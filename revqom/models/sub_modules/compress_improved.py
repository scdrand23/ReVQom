import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
import numpy as np

class ImprovedChannelRVQ(nn.Module):
    """
    Improved Residual VQ with:
    - K-means initialization
    - Entropy regularization for better code usage
    - Perplexity tracking
    - Dead code revival
    - Higher commitment loss for better utilization
    """
    def __init__(self, C, ratio, codebook_size=32, n_q=3,
                 beta_commit=0.5, ema_decay=0.99, eps=1e-5,
                 use_groupnorm=True, ortho_reg=1e-4, 
                 entropy_weight=0.01, kmeans_init=True):
        super().__init__()
        self.C = C
        self.Cb = max(C // ratio, 16)
        self.K = codebook_size  # Reduced from 128 to match actual usage
        self.n_q = n_q
        self.beta_commit = beta_commit  # Increased from 0.1
        self.ema_decay = ema_decay
        self.eps = eps
        self.ortho_reg = ortho_reg
        self.entropy_weight = entropy_weight
        self.kmeans_init = kmeans_init
        self.initialized = False

        # Encoder/decoder: 1x1 conv bottleneck
        self.enc = nn.Conv2d(C, self.Cb, 1, bias=True)
        self.dec = nn.Sequential(
            nn.Conv2d(self.Cb, C, 1, bias=True),
            nn.ReLU(inplace=True)
        )
        
        if use_groupnorm:
            self.enc_norm = nn.GroupNorm(max(1, self.Cb // 8), self.Cb)
            self.dec_norm = nn.GroupNorm(max(1, C // 16), C)
        else:
            self.enc_norm = nn.Identity()
            self.dec_norm = nn.Identity()

        # Codebooks with better initialization
        if kmeans_init:
            # Start with random but will be replaced by k-means
            cb = torch.randn(self.n_q, self.K, self.Cb) * 0.1
        else:
            # Larger initialization for better diversity
            cb = torch.randn(self.n_q, self.K, self.Cb) * 0.5
            
        self.register_buffer("codebooks", cb)
        self.register_buffer("ema_cluster_size", torch.zeros(self.n_q, self.K))
        self.register_buffer("ema_codebooks", cb.clone())
        self.register_buffer("usage_ema", torch.zeros(self.n_q, self.K))
        
        # Track perplexity for monitoring
        self.register_buffer("perplexity", torch.zeros(self.n_q))
        self.register_buffer("code_usage", torch.zeros(self.n_q, self.K))

        # Learnable affine to restore scale/shift
        self.post_affine = nn.Sequential(
            nn.Conv2d(self.Cb, self.Cb, 1, bias=True),
            nn.ReLU(inplace=True)
        )

        # Init
        nn.init.kaiming_normal_(self.enc.weight, mode='fan_out', nonlinearity='relu')
        if self.enc.bias is not None: 
            nn.init.constant_(self.enc.bias, 0)
        for m in self.dec:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: 
                    nn.init.constant_(m.bias, 0)

    @torch.no_grad()
    def _kmeans_init(self, z):
        """Initialize codebooks with k-means on first batch"""
        if self.initialized or not self.kmeans_init:
            return
            
        B, Cb, H, W = z.shape
        flat = z.permute(0,2,3,1).reshape(-1, Cb).cpu().numpy()
        
        # Sample subset for efficiency
        n_samples = min(10000, flat.shape[0])
        idx = np.random.choice(flat.shape[0], n_samples, replace=False)
        samples = flat[idx]
        
        for i in range(self.n_q):
            # K-means clustering
            kmeans = KMeans(n_clusters=self.K, n_init=3, max_iter=100)
            kmeans.fit(samples)
            centers = torch.from_numpy(kmeans.cluster_centers_).to(z.device).float()
            self.codebooks[i] = centers
            self.ema_codebooks[i] = centers.clone()
            
            # For residual quantization, update samples
            if i < self.n_q - 1:
                # Compute residuals for next stage
                labels = kmeans.labels_
                samples = samples - kmeans.cluster_centers_[labels]
                
        self.initialized = True
        print(f"Initialized {self.n_q} codebooks with k-means clustering")

    @torch.no_grad()
    def _ema_update(self, flat_residual, indices):
        """EMA update with perplexity tracking and dead code revival"""
        N = flat_residual.size(0)
        
        for i in range(self.n_q):
            idx = indices[i]  # [N]
            one_hot = F.one_hot(idx, num_classes=self.K).type_as(flat_residual)  # [N, K]
            cluster_size = one_hot.sum(0)  # [K]
            
            # Update EMA
            self.ema_cluster_size[i] = self.ema_decay * self.ema_cluster_size[i] + (1 - self.ema_decay) * cluster_size
            embed_sum = one_hot.t() @ flat_residual  # [K, Cb]
            self.ema_codebooks[i] = self.ema_decay * self.ema_codebooks[i] + (1 - self.ema_decay) * embed_sum
            
            # Track usage
            self.usage_ema[i] = self.ema_decay * self.usage_ema[i] + (1 - self.ema_decay) * (cluster_size > 0).float()
            self.code_usage[i] = cluster_size / N  # Current batch usage
            
            # Calculate perplexity
            probs = cluster_size.float() / N
            probs = probs[probs > 0]  # Remove zeros for log
            if len(probs) > 0:
                entropy = -(probs * probs.log()).sum()
                self.perplexity[i] = entropy.exp()
            
            # Normalize to get new codebook
            n = self.ema_cluster_size[i] + self.eps
            self.codebooks[i] = self.ema_codebooks[i] / n.unsqueeze(1)
            
            # More aggressive dead-code revival
            dead = (self.usage_ema[i] < 0.005)  # Lower threshold
            if dead.any() and N > 0:
                # Replace dead codes with random high-residual samples
                # This encourages using underutilized codes
                residual_norms = (flat_residual ** 2).sum(dim=1)
                top_k = min(dead.sum().item() * 2, N // 4)
                top_indices = residual_norms.topk(top_k).indices
                replace_indices = top_indices[torch.randperm(top_k)[:dead.sum()]]
                self.codebooks[i][dead] = flat_residual[replace_indices]
                # Reset their EMA stats
                self.ema_cluster_size[i][dead] = 1.0
                self.ema_codebooks[i][dead] = flat_residual[replace_indices]

    def encode(self, x):
        z = self.enc_norm(self.enc(x))
        # Initialize codebooks on first forward pass
        if not self.initialized and self.training:
            self._kmeans_init(z)
        return z

    def decode(self, z_q):
        z_q = self.post_affine(z_q)
        x_hat = self.dec_norm(self.dec(z_q))
        return x_hat

    def _residual_quantize(self, z):
        """
        Residual VQ with entropy regularization
        """
        B, Cb, H, W = z.shape
        flat = z.permute(0,2,3,1).reshape(-1, Cb).contiguous()   # [N, Cb]
        residual = flat
        qsum = torch.zeros_like(flat)
        all_idx = []
        all_probs = []

        with torch.no_grad():
            for i in range(self.n_q):
                cb = self.codebooks[i]   # [K, Cb]
                
                # L2 distance
                a2 = (residual**2).sum(dim=1, keepdim=True)      # [N,1]
                b2 = (cb**2).sum(dim=1).unsqueeze(0)             # [1,K]
                ab = residual @ cb.t()                           # [N,K]
                dist = a2 + b2 - 2*ab
                
                # Convert distances to probabilities for entropy
                probs = F.softmax(-dist / 2.0, dim=1)  # Temperature=2 for smoother distribution
                all_probs.append(probs)
                
                idx = dist.argmin(dim=1)                         # [N]
                all_idx.append(idx)
                
                q = cb[idx]                                      # [N, Cb]
                qsum = qsum + q
                residual = residual - q

        # Update EMA
        if self.training:
            self._ema_update(flat, all_idx)

        # Straight-through estimator
        z_q = qsum.view(B, H, W, Cb).permute(0,3,1,2).contiguous()
        z_q_st = z + (z_q - z).detach()

        # Commitment loss (higher weight for better utilization)
        commit_loss = F.mse_loss(z_q_st, z.detach())
        
        # Entropy regularization loss
        entropy_loss = 0
        for probs in all_probs:
            # Maximize entropy of code selection
            avg_probs = probs.mean(dim=0)
            avg_probs = avg_probs[avg_probs > 0]
            if len(avg_probs) > 0:
                entropy = -(avg_probs * avg_probs.log()).sum()
                entropy_loss = entropy_loss - entropy  # Negative because we want to maximize
        
        entropy_loss = entropy_loss / self.n_q
        
        # Combined VQ loss
        vq_loss = self.beta_commit * commit_loss + self.entropy_weight * entropy_loss

        return z_q_st, vq_loss, torch.stack(all_idx)

    def forward(self, x, mode="full"):
        if mode == "encoder": 
            return self.encode(x)
        if mode == "decoder": 
            return self.decode(x)
        
        # Full forward: encode -> quantize -> decode
        z = self.encode(x)
        z_q, vq_loss, indices = self._residual_quantize(z)
        x_hat = self.decode(z_q)
        
        # Orthogonality loss
        if hasattr(self.enc, 'weight'):
            W = self.enc.weight.view(self.Cb, -1)  # [Cb, C]
            ortho = ((W @ W.t()) - torch.eye(self.Cb, device=W.device)).pow(2).mean()
            ortho_loss = self.ortho_reg * ortho
        else:
            ortho_loss = torch.tensor(0.0, device=x.device)
        
        # Log perplexity for monitoring
        avg_perplexity = self.perplexity.mean().item()
        
        return x_hat, {
            "vq_loss": vq_loss, 
            "ortho_loss": ortho_loss,
            "perplexity": avg_perplexity,
            "code_usage": self.code_usage.mean(dim=0)  # Average usage across quantizers
        }