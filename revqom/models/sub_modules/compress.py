import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class NaiveCompressor(nn.Module):
    """
    A very naive compression that only compress on the channel.
    """
    def __init__(self, input_dim, compress_raito):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, input_dim//compress_raito, kernel_size=3,
                      stride=1, padding=1),
            nn.BatchNorm2d(input_dim//compress_raito, eps=1e-3, momentum=0.01),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(input_dim//compress_raito, input_dim, kernel_size=3,
                      stride=1, padding=1),
            nn.BatchNorm2d(input_dim, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.Conv2d(input_dim, input_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(input_dim, eps=1e-3,
                           momentum=0.01),
            nn.ReLU()
        )

    def forward(self, x, mode="full"):
        if mode == "encoder":
            x = self.encoder(x)
            return x
        if mode == "decoder":
            x = self.decoder(x)
            return x
        x = self.encoder(x)
        x = self.decoder(x)
        return x



class ChannelRVQ(nn.Module):
    """
    Per-pixel channel compressor with residual VQ for V2X communication.
    No mixing with original - pure compress->transmit->reconstruct.
    """
    def __init__(self, C, ratio, codebook_size=256, n_q=2, beta=0.25, ortho_reg=1e-3):
        super().__init__()
        self.C, self.Cb = C, max(C // ratio, 16)  # Ensure minimum channels
        self.n_q, self.K, self.beta, self.ortho_reg = n_q, codebook_size, beta, ortho_reg
        
        # Encoder with normalization for stability
        self.enc = nn.Sequential(
            nn.Conv2d(C, self.Cb, 1, bias=True),
            nn.BatchNorm2d(self.Cb)
        )
        
        # Decoder with normalization
        self.dec = nn.Sequential(
            nn.Conv2d(self.Cb, C, 1, bias=True),
            nn.BatchNorm2d(C),
            nn.ReLU(inplace=True)
        )
        
        # Initialize codebooks with smaller variance for stability
        self.codebooks = nn.Parameter(torch.randn(n_q, self.K, self.Cb) * 0.02)

        # Better initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def encode(self, x):
        return self.enc(x)

    def _rvq(self, z):
        B, Cb, H, W = z.shape
        
        # Normalize features for stable quantization
        z_norm = F.normalize(z, dim=1, eps=1e-6)
        
        flat = z_norm.permute(0,2,3,1).reshape(-1, Cb).contiguous()
        residual = flat
        qsum = torch.zeros_like(flat)
        
        for i in range(self.n_q):
            # Normalize codebook too
            cb = F.normalize(self.codebooks[i], dim=-1, eps=1e-6)  # [K,Cb]
            
            # Cosine similarity (more stable than L2 for high-dim)
            sim = torch.mm(residual, cb.t())  # [N, K]
            idx = sim.argmax(1)
            q = self.codebooks[i][idx]  # Use unnormalized for actual quantization
            qsum = qsum + q
            residual = residual - q.detach()
        
        z_q = qsum.view(B, H, W, Cb).permute(0,3,1,2).contiguous()
        
        # Scale back to original magnitude
        z_scale = z.norm(dim=1, keepdim=True) / (z_norm.norm(dim=1, keepdim=True) + 1e-6)
        z_q = z_q * z_scale
        
        # Gentler VQ loss
        commit_loss = F.mse_loss(z_q.detach(), z)
        codebook_loss = F.mse_loss(z_q, z.detach()) 
        vq_loss = self.beta * (commit_loss + codebook_loss)
        
        return z_q, vq_loss

    def decode(self, z_q):
        return self.dec(z_q)

    def forward(self, x, mode="full"):
        if mode == "encoder": 
            return self.encode(x)
        if mode == "decoder": 
            return self.decode(x)
        
        # Full forward: encode -> quantize -> decode
        z = self.encode(x)
        z_q, vq_loss = self._rvq(z)
        # breakpoint()
        x_hat = self.decode(z_q)
        

        if hasattr(self.enc[0], 'weight'):
            W = self.enc[0].weight.view(self.Cb, -1)  # [Cb, C]
            ortho = ((W @ W.t()) - torch.eye(self.Cb, device=W.device)).pow(2).mean()
            ortho_loss = self.ortho_reg * ortho
        else:
            ortho_loss = torch.tensor(0.0, device=x.device)
        
        return x_hat, {"vq_loss": vq_loss, "ortho_loss": ortho_loss}



import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ChannelRVQ_EMA(nn.Module):
    """
    [B, C, H, W] -> [B, C//CRR, H, W] -> Residual VQ -> [B, C, H, W]


    """
    def __init__(self, C, C_rr, codebook_size=128, n_q=3,
                 beta_commit=0.1, ema_decay=0.99, eps=1e-5,
                 use_groupnorm=True, ortho_reg=1e-4):
        super().__init__()
        self.C = C
        self.Cr = max(C // C_rr, 1)
        self.K = codebook_size
        self.n_q = n_q
        self.beta_commit = beta_commit
        self.ema_decay = ema_decay
        self.eps = eps
        self.ortho_reg = ortho_reg


        self.enc = nn.Conv2d(C, self.Cr, kernel_size=1, bias=True)
        self.dec = nn.Sequential(
            nn.Conv2d(self.Cr, C, kernel_size=1, bias=True),
            nn.ReLU(inplace=True)
        )
        # self.dec = nn.Sequential(
        #     nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        #     nn.Conv2d(self.Cr, C, kernel_size=1, bias=True),
        #     nn.ReLU(inplace=True)
        # )
        if use_groupnorm:
            # GroupNorm is stabler for small effective batch sizes per GPU than BN
            self.enc_norm = nn.GroupNorm(max(1, self.Cr // 8), self.Cr)
            self.dec_norm = nn.GroupNorm(max(1, C // 16), C)
        else:
            self.enc_norm = nn.Identity()
            self.dec_norm = nn.Identity()

        # Codebooks (n_q, K, Cr) + EMA buffers
        cr = torch.randn(self.n_q, self.K, self.Cr) * 0.02
        self.register_buffer("codebooks", cr)        
        self.register_buffer("ema_cluster_size", torch.zeros(self.n_q, self.K))
        self.register_buffer("ema_codebooks", cr.clone())
        self.register_buffer("usage_ema", torch.zeros(self.n_q, self.K))  
        self.post_affine = nn.Sequential(
            nn.Conv2d(self.Cr, self.Cr, kernel_size=1, bias=True),
            nn.ReLU(inplace=True)
        )


        nn.init.kaiming_normal_(self.enc.weight, mode='fan_out', nonlinearity='relu')
        if self.enc.bias is not None: nn.init.constant_(self.enc.bias, 0)
        for m in self.dec:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: nn.init.constant_(m.bias, 0)

    @torch.no_grad()
    def _ema_update(self, all_residuals, indices):
        # all_residuals: [n_q, N, Cr], indices: [n_q, N]
        for i in range(self.n_q):
            flat_residual = all_residuals[i]  # [N, Cr] - residual for stage i
            idx = indices[i]  # [N]
            one_hot = F.one_hot(idx, num_classes=self.K).type_as(flat_residual)  # [N, K]
            cluster_size = one_hot.sum(0)  # [K]
            self.ema_cluster_size[i] = self.ema_decay * self.ema_cluster_size[i] + (1 - self.ema_decay) * cluster_size

            # Sum of residuals assigned to each code
            embed_sum = one_hot.t() @ flat_residual  # [K, Cr]
            self.ema_codebooks[i] = self.ema_decay * self.ema_codebooks[i] + (1 - self.ema_decay) * embed_sum

            # Track usage
            self.usage_ema[i] = self.ema_decay * self.usage_ema[i] + (1 - self.ema_decay) * (cluster_size > 0).float()

            # Normalize to get new codebook
            n = self.ema_cluster_size[i] + self.eps
            self.codebooks[i] = self.ema_codebooks[i] / n.unsqueeze(1)

            # Dead-code refresh (optional): reinit low-usage codes to random residuals
            dead = (self.usage_ema[i] < 0.01) & (n < 1.0)
            if dead.any():
                N = flat_residual.size(0)
                # sample random residuals
                sel = flat_residual[torch.randint(0, N, (dead.sum().item(),), device=flat_residual.device)]
                self.codebooks[i][dead] = sel

    def encode(self, x):
        z = self.enc_norm(self.enc(x))
        return z

    def decode(self, z_q):
        z_q = self.post_affine(z_q)
        x_hat = self.dec_norm(self.dec(z_q))
        return x_hat

    def _residual_quantize(self, z):
        """
        Residual VQ with L2 nearest neighbor and STE proxy.
        Returns:
            z_q_st: STE quantized latent (for backprop)
            vq_loss: commitment loss
            indices: [n_q, N] chosen code indices (for logging/bitrate)
            residuals: [n_q, N, Cr] residuals for each stage (for EMA)
        """
        B, Cr, H, W = z.shape
        flat = z.permute(0,2,3,1).reshape(-1, Cr).contiguous()   # [N, Cr]
        residual = flat
        qsum = torch.zeros_like(flat)
        all_idx = []
        all_residuals = []  # Store residuals for EMA

        with torch.no_grad():
            for i in range(self.n_q):
                # Store current residual for this stage's EMA update
                all_residuals.append(residual.clone())
                
                # L2 distance to codebook
                cb = self.codebooks[i]   # [K, Cr]
                # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b
                a2 = (residual**2).sum(dim=1, keepdim=True)      # [N,1]
                b2 = (cb**2).sum(dim=1).unsqueeze(0)             # [1,K]
                ab = residual @ cb.t()                           # [N,K]
                dist = a2 + b2 - 2*ab
                idx = dist.argmin(dim=1)                         # [N]
                all_idx.append(idx)
                q = cb[idx]                                      # [N, Cr]
                qsum = qsum + q
                residual = residual - q

        z_q = qsum.view(B, H, W, Cr).permute(0,3,1,2).contiguous()

        # STE: copy gradients from z to z_q
        z_q_st = z + (z_q - z).detach()

        # Commitment loss only (EMA handles codebook)
        vq_loss = self.beta_commit * F.mse_loss(z_q.detach(), z)

        return z_q_st, vq_loss, torch.stack(all_idx, dim=0), torch.stack(all_residuals, dim=0)

    def forward(self, x, mode="full"):
        if mode == "encoder":
            return self.encode(x)
        if mode == "decoder":
            return self.decode(x)

        z = self.encode(x)
        z_q_st, vq_loss, indices, all_residuals = self._residual_quantize(z)
        x_hat = self.decode(z_q_st)
        # breakpoint()
        if hasattr(self.enc, 'weight'):
            W = self.enc.weight.view(self.Cr, -1)  # [Cr, C]
            ortho = ((W @ W.t()) - torch.eye(self.Cr, device=W.device)).pow(2).mean()
            ortho_loss = self.ortho_reg * ortho
        else:
            ortho_loss = torch.tensor(0.0, device=x.device)

       
        if self.training:
            with torch.no_grad():
                self._ema_update(all_residuals, indices)

        # Perplexity (usage) for logging
        # breakpoint()
        with torch.no_grad():
            usage = []
            for i in range(self.n_q):
                hist = torch.bincount(indices[i], minlength=self.K).float()
                p = hist / (hist.sum() + 1e-6)
                perp = torch.exp(-(p * (p.clamp_min(1e-6).log())).sum())
                usage.append(perp)
            perplexity = torch.stack(usage).mean()

        return x_hat, {"vq_loss": vq_loss, "ortho_loss": ortho_loss, "perplexity": perplexity}












class ResidualFSQ(nn.Module):
    """
    Simplified Residual Finite Scalar Quantization for channel compression.
    Based on FSQ: VQ-VAE Made Simple - https://arxiv.org/abs/2309.15505
    """
    
    def __init__(self, C, ratio, levels=[8, 8, 8], num_quantizers=2, 
                 beta_commit=0.1, ortho_reg=1e-4):
        super().__init__()
        self.C = C
        self.Cb = max(C // ratio, 4)
        self.levels = levels
        self.num_quantizers = num_quantizers
        self.beta_commit = beta_commit
        self.ortho_reg = ortho_reg
        
        # Encoder/decoder
        self.enc = nn.Conv2d(C, self.Cb, 1, bias=True)
        self.dec = nn.Sequential(
            nn.Conv2d(self.Cb, C, 1, bias=True),
            nn.ReLU(inplace=True)
        )
        
        # FSQ layers for residual quantization
        self.quantizers = nn.ModuleList([
            FSQLayer(self.Cb, levels) for _ in range(num_quantizers)
        ])
        
        # Scales for each quantizer (decreasing powers)
        levels_tensor = torch.tensor(levels, dtype=torch.float32)
        scales = [(levels_tensor - 1) ** -i for i in range(num_quantizers)]
        self.register_buffer('scales', torch.stack(scales), persistent=False)
        
        # Init
        nn.init.kaiming_normal_(self.enc.weight, mode='fan_out', nonlinearity='relu')
        if self.enc.bias is not None:
            nn.init.constant_(self.enc.bias, 0)
        for m in self.dec:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def encode(self, x):
        return self.enc(x)
    
    def decode(self, z_q):
        return self.dec(z_q)
    
    def _residual_fsq(self, z):
        """Residual FSQ quantization"""
        B, Cb, H, W = z.shape
        z_flat = z.permute(0, 2, 3, 1).reshape(-1, Cb)  # [B*H*W, Cb]
        
        quantized_out = torch.zeros_like(z_flat)
        residual = z_flat
        commitment_loss = 0.0
        
        for i, (quantizer, scale) in enumerate(zip(self.quantizers, self.scales)):
            # Scale residual
            scaled_residual = residual / scale.view(1, -1)
            
            # Quantize
            quantized = quantizer(scaled_residual.view(B, H, W, Cb))
            quantized_flat = quantized.view(-1, Cb)
            
            # Scale back
            quantized_scaled = quantized_flat * scale.view(1, -1)
            
            # Accumulate
            quantized_out = quantized_out + quantized_scaled
            
            # Commitment loss
            commitment_loss = commitment_loss + F.mse_loss(quantized_scaled.detach(), residual)
            
            # Update residual
            residual = residual - quantized_scaled.detach()
        
        # Reshape back
        z_q = quantized_out.view(B, H, W, Cb).permute(0, 3, 1, 2)
        
        # STE
        z_q_st = z + (z_q - z).detach()
        
        return z_q_st, self.beta_commit * commitment_loss
    
    def forward(self, x, mode="full"):
        if mode == "encoder":
            return self.encode(x)
        if mode == "decoder":
            return self.decode(x)
        
        # Full forward
        z = self.encode(x)
        z_q_st, commit_loss = self._residual_fsq(z)
        x_hat = self.decode(z_q_st)
        
        # Orthogonality loss
        if hasattr(self.enc, 'weight'):
            W = self.enc.weight.view(self.Cb, -1)
            ortho = ((W @ W.t()) - torch.eye(self.Cb, device=W.device)).pow(2).mean()
            ortho_loss = self.ortho_reg * ortho
        else:
            ortho_loss = torch.tensor(0.0, device=x.device)
        
        return x_hat, {"vq_loss": commit_loss, "ortho_loss": ortho_loss}


class FSQLayer(nn.Module):
    """Simple FSQ layer for scalar quantization"""
    
    def __init__(self, dim, levels):
        super().__init__()
        self.levels = torch.tensor(levels)
        self.dim = dim
        assert len(levels) == dim, f"Number of levels {len(levels)} must match dim {dim}"
    
    def bound_and_quantize(self, z):
        """Bound and quantize each channel independently"""
        device = z.device
        levels = self.levels.to(device)
        
        # Bound to [-1, 1] range and quantize
        bounded = torch.tanh(z)  # Soft bound to (-1, 1)
        
        # Scale to quantization levels
        half_levels = (levels - 1) / 2.0
        quantized = torch.round(bounded * half_levels.view(1, 1, 1, -1)) / half_levels.view(1, 1, 1, -1)
        
        # STE
        return z + (quantized - z).detach()
    
    def forward(self, z):
        """
        Args:
            z: [B, H, W, dim] tensor
        Returns:
            quantized: [B, H, W, dim] tensor
        """
        return self.bound_and_quantize(z)
