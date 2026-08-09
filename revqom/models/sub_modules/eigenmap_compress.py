import torch
import torch.nn as nn
import torch.nn.functional as F


class EigenMAP_SVD(nn.Module):
    """
    EigenMAP: SVD-Based Spectral Compression for Bandwidth-Efficient Multi-Agent Perception
    
    Applies Singular Value Decomposition to BEV feature maps to achieve compression
    while preserving the semantic structure of learned features.
    """
    
    def __init__(self, input_channels, compression_ratio=16, rank_selection='fixed', 
                 energy_threshold=0.95, adaptive_threshold=True):
        super().__init__()
        
        self.input_channels = input_channels  # C (e.g., 256)
        self.compression_ratio = compression_ratio
        self.rank_selection = rank_selection  # 'fixed', 'energy', 'adaptive'
        self.energy_threshold = energy_threshold
        self.adaptive_threshold = adaptive_threshold
        
        # Fixed rank based on compression ratio
        self.fixed_rank = max(1, input_channels // compression_ratio)
        
        # Learnable reconstruction network (optional enhancement)
        self.use_learnable_reconstruction = True
        if self.use_learnable_reconstruction:
            self.reconstruction_net = nn.Sequential(
                nn.Conv2d(input_channels, input_channels, 3, padding=1, bias=False),
                nn.GroupNorm(8, input_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(input_channels, input_channels, 1)
            )
        
        # Statistics tracking
        self.register_buffer("compression_stats", torch.zeros(5))  
        # [total_samples, avg_rank, avg_compression_ratio, avg_reconstruction_error, energy_preserved]
    
    def select_rank(self, singular_values, target_rank=None):
        """
        Select optimal rank for SVD truncation based on specified criteria
        
        Args:
            singular_values: [rank] tensor of singular values (sorted descending)
            target_rank: Optional fixed rank override
            
        Returns:
            selected_rank: Number of components to keep
        """
        if target_rank is not None:
            return min(target_rank, len(singular_values))
        
        if self.rank_selection == 'fixed':
            return min(self.fixed_rank, len(singular_values))
        
        elif self.rank_selection == 'energy':
            # Keep components until energy_threshold of total energy is preserved
            total_energy = (singular_values ** 2).sum()
            cumulative_energy = torch.cumsum(singular_values ** 2, dim=0)
            energy_ratio = cumulative_energy / total_energy
            
            # Find first index where energy ratio exceeds threshold
            rank = torch.where(energy_ratio >= self.energy_threshold)[0]
            if len(rank) > 0:
                return min(rank[0].item() + 1, len(singular_values))
            else:
                return len(singular_values)
        
        elif self.rank_selection == 'adaptive':
            # Adaptive thresholding based on singular value drop-off
            if len(singular_values) < 2:
                return len(singular_values)
            
            # Find largest gap in singular values (indicates natural cutoff)
            ratios = singular_values[1:] / (singular_values[:-1] + 1e-8)
            gap_indices = torch.where(ratios < 0.1)[0]  # 10x drop indicates cutoff
            
            if len(gap_indices) > 0:
                return min(gap_indices[0].item() + 1, self.fixed_rank)
            else:
                return self.fixed_rank
        
        else:
            return self.fixed_rank
    
    def compress_sample(self, x_sample):
        """
        Compress a single sample using SVD
        
        Args:
            x_sample: [C, H, W] feature map for one sample
            
        Returns:
            compressed_data: Dictionary with U, S, Vt components
        """
        C, H, W = x_sample.shape
        
        # Reshape to [C, H*W] for SVD
        X = x_sample.reshape(C, H * W)  # [C, HW]
        
        # Add regularization for numerical stability
        X = X + torch.randn_like(X) * 1e-6  # Small noise for stability
        
        # Perform SVD with better error handling
        try:
            U, S, Vt = torch.linalg.svd(X, full_matrices=False)
        except Exception as e:
            print(f"SVD failed, using simple fallback: {e}")
            # Simple fallback: use identity matrices with reduced rank
            min_dim = min(X.shape[0], X.shape[1])
            rank = min(self.fixed_rank, min_dim)
            
            # Create truncated identity-based approximation
            U = torch.eye(X.shape[0], device=X.device)[:, :rank]
            S = torch.ones(rank, device=X.device)
            Vt = torch.eye(rank, X.shape[1], device=X.device)
            
            # Better approximation: use mean-centered version
            X_mean = X.mean(dim=1, keepdim=True)
            X_centered = X - X_mean
            if rank < min_dim:
                # Simple rank reduction by taking top-left block
                U = U * torch.norm(X_centered, dim=1, keepdim=True)[:, :rank] / (torch.norm(X_centered) + 1e-8)
                Vt = Vt * torch.norm(X_centered, dim=0, keepdim=True)[:rank, :] / (torch.norm(X_centered) + 1e-8)
        
        # Select rank
        rank = self.select_rank(S)
        
        # Truncate
        U_compressed = U[:, :rank]        # [C, rank]
        S_compressed = S[:rank]           # [rank]
        Vt_compressed = Vt[:rank, :]      # [rank, HW]
        
        return {
            'U': U_compressed,
            'S': S_compressed, 
            'Vt': Vt_compressed,
            'rank': rank,
            'original_shape': (C, H, W)
        }
    
    def decompress_sample(self, compressed_data):
        """
        Reconstruct sample from compressed SVD components
        
        Args:
            compressed_data: Dictionary with U, S, Vt components
            
        Returns:
            x_reconstructed: [C, H, W] reconstructed feature map
        """
        U = compressed_data['U']          # [C, rank] 
        S = compressed_data['S']          # [rank]
        Vt = compressed_data['Vt']        # [rank, HW]
        C, H, W = compressed_data['original_shape']
        
        # Reconstruct: X_hat = U @ diag(S) @ Vt
        X_reconstructed = U @ torch.diag(S) @ Vt  # [C, HW]
        
        # Reshape back to spatial dimensions
        x_reconstructed = X_reconstructed.reshape(C, H, W)  # [C, H, W]
        
        return x_reconstructed
    
    def forward(self, x):
        """
        Forward pass: compress and immediately decompress for training
        
        Args:
            x: [B, C, H, W] BEV feature maps
            
        Returns:
            x_reconstructed: [B, C, H, W] reconstructed features
            compression_stats: Dictionary with compression statistics
        """
        B, C, H, W = x.shape
        
        compressed_samples = []
        reconstructed_samples = []
        total_rank = 0
        
        # Process each sample in the batch
        for b in range(B):
            # Compress
            compressed = self.compress_sample(x[b])  # [C, H, W] -> compressed components
            compressed_samples.append(compressed)
            
            # Decompress
            reconstructed = self.decompress_sample(compressed)  # compressed -> [C, H, W]
            reconstructed_samples.append(reconstructed)
            
            total_rank += compressed['rank']
        
        # Stack reconstructed samples
        x_reconstructed = torch.stack(reconstructed_samples, dim=0)  # [B, C, H, W]
        
        # Apply learnable reconstruction enhancement
        if self.use_learnable_reconstruction and self.training:
            x_reconstructed = x_reconstructed + self.reconstruction_net(x_reconstructed)
        
        # Calculate statistics
        avg_rank = total_rank / B
        theoretical_compression_ratio = C / avg_rank
        
        # Reconstruction error
        reconstruction_error = F.mse_loss(x_reconstructed, x).item()
        
        # Calculate actual compression ratio (accounting for overhead)
        # Original: B * C * H * W * 32 bits
        # Compressed: B * (C*rank + rank + rank*H*W) * 32 bits  
        original_bits = B * C * H * W * 32
        compressed_bits = B * (C * avg_rank + avg_rank + avg_rank * H * W) * 32
        actual_compression_ratio = original_bits / compressed_bits
        
        # Update running statistics
        if self.training:
            self.compression_stats[0] += B  # total_samples
            self.compression_stats[1] = 0.9 * self.compression_stats[1] + 0.1 * avg_rank
            self.compression_stats[2] = 0.9 * self.compression_stats[2] + 0.1 * actual_compression_ratio
            self.compression_stats[3] = 0.9 * self.compression_stats[3] + 0.1 * reconstruction_error
        
        stats = {
            'avg_rank': avg_rank,
            'theoretical_compression_ratio': theoretical_compression_ratio,
            'actual_compression_ratio': actual_compression_ratio,
            'reconstruction_error': reconstruction_error,
            'compressed_samples': compressed_samples,  # For actual transmission
            'bits_per_pixel': compressed_bits / (B * H * W)
        }
        
        return x_reconstructed, stats
    
    def get_compression_stats(self):
        """Get accumulated compression statistics"""
        return {
            'total_samples': self.compression_stats[0].item(),
            'avg_rank': self.compression_stats[1].item(), 
            'avg_compression_ratio': self.compression_stats[2].item(),
            'avg_reconstruction_error': self.compression_stats[3].item(),
        }


# Simpler version for initial testing
class EigenMAP_Simple(nn.Module):
    """Simplified EigenMAP with fixed rank for initial experiments"""
    
    def __init__(self, input_channels, rank):
        super().__init__()
        self.input_channels = input_channels
        self.rank = rank
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Simple SVD with fixed rank
        reconstructed = []
        for b in range(B):
            X = x[b].reshape(C, H * W)
            
            # SVD
            U, S, Vt = torch.linalg.svd(X, full_matrices=False)
            
            # Truncate to fixed rank
            rank = min(self.rank, S.size(0))
            X_compressed = U[:, :rank] @ torch.diag(S[:rank]) @ Vt[:rank, :]
            
            # Reshape back
            reconstructed.append(X_compressed.reshape(C, H, W))
        
        x_reconstructed = torch.stack(reconstructed, dim=0)
        
        stats = {
            'compression_ratio': C / self.rank,
            'rank': self.rank
        }
        
        return x_reconstructed, stats