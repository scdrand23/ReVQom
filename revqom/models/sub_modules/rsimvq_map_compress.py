"""
ResidualSimVQ-MAP: Residual SimVQ for Multi-Agent Perception
"""

import torch
import torch.nn as nn
from vector_quantize_pytorch import ResidualSimVQ


class RsimvqMapCompress(nn.Module):
    """ResidualSimVQ-based compression for BEV features"""
    
    def __init__(self, dim=256, codebook_dim=512, num_quantizers=4, codebook_size=1024):
        super().__init__()
        
        # Project input to codebook dimension
        self.proj_in = nn.Conv2d(dim, codebook_dim, 1) if dim != codebook_dim else nn.Identity()
        
        # ResidualSimVQ quantization
        self.rsimvq = ResidualSimVQ(
            dim=codebook_dim,
            num_quantizers=num_quantizers,
            codebook_size=codebook_size,
            rotation_trick=True  # use rotation trick from Fifty et al.
        )
        
        # Project back to original dimension
        self.proj_out = nn.Conv2d(codebook_dim, dim, 1) if dim != codebook_dim else nn.Identity()
    
    def forward(self, x):
        """Forward pass with ResidualSimVQ compression"""
        B, C, H, W = x.shape
        
        # Project to codebook dimension
        z = self.proj_in(x)
        
        # Reshape for ResidualSimVQ: [B, C, H, W] -> [B, H*W, C]
        z_flat = z.permute(0, 2, 3, 1).reshape(B, H*W, -1)
        
        # ResidualSimVQ quantization
        z_q, indices, commit_loss = self.rsimvq(z_flat)
        
        # Reshape back: [B, H*W, C] -> [B, C, H, W]
        z_q = z_q.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        
        # Project back to original dimension
        x_rec = self.proj_out(z_q)
        
        # Reconstruction loss
        rec_loss = nn.functional.mse_loss(x_rec, x)
        
        return {
            'reconstructed': x_rec,
            'indices': indices,
            'loss': rec_loss + commit_loss,
            'rec_loss': rec_loss,
            'commit_loss': commit_loss
        }