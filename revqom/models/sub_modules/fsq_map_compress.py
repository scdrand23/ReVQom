"""
FSQ-MAP: Finite Scalar Quantization for Multi-Agent Perception
"""

import torch
import torch.nn as nn
from vector_quantize_pytorch import FSQ


class FsqMapCompress(nn.Module):
    """FSQ-based compression for BEV features"""
    
    def __init__(self, dim=256, codebook_dim=64, levels=[8, 5, 5, 5]):
        super().__init__()
        
        # Project input to codebook dimension
        self.proj_in = nn.Conv2d(dim, codebook_dim, 1) if dim != codebook_dim else nn.Identity()
        
        # Project to FSQ dimension (number of levels)
        fsq_dim = len(levels)
        self.to_fsq = nn.Linear(codebook_dim, fsq_dim)
        
        # FSQ quantization
        self.fsq = FSQ(levels=levels)
        
        # Project back from FSQ dimension
        self.from_fsq = nn.Linear(fsq_dim, codebook_dim)
        
        # Project back to original dimension
        self.proj_out = nn.Conv2d(codebook_dim, dim, 1) if dim != codebook_dim else nn.Identity()
    
    def forward(self, x):
        """Forward pass with FSQ compression"""
        B, C, H, W = x.shape
        
        # Project to codebook dimension
        z = self.proj_in(x)
        
        # Reshape for linear layers: [B, C, H, W] -> [B, H*W, C]
        z_flat = z.permute(0, 2, 3, 1).reshape(B, H*W, -1)
        
        # Project to FSQ dimension
        z_fsq = self.to_fsq(z_flat)
        
        # FSQ quantization
        z_q, indices = self.fsq(z_fsq)
        
        # Project back from FSQ dimension
        z_rec = self.from_fsq(z_q)
        
        # Reshape back: [B, H*W, C] -> [B, C, H, W]
        z_rec = z_rec.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        
        # Project back to original dimension
        x_rec = self.proj_out(z_rec)
        
        # Reconstruction loss
        rec_loss = nn.functional.mse_loss(x_rec, x)
        
        return {
            'reconstructed': x_rec,
            'indices': indices,
            'loss': rec_loss,
            'rec_loss': rec_loss,
            'commit_loss': torch.tensor(0.0, device=x.device)  # FSQ has no commitment loss
        }