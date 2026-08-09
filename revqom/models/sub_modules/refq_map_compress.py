"""
REFQ-MAP: Clean residual VQ using vector-quantize-pytorch library
"""

import torch
import torch.nn as nn
from vector_quantize_pytorch import ResidualVQ


class RefqMapCompress(nn.Module):
    """Simple compression using the proven ResidualVQ library"""
    
    def __init__(self, dim=256, codebook_dim=64, codebook_size=256, num_quantizers=3, 
                 decay=0.99, commitment_weight=0.25, use_cosine_sim=False):
        super().__init__()
        
        # Project input to codebook dimension
        
        self.dim, self.codebook_dim = dim, codebook_dim
        self.proj_in = nn.Sequential(
            nn.Conv2d(dim, codebook_dim, 1, bias=True),
            nn.GroupNorm(max(1, codebook_dim // 8), codebook_dim)
        ) if dim != codebook_dim else nn.Identity() 
        # ResidualVQ from the library
        self.rvq = ResidualVQ(
            dim=codebook_dim,
            num_quantizers=num_quantizers,
            codebook_size=codebook_size,
            decay=decay,
            commitment_weight=commitment_weight,
            kmeans_init=True,
            threshold_ema_dead_code=2,
            use_cosine_sim=use_cosine_sim,
            accept_image_fmap=True   # <-- avoids manual reshape
        )
        
        # Project back to original dimension
        self.proj_out = nn.Sequential(
            nn.Conv2d(codebook_dim, dim, 1, bias=True),
            nn.GroupNorm(max(1, dim // 16), dim),
            nn.ReLU(inplace=True)
        ) if dim != codebook_dim else nn.Identity()
    
    # @torch.no_grad()
    # def bitrate_bits(self, indices, valid: torch.Tensor | None, H: int, W: int) -> float:
    #     """Rough payload size in bits/frame given indices."""
    #     # indices: (B, n_q, H, W) or (B, HW, n_q) depending on lib version; handle fmap case:
    #     if indices.dim() == 4:
    #         b, n_q, h, w = indices.shape
    #         n_valid = valid.sum().item() if valid is not None else (b * h * w)
    #     else:
    #         b, hw, n_q = indices.shape
    #         n_valid = valid.sum().item() if valid is not None else (b * hw)
    #     # ceil(log2 K) ≈ bits per index
    #     return n_valid * self.rvq.codebook_size.bit_length()

    def forward(self, x, valid_mask=None):
        """
        x: (B, C, H, W)
        valid_mask: (B, 1, H, W) with 1 for valid BEV cells (optional but recommended)
        """
        B, C, H, W = x.shape

        # project to latent
        z = self.proj_in(x) if not isinstance(self.proj_in, nn.Identity) else x

        # RVQ on fmap directly; returns (quantized, indices, commit_loss)
        z_q, indices, commit_loss = self.rvq(z)   # z_q: (B, Cb, H, W)

        # back-project
        x_rec = self.proj_out(z_q) if not isinstance(self.proj_out, nn.Identity) else z_q

        # -------- losses --------
        if valid_mask is None:
            # default to activity-based mask to avoid empty cells dominating
            valid_mask = (x.abs().sum(dim=1, keepdim=True) > 1e-6).float()

        # robust L1 on valid cells only
        w = valid_mask.float()
        rec_loss = (w * (x_rec - x).abs()).sum() / (w.sum() + 1e-6)

        # IMPORTANT: commit_loss from the module is already scaled by commitment_weight
        # so you usually add it directly without extra weighting.
        total_loss = rec_loss + commit_loss

        # optional: perplexity / usage (if exposed by your version)
        # Some versions expose .perplexity on the last forward; if not, skip.
        aux = {
            'indices': indices,
            'rec_loss': rec_loss.detach(),
            'commit_loss': commit_loss.detach(),
            'total_loss': total_loss.detach(),
        }

        return x_rec, total_loss, aux

    # Convenience encode/decode if you’ll transmit indices later
    def encode(self, x):
        z = self.proj_in(x) if not isinstance(self.proj_in, nn.Identity) else x
        _, indices, _ = self.rvq(z, return_all_codes=True)  # depending on version
        return indices

    def decode(self, indices, shape_hw):
        H, W = shape_hw
        z_q = self.rvq.get_quantized_from_indices(indices, shape=(H, W))  # API name may vary by version
        x_rec = self.proj_out(z_q) if not isinstance(self.proj_out, nn.Identity) else z_q
        return x_rec