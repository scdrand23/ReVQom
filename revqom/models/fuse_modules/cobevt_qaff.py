import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from revqom.models.fuse_modules.swap_fusion_modules import SwapFusionBlockMask
from revqom.models.sub_modules.torch_transformation_utils import warp_affine_simple
from revqom.models.fuse_modules.fuse_utils import regroup as Regroup
from einops.layers.torch import Rearrange, Reduce

class CoBEVTQAFF(nn.Module):
    def __init__(self, args):
        super(CoBEVTQAFF, self).__init__()
        
        self.layers = nn.ModuleList([])
        self.depth = args.get('depth', 1)
        
        input_dim = args.get('input_dim', 256)
        mlp_dim = args.get('mlp_dim', 256)
        agent_size = args.get('agent_size', 5)
        window_size = args.get('window_size', 7)
        drop_out = args.get('drop_out', 0.1)
        dim_head = args.get('dim_head', 32)
        self.num_stages = args.get('num_stages', 3)
        
        for i in range(self.depth):
            # Note: input_dim will be 2*input_dim after concatenating query+agent features
            block = SwapFusionBlockMask(input_dim * 2,  # Account for concatenated features
                                       mlp_dim,
                                       dim_head,
                                       window_size,
                                       agent_size,
                                       drop_out)
            self.layers.append(block)
        
        self.stage_attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim // 4, 1)
        )
        
        self.query_proj = nn.Linear(input_dim, input_dim)
        self.norm_query = nn.LayerNorm(input_dim)
        
        self.mlp_head = nn.Sequential(
            Reduce('b m d h w -> b d h w', 'mean'),
            Rearrange('b d h w -> b h w d'),
            nn.LayerNorm(input_dim * 2),  # Account for concatenated features
            nn.Linear(input_dim * 2, input_dim),  # Project back to original dim
            Rearrange('b h w d -> b d h w')
        )
        
        self.agent_attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim // 2, 1)
        )

    def forward(self, query_features, agent_features, record_len, affine_matrix):
        B, K, C, H, W = agent_features.shape
        L = affine_matrix.shape[1]
        
        # query_features = query_features.unsqueeze(1).expand(-1, K, -1, -1, -1)
        
        valid_mask = torch.arange(K, device=record_len.device)[None, :] < record_len[:, None]
        
        stage_queries = torch.chunk(query_features, self.num_stages, dim=2)
        stage_weights = []
        
        for stage_idx in range(self.num_stages):
            stage_q = rearrange(stage_queries[stage_idx], 'b k c h w -> b k (h w) c')
            stage_feat = stage_q.mean(dim=2)  # [B, K, C]
            weight = self.stage_attention(stage_feat)  # [B, K, 1]
            stage_weights.append(weight.squeeze(-1))  # [B, K]
        
        stage_weights = torch.stack(stage_weights, dim=-1)  # [B, K, num_stages]
        stage_weights = F.softmax(stage_weights, dim=-1)
        
        # Fix: apply weights per agent - stage_weights [B, K, num_stages], stage_queries[i] [B, K, C//num_stages, H, W]
        combined_queries = sum(stage_queries[i] * stage_weights[:, :, i:i+1, None, None] 
                              for i in range(self.num_stages))
        
        combined_queries = rearrange(combined_queries, 'b k c h w -> (b k) c h w')
        combined_queries = self.norm_query(combined_queries.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        combined_queries = self.query_proj(combined_queries.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
        # Fix: agent_features is already [B, K, C, H, W], no need to flatten and regroup
        # Just use it directly and create appropriate mask
        agent_flat = rearrange(agent_features, 'b k c h w -> (b k) c h w')
        query_flat = combined_queries
        
        # Create mask for valid agents
        mask = torch.arange(K, device=record_len.device)[None, :] < record_len[:, None]
        
        # Simplify: just concatenate query and agent features along agent dimension
        # query_flat: [B*K, C, H, W], agent_flat: [B*K, C, H, W]
        combined_flat = torch.cat([query_flat, agent_flat], dim=1)  # [B*K, 2*C, H, W]
        
        # Reshape back to batch format for processing
        combined_features = rearrange(combined_flat, '(b k) c h w -> b k c h w', b=B, k=K)
        
        # Create simple mask for agents (no complex query masking)
        agent_mask = mask.unsqueeze(2).unsqueeze(3).unsqueeze(4)  # [B, K, 1, 1, 1]
        agent_mask = agent_mask.expand(-1, -1, combined_features.shape[2], H, W)
        
        # Apply transformation using ego perspective (agent 0)
        transformed_features = []
        for b in range(B):
            ego = 0
            # Transform all agent features to ego coordinate frame
            transformed = warp_affine_simple(combined_features[b], affine_matrix[b, ego], (H, W))
            transformed_features.append(transformed)
        x = torch.stack(transformed_features)
        
        # Create mask for SwapFusion - format expected by SwapFusionBlockMask
        swap_mask = mask.unsqueeze(2).unsqueeze(3).unsqueeze(4)  # [B, K, 1, 1, 1]
        swap_mask = repeat(swap_mask, 'b k c h w -> b (h new_h) (w new_w) c k', new_h=H, new_w=W)
        
        for stage in self.layers:
            x = stage(x, mask=swap_mask)
        
        fused_features = self.mlp_head(x)
        
        return fused_features