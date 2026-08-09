import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

class QAFFV2(nn.Module):
    def __init__(self, hidden_dim=256, num_heads=4, dropout=0.1, num_stages=3, compress_ratio=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_stages = num_stages
        self.compress_ratio = compress_ratio
        
        self.stage_self_attns = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
            for _ in range(num_stages)
        ])
        
        self.stage_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 4, 1)
        )
        
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        
        self.proj_k = nn.Linear(hidden_dim, hidden_dim)
        self.proj_v = nn.Linear(hidden_dim, hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
       
        self.norm_stages = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_stages)])
        self.norm_combined = nn.LayerNorm(hidden_dim)
        self.norm_kv = nn.LayerNorm(hidden_dim)
        self.norm_out = nn.LayerNorm(hidden_dim)

        self.agent_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.sample_id = 0


    def forward(self, query_features, agent_features, record_len, hard_instance_maps=None):
        B, K, C, H, W = agent_features.shape
        
        # query_features is already [B, K, num_stages*hidden_dim, H, W] from multi-agent HIM
        valid_mask = torch.arange(K, device=record_len.device)[None, :] < record_len[:, None]
        attn_mask = ~valid_mask
        
        stage_queries = torch.chunk(query_features, self.num_stages, dim=2)
        
        # Process hard instance maps for attention enhancement
        # if hard_instance_maps is not None:
        #     # hard_instance_maps: List of [B*N, num_classes, H, W] per stage
        #     # Need to regroup to [B, K, num_classes, H, W] to match agent_features format
        #     from revqom.models.fuse_modules.fuse_utils import regroup
        #     
        #     regrouped_hard_maps = []
        #     for stage_map in hard_instance_maps:
        #         # Regroup from [B*N, num_classes, H, W] to [B, K, num_classes, H, W]
        #         regrouped_map, _ = regroup(stage_map, record_len, K)
        #         regrouped_hard_maps.append(regrouped_map)
        #     
        #     # Combine stage maps: max across classes and stages for spatial attention
        #     # Stack: [B, K, num_stages, num_classes, H, W] -> [B, K, H, W]
        #     combined_hard_map = torch.stack(regrouped_hard_maps, dim=2).max(dim=2)[0].max(dim=2)[0]  # [B, K, H, W]
        #     hard_attention_weight = 1.0 + combined_hard_map  # [B, K, H, W] - boost hard regions
        # else:
        #     hard_attention_weight = None
        hard_attention_weight = None
        
        stage_outputs = []
        for stage_idx in range(self.num_stages):
            stage_q = rearrange(stage_queries[stage_idx], 'b k c h w -> (b h w) k c')
            stage_q = self.norm_stages[stage_idx](stage_q)
            
            stage_out, _ = self.stage_self_attns[stage_idx](
                query=stage_q,
                key=stage_q,
                value=stage_q,
                key_padding_mask=attn_mask.repeat_interleave(H*W, dim=0)
            )
            stage_outputs.append(stage_out)
        
        stage_weights = []
        for stage_out in stage_outputs:
            stage_feat = stage_out.mean(dim=1)
            weight = self.stage_attention(stage_feat)
            stage_weights.append(weight)
        
        stage_weights = torch.stack(stage_weights, dim=-1)
        stage_weights = F.softmax(stage_weights, dim=-1)
        
        combined_queries = sum(stage_outputs[i] * stage_weights[..., i:i+1] 
                              for i in range(self.num_stages))
        combined_queries = self.norm_combined(combined_queries)
        
        agent_feats_flat = rearrange(agent_features, 'b k c h w -> (b h w) k c')
        agent_feats_flat = self.norm_kv(agent_feats_flat)
        
        keys = self.proj_k(agent_feats_flat)
        values = self.proj_v(agent_feats_flat)
        
        # Apply hard instance attention enhancement
        # if hard_attention_weight is not None:
        #     # Enhance keys and values for hard instance regions
        #     # hard_attention_weight: [B, K, H, W] -> [(B H W), K, 1]
        #     hard_weight_flat = rearrange(hard_attention_weight, 'b k h w -> (b h w) k 1')
        #     keys = keys * hard_weight_flat
        #     values = values * hard_weight_flat
        
        cross_out, cross_attn_weights = self.cross_attn(
            query=combined_queries,
            key=keys,
            value=values,
            key_padding_mask=attn_mask.repeat_interleave(H*W, dim=0)
        )
        
        cross_out = self.norm_out(cross_out)
        cross_out = self.mlp(cross_out)
        
        cross_out = rearrange(cross_out, '(b h w) k c -> b k c h w', b=B, h=H, w=W)
        
        agent_feats_pooled = cross_out.mean(dim=(3, 4))
        agent_weights = self.agent_attention(agent_feats_pooled)
        
        agent_weights = agent_weights * valid_mask.float().unsqueeze(-1) + (~valid_mask).float().unsqueeze(-1) * (-1e9)
        agent_weights = F.softmax(agent_weights, dim=1)
        
        fused_out = (cross_out * agent_weights.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)
        
        self.sample_id += 1
        return fused_out
