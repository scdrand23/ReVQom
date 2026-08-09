import torch
import torch.nn as nn
import torch.nn.functional as F

class DirectQAFF(nn.Module):
    """
    Direct Query-guided Adaptive Feature Fusion
    - Minimal processing of neighbor features
    - Uses HIM outputs to create attention masks
    - Preserves raw neighbor information
    """
    def __init__(self, hidden_dim=256, num_heads=4, dropout=0.1, num_stages=3, num_classes=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Simple attention weight generator from hard regions
        self.attention_generator = nn.Sequential(
            nn.Conv2d(1, 1, 3, padding=1),
            nn.Sigmoid()
        )
        
        # Direct fusion with minimal processing
        # Ego + all neighbors concatenated
        self.fusion_layer = nn.Conv2d(hidden_dim * 2, hidden_dim, 1)
        
        # Optional: dropout for regularization
        self.dropout = nn.Dropout2d(dropout)
        
    def forward(self, him_outputs, agent_features, record_len):
        B, K, C, H, W = agent_features.shape
        ego_features = agent_features[:, 0]  # [B, C, H, W]
        
        if K == 1:  # No collaboration
            return ego_features
            
        # Get hard regions from HIM
        hard_mask = him_outputs.get('accumulated_positive_mask', None)
        
        if hard_mask is not None:
            # Hard regions: where ego needs help (1 - accumulated positive mask)
            hard_regions = 1.0 - hard_mask.max(dim=1, keepdim=True)[0]  # [B, 1, H, W]
            
            # Generate spatial attention from hard regions
            spatial_attention = self.attention_generator(hard_regions)  # [B, 1, H, W]
        else:
            # No HIM info: use uniform attention
            spatial_attention = torch.ones(B, 1, H, W, device=ego_features.device) * 0.5
        
        # Get all neighbor features
        all_neighbor_features = []
        for k in range(1, min(K, record_len[0])):
            neighbor_k = agent_features[:, k]  # [B, C, H, W]
            all_neighbor_features.append(neighbor_k)
        
        if all_neighbor_features:
            # Simple average of all neighbors (preserving raw features)
            neighbor_avg = torch.stack(all_neighbor_features, dim=0).mean(dim=0)  # [B, C, H, W]
            
            # Apply spatial attention to neighbor features
            attended_neighbors = neighbor_avg * spatial_attention
            
            # Concatenate ego and attended neighbor features
            concat_features = torch.cat([
                ego_features,
                attended_neighbors
            ], dim=1)  # [B, 2*C, H, W]
            
            # Single conv to fuse
            fused = self.fusion_layer(concat_features)  # [B, C, H, W]
            
            # Apply dropout
            fused = self.dropout(fused)
            
            # Strong residual connection to ego
            output = fused + ego_features
        else:
            output = ego_features
            
        return output