import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleQAFF(nn.Module):
    """
    Simplified Query-guided Adaptive Feature Fusion
    - Uses HIM hard regions to weight neighbor contributions
    - Provides more raw neighbor features to final fusion
    - Less processing, more direct fusion
    """
    def __init__(self, hidden_dim=256, num_heads=4, dropout=0.1, num_stages=3, num_classes=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_stages = num_stages
        
        # Simple MLP to decode importance from hard regions
        self.importance_decoder = nn.Sequential(
            nn.Conv2d(1, hidden_dim // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 4, 1, 1),
            nn.Sigmoid()
        )
        
        # Lightweight fusion - just combine ego and weighted neighbors
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # Final refinement
        self.output_conv = nn.Conv2d(hidden_dim, hidden_dim, 1)
        
    def forward(self, him_outputs, agent_features, record_len):
        B, K, C, H, W = agent_features.shape
        ego_features = agent_features[:, 0]  # [B, C, H, W]
        
        if K == 1:  # No collaboration
            return ego_features
            
        # Get accumulated hard mask from HIM (regions where ego struggles)
        hard_mask = him_outputs.get('accumulated_positive_mask', None)  # [B, num_classes, H, W]
        
        if hard_mask is not None:
            # Convert to single channel hard region indicator
            hard_regions = 1.0 - hard_mask.max(dim=1, keepdim=True)[0]  # [B, 1, H, W]
            
            # Decode importance weights from hard regions
            importance_weights = self.importance_decoder(hard_regions)  # [B, 1, H, W]
        else:
            # Fallback: uniform importance
            importance_weights = torch.ones(B, 1, H, W, device=ego_features.device) * 0.5
        
        # Simple weighted average of neighbor features
        neighbor_features = agent_features[:, 1:K]  # [B, K-1, C, H, W]
        
        # Weight each neighbor by importance in hard regions
        weighted_neighbors = []
        for k in range(min(K-1, record_len[0]-1)):
            neighbor_k = neighbor_features[:, k] * importance_weights
            weighted_neighbors.append(neighbor_k)
        
        if weighted_neighbors:
            # Average weighted neighbor features
            avg_neighbor_features = torch.stack(weighted_neighbors, dim=0).mean(dim=0)
            
            # Concatenate ego and averaged neighbor features
            combined_features = torch.cat([ego_features, avg_neighbor_features], dim=1)
            
            # Simple fusion
            fused = self.fusion_conv(combined_features)
            
            # Residual connection with ego features
            output = self.output_conv(fused) + ego_features
        else:
            output = ego_features
            
        return output