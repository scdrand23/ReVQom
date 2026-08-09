import torch
import torch.nn as nn
import torch.nn.functional as F

class ConcatQAFF(nn.Module):
    """
    Concatenation-based QAFF
    - Concatenates ego and neighbor features with HIM-guided weighting
    - Lets the detection head learn what to use
    - Maximum information preservation
    """
    def __init__(self, hidden_dim=256, num_heads=4, dropout=0.1, num_stages=3, num_classes=3, max_agents=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_agents = max_agents
        
        # Weight generator for each agent based on HIM outputs
        self.agent_weight_generator = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, max_agents, 1),
            nn.Softmax(dim=1)  # Weights sum to 1 across agents
        )
        
        # Channel reduction if needed (from K*C to C)
        self.channel_reducer = nn.Conv2d(hidden_dim * max_agents, hidden_dim, 1)
        
    def forward(self, him_outputs, agent_features, record_len):
        B, K, C, H, W = agent_features.shape
        
        # Get hard regions from HIM
        hard_mask = him_outputs.get('accumulated_positive_mask', None)
        
        if hard_mask is not None:
            hard_regions = 1.0 - hard_mask.max(dim=1, keepdim=True)[0]  # [B, 1, H, W]
            # Generate per-agent weights based on hard regions
            agent_weights = self.agent_weight_generator(hard_regions)  # [B, max_agents, H, W]
            agent_weights = agent_weights[:, :K]  # Take only valid agents
        else:
            # Equal weights for all agents
            agent_weights = torch.ones(B, K, H, W, device=agent_features.device) / K
        
        # Apply agent-specific spatial weights
        weighted_features = []
        for k in range(K):
            if k < record_len[0]:
                weight_k = agent_weights[:, k:k+1]  # [B, 1, H, W]
                weighted_feat = agent_features[:, k] * weight_k
                weighted_features.append(weighted_feat)
            else:
                # Pad with zeros for invalid agents
                weighted_features.append(torch.zeros_like(agent_features[:, 0]))
        
        # Pad to max_agents if needed
        while len(weighted_features) < self.max_agents:
            weighted_features.append(torch.zeros_like(agent_features[:, 0]))
        
        # Concatenate all weighted features
        concat_features = torch.cat(weighted_features, dim=1)  # [B, max_agents*C, H, W]
        
        # Reduce channels back to C
        output = self.channel_reducer(concat_features)  # [B, C, H, W]
        
        # Residual with ego
        output = output + agent_features[:, 0]
        
        return output