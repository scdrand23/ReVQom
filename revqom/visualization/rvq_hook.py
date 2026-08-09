import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import numpy as np


class RVQVisualizationHook:
    """
    Hook into ChannelRVQ_EMA to capture intermediate states for visualization.
    """
    
    def __init__(self, visualizer):
        self.visualizer = visualizer
        self.reset()
        
    def reset(self):
        """Reset captured data"""
        self.original_features = None
        self.encoded_features = None
        self.quantized_stages = []
        self.residuals = []
        self.indices = None
        self.reconstructed = None
        
    def register_hooks(self, model):
        """Register forward hooks on RVQ module"""
        # Find RVQ module - check different possible names
        rvq_module = None
        found_module_name = ""
        
        for name, module in model.named_modules():
            class_name = module.__class__.__name__
            if any(rvq_name in class_name for rvq_name in ['ChannelRVQ_EMA', 'ChannelRVQ', 'RVQ']):
                rvq_module = module
                found_module_name = name
                print(f"Found RVQ module: {name} ({class_name})")
                break
        
        if rvq_module is None:
            print("Warning: No RVQ module found. Available modules:")
            for name, module in model.named_modules():
                if 'compress' in name.lower() or 'rvq' in name.lower():
                    print(f"  - {name}: {module.__class__.__name__}")
            return
        
        # Hook to capture the input to encoder (original 256 channels)
        def encoder_input_hook(module, input, output):
            # Input to encoder is the original features
            self.original_features = input[0].detach().clone() if isinstance(input, tuple) else input.detach().clone()
            # Output is encoded features (16 channels)
            self.encoded_features = output.detach().clone()
            
        # Hook to capture decoder output (should be 256 channels)
        def decoder_hook(module, input, output):
            self.reconstructed = output.detach().clone()
        
        # Register hooks - use enc for encoder since enc_norm might not exist
        if hasattr(rvq_module, 'enc'):
            rvq_module.enc.register_forward_hook(encoder_input_hook)
        if hasattr(rvq_module, 'dec_norm'):
            rvq_module.dec_norm.register_forward_hook(decoder_hook)
        elif hasattr(rvq_module, 'dec'):
            # For ChannelRVQ which doesn't have dec_norm
            rvq_module.dec.register_forward_hook(decoder_hook)
        
        # Monkey-patch the _residual_quantize method to capture stages
        original_quantize = rvq_module._residual_quantize
        
        def patched_quantize(z):
            B, Cb, H, W = z.shape
            flat = z.permute(0, 2, 3, 1).reshape(-1, Cb).contiguous()
            
            residual = flat
            qsum = torch.zeros_like(flat)
            all_idx = []
            
            # z here is already encoded (16 channels), not the original
            
            # Clear previous stages
            self.quantized_stages = []
            self.residuals = []
            
            with torch.no_grad():
                for i in range(rvq_module.n_q):
                    # Capture residual before quantization
                    self.residuals.append(
                        residual.view(B, H, W, Cb).permute(0, 3, 1, 2).contiguous().detach().clone()
                    )
                    
                    # Quantization logic
                    cb = rvq_module.codebooks[i]
                    a2 = (residual**2).sum(dim=1, keepdim=True)
                    b2 = (cb**2).sum(dim=1).unsqueeze(0)
                    ab = residual @ cb.t()
                    dist = a2 + b2 - 2*ab
                    idx = dist.argmin(dim=1)
                    all_idx.append(idx)
                    q = cb[idx]
                    qsum = qsum + q
                    
                    # Capture cumulative quantized result
                    self.quantized_stages.append(
                        qsum.view(B, H, W, Cb).permute(0, 3, 1, 2).contiguous().detach().clone()
                    )
                    
                    residual = residual - q
            
            # Store indices
            self.indices = torch.stack(all_idx, dim=0).detach().clone()
            
            # Call original method for actual computation
            return original_quantize(z)
        
        rvq_module._residual_quantize = patched_quantize
        
        print("RVQ visualization hooks registered successfully")
    
    def visualize_current_batch(
        self, 
        gt_boxes: Optional[torch.Tensor] = None,
        pred_boxes: Optional[torch.Tensor] = None,
        frame_id: str = "frame_0",
        voxel_size: float = 0.2
    ):
        """Create visualization for current batch"""
        if self.original_features is None:
            print("No data captured yet")
            return None
            
        return self.visualizer.visualize_quantization_stages(
            self.original_features,
            self.quantized_stages,
            self.residuals,
            self.indices,
            gt_boxes,
            pred_boxes,
            frame_id,
            voxel_size
        )
    
    def get_compression_metrics(self) -> Dict:
        """Calculate compression metrics for current batch"""
        if self.original_features is None or self.reconstructed is None:
            return {}
            
        mse = torch.nn.functional.mse_loss(
            self.reconstructed, self.original_features
        ).item()
        
        psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else 100
        
        # Calculate codebook usage
        if self.indices is not None:
            usage_stats = []
            for i in range(self.indices.shape[0]):
                unique = torch.unique(self.indices[i])
                usage_stats.append(len(unique))
            avg_usage = np.mean(usage_stats)
        else:
            avg_usage = 0
        
        return {
            'mse': mse,
            'psnr': psnr,
            'avg_codebook_usage': avg_usage,
            'compression_ratio': 16.0  # Assuming 16x from config
        }