#!/usr/bin/env python
"""
Visualize RVQ reconstruction quality and stage-wise refinement.
Shows actual BEV features and how they're reconstructed through RVQ stages.
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import yaml
from pathlib import Path

from revqom.tools import train_utils, inference_utils
from revqom.data_utils.datasets import build_dataset
from torch.utils.data import DataLoader
import revqom.hypes_yaml.yaml_utils as yaml_utils


class RVQReconstructor(nn.Module):
    """Extract stage-wise reconstructions from RVQ"""
    
    def __init__(self, rvq_module):
        super().__init__()
        self.rvq = rvq_module
        
    def get_stage_reconstructions(self, x):
        """Get reconstruction at each RVQ stage"""
        B, C, H, W = x.shape
        
        # Encode
        z = self.rvq.encode(x)
        
        # Get reconstructions at each stage
        flat = z.permute(0, 2, 3, 1).reshape(-1, self.rvq.Cb).contiguous()
        
        stage_reconstructions = []
        residual = flat
        cumulative_q = torch.zeros_like(flat)
        
        for i in range(self.rvq.n_q):
            # Find nearest codebook entry
            cb = self.rvq.codebooks[i]
            
            # L2 distance
            a2 = (residual**2).sum(dim=1, keepdim=True)
            b2 = (cb**2).sum(dim=1).unsqueeze(0)
            ab = residual @ cb.t()
            dist = a2 + b2 - 2*ab
            
            idx = dist.argmin(dim=1)
            q = cb[idx]
            
            # Add to cumulative quantization
            cumulative_q = cumulative_q + q
            
            # Reshape and decode current cumulative
            z_q = cumulative_q.view(B, H, W, self.rvq.Cb).permute(0, 3, 1, 2).contiguous()
            x_recon = self.rvq.decode(z_q)
            
            stage_reconstructions.append(x_recon)
            
            # Update residual for next stage
            residual = residual - q.detach()
        
        return stage_reconstructions


def visualize_rvq_stages(model, batch_data, save_path, sample_idx=0):
    """Create meaningful visualization of RVQ stages"""
    
    # Get compressor
    compressor = None
    for name, module in model.named_modules():
        if name == 'compressor':
            compressor = module
            break
    
    if compressor is None:
        print("No compressor found!")
        return
    
    # Forward through model
    with torch.no_grad():
        # First run inference to get the processed data format
        if 'processed_lidar' not in batch_data:
            # Run through inference utils to get proper format
            from revqom.tools import inference_utils
            from revqom.data_utils.datasets import build_dataset
            
            # Use inference method to get proper output
            if hasattr(inference_utils, 'inference_intermediate_fusion'):
                infer_result = inference_utils.inference_intermediate_fusion(
                    batch_data, model, None
                )
                output = infer_result
            else:
                output = {}
            
            # Now get the features we need by re-running parts
            # Get the actual voxel features from the batch
            if 'ego' in batch_data:
                # This is the format from the dataloader
                batch_data_proc = batch_data
            else:
                batch_data_proc = batch_data
        
        # Get original BEV features (before compression)
        # Re-run part of the forward pass
        if 'ego' in batch_data:
            # Get voxel features from ego
            voxel_dict = batch_data['ego']
            voxel_features = voxel_dict['voxel_features']
            voxel_coords = voxel_dict['voxel_coords']
            voxel_num_points = voxel_dict['voxel_num_points']
            record_len = batch_data['record_len']
        else:
            # Assume it's already in the right format
            voxel_features = batch_data.get('voxel_features')
            voxel_coords = batch_data.get('voxel_coords')
            voxel_num_points = batch_data.get('voxel_num_points')
            record_len = batch_data.get('record_len', torch.tensor([1]))
        
        batch_dict = {
            'voxel_features': voxel_features,
            'voxel_coords': voxel_coords,
            'voxel_num_points': voxel_num_points,
            'batch_size': torch.sum(record_len).cpu().numpy(),
            'record_len': record_len
        }
        
        # Forward to get BEV features
        batch_dict = model.mean_vfe(batch_dict)
        batch_dict = model.backbone_3d(batch_dict)
        batch_dict = model.height_compression(batch_dict)
        
        original_bev = batch_dict['spatial_features']  # [B, 256, H, W]
        
        # Get stage-wise reconstructions
        reconstructor = RVQReconstructor(compressor)
        stage_recons = reconstructor.get_stage_reconstructions(original_bev)
        
    # Create visualization
    n_stages = len(stage_recons)
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, n_stages + 1, figure=fig, hspace=0.3, wspace=0.2)
    
    # Convert to numpy for visualization
    orig_np = original_bev[0].mean(0).cpu().numpy()  # Average across channels
    
    # Row 1: Progressive reconstruction
    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(orig_np, cmap='viridis', aspect='auto')
    ax.set_title("Original BEV", fontsize=12, fontweight='bold')
    ax.axis('off')
    
    for i, recon in enumerate(stage_recons):
        ax = fig.add_subplot(gs[0, i + 1])
        recon_np = recon[0].mean(0).cpu().numpy()
        ax.imshow(recon_np, cmap='viridis', aspect='auto')
        ax.set_title(f"After Stage {i+1}", fontsize=12, fontweight='bold')
        ax.axis('off')
    
    # Row 2: Reconstruction error at each stage
    for i, recon in enumerate(stage_recons):
        ax = fig.add_subplot(gs[1, i])
        recon_np = recon[0].mean(0).cpu().numpy()
        error = np.abs(orig_np - recon_np)
        im = ax.imshow(error, cmap='hot', aspect='auto', vmin=0, vmax=np.max(error))
        ax.set_title(f"Stage {i+1} Error", fontsize=11)
        ax.axis('off')
        
        # Add MSE text
        mse = torch.nn.functional.mse_loss(recon, original_bev).item()
        ax.text(0.5, -0.1, f"MSE: {mse:.4f}", transform=ax.transAxes,
                ha='center', fontsize=10)
    
    # Add final metrics
    ax = fig.add_subplot(gs[1, -1])
    final_recon = stage_recons[-1]
    final_mse = torch.nn.functional.mse_loss(final_recon, original_bev).item()
    final_psnr = 20 * np.log10(1.0 / np.sqrt(final_mse)) if final_mse > 0 else 100
    
    metrics_text = f"""Final Metrics:
    
MSE: {final_mse:.6f}
PSNR: {final_psnr:.2f} dB

Compression: 16x
Stages: {n_stages}
Codebook: {compressor.K} codes
Dims: 256 → {compressor.Cb}"""
    
    ax.text(0.5, 0.5, metrics_text, transform=ax.transAxes,
            fontsize=11, ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    ax.axis('off')
    
    # Row 3: Feature distributions
    for i, recon in enumerate(stage_recons):
        ax = fig.add_subplot(gs[2, i])
        
        # Plot histogram of feature values
        recon_flat = recon[0].flatten().cpu().numpy()
        orig_flat = original_bev[0].flatten().cpu().numpy()
        
        ax.hist(orig_flat, bins=50, alpha=0.5, label='Original', density=True, color='blue')
        ax.hist(recon_flat, bins=50, alpha=0.5, label=f'Stage {i+1}', density=True, color='red')
        ax.set_xlabel("Feature Value")
        ax.set_ylabel("Density")
        ax.legend(loc='upper right', fontsize=9)
        ax.set_title(f"Distribution Stage {i+1}", fontsize=11)
        ax.grid(True, alpha=0.3)
    
    # Add detection results if available
    if 'pred_box_tensor' in output:
        ax = fig.add_subplot(gs[2, -1])
        
        # Simple detection stats
        pred_boxes = output.get('pred_box_tensor', None)
        n_detections = len(pred_boxes) if pred_boxes is not None else 0
        
        det_text = f"""Detection Results:
        
Predictions: {n_detections}
With 16x compression

mAP@0.5: ~0.559
(from your results)"""
        
        ax.text(0.5, 0.5, det_text, transform=ax.transAxes,
                fontsize=11, ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        ax.axis('off')
    
    plt.suptitle(f"RVQ Stage-wise Reconstruction - Sample {sample_idx}", 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved visualization: {save_path}")
    return save_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--fusion_method', type=str, default='intermediate')
    parser.add_argument('--output_dir', type=str, default='./rvq_reconstruction_viz')
    parser.add_argument('--num_samples', type=int, default=5)
    
    args = parser.parse_args()
    
    # Load config
    hypes = yaml_utils.load_yaml(None, args)
    
    # Create dataset
    print("Building dataset...")
    dataset = build_dataset(hypes, visualize=True, train=False)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=dataset.collate_batch_test
    )
    
    # Create model
    print("Loading model...")
    model = train_utils.create_model(hypes)
    if torch.cuda.is_available():
        model.cuda()
    
    saved_path = args.model_dir
    _, model = train_utils.load_saved_model(saved_path, model)
    model.eval()
    
    # Create output directory
    Path(args.output_dir).mkdir(exist_ok=True, parents=True)
    
    # Process samples
    for i, batch_data in enumerate(dataloader):
        if i >= args.num_samples:
            break
        
        print(f"Processing sample {i}...")
        
        # Move to GPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        batch_data = train_utils.to_device(batch_data, device)
        
        # Create visualization
        save_path = Path(args.output_dir) / f"rvq_reconstruction_sample_{i:04d}.png"
        visualize_rvq_stages(model, batch_data, save_path, sample_idx=i)
    
    print(f"\nDone! Visualizations saved to {args.output_dir}")


if __name__ == '__main__':
    main()