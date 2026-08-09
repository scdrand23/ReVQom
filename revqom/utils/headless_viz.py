
import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime


def save_sparse_voxel_viz(sparse_tensor, save_dir='viz_output'):
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"\nSaving visualizations to {save_dir}/")
    
    dense = sparse_tensor.dense()
    B, C, D, H, W = dense.shape
    
    print(f"Sparse tensor info:")
    print(f"  Dense shape: {dense.shape}")
    print(f"  Sparse features: {sparse_tensor.features.shape}")
    print(f"  Indices: {sparse_tensor.indices.shape}")
    
    torch.save({
        'dense': dense.cpu(),
        'features': sparse_tensor.features.cpu(),
        'indices': sparse_tensor.indices.cpu(),
        'spatial_shape': sparse_tensor.spatial_shape,
        'batch_size': sparse_tensor.batch_size
    }, os.path.join(save_dir, f'sparse_tensor_{timestamp}.pt'))
    print(f"  Saved tensor to sparse_tensor_{timestamp}.pt")
    
    bev_max = dense[0].max(dim=1)[0].detach().cpu().numpy()
    bev_mean = dense[0].mean(dim=1).detach().cpu().numpy()
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    im1 = axes[0, 0].imshow(bev_max.mean(axis=0), cmap='viridis', aspect='auto')
    axes[0, 0].set_title('BEV Max Projection (avg channels)')
    axes[0, 0].set_xlabel('Width')
    axes[0, 0].set_ylabel('Height')
    plt.colorbar(im1, ax=axes[0, 0])
    
    im2 = axes[0, 1].imshow(bev_mean.mean(axis=0), cmap='hot', aspect='auto')
    axes[0, 1].set_title('BEV Mean Projection (avg channels)')
    axes[0, 1].set_xlabel('Width')
    axes[0, 1].set_ylabel('Height')
    plt.colorbar(im2, ax=axes[0, 1])
    
    im3 = axes[1, 0].imshow(bev_max[0], cmap='viridis', aspect='auto')
    axes[1, 0].set_title('BEV Max - Channel 0')
    plt.colorbar(im3, ax=axes[1, 0])
    
    feat_vals = sparse_tensor.features.detach().cpu().numpy()
    axes[1, 1].hist(feat_vals.flatten(), bins=50, alpha=0.7)
    axes[1, 1].set_title('Feature Value Distribution')
    axes[1, 1].set_xlabel('Value')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].axvline(0, color='r', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'sparse_voxel_viz_{timestamp}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved visualization to sparse_voxel_viz_{timestamp}.png")
    
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    axes = axes.flatten()
    
    for i in range(min(16, C)):
        channel_bev = bev_max[i]
        im = axes[i].imshow(channel_bev, cmap='viridis', aspect='auto')
        axes[i].set_title(f'Ch {i}')
        axes[i].axis('off')
    
    plt.suptitle('First 16 Channels - BEV Max Projection')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'channels_{timestamp}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved channel visualization to channels_{timestamp}.png")
    
    stats_file = os.path.join(save_dir, f'stats_{timestamp}.txt')
    with open(stats_file, 'w') as f:
        f.write("="*50 + "\n")
        f.write("SPARSE VOXEL TENSOR STATISTICS\n")
        f.write("="*50 + "\n\n")
        
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Dense shape (B,C,D,H,W): {dense.shape}\n")
        f.write(f"Sparse features shape: {sparse_tensor.features.shape}\n")
        f.write(f"Number of active voxels: {sparse_tensor.features.shape[0]}\n")
        f.write(f"Sparsity: {1 - sparse_tensor.features.shape[0] / (B*D*H*W):.2%}\n\n")
        
        f.write("Feature Statistics:\n")
        f.write(f"  Min: {sparse_tensor.features.min().item():.6f}\n")
        f.write(f"  Max: {sparse_tensor.features.max().item():.6f}\n")
        f.write(f"  Mean: {sparse_tensor.features.mean().item():.6f}\n")
        f.write(f"  Std: {sparse_tensor.features.std().item():.6f}\n")
        f.write(f"  % near zero (<0.01): {(sparse_tensor.features.abs() < 0.01).float().mean().item():.2%}\n")
        
        f.write("\nPer-Channel Statistics (first 10 channels):\n")
        for i in range(min(10, C)):
            ch_data = sparse_tensor.features[:, i].cpu()
            f.write(f"  Channel {i:2d}: mean={ch_data.mean():.4f}, std={ch_data.std():.4f}, "
                   f"min={ch_data.min():.4f}, max={ch_data.max():.4f}\n")
    
    print(f"  Saved statistics to stats_{timestamp}.txt")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    depth_indices = np.linspace(0, D-1, 6, dtype=int)
    
    for idx, d_idx in enumerate(depth_indices):
        slice_data = dense[0, :, d_idx].mean(dim=0).detach().cpu().numpy()
        im = axes[idx].imshow(slice_data, cmap='viridis', aspect='auto')
        axes[idx].set_title(f'Depth slice {d_idx}/{D}')
        axes[idx].axis('off')
        plt.colorbar(im, ax=axes[idx], fraction=0.046)
    
    plt.suptitle('Voxel Features at Different Depths')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'depth_slices_{timestamp}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved depth slices to depth_slices_{timestamp}.png")
    
    return save_dir


def save_bev_features(spatial_features, save_dir='viz_output'):
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"\nSaving BEV features to {save_dir}/")
    
    feat = spatial_features.detach().cpu()
    B, C, H, W = feat.shape
    
    torch.save(feat, os.path.join(save_dir, f'bev_features_{timestamp}.pt'))
    print(f"  Saved tensor to bev_features_{timestamp}.pt")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    bev_mean = feat[0].mean(dim=0).numpy()
    im1 = axes[0, 0].imshow(bev_mean, cmap='viridis', aspect='auto')
    axes[0, 0].set_title('BEV Mean (all channels)')
    plt.colorbar(im1, ax=axes[0, 0])
    
    bev_max = feat[0].max(dim=0)[0].numpy()
    im2 = axes[0, 1].imshow(bev_max, cmap='hot', aspect='auto')
    axes[0, 1].set_title('BEV Max (all channels)')
    plt.colorbar(im2, ax=axes[0, 1])
    
    bev_std = feat[0].std(dim=0).numpy()
    im3 = axes[0, 2].imshow(bev_std, cmap='coolwarm', aspect='auto')
    axes[0, 2].set_title('BEV Std (all channels)')
    plt.colorbar(im3, ax=axes[0, 2])
    
    for i in range(3):
        im = axes[1, i].imshow(feat[0, i].numpy(), cmap='viridis', aspect='auto')
        axes[1, i].set_title(f'Channel {i}')
        plt.colorbar(im, ax=axes[1, i])
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'bev_features_{timestamp}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved visualization to bev_features_{timestamp}.png")
    
    n_channels_to_show = min(32, C)
    grid_size = int(np.ceil(np.sqrt(n_channels_to_show)))
    
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(20, 20))
    axes = axes.flatten()
    
    for i in range(n_channels_to_show):
        im = axes[i].imshow(feat[0, i].numpy(), cmap='viridis', aspect='auto')
        axes[i].set_title(f'Ch {i}', fontsize=8)
        axes[i].axis('off')
    
    for i in range(n_channels_to_show, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f'First {n_channels_to_show} Channels')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'channel_grid_{timestamp}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved channel grid to channel_grid_{timestamp}.png")
    
    return save_dir


def debug_save_all(batch_dict, save_dir='viz_debug'):
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("\n" + "="*60)
    print(f"SAVING DEBUG VISUALIZATIONS TO: {save_dir}/")
    print("="*60)
    
    info_file = os.path.join(save_dir, f'batch_info_{timestamp}.txt')
    with open(info_file, 'w') as f:
        f.write("BATCH DICT CONTENTS\n")
        f.write("="*50 + "\n\n")
        
        for key in batch_dict.keys():
            f.write(f"{key}:\n")
            if hasattr(batch_dict[key], 'shape'):
                f.write(f"  Shape: {batch_dict[key].shape}\n")
            elif hasattr(batch_dict[key], 'dense'):
                f.write(f"  Sparse tensor - dense shape: {batch_dict[key].dense().shape}\n")
            else:
                f.write(f"  Type: {type(batch_dict[key])}\n")
                if isinstance(batch_dict[key], (int, float, list)):
                    f.write(f"  Value: {batch_dict[key]}\n")
            f.write("\n")
    
    print(f"Saved batch info to {info_file}")
    
    if 'encoded_spconv_tensor' in batch_dict:
        print("\nProcessing encoded_spconv_tensor...")
        save_sparse_voxel_viz(batch_dict['encoded_spconv_tensor'], save_dir)
    
    if 'spatial_features' in batch_dict:
        print("\nProcessing spatial_features...")
        save_bev_features(batch_dict['spatial_features'], save_dir)
    
    if 'multi_scale_3d_features' in batch_dict:
        print("\nSaving multi-scale 3D features...")
        for i, feat_dict in enumerate(batch_dict['multi_scale_3d_features']):
            torch.save(feat_dict, os.path.join(save_dir, f'multiscale_3d_{i}_{timestamp}.pt'))
        print(f"  Saved {len(batch_dict['multi_scale_3d_features'])} multi-scale features")
    
    print("\n" + "="*60)
    print(f"ALL FILES SAVED TO: {save_dir}/")
    print("You can download these files for local analysis")
    print("="*60 + "\n")
    
    return save_dir




