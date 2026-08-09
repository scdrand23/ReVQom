import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime

def viz_3d_voxels(batch_dict, save_dir='viz_3d'):
    """
    Create 3D visualizations of voxel features
    """
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if 'encoded_spconv_tensor' not in batch_dict:
        print("No sparse tensor found")
        return
    
    sparse = batch_dict['encoded_spconv_tensor']
    
    # Method 1: Visualize sparse voxel locations in 3D space
    print("\nCreating 3D voxel visualizations...")
    
    # Get voxel coordinates and features
    indices = sparse.indices.detach().cpu().numpy()  # (N, 4) [batch, z, y, x]
    features = sparse.features.detach().cpu().numpy()  # (N, C)
    
    # Filter to first batch
    batch_0_mask = indices[:, 0] == 0
    coords = indices[batch_0_mask, 1:]  # [z, y, x]
    feats = features[batch_0_mask]
    
    # Compute feature magnitude for coloring
    feat_magnitude = np.linalg.norm(feats, axis=1)
    
    # Create 3D scatter plot of active voxels
    fig = plt.figure(figsize=(15, 12))
    
    # Plot 1: 3D scatter of voxel positions colored by feature magnitude
    ax1 = fig.add_subplot(221, projection='3d')
    
    # Subsample if too many points
    if len(coords) > 5000:
        idx = np.random.choice(len(coords), 5000, replace=False)
        coords_vis = coords[idx]
        feat_vis = feat_magnitude[idx]
    else:
        coords_vis = coords
        feat_vis = feat_magnitude
    
    scatter = ax1.scatter(coords_vis[:, 2], coords_vis[:, 1], coords_vis[:, 0],
                         c=feat_vis, cmap='viridis', s=10, alpha=0.6)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z (depth)')
    ax1.set_title(f'3D Active Voxels ({len(coords)} total, showing {len(coords_vis)})')
    plt.colorbar(scatter, ax=ax1, label='Feature Magnitude')
    
    # Plot 2: 3D density plot
    ax2 = fig.add_subplot(222, projection='3d')
    
    # Create density grid
    x_bins = np.linspace(coords[:, 2].min(), coords[:, 2].max(), 20)
    y_bins = np.linspace(coords[:, 1].min(), coords[:, 1].max(), 20)
    z_bins = np.linspace(coords[:, 0].min(), coords[:, 0].max(), 3)
    
    H, edges = np.histogramdd(coords, bins=[z_bins, y_bins, x_bins])
    
    # Plot density as 3D bars
    xpos, ypos, zpos = np.meshgrid(edges[2][:-1], edges[1][:-1], edges[0][:-1], indexing='ij')
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = zpos.ravel()
    
    dx = dy = dz = 1
    values = H.T.ravel()
    
    # Only plot non-zero densities
    mask = values > 0
    colors = plt.cm.viridis(values[mask] / values.max())
    
    ax2.bar3d(xpos[mask], ypos[mask], zpos[mask], dx, dy, values[mask]/10, 
             color=colors, alpha=0.7)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Density')
    ax2.set_title('Voxel Density Distribution')
    
    # Plot 3: Feature distribution by depth
    ax3 = fig.add_subplot(223)
    
    for z in np.unique(coords[:, 0]):
        z_mask = coords[:, 0] == z
        z_feats = feat_magnitude[z_mask]
        ax3.hist(z_feats, bins=30, alpha=0.5, label=f'Depth {int(z)}')
    
    ax3.set_xlabel('Feature Magnitude')
    ax3.set_ylabel('Count')
    ax3.set_title('Feature Distribution by Depth')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: XY projection with depth info
    ax4 = fig.add_subplot(224)
    
    # Color by depth
    scatter2d = ax4.scatter(coords[:, 2], coords[:, 1], c=coords[:, 0], 
                          cmap='coolwarm', s=1, alpha=0.5)
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.set_title('XY Projection (colored by depth)')
    plt.colorbar(scatter2d, ax=ax4, label='Z depth')
    ax4.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'voxels_3d_{timestamp}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved 3D visualization to voxels_3d_{timestamp}.png")
    
    # Create additional 3D views from different angles
    fig = plt.figure(figsize=(18, 6))
    
    angles = [(30, 30), (30, 120), (60, 60)]
    titles = ['View 1 (30°, 30°)', 'View 2 (30°, 120°)', 'View 3 (60°, 60°)']
    
    for i, (elev, azim) in enumerate(angles):
        ax = fig.add_subplot(1, 3, i+1, projection='3d')
        
        scatter = ax.scatter(coords_vis[:, 2], coords_vis[:, 1], coords_vis[:, 0],
                           c=feat_vis, cmap='plasma', s=5, alpha=0.7)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(titles[i])
        ax.view_init(elev=elev, azim=azim)
        
        if i == 2:  # Add colorbar to last subplot
            plt.colorbar(scatter, ax=ax, label='Feature Magnitude', fraction=0.046, pad=0.1)
    
    plt.suptitle(f'3D Voxel Features - Multiple Views\nShape: {sparse.dense().shape}, Active voxels: {len(coords)}')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'voxels_3d_views_{timestamp}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved multiple 3D views to voxels_3d_views_{timestamp}.png")
    
    # Save dense tensor slices for full 3D understanding
    dense = sparse.dense()  # (B, C, D, H, W)
    B, C, D, H, W = dense.shape
    
    # Create slice visualization
    fig, axes = plt.subplots(D, 4, figsize=(16, D*4))
    if D == 1:
        axes = axes.reshape(1, -1)
    
    for d in range(D):
        # Show different channel aggregations for each depth
        depth_slice = dense[0, :, d].detach().cpu().numpy()  # (C, H, W)
        
        # Mean across channels
        axes[d, 0].imshow(depth_slice.mean(axis=0), cmap='viridis')
        axes[d, 0].set_title(f'Depth {d}: Mean')
        axes[d, 0].axis('off')
        
        # Max across channels
        axes[d, 1].imshow(depth_slice.max(axis=0), cmap='hot')
        axes[d, 1].set_title(f'Depth {d}: Max')
        axes[d, 1].axis('off')
        
        # Std across channels
        axes[d, 2].imshow(depth_slice.std(axis=0), cmap='coolwarm')
        axes[d, 2].set_title(f'Depth {d}: Std')
        axes[d, 2].axis('off')
        
        # First channel
        axes[d, 3].imshow(depth_slice[0], cmap='plasma')
        axes[d, 3].set_title(f'Depth {d}: Channel 0')
        axes[d, 3].axis('off')
    
    plt.suptitle(f'3D Voxel Slices at Each Depth (Total depths: {D})')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'voxels_depth_slices_{timestamp}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved depth slices to voxels_depth_slices_{timestamp}.png")
    
    # Save statistics
    stats_file = os.path.join(save_dir, f'3d_stats_{timestamp}.txt')
    with open(stats_file, 'w') as f:
        f.write("3D VOXEL STATISTICS\n")
        f.write("="*50 + "\n\n")
        f.write(f"Dense shape (B, C, D, H, W): {dense.shape}\n")
        f.write(f"Total voxels: {B * D * H * W:,}\n")
        f.write(f"Active voxels: {len(indices):,}\n")
        f.write(f"Sparsity: {1 - len(indices)/(B*D*H*W):.2%}\n\n")
        
        f.write("Voxel distribution by depth:\n")
        for z in range(D):
            z_count = np.sum(indices[:, 1] == z)
            f.write(f"  Depth {z}: {z_count:,} voxels ({z_count/len(indices)*100:.1f}%)\n")
        
        f.write(f"\nFeature statistics:\n")
        f.write(f"  Min: {features.min():.6f}\n")
        f.write(f"  Max: {features.max():.6f}\n")
        f.write(f"  Mean: {features.mean():.6f}\n")
        f.write(f"  Std: {features.std():.6f}\n")
    
    print(f"Saved statistics to 3d_stats_{timestamp}.txt")
    print(f"\nAll 3D visualizations saved to {save_dir}/")
    
    return save_dir

# For use in pdb
# if __name__ == "__main__":
#     print("In pdb, run:")
#     print("  exec(open('viz_3d.py').read()); viz_3d_voxels(batch_dict)")
#     print("Or import:")
#     print("  import viz_3d; viz_3d.viz_3d_voxels(batch_dict)")