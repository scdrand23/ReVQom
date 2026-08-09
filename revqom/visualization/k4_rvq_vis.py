import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
import torch.nn.functional as F


def visualize_k4_compression(original, reconstructed, indices, codebooks, sample_id, save_path):
    """
    Simple visualization for K=4, n_q=1 case showing the 4-code compression
    """
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 4, figure=fig, hspace=0.4, wspace=0.3)
    
    # Move to CPU
    original = original.cpu().detach()
    reconstructed = reconstructed.cpu().detach()
    
    # For K=4, n_q=1, indices should be [1, H, W] with values 0-3
    if indices is not None:
        spatial_indices = indices[0].cpu().detach()  # First (and only) stage
        print(f"Spatial indices shape: {spatial_indices.shape}")
        print(f"Unique indices: {torch.unique(spatial_indices)}")
        
        # If flattened, reshape to spatial dimensions
        if len(spatial_indices.shape) == 1:
            # Assume square spatial dimensions
            spatial_size = int(np.sqrt(spatial_indices.shape[0]))
            if spatial_size * spatial_size == spatial_indices.shape[0]:
                spatial_indices = spatial_indices.view(spatial_size, spatial_size)
                print(f"Reshaped to: {spatial_indices.shape}")
            else:
                print(f"Cannot reshape {spatial_indices.shape[0]} to square - using 128x128")
                spatial_indices = spatial_indices.view(128, 128)
    else:
        print("Warning: No indices captured")
        spatial_indices = None
    
    # 1. Original BEV (average across channels)
    ax1 = fig.add_subplot(gs[0, 0])
    orig_vis = original[0].mean(0).numpy()
    im1 = ax1.imshow(orig_vis, cmap='viridis', aspect='auto')
    ax1.set_title(f'Original BEV\n({original.shape[1]} channels)')
    ax1.axis('off')
    plt.colorbar(im1, ax=ax1, fraction=0.046)
    
    # 2. Reconstructed BEV
    ax2 = fig.add_subplot(gs[0, 1])
    recon_vis = reconstructed[0].mean(0).numpy()
    im2 = ax2.imshow(recon_vis, cmap='viridis', aspect='auto')
    ax2.set_title(f'Reconstructed BEV\n({reconstructed.shape[1]} channels)')
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, fraction=0.046)
    
    # 3. 4-Color Code Map
    if spatial_indices is not None:
        ax3 = fig.add_subplot(gs[0, 2])
        spatial_map = spatial_indices.numpy()
        colors = ['red', 'green', 'blue', 'yellow']
        cmap = plt.matplotlib.colors.ListedColormap(colors)
        im3 = ax3.imshow(spatial_map, cmap=cmap, vmin=0, vmax=3, aspect='auto')
        ax3.set_title('Code Assignment\n(4 colors for 4 codes)')
        ax3.axis('off')
        
        # Colorbar with labels
        cbar = plt.colorbar(im3, ax=ax3, fraction=0.046, ticks=[0, 1, 2, 3])
        cbar.set_ticklabels(['Code 0', 'Code 1', 'Code 2', 'Code 3'])
    
    # 4. Compression Stats
    ax4 = fig.add_subplot(gs[0, 3])
    mse = F.mse_loss(original, reconstructed).item()
    psnr = 20 * np.log10(1.0 / (np.sqrt(mse) + 1e-8))
    
    # Bit calculation
    original_bits = original.shape[1] * 32  # 256 channels × 32 bits
    compressed_bits = 2  # log2(4) = 2 bits per pixel
    compression_ratio = original_bits / compressed_bits
    
    stats_text = f"""Extreme Compression:
    
Original: {original_bits} bits/pixel
K=4: {compressed_bits} bits/pixel
Ratio: {compression_ratio:.0f}x

Quality:
MSE: {mse:.4f}
PSNR: {psnr:.2f} dB

Only 4 prototype vectors
to represent ALL features!"""
    
    ax4.text(0.1, 0.5, stats_text, fontsize=11, transform=ax4.transAxes,
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightblue'))
    ax4.axis('off')
    
    # 5-8. Show the 4 learned codes
    if codebooks is not None:
        codebook = codebooks[0].cpu().numpy()  # [4, Cb] where Cb is compressed dims
        colors = ['red', 'green', 'blue', 'yellow']
        
        for i in range(4):
            ax = fig.add_subplot(gs[1, i])
            code_vector = codebook[i]  # [Cb] dimensions
            
            # Show as heatmap
            # Reshape to roughly square for visualization
            cb_size = len(code_vector)
            if cb_size == 16:
                code_2d = code_vector.reshape(4, 4)
            else:
                # Make it 1D visualization
                code_2d = code_vector.reshape(-1, 1)
            
            im = ax.imshow(code_2d, cmap='RdBu_r', aspect='auto')
            ax.set_title(f'Code {i}', color=colors[i], fontweight='bold', fontsize=12)
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046)
    
    # 9. Code Usage Statistics
    if spatial_indices is not None:
        ax9 = fig.add_subplot(gs[2, :2])
        indices_flat = spatial_indices.numpy().flatten()
        usage_counts = np.bincount(indices_flat, minlength=4)
        total_pixels = len(indices_flat)
        
        bars = ax9.bar(range(4), usage_counts, color=colors, alpha=0.7)
        ax9.set_title('Code Usage Across BEV', fontsize=12, fontweight='bold')
        ax9.set_xlabel('Code Index')
        ax9.set_ylabel('Pixel Count')
        ax9.set_xticks(range(4))
        
        # Add percentages
        for i, (bar, count) in enumerate(zip(bars, usage_counts)):
            height = bar.get_height()
            ax9.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{count}\\n({100*count/total_pixels:.1f}%)',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 10. Compression Comparison
    ax10 = fig.add_subplot(gs[2, 2:])
    methods = ['No Compression\\n(32-bit float)', 'K=4, n_q=1\\n(2 bits)', 'K=256, n_q=3\\n(24 bits)']
    bits = [8192, 2, 24]  # 256*32, log2(4), 3*log2(256)
    ratios = [8192/b for b in bits]
    
    bars = ax10.bar(methods, bits, color=['red', 'green', 'orange'], alpha=0.7)
    ax10.set_title('Bits per Pixel Comparison', fontsize=12, fontweight='bold')
    ax10.set_ylabel('Bits per Pixel')
    ax10.set_yscale('log')
    
    # Add compression ratios
    for bar, bit_count, ratio in zip(bars, bits, ratios):
        height = bar.get_height()
        ax10.text(bar.get_x() + bar.get_width()/2., height * 1.2,
                 f'{bit_count} bits\\n{ratio:.0f}x',
                 ha='center', va='bottom', fontweight='bold')
    
    ax10.grid(True, alpha=0.3, axis='y')
    
    # Main title
    fig.suptitle(f'Understanding RVQ: K=4 Compression (Sample {sample_id})\\n' +
                 'Every pixel mapped to 1 of 4 prototype vectors (2 bits each)',
                 fontsize=14, fontweight='bold')
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return {
        'mse': mse,
        'psnr': psnr,
        'compression_ratio': compression_ratio,
        'code_usage': usage_counts.tolist() if spatial_indices is not None else None
    }