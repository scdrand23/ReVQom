import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
import torch.nn.functional as F


def create_rvq_visualization(original, encoded, reconstructed, indices, codebooks, sample_id, save_path):
    """
    Create comprehensive RVQ visualization showing:
    - Original vs reconstructed BEV features
    - Stage-wise reconstruction
    - Codebook usage statistics
    - Reconstruction error heatmap
    """
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
    
    # Move to CPU for visualization
    original = original.cpu().detach()
    reconstructed = reconstructed.cpu().detach()
    encoded = encoded.cpu().detach() if encoded is not None else None
    
    # 1. Original feature map (average across channels)
    ax1 = fig.add_subplot(gs[0, 0])
    orig_vis = original[0].mean(0).numpy()  # Average across channels
    im1 = ax1.imshow(orig_vis, cmap='viridis', aspect='auto')
    ax1.set_title(f'Original BEV Features\n({original.shape[1]} channels)')
    ax1.axis('off')
    plt.colorbar(im1, ax=ax1, fraction=0.046)
    
    # 2. Reconstructed feature map
    ax2 = fig.add_subplot(gs[0, 1])
    recon_vis = reconstructed[0].mean(0).numpy()
    im2 = ax2.imshow(recon_vis, cmap='viridis', aspect='auto')
    ax2.set_title(f'Reconstructed BEV\n({reconstructed.shape[1]} channels)')
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, fraction=0.046)
    
    # 3. Error heatmap
    ax3 = fig.add_subplot(gs[0, 2])
    error = torch.abs(original - reconstructed)
    error_map = error[0].mean(0).numpy()
    im3 = ax3.imshow(error_map, cmap='hot', aspect='auto')
    ax3.set_title(f'Reconstruction Error\nMSE: {F.mse_loss(original, reconstructed):.4f}')
    ax3.axis('off')
    plt.colorbar(im3, ax=ax3, fraction=0.046)
    
    # 4. Metrics
    ax4 = fig.add_subplot(gs[0, 3])
    mse = F.mse_loss(original, reconstructed).item()
    psnr = 20 * np.log10(1.0 / (np.sqrt(mse) + 1e-8))
    
    # Calculate compression ratio
    orig_size = original.shape[1] * original.shape[2] * original.shape[3]
    if indices is not None:
        compressed_size = indices.shape[0] * indices.shape[1] * np.log2(codebooks.shape[1]) / 8
        compression_ratio = orig_size * 4 / compressed_size  # 4 bytes per float
    else:
        compression_ratio = original.shape[1] / encoded.shape[1] if encoded is not None else 0
    
    metrics_text = f"""Compression Metrics:
    
MSE: {mse:.6f}
PSNR: {psnr:.2f} dB
Compression Ratio: {compression_ratio:.1f}x

Original shape: {list(original.shape)}
Reconstructed: {list(reconstructed.shape)}"""
    
    if encoded is not None:
        metrics_text += f"\nEncoded shape: {list(encoded.shape)}"
    
    ax4.text(0.1, 0.5, metrics_text, fontsize=11, 
             transform=ax4.transAxes, verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    ax4.axis('off')
    
    # 5. Codebook usage (if indices available)
    if indices is not None and codebooks is not None:
        n_stages = min(3, indices.shape[0])
        for stage in range(n_stages):
            ax = fig.add_subplot(gs[1, stage])
            
            # Count usage of each code
            stage_indices = indices[stage].cpu().numpy().flatten()
            usage = np.bincount(stage_indices, minlength=codebooks.shape[1])
            
            # Plot usage histogram
            ax.bar(range(len(usage)), usage, color='steelblue', alpha=0.7)
            ax.set_title(f'Stage {stage+1} Code Usage')
            ax.set_xlabel('Code Index')
            ax.set_ylabel('Usage Count')
            ax.set_xlim([0, len(usage)])
            
            # Add statistics
            used_codes = np.sum(usage > 0)
            ax.text(0.98, 0.98, f'Active: {used_codes}/{len(usage)}',
                   transform=ax.transAxes, ha='right', va='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 6. Channel-wise reconstruction quality
    ax6 = fig.add_subplot(gs[2, :2])
    channel_mse = []
    for c in range(min(original.shape[1], 50)):  # Show first 50 channels
        channel_mse.append(F.mse_loss(original[0, c], reconstructed[0, c]).item())
    
    ax6.plot(channel_mse, 'b-', alpha=0.7, linewidth=1)
    ax6.fill_between(range(len(channel_mse)), channel_mse, alpha=0.3)
    ax6.set_title('Per-Channel Reconstruction Error')
    ax6.set_xlabel('Channel Index')
    ax6.set_ylabel('MSE')
    ax6.grid(True, alpha=0.3)
    
    # 7. Spatial error distribution
    ax7 = fig.add_subplot(gs[2, 2:])
    error_flat = error_map.flatten()
    ax7.hist(error_flat, bins=50, color='coral', alpha=0.7, edgecolor='black')
    ax7.set_title('Spatial Error Distribution')
    ax7.set_xlabel('Reconstruction Error')
    ax7.set_ylabel('Pixel Count')
    ax7.axvline(error_flat.mean(), color='red', linestyle='--', 
                label=f'Mean: {error_flat.mean():.4f}')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # Overall title
    fig.suptitle(f'RVQ Compression Analysis - Sample {sample_id}', fontsize=14, fontweight='bold')
    
    # Save figure
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return {
        'mse': mse,
        'psnr': psnr,
        'compression_ratio': compression_ratio,
        'active_codes': [np.sum(np.bincount(indices[i].cpu().numpy().flatten(), 
                                            minlength=codebooks.shape[1]) > 0) 
                         for i in range(indices.shape[0])] if indices is not None else []
    }