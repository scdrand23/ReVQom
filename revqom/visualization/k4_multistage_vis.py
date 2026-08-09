import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
import torch.nn.functional as F
from revqom.utils import box_utils
import revqom.visualization.simple_plot3d.canvas_bev as canvas_bev


def visualize_k4_multistage(original, reconstructed, indices, codebooks, 
                            quantized_stages, gt_boxes=None, gt_labels=None,
                            pred_boxes=None, pred_scores=None,
                            sample_id=0, save_path=None):
    """
    Visualization for K=4 with multiple stages (n_q=3) showing progressive refinement
    and ground truth comparison with per-class analysis
    """
    # Define class information
    class_names = {1: 'Vehicle', 2: 'Pedestrian', 3: 'Truck'}
    class_colors = {
        1: [0, 255, 0],      # Green for vehicles
        2: [255, 165, 0],    # Orange for pedestrians  
        3: [0, 255, 255]     # Cyan for trucks
    }
    pred_class_colors = {
        1: [255, 0, 0],      # Red for vehicle predictions
        2: [255, 0, 255],    # Magenta for pedestrian predictions
        3: [0, 0, 255]       # Blue for truck predictions
    }
    
    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(4, 5, figure=fig, hspace=0.3, wspace=0.3)
    
    # Move to CPU
    original = original.cpu().detach()
    reconstructed = reconstructed.cpu().detach()
    
    # Get number of stages
    n_stages = indices.shape[0] if indices is not None else 1
    print(f"Visualizing {n_stages} stages of RVQ")
    
    # --- Row 1: Original, GT, and Progressive Stages ---
    
    # 1. Original BEV
    ax1 = fig.add_subplot(gs[0, 0])
    orig_vis = original[0].mean(0).numpy()
    im1 = ax1.imshow(orig_vis, cmap='viridis', aspect='auto')
    ax1.set_title('Original BEV\n(256 channels)', fontsize=10)
    ax1.axis('off')
    
    # 2. Ground Truth with proper BEV visualization
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Draw GT boxes using proper BEV canvas
    if gt_boxes is not None and len(gt_boxes) > 0:
        # Create BEV canvas with gray background to see boxes
        pc_range = [-102.4, -102.4, -10, 102.4, 102.4, 6]
        canvas = canvas_bev.Canvas_BEV_heading_right(
            canvas_shape=(128, 128),
            canvas_x_range=(pc_range[0], pc_range[3]), 
            canvas_y_range=(pc_range[1], pc_range[4]),
            left_hand=True
        )
        # Set gray background
        canvas.canvas[:] = [50, 50, 50]
        
        # Convert GT boxes from [N,7] to [N,8,3] corner format
        if torch.is_tensor(gt_boxes):
            gt_boxes_np = gt_boxes.cpu().numpy()
        else:
            gt_boxes_np = gt_boxes
        
        # Count objects by class
        gt_class_counts = {1: 0, 2: 0, 3: 0}
        if gt_labels is not None:
            if torch.is_tensor(gt_labels):
                gt_labels_np = gt_labels.cpu().numpy()
            else:
                gt_labels_np = gt_labels
            for label in gt_labels_np:
                if int(label) in gt_class_counts:
                    gt_class_counts[int(label)] += 1
        
        if len(gt_boxes_np.shape) == 3 and gt_boxes_np.shape[1] == 8:
            # Already in corner format [N,8,3] - use directly
            print(f"GT boxes already in corner format: {gt_boxes_np.shape}")
            
            # Color boxes by class
            if gt_labels is not None and len(gt_labels_np) == len(gt_boxes_np):
                gt_colors = np.array([class_colors.get(int(label), [255, 255, 255]) for label in gt_labels_np])
            else:
                gt_colors = np.array([[0, 255, 0]] * len(gt_boxes_np))  # Default green
            canvas.draw_boxes(gt_boxes_np, colors=gt_colors, texts=[''] * len(gt_boxes_np))
            
        elif len(gt_boxes_np.shape) == 2 and gt_boxes_np.shape[1] >= 7:
            # Convert from center format [N,7] to corner format [N,8,3]
            print(f"Converting GT boxes from center format: {gt_boxes_np.shape}")
            gt_corners = box_utils.boxes_to_corners_3d(
                torch.from_numpy(gt_boxes_np).float(), order='lwh')
            gt_corners_np = gt_corners.numpy()
            
            # Color boxes by class
            if gt_labels is not None and len(gt_labels_np) == len(gt_corners_np):
                gt_colors = np.array([class_colors.get(int(label), [255, 255, 255]) for label in gt_labels_np])
            else:
                gt_colors = np.array([[0, 255, 0]] * len(gt_corners_np))  # Default green
            canvas.draw_boxes(gt_corners_np, colors=gt_colors, texts=[''] * len(gt_corners_np))
        else:
            print(f"Unexpected GT boxes shape: {gt_boxes_np.shape}")
        
        # Display the canvas
        ax2.imshow(canvas.canvas, aspect='auto')
        
        # Create title with class breakdown
        gt_title = f'GT: {sum(gt_class_counts.values())} total\n'
        for class_id, count in gt_class_counts.items():
            if count > 0:
                gt_title += f'{class_names[class_id]}: {count} '
        ax2.set_title(gt_title.strip(), fontsize=10, color='green')
    else:
        ax2.imshow(orig_vis, cmap='viridis', aspect='auto')
        ax2.set_title('GT Not Available', fontsize=10)
    ax2.axis('off')
    
    # 3-5. Progressive Stages (for n_q=3)
    if quantized_stages is not None and len(quantized_stages) > 0:
        for stage_idx in range(min(3, len(quantized_stages))):
            ax = fig.add_subplot(gs[0, 2 + stage_idx])
            stage_features = quantized_stages[stage_idx].cpu().detach()
            stage_vis = stage_features[0].mean(0).numpy()
            ax.imshow(stage_vis, cmap='viridis', aspect='auto')
            ax.set_title(f'Stage {stage_idx+1}\nReconstruction', fontsize=10)
            ax.axis('off')
    else:
        # If no stages, show final reconstruction in stages
        for i in range(3):
            ax = fig.add_subplot(gs[0, 2 + i])
            ax.imshow(reconstructed[0].mean(0).numpy(), cmap='viridis', aspect='auto', 
                     alpha=0.3 + 0.3*i)  # Increasing opacity
            ax.set_title(f'Stage {i+1}\n(Simulated)', fontsize=10)
            ax.axis('off')
    
    # --- Row 2: Code Assignment Maps for Each Stage ---
    
    colors = ['red', 'green', 'blue', 'yellow']
    cmap = plt.matplotlib.colors.ListedColormap(colors)
    
    for stage_idx in range(n_stages):
        ax = fig.add_subplot(gs[1, stage_idx])
        
        # Get indices for this stage
        stage_indices = indices[stage_idx].cpu().detach()
        
        # Reshape if needed
        if len(stage_indices.shape) == 1:
            spatial_size = int(np.sqrt(stage_indices.shape[0]))
            stage_indices = stage_indices.view(spatial_size, spatial_size)
        
        im = ax.imshow(stage_indices.numpy(), cmap=cmap, vmin=0, vmax=3, aspect='auto')
        ax.set_title(f'Stage {stage_idx+1} Codes\n(2 bits)', fontsize=10)
        ax.axis('off')
        
        # Add colorbar for first stage
        if stage_idx == 0:
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, ticks=[0, 1, 2, 3])
            cbar.ax.tick_size = 8
    
    # Combined final code visualization
    if n_stages > 1:
        ax_final = fig.add_subplot(gs[1, 3])
        # Create combined visualization (simplified - you'd combine all stages)
        combined = indices[0].cpu().detach()
        if len(combined.shape) == 1:
            spatial_size = int(np.sqrt(combined.shape[0]))
            combined = combined.view(spatial_size, spatial_size)
        ax_final.imshow(combined.numpy(), cmap=cmap, vmin=0, vmax=3, aspect='auto')
        ax_final.set_title(f'Combined {n_stages} Stages\n({n_stages*2} bits total)', fontsize=10)
        ax_final.axis('off')
    
    # Prediction comparison with proper BEV visualization
    ax_pred = fig.add_subplot(gs[1, 4])
    if pred_boxes is not None and len(pred_boxes) > 0:
        # Create BEV canvas for predictions with gray background
        pc_range = [-102.4, -102.4, -10, 102.4, 102.4, 6]
        pred_canvas = canvas_bev.Canvas_BEV_heading_right(
            canvas_shape=(128, 128),
            canvas_x_range=(pc_range[0], pc_range[3]), 
            canvas_y_range=(pc_range[1], pc_range[4]),
            left_hand=True
        )
        # Set gray background
        pred_canvas.canvas[:] = [50, 50, 50]
        
        # Convert prediction boxes if they're in corner format already
        if torch.is_tensor(pred_boxes):
            pred_boxes_np = pred_boxes.cpu().numpy()
        else:
            pred_boxes_np = pred_boxes
        
        # Count predictions by class
        pred_class_counts = {1: 0, 2: 0, 3: 0}
        pred_labels_np = None
        if pred_scores is not None:
            if torch.is_tensor(pred_scores):
                pred_scores_np = pred_scores.cpu().numpy()
            else:
                pred_scores_np = pred_scores
            
            # Extract class labels from scores (assuming last column is class)
            if len(pred_scores_np.shape) == 2 and pred_scores_np.shape[1] >= 2:
                pred_labels_np = pred_scores_np[:, -1].astype(int)  # Last column is class
                for label in pred_labels_np:
                    if int(label) in pred_class_counts:
                        pred_class_counts[int(label)] += 1
        
        if len(pred_boxes_np.shape) == 3 and pred_boxes_np.shape[1] == 8:
            # Already in corner format [N,8,3]
            print(f"Pred boxes already in corner format: {pred_boxes_np.shape}")
            
            # Color boxes by predicted class
            if pred_labels_np is not None and len(pred_labels_np) == len(pred_boxes_np):
                pred_colors = np.array([pred_class_colors.get(int(label), [255, 255, 255]) for label in pred_labels_np])
            else:
                pred_colors = np.array([[255, 0, 0]] * len(pred_boxes_np))  # Default red
            pred_canvas.draw_boxes(pred_boxes_np, colors=pred_colors, texts=[''] * len(pred_boxes_np))
            
        elif len(pred_boxes_np.shape) == 2 and pred_boxes_np.shape[1] >= 7:
            # Convert from center format [N,7] to corner format [N,8,3]
            print(f"Converting pred boxes from center format: {pred_boxes_np.shape}")
            pred_corners = box_utils.boxes_to_corners_3d(
                torch.from_numpy(pred_boxes_np).float(), order='lwh')
            pred_corners_np = pred_corners.numpy()
            
            # Color boxes by predicted class
            if pred_labels_np is not None and len(pred_labels_np) == len(pred_corners_np):
                pred_colors = np.array([pred_class_colors.get(int(label), [255, 255, 255]) for label in pred_labels_np])
            else:
                pred_colors = np.array([[255, 0, 0]] * len(pred_corners_np))  # Default red
            pred_canvas.draw_boxes(pred_corners_np, colors=pred_colors, texts=[''] * len(pred_corners_np))
        else:
            print(f"Unexpected pred boxes shape: {pred_boxes_np.shape}")
        
        ax_pred.imshow(pred_canvas.canvas, aspect='auto')
        
        # Create title with class breakdown
        pred_title = f'Pred: {sum(pred_class_counts.values())} total\n'
        for class_id, count in pred_class_counts.items():
            if count > 0:
                pred_title += f'{class_names[class_id]}: {count} '
        ax_pred.set_title(pred_title.strip(), fontsize=10, color='orange')
    else:
        ax_pred.imshow(reconstructed[0].mean(0).numpy(), cmap='viridis', aspect='auto')
        ax_pred.set_title('Final\nReconstructed', fontsize=10)
    ax_pred.axis('off')
    
    # --- Row 3: Code Usage Statistics for Each Stage ---
    
    for stage_idx in range(min(n_stages, 3)):
        ax = fig.add_subplot(gs[2, stage_idx])
        
        stage_indices = indices[stage_idx].cpu().detach()
        indices_flat = stage_indices.numpy().flatten()
        usage_counts = np.bincount(indices_flat, minlength=4)
        total_pixels = len(indices_flat)
        
        bars = ax.bar(range(4), usage_counts, color=colors, alpha=0.7)
        ax.set_title(f'Stage {stage_idx+1} Usage', fontsize=10)
        ax.set_xlabel('Code', fontsize=9)
        ax.set_ylabel('Pixels', fontsize=9)
        ax.set_xticks(range(4))
        
        # Add percentages
        for i, (bar, count) in enumerate(zip(bars, usage_counts)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{100*count/total_pixels:.1f}%',
                   ha='center', va='bottom', fontsize=8)
    
    # --- Row 3, Col 3-4: Compression Metrics ---
    
    ax_metrics = fig.add_subplot(gs[2, 3:])
    
    mse = F.mse_loss(original, reconstructed).item()
    psnr = 20 * np.log10(1.0 / (np.sqrt(mse) + 1e-8))
    
    # Calculate progressive improvement if we have stages
    stage_metrics = []
    # Note: quantized_stages are in compressed/encoded space (16 channels)
    # original is in full space (256 channels) - can't compare directly
    
    metrics_text = f"""K=4, n_q={n_stages} Performance:
    
Compression: {8192/(n_stages*2):.0f}x
Bits per pixel: {n_stages*2} bits
Total bits: {128*128*n_stages*2:,} bits

Final Quality:
MSE: {mse:.4f}
PSNR: {psnr:.2f} dB"""
    
    # Removed per-stage PSNR since we can't compare encoded vs original
    
    ax_metrics.text(0.1, 0.5, metrics_text, fontsize=10, transform=ax_metrics.transAxes,
                   verticalalignment='center', 
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    ax_metrics.axis('off')
    
    # --- Row 4: Show the 4 Codebook Vectors ---
    
    if codebooks is not None:
        # For multi-stage, we might have multiple codebooks
        # Show the first stage's codebook
        codebook = codebooks[0].cpu().numpy() if n_stages > 0 else codebooks.cpu().numpy()
        
        for i in range(4):
            ax = fig.add_subplot(gs[3, i])
            code_vector = codebook[i]
            
            # Reshape for visualization
            if len(code_vector) == 16:
                code_2d = code_vector.reshape(4, 4)
            else:
                code_2d = code_vector.reshape(-1, 1)
            
            im = ax.imshow(code_2d, cmap='RdBu_r', aspect='auto')
            ax.set_title(f'Code {i}', color=colors[i], fontweight='bold', fontsize=10)
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046)
    
    # Bit comparison
    ax_bits = fig.add_subplot(gs[3, 4])
    methods = ['Uncompressed', f'K=4,n={n_stages}', 'K=256,n=3']
    bits = [8192, n_stages*2, 24]
    colors_bar = ['red', 'green', 'orange']
    
    bars = ax_bits.bar(methods, bits, color=colors_bar, alpha=0.7)
    ax_bits.set_ylabel('Bits/pixel', fontsize=9)
    ax_bits.set_yscale('log')
    ax_bits.set_title('Compression', fontsize=10)
    
    for bar, bit_count in zip(bars, bits):
        height = bar.get_height()
        ax_bits.text(bar.get_x() + bar.get_width()/2., height * 1.1,
                    f'{bit_count}', ha='center', va='bottom', fontsize=8)
    
    # Main title
    fig.suptitle(f'K=4 Multi-Stage RVQ Analysis (Sample {sample_id})\n' +
                 f'Progressive Refinement through {n_stages} Stages',
                 fontsize=14, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return {
        'mse': mse,
        'psnr': psnr,
        'compression_ratio': 8192 / (n_stages * 2),  # bits per pixel compression
        'n_stages': n_stages,
        'stage_metrics': stage_metrics if stage_metrics else None
    }