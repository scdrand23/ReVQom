import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch


def convert_box_to_pixel_coords(box, gt_range, grid_size=128):
    """
    Convert box coordinates from metric to pixel coordinates
    
    Args:
        box: [x, y, z, dx, dy, dz, heading] in metric coordinates
        gt_range: [x_min, y_min, z_min, x_max, y_max, z_max]
        grid_size: BEV grid size (128 for 128x128)
    
    Returns:
        pixel_box: [x_pixel, y_pixel, width_pixel, height_pixel, heading]
    """
    x, y, z, dx, dy, dz, heading = box[:7]
    
    # Convert center to pixel coordinates
    x_pixel = (x - gt_range[0]) / (gt_range[3] - gt_range[0]) * grid_size
    y_pixel = (y - gt_range[1]) / (gt_range[4] - gt_range[1]) * grid_size
    
    # Convert size to pixel dimensions
    width_pixel = dx / (gt_range[3] - gt_range[0]) * grid_size
    height_pixel = dy / (gt_range[4] - gt_range[1]) * grid_size
    
    return [x_pixel, y_pixel, width_pixel, height_pixel, heading]


def overlay_boxes_on_bev(bev_image, boxes, gt_range, color='red', label='', grid_size=128):
    """
    Overlay bounding boxes on BEV feature visualization
    
    Args:
        bev_image: numpy array of BEV features (H, W)
        boxes: tensor or array of boxes [N, 7] (x, y, z, dx, dy, dz, heading)
        gt_range: detection range
        color: box color
        label: box label
        grid_size: BEV grid size
    
    Returns:
        fig, ax with boxes overlaid
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Show BEV image
    ax.imshow(bev_image, cmap='viridis', aspect='auto')
    
    if boxes is not None and len(boxes) > 0:
        if torch.is_tensor(boxes):
            boxes = boxes.cpu().numpy()
        
        for box in boxes:
            # Convert to pixel coordinates
            x_pixel, y_pixel, w_pixel, h_pixel, heading = convert_box_to_pixel_coords(
                box, gt_range, grid_size
            )
            
            # Create rotated rectangle
            # Note: matplotlib uses (x,y) as bottom-left corner
            rect = patches.Rectangle(
                (x_pixel - w_pixel/2, y_pixel - h_pixel/2),
                w_pixel, h_pixel,
                linewidth=2,
                edgecolor=color,
                facecolor='none',
                angle=np.degrees(heading),
                rotation_point='center'
            )
            ax.add_patch(rect)
    
    ax.set_xlim(0, grid_size)
    ax.set_ylim(grid_size, 0)  # Flip y-axis to match image coordinates
    ax.set_aspect('equal')
    ax.set_title(f'{label} Boxes on BEV Features')
    
    return fig, ax


def create_aligned_visualization(original_features, reconstructed_features, 
                                gt_boxes, pred_boxes, gt_range, sample_id=0):
    """
    Create side-by-side visualization with aligned GT and pred boxes
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Average features across channels for visualization
    orig_vis = original_features[0].mean(0).cpu().numpy()
    recon_vis = reconstructed_features[0].mean(0).cpu().numpy()
    
    # 1. Original with GT boxes
    axes[0].imshow(orig_vis, cmap='viridis', aspect='auto')
    if gt_boxes is not None and len(gt_boxes) > 0:
        boxes_np = gt_boxes.cpu().numpy() if torch.is_tensor(gt_boxes) else gt_boxes
        for box in boxes_np:
            x_pix, y_pix, w_pix, h_pix, heading = convert_box_to_pixel_coords(
                box, gt_range, orig_vis.shape[0]
            )
            rect = patches.Rectangle(
                (x_pix - w_pix/2, y_pix - h_pix/2),
                w_pix, h_pix,
                linewidth=2, edgecolor='green', facecolor='none',
                angle=np.degrees(heading), rotation_point='center'
            )
            axes[0].add_patch(rect)
    axes[0].set_title('Original + GT (green)')
    axes[0].axis('off')
    
    # 2. Original with Pred boxes
    axes[1].imshow(orig_vis, cmap='viridis', aspect='auto')
    if pred_boxes is not None and len(pred_boxes) > 0:
        boxes_np = pred_boxes.cpu().numpy() if torch.is_tensor(pred_boxes) else pred_boxes
        for box in boxes_np:
            x_pix, y_pix, w_pix, h_pix, heading = convert_box_to_pixel_coords(
                box, gt_range, orig_vis.shape[0]
            )
            rect = patches.Rectangle(
                (x_pix - w_pix/2, y_pix - h_pix/2),
                w_pix, h_pix,
                linewidth=2, edgecolor='red', facecolor='none',
                angle=np.degrees(heading), rotation_point='center'
            )
            axes[1].add_patch(rect)
    axes[1].set_title('Original + Pred (red)')
    axes[1].axis('off')
    
    # 3. Reconstructed with GT
    axes[2].imshow(recon_vis, cmap='viridis', aspect='auto')
    if gt_boxes is not None and len(gt_boxes) > 0:
        boxes_np = gt_boxes.cpu().numpy() if torch.is_tensor(gt_boxes) else gt_boxes
        for box in boxes_np:
            x_pix, y_pix, w_pix, h_pix, heading = convert_box_to_pixel_coords(
                box, gt_range, recon_vis.shape[0]
            )
            rect = patches.Rectangle(
                (x_pix - w_pix/2, y_pix - h_pix/2),
                w_pix, h_pix,
                linewidth=2, edgecolor='green', facecolor='none',
                angle=np.degrees(heading), rotation_point='center'
            )
            axes[2].add_patch(rect)
    axes[2].set_title('Reconstructed + GT')
    axes[2].axis('off')
    
    # 4. Reconstructed with Pred
    axes[3].imshow(recon_vis, cmap='viridis', aspect='auto')
    if pred_boxes is not None and len(pred_boxes) > 0:
        boxes_np = pred_boxes.cpu().numpy() if torch.is_tensor(pred_boxes) else pred_boxes
        for box in boxes_np:
            x_pix, y_pix, w_pix, h_pix, heading = convert_box_to_pixel_coords(
                box, gt_range, recon_vis.shape[0]
            )
            rect = patches.Rectangle(
                (x_pix - w_pix/2, y_pix - h_pix/2),
                w_pix, h_pix,
                linewidth=2, edgecolor='red', facecolor='none',
                angle=np.degrees(heading), rotation_point='center'
            )
            axes[3].add_patch(rect)
    axes[3].set_title('Reconstructed + Pred')
    axes[3].axis('off')
    
    plt.suptitle(f'Sample {sample_id}: BEV Features with Aligned Boxes')
    plt.tight_layout()
    
    return fig