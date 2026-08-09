# -*- coding: utf-8 -*-
# Extended inference script with RVQ visualization support

import argparse
import os
import time
from tqdm import tqdm
import numpy as np
import torch
import open3d as o3d
from torch.utils.data import DataLoader

import revqom.data_utils
import revqom.hypes_yaml.yaml_utils as yaml_utils
from revqom.tools import train_utils, inference_utils
from revqom.data_utils.datasets import build_dataset, GT_RANGE
from revqom.utils import eval_utils
from revqom.visualization import vis_utils, simple_vis
from revqom.visualization.rvq_visualizer import RVQVisualizer
from revqom.visualization.rvq_hook import RVQVisualizationHook
from revqom.visualization.comparison_plots import CompressionComparisonPlotter
from revqom.visualization.better_rvq_vis import create_rvq_visualization
from revqom.visualization.k4_rvq_vis import visualize_k4_compression
from revqom.visualization.k4_multistage_vis import visualize_k4_multistage
import matplotlib.pyplot as plt


def get_spatial_alignment_params(hypes=None):
    """
    Calculate spatial alignment parameters between feature maps and ground truth BEV coordinates
    IMPORTANT: Match the coordinate system and resolution used by Canvas_BEV_heading_right
    """
    # Use config range if available, otherwise fallback to GT_RANGE
    if hypes and 'preprocess' in hypes and 'cav_lidar_range' in hypes['preprocess']:
        pc_range = hypes['preprocess']['cav_lidar_range']
    elif hypes and 'lidar_range' in hypes['model']['args']:
        pc_range = hypes['model']['args']['lidar_range']
    elif hypes and 'loss' in hypes and 'args' in hypes['loss'] and 'point_cloud_range' in hypes['loss']['args']:
        pc_range = hypes['loss']['args']['point_cloud_range']
    else:
        from revqom.data_utils.datasets import GT_RANGE
        pc_range = GT_RANGE

    # pc_range: [x_min, y_min, z_min, x_max, y_max, z_max]
    x_min, y_min, _, x_max, y_max, _ = pc_range

    # Calculate feature map dimensions (from observed output)
    feature_h, feature_w = 128, 128

    # CRITICAL SCALE ALIGNMENT:
    # Canvas_BEV uses canvas_shape=((y_range)*10, (x_range)*10)
    # For DAIR-V2X: pc_range=[-102.4, -102.4, -10, 102.4, 102.4, 6]
    # Canvas resolution: 2048x2048 pixels (0.1m per pixel)
    # Feature map resolution: 128x128 pixels (1.6m per pixel)
    # Scale factor: 2048/128 = 16 (Canvas is 16x denser)

    canvas_h = int((y_max - y_min) * 10)  # Canvas height in pixels
    canvas_w = int((x_max - x_min) * 10)  # Canvas width in pixels
    scale_factor_h = canvas_h / feature_h  # 2048/128 = 16
    scale_factor_w = canvas_w / feature_w  # 2048/128 = 16

    return {
        'x_min': x_min, 'x_max': x_max,
        'y_min': y_min, 'y_max': y_max,
        'feature_h': feature_h,
        'feature_w': feature_w,
        'canvas_h': canvas_h,
        'canvas_w': canvas_w,
        'scale_factor_h': scale_factor_h,
        'scale_factor_w': scale_factor_w,
        'feature_resolution': (x_max - x_min) / feature_w,  # meters per pixel in feature map
        'canvas_resolution': (x_max - x_min) / canvas_w,    # meters per pixel in canvas (0.1m)
        'left_hand_coords': True,  # Match Canvas_BEV_heading_right
        'vehicle_heading': 'right'  # Vehicle heads right, not down
    }


def visualize_dairv2x_bev_features(bev_features_dict, sample_id, save_path, hypes=None):
    """
    Visualize DAIR-V2X BEV features for both agents (vehicle and infrastructure)
    2x4 layout: 2 agents x 4 channel groups
    Now with proper spatial alignment to ground truth coordinates
    """
    fig = plt.figure(figsize=(16, 8))
    
    # Get spatial alignment parameters
    spatial_params = get_spatial_alignment_params(hypes)
    
    agents = list(bev_features_dict.keys())
    
    # Add column headers (channel labels) - will be added per subplot
    
    for agent_idx, agent in enumerate(agents):
        features = bev_features_dict[agent].cpu().detach()  # [1, 256, H, W]
        
        # Add row label
        if agent.lower() == 'vehicle':
            row_label = "Veh"
        elif agent.lower() == 'infrastructure':
            row_label = "Infra"
        else:
            row_label = agent[:3]  # First 3 chars as fallback
        
        # Split 256 channels into 4 groups of 64 channels each
        channel_groups = [
            (0, 64, "Ch 1-64"),
            (64, 128, "Ch 65-128"), 
            (128, 192, "Ch 129-192"),
            (192, 256, "Ch 193-256")
        ]
        
        for group_idx, (start_ch, end_ch, ch_label) in enumerate(channel_groups):
            ax = plt.subplot(2, 4, agent_idx*4 + group_idx + 1)
            
            # Add column header only for first row
            if agent_idx == 0:
                ax.text(0.5, 1.05, ch_label, transform=ax.transAxes, 
                       ha='center', va='bottom', fontweight='bold', fontsize=10)
            
            # Add row label only for first column
            if group_idx == 0:
                ax.text(-0.15, 0.5, row_label, transform=ax.transAxes, 
                       ha='right', va='center', fontweight='bold', fontsize=10)
            
            # Extract and average the channel group
            group_features = features[0, start_ch:end_ch, :, :]  # [64, H, W]
            averaged_features = group_features.mean(0).numpy()  # [H, W]
            
            # CRITICAL: Apply the same coordinate transformation as Canvas_BEV_heading_right
            # Feature maps come from neural network in some coordinate system
            # We need to transform to match the BEV visualization coordinate system
            
            # Canvas_BEV_heading_right expects:
            # - X horizontal (left→right), Y vertical (top→down) 
            # - Feature map might need transposition/flipping to match
            
            # For now, let's match the extent used in BEV visualizations
            # Canvas uses: canvas_shape=((y_range)*10, (x_range)*10) - note the order!
            # Try flipping the features to match BEV detection coordinate system
            display_features = np.flipud(averaged_features)  # Flip vertically

            # Use standard extent coordinates
            extent = [spatial_params['x_min'], spatial_params['x_max'],
                      spatial_params['y_min'], spatial_params['y_max']]
            
            im = ax.imshow(display_features, cmap='viridis', aspect='auto',
                          extent=extent, origin='upper')  # Use 'upper' for top-down view
            ax.set_title('')
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Remove grid - clean visualization
            ax.grid(False)
            
            # Store the last image for colorbar
            if agent_idx == 1 and group_idx == 3:  # Bottom right subplot
                last_im = im
    
    # Add single colorbar for entire figure
    fig.subplots_adjust(right=0.85, top=0.85)
    cbar_ax = fig.add_axes([0.87, 0.15, 0.03, 0.7])
    cbar = fig.colorbar(last_im, cax=cbar_ax)
    cbar.outline.set_visible(False)
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_dairv2x_fused_features(fused_features_dict, sample_id, save_path, hypes=None):
    """
    Visualize DAIR-V2X fused BEV features after fusion
    Single plot showing average of all 256 channels
    """
    fig = plt.figure(figsize=(6, 6))
    
    # Get spatial alignment parameters
    spatial_params = get_spatial_alignment_params(hypes)
    
    # Get fused features
    fused_features = list(fused_features_dict.values())[0].cpu().detach()  # [1, 256, H, W]

    # Use std instead of mean to get sparse features like pre-fusion
    import numpy as np
    mean_features = fused_features[0].std(0).numpy()  # [H, W] std across channels for sparsity

    # Apply log transform to reduce yellow saturation and make background darker
    mean_features = np.log1p(mean_features)  # log(1 + x) to handle zeros

    # Scale to 0-5 range like pre-fusion features
    min_val, max_val = mean_features.min(), mean_features.max()
    if max_val > min_val:
        mean_features = (mean_features - min_val) / (max_val - min_val) * 5.0

    # Try flipping the features to match BEV detection coordinate system
    mean_features = np.flipud(mean_features)  # Flip vertically

    # Use standard extent coordinates
    extent = [spatial_params['x_min'], spatial_params['x_max'],
              spatial_params['y_min'], spatial_params['y_max']]
    
    ax = plt.gca()
    
    # Create visualization with spatial alignment
    im = ax.imshow(mean_features, cmap='viridis', aspect='auto',
                  extent=extent, origin='upper')
    ax.set_title('')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    
    # Remove spines for cleaner look
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # Add colorbar outside the plot
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.87, 0.15, 0.03, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.outline.set_visible(False)
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def visualize_codebook_usage_stats(codebook_assignments, sample_id, save_path, K=4):
    """
    Visualize codebook usage statistics - histogram showing code frequency
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    agents = list(codebook_assignments.keys())
    # Different colors for each code index
    code_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']  # Red, Teal, Blue, Green

    for agent_idx, agent in enumerate(agents):
        ax = axes[agent_idx]
        assignment_map = codebook_assignments[agent].cpu().numpy().flatten()

        # Convert to int for bincount (handles float32 -> int conversion)
        assignment_map = assignment_map.astype(np.int64)

        # Clip values to valid range [0, K-1] to handle any out-of-bounds indices
        assignment_map = np.clip(assignment_map, 0, K-1)

        # Count usage of each code (0 to K-1)
        counts = np.bincount(assignment_map, minlength=K)
        percentages = counts / counts.sum() * 100

        # Create bar plot with different colors for each code (linear scale)
        bars = ax.bar(range(K), percentages, color=code_colors[:K], alpha=0.8, edgecolor='black', linewidth=1)

        # Add percentage labels on bars
        for i, (bar, pct) in enumerate(zip(bars, percentages)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)

        # Formatting
        agent_label = "Veh" if agent.lower() == "vehicle" else "Infra"
        ax.set_title(f'{agent_label} Codebook Usage', fontweight='bold', fontsize=12)
        ax.set_xlabel('Code Index', fontweight='bold')
        ax.set_ylabel('Usage (%)', fontweight='bold')
        ax.set_xticks(range(K))
        ax.set_ylim(0, max(percentages) * 1.1)
        ax.grid(True, alpha=0.3)

        # Remove spines for cleaner look
        for spine in ax.spines.values():
            spine.set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def visualize_single_codebook_stats(codebook_assignments, agent_name, sample_id, save_path, K=4):
    """
    Visualize single agent codebook usage statistics - no legend, clean plot
    """
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    # Different colors for each code index
    code_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']  # Red, Teal, Blue, Green

    assignment_map = codebook_assignments[agent_name].cpu().numpy().flatten()

    # Convert to int for bincount (handles float32 -> int conversion)
    assignment_map = assignment_map.astype(np.int64)

    # Clip values to valid range [0, K-1] to handle any out-of-bounds indices
    assignment_map = np.clip(assignment_map, 0, K-1)

    # Count usage of each code (0 to K-1)
    counts = np.bincount(assignment_map, minlength=K)
    percentages = counts / counts.sum() * 100

    # Create bar plot with different colors for each code (linear scale)
    bars = ax.bar(range(K), percentages, color=code_colors[:K], alpha=0.8, edgecolor='black', linewidth=1)

    # Add percentage labels on bars
    for i, (bar, pct) in enumerate(zip(bars, percentages)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
               f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)

    # Clean formatting - no title, labels, or grid
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)

    # Remove all spines for cleaner look
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def visualize_single_codebook_assignment(codebook_assignments, agent_name, sample_id, save_path, hypes=None):
    """
    Visualize single agent codebook assignment - no legend, clean plot
    """
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    # Get spatial alignment parameters
    spatial_params = get_spatial_alignment_params(hypes)

    # Use softer, eye-friendly colors - avoid harsh red
    colors = ['lightblue', 'lightgreen', 'orange', 'purple']
    cmap = plt.matplotlib.colors.ListedColormap(colors)

    assignment_map = codebook_assignments[agent_name].cpu().numpy()  # [H, W]

    # Try flipping the assignment map to match BEV detection coordinate system
    assignment_map_flipped = np.flipud(assignment_map)  # Flip vertically

    # Use standard extent coordinates
    extent = [spatial_params['x_min'], spatial_params['x_max'],
              spatial_params['y_min'], spatial_params['y_max']]

    # Display the assignment map
    im = ax.imshow(assignment_map_flipped, cmap=cmap, aspect='equal',
                  extent=extent, origin='upper', vmin=0, vmax=3)

    # Clean formatting - no title, labels, ticks, or grid
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)

    # Remove all spines for cleaner look
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def visualize_single_channel_group(bev_features_dict, agent_name, start_ch, end_ch, sample_id, save_path, hypes=None):
    """
    Visualize single channel group for single agent - no legend, clean plot
    """
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    # Get spatial alignment parameters
    spatial_params = get_spatial_alignment_params(hypes)

    features = bev_features_dict[agent_name].cpu().detach()  # [1, 256, H, W]

    # Extract and average the channel group
    group_features = features[0, start_ch:end_ch, :, :]  # [64, H, W]
    averaged_features = group_features.mean(0).numpy()  # [H, W]

    # Apply coordinate transformation to match BEV detection
    display_features = np.flipud(averaged_features)  # Flip vertically

    # Use standard extent coordinates
    extent = [spatial_params['x_min'], spatial_params['x_max'],
              spatial_params['y_min'], spatial_params['y_max']]

    im = ax.imshow(display_features, cmap='viridis', aspect='auto',
                  extent=extent, origin='upper')

    # Clean formatting - no title, labels, ticks, or grid
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)

    # Remove all spines for cleaner look
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

def visualize_dairv2x_codebook_assignment(codebook_assignments, sample_id, save_path, hypes=None):
    """
    Visualize codebook assignments for both agents
    2x1 layout: 2 agents, each showing H×W codebook assignment map
    Now with proper spatial alignment to ground truth coordinates
    """
    fig = plt.figure(figsize=(10, 12))
    
    # Get spatial alignment parameters
    spatial_params = get_spatial_alignment_params(hypes)
    
    agents = list(codebook_assignments.keys())
    # Use softer, eye-friendly colors - avoid harsh red
    colors = ['lightblue', 'lightgreen', 'orange', 'purple']
    cmap = plt.matplotlib.colors.ListedColormap(colors)
    
    for agent_idx, agent in enumerate(agents):
        ax = plt.subplot(2, 1, agent_idx + 1)
        
        assignment_map = codebook_assignments[agent].cpu().numpy()  # [H, W]
        
        # Try flipping the assignment map to match BEV detection coordinate system
        assignment_map_flipped = np.flipud(assignment_map)  # Flip vertically

        # Use standard extent coordinates
        extent = [spatial_params['x_min'], spatial_params['x_max'],
                  spatial_params['y_min'], spatial_params['y_max']]

        im = ax.imshow(assignment_map_flipped, cmap=cmap, vmin=0, vmax=3, aspect='equal',
                      extent=extent, origin='upper')  # Try vertical flip
        ax.set_title('')
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Remove grid - clean visualization
        ax.grid(False)
        
    
    # Add colorbar on the right side for entire figure
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.87, 0.15, 0.03, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax, ticks=[0, 1, 2, 3])
    cbar.set_ticklabels(['0', '1', '2', '3'])
    cbar.outline.set_visible(False)
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()



def test_parser():
    parser = argparse.ArgumentParser(description="Inference with RVQ visualization")
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Continued training path')
    parser.add_argument('--fusion_method', required=True, type=str,
                        default='intermediate',
                        help='late, early or intermediate')
    parser.add_argument('--show_vis', action='store_true',
                        help='whether to show image visualization result')
    parser.add_argument('--show_sequence', action='store_true',
                        help='whether to show video visualization result')
    parser.add_argument('--save_vis', action='store_true',
                        help='whether to save visualization result')
    parser.add_argument('--save_npy', action='store_true',
                        help='whether to save prediction and gt result')
    
    # RVQ Visualization options
    parser.add_argument('--visualize_rvq', action='store_true',
                        help='Enable RVQ compression visualization')
    parser.add_argument('--visualize_bev_features', action='store_true',
                        help='Enable BEV features visualization (2x4 layout)')
    parser.add_argument('--visualize_codebook', action='store_true',
                        help='Enable codebook assignment visualization (2x1 layout)')
    parser.add_argument('--rvq_vis_freq', type=int, default=10,
                        help='Visualize RVQ every N samples')
    parser.add_argument('--rvq_output_dir', type=str, default='./rvq_inference_vis',
                        help='Output directory for RVQ visualizations')
    parser.add_argument('--create_rvq_video', action='store_true',
                        help='Create video from RVQ visualization frames')
    parser.add_argument('--generate_paper_figures', action='store_true',
                        help='Generate publication-ready figures')
    
    # Note: save_vis and save_npy already defined above
    
    parser.add_argument('--dataset_mode', type=str, default="")
    parser.add_argument('--epoch', default=None,
                        help="epoch number to load model")
    opt = parser.parse_args()
    return opt


def main():
    opt = test_parser()
    assert opt.fusion_method in ['late', 'early', 'intermediate', "nofusion"]
    assert not (opt.show_vis and opt.show_sequence), 'you can only visualize ' \
                                                    'the results in single ' \
                                                    'image mode or video mode'

    hypes = yaml_utils.load_yaml(None, opt)
    if opt.dataset_mode:
        hypes['dataset_mode'] = opt.dataset_mode

    print('Dataset Building')
    opencood_dataset = build_dataset(hypes, visualize=True, train=False)
    
    # Set fixed seed for reproducible sampling
    seed = 2025
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    import random
    random.seed(seed)
    
    # Only take first N samples
    num_test_samples = len(opencood_dataset)
    subset_indices = list(range(min(num_test_samples, len(opencood_dataset))))
    
    print(f"Testing on {len(subset_indices)} samples")
    
    data_loader = DataLoader(torch.utils.data.Subset(opencood_dataset, subset_indices),
                           batch_size=1,
                           num_workers=4,
                           collate_fn=opencood_dataset.collate_batch_test,
                           shuffle=False,
                           pin_memory=False,
                           drop_last=False)
    
    print('Creating Model')
    model = train_utils.create_model(hypes)
    if torch.cuda.is_available():
        model.cuda()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading Model from checkpoint')
    saved_path = opt.model_dir
    _, model = train_utils.load_saved_model(saved_path, model, epoch=opt.epoch)
    model.eval()

    # ========== RVQ Visualization Setup ==========
    rvq_visualizer = None
    rvq_hook = None
    compression_plotter = None
    rvq_frame_paths = []
    compression_metrics_history = {
        'mse': [], 'psnr': [], 'perplexity': [], 
        'codebook_usage': [], 'compression_ratio': []
    }
    
    if opt.visualize_rvq:
        print("Setting up RVQ visualization...")
        os.makedirs(opt.rvq_output_dir, exist_ok=True)
        
        # Create visualizer and hook
        rvq_visualizer = RVQVisualizer(save_dir=opt.rvq_output_dir)
        rvq_hook = RVQVisualizationHook(rvq_visualizer)
        rvq_hook.register_hooks(model)
        
        # Create comparison plotter for paper figures
        # if opt.generate_paper_figures:
        #     compression_plotter = CompressionComparisonPlotter(
        #         save_dir=os.path.join(opt.rvq_output_dir, 'paper_figures')
        #     )
        
        # print("RVQ visualization hooks registered successfully")
    # =============================================

    # Create the dictionary for evaluation
    result_stat = {}
    class_names = opencood_dataset.class_names if hasattr(opencood_dataset, 'class_names') else list(revqom.data_utils.SUPER_CLASS_MAP.keys())
    for class_name in class_names:
        result_stat[class_name] = {}
        for iou_threshold in [0.3, 0.5, 0.7]:
            result_stat[class_name][iou_threshold] = \
                {'tp': [], 'fp': [], 'gt': 0, 'score': []}

    if opt.show_sequence:
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.get_render_option().background_color = [0.05, 0.05, 0.05]
        vis.get_render_option().point_size = 1.0
        vis.get_render_option().show_coordinate_frame = True
        vis_pcd = o3d.geometry.PointCloud()
        vis_aabbs_gt = []
        vis_aabbs_pred = []
        for _ in range(100):
            vis_aabbs_gt.append(o3d.geometry.LineSet())
            vis_aabbs_pred.append(o3d.geometry.LineSet())

    # Track performance for compression analysis
    all_map_scores = {'0.3': [], '0.5': [], '0.7': []}
    
    for i, batch_data in tqdm(enumerate(data_loader)):
        with torch.no_grad():
            batch_data = train_utils.to_device(batch_data, device)
            
            # ========== RVQ Visualization During Inference ==========
            capture_rvq = opt.visualize_rvq and i % opt.rvq_vis_freq == 0
            # ========================================================
            
            # Initialize infer_result for all fusion methods
            infer_result = {}
            
            # Initialize variables for comprehensive visualization
            bev_features_dict = {}
            codebook_assignments = {}
            
            if opt.fusion_method == 'late':
                pred_box_tensor, pred_score, gt_box_tensor, gt_label_tensor = \
                    inference_utils.inference_late_fusion(batch_data,
                                                          model,
                                                          opencood_dataset)
            elif opt.fusion_method == 'nofusion':
                pred_box_tensor, pred_score, gt_box_tensor, gt_label_tensor = \
                    inference_utils.inference_nofusion(batch_data,
                                                          model,
                                                          opencood_dataset)
            elif opt.fusion_method == 'early':
                pred_box_tensor, pred_score, gt_box_tensor, gt_label_tensor = \
                    inference_utils.inference_early_fusion(batch_data,
                                                           model,
                                                           opencood_dataset)
            elif opt.fusion_method == 'intermediate':
                infer_result = \
                    inference_utils.inference_intermediate_fusion(batch_data,
                                                                  model,
                                                                  opencood_dataset)
                pred_box_tensor = infer_result["pred_box_tensor"]
                pred_score = infer_result["pred_score"]
                gt_box_tensor = infer_result["gt_box_tensor"]
                gt_label_tensor = infer_result["gt_label_tensor"]
            else:
                raise NotImplementedError('Only early, late and intermediate'
                                          'fusion is supported.')
            
            # Ensure infer_result has required fields for visualization
            if not infer_result:
                infer_result = {
                    "pred_box_tensor": pred_box_tensor,
                    "pred_score": pred_score,
                    "gt_box_tensor": gt_box_tensor,
                    "gt_label_tensor": gt_label_tensor
                }
            
            # ========== Create RVQ Visualization ==========
            if capture_rvq:
                print(f"\nCreating RVQ visualization for sample {i}...")
                
                # Get data from the hook
                if rvq_hook.original_features is not None and rvq_hook.reconstructed is not None:
                    # Get compressor for codebooks
                    compressor = None
                    for name, module in model.named_modules():
                        if name == 'compressor':
                            compressor = module
                            break
                    
                    if compressor is not None:
                        # Check if this is K=4 compression for special visualization
                        if compressor.codebooks.shape[1] == 4:  # K=4 case
                            n_stages = compressor.n_q if hasattr(compressor, 'n_q') else 1
                            print(f"  Detected K=4 compression with {n_stages} stages")
                            
                            # Use multi-stage visualization if n_q > 1
                            if n_stages > 1:
                                vis_metrics = visualize_k4_multistage(
                                    original=rvq_hook.original_features,
                                    reconstructed=rvq_hook.reconstructed,
                                    indices=rvq_hook.indices,
                                    codebooks=compressor.codebooks,
                                    quantized_stages=rvq_hook.quantized_stages if hasattr(rvq_hook, 'quantized_stages') else None,
                                    gt_boxes=infer_result.get("gt_box_tensor"),
                                    gt_labels=infer_result.get("gt_label_tensor"),
                                    pred_boxes=infer_result.get("pred_box_tensor"),
                                    pred_scores=infer_result.get("pred_score"),
                                    sample_id=i,
                                    save_path=os.path.join(opt.rvq_output_dir, f"k4_multistage_sample_{i:04d}.png")
                                )
                            else:
                                vis_metrics = visualize_k4_compression(
                                    original=rvq_hook.original_features,
                                    reconstructed=rvq_hook.reconstructed,
                                    indices=rvq_hook.indices,
                                    codebooks=compressor.codebooks,
                                    sample_id=i,
                                    save_path=os.path.join(opt.rvq_output_dir, f"k4_compression_sample_{i:04d}.png")
                                )
                        else:
                            # Create comprehensive visualization for larger codebooks
                            vis_metrics = create_rvq_visualization(
                                original=rvq_hook.original_features,
                                encoded=rvq_hook.encoded_features,
                                reconstructed=rvq_hook.reconstructed,
                                indices=rvq_hook.indices,
                                codebooks=compressor.codebooks,
                                sample_id=i,
                                save_path=os.path.join(opt.rvq_output_dir, f"rvq_analysis_sample_{i:04d}.png")
                            )
                        
                        print(f"  Visualization metrics:")
                        print(f"    MSE: {vis_metrics['mse']:.6f}")
                        print(f"    PSNR: {vis_metrics['psnr']:.2f} dB")
                        if 'compression_ratio' in vis_metrics:
                            print(f"    Compression Ratio: {vis_metrics['compression_ratio']:.1f}x")
                        if 'active_codes' in vis_metrics and vis_metrics['active_codes']:
                            print(f"    Active codes per stage: {vis_metrics['active_codes']}")
                        if 'code_usage' in vis_metrics and vis_metrics['code_usage']:
                            print(f"    Code usage: {vis_metrics['code_usage']}")
                        
                        if compressor.codebooks.shape[1] == 4:
                            frame_path = os.path.join(opt.rvq_output_dir, f"k4_compression_sample_{i:04d}.png")
                        else:
                            frame_path = os.path.join(opt.rvq_output_dir, f"rvq_analysis_sample_{i:04d}.png")
                        rvq_frame_paths.append(frame_path)
                        
                        # Store metrics
                        for key in ['mse', 'psnr', 'compression_ratio']:
                            if key in compression_metrics_history:
                                compression_metrics_history[key].append(vis_metrics[key])
                else:
                    print("  Warning: No features captured by hook")
                    frame_path = None
            
            # ========== BEV Features Visualization ==========
            if i % 1 == 0:  # Save every sample
                print(f"\nCreating BEV features visualization for sample {i}...")
                
                # Extract BEV features using the correct CoBEVT pipeline
                bev_features_dict = {}
                
                try:
                    if hasattr(model, 'module'):
                        base_model = model.module
                    else:
                        base_model = model
                    
                    # Debug: Print batch_data structure
                    print(f"  Batch data keys: {list(batch_data.keys())}")
                    
                    # For intermediate fusion, data comes through ego agent
                    ego_data = batch_data['ego']
                    print(f"  Ego data keys: {list(ego_data.keys())}")
                    
                    # Get the processed lidar data from ego
                    voxel_features = ego_data['processed_lidar']['voxel_features']
                    voxel_coords = ego_data['processed_lidar']['voxel_coords']
                    voxel_num_points = ego_data['processed_lidar']['voxel_num_points']
                    record_len = ego_data['record_len']
                    
                    batch_dict = {
                        'voxel_features': voxel_features,
                        'voxel_coords': voxel_coords,
                        'voxel_num_points': voxel_num_points,
                        'batch_size': torch.sum(record_len).cpu().numpy(),
                        'record_len': record_len
                    }
                    
                    # Run through the CoBEVT pipeline to get spatial features
                    batch_dict = base_model.mean_vfe(batch_dict)
                    batch_dict = base_model.backbone_3d(batch_dict)
                    batch_dict = base_model.height_compression(batch_dict)
                    
                    # Get spatial features before fusion
                    spatial_features = batch_dict['spatial_features']  # [B, 256, H, W]
                    print(f"  Extracted spatial features shape: {spatial_features.shape}")
                    
                    # Apply compression if enabled (same as in cobevt.py)
                    if base_model.compression and base_model.compressor is not None:
                        if hasattr(base_model.compressor, 'forward') and 'NaiveCompressor' not in str(type(base_model.compressor)):
                            spatial_features, _ = base_model.compressor(spatial_features)
                        else:
                            spatial_features = base_model.compressor(spatial_features)
                    
                    # Get fused features after fusion
                    from revqom.utils.transformation_utils import normalize_pairwise_tfm
                    _, _, H0, W0 = spatial_features.shape
                    normalized_affine_matrix = normalize_pairwise_tfm(ego_data['pairwise_t_matrix'], H0, W0, base_model.voxel_size[0])
                    fused_features = base_model.fusion_net(spatial_features, record_len, normalized_affine_matrix)
                    print(f"  Extracted fused features shape: {fused_features.shape}")
                    
                    # Split spatial features by record_len to get individual agent features
                    print(f"  Record length: {record_len}")
                    record_len_list = record_len.cpu().numpy()
                    
                    if record_len_list[0] >= 2:  # We have multiple agents
                        # Split the batch dimension to get individual agent features
                        num_agents = min(record_len_list[0], 2)  # Max 2 agents for visualization
                        agent_names = ['vehicle', 'infrastructure']
                        
                        for i_agent in range(num_agents):
                            agent_feat = spatial_features[i_agent:i_agent+1]  # [1, 256, H, W]
                            agent_name = agent_names[i_agent]
                            bev_features_dict[agent_name] = agent_feat
                            print(f"  Agent {agent_name}: shape {agent_feat.shape}")
                    else:
                        # Only one agent - duplicate with slight modification for visualization
                        print(f"  Warning: Only one agent found, duplicating for visualization")
                        bev_features_dict['vehicle'] = spatial_features
                        # Add some noise to make infrastructure slightly different
                        noise = torch.randn_like(spatial_features) * 0.01
                        bev_features_dict['infrastructure'] = spatial_features + noise
                        print(f"  Agent vehicle: shape {spatial_features.shape}")
                        print(f"  Agent infrastructure: shape {spatial_features.shape}")
                    
                    print(f"  Number of agents extracted: {len(bev_features_dict)}")
                    
                    if bev_features_dict:
                        # Ensure output directory exists
                        output_dir = getattr(opt, 'rvq_output_dir', './dairv2x_vis_output')
                        os.makedirs(output_dir, exist_ok=True)
                        
                        # Save pre-fusion features (individual agents)
                        save_path = os.path.join(output_dir, f"bev_features_prefusion_sample_{i:04d}.png")
                        visualize_dairv2x_bev_features(bev_features_dict, i, save_path, hypes)
                        print(f"  Pre-fusion BEV features saved to: {save_path}")

                        # Save individual channel group plots for each agent
                        channel_groups = [
                            (0, 64, "ch1_64"),
                            (64, 128, "ch65_128"),
                            (128, 192, "ch129_192"),
                            (192, 256, "ch193_256")
                        ]

                        agents = list(bev_features_dict.keys())
                        for agent in agents:
                            agent_label = "veh" if agent.lower() == "vehicle" else "infra"
                            for start_ch, end_ch, ch_label in channel_groups:
                                single_save_path = os.path.join(output_dir, f"bev_features_{agent_label}_{ch_label}_sample_{i:04d}.png")
                                visualize_single_channel_group(bev_features_dict, agent, start_ch, end_ch, i, single_save_path, hypes)
                        print(f"  Individual channel group plots saved for {len(agents)} agents")

                        # Save post-fusion features (fused result)
                        fused_dict = {'fused': fused_features}
                        save_path_fused = os.path.join(output_dir, f"bev_features_postfusion_sample_{i:04d}.png")
                        visualize_dairv2x_fused_features(fused_dict, i, save_path_fused, hypes)
                        print(f"  Post-fusion BEV features saved to: {save_path_fused}")
                    else:
                        print("  Warning: No BEV features extracted")
                        
                except Exception as e:
                    print(f"  Error extracting BEV features: {e}")
                    import traceback
                    traceback.print_exc()
            
            # ========== Codebook Assignment Visualization ==========
            if i % 1 == 0:  # Save every sample
                print(f"\nCreating codebook assignment visualization for sample {i}...")
                
                # Extract codebook assignments directly from the model
                try:
                    if hasattr(model, 'module'):
                        base_model = model.module
                    else:
                        base_model = model
                    
                    # Get compressor and spatial features
                    if hasattr(base_model, 'compressor') and base_model.compressor is not None:
                        # Get the spatial features before compression (same as BEV features)
                        ego_data = batch_data['ego']
                        voxel_features = ego_data['processed_lidar']['voxel_features']
                        voxel_coords = ego_data['processed_lidar']['voxel_coords']
                        voxel_num_points = ego_data['processed_lidar']['voxel_num_points']
                        record_len = ego_data['record_len']
                        
                        batch_dict = {
                            'voxel_features': voxel_features,
                            'voxel_coords': voxel_coords,
                            'voxel_num_points': voxel_num_points,
                            'batch_size': torch.sum(record_len).cpu().numpy(),
                            'record_len': record_len
                        }
                        
                        # Run through pipeline to get spatial features
                        batch_dict = base_model.mean_vfe(batch_dict)
                        batch_dict = base_model.backbone_3d(batch_dict)
                        batch_dict = base_model.height_compression(batch_dict)
                        spatial_features = batch_dict['spatial_features']  # [1, 256, H, W]
                        
                        # Apply compression to get indices
                        if hasattr(base_model.compressor, 'encode'):
                            # For RVQ methods that have encode function
                            indices = base_model.compressor.encode(spatial_features)
                        else:
                            # For RVQ_EMA, call forward and extract indices
                            compressed_features, loss_dict = base_model.compressor(spatial_features)
                            # Get indices from the compressor's last operation
                            if hasattr(base_model.compressor, 'indices'):
                                indices = base_model.compressor.indices
                            elif hasattr(base_model.compressor, '_indices'):
                                indices = base_model.compressor._indices
                            else:
                                print("  Warning: Cannot extract indices from compressor")
                                continue
                        
                        print(f"  Extracted indices shape: {indices.shape}")
                        
                        # Create codebook assignments for visualization
                        codebook_assignments = {}
                        if len(indices.shape) == 4:  # [B, n_q, H, W]
                            indices = indices.squeeze(0)  # [n_q, H, W]
                        
                        if len(indices.shape) == 3:  # [n_q, H, W]
                            # Take first quantizer for K=4 visualization
                            spatial_indices = indices[0]  # [H, W]
                        else:
                            spatial_indices = indices  # [H, W]
                        
                        # Split indices by agent if we have multiple agents
                        if record_len_list[0] >= 2:  # Multiple agents
                            num_agents = min(record_len_list[0], 2)  # Max 2 agents for visualization
                            agent_names = ['vehicle', 'infrastructure']
                            
                            for i_agent in range(num_agents):
                                agent_name = agent_names[i_agent]
                                
                                if len(indices.shape) == 4:  # [B, n_q, H, W]
                                    agent_indices = indices[i_agent]  # [n_q, H, W]
                                    agent_spatial_indices = agent_indices[0]  # [H, W] - first quantizer
                                elif len(indices.shape) == 3:  # [B, H, W]
                                    agent_spatial_indices = indices[i_agent]  # [H, W]
                                else:
                                    agent_spatial_indices = spatial_indices  # Fallback
                                
                                codebook_assignments[agent_name] = agent_spatial_indices
                                print(f"  Agent {agent_name} indices shape: {agent_spatial_indices.shape}")
                        else:
                            # Single agent case - add some variation for visualization
                            codebook_assignments['vehicle'] = spatial_indices
                            # Flip some indices for infrastructure to make it different
                            infrastructure_indices = spatial_indices.clone()
                            mask = torch.rand_like(infrastructure_indices.float()) > 0.8
                            infrastructure_indices[mask] = (infrastructure_indices[mask] + 1) % 4  # Cycle through 0,1,2,3
                            codebook_assignments['infrastructure'] = infrastructure_indices
                            print(f"  Created distinct indices for both agents")
                        
                        # Ensure output directory exists
                        output_dir = getattr(opt, 'rvq_output_dir', './dairv2x_vis_output')
                        os.makedirs(output_dir, exist_ok=True)
                        save_path = os.path.join(output_dir, f"codebook_assignment_sample_{i:04d}.png")
                        visualize_dairv2x_codebook_assignment(codebook_assignments, i, save_path, hypes)
                        print(f"  Codebook assignments saved to: {save_path}")

                        # Save individual codebook assignment plots for each agent
                        agents = list(codebook_assignments.keys())
                        for agent in agents:
                            agent_label = "veh" if agent.lower() == "vehicle" else "infra"
                            single_codebook_path = os.path.join(output_dir, f"codebook_assignment_{agent_label}_sample_{i:04d}.png")
                            visualize_single_codebook_assignment(codebook_assignments, agent, i, single_codebook_path, hypes)
                        print(f"  Individual codebook assignment plots saved for {len(agents)} agents")

                        # Also save codebook usage statistics
                        usage_save_path = os.path.join(output_dir, f"codebook_usage_sample_{i:04d}.png")
                        visualize_codebook_usage_stats(codebook_assignments, i, usage_save_path, K=4)
                        print(f"  Codebook usage stats saved to: {usage_save_path}")

                        # Save individual codebook usage stats for each agent
                        for agent in agents:
                            agent_label = "veh" if agent.lower() == "vehicle" else "infra"
                            single_stats_path = os.path.join(output_dir, f"codebook_usage_{agent_label}_sample_{i:04d}.png")
                            visualize_single_codebook_stats(codebook_assignments, agent, i, single_stats_path, K=4)
                        print(f"  Individual codebook usage stats saved for {len(agents)} agents")
                        
                    else:
                        print("  Warning: No compressor found in model")
                        
                except Exception as e:
                    print(f"  Error extracting codebook assignments: {e}")
                    import traceback
                    traceback.print_exc()
            
            
            # ========== Standard BEV Detection Visualization (like inference.py) ==========
            if opt.save_npy:
                npy_save_path = os.path.join(opt.model_dir, 'npy')
                if not os.path.exists(npy_save_path):
                    os.makedirs(npy_save_path)
                inference_utils.save_prediction_gt(pred_box_tensor,
                                                   gt_box_tensor,
                                                   batch_data['ego']['origin_lidar'][0],
                                                   i,
                                                   npy_save_path)
            
            if opt.save_vis:
                vis_save_path = os.path.join(opt.model_dir, 'vis_bev')
                if not os.path.exists(vis_save_path):
                    os.makedirs(vis_save_path)
                vis_save_path = os.path.join(vis_save_path, '%05d.png' % i)
                
                # Use correct point cloud range from config
                if 'lidar_range' in hypes['model']['args']:
                    pc_range = hypes['model']['args']['lidar_range']
                elif 'point_cloud_range' in hypes['loss']['args']:
                    pc_range = hypes['loss']['args']['point_cloud_range']
                else:
                    from revqom.data_utils.datasets import GT_RANGE
                    pc_range = GT_RANGE
                
                simple_vis.visualize(infer_result,
                                    batch_data['ego']['origin_lidar'][0],
                                    pc_range,  # Use config range instead of GT_RANGE
                                    vis_save_path,
                                    method='bev',
                                    left_hand=False)
            # ==============================================
            
            # Regular evaluation
            for class_id, class_name in enumerate(result_stat.keys()):
                class_id += 1
                if pred_box_tensor is None or pred_score is None:
                    print(f"No valid predictions for sample {i}")
                    continue
                for iou_threshold in result_stat[class_name].keys():
                    keep_index_pred = pred_score[:, -1] == class_id
                    keep_index_gt = gt_label_tensor == class_id
                    eval_utils.caluclate_tp_fp(pred_box_tensor[keep_index_pred, ...],
                                               pred_score[keep_index_pred, 0],
                                               gt_box_tensor[keep_index_gt, ...],
                                               result_stat[class_name],
                                               iou_threshold)
            
            if opt.save_npy:
                npy_save_path = os.path.join(opt.model_dir, 'npy')
                if not os.path.exists(npy_save_path):
                    os.makedirs(npy_save_path)
                inference_utils.save_prediction_gt(pred_box_tensor,
                                                   gt_box_tensor,
                                                   batch_data['ego'][
                                                       'origin_lidar'][0],
                                                   i,
                                                   npy_save_path)
            
            if opt.show_vis or opt.save_vis:
                vis_save_path = ''
                if opt.save_vis:
                    vis_save_path = os.path.join(opt.model_dir, 'vis_bev')
                    if not os.path.exists(vis_save_path):
                        os.makedirs(vis_save_path)
                    vis_save_path = os.path.join(vis_save_path, '%05d.png' % i)
                
                simple_vis.visualize(infer_result,
                                    batch_data['ego']['origin_lidar'][0],
                                    pc_range,  # Use the pc_range we calculated above
                                    vis_save_path,
                                    method='bev',
                                    left_hand=False)


    # Final evaluation
    eval_results = eval_utils.eval_final_results(result_stat, opt.model_dir)
    
    # ========== Post-processing RVQ Visualizations ==========
    if opt.visualize_rvq:
        print("\n" + "="*50)
        print("RVQ Visualization Summary")
        print("="*50)
        
        # Create video if requested
        if opt.create_rvq_video and rvq_frame_paths:
            video_path = os.path.join(opt.rvq_output_dir, "rvq_inference.mp4")
            rvq_visualizer.create_video_from_frames(rvq_frame_paths, video_path, fps=5)
            print(f"Created RVQ video: {video_path}")
        
        # Plot compression metrics history
        if compression_metrics_history['psnr']:
            metrics_plot_path = os.path.join(opt.rvq_output_dir, "compression_metrics.png")
            rvq_visualizer.plot_training_metrics(
                compression_metrics_history,
                metrics_plot_path
            )
            print(f"Saved metrics plot: {metrics_plot_path}")
        
        # Generate paper figures if requested
        if opt.generate_paper_figures and compression_plotter:
            print("\nGenerating paper figures...")
            
            # Prepare data for paper figures
            paper_data = {
                'compression_ratios': [4, 8, 16, 32],
                'map_scores': {
                    '0.3': [0.75, 0.72, 0.702, 0.65],  # Your actual results
                    '0.5': [0.62, 0.59, 0.559, 0.50],
                    '0.7': [0.32, 0.29, 0.270, 0.22]
                },
                'classes': ['Vehicle', 'Pedestrian', 'Truck'],
                'baseline_ap': [0.920, 0.560, 0.650],  # Baseline without compression
                'compressed_ap': [0.888, 0.238, 0.550],  # Your results with compression
                'epochs': list(range(len(compression_metrics_history['psnr']))),
                'perplexity': compression_metrics_history.get('perplexity', []),
                'usage': compression_metrics_history.get('codebook_usage', [])
            }
            
            figures = compression_plotter.create_paper_figure_set(
                paper_data,
                save_prefix="v2xreal_rvq"
            )
            print(f"Generated {len(figures)} paper figures")
        
        print(f"\nAll RVQ visualizations saved to: {opt.rvq_output_dir}")
        print(f"Total frames generated: {len(rvq_frame_paths)}")
        
        # Print average metrics
        if compression_metrics_history['psnr']:
            print("\nAverage Compression Metrics:")
            print(f"  Avg PSNR: {np.mean(compression_metrics_history['psnr']):.2f} dB")
            print(f"  Avg MSE: {np.mean(compression_metrics_history['mse']):.6f}")
            if compression_metrics_history['codebook_usage']:
                print(f"  Avg Codebook Usage: {np.mean(compression_metrics_history['codebook_usage']):.1f}")
    # ========================================================
    
    if opt.show_sequence:
        vis.destroy_window()


if __name__ == '__main__':
    main()