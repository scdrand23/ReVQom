import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import cv2
from pathlib import Path


class RVQVisualizer:
    """
    Comprehensive visualization for Residual Vector Quantization in BEV detection.
    Shows stage-wise quantization, compression quality, and detection performance.
    """
    
    def __init__(self, save_dir: str = "./vis_outputs"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
    def visualize_quantization_stages(
        self,
        original_features: torch.Tensor,
        quantized_stages: List[torch.Tensor],
        residuals: List[torch.Tensor],
        indices: torch.Tensor,
        gt_boxes: Optional[torch.Tensor] = None,
        pred_boxes: Optional[torch.Tensor] = None,
        frame_id: str = "frame_0",
        voxel_size: float = 0.2
    ):
        """
        Visualize multi-stage RVQ process with detection results.
        
        Args:
            original_features: [B, C, H, W] original BEV features
            quantized_stages: List of [B, C, H, W] features after each RVQ stage
            residuals: List of [B, C, H, W] residuals at each stage
            indices: [n_q, B*H*W] codebook indices used
            gt_boxes: Ground truth boxes [N, 7] (x, y, z, w, l, h, yaw)
            pred_boxes: Predicted boxes [M, 7]
        """
        B, C, H, W = original_features.shape
        n_stages = len(quantized_stages)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(24, 16))
        gs = GridSpec(4, n_stages + 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # === Row 1: Original vs Final Quantized ===
        ax_orig = fig.add_subplot(gs[0, 0])
        self._plot_feature_map(original_features[0], ax_orig, "Original Features")
        
        ax_final = fig.add_subplot(gs[0, -1])
        self._plot_feature_map(quantized_stages[-1][0], ax_final, "Final Quantized")
        
        # Compression stats
        ax_stats = fig.add_subplot(gs[0, 1:-1])
        self._plot_compression_stats(original_features, quantized_stages[-1], ax_stats)
        
        # === Row 2: Stage-wise Quantization ===
        for i, (quant, resid) in enumerate(zip(quantized_stages, residuals)):
            ax = fig.add_subplot(gs[1, i])
            self._plot_feature_map(quant[0], ax, f"Stage {i+1} Quantized")
            
            ax_res = fig.add_subplot(gs[1, n_stages + i])
            self._plot_residual_map(resid[0], ax_res, f"Stage {i+1} Residual")
        
        # === Row 3: Detection Results ===
        # Ground Truth
        ax_gt = fig.add_subplot(gs[2, :n_stages+1])
        self._plot_bev_with_boxes(
            original_features[0], gt_boxes, ax_gt, 
            "Ground Truth Detections", voxel_size, color='green'
        )
        
        # Predictions
        ax_pred = fig.add_subplot(gs[2, n_stages+1:])
        self._plot_bev_with_boxes(
            quantized_stages[-1][0], pred_boxes, ax_pred,
            "Predicted Detections (After Quantization)", voxel_size, color='red'
        )
        
        # === Row 4: Codebook Usage & Perplexity ===
        ax_usage = fig.add_subplot(gs[3, :n_stages])
        self._plot_codebook_usage(indices, ax_usage)
        
        ax_quality = fig.add_subplot(gs[3, n_stages:])
        self._plot_reconstruction_quality(original_features, quantized_stages, ax_quality)
        
        plt.suptitle(f"RVQ Visualization - {frame_id}", fontsize=16, fontweight='bold')
        
        # Save figure
        save_path = self.save_dir / f"rvq_vis_{frame_id}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def _plot_feature_map(self, features: torch.Tensor, ax, title: str):
        """Plot BEV feature map (channel-averaged)"""
        # Average across channels and convert to numpy
        feat_avg = features.mean(dim=0).cpu().numpy()
        
        im = ax.imshow(feat_avg, cmap='viridis', aspect='auto')
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    def _plot_residual_map(self, residual: torch.Tensor, ax, title: str):
        """Plot residual map with diverging colormap"""
        resid_avg = residual.mean(dim=0).cpu().numpy()
        
        # Use diverging colormap centered at 0
        vmax = np.abs(resid_avg).max()
        im = ax.imshow(resid_avg, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    def _plot_bev_with_boxes(
        self, features: torch.Tensor, boxes: Optional[torch.Tensor], 
        ax, title: str, voxel_size: float, color: str = 'red'
    ):
        """Plot BEV features with detection boxes overlaid"""
        # Plot feature map
        feat_avg = features.mean(dim=0).cpu().numpy()
        ax.imshow(feat_avg, cmap='gray', alpha=0.7, aspect='auto')
        
        if boxes is not None and len(boxes) > 0:
            boxes = boxes.cpu().numpy()
            H, W = features.shape[1:]
            
            for box in boxes:
                x, y, z, w, l, h, yaw = box[:7]
                
                # Convert to pixel coordinates
                px = int((x / voxel_size) + W // 2)
                py = int((y / voxel_size) + H // 2)
                pw = int(w / voxel_size)
                pl = int(l / voxel_size)
                
                # Create rotated rectangle
                rect = patches.Rectangle(
                    (px - pw//2, py - pl//2), pw, pl,
                    linewidth=2, edgecolor=color, facecolor='none',
                    angle=np.degrees(yaw), rotation_point='center'
                )
                ax.add_patch(rect)
                
                # Add arrow for orientation
                arrow_len = pl // 2
                arrow_x = px + arrow_len * np.cos(yaw)
                arrow_y = py + arrow_len * np.sin(yaw)
                ax.arrow(px, py, arrow_x - px, arrow_y - py,
                        head_width=3, head_length=2, fc=color, ec=color)
        
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.axis('off')
    
    def _plot_compression_stats(
        self, original: torch.Tensor, quantized: torch.Tensor, ax
    ):
        """Plot compression statistics and metrics"""
        # Calculate metrics
        mse = F.mse_loss(quantized, original).item()
        psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else 100
        
        # Calculate compression ratio (assuming indices storage)
        orig_size = original.numel() * 32  # 32 bits per float
        # Assuming log2(codebook_size) bits per index
        compressed_size = quantized.shape[2] * quantized.shape[3] * 8  # ~8 bits per index
        compression_ratio = orig_size / compressed_size
        
        # Plot metrics
        metrics_text = f"""Compression Metrics:
        
MSE: {mse:.6f}
PSNR: {psnr:.2f} dB
Compression Ratio: {compression_ratio:.1f}x

Original Size: {orig_size/8/1024:.1f} KB
Compressed: {compressed_size/8/1024:.1f} KB
Saved: {100*(1-compressed_size/orig_size):.1f}%"""
        
        ax.text(0.5, 0.5, metrics_text, transform=ax.transAxes,
                fontsize=12, verticalalignment='center',
                horizontalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        ax.axis('off')
        ax.set_title("Compression Performance", fontsize=12, fontweight='bold')
    
    def _plot_codebook_usage(self, indices: torch.Tensor, ax):
        """Plot histogram of codebook usage across stages"""
        n_q, N = indices.shape
        
        colors = plt.cm.Set2(np.linspace(0, 1, n_q))
        
        for i in range(n_q):
            idx = indices[i].cpu().numpy()
            unique, counts = np.unique(idx, return_counts=True)
            
            # Plot histogram
            ax.bar(unique, counts, alpha=0.7, label=f'Stage {i+1}', 
                  color=colors[i], width=0.8)
        
        ax.set_xlabel("Codebook Index", fontsize=10)
        ax.set_ylabel("Usage Count", fontsize=10)
        ax.set_title("Codebook Utilization", fontsize=11, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    
    def _plot_reconstruction_quality(
        self, original: torch.Tensor, quantized_stages: List[torch.Tensor], ax
    ):
        """Plot reconstruction quality across stages"""
        stages = []
        mse_vals = []
        psnr_vals = []
        
        for i, quant in enumerate(quantized_stages):
            stages.append(f"Stage {i+1}")
            mse = F.mse_loss(quant, original).item()
            mse_vals.append(mse)
            psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else 100
            psnr_vals.append(psnr)
        
        # Create dual y-axis plot
        ax2 = ax.twinx()
        
        # Plot MSE
        line1 = ax.plot(stages, mse_vals, 'b-o', linewidth=2, 
                       markersize=8, label='MSE')
        ax.set_ylabel("MSE", color='b', fontsize=10)
        ax.tick_params(axis='y', labelcolor='b')
        
        # Plot PSNR
        line2 = ax2.plot(stages, psnr_vals, 'r-s', linewidth=2,
                        markersize=8, label='PSNR (dB)')
        ax2.set_ylabel("PSNR (dB)", color='r', fontsize=10)
        ax2.tick_params(axis='y', labelcolor='r')
        
        ax.set_xlabel("Quantization Stage", fontsize=10)
        ax.set_title("Reconstruction Quality Progression", fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='center right')
    
    def create_video_from_frames(
        self, frame_paths: List[Path], output_path: str, fps: int = 10
    ):
        """Create video from visualization frames"""
        if not frame_paths:
            return
        
        # Read first frame to get dimensions
        first_frame = cv2.imread(str(frame_paths[0]))
        height, width = first_frame.shape[:2]
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for frame_path in frame_paths:
            frame = cv2.imread(str(frame_path))
            out.write(frame)
        
        out.release()
        print(f"Video saved to {output_path}")
    
    def plot_training_metrics(
        self, metrics_history: Dict[str, List[float]], save_path: str
    ):
        """Plot training metrics over time"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        metric_names = ['vq_loss', 'ortho_loss', 'perplexity', 
                       'mAP@0.3', 'mAP@0.5', 'mAP@0.7']
        
        for i, metric in enumerate(metric_names):
            if metric in metrics_history:
                ax = axes[i]
                values = metrics_history[metric]
                ax.plot(values, linewidth=2)
                ax.set_title(metric, fontsize=12, fontweight='bold')
                ax.set_xlabel("Epoch")
                ax.set_ylabel(metric)
                ax.grid(True, alpha=0.3)
                
                # Add trend line
                if len(values) > 5:
                    z = np.polyfit(range(len(values)), values, 1)
                    p = np.poly1d(z)
                    ax.plot(range(len(values)), p(range(len(values))), 
                           "r--", alpha=0.5, label=f"Trend")
                    ax.legend()
        
        plt.suptitle("Training Metrics - RVQ Compression", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()