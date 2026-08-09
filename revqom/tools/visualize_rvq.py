#!/usr/bin/env python
"""
Script to visualize RVQ compression during training or inference.
Usage: python visualize_rvq.py --model_path /path/to/model.pth --config /path/to/config.yaml
"""

import argparse
import torch
import yaml
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from revqom.visualization.rvq_visualizer import RVQVisualizer
from revqom.visualization.rvq_hook import RVQVisualizationHook
from revqom.models.cobevt import Cobevt
from revqom.dataset import build_dataset
from torch.utils.data import DataLoader


def visualize_rvq_compression(args):
    """Main visualization function"""
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create model
    model = Cobevt(config['model']['args'])
    
    # Load checkpoint if provided
    if args.model_path:
        checkpoint = torch.load(args.model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from {args.model_path}")
    
    model = model.cuda()
    model.eval()
    
    # Create visualizer and hook
    visualizer = RVQVisualizer(save_dir=args.output_dir)
    hook = RVQVisualizationHook(visualizer)
    hook.register_hooks(model)
    
    # Create dataset
    if args.use_val:
        dataset = build_dataset(config, visualize=True, train=False)
    else:
        dataset = build_dataset(config, visualize=True, train=True)
    
    dataloader = DataLoader(
        dataset, 
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=dataset.collate_batch
    )
    
    # Process samples
    num_samples = min(args.num_samples, len(dataloader))
    frame_paths = []
    
    print(f"Processing {num_samples} samples...")
    
    for idx, batch in enumerate(dataloader):
        if idx >= num_samples:
            break
            
        # Move batch to GPU
        batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # Forward pass
        with torch.no_grad():
            output = model(batch)
        
        # Get ground truth and predictions
        gt_boxes = batch.get('gt_boxes', None)
        pred_boxes = output.get('pred_boxes', None)
        
        # Create visualization
        frame_id = f"frame_{idx:04d}"
        frame_path = hook.visualize_current_batch(
            gt_boxes=gt_boxes,
            pred_boxes=pred_boxes,
            frame_id=frame_id,
            voxel_size=config['model']['args']['voxel_size'][0]
        )
        
        if frame_path:
            frame_paths.append(frame_path)
            print(f"Saved visualization {idx+1}/{num_samples}: {frame_path}")
        
        # Print compression metrics
        metrics = hook.get_compression_metrics()
        print(f"  MSE: {metrics.get('mse', 0):.6f}")
        print(f"  PSNR: {metrics.get('psnr', 0):.2f} dB")
        print(f"  Avg Codebook Usage: {metrics.get('avg_codebook_usage', 0):.1f}")
        
        # Reset hook for next batch
        hook.reset()
    
    # Create video if requested
    if args.create_video and frame_paths:
        video_path = Path(args.output_dir) / "rvq_visualization.mp4"
        visualizer.create_video_from_frames(frame_paths, str(video_path), fps=args.fps)
        print(f"Created video: {video_path}")
    
    print(f"Visualizations saved to {args.output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Visualize RVQ compression')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='./rvq_visualizations',
                       help='Output directory for visualizations')
    parser.add_argument('--num_samples', type=int, default=10,
                       help='Number of samples to visualize')
    parser.add_argument('--use_val', action='store_true',
                       help='Use validation set instead of training')
    parser.add_argument('--create_video', action='store_true',
                       help='Create video from visualization frames')
    parser.add_argument('--fps', type=int, default=5,
                       help='FPS for output video')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    visualize_rvq_compression(args)


if __name__ == '__main__':
    main()