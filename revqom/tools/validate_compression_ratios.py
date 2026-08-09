#!/usr/bin/env python3
"""
Validation script for testing EIGEN-MAP v2 at different compression ratios.
Evaluates performance vs compression trade-offs.
"""

import argparse
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
import yaml
import json
from collections import defaultdict

# Import EigenMAP modules
import revqom.utils.common_utils as utils
from revqom.tools import train_utils
from revqom.data_utils.datasets import build_dataset


def validate_compression_ratios(hypes_yaml_path, model_path, compression_ratios=[1, 2, 4, 8, 16, 32]):
    """
    Validate model performance at different compression ratios.
    
    Args:
        hypes_yaml_path: Path to YAML configuration
        model_path: Path to trained model checkpoint
        compression_ratios: List of compression ratios to test
    
    Returns:
        results: Dictionary with performance metrics for each ratio
    """
    print("=" * 60)
    print("EIGEN-MAP v2: Compression Ratio Validation")
    print("=" * 60)
    
    # Load configuration
    hypes = utils.load_yaml(hypes_yaml_path, None)
    
    # Build validation dataset
    print("Building validation dataset...")
    validate_dataset = build_dataset(hypes, visualize=False, train=False)
    validate_loader = DataLoader(
        validate_dataset,
        batch_size=1,  # Use batch_size=1 for validation
        shuffle=False,
        num_workers=4,
        collate_fn=validate_dataset.collate_batch,
        pin_memory=True
    )
    
    # Load model
    print(f"Loading model from {model_path}...")
    model = train_utils.create_model(hypes)
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['net'], strict=False)
    model = model.cuda()
    model.eval()
    
    # Results storage
    results = {}
    
    # Test each compression ratio
    for ratio in compression_ratios:
        print(f"\n--- Testing Compression Ratio: {ratio}x ---")
        
        # Initialize metrics
        metrics = {
            'total_loss': [],
            'detection_loss': [],
            'compression_loss': [],
            'reconstruction_mse': [],
            'ious': [],
            'per_class_ious': defaultdict(list)
        }
        
        with torch.no_grad():
            for i, batch_data in enumerate(validate_loader):
                if i >= 100:  # Limit validation samples for speed
                    break
                
                # Move to CUDA
                batch_data = train_utils.to_device(batch_data, 'cuda')
                
                # Set compression ratio for inference
                if ratio > 1:
                    batch_data['ego']['compress_ratio'] = ratio
                
                # Forward pass
                try:
                    output_dict = model(batch_data['ego'])
                except Exception as e:
                    print(f"Error in forward pass: {e}")
                    continue
                
                # Compute losses (without backward pass)
                criterion = train_utils.create_loss(hypes)
                criterion = criterion.cuda()
                
                try:
                    loss, loss_dict = criterion(output_dict, batch_data['ego']['gt_boxes'])
                    
                    # Store metrics
                    metrics['total_loss'].append(loss.item())
                    
                    if hasattr(criterion, 'loss_dict'):
                        if 'detection_loss' in criterion.loss_dict:
                            metrics['detection_loss'].append(criterion.loss_dict['detection_loss'])
                        if 'compression_loss' in criterion.loss_dict:
                            metrics['compression_loss'].append(criterion.loss_dict['compression_loss'])
                    
                    # IoU metrics
                    if hasattr(criterion, 'iou'):
                        metrics['ious'].append(criterion.iou)
                    
                    if hasattr(criterion, 'per_class_iou'):
                        for class_idx, iou_val in criterion.per_class_iou.items():
                            metrics['per_class_ious'][class_idx].append(iou_val)
                
                except Exception as e:
                    print(f"Error in loss computation: {e}")
                    continue
                
                # Print progress
                if (i + 1) % 20 == 0:
                    avg_loss = np.mean(metrics['total_loss'][-20:])
                    avg_iou = np.mean(metrics['ious'][-20:]) if metrics['ious'] else 0
                    print(f"  Batch {i+1}/100: Loss={avg_loss:.4f}, IoU={avg_iou:.4f}")
        
        # Compute average metrics for this compression ratio
        ratio_results = {
            'compression_ratio': ratio,
            'avg_total_loss': np.mean(metrics['total_loss']) if metrics['total_loss'] else float('inf'),
            'avg_detection_loss': np.mean(metrics['detection_loss']) if metrics['detection_loss'] else float('inf'),
            'avg_compression_loss': np.mean(metrics['compression_loss']) if metrics['compression_loss'] else 0.0,
            'avg_iou': np.mean(metrics['ious']) if metrics['ious'] else 0.0,
            'per_class_avg_iou': {
                class_idx: np.mean(iou_list) 
                for class_idx, iou_list in metrics['per_class_ious'].items()
            },
            'num_samples': len(metrics['total_loss'])
        }
        
        results[ratio] = ratio_results
        
        # Print results for this ratio
        print(f"Results for {ratio}x compression:")
        print(f"  Total Loss: {ratio_results['avg_total_loss']:.4f}")
        print(f"  Detection Loss: {ratio_results['avg_detection_loss']:.4f}")
        print(f"  Compression Loss: {ratio_results['avg_compression_loss']:.4f}")
        print(f"  Average IoU: {ratio_results['avg_iou']:.4f}")
        
        if ratio_results['per_class_avg_iou']:
            class_names = ['vehicle', 'pedestrian', 'truck']  # Adjust based on dataset
            iou_strs = []
            for i, class_name in enumerate(class_names):
                if i in ratio_results['per_class_avg_iou']:
                    iou_strs.append(f"{class_name}: {ratio_results['per_class_avg_iou'][i]:.3f}")
            print(f"  Per-class IoU: {' | '.join(iou_strs)}")
    
    return results


def analyze_compression_trade_offs(results):
    """
    Analyze and visualize compression vs performance trade-offs.
    """
    print("\n" + "=" * 60)
    print("COMPRESSION VS PERFORMANCE ANALYSIS")
    print("=" * 60)
    
    # Create summary table
    print(f"{'Ratio':<8} {'Loss':<10} {'IoU':<10} {'Degradation':<12} {'Bandwidth':<12}")
    print("-" * 60)
    
    baseline_iou = results[1]['avg_iou'] if 1 in results else None
    
    for ratio in sorted(results.keys()):
        result = results[ratio]
        
        degradation = 0.0
        if baseline_iou and baseline_iou > 0:
            degradation = ((baseline_iou - result['avg_iou']) / baseline_iou) * 100
        
        bandwidth_reduction = (1 - 1/ratio) * 100 if ratio > 1 else 0
        
        print(f"{ratio:<8} {result['avg_total_loss']:<10.4f} {result['avg_iou']:<10.4f} "
              f"{degradation:<12.1f}% {bandwidth_reduction:<12.1f}%")
    
    # Find optimal compression ratio (best IoU/compression trade-off)
    if baseline_iou:
        best_ratio = 1
        best_score = 0
        
        for ratio in sorted(results.keys()):
            if ratio == 1:
                continue
                
            result = results[ratio]
            iou_retention = result['avg_iou'] / baseline_iou if baseline_iou > 0 else 0
            bandwidth_saving = (1 - 1/ratio)
            
            # Trade-off score: balance IoU retention and bandwidth saving
            score = iou_retention * 0.7 + bandwidth_saving * 0.3
            
            if score > best_score:
                best_score = score
                best_ratio = ratio
        
        print(f"\nOptimal compression ratio: {best_ratio}x")
        print(f"  IoU: {results[best_ratio]['avg_iou']:.4f}")
        print(f"  IoU retention: {results[best_ratio]['avg_iou']/baseline_iou*100:.1f}%")
        print(f"  Bandwidth reduction: {(1-1/best_ratio)*100:.1f}%")


def save_results(results, output_path):
    """Save results to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Validate EIGEN-MAP v2 compression ratios')
    parser.add_argument('--hypes_yaml', type=str, required=True,
                       help='Path to YAML configuration file')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--compression_ratios', nargs='+', type=int, 
                       default=[1, 2, 4, 8, 16, 32],
                       help='Compression ratios to test')
    parser.add_argument('--output_path', type=str, 
                       default='compression_validation_results.json',
                       help='Path to save results JSON')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.hypes_yaml):
        raise FileNotFoundError(f"Configuration file not found: {args.hypes_yaml}")
    
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {args.model_path}")
    
    # Run validation
    print(f"Configuration: {args.hypes_yaml}")
    print(f"Model: {args.model_path}")
    print(f"Compression ratios: {args.compression_ratios}")
    
    results = validate_compression_ratios(
        args.hypes_yaml, 
        args.model_path, 
        args.compression_ratios
    )
    
    # Analyze results
    analyze_compression_trade_offs(results)
    
    # Save results
    save_results(results, args.output_path)
    
    print("\nValidation complete! 🎉")


if __name__ == '__main__':
    main()