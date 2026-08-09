import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from typing import Dict, List, Optional
from pathlib import Path


class CompressionComparisonPlotter:
    """
    Create compelling comparison plots for RVQ compression paper figures.
    """
    
    def __init__(self, save_dir: str = "./figures"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        
        # Set publication-quality style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_context("paper", font_scale=1.2)
        
    def plot_progressive_refinement(
        self,
        original: torch.Tensor,
        stages: List[torch.Tensor],
        residuals: List[torch.Tensor],
        save_name: str = "progressive_refinement.pdf"
    ):
        """
        Figure 1: Show progressive refinement through RVQ stages.
        Clean, publication-ready figure showing how each stage adds detail.
        """
        n_stages = len(stages)
        fig = plt.figure(figsize=(16, 4))
        
        # Original
        ax = plt.subplot(1, n_stages + 2, 1)
        feat = original[0].mean(0).cpu().numpy()
        im = ax.imshow(feat, cmap='viridis', aspect='auto')
        ax.set_title("Original", fontweight='bold')
        ax.axis('off')
        
        # Each stage
        for i, stage in enumerate(stages):
            ax = plt.subplot(1, n_stages + 2, i + 2)
            feat = stage[0].mean(0).cpu().numpy()
            ax.imshow(feat, cmap='viridis', aspect='auto')
            ax.set_title(f"Stage {i+1}", fontweight='bold')
            ax.axis('off')
            
            # Add arrow
            if i < n_stages - 1:
                ax.annotate('', xy=(1.05, 0.5), xytext=(0.95, 0.5),
                           xycoords='axes fraction',
                           arrowprops=dict(arrowstyle='->', lw=2))
        
        # Error heatmap
        ax = plt.subplot(1, n_stages + 2, n_stages + 2)
        error = torch.abs(original - stages[-1])[0].mean(0).cpu().numpy()
        im = ax.imshow(error, cmap='hot', aspect='auto')
        ax.set_title("Reconstruction Error", fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
        
        plt.tight_layout()
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def plot_compression_performance_tradeoff(
        self,
        compression_ratios: List[float],
        map_scores: Dict[str, List[float]],
        save_name: str = "compression_tradeoff.pdf"
    ):
        """
        Figure 2: Compression ratio vs detection performance.
        Shows mAP@0.3, 0.5, 0.7 vs compression ratio.
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        colors = {'0.3': '#2E86AB', '0.5': '#A23B72', '0.7': '#F18F01'}
        markers = {'0.3': 'o', '0.5': 's', '0.7': '^'}
        
        for threshold, scores in map_scores.items():
            ax.plot(compression_ratios, scores, 
                   color=colors[threshold],
                   marker=markers[threshold],
                   markersize=8,
                   linewidth=2,
                   label=f'mAP@{threshold}')
        
        ax.set_xlabel("Compression Ratio", fontsize=14)
        ax.set_ylabel("mAP Score", fontsize=14)
        ax.set_title("Detection Performance vs Compression Ratio", 
                    fontsize=16, fontweight='bold')
        ax.legend(loc='best', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)
        ax.set_ylim([0, 1])
        
        # Add annotations for key points
        for ratio, score in zip(compression_ratios, map_scores['0.5']):
            if ratio in [4, 8, 16]:  # Highlight key ratios
                ax.annotate(f'{ratio}x\n{score:.3f}',
                          xy=(ratio, score),
                          xytext=(5, 5),
                          textcoords='offset points',
                          fontsize=10,
                          bbox=dict(boxstyle='round,pad=0.3', 
                                  facecolor='yellow', alpha=0.5))
        
        plt.tight_layout()
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def plot_codebook_learning_progression(
        self,
        epochs: List[int],
        perplexity: List[float],
        usage: List[float],
        save_name: str = "codebook_learning.pdf"
    ):
        """
        Figure 3: Codebook learning over training epochs.
        Shows perplexity and usage statistics.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Perplexity
        ax1.plot(epochs, perplexity, 'b-', linewidth=2)
        ax1.fill_between(epochs, perplexity, alpha=0.3)
        ax1.set_xlabel("Epoch", fontsize=12)
        ax1.set_ylabel("Perplexity", fontsize=12)
        ax1.set_title("Codebook Perplexity", fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Usage
        ax2.plot(epochs, usage, 'g-', linewidth=2)
        ax2.fill_between(epochs, usage, alpha=0.3)
        ax2.set_xlabel("Epoch", fontsize=12)
        ax2.set_ylabel("Average Usage (%)", fontsize=12)
        ax2.set_title("Codebook Utilization", fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 100])
        
        plt.tight_layout()
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def plot_stage_wise_contribution(
        self,
        stage_names: List[str],
        contributions: Dict[str, List[float]],
        save_name: str = "stage_contributions.pdf"
    ):
        """
        Figure 4: What each RVQ stage captures.
        Bar chart showing what features each stage learns.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(stage_names))
        width = 0.25
        
        feature_types = ['Coarse Structure', 'Object Shapes', 'Fine Details']
        colors = ['#8B4513', '#4169E1', '#32CD32']
        
        for i, (feature, color) in enumerate(zip(feature_types, colors)):
            values = contributions[feature]
            offset = (i - 1) * width
            ax.bar(x + offset, values, width, label=feature, color=color)
        
        ax.set_xlabel("RVQ Stage", fontsize=12)
        ax.set_ylabel("Contribution (%)", fontsize=12)
        ax.set_title("Feature Contribution by RVQ Stage", 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(stage_names)
        ax.legend(loc='upper left', fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def plot_class_wise_impact(
        self,
        classes: List[str],
        baseline_ap: List[float],
        compressed_ap: List[float],
        save_name: str = "class_impact.pdf"
    ):
        """
        Figure 5: Impact of compression on different object classes.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(classes))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, baseline_ap, width, 
                      label='Baseline', color='#2E86AB')
        bars2 = ax.bar(x + width/2, compressed_ap, width,
                      label='Compressed (16x)', color='#F18F01')
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}',
                          xy=(bar.get_x() + bar.get_width() / 2, height),
                          xytext=(0, 3),
                          textcoords="offset points",
                          ha='center', va='bottom',
                          fontsize=10)
        
        ax.set_xlabel("Object Class", fontsize=12)
        ax.set_ylabel("AP@0.5", fontsize=12)
        ax.set_title("Per-Class Detection Performance: Baseline vs Compressed",
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(classes)
        ax.legend(loc='upper right', fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def create_paper_figure_set(
        self,
        data: Dict,
        save_prefix: str = "rvq"
    ):
        """
        Create complete set of figures for paper.
        """
        figures = {}
        
        # Progressive refinement
        if 'stages' in data:
            figures['refinement'] = self.plot_progressive_refinement(
                data['original'],
                data['stages'],
                data['residuals'],
                f"{save_prefix}_refinement.pdf"
            )
        
        # Compression tradeoff
        if 'compression_ratios' in data:
            figures['tradeoff'] = self.plot_compression_performance_tradeoff(
                data['compression_ratios'],
                data['map_scores'],
                f"{save_prefix}_tradeoff.pdf"
            )
        
        # Codebook learning
        if 'epochs' in data:
            figures['codebook'] = self.plot_codebook_learning_progression(
                data['epochs'],
                data['perplexity'],
                data['usage'],
                f"{save_prefix}_codebook.pdf"
            )
        
        # Stage contributions
        if 'stage_contributions' in data:
            figures['contributions'] = self.plot_stage_wise_contribution(
                data['stage_names'],
                data['stage_contributions'],
                f"{save_prefix}_contributions.pdf"
            )
        
        # Class impact
        if 'class_performance' in data:
            figures['class_impact'] = self.plot_class_wise_impact(
                data['classes'],
                data['baseline_ap'],
                data['compressed_ap'],
                f"{save_prefix}_class_impact.pdf"
            )
        
        print(f"Created {len(figures)} figures:")
        for name, path in figures.items():
            print(f"  - {name}: {path}")
        
        return figures