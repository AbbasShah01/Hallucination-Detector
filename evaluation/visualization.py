"""
Advanced Visualization for Evaluation Results
Generates comprehensive plots for research papers.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import pandas as pd

from sklearn.metrics import (
    roc_curve, precision_recall_curve, confusion_matrix,
    roc_auc_score, average_precision_score
)


class EvaluationVisualizer:
    """
    Creates comprehensive visualizations for evaluation results.
    """
    
    def __init__(self, style: str = "seaborn-v0_8-darkgrid", figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize visualizer.
        
        Args:
            style: Matplotlib style
            figsize: Default figure size
        """
        plt.style.use(style)
        self.figsize = figsize
        sns.set_palette("husl")
    
    def plot_roc_curves(
        self,
        y_true: np.ndarray,
        y_probs_dict: Dict[str, np.ndarray],
        output_path: str,
        title: str = "ROC Curves Comparison"
    ):
        """
        Plot multiple ROC curves for comparison.
        
        Args:
            y_true: True labels
            y_probs_dict: Dictionary of model_name -> probabilities
            output_path: Path to save plot
            title: Plot title
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot diagonal (random)
        ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier (AUC = 0.50)', linewidth=2)
        
        # Plot each model
        for model_name, y_prob in y_probs_dict.items():
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            auc_score = roc_auc_score(y_true, y_prob)
            ax.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.3f})', linewidth=2)
        
        ax.set_xlabel('False Positive Rate', fontsize=14, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=14, fontweight='bold')
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.legend(loc='lower right', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"ROC curves saved to {output_path}")
    
    def plot_precision_recall_curves(
        self,
        y_true: np.ndarray,
        y_probs_dict: Dict[str, np.ndarray],
        output_path: str,
        title: str = "Precision-Recall Curves Comparison"
    ):
        """Plot precision-recall curves."""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        for model_name, y_prob in y_probs_dict.items():
            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            ap_score = average_precision_score(y_true, y_prob)
            ax.plot(recall, precision, label=f'{model_name} (AP = {ap_score:.3f})', linewidth=2)
        
        ax.set_xlabel('Recall', fontsize=14, fontweight='bold')
        ax.set_ylabel('Precision', fontsize=14, fontweight='bold')
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.legend(loc='lower left', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Precision-Recall curves saved to {output_path}")
    
    def plot_metrics_comparison(
        self,
        metrics_dict: Dict[str, Dict[str, float]],
        output_path: str,
        title: str = "Metrics Comparison"
    ):
        """
        Plot bar chart comparing metrics across models.
        
        Args:
            metrics_dict: Dictionary of model_name -> metrics dict
            output_path: Path to save plot
            title: Plot title
        """
        # Prepare data
        models = list(metrics_dict.keys())
        metric_names = ['accuracy', 'precision', 'recall', 'f1']
        
        # Create DataFrame
        data = []
        for model in models:
            for metric in metric_names:
                value = metrics_dict[model].get(metric, 0.0)
                data.append({'Model': model, 'Metric': metric.capitalize(), 'Value': value})
        
        df = pd.DataFrame(data)
        
        # Create plot
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Pivot for grouped bar chart
        df_pivot = df.pivot(index='Model', columns='Metric', values='Value')
        df_pivot.plot(kind='bar', ax=ax, width=0.8, edgecolor='black', linewidth=1.5)
        
        ax.set_xlabel('Model', fontsize=14, fontweight='bold')
        ax.set_ylabel('Score', fontsize=14, fontweight='bold')
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.legend(title='Metric', fontsize=11, title_fontsize=12)
        ax.set_ylim([0, 1.1])
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Metrics comparison saved to {output_path}")
    
    def plot_confusion_matrices(
        self,
        y_true: np.ndarray,
        predictions_dict: Dict[str, np.ndarray],
        output_path: str,
        title: str = "Confusion Matrices Comparison"
    ):
        """Plot confusion matrices for multiple models."""
        n_models = len(predictions_dict)
        n_cols = min(3, n_models)
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
        if n_models == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes if isinstance(axes, list) else [axes]
        else:
            axes = axes.flatten()
        
        for idx, (model_name, y_pred) in enumerate(predictions_dict.items()):
            ax = axes[idx] if n_models > 1 else axes[0]
            
            cm = confusion_matrix(y_true, y_pred)
            sns.heatmap(
                cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Correct', 'Hallucination'],
                yticklabels=['Correct', 'Hallucination'],
                ax=ax, cbar_kws={'label': 'Count'}
            )
            
            ax.set_title(f'{model_name}', fontsize=14, fontweight='bold')
            ax.set_xlabel('Predicted', fontsize=12)
            ax.set_ylabel('True', fontsize=12)
        
        # Hide unused subplots
        for idx in range(n_models, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Confusion matrices saved to {output_path}")
    
    def plot_ablation_results(
        self,
        ablation_results: pd.DataFrame,
        output_path: str,
        title: str = "Ablation Study Results"
    ):
        """Plot ablation study results."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # F1 Score by configuration
        ax = axes[0, 0]
        ablation_results_sorted = ablation_results.sort_values('f1_score', ascending=True)
        ax.barh(range(len(ablation_results_sorted)), ablation_results_sorted['f1_score'])
        ax.set_yticks(range(len(ablation_results_sorted)))
        ax.set_yticklabels(ablation_results_sorted['config_name'], fontsize=9)
        ax.set_xlabel('F1 Score', fontsize=12, fontweight='bold')
        ax.set_title('F1 Score by Configuration', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # F1 Drop
        ax = axes[0, 1]
        ablation_results_sorted = ablation_results.sort_values('f1_drop', ascending=True)
        ax.barh(range(len(ablation_results_sorted)), ablation_results_sorted['f1_drop'])
        ax.set_yticks(range(len(ablation_results_sorted)))
        ax.set_yticklabels(ablation_results_sorted['config_name'], fontsize=9)
        ax.set_xlabel('F1 Drop', fontsize=12, fontweight='bold')
        ax.set_title('Performance Drop from Baseline', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Component contribution
        ax = axes[1, 0]
        # Aggregate by disabled components
        component_drops = {}
        for _, row in ablation_results.iterrows():
            disabled = row['disabled_components']
            if disabled and disabled != 'none':
                for comp in disabled.split(', '):
                    if comp not in component_drops:
                        component_drops[comp] = []
                    component_drops[comp].append(row['f1_drop'])
        
        if component_drops:
            avg_drops = {comp: np.mean(drops) for comp, drops in component_drops.items()}
            sorted_comps = sorted(avg_drops.items(), key=lambda x: x[1], reverse=True)
            comps, drops = zip(*sorted_comps[:10])  # Top 10
            ax.barh(range(len(comps)), drops)
            ax.set_yticks(range(len(comps)))
            ax.set_yticklabels(comps, fontsize=10)
            ax.set_xlabel('Average F1 Drop', fontsize=12, fontweight='bold')
            ax.set_title('Component Contribution', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x')
        
        # Metrics comparison
        ax = axes[1, 1]
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score']
        x = np.arange(len(ablation_results))
        width = 0.2
        
        for i, metric in enumerate(metrics_to_plot):
            ax.bar(x + i*width, ablation_results[metric], width, label=metric.replace('_', ' ').title())
        
        ax.set_xlabel('Configuration', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Metrics Across Configurations', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(ablation_results['config_name'], rotation=45, ha='right', fontsize=8)
        ax.legend(fontsize=9)
        ax.set_ylim([0, 1.1])
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Ablation results plot saved to {output_path}")
    
    def plot_advanced_metrics(
        self,
        advanced_metrics: Dict[str, Dict],
        output_path: str,
        title: str = "Advanced Metrics Comparison"
    ):
        """Plot advanced metrics comparison."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        models = list(advanced_metrics.keys())
        
        # Truthfulness Confidence
        ax = axes[0]
        values = [advanced_metrics[m].get('truthfulness_confidence', {}).get('value', 0) for m in models]
        ax.bar(models, values, color='skyblue', edgecolor='black', linewidth=1.5)
        ax.set_ylabel('Truthfulness Confidence', fontsize=12, fontweight='bold')
        ax.set_title('Truthfulness Confidence', fontsize=14, fontweight='bold')
        ax.set_ylim([0, 1.1])
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_xticklabels(models, rotation=45, ha='right')
        
        # Semantic Divergence
        ax = axes[1]
        values = [advanced_metrics[m].get('semantic_divergence', {}).get('value', 0) for m in models]
        ax.bar(models, values, color='lightcoral', edgecolor='black', linewidth=1.5)
        ax.set_ylabel('Semantic Divergence', fontsize=12, fontweight='bold')
        ax.set_title('Semantic Fact Divergence', fontsize=14, fontweight='bold')
        ax.set_ylim([0, 1.1])
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_xticklabels(models, rotation=45, ha='right')
        
        # Causal Chains
        ax = axes[2]
        values = [advanced_metrics[m].get('causal_chains', {}).get('value', 0) for m in models]
        ax.bar(models, values, color='lightgreen', edgecolor='black', linewidth=1.5)
        ax.set_ylabel('Avg Chain Length', fontsize=12, fontweight='bold')
        ax.set_title('Causal Hallucination Chains', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_xticklabels(models, rotation=45, ha='right')
        
        plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Advanced metrics plot saved to {output_path}")

