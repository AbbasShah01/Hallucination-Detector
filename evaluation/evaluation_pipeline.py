"""
Automated Evaluation Pipeline
Orchestrates complete evaluation including metrics, baselines, ablation, and visualization.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Callable
from pathlib import Path
from datetime import datetime

from .metrics import AdvancedMetrics
from .ablation_study import AblationStudy
from .baseline_comparison import BaselineComparator
from .visualization import EvaluationVisualizer

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)


class EvaluationPipeline:
    """
    Automated evaluation pipeline for hallucination detection.
    """
    
    def __init__(self, output_dir: str = "results/evaluation"):
        """
        Initialize evaluation pipeline.
        
        Args:
            output_dir: Directory for output files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.advanced_metrics = AdvancedMetrics()
        self.visualizer = EvaluationVisualizer()
        self.baseline_comparator = BaselineComparator()
        self.ablation_study = None
        
        self.results = {}
    
    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
        responses: List[str],
        model_name: str = "model",
        confidence: Optional[np.ndarray] = None,
        conversation_contexts: Optional[List[List[str]]] = None
    ) -> Dict:
        """
        Run complete evaluation.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities
            responses: Response texts
            model_name: Name of the model
            confidence: Optional confidence scores
            conversation_contexts: Optional conversation contexts
        
        Returns:
            Dictionary with all evaluation results
        """
        print("=" * 80)
        print(f"Evaluating Model: {model_name}")
        print("=" * 80)
        
        results = {
            'model_name': model_name,
            'timestamp': datetime.now().isoformat(),
            'standard_metrics': {},
            'advanced_metrics': {},
            'baseline_comparison': None,
            'ablation_study': None
        }
        
        # Standard metrics
        print("\n1. Computing Standard Metrics...")
        results['standard_metrics'] = self._compute_standard_metrics(y_true, y_pred, y_prob)
        self._print_metrics(results['standard_metrics'])
        
        # Advanced metrics
        print("\n2. Computing Advanced Metrics...")
        advanced_results = self.advanced_metrics.compute_all(
            y_true, y_pred, y_prob, responses, confidence, conversation_contexts
        )
        results['advanced_metrics'] = {
            name: {
                'value': r.value,
                'confidence_interval': r.confidence_interval,
                'metadata': r.metadata
            }
            for name, r in advanced_results.items()
        }
        print(self.advanced_metrics.summary(advanced_results))
        
        # Save results
        self.results[model_name] = results
        self._save_results(model_name, results)
        
        return results
    
    def compare_baselines(
        self,
        test_data: Dict,
        custom_baselines: Optional[List[Callable]] = None
    ) -> pd.DataFrame:
        """
        Compare with baseline models.
        
        Args:
            test_data: Test dataset with 'labels' key
            custom_baselines: Optional custom baseline predictors
        
        Returns:
            Comparison DataFrame
        """
        print("\n3. Comparing with Baselines...")
        
        # Add custom baselines
        if custom_baselines:
            for i, predictor in enumerate(custom_baselines):
                self.baseline_comparator.add_baseline(
                    name=f"custom_baseline_{i+1}",
                    predictor=predictor
                )
        
        # Compare all
        comparison_df = self.baseline_comparator.compare_all(test_data)
        
        # Save
        comparison_path = self.output_dir / "baseline_comparison.json"
        self.baseline_comparator.save_comparison(str(comparison_path), comparison_df)
        
        # Generate report
        report_path = self.output_dir / "baseline_comparison_report.txt"
        self.baseline_comparator.generate_report(comparison_df, str(report_path))
        
        print("\nBaseline Comparison:")
        print(comparison_df.to_string(index=False))
        
        return comparison_df
    
    def run_ablation_study(
        self,
        base_config: Dict[str, bool],
        test_data: Dict,
        model_predictor: Callable
    ) -> pd.DataFrame:
        """
        Run ablation study.
        
        Args:
            base_config: Base configuration with component flags
            test_data: Test dataset
            model_predictor: Function that takes config and returns predictions
        
        Returns:
            Ablation results DataFrame
        """
        print("\n4. Running Ablation Study...")
        
        self.ablation_study = AblationStudy(base_config)
        ablation_results = self.ablation_study.run_full_ablation(test_data, model_predictor)
        
        # Analyze contributions
        contributions_df = self.ablation_study.analyze_contributions()
        
        # Save results
        results_path = self.output_dir / "ablation_results.json"
        self.ablation_study.save_results(str(results_path))
        
        # Generate report
        report_path = self.output_dir / "ablation_report.txt"
        self.ablation_study.generate_report(str(report_path))
        
        print("\nAblation Study Results:")
        print(contributions_df.to_string(index=False))
        
        return contributions_df
    
    def generate_all_visualizations(
        self,
        y_true: np.ndarray,
        models_data: Dict[str, Dict],
        ablation_results: Optional[pd.DataFrame] = None
    ):
        """
        Generate all visualizations.
        
        Args:
            y_true: True labels
            models_data: Dictionary of model_name -> {predictions, probabilities, metrics}
            ablation_results: Optional ablation study results
        """
        print("\n5. Generating Visualizations...")
        
        # Prepare data
        y_probs_dict = {name: data['probabilities'] for name, data in models_data.items()}
        y_preds_dict = {name: data['predictions'] for name, data in models_data.items()}
        metrics_dict = {name: data['metrics'] for name, data in models_data.items()}
        
        # ROC curves
        roc_path = self.output_dir / "roc_curves.png"
        self.visualizer.plot_roc_curves(y_true, y_probs_dict, str(roc_path))
        
        # Precision-Recall curves
        pr_path = self.output_dir / "precision_recall_curves.png"
        self.visualizer.plot_precision_recall_curves(y_true, y_probs_dict, str(pr_path))
        
        # Metrics comparison
        metrics_path = self.output_dir / "metrics_comparison.png"
        self.visualizer.plot_metrics_comparison(metrics_dict, str(metrics_path))
        
        # Confusion matrices
        cm_path = self.output_dir / "confusion_matrices.png"
        self.visualizer.plot_confusion_matrices(y_true, y_preds_dict, str(cm_path))
        
        # Ablation results (if available)
        if ablation_results is not None:
            ablation_path = self.output_dir / "ablation_results.png"
            self.visualizer.plot_ablation_results(ablation_results, str(ablation_path))
        
        print("All visualizations generated!")
    
    def _compute_standard_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray
    ) -> Dict[str, float]:
        """Compute standard classification metrics."""
        return {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred, zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, zero_division=0)),
            'f1': float(f1_score(y_true, y_pred, zero_division=0)),
            'roc_auc': float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.0
        }
    
    def _print_metrics(self, metrics: Dict[str, float]):
        """Print metrics in formatted way."""
        print("\nStandard Metrics:")
        print("-" * 40)
        for name, value in metrics.items():
            print(f"  {name.capitalize()}: {value:.4f}")
    
    def _save_results(self, model_name: str, results: Dict):
        """Save evaluation results to file."""
        output_path = self.output_dir / f"{model_name}_evaluation.json"
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")
    
    def generate_final_report(self, output_path: Optional[str] = None):
        """Generate comprehensive final evaluation report."""
        if output_path is None:
            output_path = self.output_dir / "final_evaluation_report.txt"
        
        report_lines = [
            "=" * 80,
            "COMPREHENSIVE EVALUATION REPORT",
            "=" * 80,
            f"Generated: {datetime.now().isoformat()}",
            "",
        ]
        
        # Summary of all models
        for model_name, results in self.results.items():
            report_lines.extend([
                f"\nModel: {model_name}",
                "-" * 80,
                "\nStandard Metrics:",
            ])
            
            for metric, value in results['standard_metrics'].items():
                report_lines.append(f"  {metric}: {value:.4f}")
            
            report_lines.append("\nAdvanced Metrics:")
            for metric_name, metric_data in results['advanced_metrics'].items():
                report_lines.append(f"  {metric_name}: {metric_data['value']:.4f}")
                if metric_data.get('confidence_interval'):
                    ci = metric_data['confidence_interval']
                    report_lines.append(f"    95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")
        
        report = "\n".join(report_lines)
        
        with open(output_path, 'w') as f:
            f.write(report)
        
        print(f"\nFinal report saved to {output_path}")
        return report

