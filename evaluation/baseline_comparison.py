"""
Baseline Model Comparison Framework
Compares multiple baseline models and methods.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass, asdict
import json
from pathlib import Path

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)


@dataclass
class BaselineModel:
    """Represents a baseline model."""
    name: str
    description: str
    predictor: Callable  # Function that takes data and returns (predictions, probabilities)
    metadata: Optional[Dict] = None


@dataclass
class BaselineResult:
    """Result of baseline evaluation."""
    model_name: str
    metrics: Dict[str, float]
    predictions: np.ndarray
    probabilities: np.ndarray
    confusion_matrix: np.ndarray
    classification_report: str
    metadata: Optional[Dict] = None


class BaselineComparator:
    """
    Compares multiple baseline models.
    """
    
    # Standard baseline models
    BASELINE_MODELS = {
        'random': {
            'description': 'Random baseline (50/50)',
            'predictor': lambda data: (
                np.random.randint(0, 2, len(data['labels'])),
                np.random.uniform(0, 1, len(data['labels']))
            )
        },
        'always_correct': {
            'description': 'Always predict correct (0)',
            'predictor': lambda data: (
                np.zeros(len(data['labels']), dtype=int),
                np.zeros(len(data['labels']))
            )
        },
        'always_hallucination': {
            'description': 'Always predict hallucination (1)',
            'predictor': lambda data: (
                np.ones(len(data['labels']), dtype=int),
                np.ones(len(data['labels']))
            )
        },
        'majority_class': {
            'description': 'Predict majority class',
            'predictor': lambda data: (
                np.full(len(data['labels']), int(np.bincount(data['labels']).argmax())),
                np.full(len(data['labels']), np.bincount(data['labels']).max() / len(data['labels']))
            )
        }
    }
    
    def __init__(self):
        """Initialize comparator."""
        self.baselines: List[BaselineModel] = []
        self.results: List[BaselineResult] = []
    
    def add_baseline(
        self,
        name: str,
        predictor: Callable,
        description: Optional[str] = None,
        metadata: Optional[Dict] = None
    ):
        """
        Add a baseline model.
        
        Args:
            name: Model name
            predictor: Prediction function (data -> (predictions, probabilities))
            description: Optional description
            metadata: Optional metadata
        """
        baseline = BaselineModel(
            name=name,
            description=description or f"Custom baseline: {name}",
            predictor=predictor,
            metadata=metadata
        )
        self.baselines.append(baseline)
    
    def add_standard_baselines(self):
        """Add standard baseline models."""
        for name, config in self.BASELINE_MODELS.items():
            self.add_baseline(
                name=name,
                predictor=config['predictor'],
                description=config['description']
            )
    
    def evaluate_baseline(
        self,
        baseline: BaselineModel,
        test_data: Dict
    ) -> BaselineResult:
        """
        Evaluate a single baseline.
        
        Args:
            baseline: Baseline model to evaluate
            test_data: Test dataset with 'labels' key
        
        Returns:
            BaselineResult
        """
        # Get predictions
        predictions, probabilities = baseline.predictor(test_data)
        y_true = test_data['labels']
        
        # Ensure same length
        min_len = min(len(y_true), len(predictions), len(probabilities))
        y_true = y_true[:min_len]
        predictions = predictions[:min_len]
        probabilities = probabilities[:min_len]
        
        # Compute metrics
        metrics = {
            'accuracy': float(accuracy_score(y_true, predictions)),
            'precision': float(precision_score(y_true, predictions, zero_division=0)),
            'recall': float(recall_score(y_true, predictions, zero_division=0)),
            'f1': float(f1_score(y_true, predictions, zero_division=0)),
            'roc_auc': float(roc_auc_score(y_true, probabilities)) if len(np.unique(y_true)) > 1 else 0.0
        }
        
        # Confusion matrix
        cm = confusion_matrix(y_true, predictions)
        
        # Classification report
        report = classification_report(
            y_true, predictions,
            target_names=['Correct', 'Hallucination'],
            output_dict=False
        )
        
        return BaselineResult(
            model_name=baseline.name,
            metrics=metrics,
            predictions=predictions,
            probabilities=probabilities,
            confusion_matrix=cm.tolist(),
            classification_report=report,
            metadata=baseline.metadata
        )
    
    def compare_all(self, test_data: Dict) -> pd.DataFrame:
        """
        Compare all baselines.
        
        Args:
            test_data: Test dataset
        
        Returns:
            DataFrame with comparison results
        """
        if not self.baselines:
            self.add_standard_baselines()
        
        results = []
        print(f"Evaluating {len(self.baselines)} baseline models...")
        
        for baseline in self.baselines:
            print(f"  Evaluating {baseline.name}...")
            result = self.evaluate_baseline(baseline, test_data)
            results.append(result)
            self.results.append(result)
        
        # Create comparison DataFrame
        comparison_data = []
        for result in results:
            comparison_data.append({
                'Model': result.model_name,
                'Accuracy': result.metrics['accuracy'],
                'Precision': result.metrics['precision'],
                'Recall': result.metrics['recall'],
                'F1-Score': result.metrics['f1'],
                'ROC-AUC': result.metrics['roc_auc']
            })
        
        df = pd.DataFrame(comparison_data)
        df = df.sort_values('F1-Score', ascending=False)
        
        return df
    
    def save_comparison(self, output_path: str, comparison_df: pd.DataFrame):
        """Save comparison results."""
        results_dict = {
            'baseline_comparison': {
                'num_models': len(self.results),
                'comparison_table': comparison_df.to_dict('records'),
                'detailed_results': [
                    {
                        'model_name': r.model_name,
                        'metrics': r.metrics,
                        'confusion_matrix': r.confusion_matrix,
                        'classification_report': r.classification_report,
                        'metadata': r.metadata
                    }
                    for r in self.results
                ]
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"Baseline comparison saved to {output_path}")
    
    def generate_report(self, comparison_df: pd.DataFrame, output_path: str):
        """Generate comparison report."""
        report_lines = [
            "Baseline Model Comparison Report",
            "=" * 80,
            "",
            f"Total Models Compared: {len(self.results)}",
            "",
            "Comparison Table (sorted by F1-Score):",
            "-" * 80
        ]
        
        report_lines.append(comparison_df.to_string(index=False))
        
        # Best model
        best_model = comparison_df.iloc[0]
        report_lines.extend([
            "",
            "Best Model:",
            "-" * 80,
            f"  Name: {best_model['Model']}",
            f"  F1-Score: {best_model['F1-Score']:.4f}",
            f"  Accuracy: {best_model['Accuracy']:.4f}",
            f"  Precision: {best_model['Precision']:.4f}",
            f"  Recall: {best_model['Recall']:.4f}",
            f"  ROC-AUC: {best_model['ROC-AUC']:.4f}",
        ])
        
        report = "\n".join(report_lines)
        
        with open(output_path, 'w') as f:
            f.write(report)
        
        print(f"Comparison report saved to {output_path}")
        return report

