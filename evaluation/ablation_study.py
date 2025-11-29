"""
Ablation Study Framework for Hallucination Detection
Systematically removes components to measure their contribution.
"""

import json
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import pandas as pd

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)


@dataclass
class AblationConfig:
    """Configuration for an ablation study."""
    name: str
    description: str
    components: Dict[str, bool]  # Component name -> enabled/disabled
    metadata: Optional[Dict] = None


@dataclass
class AblationResult:
    """Result of an ablation study."""
    config: AblationConfig
    metrics: Dict[str, float]
    predictions: np.ndarray
    probabilities: np.ndarray
    metadata: Optional[Dict] = None


class AblationStudy:
    """
    Conducts ablation studies by systematically removing components.
    """
    
    def __init__(self, base_config: Dict[str, bool]):
        """
        Initialize ablation study.
        
        Args:
            base_config: Base configuration with all components enabled
        """
        self.base_config = base_config
        self.components = list(base_config.keys())
        self.results = []
    
    def generate_ablation_configs(self) -> List[AblationConfig]:
        """
        Generate all possible ablation configurations.
        
        Returns:
            List of ablation configurations
        """
        configs = []
        
        # Full model (baseline)
        configs.append(AblationConfig(
            name="full_model",
            description="All components enabled",
            components=self.base_config.copy()
        ))
        
        # Remove each component individually
        for component in self.components:
            config = self.base_config.copy()
            config[component] = False
            configs.append(AblationConfig(
                name=f"no_{component}",
                description=f"Without {component}",
                components=config
            ))
        
        # Remove pairs of components
        for i, comp1 in enumerate(self.components):
            for comp2 in self.components[i+1:]:
                config = self.base_config.copy()
                config[comp1] = False
                config[comp2] = False
                configs.append(AblationConfig(
                    name=f"no_{comp1}_no_{comp2}",
                    description=f"Without {comp1} and {comp2}",
                    components=config
                ))
        
        # Single component only
        for component in self.components:
            config = {comp: False for comp in self.components}
            config[component] = True
            configs.append(AblationConfig(
                name=f"only_{component}",
                description=f"Only {component}",
                components=config
            ))
        
        return configs
    
    def run_ablation(
        self,
        config: AblationConfig,
        test_data: Dict,
        model_predictor
    ) -> AblationResult:
        """
        Run ablation study for a configuration.
        
        Args:
            config: Ablation configuration
            test_data: Test dataset
            model_predictor: Function that takes config and data, returns predictions
        
        Returns:
            AblationResult
        """
        # Get predictions with this configuration
        predictions, probabilities = model_predictor(config.components, test_data)
        
        # Compute metrics
        y_true = test_data['labels']
        metrics = self._compute_metrics(y_true, predictions, probabilities)
        
        return AblationResult(
            config=config,
            metrics=metrics,
            predictions=predictions,
            probabilities=probabilities
        )
    
    def _compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray
    ) -> Dict[str, float]:
        """Compute standard metrics."""
        return {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred, zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, zero_division=0)),
            'f1': float(f1_score(y_true, y_pred, zero_division=0)),
            'roc_auc': float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.0
        }
    
    def run_full_ablation(
        self,
        test_data: Dict,
        model_predictor,
        configs: Optional[List[AblationConfig]] = None
    ) -> List[AblationResult]:
        """
        Run full ablation study.
        
        Args:
            test_data: Test dataset
            model_predictor: Prediction function
            configs: Optional custom configs (uses generated if None)
        
        Returns:
            List of ablation results
        """
        if configs is None:
            configs = self.generate_ablation_configs()
        
        results = []
        print(f"Running ablation study with {len(configs)} configurations...")
        
        for i, config in enumerate(configs):
            print(f"  [{i+1}/{len(configs)}] {config.name}...")
            result = self.run_ablation(config, test_data, model_predictor)
            results.append(result)
        
        self.results = results
        return results
    
    def analyze_contributions(self) -> pd.DataFrame:
        """
        Analyze component contributions.
        
        Returns:
            DataFrame with contribution analysis
        """
        if not self.results:
            raise ValueError("No results available. Run ablation study first.")
        
        # Find baseline (full model)
        baseline = next((r for r in self.results if r.config.name == "full_model"), None)
        if baseline is None:
            raise ValueError("Baseline (full_model) not found in results.")
        
        baseline_f1 = baseline.metrics['f1']
        
        # Compute contributions
        contributions = []
        for result in self.results:
            if result.config.name == "full_model":
                continue
            
            # Compute drop in performance
            f1_drop = baseline_f1 - result.metrics['f1']
            relative_drop = f1_drop / baseline_f1 if baseline_f1 > 0 else 0
            
            # Identify which components are disabled
            disabled = [comp for comp, enabled in result.config.components.items() if not enabled]
            enabled = [comp for comp, enabled in result.config.components.items() if enabled]
            
            contributions.append({
                'config_name': result.config.name,
                'disabled_components': ', '.join(disabled) if disabled else 'none',
                'enabled_components': ', '.join(enabled),
                'f1_score': result.metrics['f1'],
                'f1_drop': f1_drop,
                'relative_drop': relative_drop,
                'accuracy': result.metrics['accuracy'],
                'precision': result.metrics['precision'],
                'recall': result.metrics['recall']
            })
        
        df = pd.DataFrame(contributions)
        return df.sort_values('f1_drop', ascending=False)
    
    def save_results(self, output_path: str):
        """Save ablation results to file."""
        results_dict = {
            'ablation_study': {
                'base_config': self.base_config,
                'num_configs': len(self.results),
                'results': [
                    {
                        'config': asdict(result.config),
                        'metrics': result.metrics,
                        'metadata': result.metadata
                    }
                    for result in self.results
                ]
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"Ablation results saved to {output_path}")
    
    def generate_report(self, output_path: str):
        """Generate comprehensive ablation report."""
        contributions = self.analyze_contributions()
        
        report_lines = [
            "Ablation Study Report",
            "=" * 80,
            "",
            f"Base Configuration: {self.base_config}",
            f"Total Configurations Tested: {len(self.results)}",
            "",
            "Component Contributions (sorted by F1 drop):",
            "-" * 80
        ]
        
        report_lines.append(contributions.to_string(index=False))
        
        # Summary statistics
        report_lines.extend([
            "",
            "Summary Statistics:",
            "-" * 80,
            f"Average F1 Drop: {contributions['f1_drop'].mean():.4f}",
            f"Max F1 Drop: {contributions['f1_drop'].max():.4f}",
            f"Min F1 Drop: {contributions['f1_drop'].min():.4f}",
        ])
        
        report = "\n".join(report_lines)
        
        with open(output_path, 'w') as f:
            f.write(report)
        
        print(f"Ablation report saved to {output_path}")
        return report

