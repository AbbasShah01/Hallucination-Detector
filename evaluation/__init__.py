"""
Research-Grade Evaluation Framework for Hallucination Detection

This package provides comprehensive evaluation tools comparable to academic papers:
- Advanced metrics (truthfulness confidence, semantic divergence, causal chains)
- Ablation studies
- Baseline comparisons
- Comprehensive visualizations
- Automated evaluation pipeline
"""

from .metrics import (
    TruthfulnessConfidenceMetric,
    SemanticFactDivergenceMetric,
    CausalHallucinationChainMetric,
    AdvancedMetrics
)
from .ablation_study import AblationStudy
from .baseline_comparison import BaselineComparator
from .visualization import EvaluationVisualizer
from .evaluation_pipeline import EvaluationPipeline

__version__ = "1.0.0"
__all__ = [
    'TruthfulnessConfidenceMetric',
    'SemanticFactDivergenceMetric',
    'CausalHallucinationChainMetric',
    'AdvancedMetrics',
    'AblationStudy',
    'BaselineComparator',
    'EvaluationVisualizer',
    'EvaluationPipeline'
]

