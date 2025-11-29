"""
Uncertainty-Driven Hallucination Score Module

This module implements a novel uncertainty-driven scoring mechanism that uses
epistemic and aleatoric uncertainty to refine hallucination predictions.
The key insight: high uncertainty in predictions often correlates with
hallucinations, and uncertainty decomposition enables targeted improvements.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

try:
    from sklearn.metrics import roc_auc_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


@dataclass
class UncertaintyScore:
    """Result of uncertainty-driven scoring."""
    base_prediction: float  # Base hallucination probability
    epistemic_uncertainty: float  # Model uncertainty
    aleatoric_uncertainty: float  # Data uncertainty
    total_uncertainty: float  # Combined uncertainty
    uncertainty_driven_score: float  # Final uncertainty-adjusted score
    confidence: float  # Confidence in the prediction
    uncertainty_type: str  # Dominant uncertainty type


class MonteCarloDropout(nn.Module):
    """
    Monte Carlo Dropout wrapper for uncertainty estimation.
    Enables dropout during inference to estimate epistemic uncertainty.
    """
    
    def __init__(self, base_model, dropout_rate: float = 0.1):
        """
        Initialize MC Dropout wrapper.
        
        Args:
            base_model: Base neural network model
            dropout_rate: Dropout probability
        """
        super().__init__()
        self.base_model = base_model
        self.dropout_rate = dropout_rate
        
        # Enable dropout in all dropout layers
        self._enable_dropout()
    
    def _enable_dropout(self):
        """Enable dropout in all dropout layers."""
        for module in self.base_model.modules():
            if isinstance(module, nn.Dropout):
                module.train()  # Keep dropout active during eval
    
    def forward(self, x, num_samples: int = 10):
        """
        Forward pass with MC sampling.
        
        Args:
            x: Input tensor
            num_samples: Number of MC samples
        
        Returns:
            Mean prediction and uncertainty estimates
        """
        predictions = []
        
        for _ in range(num_samples):
            with torch.no_grad():
                pred = self.base_model(x)
                predictions.append(pred)
        
        predictions = torch.stack(predictions)
        mean_pred = torch.mean(predictions, dim=0)
        var_pred = torch.var(predictions, dim=0)
        
        return mean_pred, var_pred


class EnsembleUncertainty:
    """
    Ensemble-based uncertainty estimation.
    Uses multiple model variants to estimate epistemic uncertainty.
    """
    
    def __init__(self, models: List[nn.Module]):
        """
        Initialize ensemble.
        
        Args:
            models: List of trained models (different initializations)
        """
        self.models = models
    
    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        return_individual: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get predictions with uncertainty from ensemble.
        
        Args:
            x: Input tensor
            return_individual: Whether to return individual predictions
        
        Returns:
            Mean prediction, epistemic uncertainty, individual predictions (optional)
        """
        predictions = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model(x)
                predictions.append(pred)
        
        predictions = torch.stack(predictions)
        mean_pred = torch.mean(predictions, dim=0)
        
        # Epistemic uncertainty = variance across models
        epistemic_uncertainty = torch.var(predictions, dim=0)
        
        if return_individual:
            return mean_pred, epistemic_uncertainty, predictions
        return mean_pred, epistemic_uncertainty


class UncertaintyDrivenScorer:
    """
    Main uncertainty-driven hallucination scorer.
    Uses uncertainty decomposition to refine predictions.
    """
    
    def __init__(
        self,
        uncertainty_method: str = "mc_dropout",
        uncertainty_weight: float = 0.3,
        uncertainty_threshold: float = 0.5
    ):
        """
        Initialize uncertainty-driven scorer.
        
        Args:
            uncertainty_method: "mc_dropout", "ensemble", or "both"
            uncertainty_weight: Weight for uncertainty in final score
            uncertainty_threshold: Threshold for high uncertainty
        """
        self.uncertainty_method = uncertainty_method
        self.uncertainty_weight = uncertainty_weight
        self.uncertainty_threshold = uncertainty_threshold
        
        self.mc_model = None
        self.ensemble = None
    
    def set_mc_model(self, model: nn.Module, dropout_rate: float = 0.1):
        """Set Monte Carlo Dropout model."""
        self.mc_model = MonteCarloDropout(model, dropout_rate)
    
    def set_ensemble(self, models: List[nn.Module]):
        """Set ensemble models."""
        self.ensemble = EnsembleUncertainty(models)
    
    def compute_uncertainty(
        self,
        input_tensor: torch.Tensor,
        num_samples: int = 10
    ) -> Tuple[float, float, float]:
        """
        Compute epistemic and aleatoric uncertainty.
        
        Args:
            input_tensor: Input to model
            num_samples: Number of MC samples
        
        Returns:
            Tuple of (epistemic_uncertainty, aleatoric_uncertainty, total_uncertainty)
        """
        if self.uncertainty_method == "mc_dropout" and self.mc_model is not None:
            mean_pred, var_pred = self.mc_model(input_tensor, num_samples)
            
            # Epistemic uncertainty = variance across samples
            epistemic = torch.mean(var_pred).item()
            
            # Aleatoric uncertainty = expected variance (approximated)
            # In practice, this requires multiple forward passes with different inputs
            # For now, we use a heuristic based on prediction confidence
            aleatoric = self._estimate_aleatoric(mean_pred)
            
        elif self.uncertainty_method == "ensemble" and self.ensemble is not None:
            mean_pred, epistemic_tensor = self.ensemble.predict_with_uncertainty(input_tensor)
            epistemic = torch.mean(epistemic_tensor).item()
            aleatoric = self._estimate_aleatoric(mean_pred)
        
        else:
            # Fallback: use prediction variance as proxy
            epistemic = 0.1  # Default
            aleatoric = 0.1  # Default
        
        total_uncertainty = epistemic + aleatoric
        
        return epistemic, aleatoric, total_uncertainty
    
    def _estimate_aleatoric(self, mean_pred: torch.Tensor) -> float:
        """
        Estimate aleatoric uncertainty.
        
        Aleatoric uncertainty is data-dependent and can be estimated
        from prediction confidence or entropy.
        """
        # Convert to probabilities if logits
        if mean_pred.dim() > 1 and mean_pred.size(-1) > 1:
            probs = torch.softmax(mean_pred, dim=-1)
        else:
            probs = torch.sigmoid(mean_pred)
        
        # Entropy as proxy for aleatoric uncertainty
        # Higher entropy = higher uncertainty
        if probs.dim() > 1:
            # Multi-class: use entropy
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
            aleatoric = torch.mean(entropy).item()
        else:
            # Binary: use binary entropy
            p = torch.clamp(probs, 1e-10, 1 - 1e-10)
            entropy = -(p * torch.log(p) + (1 - p) * torch.log(1 - p))
            aleatoric = torch.mean(entropy).item()
        
        # Normalize to [0, 1]
        return min(1.0, aleatoric)
    
    def score(
        self,
        base_prediction: float,
        epistemic_uncertainty: float,
        aleatoric_uncertainty: float,
        use_uncertainty_boost: bool = True
    ) -> UncertaintyScore:
        """
        Compute uncertainty-driven hallucination score.
        
        Key insight: High uncertainty often indicates hallucinations.
        We adjust the base prediction based on uncertainty levels.
        
        Args:
            base_prediction: Base hallucination probability (0-1)
            epistemic_uncertainty: Model uncertainty (0-1)
            aleatoric_uncertainty: Data uncertainty (0-1)
            use_uncertainty_boost: Whether to boost score for high uncertainty
        
        Returns:
            UncertaintyScore object
        """
        total_uncertainty = epistemic_uncertainty + aleatoric_uncertainty
        
        # Uncertainty-driven adjustment
        if use_uncertainty_boost:
            # High uncertainty increases hallucination probability
            # Formula: score = base + uncertainty_penalty
            uncertainty_penalty = total_uncertainty * self.uncertainty_weight
            
            # But only if uncertainty is above threshold
            if total_uncertainty > self.uncertainty_threshold:
                uncertainty_penalty *= 1.5  # Boost for high uncertainty
            
            uncertainty_driven_score = base_prediction + uncertainty_penalty
        else:
            # Alternative: uncertainty reduces confidence but doesn't change prediction
            uncertainty_driven_score = base_prediction
        
        # Clamp to valid range
        uncertainty_driven_score = max(0.0, min(1.0, uncertainty_driven_score))
        
        # Compute confidence (inverse of uncertainty)
        confidence = 1.0 - min(1.0, total_uncertainty)
        
        # Determine dominant uncertainty type
        if epistemic_uncertainty > aleatoric_uncertainty:
            uncertainty_type = "epistemic"
        elif aleatoric_uncertainty > epistemic_uncertainty:
            uncertainty_type = "aleatoric"
        else:
            uncertainty_type = "balanced"
        
        return UncertaintyScore(
            base_prediction=base_prediction,
            epistemic_uncertainty=epistemic_uncertainty,
            aleatoric_uncertainty=aleatoric_uncertainty,
            total_uncertainty=total_uncertainty,
            uncertainty_driven_score=uncertainty_driven_score,
            confidence=confidence,
            uncertainty_type=uncertainty_type
        )
    
    def score_batch(
        self,
        base_predictions: np.ndarray,
        epistemic_uncertainties: np.ndarray,
        aleatoric_uncertainties: np.ndarray
    ) -> List[UncertaintyScore]:
        """
        Score a batch of predictions.
        
        Args:
            base_predictions: Array of base predictions
            epistemic_uncertainties: Array of epistemic uncertainties
            aleatoric_uncertainties: Array of aleatoric uncertainties
        
        Returns:
            List of UncertaintyScore objects
        """
        scores = []
        for base_pred, epi_unc, ale_unc in zip(
            base_predictions, epistemic_uncertainties, aleatoric_uncertainties
        ):
            score = self.score(base_pred, epi_unc, ale_unc)
            scores.append(score)
        return scores


def integrate_with_hybrid_fusion(
    transformer_prob: float,
    factual_score: float,
    agentic_score: Optional[float],
    uncertainty_score: UncertaintyScore,
    alpha: float = 0.5,
    beta: float = 0.2,
    gamma: float = 0.2,
    delta: float = 0.1
) -> float:
    """
    Integrate uncertainty-driven score with hybrid fusion.
    
    Four-way fusion:
    - transformer_prob: Transformer model prediction
    - factual_score: Entity verification score
    - agentic_score: Agentic verification score (optional)
    - uncertainty_score: Uncertainty-driven score
    
    Args:
        transformer_prob: Transformer hallucination probability
        factual_score: Entity verification score (0-1, higher = more correct)
        agentic_score: Agentic verification score (0-1, higher = more correct)
        uncertainty_score: UncertaintyScore object
        alpha: Weight for transformer
        beta: Weight for factual
        gamma: Weight for agentic
        delta: Weight for uncertainty
    
    Returns:
        Final fused hallucination probability
    """
    # Normalize weights
    total_weight = alpha + beta + gamma + delta
    alpha = alpha / total_weight
    beta = beta / total_weight
    gamma = gamma / total_weight
    delta = delta / total_weight
    
    # Convert scores to hallucination probabilities
    factual_hallucination = 1.0 - factual_score
    agentic_hallucination = 1.0 - agentic_score if agentic_score is not None else 0.5
    uncertainty_hallucination = uncertainty_score.uncertainty_driven_score
    
    # Four-way weighted fusion
    fusion_prob = (
        alpha * transformer_prob +
        beta * factual_hallucination +
        gamma * agentic_hallucination +
        delta * uncertainty_hallucination
    )
    
    return max(0.0, min(1.0, fusion_prob))


# Example usage and tests
if __name__ == "__main__":
    print("=" * 70)
    print("Uncertainty-Driven Hallucination Score Module - Test")
    print("=" * 70)
    
    # Initialize scorer
    scorer = UncertaintyDrivenScorer(
        uncertainty_method="mc_dropout",
        uncertainty_weight=0.3,
        uncertainty_threshold=0.5
    )
    
    # Test scoring
    print("\n1. Testing Uncertainty-Driven Scoring")
    print("-" * 70)
    
    test_cases = [
        {
            "name": "High uncertainty, low base prediction",
            "base": 0.3,
            "epistemic": 0.7,
            "aleatoric": 0.2
        },
        {
            "name": "Low uncertainty, high base prediction",
            "base": 0.8,
            "epistemic": 0.1,
            "aleatoric": 0.1
        },
        {
            "name": "High epistemic uncertainty",
            "base": 0.5,
            "epistemic": 0.8,
            "aleatoric": 0.1
        },
        {
            "name": "High aleatoric uncertainty",
            "base": 0.5,
            "epistemic": 0.1,
            "aleatoric": 0.8
        }
    ]
    
    for case in test_cases:
        result = scorer.score(
            case["base"],
            case["epistemic"],
            case["aleatoric"]
        )
        
        print(f"\n{case['name']}:")
        print(f"  Base prediction: {result.base_prediction:.3f}")
        print(f"  Epistemic uncertainty: {result.epistemic_uncertainty:.3f}")
        print(f"  Aleatoric uncertainty: {result.aleatoric_uncertainty:.3f}")
        print(f"  Total uncertainty: {result.total_uncertainty:.3f}")
        print(f"  Uncertainty-driven score: {result.uncertainty_driven_score:.3f}")
        print(f"  Confidence: {result.confidence:.3f}")
        print(f"  Uncertainty type: {result.uncertainty_type}")
    
    # Test integration with hybrid fusion
    print("\n2. Testing Integration with Hybrid Fusion")
    print("-" * 70)
    
    uncertainty_result = scorer.score(0.4, 0.6, 0.3)
    fusion_prob = integrate_with_hybrid_fusion(
        transformer_prob=0.3,
        factual_score=0.9,
        agentic_score=0.85,
        uncertainty_score=uncertainty_result,
        alpha=0.5,
        beta=0.2,
        gamma=0.2,
        delta=0.1
    )
    
    print(f"Transformer prob: 0.300")
    print(f"Factual score: 0.900")
    print(f"Agentic score: 0.850")
    print(f"Uncertainty score: {uncertainty_result.uncertainty_driven_score:.3f}")
    print(f"Final fusion prob: {fusion_prob:.3f}")
    
    print("\n" + "=" * 70)
    print("Tests completed!")
    print("=" * 70)

