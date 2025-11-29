"""
Dynamic Multi-Signal Fusion (DMSF)

A novel fusion algorithm that dynamically adjusts weights based on:
- Inter-signal agreement/disagreement
- Uncertainty levels
- Signal reliability

Formula:
    H = α*C + β*E + γ*A + δ*SHDS + DynamicBias

Where:
- C = classifier score
- E = entity verification score
- A = agentic verification score
- SHDS = Semantic Hallucination Divergence Score
- DynamicBias = computed from signal disagreement and uncertainty

Dynamic Weight Adjustment:
- High disagreement → increase SHDS weight
- High agreement → trust agreement, reduce SHDS
- High uncertainty → upweight agent and SHDS
- Strong entity mismatch → upweight entity verification

This represents a novel research contribution that adapts fusion
weights based on signal characteristics rather than using fixed weights.
"""

import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

try:
    from modules.novel_metric.shds import SHDS, SHDSResult
    SHDS_AVAILABLE = True
except ImportError:
    SHDS_AVAILABLE = False
    print("Warning: SHDS not available. DMSF will use fallback.")


@dataclass
class DMSFResult:
    """Result of DMSF fusion."""
    final_score: float  # Final hallucination probability [0,1]
    base_fusion: float  # Base weighted fusion score
    dynamic_bias: float  # Dynamic adjustment
    adjusted_weights: Dict[str, float]  # Final adjusted weights
    signal_agreement: float  # Agreement score [0,1]
    uncertainty_level: float  # Overall uncertainty [0,1]
    fusion_method: str  # "dmsf"


class DMSF:
    """
    Dynamic Multi-Signal Fusion calculator.
    
    This novel fusion algorithm adapts weights based on signal characteristics,
    providing more robust hallucination detection than fixed-weight fusion.
    
    Research Contribution:
    - First fusion method to dynamically adjust weights based on signal agreement
    - Incorporates uncertainty-aware weighting
    - Adapts to different hallucination types automatically
    - Integrates novel SHDS metric for comprehensive assessment
    """
    
    def __init__(
        self,
        alpha: float = 0.4,  # Base weight for classifier
        beta: float = 0.25,  # Base weight for entity verification
        gamma: float = 0.2,  # Base weight for agentic verification
        delta: float = 0.15,  # Base weight for SHDS
        agreement_threshold: float = 0.7,  # Threshold for high agreement
        disagreement_threshold: float = 0.3,  # Threshold for high disagreement
        uncertainty_threshold: float = 0.5  # Threshold for high uncertainty
    ):
        """
        Initialize DMSF calculator.
        
        Args:
            alpha: Base weight for classifier score
            beta: Base weight for entity verification
            gamma: Base weight for agentic verification
            delta: Base weight for SHDS
            agreement_threshold: Threshold for considering signals in agreement
            disagreement_threshold: Threshold for considering signals in disagreement
            uncertainty_threshold: Threshold for high uncertainty
        """
        # Normalize base weights
        total = alpha + beta + gamma + delta
        self.alpha_base = alpha / total
        self.beta_base = beta / total
        self.gamma_base = gamma / total
        self.delta_base = delta / total
        
        self.agreement_threshold = agreement_threshold
        self.disagreement_threshold = disagreement_threshold
        self.uncertainty_threshold = uncertainty_threshold
        
        # Initialize SHDS if available
        self.shds_calculator = None
        if SHDS_AVAILABLE:
            self.shds_calculator = SHDS()
    
    def compute_signal_agreement(
        self,
        classifier_score: float,
        entity_score: float,
        agentic_score: Optional[float],
        shds_score: Optional[float] = None
    ) -> float:
        """
        Compute agreement between signals.
        
        Measures how much signals agree on hallucination probability.
        
        Args:
            classifier_score: Classifier hallucination probability
            entity_score: Entity verification score (higher = more correct, so invert)
            agentic_score: Agent verification score (higher = more correct, so invert)
            shds_score: Optional SHDS score
        
        Returns:
            Agreement score [0,1] (higher = more agreement)
        """
        # Convert verification scores to hallucination probabilities
        entity_hallucination = 1.0 - entity_score
        agentic_hallucination = 1.0 - agentic_score if agentic_score is not None else None
        
        # Collect all scores
        scores = [classifier_score, entity_hallucination]
        if agentic_hallucination is not None:
            scores.append(agentic_hallucination)
        if shds_score is not None:
            scores.append(shds_score)
        
        if len(scores) < 2:
            return 0.5  # Neutral if not enough signals
        
        # Compute variance as measure of disagreement
        scores_array = np.array(scores)
        variance = np.var(scores_array)
        
        # Convert variance to agreement (lower variance = higher agreement)
        # Normalize: variance in [0, 0.25] (max for binary scores)
        agreement = 1.0 - min(1.0, variance * 4.0)
        
        return max(0.0, min(1.0, agreement))
    
    def compute_uncertainty_level(
        self,
        classifier_score: float,
        entity_score: float,
        agentic_score: Optional[float],
        shds_score: Optional[float] = None
    ) -> float:
        """
        Compute overall uncertainty level.
        
        Args:
            classifier_score: Classifier score
            entity_score: Entity verification score
            agentic_score: Agent verification score
            shds_score: Optional SHDS score
        
        Returns:
            Uncertainty level [0,1] (higher = more uncertain)
        """
        # Uncertainty is high when scores are near 0.5 (ambiguous)
        scores = [classifier_score]
        if entity_score is not None:
            entity_hallucination = 1.0 - entity_score
            scores.append(entity_hallucination)
        if agentic_score is not None:
            agentic_hallucination = 1.0 - agentic_score
            scores.append(agentic_hallucination)
        if shds_score is not None:
            scores.append(shds_score)
        
        # Average distance from 0.5 (uncertainty center)
        distances = [abs(s - 0.5) for s in scores]
        avg_distance = np.mean(distances)
        
        # Uncertainty is inverse of distance from 0.5
        uncertainty = 1.0 - (avg_distance * 2.0)  # Scale to [0,1]
        
        return max(0.0, min(1.0, uncertainty))
    
    def compute_dynamic_bias(
        self,
        agreement: float,
        uncertainty: float,
        entity_mismatch: Optional[float] = None
    ) -> float:
        """
        Compute dynamic bias adjustment.
        
        Args:
            agreement: Signal agreement score
            uncertainty: Uncertainty level
            entity_mismatch: Optional entity mismatch strength
        
        Returns:
            Dynamic bias adjustment [-0.2, 0.2]
        """
        bias = 0.0
        
        # High disagreement → positive bias (trust SHDS more)
        if agreement < self.disagreement_threshold:
            bias += 0.1  # Increase hallucination probability
        
        # High agreement → negative bias (trust agreement)
        if agreement > self.agreement_threshold:
            bias -= 0.05  # Slight decrease
        
        # High uncertainty → positive bias (be more cautious)
        if uncertainty > self.uncertainty_threshold:
            bias += 0.1
        
        # Strong entity mismatch → positive bias
        if entity_mismatch is not None and entity_mismatch > 0.7:
            bias += 0.05
        
        # Clamp bias
        bias = max(-0.2, min(0.2, bias))
        
        return bias
    
    def adjust_weights(
        self,
        agreement: float,
        uncertainty: float,
        entity_mismatch: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Dynamically adjust fusion weights.
        
        Args:
            agreement: Signal agreement score
            uncertainty: Uncertainty level
            entity_mismatch: Optional entity mismatch strength
        
        Returns:
            Adjusted weights dict
        """
        alpha = self.alpha_base
        beta = self.beta_base
        gamma = self.gamma_base
        delta = self.delta_base
        
        # High disagreement → increase SHDS weight
        if agreement < self.disagreement_threshold:
            delta += 0.15
            alpha -= 0.05
            beta -= 0.05
            gamma -= 0.05
        
        # High agreement → reduce SHDS, trust signals
        elif agreement > self.agreement_threshold:
            delta -= 0.1
            alpha += 0.03
            beta += 0.03
            gamma += 0.04
        
        # High uncertainty → upweight agent and SHDS
        if uncertainty > self.uncertainty_threshold:
            gamma += 0.1
            delta += 0.05
            alpha -= 0.08
            beta -= 0.07
        
        # Strong entity mismatch → upweight entity verification
        if entity_mismatch is not None and entity_mismatch > 0.7:
            beta += 0.15
            alpha -= 0.05
            gamma -= 0.05
            delta -= 0.05
        
        # Ensure non-negative and normalize
        alpha = max(0.0, alpha)
        beta = max(0.0, beta)
        gamma = max(0.0, gamma)
        delta = max(0.0, delta)
        
        total = alpha + beta + gamma + delta
        if total > 0:
            alpha = alpha / total
            beta = beta / total
            gamma = gamma / total
            delta = delta / total
        
        return {
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "delta": delta
        }
    
    def fuse(
        self,
        classifier_score: float,
        entity_score: float,
        agentic_score: Optional[float] = None,
        shds_score: Optional[float] = None,
        entity_mismatch: Optional[float] = None,
        compute_shds: bool = True,
        span: Optional[str] = None,
        **shds_kwargs
    ) -> DMSFResult:
        """
        Perform dynamic multi-signal fusion.
        
        Args:
            classifier_score: Classifier hallucination probability [0,1]
            entity_score: Entity verification correctness [0,1] (higher = more correct)
            agentic_score: Agent verification correctness [0,1] (higher = more correct)
            shds_score: Optional pre-computed SHDS score
            entity_mismatch: Optional entity mismatch strength [0,1]
            compute_shds: Whether to compute SHDS if not provided
            span: Text span (for SHDS computation if needed)
            **shds_kwargs: Additional arguments for SHDS computation
        
        Returns:
            DMSFResult with final score and metadata
        """
        # Compute SHDS if needed
        if shds_score is None and compute_shds and self.shds_calculator is not None:
            if span is not None:
                shds_result = self.shds_calculator.compute(span, **shds_kwargs)
                shds_score = shds_result.shds_score
                # Extract entity mismatch from SHDS if available
                if entity_mismatch is None:
                    entity_mismatch = shds_result.entity_mismatch_penalty
            else:
                shds_score = None
        
        # Convert verification scores to hallucination probabilities
        entity_hallucination = 1.0 - entity_score
        agentic_hallucination = 1.0 - agentic_score if agentic_score is not None else None
        
        # Compute signal characteristics
        agreement = self.compute_signal_agreement(
            classifier_score, entity_score, agentic_score, shds_score
        )
        uncertainty = self.compute_uncertainty_level(
            classifier_score, entity_score, agentic_score, shds_score
        )
        
        # Adjust weights dynamically
        adjusted_weights = self.adjust_weights(agreement, uncertainty, entity_mismatch)
        
        # Base fusion with adjusted weights
        base_fusion = (
            adjusted_weights["alpha"] * classifier_score +
            adjusted_weights["beta"] * entity_hallucination
        )
        
        if agentic_hallucination is not None:
            base_fusion += adjusted_weights["gamma"] * agentic_hallucination
        
        if shds_score is not None:
            base_fusion += adjusted_weights["delta"] * shds_score
        
        # Compute dynamic bias
        dynamic_bias = self.compute_dynamic_bias(agreement, uncertainty, entity_mismatch)
        
        # Final score
        final_score = base_fusion + dynamic_bias
        
        # Clamp to [0,1]
        final_score = max(0.0, min(1.0, final_score))
        
        return DMSFResult(
            final_score=final_score,
            base_fusion=base_fusion,
            dynamic_bias=dynamic_bias,
            adjusted_weights=adjusted_weights,
            signal_agreement=agreement,
            uncertainty_level=uncertainty,
            fusion_method="dmsf"
        )


def compute_dmsf(
    classifier_score: float,
    entity_score: float,
    agentic_score: Optional[float] = None,
    **kwargs
) -> DMSFResult:
    """
    Convenience function to compute DMSF.
    
    Args:
        classifier_score: Classifier score
        entity_score: Entity verification score
        agentic_score: Agent verification score
        **kwargs: Additional arguments for DMSF.fuse()
    
    Returns:
        DMSFResult
    """
    dmsf = DMSF()
    return dmsf.fuse(classifier_score, entity_score, agentic_score, **kwargs)


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("Dynamic Multi-Signal Fusion (DMSF) Test")
    print("=" * 70)
    
    dmsf = DMSF()
    
    # Test cases
    test_cases = [
        {
            "name": "High agreement (all signals agree)",
            "classifier": 0.8,
            "entity": 0.2,  # Low correctness = high hallucination
            "agentic": 0.15,
            "span": "The moon is made of cheese."
        },
        {
            "name": "High disagreement (signals conflict)",
            "classifier": 0.3,
            "entity": 0.9,  # High correctness = low hallucination
            "agentic": 0.85,
            "span": "Barack Obama was the 44th President."
        },
        {
            "name": "High uncertainty (ambiguous scores)",
            "classifier": 0.5,
            "entity": 0.5,
            "agentic": 0.5,
            "span": "This is an ambiguous statement."
        }
    ]
    
    print("\nComputing DMSF fusion...")
    for case in test_cases:
        result = dmsf.fuse(
            classifier_score=case["classifier"],
            entity_score=case["entity"],
            agentic_score=case["agentic"],
            span=case["span"],
            compute_shds=True
        )
        
        print(f"\n{case['name']}:")
        print(f"  Final Score: {result.final_score:.4f}")
        print(f"  Base Fusion: {result.base_fusion:.4f}")
        print(f"  Dynamic Bias: {result.dynamic_bias:.4f}")
        print(f"  Signal Agreement: {result.signal_agreement:.4f}")
        print(f"  Uncertainty Level: {result.uncertainty_level:.4f}")
        print(f"  Adjusted Weights:")
        for key, value in result.adjusted_weights.items():
            print(f"    {key}: {value:.4f}")
    
    print("\n" + "=" * 70)

