"""
Hybrid Fusion Module for Hallucination Detection
Combines transformer model predictions with rule-based factual scores
using weighted fusion and threshold-based classification.
"""

from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class FusionResult:
    """Result of hybrid fusion prediction."""
    transformer_prob: float  # Transformer model's hallucination probability
    factual_score: float     # Rule-based factual correctness score (0-1)
    fusion_prob: float       # Final fused hallucination probability
    is_hallucination: bool   # Binary classification result
    confidence: float         # Confidence in the prediction
    alpha: float             # Weight parameter used


def compute_fusion_probability(
    transformer_prob: float,
    factual_score: float,
    alpha: float = 0.7
) -> float:
    """
    Compute final hallucination probability using weighted fusion.
    
    Fusion Logic:
    - transformer_prob: Probability from transformer model (0-1, higher = more hallucination)
    - factual_score: Rule-based factual correctness (0-1, higher = more correct)
    - alpha: Weight for transformer model (0-1)
    - (1-alpha): Weight for factual score
    
    The fusion combines:
    1. Transformer prediction (weighted by alpha)
    2. Inverted factual score (1 - factual_score) as hallucination indicator
       (weighted by 1-alpha)
    
    Formula: fusion_prob = alpha * transformer_prob + (1-alpha) * (1 - factual_score)
    
    Args:
        transformer_prob: Transformer model's hallucination probability (0-1)
        factual_score: Rule-based factual correctness score (0-1)
        alpha: Weight parameter for transformer model (default 0.7)
              - alpha=1.0: Use only transformer model
              - alpha=0.0: Use only factual score
              - alpha=0.7: 70% transformer, 30% factual (default)
    
    Returns:
        Final fused hallucination probability (0-1)
    
    Raises:
        ValueError: If inputs are not in valid range [0, 1]
    """
    # Validate inputs
    if not (0 <= transformer_prob <= 1):
        raise ValueError(f"transformer_prob must be in [0, 1], got {transformer_prob}")
    if not (0 <= factual_score <= 1):
        raise ValueError(f"factual_score must be in [0, 1], got {factual_score}")
    if not (0 <= alpha <= 1):
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    
    # Convert factual score to hallucination probability
    # Higher factual_score = lower hallucination probability
    factual_hallucination_prob = 1.0 - factual_score
    
    # Weighted fusion: combine transformer and factual predictions
    # alpha controls the relative importance of transformer vs factual
    fusion_prob = (alpha * transformer_prob) + ((1 - alpha) * factual_hallucination_prob)
    
    # Ensure result is in valid range (should be, but for safety)
    fusion_prob = max(0.0, min(1.0, fusion_prob))
    
    return fusion_prob


def classify_hallucination(
    fusion_prob: float,
    threshold: float = 0.5
) -> Tuple[bool, float]:
    """
    Classify response as hallucinated or correct based on threshold.
    
    Classification Logic:
    - If fusion_prob >= threshold: classified as HALLUCINATION
    - If fusion_prob < threshold: classified as CORRECT
    - Confidence is calculated as distance from threshold
    
    Args:
        fusion_prob: Fused hallucination probability (0-1)
        threshold: Classification threshold (default 0.5)
    
    Returns:
        Tuple of (is_hallucination, confidence)
        - is_hallucination: True if hallucination, False if correct
        - confidence: Confidence in classification (0-1)
    
    Raises:
        ValueError: If threshold is not in valid range [0, 1]
    """
    if not (0 <= threshold <= 1):
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")
    
    # Binary classification
    is_hallucination = fusion_prob >= threshold
    
    # Calculate confidence as distance from threshold
    # Maximum confidence when probability is at extremes (0 or 1)
    # Minimum confidence when probability is at threshold
    if is_hallucination:
        # For hallucination: confidence increases as prob approaches 1.0
        confidence = (fusion_prob - threshold) / (1.0 - threshold) if threshold < 1.0 else 1.0
    else:
        # For correct: confidence increases as prob approaches 0.0
        confidence = (threshold - fusion_prob) / threshold if threshold > 0.0 else 1.0
    
    # Ensure confidence is in valid range
    confidence = max(0.0, min(1.0, confidence))
    
    return is_hallucination, confidence


def hybrid_predict(
    transformer_prob: float,
    factual_score: float,
    alpha: float = 0.7,
    threshold: float = 0.5
) -> FusionResult:
    """
    Complete hybrid prediction pipeline: fusion + classification.
    
    This is the main function that combines:
    1. Weighted fusion of transformer and factual scores
    2. Threshold-based binary classification
    3. Confidence calculation
    
    Args:
        transformer_prob: Transformer model's hallucination probability (0-1)
        factual_score: Rule-based factual correctness score (0-1)
        alpha: Weight parameter for transformer model (default 0.7)
        threshold: Classification threshold (default 0.5)
    
    Returns:
        FusionResult with all prediction details
    """
    # Step 1: Compute fused probability
    fusion_prob = compute_fusion_probability(
        transformer_prob=transformer_prob,
        factual_score=factual_score,
        alpha=alpha
    )
    
    # Step 2: Classify using threshold
    is_hallucination, confidence = classify_hallucination(
        fusion_prob=fusion_prob,
        threshold=threshold
    )
    
    return FusionResult(
        transformer_prob=transformer_prob,
        factual_score=factual_score,
        fusion_prob=fusion_prob,
        is_hallucination=is_hallucination,
        confidence=confidence,
        alpha=alpha
    )


def batch_predict(
    transformer_probs: List[float],
    factual_scores: List[float],
    alpha: float = 0.7,
    threshold: float = 0.5
) -> List[FusionResult]:
    """
    Perform batch predictions on multiple responses.
    
    Args:
        transformer_probs: List of transformer hallucination probabilities
        factual_scores: List of factual correctness scores
        alpha: Weight parameter for transformer model
        threshold: Classification threshold
    
    Returns:
        List of FusionResult objects
    
    Raises:
        ValueError: If lists have different lengths
    """
    if len(transformer_probs) != len(factual_scores):
        raise ValueError(
            f"Lists must have same length: "
            f"transformer_probs={len(transformer_probs)}, "
            f"factual_scores={len(factual_scores)}"
        )
    
    results = []
    for trans_prob, fact_score in zip(transformer_probs, factual_scores):
        result = hybrid_predict(
            transformer_prob=trans_prob,
            factual_score=fact_score,
            alpha=alpha,
            threshold=threshold
        )
        results.append(result)
    
    return results


def find_optimal_alpha(
    transformer_probs: List[float],
    factual_scores: List[float],
    true_labels: List[bool],
    threshold: float = 0.5,
    alpha_range: Optional[Tuple[float, float]] = None,
    num_points: int = 20
) -> Tuple[float, float]:
    """
    Find optimal alpha value that maximizes accuracy.
    
    Args:
        transformer_probs: List of transformer probabilities
        factual_scores: List of factual scores
        true_labels: List of true labels (True=hallucination, False=correct)
        threshold: Classification threshold
        alpha_range: Range to search (min, max), defaults to (0, 1)
        num_points: Number of alpha values to test
    
    Returns:
        Tuple of (optimal_alpha, best_accuracy)
    """
    if alpha_range is None:
        alpha_range = (0.0, 1.0)
    
    alphas = np.linspace(alpha_range[0], alpha_range[1], num_points)
    best_alpha = 0.5
    best_accuracy = 0.0
    
    for alpha in alphas:
        # Make predictions with this alpha
        results = batch_predict(
            transformer_probs=transformer_probs,
            factual_scores=factual_scores,
            alpha=alpha,
            threshold=threshold
        )
        
        # Calculate accuracy
        predictions = [r.is_hallucination for r in results]
        accuracy = sum(p == t for p, t in zip(predictions, true_labels)) / len(true_labels)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_alpha = alpha
    
    return best_alpha, best_accuracy


# Example usage and demonstrations
if __name__ == "__main__":
    print("=" * 70)
    print("Hybrid Fusion Module - Demonstration")
    print("=" * 70)
    
    # Sample responses with their transformer predictions and factual scores
    sample_responses = [
        {
            "text": "Barack Obama was the 44th President of the United States.",
            "transformer_prob": 0.15,  # Model thinks: low hallucination (15%)
            "factual_score": 0.95       # Entity verification: high correctness (95%)
        },
        {
            "text": "Dr. Quantum invented the time machine in 2025.",
            "transformer_prob": 0.85,   # Model thinks: high hallucination (85%)
            "factual_score": 0.20       # Entity verification: low correctness (20%)
        },
        {
            "text": "Albert Einstein developed the theory of relativity.",
            "transformer_prob": 0.25,   # Model thinks: low hallucination (25%)
            "factual_score": 0.90       # Entity verification: high correctness (90%)
        },
        {
            "text": "The moon is made of cheese according to NASA scientists.",
            "transformer_prob": 0.90,   # Model thinks: high hallucination (90%)
            "factual_score": 0.10       # Entity verification: low correctness (10%)
        },
        {
            "text": "Microsoft is a technology company founded in 1975.",
            "transformer_prob": 0.30,   # Model thinks: medium-low hallucination (30%)
            "factual_score": 0.85       # Entity verification: high correctness (85%)
        },
        {
            "text": "The fictional character Harry Potter discovered quantum physics.",
            "transformer_prob": 0.75,   # Model thinks: high hallucination (75%)
            "factual_score": 0.40       # Entity verification: medium correctness (40%)
        }
    ]
    
    # Test with different alpha values
    alpha_values = [0.5, 0.7, 0.9]
    threshold = 0.5
    
    print(f"\nClassification threshold: {threshold}")
    print(f"Testing with different alpha values (weight for transformer model)")
    print("\n" + "-" * 70)
    
    for alpha in alpha_values:
        print(f"\nAlpha = {alpha} (Transformer: {alpha*100:.0f}%, Factual: {(1-alpha)*100:.0f}%)")
        print("-" * 70)
        
        for i, sample in enumerate(sample_responses, 1):
            result = hybrid_predict(
                transformer_prob=sample["transformer_prob"],
                factual_score=sample["factual_score"],
                alpha=alpha,
                threshold=threshold
            )
            
            status = "HALLUCINATION" if result.is_hallucination else "CORRECT"
            confidence_level = "HIGH" if result.confidence > 0.7 else "MEDIUM" if result.confidence > 0.4 else "LOW"
            
            print(f"\nResponse {i}: {sample['text'][:50]}...")
            print(f"  Transformer prob: {result.transformer_prob:.3f}")
            print(f"  Factual score:    {result.factual_score:.3f}")
            print(f"  Fusion prob:      {result.fusion_prob:.3f}")
            print(f"  Classification:   {status} (confidence: {confidence_level})")
    
    # Demonstrate threshold sensitivity
    print("\n" + "=" * 70)
    print("Threshold Sensitivity Analysis")
    print("=" * 70)
    
    # Use a borderline case
    borderline_transformer = 0.45
    borderline_factual = 0.55
    alpha = 0.7
    
    print(f"\nBorderline case:")
    print(f"  Transformer prob: {borderline_transformer:.3f}")
    print(f"  Factual score:    {borderline_factual:.3f}")
    print(f"  Alpha:            {alpha:.3f}")
    
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    print(f"\nClassification at different thresholds:")
    print("-" * 70)
    
    for thresh in thresholds:
        result = hybrid_predict(
            transformer_prob=borderline_transformer,
            factual_score=borderline_factual,
            alpha=alpha,
            threshold=thresh
        )
        status = "HALLUCINATION" if result.is_hallucination else "CORRECT"
        print(f"  Threshold {thresh:.1f}: {status} (fusion_prob={result.fusion_prob:.3f}, confidence={result.confidence:.3f})")
    
    # Demonstrate batch prediction
    print("\n" + "=" * 70)
    print("Batch Prediction Example")
    print("=" * 70)
    
    transformer_probs = [s["transformer_prob"] for s in sample_responses]
    factual_scores = [s["factual_score"] for s in sample_responses]
    
    batch_results = batch_predict(
        transformer_probs=transformer_probs,
        factual_scores=factual_scores,
        alpha=0.7,
        threshold=0.5
    )
    
    print(f"\nBatch predictions for {len(batch_results)} responses:")
    print("-" * 70)
    for i, (sample, result) in enumerate(zip(sample_responses, batch_results), 1):
        status = "HALLUCINATION" if result.is_hallucination else "CORRECT"
        print(f"{i}. {status:12s} | Fusion: {result.fusion_prob:.3f} | Conf: {result.confidence:.3f}")
    
    print("\n" + "=" * 70)
    print("Demonstration completed!")
    print("=" * 70)

