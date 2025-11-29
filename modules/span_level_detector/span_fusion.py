"""
Span-Level Fusion Module

Combines classification, entity verification, and agent verification scores
at the sentence level using weighted fusion.
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import sys
import os

# Add parent directories to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

try:
    from src.hybrid_fusion import compute_fusion_probability
    FUSION_AVAILABLE = True
except ImportError:
    FUSION_AVAILABLE = False

# Import novel DMSF fusion
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))
    from modules.fusion.dmsf import DMSF
    from modules.novel_metric.shds import SHDS
    DMSF_AVAILABLE = True
except ImportError:
    DMSF_AVAILABLE = False
    DMSF = None
    SHDS = None


@dataclass
class SpanFusionResult:
    """Result of sentence-level fusion."""
    sentence: str
    sentence_index: int
    classification_score: float
    entity_verification_score: float
    agent_verification_score: Optional[float]
    final_hallucination_score: float
    label: str  # "hallucinated" or "factual"
    confidence: float
    fusion_weights: Dict[str, float]


class SpanFusion:
    """
    Fuses multiple detection signals at the sentence level.
    
    Combines:
    - Transformer classification score
    - Entity verification score
    - Agent verification score (optional)
    
    Uses weighted fusion similar to response-level fusion but adapted
    for sentence-level granularity.
    """
    
    def __init__(
        self,
        alpha: float = 0.5,  # Weight for classification
        beta: float = 0.3,   # Weight for entity verification
        gamma: float = 0.2,  # Weight for agent verification
        threshold: float = 0.5,
        fusion_method: str = "classic"  # "classic" or "dmsf"
    ):
        """
        Initialize span fusion.
        
        Args:
            alpha: Weight for classification score
            beta: Weight for entity verification
            gamma: Weight for agent verification
            threshold: Threshold for hallucination classification
        """
        # Normalize weights
        total = alpha + beta + gamma
        self.alpha = alpha / total
        self.beta = beta / total
        self.gamma = gamma / total
        self.threshold = threshold
        self.fusion_method = fusion_method
        
        # Initialize DMSF if requested
        self.dmsf = None
        self.shds = None
        if fusion_method == "dmsf" and DMSF_AVAILABLE:
            try:
                self.dmsf = DMSF(alpha=alpha, beta=beta, gamma=gamma, delta=0.15)
                self.shds = SHDS()
            except Exception as e:
                print(f"Warning: Could not initialize DMSF: {e}")
                self.fusion_method = "classic"
    
    def fuse(
        self,
        classification_score: float,
        entity_verification_score: float,
        agent_verification_score: Optional[float] = None,
        span: Optional[str] = None
    ) -> Dict:
        """
        Fuse scores for a single sentence.
        
        Args:
            classification_score: Transformer hallucination probability (0-1)
            entity_verification_score: Entity correctness score (0-1, higher = more correct)
            agent_verification_score: Agent correctness score (0-1, higher = more correct)
        
        Returns:
            Dict with fused score and label
        """
        # Convert verification scores to hallucination probabilities
        # (inverse: high correctness = low hallucination)
        entity_hallucination = 1.0 - entity_verification_score
        
        if agent_verification_score is not None:
            agent_hallucination = 1.0 - agent_verification_score
        else:
            agent_hallucination = None
        
        # Weighted fusion
        if agent_verification_score is not None:
            # Three-way fusion
            fusion_score = (
                self.alpha * classification_score +
                self.beta * entity_hallucination +
                self.gamma * agent_hallucination
            )
        else:
            # Two-way fusion (no agent verification)
            # Adjust weights to sum to 1
            alpha_adj = self.alpha / (self.alpha + self.beta)
            beta_adj = self.beta / (self.alpha + self.beta)
            
            fusion_score = (
                alpha_adj * classification_score +
                beta_adj * entity_hallucination
            )
        
        # Classify
        label = "hallucinated" if fusion_score >= self.threshold else "factual"
        
        # Confidence: distance from threshold (if not already computed)
        if 'confidence' not in locals():
            confidence = abs(fusion_score - self.threshold) * 2  # Scale to [0, 1]
        
        # Fusion weights (if not already computed)
        if 'fusion_weights' not in locals():
            fusion_weights = {
                "alpha": self.alpha,
                "beta": self.beta,
                "gamma": self.gamma if agent_verification_score is not None else 0.0
            }
        
        return {
            "final_hallucination_score": fusion_score,
            "label": label,
            "confidence": confidence,
            "fusion_weights": fusion_weights
        }
    
    def _classic_fuse(
        self,
        classification_score: float,
        entity_verification_score: float,
        agent_verification_score: Optional[float] = None
    ) -> Tuple[float, float, Dict]:
        """Classic weighted fusion."""
        # Convert verification scores to hallucination probabilities
        entity_hallucination = 1.0 - entity_verification_score
        
        if agent_verification_score is not None:
            agent_hallucination = 1.0 - agent_verification_score
            # Three-way fusion
            fusion_score = (
                self.alpha * classification_score +
                self.beta * entity_hallucination +
                self.gamma * agent_hallucination
            )
        else:
            # Two-way fusion
            alpha_adj = self.alpha / (self.alpha + self.beta)
            beta_adj = self.beta / (self.alpha + self.beta)
            fusion_score = (
                alpha_adj * classification_score +
                beta_adj * entity_hallucination
            )
        
        # Clamp to [0, 1]
        fusion_score = max(0.0, min(1.0, fusion_score))
        
        # Confidence
        confidence = abs(fusion_score - self.threshold) * 2
        
        # Weights
        fusion_weights = {
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma if agent_verification_score is not None else 0.0
        }
        
        return fusion_score, confidence, fusion_weights
    
    def fuse_span_result(
        self,
        sentence: str,
        sentence_index: int,
        classification_score: float,
        entity_verification_score: float,
        agent_verification_score: Optional[float] = None
    ) -> SpanFusionResult:
        """
        Fuse scores and return SpanFusionResult.
        
        Args:
            sentence: Sentence text
            sentence_index: Index of sentence
            classification_score: Transformer score
            entity_verification_score: Entity verification score
            agent_verification_score: Agent verification score (optional)
        
        Returns:
            SpanFusionResult object
        """
        fusion_dict = self.fuse(
            classification_score,
            entity_verification_score,
            agent_verification_score
        )
        
        return SpanFusionResult(
            sentence=sentence,
            sentence_index=sentence_index,
            classification_score=classification_score,
            entity_verification_score=entity_verification_score,
            agent_verification_score=agent_verification_score,
            final_hallucination_score=fusion_dict["final_hallucination_score"],
            label=fusion_dict["label"],
            confidence=fusion_dict["confidence"],
            fusion_weights=fusion_dict["fusion_weights"]
        )
    
    def fuse_batch(
        self,
        sentences: List[str],
        classification_scores: List[float],
        entity_verification_scores: List[float],
        agent_verification_scores: Optional[List[float]] = None
    ) -> List[SpanFusionResult]:
        """
        Fuse scores for multiple sentences.
        
        Args:
            sentences: List of sentences
            classification_scores: List of classification scores
            entity_verification_scores: List of entity verification scores
            agent_verification_scores: Optional list of agent verification scores
        
        Returns:
            List of SpanFusionResult objects
        """
        results = []
        
        for i, (sentence, cls_score, ent_score) in enumerate(zip(
            sentences, classification_scores, entity_verification_scores
        )):
            agent_score = agent_verification_scores[i] if agent_verification_scores else None
            
            result = self.fuse_span_result(
                sentence=sentence,
                sentence_index=i,
                classification_score=cls_score,
                entity_verification_score=ent_score,
                agent_verification_score=agent_score
            )
            results.append(result)
        
        return results


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("Span Fusion Test")
    print("=" * 70)
    
    fusion = SpanFusion(alpha=0.5, beta=0.3, gamma=0.2, threshold=0.5)
    
    # Example scores
    test_cases = [
        {
            "sentence": "Barack Obama was the 44th President.",
            "classification": 0.2,  # Low hallucination
            "entity": 0.9,  # High correctness
            "agent": 0.85  # High correctness
        },
        {
            "sentence": "The moon is made of cheese.",
            "classification": 0.8,  # High hallucination
            "entity": 0.3,  # Low correctness
            "agent": 0.2  # Low correctness
        }
    ]
    
    print("\nFusing scores...")
    for case in test_cases:
        result = fusion.fuse_span_result(
            sentence=case["sentence"],
            sentence_index=0,
            classification_score=case["classification"],
            entity_verification_score=case["entity"],
            agent_verification_score=case["agent"]
        )
        
        print(f"\nSentence: {case['sentence']}")
        print(f"  Classification: {result.classification_score:.3f}")
        print(f"  Entity verification: {result.entity_verification_score:.3f}")
        print(f"  Agent verification: {result.agent_verification_score:.3f}")
        print(f"  Final score: {result.final_hallucination_score:.3f}")
        print(f"  Label: {result.label.upper()}")
        print(f"  Confidence: {result.confidence:.3f}")
    
    print("\n" + "=" * 70)

