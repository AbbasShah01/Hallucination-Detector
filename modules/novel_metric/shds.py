"""
Semantic Hallucination Divergence Score (SHDS)

A novel metric for measuring hallucination severity by combining multiple signals:
1. Semantic embedding divergence from factual content
2. Entity mismatch penalty
3. Reasoning inconsistency score
4. Token-level uncertainty

Formula:
    SHDS = w1 * EmbeddingDivergence
         + w2 * EntityMismatchPenalty
         + w3 * ReasoningInconsistency
         + w4 * TokenUncertainty

Where all components are normalized to [0,1] and weights sum to 1.

This metric provides a fine-grained severity score that captures multiple
dimensions of hallucination beyond binary classification.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import torch
import torch.nn.functional as F

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence-transformers not available. SHDS embedding divergence will use fallback.")

try:
    from transformers import AutoTokenizer, AutoModelForMaskedLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not available. SHDS token uncertainty will use fallback.")


@dataclass
class SHDSResult:
    """Result of SHDS computation."""
    shds_score: float  # Final SHDS score [0,1]
    embedding_divergence: float  # Semantic distance component
    entity_mismatch_penalty: float  # Entity verification component
    reasoning_inconsistency: float  # Reasoning coherence component
    token_uncertainty: float  # Token-level uncertainty component
    components: Dict[str, float]  # Individual component scores
    normalized: bool  # Whether scores are normalized


class SHDS:
    """
    Semantic Hallucination Divergence Score calculator.
    
    This novel metric combines multiple signals to provide a comprehensive
    hallucination severity score that goes beyond binary classification.
    
    Research Contribution:
    - First metric to combine semantic, factual, reasoning, and uncertainty signals
    - Provides fine-grained severity assessment
    - Enables adaptive fusion based on hallucination type
    """
    
    def __init__(
        self,
        w1: float = 0.3,  # Weight for embedding divergence
        w2: float = 0.3,  # Weight for entity mismatch
        w3: float = 0.2,  # Weight for reasoning inconsistency
        w4: float = 0.2,  # Weight for token uncertainty
        embedding_model: str = "all-MiniLM-L6-v2",
        normalize: bool = True
    ):
        """
        Initialize SHDS calculator.
        
        Args:
            w1: Weight for semantic embedding divergence
            w2: Weight for entity mismatch penalty
            w3: Weight for reasoning inconsistency
            w4: Weight for token uncertainty
            embedding_model: Sentence transformer model name
            normalize: Whether to normalize components to [0,1]
        """
        # Normalize weights to sum to 1
        total_weight = w1 + w2 + w3 + w4
        self.w1 = w1 / total_weight
        self.w2 = w2 / total_weight
        self.w3 = w3 / total_weight
        self.w4 = w4 / total_weight
        
        self.normalize = normalize
        
        # Initialize embedding model
        self.embedding_model = None
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.embedding_model = SentenceTransformer(embedding_model)
            except Exception as e:
                print(f"Warning: Could not load embedding model: {e}")
        
        # Initialize tokenizer for uncertainty
        self.tokenizer = None
        self.uncertainty_model = None
        if TRANSFORMERS_AVAILABLE:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
                # We'll use a simple approach for token uncertainty
            except Exception as e:
                print(f"Warning: Could not load tokenizer: {e}")
    
    def compute_embedding_divergence(
        self,
        span: str,
        factual_correction: Optional[str] = None,
        reference_embeddings: Optional[np.ndarray] = None
    ) -> float:
        """
        Compute semantic embedding divergence.
        
        Measures cosine distance between the span and its factual correction
        or reference embeddings.
        
        Args:
            span: Text span to evaluate
            factual_correction: Optional factual correction text
            reference_embeddings: Optional reference embeddings
        
        Returns:
            Embedding divergence score [0,1] (higher = more divergent)
        """
        if self.embedding_model is None:
            # Fallback: use simple heuristic
            return 0.5
        
        try:
            # Get span embedding
            span_embedding = self.embedding_model.encode(span, convert_to_numpy=True)
            span_embedding = span_embedding / np.linalg.norm(span_embedding)
            
            if factual_correction:
                # Compare with factual correction
                fact_embedding = self.embedding_model.encode(
                    factual_correction, convert_to_numpy=True
                )
                fact_embedding = fact_embedding / np.linalg.norm(fact_embedding)
                
                # Cosine distance = 1 - cosine similarity
                cosine_sim = np.dot(span_embedding, fact_embedding)
                divergence = 1.0 - cosine_sim
            
            elif reference_embeddings is not None:
                # Compare with reference embeddings
                reference_embeddings = reference_embeddings / np.linalg.norm(
                    reference_embeddings, axis=1, keepdims=True
                )
                cosine_sims = np.dot(reference_embeddings, span_embedding)
                divergence = 1.0 - np.max(cosine_sims)
            
            else:
                # No reference: use self-consistency heuristic
                # Split span and check internal consistency
                words = span.split()
                if len(words) < 2:
                    return 0.3  # Short spans have lower divergence
                
                # Embed first and second half
                mid = len(words) // 2
                first_half = " ".join(words[:mid])
                second_half = " ".join(words[mid:])
                
                first_emb = self.embedding_model.encode(first_half, convert_to_numpy=True)
                second_emb = self.embedding_model.encode(second_half, convert_to_numpy=True)
                
                first_emb = first_emb / np.linalg.norm(first_emb)
                second_emb = second_emb / np.linalg.norm(second_emb)
                
                consistency = np.dot(first_emb, second_emb)
                divergence = 1.0 - consistency
            
            # Normalize to [0,1] and ensure non-negative
            divergence = max(0.0, min(1.0, divergence))
            return divergence
        
        except Exception as e:
            print(f"Warning: Embedding divergence computation failed: {e}")
            return 0.5  # Neutral fallback
    
    def compute_entity_mismatch_penalty(
        self,
        failed_entity_checks: int,
        total_entities: int,
        entity_verification_scores: Optional[List[float]] = None
    ) -> float:
        """
        Compute entity mismatch penalty.
        
        Penalizes spans with failed entity verifications.
        
        Args:
            failed_entity_checks: Number of entities that failed verification
            total_entities: Total number of entities found
            entity_verification_scores: Optional list of verification scores
        
        Returns:
            Entity mismatch penalty [0,1] (higher = more mismatches)
        """
        if total_entities == 0:
            # No entities: neutral penalty (no factual claims)
            return 0.3
        
        if entity_verification_scores is not None:
            # Use verification scores if available
            avg_score = np.mean(entity_verification_scores)
            # Invert: low verification = high penalty
            penalty = 1.0 - avg_score
        else:
            # Use failure ratio
            failure_ratio = failed_entity_checks / total_entities
            penalty = failure_ratio
        
        # Normalize to [0,1]
        penalty = max(0.0, min(1.0, penalty))
        return penalty
    
    def compute_reasoning_inconsistency(
        self,
        contradiction_score: Optional[float] = None,
        agentic_verification_score: Optional[float] = None,
        logical_coherence: Optional[float] = None
    ) -> float:
        """
        Compute reasoning inconsistency score.
        
        Measures contradictions and logical incoherence from agentic verification.
        
        Args:
            contradiction_score: Direct contradiction score [0,1]
            agentic_verification_score: Agent verification score (higher = more correct)
            logical_coherence: Logical coherence score [0,1]
        
        Returns:
            Reasoning inconsistency [0,1] (higher = more inconsistent)
        """
        if contradiction_score is not None:
            return contradiction_score
        
        if agentic_verification_score is not None:
            # Invert: low verification = high inconsistency
            inconsistency = 1.0 - agentic_verification_score
            return max(0.0, min(1.0, inconsistency))
        
        if logical_coherence is not None:
            # Invert coherence to get inconsistency
            inconsistency = 1.0 - logical_coherence
            return max(0.0, min(1.0, inconsistency))
        
        # Fallback: neutral
        return 0.5
    
    def compute_token_uncertainty(
        self,
        span: str,
        model_logits: Optional[torch.Tensor] = None
    ) -> float:
        """
        Compute token-level uncertainty.
        
        Measures average token entropy for the span.
        
        Args:
            span: Text span to evaluate
            model_logits: Optional model logits for tokens
        
        Returns:
            Token uncertainty [0,1] (higher = more uncertain)
        """
        if model_logits is not None:
            # Compute entropy from logits
            probs = F.softmax(model_logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
            avg_entropy = torch.mean(entropy).item()
            
            # Normalize entropy (max entropy for vocab size V is log(V))
            # Approximate normalization
            normalized_entropy = min(1.0, avg_entropy / 10.0)  # Rough normalization
            return normalized_entropy
        
        # Fallback: use span characteristics
        # Longer spans with rare words = higher uncertainty
        words = span.split()
        if len(words) == 0:
            return 0.0
        
        # Heuristic: check for unusual patterns
        # (This is a simplified fallback)
        unusual_indicators = [
            len([w for w in words if w.isupper() and len(w) > 1]) / len(words),
            len([w for w in words if any(c.isdigit() for c in w)]) / len(words),
        ]
        
        uncertainty = np.mean(unusual_indicators) if unusual_indicators else 0.3
        return max(0.0, min(1.0, uncertainty))
    
    def compute(
        self,
        span: str,
        factual_correction: Optional[str] = None,
        failed_entity_checks: int = 0,
        total_entities: int = 0,
        entity_verification_scores: Optional[List[float]] = None,
        contradiction_score: Optional[float] = None,
        agentic_verification_score: Optional[float] = None,
        model_logits: Optional[torch.Tensor] = None,
        reference_embeddings: Optional[np.ndarray] = None
    ) -> SHDSResult:
        """
        Compute complete SHDS score.
        
        Args:
            span: Text span to evaluate
            factual_correction: Optional factual correction
            failed_entity_checks: Number of failed entity verifications
            total_entities: Total entities found
            entity_verification_scores: Entity verification scores
            contradiction_score: Contradiction score
            agentic_verification_score: Agent verification score
            model_logits: Model logits for uncertainty
            reference_embeddings: Reference embeddings
        
        Returns:
            SHDSResult with all components and final score
        """
        # Compute all components
        embedding_div = self.compute_embedding_divergence(
            span, factual_correction, reference_embeddings
        )
        
        entity_penalty = self.compute_entity_mismatch_penalty(
            failed_entity_checks, total_entities, entity_verification_scores
        )
        
        reasoning_inconsistency = self.compute_reasoning_inconsistency(
            contradiction_score, agentic_verification_score
        )
        
        token_uncertainty = self.compute_token_uncertainty(span, model_logits)
        
        # Normalize components if requested
        if self.normalize:
            # Components are already in [0,1], but ensure they are
            embedding_div = max(0.0, min(1.0, embedding_div))
            entity_penalty = max(0.0, min(1.0, entity_penalty))
            reasoning_inconsistency = max(0.0, min(1.0, reasoning_inconsistency))
            token_uncertainty = max(0.0, min(1.0, token_uncertainty))
        
        # Weighted combination
        shds_score = (
            self.w1 * embedding_div +
            self.w2 * entity_penalty +
            self.w3 * reasoning_inconsistency +
            self.w4 * token_uncertainty
        )
        
        # Ensure final score is in [0,1]
        shds_score = max(0.0, min(1.0, shds_score))
        
        return SHDSResult(
            shds_score=shds_score,
            embedding_divergence=embedding_div,
            entity_mismatch_penalty=entity_penalty,
            reasoning_inconsistency=reasoning_inconsistency,
            token_uncertainty=token_uncertainty,
            components={
                "embedding_divergence": embedding_div,
                "entity_mismatch_penalty": entity_penalty,
                "reasoning_inconsistency": reasoning_inconsistency,
                "token_uncertainty": token_uncertainty,
                "weights": {
                    "w1": self.w1,
                    "w2": self.w2,
                    "w3": self.w3,
                    "w4": self.w4
                }
            },
            normalized=self.normalize
        )


def compute_shds(
    span: str,
    **kwargs
) -> SHDSResult:
    """
    Convenience function to compute SHDS.
    
    Args:
        span: Text span to evaluate
        **kwargs: Additional arguments for SHDS.compute()
    
    Returns:
        SHDSResult
    """
    shds = SHDS()
    return shds.compute(span, **kwargs)


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("Semantic Hallucination Divergence Score (SHDS) Test")
    print("=" * 70)
    
    shds = SHDS(w1=0.3, w2=0.3, w3=0.2, w4=0.2)
    
    # Test cases
    test_cases = [
        {
            "span": "The moon is made of cheese.",
            "factual_correction": "The moon is composed of rock and dust.",
            "failed_entity_checks": 2,
            "total_entities": 2,
            "agentic_verification_score": 0.1
        },
        {
            "span": "Barack Obama was the 44th President of the United States.",
            "factual_correction": None,
            "failed_entity_checks": 0,
            "total_entities": 3,
            "agentic_verification_score": 0.95
        }
    ]
    
    print("\nComputing SHDS scores...")
    for i, case in enumerate(test_cases):
        result = shds.compute(
            span=case["span"],
            factual_correction=case.get("factual_correction"),
            failed_entity_checks=case["failed_entity_checks"],
            total_entities=case["total_entities"],
            agentic_verification_score=case["agentic_verification_score"]
        )
        
        print(f"\nTest Case {i+1}:")
        print(f"  Span: {case['span']}")
        print(f"  SHDS Score: {result.shds_score:.4f}")
        print(f"  Components:")
        print(f"    Embedding Divergence: {result.embedding_divergence:.4f}")
        print(f"    Entity Mismatch: {result.entity_mismatch_penalty:.4f}")
        print(f"    Reasoning Inconsistency: {result.reasoning_inconsistency:.4f}")
        print(f"    Token Uncertainty: {result.token_uncertainty:.4f}")
    
    print("\n" + "=" * 70)

