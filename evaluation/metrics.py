"""
Advanced Evaluation Metrics for Hallucination Detection
Implements research-grade metrics beyond standard classification metrics.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import json

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False


@dataclass
class MetricResult:
    """Result of a metric computation."""
    metric_name: str
    value: float
    confidence_interval: Optional[Tuple[float, float]] = None
    per_sample_scores: Optional[List[float]] = None
    metadata: Optional[Dict] = None


class TruthfulnessConfidenceMetric:
    """
    Measures the confidence-calibrated truthfulness of predictions.
    Combines prediction accuracy with confidence calibration.
    """
    
    def __init__(self, num_bins: int = 10):
        """
        Initialize metric.
        
        Args:
            num_bins: Number of bins for calibration
        """
        self.num_bins = num_bins
    
    def compute(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
        confidence: Optional[np.ndarray] = None
    ) -> MetricResult:
        """
        Compute truthfulness confidence metric.
        
        Args:
            y_true: True labels (0=correct, 1=hallucination)
            y_pred: Predicted labels
            y_prob: Predicted probabilities
            confidence: Optional confidence scores
        
        Returns:
            MetricResult with truthfulness confidence score
        """
        # If confidence not provided, use probability as proxy
        if confidence is None:
            confidence = np.abs(y_prob - 0.5) * 2  # Convert to 0-1 scale
        
        # Compute calibration error
        calibration_error = self._compute_calibration_error(
            y_true, y_pred, y_prob
        )
        
        # Compute confidence-weighted accuracy
        correct = (y_true == y_pred).astype(float)
        confidence_weighted_acc = np.sum(correct * confidence) / np.sum(confidence)
        
        # Truthfulness confidence combines accuracy and calibration
        truthfulness_score = confidence_weighted_acc * (1 - calibration_error)
        
        # Compute per-sample scores
        per_sample = correct * confidence * (1 - calibration_error)
        
        return MetricResult(
            metric_name="truthfulness_confidence",
            value=float(truthfulness_score),
            confidence_interval=self._bootstrap_ci(truthfulness_score, per_sample),
            per_sample_scores=per_sample.tolist(),
            metadata={
                "calibration_error": float(calibration_error),
                "confidence_weighted_accuracy": float(confidence_weighted_acc)
            }
        )
    
    def _compute_calibration_error(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray
    ) -> float:
        """Compute Expected Calibration Error (ECE)."""
        # Bin predictions
        bin_boundaries = np.linspace(0, 1, self.num_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find samples in this bin
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                # Accuracy in this bin
                accuracy_in_bin = (y_true[in_bin] == y_pred[in_bin]).mean()
                # Average confidence in this bin
                avg_confidence_in_bin = y_prob[in_bin].mean()
                # Add to ECE
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return float(ece)
    
    def _bootstrap_ci(
        self,
        statistic: float,
        per_sample: np.ndarray,
        n_bootstrap: int = 1000,
        confidence_level: float = 0.95
    ) -> Tuple[float, float]:
        """Compute bootstrap confidence interval."""
        n = len(per_sample)
        bootstrap_stats = []
        
        for _ in range(n_bootstrap):
            indices = np.random.choice(n, size=n, replace=True)
            bootstrap_stat = np.mean(per_sample[indices])
            bootstrap_stats.append(bootstrap_stat)
        
        alpha = 1 - confidence_level
        lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
        upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
        
        return (float(lower), float(upper))


class SemanticFactDivergenceMetric:
    """
    Measures semantic divergence between predicted and ground truth facts.
    Uses semantic embeddings to quantify how far off predictions are.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize metric.
        
        Args:
            model_name: Sentence transformer model name
        """
        self.model = None
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.model = SentenceTransformer(model_name)
            except:
                pass
    
    def compute(
        self,
        responses: List[str],
        y_true: np.ndarray,
        y_pred: np.ndarray,
        reference_texts: Optional[List[str]] = None
    ) -> MetricResult:
        """
        Compute semantic fact divergence.
        
        Args:
            responses: Original response texts
            y_true: True labels
            y_pred: Predicted labels
            reference_texts: Optional reference correct texts
        
        Returns:
            MetricResult with divergence score
        """
        if self.model is None:
            # Fallback: use simple word overlap
            return self._compute_word_overlap(responses, y_true, y_pred)
        
        # Separate correct and hallucinated responses
        hallucinated_indices = np.where(y_true == 1)[0]
        correct_indices = np.where(y_true == 0)[0]
        
        if len(hallucinated_indices) == 0:
            return MetricResult(
                metric_name="semantic_fact_divergence",
                value=0.0,
                metadata={"note": "No hallucinated examples"}
            )
        
        # Compute embeddings
        hallucinated_texts = [responses[i] for i in hallucinated_indices]
        correct_texts = [responses[i] for i in correct_indices] if len(correct_indices) > 0 else None
        
        hallucinated_embeddings = self.model.encode(hallucinated_texts)
        
        # Compute divergence
        if correct_texts and len(correct_texts) > 0:
            correct_embeddings = self.model.encode(correct_texts)
            # Divergence = distance from hallucinated to correct cluster
            divergence = self._compute_cluster_divergence(
                hallucinated_embeddings, correct_embeddings
            )
        else:
            # No correct examples, use self-divergence
            divergence = self._compute_self_divergence(hallucinated_embeddings)
        
        # Per-sample divergences
        per_sample = self._compute_per_sample_divergence(
            responses, y_true, hallucinated_indices
        )
        
        return MetricResult(
            metric_name="semantic_fact_divergence",
            value=float(divergence),
            per_sample_scores=per_sample,
            metadata={"num_hallucinated": len(hallucinated_indices)}
        )
    
    def _compute_cluster_divergence(
        self,
        hallucinated_emb: np.ndarray,
        correct_emb: np.ndarray
    ) -> float:
        """Compute divergence between hallucinated and correct clusters."""
        # Mean embeddings
        hallucinated_mean = np.mean(hallucinated_emb, axis=0)
        correct_mean = np.mean(correct_emb, axis=0)
        
        # Cosine distance
        divergence = 1 - cosine_similarity(
            [hallucinated_mean], [correct_mean]
        )[0][0]
        
        return float(divergence)
    
    def _compute_self_divergence(self, embeddings: np.ndarray) -> float:
        """Compute self-divergence (variance) of embeddings."""
        if len(embeddings) < 2:
            return 0.0
        
        # Compute pairwise distances
        distances = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                dist = 1 - cosine_similarity(
                    [embeddings[i]], [embeddings[j]]
                )[0][0]
                distances.append(dist)
        
        return float(np.mean(distances))
    
    def _compute_per_sample_divergence(
        self,
        responses: List[str],
        y_true: np.ndarray,
        hallucinated_indices: np.ndarray
    ) -> List[float]:
        """Compute divergence per sample."""
        if self.model is None:
            return [0.5] * len(hallucinated_indices)
        
        # For each hallucinated response, compute distance to nearest correct
        correct_indices = np.where(y_true == 0)[0]
        if len(correct_indices) == 0:
            return [0.5] * len(hallucinated_indices)
        
        hallucinated_texts = [responses[i] for i in hallucinated_indices]
        correct_texts = [responses[i] for i in correct_indices]
        
        hallucinated_emb = self.model.encode(hallucinated_texts)
        correct_emb = self.model.encode(correct_texts)
        
        per_sample = []
        for h_emb in hallucinated_emb:
            # Distance to nearest correct
            similarities = cosine_similarity([h_emb], correct_emb)[0]
            max_sim = np.max(similarities)
            divergence = 1 - max_sim
            per_sample.append(float(divergence))
        
        return per_sample
    
    def _compute_word_overlap(
        self,
        responses: List[str],
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> MetricResult:
        """Fallback: compute word overlap divergence."""
        # Simple word-based divergence
        hallucinated = [responses[i] for i in np.where(y_true == 1)[0]]
        correct = [responses[i] for i in np.where(y_true == 0)[0]]
        
        if not hallucinated or not correct:
            return MetricResult(
                metric_name="semantic_fact_divergence",
                value=0.0
            )
        
        # Average word overlap
        overlaps = []
        for h_text in hallucinated[:10]:  # Sample
            h_words = set(h_text.lower().split())
            for c_text in correct[:10]:
                c_words = set(c_text.lower().split())
                overlap = len(h_words & c_words) / max(len(h_words | c_words), 1)
                overlaps.append(1 - overlap)  # Divergence
        
        divergence = np.mean(overlaps) if overlaps else 0.5
        
        return MetricResult(
            metric_name="semantic_fact_divergence",
            value=float(divergence),
            metadata={"method": "word_overlap"}
        )


class CausalHallucinationChainMetric:
    """
    Detects and measures causal chains of hallucinations.
    Identifies when one hallucination leads to another.
    """
    
    def __init__(self):
        """Initialize metric."""
        self.nlp = None
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except:
                pass
    
    def compute(
        self,
        responses: List[str],
        y_true: np.ndarray,
        conversation_contexts: Optional[List[List[str]]] = None
    ) -> MetricResult:
        """
        Compute causal hallucination chain metric.
        
        Args:
            responses: Response texts
            y_true: True labels (1=hallucination)
            conversation_contexts: Optional conversation history
        
        Returns:
            MetricResult with chain metrics
        """
        # Identify hallucinated responses
        hallucinated_indices = np.where(y_true == 1)[0]
        
        if len(hallucinated_indices) < 2:
            return MetricResult(
                metric_name="causal_hallucination_chain",
                value=0.0,
                metadata={"note": "Insufficient hallucinated examples for chains"}
            )
        
        # Detect chains
        chains = self._detect_chains(
            responses, hallucinated_indices, conversation_contexts
        )
        
        # Compute chain metrics
        chain_lengths = [len(chain) for chain in chains]
        avg_chain_length = np.mean(chain_lengths) if chain_lengths else 0.0
        max_chain_length = max(chain_lengths) if chain_lengths else 0
        num_chains = len(chains)
        chain_ratio = num_chains / len(hallucinated_indices) if hallucinated_indices.size > 0 else 0.0
        
        return MetricResult(
            metric_name="causal_hallucination_chain",
            value=float(avg_chain_length),
            metadata={
                "num_chains": num_chains,
                "max_chain_length": int(max_chain_length),
                "chain_ratio": float(chain_ratio),
                "chain_lengths": chain_lengths
            }
        )
    
    def _detect_chains(
        self,
        responses: List[str],
        hallucinated_indices: np.ndarray,
        contexts: Optional[List[List[str]]]
    ) -> List[List[int]]:
        """Detect causal chains of hallucinations."""
        chains = []
        processed = set()
        
        for idx in hallucinated_indices:
            if idx in processed:
                continue
            
            # Start new chain
            chain = [idx]
            current_idx = idx
            
            # Follow causal links
            while True:
                next_idx = self._find_causal_link(
                    responses, current_idx, hallucinated_indices, contexts
                )
                
                if next_idx is None or next_idx in chain:
                    break
                
                chain.append(next_idx)
                processed.add(next_idx)
                current_idx = next_idx
            
            if len(chain) > 1:
                chains.append(chain)
                processed.update(chain)
        
        return chains
    
    def _find_causal_link(
        self,
        responses: List[str],
        current_idx: int,
        hallucinated_indices: np.ndarray,
        contexts: Optional[List[List[str]]]
    ) -> Optional[int]:
        """Find next hallucination in causal chain."""
        current_text = responses[current_idx]
        
        # Extract entities/concepts from current hallucination
        current_entities = self._extract_entities(current_text)
        
        # Look for other hallucinations that reference these entities
        for other_idx in hallucinated_indices:
            if other_idx == current_idx:
                continue
            
            other_text = responses[other_idx]
            other_entities = self._extract_entities(other_text)
            
            # Check for entity overlap (potential causal link)
            if current_entities & other_entities:
                # Check if contexts suggest causal relationship
                if contexts and self._check_causal_relationship(
                    current_text, other_text, contexts[current_idx], contexts[other_idx]
                ):
                    return other_idx
        
        return None
    
    def _extract_entities(self, text: str) -> set:
        """Extract entities from text."""
        if self.nlp:
            doc = self.nlp(text)
            return {ent.text.lower() for ent in doc.ents}
        else:
            # Simple: capitalized words
            import re
            return {w.lower() for w in re.findall(r'\b[A-Z][a-z]+\b', text)}
    
    def _check_causal_relationship(
        self,
        text1: str,
        text2: str,
        context1: List[str],
        context2: List[str]
    ) -> bool:
        """Check if two texts have causal relationship."""
        # Simple heuristic: check for causal keywords
        causal_keywords = ['because', 'due to', 'caused', 'led to', 'resulted in']
        
        combined = ' '.join([text1, text2] + context1 + context2).lower()
        return any(keyword in combined for keyword in causal_keywords)


class AdvancedMetrics:
    """
    Comprehensive advanced metrics aggregator.
    Computes all advanced metrics and provides summary.
    """
    
    def __init__(self):
        """Initialize metrics."""
        self.truthfulness_metric = TruthfulnessConfidenceMetric()
        self.semantic_metric = SemanticFactDivergenceMetric()
        self.causal_metric = CausalHallucinationChainMetric()
    
    def compute_all(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
        responses: List[str],
        confidence: Optional[np.ndarray] = None,
        conversation_contexts: Optional[List[List[str]]] = None
    ) -> Dict[str, MetricResult]:
        """
        Compute all advanced metrics.
        
        Returns:
            Dictionary of metric results
        """
        results = {}
        
        # Truthfulness confidence
        results['truthfulness_confidence'] = self.truthfulness_metric.compute(
            y_true, y_pred, y_prob, confidence
        )
        
        # Semantic fact divergence
        results['semantic_divergence'] = self.semantic_metric.compute(
            responses, y_true, y_pred
        )
        
        # Causal hallucination chains
        results['causal_chains'] = self.causal_metric.compute(
            responses, y_true, conversation_contexts
        )
        
        return results
    
    def summary(self, results: Dict[str, MetricResult]) -> str:
        """Generate summary of all metrics."""
        lines = ["Advanced Metrics Summary", "=" * 50]
        
        for name, result in results.items():
            lines.append(f"\n{name}:")
            lines.append(f"  Value: {result.value:.4f}")
            if result.confidence_interval:
                ci = result.confidence_interval
                lines.append(f"  95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")
            if result.metadata:
                for key, value in result.metadata.items():
                    lines.append(f"  {key}: {value}")
        
        return "\n".join(lines)

