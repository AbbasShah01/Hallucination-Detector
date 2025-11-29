"""
Span-Level Inference Pipeline

Main orchestrator for sentence-level hallucination detection.
Coordinates all components: splitting, classification, verification, and fusion.
"""

from typing import List, Dict, Optional, Union
import json
from dataclasses import dataclass, asdict

from .sentence_splitter import SentenceSplitter
from .span_classifier import SpanClassifier
from .span_entity_verifier import SpanEntityVerifier
from .span_agent_verifier import SpanAgentVerifier
from .span_fusion import SpanFusion


@dataclass
class SpanDetectionResult:
    """Complete result for a single sentence."""
    sentence: str
    classification_score: float
    entity_verification_score: float
    agent_verification_score: Optional[float]
    final_hallucination_score: float
    label: str  # "hallucinated" or "factual"
    confidence: float
    sentence_index: int


class SpanInferencePipeline:
    """
    Complete pipeline for sentence-level hallucination detection.
    
    Orchestrates:
    1. Sentence splitting
    2. Per-sentence classification
    3. Per-sentence entity verification
    4. Per-sentence agent verification
    5. Per-sentence fusion
    6. Result aggregation
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        splitter_method: str = "nltk",
        extractor_method: str = "transformers",
        use_entity_verification: bool = True,
        use_agent_verification: bool = False,
        fusion_alpha: float = 0.5,
        fusion_beta: float = 0.3,
        fusion_gamma: float = 0.2,
        fusion_threshold: float = 0.5
    ):
        """
        Initialize span inference pipeline.
        
        Args:
            model_path: Path to fine-tuned transformer model
            splitter_method: "nltk", "spacy", or "regex"
            extractor_method: "spacy" or "transformers"
            use_entity_verification: Whether to use entity verification
            use_agent_verification: Whether to use agent verification
            fusion_alpha: Weight for classification
            fusion_beta: Weight for entity verification
            fusion_gamma: Weight for agent verification
            fusion_threshold: Threshold for hallucination classification
        """
        # Initialize components
        self.splitter = SentenceSplitter(method=splitter_method)
        self.classifier = SpanClassifier(model_path=model_path)
        self.entity_verifier = SpanEntityVerifier(
            extractor_method=extractor_method
        ) if use_entity_verification else None
        self.agent_verifier = SpanAgentVerifier() if use_agent_verification else None
        # Get fusion method from config or default
        fusion_method = "classic"  # Default
        
        self.fusion = SpanFusion(
            alpha=fusion_alpha,
            beta=fusion_beta,
            gamma=fusion_gamma,
            threshold=fusion_threshold,
            fusion_method=fusion_method
        )
        
        self.use_entity_verification = use_entity_verification
        self.use_agent_verification = use_agent_verification
    
    def detect(
        self,
        text: str,
        return_json: bool = True
    ) -> Union[List[Dict], List[SpanDetectionResult]]:
        """
        Detect hallucinations at sentence level.
        
        Args:
            text: Input text to analyze
            return_json: If True, return list of dicts; else return SpanDetectionResult objects
        
        Returns:
            List of detection results (dicts or SpanDetectionResult objects)
        """
        # Step 1: Split into sentences
        sentences_with_context = self.splitter.split_with_context(text, context_window=1)
        
        if not sentences_with_context:
            return []
        
        # Step 2: Classify sentences
        classification_results = self.classifier.classify_with_metadata(sentences_with_context)
        
        # Step 3: Verify entities (if enabled)
        if self.use_entity_verification and self.entity_verifier:
            entity_results = self.entity_verifier.verify_with_metadata(sentences_with_context)
        else:
            # Use neutral scores
            entity_results = [
                type('obj', (object,), {
                    'sentence': s['sentence'],
                    'sentence_index': s['sentence_index'],
                    'entity_verification_score': 0.5
                })()
                for s in sentences_with_context
            ]
        
        # Step 4: Agent verification (if enabled)
        if self.use_agent_verification and self.agent_verifier:
            agent_results = self.agent_verifier.verify_with_metadata(sentences_with_context)
        else:
            agent_results = None
        
        # Step 5: Fuse scores
        classification_scores = [r.classification_score for r in classification_results]
        entity_scores = [
            r.entity_verification_score if hasattr(r, 'entity_verification_score') else 0.5
            for r in entity_results
        ]
        agent_scores = (
            [r.agent_verification_score for r in agent_results]
            if agent_results else None
        )
        
        fusion_results = self.fusion.fuse_batch(
            sentences=[s['sentence'] for s in sentences_with_context],
            classification_scores=classification_scores,
            entity_verification_scores=entity_scores,
            agent_verification_scores=agent_scores
        )
        
        # Step 6: Format results
        if return_json:
            return self._format_as_json(fusion_results, entity_results, agent_results)
        else:
            return self._format_as_objects(fusion_results, entity_results, agent_results)
    
    def _format_as_json(
        self,
        fusion_results: List,
        entity_results: List,
        agent_results: Optional[List]
    ) -> List[Dict]:
        """Format results as JSON-serializable dicts."""
        results = []
        
        for i, fusion_result in enumerate(fusion_results):
            entity_result = entity_results[i] if i < len(entity_results) else None
            agent_result = agent_results[i] if agent_results and i < len(agent_results) else None
            
            result_dict = {
                "sentence": fusion_result.sentence,
                "classification_score": round(fusion_result.classification_score, 4),
                "entity_verification_score": (
                    round(entity_result.entity_verification_score, 4)
                    if entity_result and hasattr(entity_result, 'entity_verification_score')
                    else None
                ),
                "agent_verification_score": (
                    round(agent_result.agent_verification_score, 4)
                    if agent_result and hasattr(agent_result, 'agent_verification_score')
                    else None
                ),
                "final_hallucination_score": round(fusion_result.final_hallucination_score, 4),
                "label": fusion_result.label,
                "confidence": round(fusion_result.confidence, 4),
                "sentence_index": fusion_result.sentence_index
            }
            
            results.append(result_dict)
        
        return results
    
    def _format_as_objects(
        self,
        fusion_results: List,
        entity_results: List,
        agent_results: Optional[List]
    ) -> List[SpanDetectionResult]:
        """Format results as SpanDetectionResult objects."""
        results = []
        
        for i, fusion_result in enumerate(fusion_results):
            entity_result = entity_results[i] if i < len(entity_results) else None
            agent_result = agent_results[i] if agent_results and i < len(agent_results) else None
            
            result = SpanDetectionResult(
                sentence=fusion_result.sentence,
                classification_score=fusion_result.classification_score,
                entity_verification_score=(
                    entity_result.entity_verification_score
                    if entity_result and hasattr(entity_result, 'entity_verification_score')
                    else 0.5
                ),
                agent_verification_score=(
                    agent_result.agent_verification_score
                    if agent_result and hasattr(agent_result, 'agent_verification_score')
                    else None
                ),
                final_hallucination_score=fusion_result.final_hallucination_score,
                label=fusion_result.label,
                confidence=fusion_result.confidence,
                sentence_index=fusion_result.sentence_index
            )
            
            results.append(result)
        
        return results
    
    def detect_batch(
        self,
        texts: List[str],
        return_json: bool = True
    ) -> List[List[Union[Dict, SpanDetectionResult]]]:
        """
        Detect hallucinations for multiple texts.
        
        Args:
            texts: List of input texts
            return_json: If True, return list of dicts
        
        Returns:
            List of lists of detection results
        """
        return [self.detect(text, return_json=return_json) for text in texts]
    
    def save_results(
        self,
        results: List[Dict],
        output_path: str
    ):
        """Save results to JSON file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    
    def get_summary(
        self,
        results: List[Dict]
    ) -> Dict:
        """
        Generate summary statistics from results.
        
        Args:
            results: List of detection result dicts
        
        Returns:
            Summary dict with statistics
        """
        total_sentences = len(results)
        hallucinated = sum(1 for r in results if r['label'] == 'hallucinated')
        factual = total_sentences - hallucinated
        
        avg_hallucination_score = (
            sum(r['final_hallucination_score'] for r in results) / total_sentences
            if total_sentences > 0 else 0.0
        )
        
        avg_confidence = (
            sum(r['confidence'] for r in results) / total_sentences
            if total_sentences > 0 else 0.0
        )
        
        return {
            "total_sentences": total_sentences,
            "hallucinated_sentences": hallucinated,
            "factual_sentences": factual,
            "hallucination_rate": hallucinated / total_sentences if total_sentences > 0 else 0.0,
            "average_hallucination_score": round(avg_hallucination_score, 4),
            "average_confidence": round(avg_confidence, 4)
        }


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("Span Inference Pipeline Test")
    print("=" * 70)
    
    pipeline = SpanInferencePipeline(
        use_entity_verification=True,
        use_agent_verification=False
    )
    
    test_text = """
    Large Language Models (LLMs) have achieved remarkable success in natural language processing.
    However, they frequently generate hallucinations that appear plausible but contain factual errors.
    This module detects hallucinations at the sentence level, providing fine-grained localization.
    The moon is made of cheese and orbits Mars every 24 hours.
    """
    
    print("\nDetecting hallucinations at sentence level...")
    results = pipeline.detect(test_text, return_json=True)
    
    print(f"\nDetected {len(results)} sentences:")
    for result in results:
        label_icon = "❌" if result['label'] == 'hallucinated' else "✅"
        print(f"\n{label_icon} [{result['sentence_index']}] {result['sentence'][:60]}...")
        print(f"   Final score: {result['final_hallucination_score']:.3f} ({result['label']})")
        print(f"   Confidence: {result['confidence']:.3f}")
    
    # Summary
    summary = pipeline.get_summary(results)
    print(f"\n{'='*70}")
    print("Summary:")
    print(f"  Total sentences: {summary['total_sentences']}")
    print(f"  Hallucinated: {summary['hallucinated_sentences']}")
    print(f"  Factual: {summary['factual_sentences']}")
    print(f"  Hallucination rate: {summary['hallucination_rate']:.2%}")
    print(f"  Average score: {summary['average_hallucination_score']:.3f}")
    print("=" * 70)

