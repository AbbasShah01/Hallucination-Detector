"""
Span-Level Entity Verifier Module

Performs entity extraction and verification for each sentence individually.
Reuses existing entity verification module at the sentence level.
"""

from typing import List, Dict, Optional
from dataclasses import dataclass
import sys
import os

# Add parent directories to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

try:
    from src.entity_verification import EntityExtractor, EntityVerifier
    ENTITY_VERIFICATION_AVAILABLE = True
except ImportError:
    ENTITY_VERIFICATION_AVAILABLE = False
    print("Warning: Entity verification module not available.")


@dataclass
class SpanEntityVerificationResult:
    """Result of sentence-level entity verification."""
    sentence: str
    sentence_index: int
    entity_verification_score: float  # Correctness score (0-1, higher = more correct)
    entities_found: int
    entities_verified: int
    entities_failed: int
    details: Optional[Dict] = None


class SpanEntityVerifier:
    """
    Verifies entities in individual sentences.
    
    This module applies entity extraction and verification to each sentence
    separately, providing fine-grained factual correctness scores.
    """
    
    def __init__(
        self,
        extractor_method: str = "transformers",
        use_wikipedia: bool = True
    ):
        """
        Initialize span entity verifier.
        
        Args:
            extractor_method: "spacy" or "transformers"
            use_wikipedia: Whether to verify against Wikipedia
        """
        self.extractor_method = extractor_method
        self.use_wikipedia = use_wikipedia
        
        self.extractor = None
        self.verifier = None
        
        if ENTITY_VERIFICATION_AVAILABLE:
            self._initialize_components()
    
    def _initialize_components(self):
        """Initialize entity extractor and verifier."""
        try:
            self.extractor = EntityExtractor(method=self.extractor_method)
            
            if self.use_wikipedia:
                from src.entity_verification import WikipediaVerifier, EntityVerifier
                wiki_verifier = WikipediaVerifier()
                self.verifier = EntityVerifier(wiki_verifier)
            else:
                # Use mock verifier (always returns verified)
                self.verifier = None
                
        except Exception as e:
            print(f"Warning: Could not initialize entity verification: {e}")
            self.extractor = None
            self.verifier = None
    
    def verify_sentence(
        self,
        sentence: str,
        context: Optional[str] = None
    ) -> SpanEntityVerificationResult:
        """
        Verify entities in a single sentence.
        
        Args:
            sentence: Sentence to verify
            context: Optional surrounding context (not used for entity verification)
        
        Returns:
            SpanEntityVerificationResult with verification score
        """
        if self.extractor is None:
            # Fallback: return neutral score
            return SpanEntityVerificationResult(
                sentence=sentence,
                sentence_index=0,
                entity_verification_score=0.5,
                entities_found=0,
                entities_verified=0,
                entities_failed=0
            )
        
        # Extract entities
        entities = self.extractor.extract_entities(sentence)
        entities_found = len(entities)
        
        if entities_found == 0:
            # No entities to verify - return high score (no factual claims)
            return SpanEntityVerificationResult(
                sentence=sentence,
                sentence_index=0,
                entity_verification_score=1.0,  # No entities = no errors
                entities_found=0,
                entities_verified=0,
                entities_failed=0,
                details={"reason": "no_entities"}
            )
        
        # Verify entities
        entities_verified = 0
        entities_failed = 0
        verification_details = []
        
        if self.verifier:
            for entity in entities:
                try:
                    result = self.verifier.verify_entity(entity.text)
                    if result.is_verified:
                        entities_verified += 1
                    else:
                        entities_failed += 1
                    
                    verification_details.append({
                        "entity": entity.text,
                        "type": entity.entity_type,
                        "verified": result.is_verified,
                        "confidence": result.confidence
                    })
                except Exception as e:
                    # If verification fails, count as unverified
                    entities_failed += 1
                    verification_details.append({
                        "entity": entity.text,
                        "type": entity.entity_type,
                        "verified": False,
                        "error": str(e)
                    })
        else:
            # No verifier available - assume all verified
            entities_verified = entities_found
            verification_details = [
                {
                    "entity": e.text,
                    "type": e.entity_type,
                    "verified": True,
                    "note": "no_verifier_available"
                }
                for e in entities
            ]
        
        # Calculate verification score
        # Score = proportion of verified entities
        if entities_found > 0:
            verification_score = entities_verified / entities_found
        else:
            verification_score = 1.0  # No entities = perfect score
        
        return SpanEntityVerificationResult(
            sentence=sentence,
            sentence_index=0,
            entity_verification_score=verification_score,
            entities_found=entities_found,
            entities_verified=entities_verified,
            entities_failed=entities_failed,
            details={
                "verification_details": verification_details,
                "total_entities": entities_found
            }
        )
    
    def verify_sentences(
        self,
        sentences: List[str],
        contexts: Optional[List[str]] = None
    ) -> List[SpanEntityVerificationResult]:
        """
        Verify entities in multiple sentences.
        
        Args:
            sentences: List of sentences to verify
            contexts: Optional list of context strings (not used)
        
        Returns:
            List of SpanEntityVerificationResult objects
        """
        results = []
        
        for i, sentence in enumerate(sentences):
            result = self.verify_sentence(sentence, None)  # Context not used
            result.sentence_index = i
            results.append(result)
        
        return results
    
    def verify_with_metadata(
        self,
        sentences: List[Dict]
    ) -> List[SpanEntityVerificationResult]:
        """
        Verify sentences with metadata (from sentence splitter).
        
        Args:
            sentences: List of dicts with 'sentence' and optional 'context'
        
        Returns:
            List of SpanEntityVerificationResult objects
        """
        sentence_texts = [s['sentence'] for s in sentences]
        results = self.verify_sentences(sentence_texts)
        
        # Update indices from metadata
        for i, (result, sent_meta) in enumerate(zip(results, sentences)):
            result.sentence_index = sent_meta.get('sentence_index', i)
        
        return results


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("Span Entity Verifier Test")
    print("=" * 70)
    
    verifier = SpanEntityVerifier(extractor_method="transformers", use_wikipedia=False)
    
    test_sentences = [
        "Barack Obama was the 44th President of the United States.",
        "The moon is made of cheese and orbits Mars.",
        "Python is a programming language used in data science."
    ]
    
    print("\nVerifying entities in sentences...")
    results = verifier.verify_sentences(test_sentences)
    
    for result in results:
        print(f"\n[{result.sentence_index}] {result.sentence[:50]}...")
        print(f"  Entities found: {result.entities_found}")
        print(f"  Verified: {result.entities_verified}, Failed: {result.entities_failed}")
        print(f"  Verification score: {result.entity_verification_score:.3f}")
    
    print("\n" + "=" * 70)

