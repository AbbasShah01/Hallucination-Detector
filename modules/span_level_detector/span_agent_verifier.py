"""
Span-Level Agent Verifier Module

Performs agent-based (LLM) verification for each sentence individually.
Reuses existing agentic verification module at the sentence level.
"""

from typing import List, Dict, Optional
from dataclasses import dataclass
import sys
import os

# Add parent directories to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

try:
    from src.agentic_verification import AgenticVerifier
    AGENTIC_VERIFICATION_AVAILABLE = True
except ImportError:
    AGENTIC_VERIFICATION_AVAILABLE = False
    print("Warning: Agentic verification module not available.")


@dataclass
class SpanAgentVerificationResult:
    """Result of sentence-level agent verification."""
    sentence: str
    sentence_index: int
    agent_verification_score: float  # Correctness score (0-1, higher = more correct)
    verification_reasoning: Optional[str] = None
    confidence: float = 0.0


class SpanAgentVerifier:
    """
    Verifies sentences using agent-based (LLM) verification.
    
    This module applies agentic verification to each sentence separately,
    providing LLM-based fact-checking at the sentence level.
    """
    
    def __init__(
        self,
        method: str = "local",
        provider: Optional[str] = None,
        model_name: Optional[str] = None
    ):
        """
        Initialize span agent verifier.
        
        Args:
            method: "local" or "api"
            provider: "openai" or "anthropic" (if using API)
            model_name: Model name for local LLM
        """
        self.method = method
        self.provider = provider
        self.model_name = model_name
        
        self.verifier = None
        
        if AGENTIC_VERIFICATION_AVAILABLE:
            self._initialize_verifier()
    
    def _initialize_verifier(self):
        """Initialize agentic verifier."""
        try:
            self.verifier = AgenticVerifier(
                method=self.method,
                provider=self.provider,
                model_name=self.model_name
            )
        except Exception as e:
            print(f"Warning: Could not initialize agentic verifier: {e}")
            self.verifier = None
    
    def verify_sentence(
        self,
        sentence: str,
        context: Optional[str] = None
    ) -> SpanAgentVerificationResult:
        """
        Verify a single sentence using agent-based verification.
        
        Args:
            sentence: Sentence to verify
            context: Optional surrounding context for better verification
        
        Returns:
            SpanAgentVerificationResult with verification score
        """
        if self.verifier is None:
            # Fallback: return neutral score
            return SpanAgentVerificationResult(
                sentence=sentence,
                sentence_index=0,
                agent_verification_score=0.5,
                confidence=0.0
            )
        
        # Prepare prompt with context if available
        if context:
            prompt = f"Context: {context}\n\nSentence to verify: {sentence}"
        else:
            prompt = sentence
        
        try:
            # Verify using agentic verifier
            result = self.verifier.verify(prompt)
            
            return SpanAgentVerificationResult(
                sentence=sentence,
                sentence_index=0,
                agent_verification_score=result.verification_score,
                verification_reasoning=result.reasoning,
                confidence=result.confidence
            )
        
        except Exception as e:
            # If verification fails, return neutral score
            print(f"Warning: Agentic verification failed: {e}")
            return SpanAgentVerificationResult(
                sentence=sentence,
                sentence_index=0,
                agent_verification_score=0.5,
                confidence=0.0,
                verification_reasoning=f"Verification error: {str(e)}"
            )
    
    def verify_sentences(
        self,
        sentences: List[str],
        contexts: Optional[List[str]] = None
    ) -> List[SpanAgentVerificationResult]:
        """
        Verify multiple sentences.
        
        Args:
            sentences: List of sentences to verify
            contexts: Optional list of context strings
        
        Returns:
            List of SpanAgentVerificationResult objects
        """
        results = []
        
        for i, sentence in enumerate(sentences):
            context = contexts[i] if contexts and i < len(contexts) else None
            result = self.verify_sentence(sentence, context)
            result.sentence_index = i
            results.append(result)
        
        return results
    
    def verify_with_metadata(
        self,
        sentences: List[Dict]
    ) -> List[SpanAgentVerificationResult]:
        """
        Verify sentences with metadata (from sentence splitter).
        
        Args:
            sentences: List of dicts with 'sentence' and optional 'context'
        
        Returns:
            List of SpanAgentVerificationResult objects
        """
        sentence_texts = [s['sentence'] for s in sentences]
        contexts = [s.get('context') for s in sentences]
        
        results = self.verify_sentences(sentence_texts, contexts)
        
        # Update indices from metadata
        for i, (result, sent_meta) in enumerate(zip(results, sentences)):
            result.sentence_index = sent_meta.get('sentence_index', i)
        
        return results


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("Span Agent Verifier Test")
    print("=" * 70)
    
    verifier = SpanAgentVerifier(method="local")
    
    test_sentences = [
        "Barack Obama was the 44th President of the United States.",
        "The moon is made of cheese.",
        "Python is a programming language."
    ]
    
    print("\nVerifying sentences with agent-based verification...")
    print("(Note: This requires LLM setup. Using fallback if unavailable.)")
    
    results = verifier.verify_sentences(test_sentences)
    
    for result in results:
        print(f"\n[{result.sentence_index}] {result.sentence[:50]}...")
        print(f"  Agent verification score: {result.agent_verification_score:.3f}")
        print(f"  Confidence: {result.confidence:.3f}")
        if result.verification_reasoning:
            print(f"  Reasoning: {result.verification_reasoning[:100]}...")
    
    print("\n" + "=" * 70)

