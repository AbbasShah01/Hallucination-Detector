"""
Span-Level Classifier Module

Applies transformer-based classification to each sentence individually.
Reuses existing transformer model for per-sentence hallucination detection.
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


@dataclass
class SpanClassificationResult:
    """Result of sentence-level classification."""
    sentence: str
    sentence_index: int
    classification_score: float  # Hallucination probability (0-1)
    logits: Optional[torch.Tensor] = None
    confidence: float = 0.0


class SpanClassifier:
    """
    Classifies individual sentences using a transformer model.
    
    This module applies the existing transformer classifier to each sentence
    separately, enabling fine-grained hallucination detection.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        model_name: str = "distilbert-base-uncased",
        device: Optional[str] = None
    ):
        """
        Initialize span classifier.
        
        Args:
            model_path: Path to fine-tuned model (if available)
            model_name: HuggingFace model name
            device: "cuda" or "cpu" (auto-detected if None)
        """
        self.model_path = model_path
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.tokenizer = None
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load tokenizer and model."""
        if not TRANSFORMERS_AVAILABLE:
            print("Warning: transformers not available. SpanClassifier will not work.")
            return
        
        try:
            if self.model_path:
                # Load fine-tuned model
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_path
                )
            else:
                # Load base model (will need fine-tuning)
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_name,
                    num_labels=2  # Binary classification
                )
            
            self.model.to(self.device)
            self.model.eval()
            
        except Exception as e:
            print(f"Warning: Could not load model: {e}")
            print("SpanClassifier will use fallback scoring.")
            self.model = None
            self.tokenizer = None
    
    def classify_sentence(
        self, 
        sentence: str, 
        context: Optional[str] = None
    ) -> SpanClassificationResult:
        """
        Classify a single sentence.
        
        Args:
            sentence: Sentence to classify
            context: Optional surrounding context
        
        Returns:
            SpanClassificationResult with classification score
        """
        if self.model is None or self.tokenizer is None:
            # Fallback: return neutral score
            return SpanClassificationResult(
                sentence=sentence,
                sentence_index=0,
                classification_score=0.5,
                confidence=0.0
            )
        
        # Prepare input (optionally include context)
        if context:
            # Use context + sentence for better classification
            input_text = f"{context} {sentence}"
        else:
            input_text = sentence
        
        # Tokenize
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.device)
        
        # Classify
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            
            # Convert to probabilities
            probs = torch.softmax(logits, dim=-1)
            
            # Get hallucination probability (assuming label 1 = hallucination)
            hallucination_prob = probs[0][1].item()
            confidence = torch.max(probs[0]).item()
        
        return SpanClassificationResult(
            sentence=sentence,
            sentence_index=0,
            classification_score=hallucination_prob,
            logits=logits,
            confidence=confidence
        )
    
    def classify_sentences(
        self, 
        sentences: List[str],
        contexts: Optional[List[str]] = None,
        batch_size: int = 8
    ) -> List[SpanClassificationResult]:
        """
        Classify multiple sentences (batch processing).
        
        Args:
            sentences: List of sentences to classify
            contexts: Optional list of context strings
            batch_size: Batch size for processing
        
        Returns:
            List of SpanClassificationResult objects
        """
        if self.model is None or self.tokenizer is None:
            # Fallback: return neutral scores
            return [
                SpanClassificationResult(
                    sentence=sent,
                    sentence_index=i,
                    classification_score=0.5,
                    confidence=0.0
                )
                for i, sent in enumerate(sentences)
            ]
        
        results = []
        
        # Process in batches
        for i in range(0, len(sentences), batch_size):
            batch_sentences = sentences[i:i + batch_size]
            batch_contexts = contexts[i:i + batch_size] if contexts else [None] * len(batch_sentences)
            
            # Prepare inputs
            input_texts = []
            for sent, ctx in zip(batch_sentences, batch_contexts):
                if ctx:
                    input_texts.append(f"{ctx} {sent}")
                else:
                    input_texts.append(sent)
            
            # Tokenize batch
            inputs = self.tokenizer(
                input_texts,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)
            
            # Classify batch
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                
                # Process each result in batch
                for j, (sent, prob) in enumerate(zip(batch_sentences, probs)):
                    hallucination_prob = prob[1].item()
                    confidence = torch.max(prob).item()
                    
                    results.append(SpanClassificationResult(
                        sentence=sent,
                        sentence_index=i + j,
                        classification_score=hallucination_prob,
                        logits=logits[j],
                        confidence=confidence
                    ))
        
        return results
    
    def classify_with_metadata(
        self,
        sentences: List[Dict],
        batch_size: int = 8
    ) -> List[SpanClassificationResult]:
        """
        Classify sentences with metadata (from sentence splitter).
        
        Args:
            sentences: List of dicts with 'sentence' and optional 'context'
            batch_size: Batch size for processing
        
        Returns:
            List of SpanClassificationResult objects
        """
        sentence_texts = [s['sentence'] for s in sentences]
        contexts = [s.get('context') for s in sentences]
        
        results = self.classify_sentences(sentence_texts, contexts, batch_size)
        
        # Update indices from metadata
        for i, (result, sent_meta) in enumerate(zip(results, sentences)):
            result.sentence_index = sent_meta.get('sentence_index', i)
        
        return results


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("Span Classifier Test")
    print("=" * 70)
    
    classifier = SpanClassifier()
    
    test_sentences = [
        "Barack Obama was the 44th President of the United States.",
        "The moon is made of cheese.",
        "Python is a programming language."
    ]
    
    print("\nClassifying sentences...")
    results = classifier.classify_sentences(test_sentences)
    
    for result in results:
        label = "HALLUCINATION" if result.classification_score > 0.5 else "FACTUAL"
        print(f"\n[{result.sentence_index}] {result.sentence[:50]}...")
        print(f"  Score: {result.classification_score:.3f} ({label})")
        print(f"  Confidence: {result.confidence:.3f}")
    
    print("\n" + "=" * 70)

