"""
Sentence Splitter Module

Splits text into sentences using NLTK or spaCy.
Provides robust sentence boundary detection with fallback mechanisms.
"""

import re
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class Sentence:
    """Represents a single sentence with metadata."""
    text: str
    start_char: int
    end_char: int
    index: int
    is_valid: bool = True


class SentenceSplitter:
    """
    Splits text into sentences using NLTK or spaCy with fallback to regex.
    
    Supports multiple backends:
    - NLTK (preferred, most accurate)
    - spaCy (fast, good accuracy)
    - Regex fallback (basic, always available)
    """
    
    def __init__(self, method: str = "nltk"):
        """
        Initialize sentence splitter.
        
        Args:
            method: "nltk", "spacy", or "regex"
        """
        self.method = method
        self._nltk_tokenizer = None
        self._spacy_model = None
        self._initialize_backend()
    
    def _initialize_backend(self):
        """Initialize the selected backend."""
        if self.method == "nltk":
            try:
                import nltk
                try:
                    nltk.data.find('tokenizers/punkt')
                except LookupError:
                    nltk.download('punkt', quiet=True)
                from nltk.tokenize import sent_tokenize
                self._nltk_tokenizer = sent_tokenize
            except ImportError:
                print("Warning: NLTK not available. Falling back to regex.")
                self.method = "regex"
        
        elif self.method == "spacy":
            try:
                import spacy
                try:
                    self._spacy_model = spacy.load("en_core_web_sm")
                except OSError:
                    print("Warning: spaCy model not found. Install with: python -m spacy download en_core_web_sm")
                    print("Falling back to regex.")
                    self.method = "regex"
            except ImportError:
                print("Warning: spaCy not available. Falling back to regex.")
                self.method = "regex"
    
    def split(self, text: str) -> List[Sentence]:
        """
        Split text into sentences.
        
        Args:
            text: Input text to split
        
        Returns:
            List of Sentence objects with metadata
        """
        if not text or not text.strip():
            return []
        
        sentences = []
        
        if self.method == "nltk" and self._nltk_tokenizer:
            sentence_texts = self._nltk_tokenizer(text)
            sentences = self._create_sentences_from_texts(text, sentence_texts)
        
        elif self.method == "spacy" and self._spacy_model:
            doc = self._spacy_model(text)
            sentence_texts = [sent.text.strip() for sent in doc.sents]
            sentences = self._create_sentences_from_texts(text, sentence_texts)
        
        else:
            # Regex fallback
            sentences = self._regex_split(text)
        
        # Filter out empty sentences
        sentences = [s for s in sentences if s.text.strip()]
        
        return sentences
    
    def _create_sentences_from_texts(
        self, 
        original_text: str, 
        sentence_texts: List[str]
    ) -> List[Sentence]:
        """Create Sentence objects from sentence texts."""
        sentences = []
        current_pos = 0
        
        for idx, sent_text in enumerate(sentence_texts):
            # Find position in original text
            start_pos = original_text.find(sent_text, current_pos)
            if start_pos == -1:
                # Fallback: approximate position
                start_pos = current_pos
            
            end_pos = start_pos + len(sent_text)
            current_pos = end_pos
            
            sentences.append(Sentence(
                text=sent_text.strip(),
                start_char=start_pos,
                end_char=end_pos,
                index=idx
            ))
        
        return sentences
    
    def _regex_split(self, text: str) -> List[Sentence]:
        """
        Fallback regex-based sentence splitting.
        
        Splits on sentence-ending punctuation followed by whitespace or end of string.
        """
        # Pattern: sentence ending (. ! ?) followed by space or end
        pattern = r'([.!?]+)\s+'
        
        # Split while keeping delimiters
        parts = re.split(pattern, text)
        
        sentences = []
        current_pos = 0
        
        i = 0
        while i < len(parts):
            if i + 1 < len(parts):
                # Combine sentence with its punctuation
                sent_text = parts[i] + parts[i + 1]
                i += 2
            else:
                sent_text = parts[i]
                i += 1
            
            sent_text = sent_text.strip()
            if sent_text:
                start_pos = current_pos
                end_pos = start_pos + len(sent_text)
                current_pos = end_pos
                
                sentences.append(Sentence(
                    text=sent_text,
                    start_char=start_pos,
                    end_char=end_pos,
                    index=len(sentences)
                ))
        
        return sentences
    
    def split_with_context(
        self, 
        text: str, 
        context_window: int = 1
    ) -> List[Dict]:
        """
        Split text into sentences with surrounding context.
        
        Args:
            text: Input text
            context_window: Number of sentences before/after to include as context
        
        Returns:
            List of dicts with sentence and context
        """
        sentences = self.split(text)
        
        result = []
        for i, sent in enumerate(sentences):
            # Get context sentences
            start_idx = max(0, i - context_window)
            end_idx = min(len(sentences), i + context_window + 1)
            
            context_sentences = sentences[start_idx:end_idx]
            context_text = " ".join([s.text for s in context_sentences])
            
            result.append({
                "sentence": sent.text,
                "sentence_index": sent.index,
                "context": context_text,
                "context_start": start_idx,
                "context_end": end_idx
            })
        
        return result


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("Sentence Splitter Test")
    print("=" * 70)
    
    test_text = """
    Large Language Models (LLMs) have achieved remarkable success. 
    However, they frequently generate hallucinations. 
    This module detects hallucinations at the sentence level.
    """
    
    splitter = SentenceSplitter(method="nltk")
    sentences = splitter.split(test_text)
    
    print(f"\nSplit {len(sentences)} sentences:")
    for sent in sentences:
        print(f"  [{sent.index}] {sent.text[:60]}...")
    
    print("\n" + "=" * 70)

