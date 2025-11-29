"""
Sentence-Level (Span-Level) Hallucination Detector Module

This module provides fine-grained hallucination detection at the sentence level,
enabling precise localization of hallucinated content within LLM responses.

Key Features:
- Sentence-level splitting using NLTK/spaCy
- Per-sentence transformer classification
- Per-sentence entity verification
- Per-sentence agent-based verification
- Sentence-level fusion and scoring
- Detailed JSON output with per-sentence labels
"""

from .sentence_splitter import SentenceSplitter
from .span_classifier import SpanClassifier
from .span_entity_verifier import SpanEntityVerifier
from .span_agent_verifier import SpanAgentVerifier
from .span_fusion import SpanFusion
from .span_inference_pipeline import SpanInferencePipeline

__all__ = [
    'SentenceSplitter',
    'SpanClassifier',
    'SpanEntityVerifier',
    'SpanAgentVerifier',
    'SpanFusion',
    'SpanInferencePipeline',
]

