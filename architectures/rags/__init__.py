"""
Retrieval-Augmented Hallucination Scoring (RAGS)

This architecture retrieves evidence for each claim in a response and scores
hallucination probability based on evidence support.
"""

from .hallucination_scorer import RAGSHallucinationScorer
from .claim_extractor import ClaimExtractor
from .evidence_retriever import EvidenceRetriever

__all__ = [
    'RAGSHallucinationScorer',
    'ClaimExtractor',
    'EvidenceRetriever'
]

