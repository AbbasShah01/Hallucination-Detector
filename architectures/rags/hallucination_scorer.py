"""
Hallucination Scorer for RAGS Architecture
Scores hallucination probability based on evidence support for claims.
"""

from typing import List, Dict
from dataclasses import dataclass
import numpy as np

from .claim_extractor import ClaimExtractor, Claim
from .evidence_retriever import EvidenceRetriever, Evidence


@dataclass
class RAGSResult:
    """Result of RAGS hallucination scoring."""
    response: str
    claims: List[Claim]
    evidence_per_claim: Dict[int, List[Evidence]]
    claim_scores: Dict[int, float]  # Hallucination score per claim
    overall_score: float  # Overall hallucination probability
    confidence: float  # Confidence in the score


class RAGSHallucinationScorer:
    """
    Main scorer for RAGS architecture.
    Extracts claims, retrieves evidence, and scores hallucination probability.
    """
    
    def __init__(self, evidence_threshold: float = 0.5):
        """
        Initialize RAGS scorer.
        
        Args:
            evidence_threshold: Minimum evidence relevance to consider
        """
        self.claim_extractor = ClaimExtractor()
        self.evidence_retriever = EvidenceRetriever()
        self.evidence_threshold = evidence_threshold
    
    def score(self, response: str) -> RAGSResult:
        """
        Score hallucination probability for a response.
        
        Args:
            response: LLM response to score
        
        Returns:
            RAGSResult with detailed scoring information
        """
        # Step 1: Extract claims
        claims = self.claim_extractor.extract_claims(response)
        
        # Step 2: Retrieve evidence for each claim
        evidence_per_claim = {}
        for i, claim in enumerate(claims):
            evidence = self.evidence_retriever.retrieve_evidence(claim.text, top_k=3)
            evidence_per_claim[i] = evidence
        
        # Step 3: Score each claim based on evidence
        claim_scores = {}
        for i, claim in enumerate(claims):
            evidence = evidence_per_claim[i]
            score = self._score_claim(claim, evidence)
            claim_scores[i] = score
        
        # Step 4: Aggregate claim scores into overall score
        overall_score = self._aggregate_scores(claim_scores)
        
        # Step 5: Compute confidence
        confidence = self._compute_confidence(claim_scores, evidence_per_claim)
        
        return RAGSResult(
            response=response,
            claims=claims,
            evidence_per_claim=evidence_per_claim,
            claim_scores=claim_scores,
            overall_score=overall_score,
            confidence=confidence
        )
    
    def _score_claim(self, claim: Claim, evidence: List[Evidence]) -> float:
        """
        Score a single claim based on evidence.
        
        Higher score = more likely to be hallucination (less evidence support)
        """
        if not evidence:
            # No evidence found - high hallucination probability
            return 0.8
        
        # Check if any evidence supports the claim
        supporting_evidence = [
            ev for ev in evidence 
            if ev.relevance_score >= self.evidence_threshold
        ]
        
        if not supporting_evidence:
            # No supporting evidence - likely hallucination
            return 0.7
        
        # Compute support score (inverse of hallucination probability)
        max_relevance = max(ev.relevance_score for ev in supporting_evidence)
        avg_credibility = np.mean([ev.credibility_score for ev in supporting_evidence])
        
        # Support score combines relevance and credibility
        support_score = (max_relevance * 0.6 + avg_credibility * 0.4)
        
        # Convert to hallucination probability (inverse)
        hallucination_prob = 1.0 - support_score
        
        # Clamp to reasonable range
        return max(0.1, min(0.9, hallucination_prob))
    
    def _aggregate_scores(self, claim_scores: Dict[int, float]) -> float:
        """Aggregate individual claim scores into overall score."""
        if not claim_scores:
            return 0.5  # Neutral if no claims
        
        scores = list(claim_scores.values())
        
        # Weighted average (could weight by claim confidence)
        overall = np.mean(scores)
        
        return float(overall)
    
    def _compute_confidence(self, claim_scores: Dict[int, float], 
                           evidence_per_claim: Dict[int, List[Evidence]]) -> float:
        """Compute confidence in the overall score."""
        if not claim_scores:
            return 0.5
        
        # Confidence based on:
        # 1. Number of claims (more claims = more confident)
        # 2. Quality of evidence (better evidence = more confident)
        # 3. Agreement between claims (consistent scores = more confident)
        
        num_claims = len(claim_scores)
        num_claims_factor = min(1.0, num_claims / 5.0)  # Normalize to 5 claims
        
        # Evidence quality
        evidence_qualities = []
        for i, evidence_list in evidence_per_claim.items():
            if evidence_list:
                avg_quality = np.mean([
                    ev.relevance_score * ev.credibility_score 
                    for ev in evidence_list
                ])
                evidence_qualities.append(avg_quality)
        
        evidence_factor = np.mean(evidence_qualities) if evidence_qualities else 0.5
        
        # Score consistency (variance)
        scores = list(claim_scores.values())
        variance = np.var(scores)
        consistency_factor = 1.0 - min(1.0, variance)  # Lower variance = higher consistency
        
        # Combine factors
        confidence = (num_claims_factor * 0.3 + 
                     evidence_factor * 0.4 + 
                     consistency_factor * 0.3)
        
        return float(confidence)

