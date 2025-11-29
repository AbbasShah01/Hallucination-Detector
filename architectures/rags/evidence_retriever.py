"""
Evidence Retrieval Module for RAGS Architecture
Retrieves relevant evidence documents for claims.
"""

import requests
from typing import List, Dict, Optional
from dataclasses import dataclass
import time

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


@dataclass
class Evidence:
    """Represents retrieved evidence for a claim."""
    text: str
    source: str  # Wikipedia, academic paper, etc.
    relevance_score: float  # How relevant to the claim
    credibility_score: float  # Source credibility
    url: Optional[str] = None


class EvidenceRetriever:
    """
    Retrieves evidence documents for claims.
    Supports multiple retrieval methods: vector search, Wikipedia, academic papers.
    """
    
    def __init__(self, method="wikipedia", use_vector_search=True):
        """
        Initialize evidence retriever.
        
        Args:
            method: "wikipedia", "academic", or "hybrid"
            use_vector_search: Whether to use vector similarity search
        """
        self.method = method
        self.use_vector_search = use_vector_search
        self.vector_model = None
        self.evidence_cache = {}
        
        if use_vector_search and SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.vector_model = SentenceTransformer('all-MiniLM-L6-v2')
            except:
                self.use_vector_search = False
    
    def retrieve_evidence(self, claim: str, top_k: int = 5) -> List[Evidence]:
        """
        Retrieve evidence for a claim.
        
        Args:
            claim: Claim text to find evidence for
            top_k: Number of evidence documents to retrieve
        
        Returns:
            List of Evidence objects
        """
        # Check cache
        if claim in self.evidence_cache:
            return self.evidence_cache[claim][:top_k]
        
        all_evidence = []
        
        # Retrieve from different sources
        if self.method in ["wikipedia", "hybrid"]:
            wiki_evidence = self._retrieve_wikipedia(claim, top_k)
            all_evidence.extend(wiki_evidence)
        
        if self.method in ["academic", "hybrid"]:
            academic_evidence = self._retrieve_academic(claim, top_k)
            all_evidence.extend(academic_evidence)
        
        # Rank by relevance
        ranked_evidence = self._rank_evidence(claim, all_evidence)
        
        # Cache results
        self.evidence_cache[claim] = ranked_evidence
        
        return ranked_evidence[:top_k]
    
    def _retrieve_wikipedia(self, claim: str, top_k: int) -> List[Evidence]:
        """Retrieve evidence from Wikipedia."""
        evidence = []
        
        # Extract key terms from claim
        key_terms = self._extract_key_terms(claim)
        
        for term in key_terms[:3]:  # Top 3 terms
            try:
                # Wikipedia API search
                search_url = "https://en.wikipedia.org/api/rest_v1/page/search/"
                params = {"q": term, "limit": 2}
                
                time.sleep(0.1)  # Rate limiting
                response = requests.get(search_url, params=params, timeout=5)
                
                if response.status_code == 200:
                    results = response.json()
                    for page in results.get("pages", [])[:2]:
                        # Get page summary
                        page_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{page['title']}"
                        page_response = requests.get(page_url, timeout=5)
                        
                        if page_response.status_code == 200:
                            page_data = page_response.json()
                            evidence.append(Evidence(
                                text=page_data.get("extract", ""),
                                source="Wikipedia",
                                relevance_score=0.7,  # Will be refined
                                credibility_score=0.8,
                                url=page_data.get("content_urls", {}).get("desktop", {}).get("page", "")
                            ))
            except Exception as e:
                print(f"Error retrieving Wikipedia evidence: {e}")
        
        return evidence
    
    def _retrieve_academic(self, claim: str, top_k: int) -> List[Evidence]:
        """Retrieve evidence from academic sources (placeholder)."""
        # In production, integrate with Semantic Scholar, arXiv, etc.
        evidence = []
        
        # Placeholder: would query academic databases
        # For now, return empty list
        return evidence
    
    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract key terms from claim."""
        # Simple extraction: capitalized words, important nouns
        import re
        
        # Capitalized phrases
        terms = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        
        # Remove common words
        stopwords = {'The', 'A', 'An', 'This', 'That', 'It', 'Is', 'Was', 'Are', 'Were'}
        terms = [t for t in terms if t not in stopwords]
        
        return terms[:5]  # Top 5 terms
    
    def _rank_evidence(self, claim: str, evidence: List[Evidence]) -> List[Evidence]:
        """Rank evidence by relevance to claim."""
        if not self.use_vector_search or not self.vector_model:
            # Simple ranking by source credibility
            return sorted(evidence, key=lambda x: x.credibility_score, reverse=True)
        
        # Compute semantic similarity
        claim_embedding = self.vector_model.encode([claim])
        
        for ev in evidence:
            ev_embedding = self.vector_model.encode([ev.text])
            similarity = cosine_similarity(claim_embedding, ev_embedding)[0][0]
            ev.relevance_score = float(similarity)
        
        # Sort by relevance
        return sorted(evidence, key=lambda x: x.relevance_score, reverse=True)

