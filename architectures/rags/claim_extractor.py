"""
Claim Extraction Module for RAGS Architecture
Extracts atomic claims from LLM responses for evidence-based verification.
"""

import re
from typing import List, Dict, Tuple
from dataclasses import dataclass
from collections import defaultdict

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


@dataclass
class Claim:
    """Represents an atomic claim extracted from text."""
    text: str
    start_pos: int
    end_pos: int
    entities: List[str]  # Entities mentioned in claim
    relations: List[Tuple[str, str, str]]  # (subject, relation, object)
    claim_type: str  # factual, numerical, temporal, causal, etc.
    confidence: float  # Extraction confidence


class ClaimExtractor:
    """
    Extracts atomic claims from LLM responses.
    Splits complex responses into verifiable atomic statements.
    """
    
    def __init__(self, method="rule_based"):
        """
        Initialize claim extractor.
        
        Args:
            method: "rule_based" or "ml_based"
        """
        self.method = method
        self.nlp = None
        
        if method == "ml_based" and SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except:
                self.method = "rule_based"
    
    def extract_claims(self, response: str) -> List[Claim]:
        """
        Extract atomic claims from response.
        
        Args:
            response: LLM response text
        
        Returns:
            List of Claim objects
        """
        if self.method == "rule_based":
            return self._extract_rule_based(response)
        else:
            return self._extract_ml_based(response)
    
    def _extract_rule_based(self, response: str) -> List[Claim]:
        """Extract claims using rule-based methods."""
        claims = []
        
        # Split by sentence
        sentences = self._split_sentences(response)
        
        for i, sentence in enumerate(sentences):
            # Find sentence boundaries
            start_pos = response.find(sentence)
            end_pos = start_pos + len(sentence)
            
            # Extract entities (simple pattern matching)
            entities = self._extract_entities_simple(sentence)
            
            # Extract relations (subject-verb-object patterns)
            relations = self._extract_relations_simple(sentence)
            
            # Determine claim type
            claim_type = self._classify_claim_type(sentence)
            
            claim = Claim(
                text=sentence.strip(),
                start_pos=start_pos,
                end_pos=end_pos,
                entities=entities,
                relations=relations,
                claim_type=claim_type,
                confidence=0.8  # Rule-based confidence
            )
            claims.append(claim)
        
        return claims
    
    def _extract_ml_based(self, response: str) -> List[Claim]:
        """Extract claims using ML-based methods."""
        if not self.nlp:
            return self._extract_rule_based(response)
        
        doc = self.nlp(response)
        claims = []
        
        # Use dependency parsing to identify claims
        for sent in doc.sents:
            entities = [ent.text for ent in sent.ents]
            relations = self._extract_relations_nlp(sent)
            
            claim = Claim(
                text=sent.text,
                start_pos=sent.start_char,
                end_pos=sent.end_char,
                entities=entities,
                relations=relations,
                claim_type=self._classify_claim_type(sent.text),
                confidence=0.9
            )
            claims.append(claim)
        
        return claims
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _extract_entities_simple(self, text: str) -> List[str]:
        """Simple entity extraction using patterns."""
        entities = []
        
        # Capitalized phrases (potential entities)
        capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        entities.extend(capitalized)
        
        # Numbers and dates
        numbers = re.findall(r'\d+', text)
        entities.extend(numbers)
        
        return list(set(entities))
    
    def _extract_relations_simple(self, text: str) -> List[Tuple[str, str, str]]:
        """Extract simple subject-verb-object relations."""
        relations = []
        
        # Pattern: "X is Y", "X was Y", "X did Y"
        patterns = [
            (r'(\w+)\s+(?:is|was|are|were)\s+(\w+)', 'is_a'),
            (r'(\w+)\s+(?:invented|discovered|created)\s+(\w+)', 'invented'),
            (r'(\w+)\s+(?:born|died)\s+in\s+(\w+)', 'location'),
        ]
        
        for pattern, rel_type in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                subject = match.group(1)
                obj = match.group(2)
                relations.append((subject, rel_type, obj))
        
        return relations
    
    def _extract_relations_nlp(self, sent) -> List[Tuple[str, str, str]]:
        """Extract relations using NLP dependency parsing."""
        relations = []
        
        # Find subject-verb-object triplets
        for token in sent:
            if token.dep_ == "nsubj":  # Subject
                subject = token.text
                verb = token.head.text
                
                # Find object
                for child in token.head.children:
                    if child.dep_ in ["dobj", "pobj"]:
                        obj = child.text
                        relations.append((subject, verb, obj))
        
        return relations
    
    def _classify_claim_type(self, text: str) -> str:
        """Classify the type of claim."""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['invented', 'discovered', 'created']):
            return 'factual'
        elif re.search(r'\d{4}|\d+', text):
            return 'numerical'
        elif any(word in text_lower for word in ['when', 'during', 'in', 'after', 'before']):
            return 'temporal'
        elif any(word in text_lower for word in ['because', 'due to', 'caused']):
            return 'causal'
        else:
            return 'factual'

