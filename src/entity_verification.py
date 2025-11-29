"""
Entity Verification Module for Hybrid Hallucination Detection
Extracts entities from LLM responses using NER and verifies them against
Wikipedia or knowledge graphs to calculate factual correctness scores.
"""

import re
import requests
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import time
from collections import Counter

try:
    import spacy
    from spacy import displacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    print("Warning: spaCy not available. Install with: pip install spacy")

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


@dataclass
class Entity:
    """Represents an extracted entity."""
    text: str
    label: str  # Entity type (PERSON, ORG, LOC, etc.)
    start: int  # Start position in text
    end: int    # End position in text
    verified: Optional[bool] = None  # Whether entity was verified
    confidence: Optional[float] = None  # Verification confidence


@dataclass
class VerificationResult:
    """Result of entity verification for a response."""
    response: str
    entities: List[Entity]
    verified_count: int
    total_count: int
    correctness_score: float  # 0-1 score
    verified_entities: List[str]  # List of verified entity texts
    unverified_entities: List[str]  # List of unverified entity texts


class EntityExtractor:
    """
    Extracts named entities from text using NER models.
    Supports both spaCy and HuggingFace transformers.
    """
    
    def __init__(self, method="spacy", model_name=None):
        """
        Initialize entity extractor.
        
        Args:
            method: "spacy" or "transformers"
            model_name: Model name (e.g., "en_core_web_sm" for spaCy,
                       "dslim/bert-base-NER" for transformers)
        """
        self.method = method
        self.model_name = model_name
        self.extractor = None
        self._initialize_extractor()
    
    def _initialize_extractor(self):
        """Initialize the NER model."""
        if self.method == "spacy":
            if not SPACY_AVAILABLE:
                raise ImportError(
                    "spaCy not available. Install with: pip install spacy\n"
                    "Then download model: python -m spacy download en_core_web_sm"
                )
            
            # Default to small English model if not specified
            model = self.model_name or "en_core_web_sm"
            try:
                self.extractor = spacy.load(model)
            except OSError:
                raise OSError(
                    f"spaCy model '{model}' not found. "
                    f"Download with: python -m spacy download {model}"
                )
        
        elif self.method == "transformers":
            if not TRANSFORMERS_AVAILABLE:
                raise ImportError(
                    "transformers not available. Install with: pip install transformers"
                )
            
            # Default to BERT NER model if not specified
            model = self.model_name or "dslim/bert-base-NER"
            self.extractor = pipeline(
                "ner",
                model=model,
                aggregation_strategy="simple"
            )
        else:
            raise ValueError(f"Unknown method: {self.method}. Use 'spacy' or 'transformers'")
    
    def extract_entities(self, text: str) -> List[Entity]:
        """
        Extract named entities from text.
        
        Args:
            text: Input text to extract entities from
        
        Returns:
            List of Entity objects
        """
        if not text or not text.strip():
            return []
        
        entities = []
        
        if self.method == "spacy":
            doc = self.extractor(text)
            for ent in doc.ents:
                # Filter for relevant entity types
                if ent.label_ in ["PERSON", "ORG", "GPE", "LOC", "EVENT", "WORK_OF_ART", "DATE"]:
                    entities.append(Entity(
                        text=ent.text,
                        label=ent.label_,
                        start=ent.start_char,
                        end=ent.end_char
                    ))
        
        elif self.method == "transformers":
            results = self.extractor(text)
            for result in results:
                entity_text = result['word']
                label = result['entity_group']
                start = result.get('start', 0)
                end = result.get('end', len(entity_text))
                
                # Filter for relevant entity types
                if label in ["PER", "ORG", "LOC", "MISC"]:
                    entities.append(Entity(
                        text=entity_text,
                        label=label,
                        start=start,
                        end=end
                    ))
        
        # Remove duplicates (same text, same position)
        unique_entities = []
        seen = set()
        for entity in entities:
            key = (entity.text.lower(), entity.start, entity.end)
            if key not in seen:
                seen.add(key)
                unique_entities.append(entity)
        
        return unique_entities


class WikipediaVerifier:
    """
    Verifies entities against Wikipedia using the Wikipedia API.
    """
    
    def __init__(self, rate_limit_delay=0.1):
        """
        Initialize Wikipedia verifier.
        
        Args:
            rate_limit_delay: Delay between API requests (seconds)
        """
        self.rate_limit_delay = rate_limit_delay
        self.base_url = "https://en.wikipedia.org/api/rest_v1/page/summary/"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'HallucinationDetection/1.0 (Educational Research)'
        })
    
    def verify_entity(self, entity_text: str) -> Tuple[bool, float]:
        """
        Verify if an entity exists in Wikipedia.
        
        Args:
            entity_text: Entity text to verify
        
        Returns:
            Tuple of (is_verified, confidence)
            - is_verified: True if entity found in Wikipedia
            - confidence: Confidence score (0-1)
        """
        # Clean entity text
        entity_text = entity_text.strip()
        if not entity_text:
            return False, 0.0
        
        # Try direct lookup
        entity_encoded = entity_text.replace(" ", "_")
        url = self.base_url + entity_encoded
        
        try:
            time.sleep(self.rate_limit_delay)  # Rate limiting
            response = self.session.get(url, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                # Check if it's a disambiguation page (lower confidence)
                if "disambiguation" in data.get("type", "").lower():
                    return True, 0.7  # Lower confidence for disambiguation
                return True, 0.9  # High confidence for direct match
            
            # Try search if direct lookup fails
            return self._search_entity(entity_text)
        
        except requests.exceptions.RequestException:
            # Network error or timeout
            return False, 0.0
    
    def _search_entity(self, entity_text: str) -> Tuple[bool, float]:
        """
        Search for entity using Wikipedia search API.
        
        Args:
            entity_text: Entity text to search
        
        Returns:
            Tuple of (is_verified, confidence)
        """
        search_url = "https://en.wikipedia.org/api/rest_v1/page/search/"
        params = {"q": entity_text, "limit": 1}
        
        try:
            time.sleep(self.rate_limit_delay)
            response = self.session.get(search_url, params=params, timeout=5)
            
            if response.status_code == 200:
                results = response.json()
                if results.get("pages"):
                    # Check if first result is a good match
                    first_result = results["pages"][0]
                    title = first_result.get("title", "").lower()
                    entity_lower = entity_text.lower()
                    
                    # Exact or close match
                    if entity_lower in title or title in entity_lower:
                        return True, 0.8
                    # Partial match
                    elif any(word in title for word in entity_lower.split() if len(word) > 3):
                        return True, 0.6
            
            return False, 0.0
        
        except requests.exceptions.RequestException:
            return False, 0.0
    
    def verify_entities_batch(self, entities: List[str]) -> Dict[str, Tuple[bool, float]]:
        """
        Verify multiple entities (with rate limiting).
        
        Args:
            entities: List of entity texts
        
        Returns:
            Dictionary mapping entity text to (is_verified, confidence)
        """
        results = {}
        for entity in entities:
            results[entity] = self.verify_entity(entity)
        return results


class EntityVerifier:
    """
    Main class for entity extraction and verification.
    Combines NER extraction with Wikipedia verification.
    """
    
    def __init__(
        self,
        extractor_method="spacy",
        extractor_model=None,
        use_wikipedia=True,
        rate_limit_delay=0.1
    ):
        """
        Initialize entity verifier.
        
        Args:
            extractor_method: "spacy" or "transformers"
            extractor_model: Model name for extractor
            use_wikipedia: Whether to use Wikipedia for verification
            rate_limit_delay: Delay between Wikipedia API requests
        """
        self.extractor = EntityExtractor(extractor_method, extractor_model)
        self.use_wikipedia = use_wikipedia
        self.verifier = WikipediaVerifier(rate_limit_delay) if use_wikipedia else None
    
    def verify_response(
        self,
        response: str,
        min_confidence: float = 0.5,
        require_verification: bool = True
    ) -> VerificationResult:
        """
        Extract entities from response and verify them.
        
        Args:
            response: LLM response text to verify
            min_confidence: Minimum confidence threshold for verification
            require_verification: If True, only count verified entities
        
        Returns:
            VerificationResult with correctness score
        """
        # Extract entities
        entities = self.extractor.extract_entities(response)
        
        if not entities:
            # No entities found - return neutral score
            return VerificationResult(
                response=response,
                entities=[],
                verified_count=0,
                total_count=0,
                correctness_score=0.5,  # Neutral score when no entities
                verified_entities=[],
                unverified_entities=[]
            )
        
        # Verify entities
        verified_entities_list = []
        unverified_entities_list = []
        verified_count = 0
        total_confidence = 0.0
        
        if self.use_wikipedia and self.verifier:
            # Verify each entity
            for entity in entities:
                is_verified, confidence = self.verifier.verify_entity(entity.text)
                entity.verified = is_verified
                entity.confidence = confidence
                
                if is_verified and confidence >= min_confidence:
                    verified_count += 1
                    verified_entities_list.append(entity.text)
                    total_confidence += confidence
                else:
                    unverified_entities_list.append(entity.text)
        else:
            # No verification - assume all entities are valid
            # (useful for testing or when verification is disabled)
            verified_count = len(entities)
            verified_entities_list = [e.text for e in entities]
            for entity in entities:
                entity.verified = True
                entity.confidence = 1.0
                total_confidence += 1.0
        
        # Calculate correctness score
        # Score is based on proportion of verified entities
        if len(entities) > 0:
            if require_verification:
                # Strict: only verified entities count
                correctness_score = verified_count / len(entities)
            else:
                # Weighted: average confidence of all entities
                correctness_score = total_confidence / len(entities)
        else:
            correctness_score = 0.5  # Neutral when no entities
        
        return VerificationResult(
            response=response,
            entities=entities,
            verified_count=verified_count,
            total_count=len(entities),
            correctness_score=correctness_score,
            verified_entities=verified_entities_list,
            unverified_entities=unverified_entities_list
        )
    
    def verify_responses_batch(
        self,
        responses: List[str],
        min_confidence: float = 0.5
    ) -> List[VerificationResult]:
        """
        Verify multiple responses in batch.
        
        Args:
            responses: List of response texts
            min_confidence: Minimum confidence threshold
        
        Returns:
            List of VerificationResult objects
        """
        results = []
        for response in responses:
            result = self.verify_response(response, min_confidence)
            results.append(result)
        return results


def calculate_hybrid_score(
    model_score: float,
    entity_score: float,
    weight_model: float = 0.7,
    weight_entity: float = 0.3
) -> float:
    """
    Calculate hybrid score combining model prediction and entity verification.
    
    Args:
        model_score: Score from classification model (0-1)
        entity_score: Score from entity verification (0-1)
        weight_model: Weight for model score (default 0.7)
        weight_entity: Weight for entity score (default 0.3)
    
    Returns:
        Combined hybrid score (0-1)
    """
    # Normalize weights
    total_weight = weight_model + weight_entity
    weight_model = weight_model / total_weight
    weight_entity = weight_entity / total_weight
    
    # Calculate weighted average
    hybrid_score = (weight_model * model_score) + (weight_entity * entity_score)
    
    # Invert entity score for hallucination detection
    # Higher entity verification = lower hallucination probability
    # So we use (1 - entity_score) for hallucination detection
    hallucination_prob = (weight_model * (1 - model_score)) + (weight_entity * (1 - entity_score))
    
    return hallucination_prob


# Example usage and test cases
if __name__ == "__main__":
    # Test cases
    print("=" * 60)
    print("Entity Verification Module - Test Cases")
    print("=" * 60)
    
    # Initialize verifier
    try:
        verifier = EntityVerifier(
            extractor_method="spacy",
            use_wikipedia=True,
            rate_limit_delay=0.2  # Slower for testing
        )
    except Exception as e:
        print(f"Error initializing verifier: {e}")
        print("Falling back to transformers method...")
        try:
            verifier = EntityVerifier(
                extractor_method="transformers",
                use_wikipedia=True,
                rate_limit_delay=0.2
            )
        except Exception as e2:
            print(f"Error: {e2}")
            exit(1)
    
    # Test case 1: Response with verifiable entities
    print("\n" + "-" * 60)
    print("Test Case 1: Response with verifiable entities")
    print("-" * 60)
    response1 = "Barack Obama was the 44th President of the United States. He was born in Hawaii."
    result1 = verifier.verify_response(response1)
    print(f"Response: {response1}")
    print(f"Entities found: {len(result1.entities)}")
    for entity in result1.entities:
        print(f"  - {entity.text} ({entity.label}): Verified={entity.verified}, Confidence={entity.confidence}")
    print(f"Correctness Score: {result1.correctness_score:.3f}")
    print(f"Verified: {result1.verified_count}/{result1.total_count}")
    
    # Test case 2: Response with unverifiable/fictional entities
    print("\n" + "-" * 60)
    print("Test Case 2: Response with potentially unverifiable entities")
    print("-" * 60)
    response2 = "Dr. John Smith invented the quantum teleporter in 2025 at MIT."
    result2 = verifier.verify_response(response2)
    print(f"Response: {response2}")
    print(f"Entities found: {len(result2.entities)}")
    for entity in result2.entities:
        print(f"  - {entity.text} ({entity.label}): Verified={entity.verified}, Confidence={entity.confidence}")
    print(f"Correctness Score: {result2.correctness_score:.3f}")
    print(f"Verified: {result2.verified_count}/{result2.total_count}")
    
    # Test case 3: Response with mixed entities
    print("\n" + "-" * 60)
    print("Test Case 3: Response with mixed verifiable/unverifiable entities")
    print("-" * 60)
    response3 = "Albert Einstein developed the theory of relativity. He worked with Dr. Quantum at the Institute for Advanced Study."
    result3 = verifier.verify_response(response3)
    print(f"Response: {response3}")
    print(f"Entities found: {len(result3.entities)}")
    for entity in result3.entities:
        print(f"  - {entity.text} ({entity.label}): Verified={entity.verified}, Confidence={entity.confidence}")
    print(f"Correctness Score: {result3.correctness_score:.3f}")
    print(f"Verified: {result3.verified_count}/{result3.total_count}")
    print(f"Verified entities: {result3.verified_entities}")
    print(f"Unverified entities: {result3.unverified_entities}")
    
    # Test case 4: Response with no entities
    print("\n" + "-" * 60)
    print("Test Case 4: Response with no entities")
    print("-" * 60)
    response4 = "This is a general statement about concepts and ideas."
    result4 = verifier.verify_response(response4)
    print(f"Response: {response4}")
    print(f"Entities found: {len(result4.entities)}")
    print(f"Correctness Score: {result4.correctness_score:.3f}")
    
    # Test case 5: Hybrid score calculation
    print("\n" + "-" * 60)
    print("Test Case 5: Hybrid score calculation")
    print("-" * 60)
    model_score = 0.3  # Model predicts 30% chance of hallucination (70% correct)
    entity_score = result1.correctness_score
    hybrid = calculate_hybrid_score(model_score, entity_score)
    print(f"Model score (hallucination prob): {model_score:.3f}")
    print(f"Entity verification score: {entity_score:.3f}")
    print(f"Hybrid hallucination probability: {hybrid:.3f}")
    
    print("\n" + "=" * 60)
    print("All test cases completed!")
    print("=" * 60)

