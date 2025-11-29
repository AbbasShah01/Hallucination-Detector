"""
Unit tests for entity verification module.
Run with: python -m pytest src/test_entity_verification.py
Or: python src/test_entity_verification.py
"""

import unittest
import sys
import os
from unittest.mock import Mock, patch, MagicMock

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from src.entity_verification import (
        Entity,
        EntityExtractor,
        WikipediaVerifier,
        EntityVerifier,
        VerificationResult,
        calculate_hybrid_score
    )
except ImportError:
    # Try relative import
    from entity_verification import (
        Entity,
        EntityExtractor,
        WikipediaVerifier,
        EntityVerifier,
        VerificationResult,
        calculate_hybrid_score
    )


class TestEntityExtractor(unittest.TestCase):
    """Test cases for EntityExtractor."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock spaCy if not available
        try:
            self.extractor = EntityExtractor(method="spacy")
        except:
            try:
                self.extractor = EntityExtractor(method="transformers")
            except:
                self.skipTest("Neither spaCy nor transformers available")
    
    def test_extract_entities_person(self):
        """Test extraction of person entities."""
        text = "Barack Obama was the President."
        entities = self.extractor.extract_entities(text)
        self.assertGreater(len(entities), 0)
        # Check that at least one entity was found
        entity_texts = [e.text.lower() for e in entities]
        self.assertTrue(any("obama" in text or "barack" in text for text in entity_texts))
    
    def test_extract_entities_organization(self):
        """Test extraction of organization entities."""
        text = "Microsoft and Google are tech companies."
        entities = self.extractor.extract_entities(text)
        entity_texts = [e.text.lower() for text in entities]
        # Should find at least one organization
        self.assertGreater(len(entities), 0)
    
    def test_extract_entities_empty_text(self):
        """Test extraction from empty text."""
        entities = self.extractor.extract_entities("")
        self.assertEqual(len(entities), 0)
    
    def test_extract_entities_no_entities(self):
        """Test extraction from text with no entities."""
        text = "This is a general statement about nothing specific."
        entities = self.extractor.extract_entities(text)
        # May or may not find entities, but should not crash
        self.assertIsInstance(entities, list)


class TestWikipediaVerifier(unittest.TestCase):
    """Test cases for WikipediaVerifier."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.verifier = WikipediaVerifier(rate_limit_delay=0.1)
    
    @patch('src.entity_verification.requests.Session.get')
    def test_verify_entity_found(self, mock_get):
        """Test verification of entity found in Wikipedia."""
        # Mock successful Wikipedia response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "title": "Barack Obama",
            "type": "standard"
        }
        mock_get.return_value = mock_response
        
        verified, confidence = self.verifier.verify_entity("Barack Obama")
        self.assertTrue(verified)
        self.assertGreater(confidence, 0.5)
    
    @patch('src.entity_verification.requests.Session.get')
    def test_verify_entity_not_found(self, mock_get):
        """Test verification of entity not found in Wikipedia."""
        # Mock 404 response
        mock_response = Mock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response
        
        # Mock search also returns nothing
        with patch.object(self.verifier, '_search_entity', return_value=(False, 0.0)):
            verified, confidence = self.verifier.verify_entity("FictionalCharacter123")
            self.assertFalse(verified)
            self.assertEqual(confidence, 0.0)
    
    @patch('src.entity_verification.requests.Session.get')
    def test_verify_entity_disambiguation(self, mock_get):
        """Test verification of disambiguation page."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "title": "Washington (disambiguation)",
            "type": "disambiguation"
        }
        mock_get.return_value = mock_response
        
        verified, confidence = self.verifier.verify_entity("Washington")
        self.assertTrue(verified)
        self.assertLess(confidence, 0.9)  # Lower confidence for disambiguation


class TestEntityVerifier(unittest.TestCase):
    """Test cases for EntityVerifier."""
    
    def setUp(self):
        """Set up test fixtures."""
        try:
            self.verifier = EntityVerifier(
                extractor_method="spacy",
                use_wikipedia=False  # Disable for faster tests
            )
        except:
            try:
                self.verifier = EntityVerifier(
                    extractor_method="transformers",
                    use_wikipedia=False
                )
            except:
                self.skipTest("Neither spaCy nor transformers available")
    
    def test_verify_response_with_entities(self):
        """Test verification of response with entities."""
        response = "Barack Obama was born in Hawaii."
        result = self.verifier.verify_response(response, require_verification=False)
        
        self.assertIsInstance(result, VerificationResult)
        self.assertEqual(result.response, response)
        self.assertGreaterEqual(result.correctness_score, 0.0)
        self.assertLessEqual(result.correctness_score, 1.0)
    
    def test_verify_response_no_entities(self):
        """Test verification of response with no entities."""
        response = "This is a general statement."
        result = self.verifier.verify_response(response)
        
        self.assertIsInstance(result, VerificationResult)
        self.assertEqual(len(result.entities), 0)
        self.assertEqual(result.correctness_score, 0.5)  # Neutral score
    
    def test_verify_responses_batch(self):
        """Test batch verification of multiple responses."""
        responses = [
            "Barack Obama was the President.",
            "This is a general statement.",
            "Microsoft is a company."
        ]
        results = self.verifier.verify_responses_batch(responses)
        
        self.assertEqual(len(results), len(responses))
        for result in results:
            self.assertIsInstance(result, VerificationResult)


class TestHybridScore(unittest.TestCase):
    """Test cases for hybrid score calculation."""
    
    def test_calculate_hybrid_score_basic(self):
        """Test basic hybrid score calculation."""
        model_score = 0.2  # 20% hallucination prob
        entity_score = 0.9  # 90% correct entities
        hybrid = calculate_hybrid_score(model_score, entity_score)
        
        self.assertGreaterEqual(hybrid, 0.0)
        self.assertLessEqual(hybrid, 1.0)
        # Should be weighted average
        self.assertLess(hybrid, 0.5)  # Both scores indicate low hallucination
    
    def test_calculate_hybrid_score_custom_weights(self):
        """Test hybrid score with custom weights."""
        model_score = 0.5
        entity_score = 0.5
        hybrid = calculate_hybrid_score(
            model_score, entity_score,
            weight_model=0.8, weight_entity=0.2
        )
        
        self.assertGreaterEqual(hybrid, 0.0)
        self.assertLessEqual(hybrid, 1.0)
    
    def test_calculate_hybrid_score_extremes(self):
        """Test hybrid score with extreme values."""
        # Both indicate high correctness
        hybrid1 = calculate_hybrid_score(0.0, 1.0)
        self.assertLess(hybrid1, 0.3)
        
        # Both indicate high hallucination
        hybrid2 = calculate_hybrid_score(1.0, 0.0)
        self.assertGreater(hybrid2, 0.7)


class TestIntegration(unittest.TestCase):
    """Integration tests for the full pipeline."""
    
    def test_full_pipeline_mock(self):
        """Test full pipeline with mocked Wikipedia."""
        try:
            verifier = EntityVerifier(
                extractor_method="spacy",
                use_wikipedia=True
            )
        except:
            try:
                verifier = EntityVerifier(
                    extractor_method="transformers",
                    use_wikipedia=True
                )
            except:
                self.skipTest("Dependencies not available")
        
        # Mock Wikipedia verification to avoid API calls
        with patch.object(verifier.verifier, 'verify_entity', return_value=(True, 0.9)):
            response = "Albert Einstein developed the theory of relativity."
            result = verifier.verify_response(response)
            
            self.assertIsInstance(result, VerificationResult)
            self.assertGreaterEqual(result.correctness_score, 0.0)
            self.assertLessEqual(result.correctness_score, 1.0)


if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2)

