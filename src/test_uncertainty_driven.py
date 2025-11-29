"""
Unit tests for Uncertainty-Driven Hallucination Score module.
"""

import unittest
import numpy as np
import torch
import torch.nn as nn

from uncertainty_driven_scorer import (
    UncertaintyDrivenScorer,
    UncertaintyScore,
    integrate_with_hybrid_fusion,
    MonteCarloDropout,
    EnsembleUncertainty
)


class TestUncertaintyDrivenScorer(unittest.TestCase):
    """Test cases for UncertaintyDrivenScorer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.scorer = UncertaintyDrivenScorer(
            uncertainty_method="mc_dropout",
            uncertainty_weight=0.3,
            uncertainty_threshold=0.5
        )
    
    def test_score_low_uncertainty(self):
        """Test scoring with low uncertainty."""
        result = self.scorer.score(
            base_prediction=0.7,
            epistemic_uncertainty=0.1,
            aleatoric_uncertainty=0.1
        )
        
        self.assertIsInstance(result, UncertaintyScore)
        self.assertGreaterEqual(result.uncertainty_driven_score, 0.0)
        self.assertLessEqual(result.uncertainty_driven_score, 1.0)
        self.assertEqual(result.uncertainty_type, "balanced")
    
    def test_score_high_uncertainty(self):
        """Test scoring with high uncertainty."""
        result = self.scorer.score(
            base_prediction=0.3,
            epistemic_uncertainty=0.8,
            aleatoric_uncertainty=0.2
        )
        
        # High uncertainty should increase score
        self.assertGreater(result.uncertainty_driven_score, result.base_prediction)
        self.assertGreater(result.total_uncertainty, 0.5)
    
    def test_score_epistemic_dominant(self):
        """Test when epistemic uncertainty dominates."""
        result = self.scorer.score(
            base_prediction=0.5,
            epistemic_uncertainty=0.7,
            aleatoric_uncertainty=0.1
        )
        
        self.assertEqual(result.uncertainty_type, "epistemic")
        self.assertGreater(result.epistemic_uncertainty, result.aleatoric_uncertainty)
    
    def test_score_aleatoric_dominant(self):
        """Test when aleatoric uncertainty dominates."""
        result = self.scorer.score(
            base_prediction=0.5,
            epistemic_uncertainty=0.1,
            aleatoric_uncertainty=0.7
        )
        
        self.assertEqual(result.uncertainty_type, "aleatoric")
        self.assertGreater(result.aleatoric_uncertainty, result.epistemic_uncertainty)
    
    def test_score_batch(self):
        """Test batch scoring."""
        base_preds = np.array([0.3, 0.5, 0.7])
        epi_uncs = np.array([0.6, 0.4, 0.2])
        ale_uncs = np.array([0.2, 0.3, 0.1])
        
        results = self.scorer.score_batch(base_preds, epi_uncs, ale_uncs)
        
        self.assertEqual(len(results), 3)
        for result in results:
            self.assertIsInstance(result, UncertaintyScore)
    
    def test_confidence_computation(self):
        """Test confidence computation."""
        result = self.scorer.score(
            base_prediction=0.5,
            epistemic_uncertainty=0.2,
            aleatoric_uncertainty=0.1
        )
        
        # Confidence should be inverse of total uncertainty
        expected_confidence = 1.0 - result.total_uncertainty
        self.assertAlmostEqual(result.confidence, expected_confidence, places=5)


class TestHybridFusionIntegration(unittest.TestCase):
    """Test integration with hybrid fusion."""
    
    def test_integration_basic(self):
        """Test basic integration."""
        scorer = UncertaintyDrivenScorer()
        uncertainty_result = scorer.score(0.4, 0.5, 0.3)
        
        fusion_prob = integrate_with_hybrid_fusion(
            transformer_prob=0.3,
            factual_score=0.9,
            agentic_score=0.8,
            uncertainty_score=uncertainty_result,
            alpha=0.5,
            beta=0.2,
            gamma=0.2,
            delta=0.1
        )
        
        self.assertGreaterEqual(fusion_prob, 0.0)
        self.assertLessEqual(fusion_prob, 1.0)
    
    def test_integration_without_agentic(self):
        """Test integration without agentic score."""
        scorer = UncertaintyDrivenScorer()
        uncertainty_result = scorer.score(0.4, 0.5, 0.3)
        
        fusion_prob = integrate_with_hybrid_fusion(
            transformer_prob=0.3,
            factual_score=0.9,
            agentic_score=None,
            uncertainty_score=uncertainty_result,
            alpha=0.6,
            beta=0.2,
            gamma=0.0,
            delta=0.2
        )
        
        self.assertGreaterEqual(fusion_prob, 0.0)
        self.assertLessEqual(fusion_prob, 1.0)
    
    def test_weight_normalization(self):
        """Test that weights are normalized."""
        scorer = UncertaintyDrivenScorer()
        uncertainty_result = scorer.score(0.4, 0.5, 0.3)
        
        # Weights don't sum to 1, should be normalized
        fusion_prob = integrate_with_hybrid_fusion(
            transformer_prob=0.3,
            factual_score=0.9,
            agentic_score=0.8,
            uncertainty_score=uncertainty_result,
            alpha=1.0,  # Doesn't sum to 1
            beta=0.5,
            gamma=0.5,
            delta=0.5
        )
        
        # Should still produce valid probability
        self.assertGreaterEqual(fusion_prob, 0.0)
        self.assertLessEqual(fusion_prob, 1.0)


class TestMonteCarloDropout(unittest.TestCase):
    """Test Monte Carlo Dropout wrapper."""
    
    def test_mc_dropout_initialization(self):
        """Test MC Dropout initialization."""
        # Create simple model
        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(5, 1)
        )
        
        mc_model = MonteCarloDropout(model, dropout_rate=0.1)
        self.assertIsNotNone(mc_model)
    
    def test_mc_dropout_forward(self):
        """Test MC Dropout forward pass."""
        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(5, 1)
        )
        
        mc_model = MonteCarloDropout(model, dropout_rate=0.1)
        x = torch.randn(1, 10)
        
        mean_pred, var_pred = mc_model(x, num_samples=5)
        
        self.assertIsNotNone(mean_pred)
        self.assertIsNotNone(var_pred)
        self.assertEqual(mean_pred.shape, (1, 1))
        self.assertEqual(var_pred.shape, (1, 1))


if __name__ == "__main__":
    unittest.main(verbosity=2)

