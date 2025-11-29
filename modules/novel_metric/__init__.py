"""
Novel Research Metric: Semantic Hallucination Divergence Score (SHDS)

This module implements a novel hallucination severity metric that combines:
- Semantic embedding distance
- Factual mismatch penalty
- Reasoning-coherence penalty
- Uncertainty penalty

This represents a novel research contribution to hallucination detection.
"""

from .shds import SHDS, compute_shds, SHDSResult

__all__ = ['SHDS', 'compute_shds', 'SHDSResult']

