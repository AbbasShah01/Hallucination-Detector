"""
Novel Research Fusion: Dynamic Multi-Signal Fusion (DMSF)

This module implements a novel fusion algorithm that dynamically adjusts
weights based on signal agreement and uncertainty.

This represents a novel research contribution to hallucination detection fusion.
"""

from .dmsf import DMSF, compute_dmsf, DMSFResult

__all__ = ['DMSF', 'compute_dmsf', 'DMSFResult']

