"""
Flow Matching Package

This package provides flow matching models for robotics applications.

Key components:
- base: Abstract base classes and common functionality
- latent_conditional: Latent conditional flow matching for pendulum and cartpole
- utils: Shared utilities for state transformations and geometry
"""

from .base.flow_matcher import BaseFlowMatcher
from .base.inference import BaseFlowMatchingInference

__all__ = [
    # Base classes
    'BaseFlowMatcher',
    'BaseFlowMatchingInference',
]