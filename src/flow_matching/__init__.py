"""
Flow Matching Package

This package provides a unified framework for flow matching models with support
for both standard and circular (topology-aware) flow matching variants.

Key components:
- base: Abstract base classes and common functionality
- standard: Standard flow matching implementation using torchcfm
- circular: Circular-aware flow matching for pendulum dynamics
- utils: Shared utilities for state transformations and geometry
"""

from .base.flow_matcher import BaseFlowMatcher
from .base.inference import BaseFlowMatchingInference

# Import specialized implementations
from .standard.flow_matcher import StandardFlowMatcher
from .standard.inference import StandardFlowMatchingInference
from .circular.flow_matcher import CircularFlowMatcher
from .circular.inference import CircularFlowMatchingInference
from .latent_circular.flow_matcher import LatentCircularFlowMatcher
from .latent_circular.inference import LatentCircularInference

__all__ = [
    # Base classes
    'BaseFlowMatcher',
    'BaseFlowMatchingInference',
    
    # Standard implementation
    'StandardFlowMatcher', 
    'StandardFlowMatchingInference',
    
    # Circular implementation
    'CircularFlowMatcher',
    'CircularFlowMatchingInference',
    
    # Latent circular implementation (VAE-style)
    'LatentCircularFlowMatcher',
    'LatentCircularInference',
]