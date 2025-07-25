"""
Circular Flow Matching Implementation

Circular-aware flow matching for systems with circular/periodic state variables,
specifically designed for pendulum dynamics with proper S¹ × ℝ topology handling.
"""

from .flow_matcher import CircularFlowMatcher
from .inference import CircularFlowMatchingInference

__all__ = [
    'CircularFlowMatcher',
    'CircularFlowMatchingInference',
]