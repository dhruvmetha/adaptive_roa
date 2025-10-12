"""
Pendulum Latent Conditional Flow Matching implementation (Facebook FM)
"""

from .flow_matcher_fb import PendulumLatentConditionalFlowMatcher
from .inference import LatentConditionalFlowMatchingInference

__all__ = [
    "PendulumLatentConditionalFlowMatcher",
    "LatentConditionalFlowMatchingInference"
]