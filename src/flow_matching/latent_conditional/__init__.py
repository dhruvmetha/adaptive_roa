"""
Latent Conditional Flow Matching implementation (Facebook FM)
"""

from .flow_matcher_fb import LatentConditionalFlowMatcher
from .inference import LatentConditionalFlowMatchingInference

__all__ = ["LatentConditionalFlowMatcher", "LatentConditionalFlowMatchingInference"]