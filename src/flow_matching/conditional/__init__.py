"""
Conditional flow matching with noise-to-endpoint generation and FiLM conditioning
"""

from .flow_matcher import ConditionalFlowMatcher
from .inference import ConditionalFlowMatchingInference

__all__ = ['ConditionalFlowMatcher', 'ConditionalFlowMatchingInference']