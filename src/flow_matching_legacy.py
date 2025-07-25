"""
Legacy compatibility layer for flow matching

This module provides backward compatibility for old imports while
transitioning to the new unified flow matching architecture.
"""

import warnings
from typing import Any

# New unified imports
from flow_matching.standard.flow_matcher import StandardFlowMatcher
from flow_matching.standard.inference import StandardFlowMatchingInference
from flow_matching.circular.flow_matcher import CircularFlowMatcher  
from flow_matching.circular.inference import CircularFlowMatchingInference


def _deprecated_import_warning(old_name: str, new_name: str):
    """Show deprecation warning for old imports"""
    warnings.warn(
        f"Importing '{old_name}' is deprecated. "
        f"Please use '{new_name}' instead. "
        f"This compatibility layer will be removed in a future version.",
        DeprecationWarning,
        stacklevel=3
    )


# Legacy aliases for backward compatibility
class FlowMatching(StandardFlowMatcher):
    """Legacy alias for StandardFlowMatcher"""
    def __init__(self, *args, **kwargs):
        _deprecated_import_warning(
            "FlowMatching", 
            "flow_matching.standard.StandardFlowMatcher"
        )
        super().__init__(*args, **kwargs)


class FlowMatchingInference(StandardFlowMatchingInference):
    """Legacy alias for StandardFlowMatchingInference"""
    def __init__(self, *args, **kwargs):
        _deprecated_import_warning(
            "FlowMatchingInference", 
            "flow_matching.standard.StandardFlowMatchingInference"
        )
        super().__init__(*args, **kwargs)


class CircularFlowMatching(CircularFlowMatcher):
    """Legacy alias for CircularFlowMatcher"""
    def __init__(self, *args, **kwargs):
        _deprecated_import_warning(
            "CircularFlowMatching", 
            "flow_matching.circular.CircularFlowMatcher"
        )
        super().__init__(*args, **kwargs)


class CircularFlowMatchingInference(CircularFlowMatchingInference):
    """Legacy alias for CircularFlowMatchingInference"""
    def __init__(self, *args, **kwargs):
        _deprecated_import_warning(
            "CircularFlowMatchingInference", 
            "flow_matching.circular.CircularFlowMatchingInference"
        )
        super().__init__(*args, **kwargs)


# Legacy module-level imports for old scripts
__all__ = [
    'FlowMatching',
    'FlowMatchingInference', 
    'CircularFlowMatching',
    'CircularFlowMatchingInference',
]