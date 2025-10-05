"""
Flow Matching Utilities

Shared utilities for state transformations, geometric computations,
and other common operations across flow matching variants.
"""

from .state_transformations import (
    embed_circular_state,
    extract_circular_state, 
    normalize_states,
    denormalize_states
)

from .geometry import (
    circular_distance,
    geodesic_interpolation,
    compute_circular_velocity
)

from .scheduler_utils import (
    create_warmup_cosine_scheduler
)

__all__ = [
    # State transformations
    'embed_circular_state',
    'extract_circular_state',
    'normalize_states', 
    'denormalize_states',
    
    # Geometry
    'circular_distance',
    'geodesic_interpolation',
    'compute_circular_velocity',
    
    # Schedulers
    'create_warmup_cosine_scheduler',
]