"""
Universal dynamical systems framework
"""
from .base import DynamicalSystem, ManifoldComponent
from .pendulum import PendulumSystem  # NEW: from refactored file
from .cartpole import CartPoleSystem

# Legacy imports for backward compatibility
try:
    from .pendulum_universal import PendulumSystem as PendulumSystemUniversal
except ImportError:
    pass

__all__ = [
    "DynamicalSystem",
    "ManifoldComponent",
    "PendulumSystem",
    "CartPoleSystem",
]