"""
Universal dynamical systems framework with Theseus Lie group support
"""
from .base import DynamicalSystem, ManifoldComponent
from .pendulum_universal import PendulumSystem
from .cartpole import CartPoleSystem

# Legacy imports for backward compatibility
from .pendulum import Pendulum

__all__ = [
    "DynamicalSystem", 
    "ManifoldComponent",
    "PendulumSystem", 
    "CartPoleSystem",
    "Pendulum"  # Legacy
]