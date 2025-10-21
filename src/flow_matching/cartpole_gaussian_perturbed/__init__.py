"""
CartPole Gaussian-Perturbed Flow Matching

Simplified flow matching variant without latent variables or conditioning.
Initial states are sampled from Gaussian distributions centered at start states.
"""

from .flow_matcher import CartPoleGaussianPerturbedFlowMatcher, SimpleVelocityWrapper

__all__ = [
    'CartPoleGaussianPerturbedFlowMatcher',
    'SimpleVelocityWrapper',
]
