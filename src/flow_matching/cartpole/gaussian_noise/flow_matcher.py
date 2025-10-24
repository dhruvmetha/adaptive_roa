"""
CartPole Gaussian Noise Flow Matching implementation (REFACTORED)

INHERITS FROM: BaseGaussianNoiseFlowMatcher

KEY DIFFERENCES from Latent Conditional variant:
1. NO latent variables (removed z ~ N(0,I))
2. NO conditioning on start state
3. Initial noise sampled from Gaussian centered at start state: x₀ ~ N(start_state, σ²I)
4. Simplified model signature: f(x_t, t) instead of f(x_t, t, z, condition)
"""
import torch
import torch.nn as nn
from typing import Dict, Optional
import sys
sys.path.append('/common/home/dm1487/robotics_research/tripods/olympics-classifier/flow_matching')

from flow_matching.utils.manifolds import Product, FlatTorus, Euclidean

from src.flow_matching.base.gaussian_noise_flow_matcher import BaseGaussianNoiseFlowMatcher
from src.systems.base import DynamicalSystem


class CartPoleGaussianNoiseFlowMatcher(BaseGaussianNoiseFlowMatcher):
    """
    CartPole Gaussian Noise Flow Matching using Facebook FM

    Simplified flow matching WITHOUT:
    - Latent variables z
    - Conditioning on start state

    WITH:
    - Gaussian-perturbed initial states: x₀ ~ N(start_state, σ²I)
    - Simplified model: f(x_t, t) → velocity

    REFACTORED: Now inherits from BaseGaussianNoiseFlowMatcher
    - ✅ All generic code moved to base class (~500 lines)
    - ✅ Only CartPole-specific code remains (~50 lines)
    - ✅ System-agnostic angle wrapping via system.get_circular_indices()
    """

    def __init__(self,
                 system: DynamicalSystem,
                 model: nn.Module,
                 optimizer,
                 scheduler,
                 model_config: Optional[dict] = None,
                 noise_std: float = 0.1,
                 mae_val_frequency: int = 10):
        """
        Initialize CartPole Gaussian-Perturbed flow matcher

        Args:
            system: DynamicalSystem (CartPole with ℝ²×S¹×ℝ structure)
            model: CartPoleGaussianPerturbedUNet1D model
            optimizer: Optimizer instance
            scheduler: Learning rate scheduler
            model_config: Configuration dict
            noise_std: Standard deviation of Gaussian perturbation around start state
            mae_val_frequency: Compute MAE validation every N epochs
        """
        super().__init__(system, model, optimizer, scheduler, model_config, noise_std, mae_val_frequency)

        print("✅ Initialized CartPole Gaussian-Perturbed FM (REFACTORED):")
        print(f"   - Manifold: ℝ²×S¹×ℝ (Euclidean × FlatTorus × Euclidean)")
        print(f"   - Path: GeodesicProbPath with CondOTScheduler")
        print(f"   - Gaussian noise std: {noise_std}")
        print(f"   - NO latent variables")
        print(f"   - NO conditioning on start state")
        print(f"   - MAE validation frequency: every {mae_val_frequency} epochs")

    def _create_manifold(self):
        """
        Create ℝ²×S¹×ℝ manifold for CartPole

        Built from system.manifold_components:
        - [0] Real(cart_position)
        - [1] SO2(pole_angle)
        - [2] Real(cart_velocity)
        - [3] Real(pole_angular_velocity)
        """
        # Verify manifold structure matches expectations
        assert len(self.system.manifold_components) == 4, \
            f"CartPole should have 4 components, got {len(self.system.manifold_components)}"
        assert self.system.manifold_components[1].manifold_type == "SO2", \
            "CartPole component [1] should be SO2 (pole angle)"

        # Build manifold: ℝ × S¹ × ℝ²
        return Product(input_dim=4, manifolds=[
            (Euclidean(), 1),   # cart_position
            (FlatTorus(), 1),   # pole_angle (circular)
            (Euclidean(), 2)    # cart_velocity, pole_angular_velocity
        ])

    def _get_dimension_name(self, dim_idx: int) -> str:
        """
        Get human-readable dimension name for CartPole

        Uses system.manifold_components for consistency
        """
        if 0 <= dim_idx < len(self.system.manifold_components):
            return self.system.manifold_components[dim_idx].name
        return f"dim_{dim_idx}"
