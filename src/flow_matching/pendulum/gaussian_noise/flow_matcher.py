"""
Pendulum Gaussian Noise Flow Matching implementation

INHERITS FROM: BaseGaussianNoiseFlowMatcher

KEY DIFFERENCES from Latent Conditional variant:
1. NO latent variables (removed z ~ N(0,I))
2. NO conditioning on start state
3. Initial noise sampled from Gaussian centered at start state: x₀ ~ N(start_state, σ²I)
4. Simplified model signature: f(x_t, t) instead of f(x_t, t, z, condition)

ADVANTAGES:
- Simpler architecture (25% fewer parameters)
- Faster training and inference
- Direct perturbation-based uncertainty quantification
"""
import torch
import torch.nn as nn
from typing import Dict, Optional
import sys
sys.path.append('/common/home/dm1487/robotics_research/tripods/olympics-classifier/flow_matching')

from flow_matching.utils.manifolds import Product, FlatTorus, Euclidean

from src.flow_matching.base.gaussian_noise_flow_matcher import BaseGaussianNoiseFlowMatcher
from src.systems.base import DynamicalSystem


class PendulumGaussianNoiseFlowMatcher(BaseGaussianNoiseFlowMatcher):
    """
    Pendulum Gaussian Noise Flow Matching using Facebook FM

    Simplified flow matching WITHOUT:
    - Latent variables z
    - Conditioning on start state

    WITH:
    - Gaussian-perturbed initial states: x₀ ~ N(start_state, σ²I)
    - Simplified model: f(x_t, t) → velocity

    State space: S¹×ℝ (angle θ ∈ S¹, angular velocity θ̇ ∈ ℝ)

    IMPLEMENTATION:
    - Inherits ~500 lines of generic code from BaseGaussianNoiseFlowMatcher
    - Only ~60 lines of Pendulum-specific code
    - System-agnostic angle wrapping via system.get_circular_indices()
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
        Initialize Pendulum Gaussian-Perturbed flow matcher

        Args:
            system: DynamicalSystem (Pendulum with S¹×ℝ structure)
            model: PendulumGaussianNoiseUNet1D model
            optimizer: Optimizer instance
            scheduler: Learning rate scheduler
            model_config: Configuration dict
            noise_std: Standard deviation of Gaussian perturbation around start state
            mae_val_frequency: Compute MAE validation every N epochs
        """
        super().__init__(system, model, optimizer, scheduler, model_config, noise_std, mae_val_frequency)

        print("✅ Initialized Pendulum Gaussian-Perturbed FM:")
        print(f"   - Manifold: S¹×ℝ (FlatTorus × Euclidean)")
        print(f"   - Path: GeodesicProbPath with CondOTScheduler")
        print(f"   - Gaussian noise std: {noise_std}")
        print(f"   - NO latent variables")
        print(f"   - NO conditioning on start state")
        print(f"   - MAE validation frequency: every {mae_val_frequency} epochs")

    def _create_manifold(self):
        """
        Create S¹×ℝ manifold for Pendulum

        Built from system.manifold_components:
        - [0] SO2(angle)
        - [1] Real(angular_velocity)
        """
        # Verify manifold structure matches expectations
        assert len(self.system.manifold_components) == 2, \
            f"Pendulum should have 2 components, got {len(self.system.manifold_components)}"
        assert self.system.manifold_components[0].manifold_type == "SO2", \
            "Pendulum component [0] should be SO2 (angle)"

        # Build manifold: S¹ × ℝ
        return Product(input_dim=2, manifolds=[
            (FlatTorus(), 1),   # angle (circular)
            (Euclidean(), 1)    # angular_velocity
        ])

    def _get_dimension_name(self, dim_idx: int) -> str:
        """
        Get human-readable dimension name for Pendulum

        Uses system.manifold_components for consistency
        """
        if 0 <= dim_idx < len(self.system.manifold_components):
            return self.system.manifold_components[dim_idx].name
        return f"dim_{dim_idx}"
