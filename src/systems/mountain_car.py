"""Mountain Car 2D System Definition

Mountain Car is a classic control problem with a 2D state space (position, velocity).
The car must navigate from a valley to reach a goal position on the right hill.

State Space: ℝ² (pure Euclidean manifold)
- position (x): Position along the track, goal at π/6 ≈ 0.524
- velocity (ẋ): Velocity of the car

Manifold Structure: ℝ² (Euclidean, no circular coordinates)
"""

import numpy as np
import torch
from typing import List, Tuple, Dict, Optional
import pickle

from src.systems.base import DynamicalSystem, ManifoldComponent


class MountainCarSystem(DynamicalSystem):
    """Mountain Car 2D dynamical system.

    State: [position, velocity]
    Manifold: ℝ² (pure Euclidean)
    Goal: Reach position ≈ π/6 ≈ 0.524 with low velocity
    """

    def __init__(self, bounds_file: Optional[str] = None, use_dynamic_bounds: bool = False):
        """Initialize Mountain Car system.

        Args:
            bounds_file: Path to pickle file with per-dimension bounds
            use_dynamic_bounds: Whether to use dynamic bounds from data
        """
        # Goal position (π/6 in radians)
        self.goal_position = np.pi / 6  # ≈ 0.524
        self.goal_tolerance = 0.05
        self.velocity_tolerance = 0.02  # Success requires |velocity| < 0.02

        # Load bounds if provided
        if bounds_file:
            with open(bounds_file, 'rb') as f:
                bounds_data = pickle.load(f)

            # Extract per-dimension limits
            self.position_limit = max(
                abs(bounds_data['bounds'][0]['min']),
                abs(bounds_data['bounds'][0]['max'])
            )
            self.velocity_limit = max(
                abs(bounds_data['bounds'][1]['min']),
                abs(bounds_data['bounds'][1]['max'])
            )

            print(f"✅ Loaded MountainCar bounds from: {bounds_file}")
            print(f"   Position limit: ±{self.position_limit:.4f}")
            print(f"   Velocity limit: ±{self.velocity_limit:.4f}")
        else:
            # Default bounds based on data analysis
            self.position_limit = 3.5  # Covers [-3.41, 0.52]
            self.velocity_limit = 0.1  # Covers [-0.086, 0.053]
            print(f"⚠️  Using default MountainCar bounds (no bounds file provided)")
            print(f"   Position limit: ±{self.position_limit:.4f}")
            print(f"   Velocity limit: ±{self.velocity_limit:.4f}")

        # Store per-dimension limits for normalization
        self.dimension_limits = {
            0: self.position_limit,
            1: self.velocity_limit
        }

        # Call parent constructor
        super().__init__()

    def define_manifold_structure(self) -> List[ManifoldComponent]:
        """Define the manifold structure: ℝ² (pure Euclidean).

        Returns:
            List of ManifoldComponent objects defining the geometry
        """
        return [
            ManifoldComponent("Real", 1, "position"),
            ManifoldComponent("Real", 1, "velocity")
        ]

    def define_state_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Define bounds for each manifold component.

        Returns:
            Dictionary mapping component names to (min, max) bounds
        """
        return {
            "position": (-self.position_limit, self.position_limit),
            "velocity": (-self.velocity_limit, self.velocity_limit)
        }

    def attractors(self) -> List[List[float]]:
        """Define system attractors (goal states).

        Mountain Car has one attractor: goal position with zero velocity.

        Returns:
            List of attractor states
        """
        return [
            [self.goal_position, 0.0]  # [position=π/6, velocity=0]
        ]

    def normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        """Normalize state to roughly [-1, 1] range using per-dimension limits.

        For pure Euclidean manifold: divide each dimension by its limit.

        Args:
            state: Tensor of shape [B, 2] with [position, velocity]

        Returns:
            Normalized state tensor
        """
        normalized = state.clone()
        normalized[:, 0] = state[:, 0] / self.position_limit
        normalized[:, 1] = state[:, 1] / self.velocity_limit
        return normalized

    def denormalize_state(self, normalized_state: torch.Tensor) -> torch.Tensor:
        """Denormalize state from [-1, 1] range back to physical units.

        Args:
            normalized_state: Tensor of shape [B, 2] with normalized values

        Returns:
            Denormalized state tensor
        """
        denormalized = normalized_state.clone()
        denormalized[:, 0] = normalized_state[:, 0] * self.position_limit
        denormalized[:, 1] = normalized_state[:, 1] * self.velocity_limit
        return denormalized

    def embed_state_for_model(self, normalized_state: torch.Tensor) -> torch.Tensor:
        """Embed state for model input.

        For pure Euclidean manifold (no circular coordinates):
        No embedding needed - return state as-is (identity mapping).

        Args:
            normalized_state: Normalized state tensor [B, 2]

        Returns:
            Embedded state (same as input for Euclidean)
        """
        # Pure Euclidean: no sin/cos embedding needed
        return normalized_state

    def is_in_attractor(self, state,
                       threshold: Optional[float] = None):
        """Check if states are in the attractor basin.

        For Mountain Car: check if position is near goal AND velocity is low.
        Success condition matches dataset: |position - π/6| < 0.05 AND |velocity| < 0.02

        Args:
            state: State tensor [B, state_dim]
            threshold: Distance threshold for position (default: use goal_tolerance)

        Returns:
            Boolean tensor [B] indicating if in attractor
        """


        # Position condition: |position - π/6| < 0.05
        single_state = False
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()

        if state.dim() == 1:
            state = state.unsqueeze(0)
            single_state = True
        

        position_error = torch.abs(state[:, 0] - self.goal_position)
        position_ok = position_error < self.goal_tolerance

        # Velocity condition: |velocity| < 0.02
        velocity_ok = torch.abs(state[:, 1]) < self.velocity_tolerance

        # Both conditions must be satisfied
        result = position_ok & velocity_ok
        if single_state:
            return result.item()
        return result

    def classify_attractor(self, state: torch.Tensor,
                          radius: float = 0.1) -> torch.Tensor:
        """Classify which attractor each state belongs to.

        Args:
            state: State tensor [B, state_dim]
            radius: Distance threshold (renamed from threshold for consistency with other systems)

        Returns:
            Tensor [B] with attractor indices (0 for goal, -1 for none)
        """
        # Check if in goal attractor (use radius as threshold internally)
        in_goal = self.is_in_attractor(state, threshold=radius)

        # Return attractor index (0) or -1 for no attractor
        attractor_class = torch.where(in_goal,
                                      torch.zeros_like(in_goal, dtype=torch.long),
                                      torch.full_like(in_goal, -1, dtype=torch.long))
        return attractor_class

    def __repr__(self) -> str:
        """String representation of the system."""
        return (
            f"MountainCarSystem(\n"
            f"  state_dim={self.state_dim},\n"
            f"  manifold=ℝ²,\n"
            f"  position_limit=±{self.position_limit:.4f},\n"
            f"  velocity_limit=±{self.velocity_limit:.4f},\n"
            f"  goal_position={self.goal_position:.4f} (π/6),\n"
            f"  goal_tolerance={self.goal_tolerance:.4f},\n"
            f"  velocity_tolerance={self.velocity_tolerance:.4f}\n"
            f")"
        )
