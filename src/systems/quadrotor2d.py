"""
Quadrotor 2D system for classification and flow matching.

State space (6D raw, 7D embedded):
- Position: x, z (Euclidean)
- Orientation: theta (S¹, pitch angle)
- Linear velocity: x_dot, z_dot (Euclidean)
- Angular velocity: theta_dot (Euclidean)

Goal state: [0, 1, 0, 0, 0, 0]
(hover at x=0, altitude z=1, level orientation, zero velocities)
"""
import torch
import numpy as np
from src.systems.base import DynamicalSystem, ManifoldComponent
from typing import List, Dict, Tuple


class Quadrotor2DSystem(DynamicalSystem):
    """
    Quadrotor 2D system with ℝ² × S¹ × ℝ³ manifold structure

    State representation: (x, z, theta, x_dot, z_dot, theta_dot) where:
    - x ∈ ℝ (horizontal position)
    - z ∈ ℝ (vertical position / altitude)
    - theta ∈ S¹ (pitch angle, circular)
    - x_dot ∈ ℝ (horizontal velocity)
    - z_dot ∈ ℝ (vertical velocity)
    - theta_dot ∈ ℝ (angular velocity)
    """

    # Goal state: hover at x=0, z=1, level, zero velocities
    GOAL_STATE = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])

    # State bounds from dataset_description.json
    X_LIMIT = 1.0
    Z_MIN = 0.1
    Z_MAX = 1.5

    X_DOT_LIMIT = 1.0
    Z_DOT_LIMIT = 1.0

    # Angular velocity limit (achieved bounds show ~±12.7 rad/s)
    THETA_DOT_LIMIT = 12.0

    def __init__(self):
        """Initialize Quadrotor 2D system."""
        super().__init__()
        self.name = "quadrotor2d"

    def define_manifold_structure(self) -> List[ManifoldComponent]:
        """
        Define ℝ² × S¹ × ℝ³ manifold structure.
        """
        return [
            ManifoldComponent("Real", 1, "x"),           # x position
            ManifoldComponent("Real", 1, "z"),           # z position (altitude)
            ManifoldComponent("SO2", 1, "theta"),        # pitch angle (S¹)
            ManifoldComponent("Real", 1, "x_dot"),       # x velocity
            ManifoldComponent("Real", 1, "z_dot"),       # z velocity
            ManifoldComponent("Real", 1, "theta_dot"),   # angular velocity
        ]

    def define_state_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Define state bounds for normalization."""
        return {
            "x": (-self.X_LIMIT, self.X_LIMIT),
            "z": (self.Z_MIN, self.Z_MAX),
            "theta": (-np.pi, np.pi),
            "x_dot": (-self.X_DOT_LIMIT, self.X_DOT_LIMIT),
            "z_dot": (-self.Z_DOT_LIMIT, self.Z_DOT_LIMIT),
            "theta_dot": (-self.THETA_DOT_LIMIT, self.THETA_DOT_LIMIT),
        }

    def attractors(self) -> List[List[float]]:
        """
        Quadrotor 2D attractor positions.

        Returns:
            List of [x, z, theta, x_dot, z_dot, theta_dot] attractor positions.
        """
        return [self.GOAL_STATE.tolist()]

    def is_in_attractor(self, state, radius: float = 0.05):
        """
        Check if states are within the attractor basin (hover goal).

        Uses Euclidean distance in the full 6D state space to check if
        the state is within the specified radius of the goal state.

        Args:
            state: States as numpy array or torch tensor.
                   Shape [6] for single state or [B, 6] for batch.
            radius: Attractor radius (Euclidean distance threshold).

        Returns:
            Boolean (single state) or boolean array/tensor [B] for batch.
        """
        # Handle numpy arrays
        if isinstance(state, np.ndarray):
            if state.ndim == 1:
                distance = np.linalg.norm(state - self.GOAL_STATE)
                return distance < radius
            else:
                distances = np.linalg.norm(state - self.GOAL_STATE, axis=1)
                return distances < radius

        # Handle torch tensors
        if isinstance(state, torch.Tensor):
            goal = torch.tensor(self.GOAL_STATE, device=state.device, dtype=state.dtype)

            if state.dim() == 1:
                distance = torch.norm(state - goal)
                return (distance < radius).item()
            else:
                distances = torch.norm(state - goal, dim=1)
                return distances < radius

        raise TypeError(f"Unsupported state type: {type(state)}")

    def normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        """
        Normalize state for model input.

        Normalization:
        - x: normalize to [-1, 1] using X_LIMIT
        - z: normalize from [0.1, 1.5] to [-1, 1]
        - theta: keep as-is (already in [-π, π])
        - x_dot, z_dot: normalize to [-1, 1] using limits
        - theta_dot: normalize to [-1, 1] using limit

        Args:
            state: [B, 6] raw state

        Returns:
            [B, 6] normalized state
        """
        normalized = state.clone()

        # x position
        normalized[:, 0] = state[:, 0] / self.X_LIMIT

        # z: normalize from [Z_MIN, Z_MAX] to [-1, 1]
        z_center = (self.Z_MAX + self.Z_MIN) / 2
        z_range = (self.Z_MAX - self.Z_MIN) / 2
        normalized[:, 1] = (state[:, 1] - z_center) / z_range

        # theta: keep as-is
        # normalized[:, 2] = state[:, 2]

        # Velocities
        normalized[:, 3] = state[:, 3] / self.X_DOT_LIMIT
        normalized[:, 4] = state[:, 4] / self.Z_DOT_LIMIT
        normalized[:, 5] = state[:, 5] / self.THETA_DOT_LIMIT

        return normalized

    def denormalize_state(self, normalized_state: torch.Tensor) -> torch.Tensor:
        """
        Denormalize state back to raw coordinates.

        Args:
            normalized_state: [B, 6] normalized state

        Returns:
            [B, 6] raw state
        """
        denormalized = normalized_state.clone()

        # x position
        denormalized[:, 0] = normalized_state[:, 0] * self.X_LIMIT

        # z: denormalize from [-1, 1] to [Z_MIN, Z_MAX]
        z_center = (self.Z_MAX + self.Z_MIN) / 2
        z_range = (self.Z_MAX - self.Z_MIN) / 2
        denormalized[:, 1] = normalized_state[:, 1] * z_range + z_center

        # theta: keep as-is
        # denormalized[:, 2] = normalized_state[:, 2]

        # Velocities
        denormalized[:, 3] = normalized_state[:, 3] * self.X_DOT_LIMIT
        denormalized[:, 4] = normalized_state[:, 4] * self.Z_DOT_LIMIT
        denormalized[:, 5] = normalized_state[:, 5] * self.THETA_DOT_LIMIT

        return denormalized

    def embed_state_for_model(self, state: torch.Tensor) -> torch.Tensor:
        """
        Embed state for neural network input.

        Converts pitch angle to sin/cos representation for proper S¹ handling.

        Input: [B, 6] (x, z, theta, x_dot, z_dot, theta_dot)
        Output: [B, 7] (x, z, sin(theta), cos(theta), x_dot, z_dot, theta_dot)

        Args:
            state: [B, 6] state

        Returns:
            [B, 7] embedded state with sin/cos for theta
        """
        x = state[:, 0:1]
        z = state[:, 1:2]
        theta = state[:, 2]
        x_dot = state[:, 3:4]
        z_dot = state[:, 4:5]
        theta_dot = state[:, 5:6]

        sin_theta = torch.sin(theta).unsqueeze(1)
        cos_theta = torch.cos(theta).unsqueeze(1)

        return torch.cat([x, z, sin_theta, cos_theta, x_dot, z_dot, theta_dot], dim=1)

    def __repr__(self) -> str:
        return f"Quadrotor2DSystem(ℝ² × S¹ × ℝ³, state_dim=6, embedded_dim=7)"
