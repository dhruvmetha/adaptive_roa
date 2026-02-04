"""
Quadrotor 3D system for classification and flow matching.

State space (13D):
- Position: x, y, z (Euclidean)
- Orientation: qw, qx, qy, qz (unit quaternion, SO(3))
- Linear velocity: x_dot, y_dot, z_dot (Euclidean)
- Angular velocity: p, q, r (body frame, Euclidean)

Goal state: [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
(hover at origin, altitude 1m, identity orientation, zero velocities)
"""
import torch
import numpy as np
from src.systems.base import DynamicalSystem, ManifoldComponent
from typing import List, Dict, Tuple


class Quadrotor3DSystem(DynamicalSystem):
    """
    Quadrotor 3D system with ℝ³ × SO(3) × ℝ⁶ manifold structure

    State representation: (x, y, z, qw, qx, qy, qz, x_dot, y_dot, z_dot, p, q, r) where:
    - x, y, z ∈ ℝ (position)
    - qw, qx, qy, qz ∈ S³ (unit quaternion for orientation)
    - x_dot, y_dot, z_dot ∈ ℝ (linear velocity)
    - p, q, r ∈ ℝ (angular velocity in body frame)
    """

    # Goal state: hover at origin, altitude 1m, identity quaternion, zero velocities
    GOAL_STATE = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # State bounds from dataset_description.json
    X_LIMIT = 1.8
    Y_LIMIT = 1.8
    Z_MIN = 0.1
    Z_MAX = 3.0

    X_DOT_LIMIT = 3.0
    Y_DOT_LIMIT = 3.0
    Z_DOT_LIMIT = 3.0

    # Angular velocity limits (achieved bounds show ~±40 rad/s during tumbling)
    P_LIMIT = 40.0
    Q_LIMIT = 40.0
    R_LIMIT = 40.0

    def __init__(self):
        """Initialize Quadrotor 3D system."""
        super().__init__()
        self.name = "quadrotor3d"

    def define_manifold_structure(self) -> List[ManifoldComponent]:
        """
        Define ℝ³ × SO(3) × ℝ⁶ manifold structure.

        Note: Quaternion is treated as unit quaternion (SO(3)), not normalized like
        Euclidean components. The constraint |q| = 1 is assumed to hold.
        """
        return [
            ManifoldComponent("Real", 1, "x"),           # x position
            ManifoldComponent("Real", 1, "y"),           # y position
            ManifoldComponent("Real", 1, "z"),           # z position
            ManifoldComponent("Quaternion", 4, "orientation"),  # qw, qx, qy, qz
            ManifoldComponent("Real", 1, "x_dot"),       # x velocity
            ManifoldComponent("Real", 1, "y_dot"),       # y velocity
            ManifoldComponent("Real", 1, "z_dot"),       # z velocity
            ManifoldComponent("Real", 1, "p"),           # roll rate
            ManifoldComponent("Real", 1, "q"),           # pitch rate
            ManifoldComponent("Real", 1, "r"),           # yaw rate
        ]

    def define_state_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Define state bounds for normalization."""
        return {
            "x": (-self.X_LIMIT, self.X_LIMIT),
            "y": (-self.Y_LIMIT, self.Y_LIMIT),
            "z": (self.Z_MIN, self.Z_MAX),
            "orientation": (-1.0, 1.0),  # Quaternion components in [-1, 1]
            "x_dot": (-self.X_DOT_LIMIT, self.X_DOT_LIMIT),
            "y_dot": (-self.Y_DOT_LIMIT, self.Y_DOT_LIMIT),
            "z_dot": (-self.Z_DOT_LIMIT, self.Z_DOT_LIMIT),
            "p": (-self.P_LIMIT, self.P_LIMIT),
            "q": (-self.Q_LIMIT, self.Q_LIMIT),
            "r": (-self.R_LIMIT, self.R_LIMIT),
        }

    def attractors(self) -> List[List[float]]:
        """
        Quadrotor 3D attractor positions.

        Returns:
            List of [x, y, z, qw, qx, qy, qz, x_dot, y_dot, z_dot, p, q, r] attractor positions.
        """
        return [self.GOAL_STATE.tolist()]

    def is_in_attractor(self, state, radius: float = 0.05):
        """
        Check if states are within the attractor basin (hover goal).

        Uses Euclidean distance in the full 13D state space to check if
        the state is within the specified radius of the goal state.

        Args:
            state: States as numpy array or torch tensor.
                   Shape [13] for single state or [B, 13] for batch.
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
        - Position (x, y): normalize to [-1, 1] using limits
        - Position (z): normalize from [0.1, 3.0] to [-1, 1]
        - Quaternion: kept as-is (already unit norm)
        - Velocities: normalize to [-1, 1] using limits
        - Angular velocities: normalize to [-1, 1] using limits

        Args:
            state: [B, 13] raw state

        Returns:
            [B, 13] normalized state
        """
        normalized = state.clone()

        # Position
        normalized[:, 0] = state[:, 0] / self.X_LIMIT
        normalized[:, 1] = state[:, 1] / self.Y_LIMIT

        # z: normalize from [Z_MIN, Z_MAX] to [-1, 1]
        z_center = (self.Z_MAX + self.Z_MIN) / 2
        z_range = (self.Z_MAX - self.Z_MIN) / 2
        normalized[:, 2] = (state[:, 2] - z_center) / z_range

        # Quaternion: keep as-is (indices 3-6)
        # normalized[:, 3:7] = state[:, 3:7]

        # Linear velocities
        normalized[:, 7] = state[:, 7] / self.X_DOT_LIMIT
        normalized[:, 8] = state[:, 8] / self.Y_DOT_LIMIT
        normalized[:, 9] = state[:, 9] / self.Z_DOT_LIMIT

        # Angular velocities
        normalized[:, 10] = state[:, 10] / self.P_LIMIT
        normalized[:, 11] = state[:, 11] / self.Q_LIMIT
        normalized[:, 12] = state[:, 12] / self.R_LIMIT

        return normalized

    def denormalize_state(self, normalized_state: torch.Tensor) -> torch.Tensor:
        """
        Denormalize state back to raw coordinates.

        Args:
            normalized_state: [B, 13] normalized state

        Returns:
            [B, 13] raw state
        """
        denormalized = normalized_state.clone()

        # Position
        denormalized[:, 0] = normalized_state[:, 0] * self.X_LIMIT
        denormalized[:, 1] = normalized_state[:, 1] * self.Y_LIMIT

        # z: denormalize from [-1, 1] to [Z_MIN, Z_MAX]
        z_center = (self.Z_MAX + self.Z_MIN) / 2
        z_range = (self.Z_MAX - self.Z_MIN) / 2
        denormalized[:, 2] = normalized_state[:, 2] * z_range + z_center

        # Quaternion: keep as-is (indices 3-6)
        # denormalized[:, 3:7] = normalized_state[:, 3:7]

        # Linear velocities
        denormalized[:, 7] = normalized_state[:, 7] * self.X_DOT_LIMIT
        denormalized[:, 8] = normalized_state[:, 8] * self.Y_DOT_LIMIT
        denormalized[:, 9] = normalized_state[:, 9] * self.Z_DOT_LIMIT

        # Angular velocities
        denormalized[:, 10] = normalized_state[:, 10] * self.P_LIMIT
        denormalized[:, 11] = normalized_state[:, 11] * self.Q_LIMIT
        denormalized[:, 12] = normalized_state[:, 12] * self.R_LIMIT

        return denormalized

    def embed_state_for_model(self, state: torch.Tensor) -> torch.Tensor:
        """
        Embed state for neural network input.

        For Quadrotor 3D, the quaternion is kept as-is (not converted to sin/cos)
        since it's already a proper SO(3) representation.

        Args:
            state: [B, 13] state

        Returns:
            [B, 13] embedded state (same as input for Quadrotor 3D)
        """
        return state

    def __repr__(self) -> str:
        return f"Quadrotor3DSystem(ℝ³ × SO(3) × ℝ⁶, state_dim=13)"
