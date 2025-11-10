"""
CartPole DeepMind Control Suite System for Latent Conditional Flow Matching

Manifold: ℝ × S¹ × ℝ² (same as regular CartPole)
State: [x, θ, ẋ, θ̇] where θ will be wrapped to [-π, π]
"""
import torch
import numpy as np
import pickle
from pathlib import Path
from src.systems.base import DynamicalSystem, ManifoldComponent
from typing import List, Dict, Tuple, Optional


class CartPoleDMControlSystem(DynamicalSystem):
    """
    CartPole DeepMind Control Suite system with ℝ × S¹ × ℝ² manifold structure

    State representation: (x, θ, ẋ, θ̇) where:
    - x ∈ ℝ (cart position, normalized to [-1, 1])
    - θ ∈ S¹ (pole angle, circular, wrapped to [-π, π])
    - ẋ ∈ ℝ (cart velocity, normalized to [-1, 1])
    - θ̇ ∈ ℝ (pole angular velocity, normalized to [-1, 1])

    Key Differences from Regular CartPole:
    - Success: radius-based √(x² + θ² + ẋ² + θ̇²) < 0.25
    - NO failure/termination thresholds
    - Higher velocity bounds (from dm_control physics)
    """

    def __init__(self,
                 bounds_file: str = "/common/users/dm1487/arcmg_datasets/cartpole_dmcontrol/cartpole_dmcontrol_data_bounds.pkl",
                 use_dynamic_bounds: bool = True):
        """
        Initialize CartPole DM Control system

        Args:
            bounds_file: Path to pickle file containing actual data bounds
            use_dynamic_bounds: If True, load bounds from file; if False, use fallback defaults
        """
        if use_dynamic_bounds and Path(bounds_file).exists():
            self._load_bounds_from_file(bounds_file)
            print(f"✅ Loaded CartPoleDMControl bounds from: {bounds_file}")
        else:
            # Fallback to default bounds if file not found
            self._use_default_bounds()
            if use_dynamic_bounds:
                print(f"⚠️  Bounds file not found at {bounds_file}, using defaults")

        # Success criterion from dataset
        self.success_radius = 0.25  # √(x² + θ² + ẋ² + θ̇²) < 0.25

        super().__init__()

    def _load_bounds_from_file(self, bounds_file: str):
        """Load actual data bounds from pickle file"""
        with open(bounds_file, 'rb') as f:
            bounds_data = pickle.load(f)

        bounds = bounds_data['bounds']
        self.cart_limit = bounds[0]['limit']
        self.velocity_limit = bounds[2]['limit']
        self.angular_velocity_limit = bounds[3]['limit']
        # For angle, use π as limit (after wrapping to [-π, π])
        self.angle_limit = np.pi

        # Store the actual bounds for reference
        self.actual_bounds = bounds_data

        # Print in state vector order: [x, θ, ẋ, θ̇]
        print(f"  [0] Cart position (x): [{bounds[0]['min']:.3f}, {bounds[0]['max']:.3f}] -> limit: ±{self.cart_limit:.3f}")
        print(f"  [1] Pole angle (θ): WILL BE WRAPPED to ±π")
        print(f"  [2] Cart velocity (ẋ): [{bounds[2]['min']:.3f}, {bounds[2]['max']:.3f}] -> limit: ±{self.velocity_limit:.3f}")
        print(f"  [3] Angular velocity (θ̇): [{bounds[3]['min']:.3f}, {bounds[3]['max']:.3f}] -> limit: ±{self.angular_velocity_limit:.3f}")

    def _use_default_bounds(self):
        """Use default fallback bounds"""
        self.cart_limit = 2.4
        self.velocity_limit = 10.0
        self.angle_limit = np.pi
        self.angular_velocity_limit = 20.0
        self.actual_bounds = None
        print("⚠️  Using default CartPoleDMControl bounds (fallback mode)")

    def define_manifold_structure(self) -> List[ManifoldComponent]:
        """
        Define ℝ × S¹ × ℝ² manifold structure (same as regular CartPole):
        - Real component for cart position
        - SO2 component for pole angle θ
        - Real components for cart velocity and angular velocity
        """
        return [
            ManifoldComponent("Real", 1, "cart_position"),        # x ∈ ℝ
            ManifoldComponent("SO2", 1, "pole_angle"),            # θ ∈ S¹
            ManifoldComponent("Real", 1, "cart_velocity"),        # ẋ ∈ ℝ
            ManifoldComponent("Real", 1, "pole_angular_velocity") # θ̇ ∈ ℝ
        ]

    def define_state_bounds(self) -> Dict[str, Tuple[float, float]]:
        """
        Define state bounds for normalization using actual data bounds
        """
        return {
            "cart_position": (-self.cart_limit, self.cart_limit),
            "cart_velocity": (-self.velocity_limit, self.velocity_limit),
            "pole_angle": (-self.angle_limit, self.angle_limit),  # [-π, π] after wrapping
            "pole_angular_velocity": (-self.angular_velocity_limit, self.angular_velocity_limit)
        }

    def attractors(self) -> List[List[float]]:
        """
        CartPole DM Control attractor: upright balanced state at origin

        Returns:
            List of [x, θ, ẋ, θ̇] attractor positions
        """
        return [
            [0.0, 0.0, 0.0, 0.0],  # Cart centered, pole upright, no velocities
        ]

    def is_in_attractor(self, state, radius: float = None):
        """
        Check if states are within attractor basin using radius-based criterion

        Success: √(x² + θ² + ẋ² + θ̇²) < radius

        Args:
            state: States [B, 4] as (x, θ, ẋ, θ̇) - numpy array or torch tensor
            radius: Attractor radius (default: 0.25 from dataset)

        Returns:
            Boolean tensor [B] indicating attractor membership
        """
        # Use default success radius if not provided
        if radius is None:
            radius = self.success_radius

        # Convert to torch tensor if needed
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()

        if state.dim() == 1:
            state = state.unsqueeze(0)

        # Compute Euclidean distance from origin
        # distance = torch.norm(state, dim=1)
        # result = distance < radius
        
        x, theta, x_dot, theta_dot = state[:, 0], state[:, 1], state[:, 2], state[:, 3]
        position_ok = torch.abs(x) < radius
        velocity_ok = torch.abs(x_dot) < radius
        angular_velocity_ok = torch.abs(theta_dot) < radius
        angle_ok = torch.abs(theta) < radius
        result = position_ok & velocity_ok & angle_ok & angular_velocity_ok

        # Convert back to scalar if input was single state
        if len(result) == 1:
            return result.item()

        return result

    def classify_attractor(self, state: torch.Tensor, radius: float = None) -> torch.Tensor:
        """
        Classify CartPole DM Control states using radius-based criterion

        Binary classification:
        - SUCCESS (label=1): √(x² + θ² + ẋ² + θ̇²) < radius
        - NOT SUCCESS (label=0): √(x² + θ² + ẋ² + θ̇²) >= radius

        Args:
            state: States [B, 4] as (x, θ, ẋ, θ̇)
            radius: Attractor radius (default: 0.25 from dataset)

        Returns:
            Integer tensor [B] with:
                1: State in attractor (success)
                0: State not in attractor
        """
        # Use default success radius if not provided
        if radius is None:
            radius = self.success_radius

        # Convert to torch tensor if needed
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()

        if state.dim() == 1:
            state = state.unsqueeze(0)

        # Check if in attractor
        in_attractor = self.is_in_attractor(state, radius=radius)

        # Convert to integer labels (1=success, 0=not success)
        labels = in_attractor.long()

        return labels

    # ===================================================================
    # NORMALIZATION & EMBEDDING FOR FLOW MATCHING
    # ===================================================================

    def normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        """
        Normalize raw state coordinates (x, theta, x_dot, theta_dot) → (x_norm, theta, x_dot_norm, theta_dot_norm)

        Normalizes linear quantities to [-1, 1] using symmetric bounds.
        Angle remains UNCHANGED (already in natural [-π, π] range after wrapping).

        Args:
            state: [B, 4] raw cartpole state (theta already wrapped to [-π, π])

        Returns:
            [B, 4] normalized state (theta unchanged)
        """
        # Extract components
        x = state[:, 0]
        theta = state[:, 1]  # Keep as-is (already wrapped to [-π, π])
        x_dot = state[:, 2]
        theta_dot = state[:, 3]

        # Normalize linear quantities to [-1, 1] range using symmetric bounds
        x_norm = x / self.cart_limit
        x_dot_norm = x_dot / self.velocity_limit
        theta_dot_norm = theta_dot / self.angular_velocity_limit

        return torch.stack([x_norm, theta, x_dot_norm, theta_dot_norm], dim=1)

    def denormalize_state(self, normalized_state: torch.Tensor) -> torch.Tensor:
        """
        Denormalize state back to raw coordinates.

        Args:
            normalized_state: [B, 4] normalized state (x_norm, theta, x_dot_norm, theta_dot_norm)

        Returns:
            [B, 4] raw state (x, theta, x_dot, theta_dot)
        """
        x_norm = normalized_state[:, 0]
        theta = normalized_state[:, 1]  # Already in natural coordinates [-π, π]
        x_dot_norm = normalized_state[:, 2]
        theta_dot_norm = normalized_state[:, 3]

        # Denormalize using system bounds
        x = x_norm * self.cart_limit
        x_dot = x_dot_norm * self.velocity_limit
        theta_dot = theta_dot_norm * self.angular_velocity_limit

        return torch.stack([x, theta, x_dot, theta_dot], dim=1)

    def embed_state_for_model(self, normalized_state: torch.Tensor) -> torch.Tensor:
        """
        Embed normalized state → (x_norm, sin(theta), cos(theta), x_dot_norm, theta_dot_norm)

        Converts circular angle to sin/cos representation for neural network input.

        Args:
            normalized_state: [B, 4] normalized state

        Returns:
            [B, 5] embedded state
        """
        x_norm = normalized_state[:, 0]
        theta = normalized_state[:, 1]
        x_dot_norm = normalized_state[:, 2]
        theta_dot_norm = normalized_state[:, 3]

        # Embed circular angle as sin/cos
        sin_theta = torch.sin(theta)
        cos_theta = torch.cos(theta)

        return torch.stack([x_norm, sin_theta, cos_theta, x_dot_norm, theta_dot_norm], dim=1)

    def __repr__(self) -> str:
        return (
            f"CartPoleDMControlSystem(\n"
            f"  manifold=ℝ × S¹ × ℝ²,\n"
            f"  limits=[{self.cart_limit:.3f}, π, {self.velocity_limit:.3f}, {self.angular_velocity_limit:.3f}],\n"
            f"  success_radius={self.success_radius}\n"
            f")"
        )
