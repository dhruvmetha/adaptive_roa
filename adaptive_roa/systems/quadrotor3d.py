"""
Quadrotor 3D system for Latent Conditional Flow Matching

Manifold: ℝ³ × SO(3) × ℝ⁶ (13-dimensional state with unit quaternion representation)
"""
import torch
import numpy as np
import json
from pathlib import Path
from adaptive_roa.systems.base import DynamicalSystem, ManifoldComponent
from adaptive_roa.utils.env_config import get_shared_data_base
from typing import List, Dict, Tuple


class Quadrotor3DSystem(DynamicalSystem):
    """
    Quadrotor 3D system with ℝ³ × SO(3) × ℝ⁶ manifold structure

    State representation: (x, y, z, qw, qx, qy, qz, ẋ, ẏ, ż, p, q, r) where:
    - (x, y, z) ∈ ℝ³ (position in world frame)
    - (qw, qx, qy, qz) ∈ SO(3) via unit quaternion (scalar-first, canonicalized qw >= 0)
    - (ẋ, ẏ, ż) ∈ ℝ³ (linear velocity in world frame)
    - (p, q, r) ∈ ℝ³ (angular velocity in body frame)

    Goal: Hover at (0, 0, 1) with identity orientation (qw=1, qx=qy=qz=0)
    """

    def __init__(self, dataset_dir: str = None):
        """
        Initialize Quadrotor 3D system

        Args:
            dataset_dir: Path to dataset directory containing dataset_description.json.
                        If None, uses default path from environment.
        """
        if dataset_dir is None:
            dataset_dir = f"{get_shared_data_base()}/quadrotor3D_lqr"

        dataset_dir = Path(dataset_dir)
        json_path = dataset_dir / "dataset_description.json"

        # Try loading from JSON
        if json_path.exists():
            self._load_bounds_from_json(json_path)
        else:
            raise FileNotFoundError(
                f"Cannot load bounds: dataset_description.json not found at {json_path}"
            )

        # Goal state: hover at (0, 0, 1) with identity orientation
        self.goal_position = np.array([0.0, 0.0, 1.0])
        self.goal_quaternion = np.array([1.0, 0.0, 0.0, 0.0])  # Identity (qw, qx, qy, qz)
        self.goal_linear_velocity = np.array([0.0, 0.0, 0.0])
        self.goal_angular_velocity = np.array([0.0, 0.0, 0.0])

        # Success threshold (Euclidean distance in state space)
        self.success_threshold = 0.05

        super().__init__()

    def _load_bounds_from_json(self, json_path: Path):
        """Load actual data bounds from dataset_description.json"""
        with open(json_path) as f:
            dataset_info = json.load(f)

        achieved = dataset_info['achieved_bounds']

        # Position bounds (x, y, z)
        self.x_limit = max(abs(achieved['x']['min']), abs(achieved['x']['max']))
        self.y_limit = max(abs(achieved['y']['min']), abs(achieved['y']['max']))
        self.z_min = achieved['z']['min']
        self.z_max = achieved['z']['max']

        # Linear velocity bounds
        self.x_dot_limit = max(abs(achieved['x_dot']['min']), abs(achieved['x_dot']['max']))
        self.y_dot_limit = max(abs(achieved['y_dot']['min']), abs(achieved['y_dot']['max']))
        self.z_dot_limit = max(abs(achieved['z_dot']['min']), abs(achieved['z_dot']['max']))

        # Angular velocity bounds (can be larger due to tumbling)
        self.p_limit = max(abs(achieved['p']['min']), abs(achieved['p']['max']))
        self.q_limit = max(abs(achieved['q']['min']), abs(achieved['q']['max']))
        self.r_limit = max(abs(achieved['r']['min']), abs(achieved['r']['max']))

        # Store full info
        self.dataset_info = dataset_info
        self.achieved_bounds = achieved

        print(f"Quadrotor3D bounds loaded from {json_path}:")
        print(f"  Position (x, y): ±{self.x_limit:.3f}, ±{self.y_limit:.3f}")
        print(f"  Position (z): [{self.z_min:.3f}, {self.z_max:.3f}]")
        print(f"  Linear velocity: ±{self.x_dot_limit:.3f}, ±{self.y_dot_limit:.3f}, ±{self.z_dot_limit:.3f}")
        print(f"  Angular velocity: ±{self.p_limit:.3f}, ±{self.q_limit:.3f}, ±{self.r_limit:.3f}")

    def define_manifold_structure(self) -> List[ManifoldComponent]:
        """
        Define ℝ³ × SO(3) × ℝ⁶ manifold structure:
        - Position (x, y, z): Euclidean ℝ³
        - Orientation (qw, qx, qy, qz): SO(3) via unit quaternion
        - Linear velocity (ẋ, ẏ, ż): Euclidean ℝ³
        - Angular velocity (p, q, r): Euclidean ℝ³
        """
        return [
            ManifoldComponent("Real", 3, "position"),          # (x, y, z) ∈ ℝ³
            ManifoldComponent("SO3", 4, "orientation"),        # (qw, qx, qy, qz) ∈ SO(3)
            ManifoldComponent("Real", 3, "linear_velocity"),   # (ẋ, ẏ, ż) ∈ ℝ³
            ManifoldComponent("Real", 3, "angular_velocity"),  # (p, q, r) ∈ ℝ³
        ]

    def define_state_bounds(self) -> Dict[str, Tuple[float, float]]:
        """
        Define state bounds for normalization using actual data bounds
        """
        return {
            "position": (-self.x_limit, self.x_limit),  # Symmetric for x, y
            "orientation": (-1.0, 1.0),  # Quaternion components are in [-1, 1]
            "linear_velocity": (-self.x_dot_limit, self.x_dot_limit),
            "angular_velocity": (-self.p_limit, self.p_limit),
        }

    def attractors(self) -> List[List[float]]:
        """
        Quadrotor 3D attractor positions
        Target state: hover at (0, 0, 1) with identity orientation and zero velocities

        Returns:
            List of 13D attractor states
        """
        return [
            [0.0, 0.0, 1.0,    # Position (x, y, z)
             1.0, 0.0, 0.0, 0.0,  # Quaternion (qw, qx, qy, qz) - identity
             0.0, 0.0, 0.0,    # Linear velocity
             0.0, 0.0, 0.0]    # Angular velocity
        ]

    def is_in_attractor(self, state, radius: float = 0.05):
        """
        Check if states are within attractor basin (hovering at goal)

        Args:
            state: States [B, 13] - numpy array or torch tensor
            radius: Attractor radius (Euclidean distance threshold)

        Returns:
            Boolean tensor [B] indicating attractor membership
        """
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()

        if state.dim() == 1:
            state = state.unsqueeze(0)

        # Goal state
        goal = torch.tensor(self.attractors()[0], device=state.device, dtype=state.dtype)

        # Compute Euclidean distance in full state space
        # Note: For quaternions, this is an approximation; proper SO(3) distance would be geodesic
        dist = torch.norm(state - goal.unsqueeze(0), dim=1)

        result = dist < radius

        if len(result) == 1:
            return result.item()

        return result

    def classify_attractor(self, state: torch.Tensor, radius: float = 0.05) -> torch.Tensor:
        """
        Classify Quadrotor 3D states into success/failure categories

        Two-way classification:
        1. SUCCESS (label=1): Within radius of goal state
        2. FAILURE (label=-1): Outside goal region

        Args:
            state: States [B, 13]
            radius: Attractor radius (default 0.05 from dataset)

        Returns:
            Integer tensor [B] with:
                 1: State in goal attractor (SUCCESS)
                -1: State outside goal region (FAILURE)
        """
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()

        if state.dim() == 1:
            state = state.unsqueeze(0)

        # Goal state
        goal = torch.tensor(self.attractors()[0], device=state.device, dtype=state.dtype)

        # Compute Euclidean distance
        dist = torch.norm(state - goal.unsqueeze(0), dim=1)

        in_attractor = dist < radius

        # Binary classification
        labels = torch.ones_like(in_attractor, dtype=torch.long) * -1
        labels[in_attractor] = 1

        return labels

    # ===================================================================
    # NORMALIZATION & EMBEDDING FOR FLOW MATCHING
    # ===================================================================

    def normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        """
        Normalize raw state coordinates for neural network input.

        Position and velocities are normalized to approximately [-1, 1].
        Quaternion is kept as-is (already unit norm).

        Args:
            state: [B, 13] raw state

        Returns:
            [B, 13] normalized state
        """
        # Extract components
        pos = state[:, 0:3]      # (x, y, z)
        quat = state[:, 3:7]     # (qw, qx, qy, qz)
        lin_vel = state[:, 7:10] # (ẋ, ẏ, ż)
        ang_vel = state[:, 10:13] # (p, q, r)

        # Normalize position (x, y symmetric, z shifted to center around 1.55)
        x_norm = pos[:, 0:1] / self.x_limit
        y_norm = pos[:, 1:2] / self.y_limit
        # z is [0.1, 3.0], center is ~1.55, normalize to roughly [-1, 1]
        z_center = (self.z_min + self.z_max) / 2
        z_scale = (self.z_max - self.z_min) / 2
        z_norm = (pos[:, 2:3] - z_center) / z_scale
        pos_norm = torch.cat([x_norm, y_norm, z_norm], dim=1)

        # Quaternion: keep as-is (already normalized to unit sphere)
        quat_norm = quat

        # Normalize linear velocity
        lin_vel_norm = torch.stack([
            lin_vel[:, 0] / self.x_dot_limit,
            lin_vel[:, 1] / self.y_dot_limit,
            lin_vel[:, 2] / self.z_dot_limit,
        ], dim=1)

        # Normalize angular velocity
        ang_vel_norm = torch.stack([
            ang_vel[:, 0] / self.p_limit,
            ang_vel[:, 1] / self.q_limit,
            ang_vel[:, 2] / self.r_limit,
        ], dim=1)

        return torch.cat([pos_norm, quat_norm, lin_vel_norm, ang_vel_norm], dim=1)

    def denormalize_state(self, normalized_state: torch.Tensor) -> torch.Tensor:
        """
        Denormalize state back to raw coordinates.

        Args:
            normalized_state: [B, 13] normalized state

        Returns:
            [B, 13] raw state
        """
        # Extract components
        pos_norm = normalized_state[:, 0:3]
        quat_norm = normalized_state[:, 3:7]
        lin_vel_norm = normalized_state[:, 7:10]
        ang_vel_norm = normalized_state[:, 10:13]

        # Denormalize position
        x = pos_norm[:, 0:1] * self.x_limit
        y = pos_norm[:, 1:2] * self.y_limit
        z_center = (self.z_min + self.z_max) / 2
        z_scale = (self.z_max - self.z_min) / 2
        z = pos_norm[:, 2:3] * z_scale + z_center
        pos = torch.cat([x, y, z], dim=1)

        # Quaternion: normalize to ensure unit norm
        quat = quat_norm / torch.norm(quat_norm, dim=1, keepdim=True).clamp(min=1e-8)

        # Denormalize linear velocity
        lin_vel = torch.stack([
            lin_vel_norm[:, 0] * self.x_dot_limit,
            lin_vel_norm[:, 1] * self.y_dot_limit,
            lin_vel_norm[:, 2] * self.z_dot_limit,
        ], dim=1)

        # Denormalize angular velocity
        ang_vel = torch.stack([
            ang_vel_norm[:, 0] * self.p_limit,
            ang_vel_norm[:, 1] * self.q_limit,
            ang_vel_norm[:, 2] * self.r_limit,
        ], dim=1)

        return torch.cat([pos, quat, lin_vel, ang_vel], dim=1)

    def embed_state_for_model(self, normalized_state: torch.Tensor) -> torch.Tensor:
        """
        Embed normalized state for neural network input.

        For Quadrotor 3D, the state is already in a good representation:
        - Position (3D): Euclidean, no transformation needed
        - Quaternion (4D): Already unit norm, no transformation needed
        - Velocities (6D): Euclidean, no transformation needed

        Total: 13D → 13D (identity embedding)

        Args:
            normalized_state: [B, 13] normalized state

        Returns:
            [B, 13] embedded state (identity for this system)
        """
        # For SO(3) represented as quaternions, we keep the quaternion as-is
        # The flow matching manifold will handle the geodesic interpolation
        return normalized_state

    def canonicalize_quaternion(self, quat: torch.Tensor) -> torch.Tensor:
        """
        Canonicalize quaternion to ensure qw >= 0 (removes double-cover ambiguity)

        Args:
            quat: Quaternion [B, 4] as (qw, qx, qy, qz)

        Returns:
            Canonicalized quaternion [B, 4] with qw >= 0
        """
        # Flip sign if qw < 0
        sign = torch.sign(quat[:, 0:1])
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        return quat * sign

    def project_to_manifold(self, state: torch.Tensor) -> torch.Tensor:
        """
        Project state onto the manifold (normalize quaternion to unit norm)

        Args:
            state: [B, 13] state (possibly with unnormalized quaternion)

        Returns:
            [B, 13] state with unit quaternion
        """
        pos = state[:, 0:3]
        quat = state[:, 3:7]
        vel = state[:, 7:13]

        # Normalize and canonicalize quaternion
        quat_norm = quat / torch.norm(quat, dim=1, keepdim=True).clamp(min=1e-8)
        quat_canon = self.canonicalize_quaternion(quat_norm)

        return torch.cat([pos, quat_canon, vel], dim=1)

    def __repr__(self) -> str:
        return (f"Quadrotor3DSystem(ℝ³ × SO(3) × ℝ⁶, "
                f"pos_limits=[±{self.x_limit:.2f}, ±{self.y_limit:.2f}, {self.z_min:.2f}-{self.z_max:.2f}], "
                f"goal=[0,0,1])")
