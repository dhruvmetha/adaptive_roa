"""
Pendulum Cartesian system for Latent Conditional Flow Matching
"""
import torch
import numpy as np
import json
from pathlib import Path
from adaptive_roa.systems.base import DynamicalSystem, ManifoldComponent
from adaptive_roa.utils.env_config import get_data_dir
from typing import List, Dict, Tuple


class PendulumCartesianSystem(DynamicalSystem):
    """
    Pendulum Cartesian system with ℝ⁴ manifold structure (pure Euclidean)

    State representation: (x, y, vx, vy) where:
    - x ∈ ℝ (x-position of pendulum bob, normalized to [-1, 1])
    - y ∈ ℝ (y-position of pendulum bob, normalized to [-1, 1])
    - vx ∈ ℝ (x-velocity of pendulum bob)
    - vy ∈ ℝ (y-velocity of pendulum bob)

    Note: Unlike the polar pendulum (θ, θ̇), this uses Cartesian coordinates.
    The position (x, y) lies on a unit circle, but we represent it as Euclidean
    dimensions rather than using a Sphere manifold.

    Goal: Pendulum swinging up to the upright position (0, 1) with zero velocity
    """

    def __init__(self, dataset_dir: str = None):
        """
        Initialize Pendulum Cartesian system

        Args:
            dataset_dir: Path to dataset directory containing dataset_description.json.
                        If None, uses default path.
        """
        if dataset_dir is None:
            dataset_dir = f"{get_data_dir()}/pendulum_cartesian_50k"

        dataset_dir = Path(dataset_dir)
        json_path = dataset_dir / "dataset_description.json"

        # Load bounds from JSON - no fallback, error if not found
        if not json_path.exists():
            raise FileNotFoundError(
                f"Cannot load bounds: dataset_description.json not found at {json_path}. "
                f"Achieved bounds are required for consistent normalization."
            )
        self._load_bounds_from_json(json_path)

        # Goal parameters (upright position)
        self.goal_position = np.array([0.0, 1.0, 0.0, 0.0])  # Top of circle (upright)
        self.position_threshold = 0.1
        self.velocity_threshold = 0.1

        super().__init__()

    def _load_bounds_from_json(self, json_path: Path):
        """Load actual data bounds from dataset_description.json"""
        with open(json_path) as f:
            dataset_info = json.load(f)

        bounds = dataset_info['achieved_bounds']

        self.x_limit = max(abs(bounds['x']['min']), abs(bounds['x']['max']))
        self.y_limit = max(abs(bounds['y']['min']), abs(bounds['y']['max']))
        self.vx_limit = max(abs(bounds['x_dot']['min']), abs(bounds['x_dot']['max']))
        self.vy_limit = max(abs(bounds['y_dot']['min']), abs(bounds['y_dot']['max']))

        # Store the full dataset info for reference
        self.dataset_info = dataset_info
        self.achieved_bounds = bounds

        # Print in state vector order: [x, y, vx, vy]
        print(f"  [0] X Position: [{bounds['x']['min']:.3f}, {bounds['x']['max']:.3f}] -> limit: ±{self.x_limit:.3f}")
        print(f"  [1] Y Position: [{bounds['y']['min']:.3f}, {bounds['y']['max']:.3f}] -> limit: ±{self.y_limit:.3f}")
        print(f"  [2] X Velocity: [{bounds['x_dot']['min']:.3f}, {bounds['x_dot']['max']:.3f}] -> limit: ±{self.vx_limit:.3f}")
        print(f"  [3] Y Velocity: [{bounds['y_dot']['min']:.3f}, {bounds['y_dot']['max']:.3f}] -> limit: ±{self.vy_limit:.3f}")

    def _compute_bounds_from_trajectories(self, trajectories_dir: Path):
        """Compute bounds from trajectory files as fallback when JSON is not accessible"""
        from adaptive_roa.utils.bounds_from_trajectories import compute_pendulum_cartesian_bounds

        bounds_info = compute_pendulum_cartesian_bounds(trajectories_dir, max_files=1000)

        self.x_limit = bounds_info['x_limit']
        self.y_limit = bounds_info['y_limit']
        self.vx_limit = bounds_info['vx_limit']
        self.vy_limit = bounds_info['vy_limit']
        self.achieved_bounds = bounds_info['achieved_bounds']
        self.dataset_info = None  # Not available when computing from trajectories

    def define_manifold_structure(self) -> List[ManifoldComponent]:
        """
        Define ℝ⁴ manifold structure (pure Euclidean):
        - Real component for x position
        - Real component for y position
        - Real component for x velocity
        - Real component for y velocity
        """
        return [
            ManifoldComponent("Real", 1, "x_position"),   # x ∈ ℝ
            ManifoldComponent("Real", 1, "y_position"),   # y ∈ ℝ
            ManifoldComponent("Real", 1, "x_velocity"),   # vx ∈ ℝ
            ManifoldComponent("Real", 1, "y_velocity")    # vy ∈ ℝ
        ]

    def define_state_bounds(self) -> Dict[str, Tuple[float, float]]:
        """
        Define state bounds for normalization using actual data bounds
        """
        return {
            "x_position": (-self.x_limit, self.x_limit),
            "y_position": (-self.y_limit, self.y_limit),
            "x_velocity": (-self.vx_limit, self.vx_limit),
            "y_velocity": (-self.vy_limit, self.vy_limit)
        }

    def attractors(self) -> List[List[float]]:
        """
        Pendulum Cartesian attractor positions in ℝ⁴ space
        Target state: pendulum upright at (0, 1) with zero velocity

        Returns:
            List of [x, y, vx, vy] attractor positions
        """
        return [
            [0.0, 1.0, 0.0, 0.0],  # Upright position: (0, 1) with no velocity
        ]

    def is_in_attractor(self, state, radius: float = None):
        """
        Check if states are within the upright attractor region

        Success condition: position near (0, 1) AND velocities near zero

        Args:
            state: States [B, 4] as (x, y, vx, vy) - numpy array or torch tensor
            radius: Ignored (uses fixed thresholds from dataset)

        Returns:
            Boolean tensor [B] indicating attractor membership
        """
        # Convert to torch tensor if needed
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()

        if state.dim() == 1:
            state = state.unsqueeze(0)

        x, y, vx, vy = state[:, 0], state[:, 1], state[:, 2], state[:, 3]

        # Success criteria: near upright position (0, 1) with low velocity
        position_error = torch.sqrt((x - self.goal_position[0])**2 + (y - self.goal_position[1])**2)
        position_ok = position_error < self.position_threshold

        velocity_magnitude = torch.sqrt(vx**2 + vy**2)
        velocity_ok = velocity_magnitude < self.velocity_threshold

        result = position_ok & velocity_ok

        if len(result) == 1:
            return result.item()

        return result

    def classify_attractor(self, state: torch.Tensor, radius: float = None) -> torch.Tensor:
        """
        Classify Pendulum Cartesian states into categories

        Two-way classification:
        1. SUCCESS (label=1): In upright attractor (near (0,1) with low velocity)
        2. FAILURE (label=-1): Outside attractor region

        Args:
            state: States [B, 4] as (x, y, vx, vy)
            radius: Ignored (uses fixed thresholds)

        Returns:
            Integer tensor [B] with:
                1: State in upright attractor (SUCCESS)
                -1: State outside attractor (FAILURE)
        """
        # Convert to torch tensor if neededs
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()

        if state.dim() == 1:
            state = state.unsqueeze(0)

        
        # print(torch.norm(state, dim=1))
        in_attractor = torch.norm(state - torch.tensor(self.goal_position, device=state.device), dim=1) < radius
        
        
        # Binary classification: SUCCESS (1) or FAILURE (-1)
        labels = torch.ones_like(in_attractor, dtype=torch.long) * -1
        labels[in_attractor] = 1.0

        return labels

    # ===================================================================
    # NORMALIZATION & EMBEDDING FOR FLOW MATCHING
    # ===================================================================

    def normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        """
        Normalize raw state coordinates (x, y, vx, vy) → (x_norm, y_norm, vx_norm, vy_norm)

        Normalizes all quantities to roughly [-1, 1] using symmetric bounds.

        Args:
            state: [B, 4] raw pendulum cartesian state

        Returns:
            [B, 4] normalized state
        """
        # Extract components
        x = state[:, 0]
        y = state[:, 1]
        vx = state[:, 2]
        vy = state[:, 3]

        # Normalize all dimensions to [-1, 1] range using symmetric bounds
        x_norm = x / self.x_limit
        y_norm = y / self.y_limit
        vx_norm = vx / self.vx_limit
        vy_norm = vy / self.vy_limit

        return torch.stack([x_norm, y_norm, vx_norm, vy_norm], dim=1)

    def denormalize_state(self, normalized_state: torch.Tensor) -> torch.Tensor:
        """
        Denormalize state back to raw coordinates.

        Args:
            normalized_state: [B, 4] normalized state (x_norm, y_norm, vx_norm, vy_norm)

        Returns:
            [B, 4] raw state (x, y, vx, vy)
        """
        x_norm = normalized_state[:, 0]
        y_norm = normalized_state[:, 1]
        vx_norm = normalized_state[:, 2]
        vy_norm = normalized_state[:, 3]

        # Denormalize using system bounds
        x = x_norm * self.x_limit
        y = y_norm * self.y_limit
        vx = vx_norm * self.vx_limit
        vy = vy_norm * self.vy_limit

        return torch.stack([x, y, vx, vy], dim=1)

    def embed_state_for_model(self, normalized_state: torch.Tensor) -> torch.Tensor:
        """
        Embed normalized state for neural network input.

        For pure Euclidean manifold (ℝ⁴), embedding is IDENTITY (no change).
        Unlike systems with SO2 angles (which need sin/cos), Euclidean dimensions
        stay as-is.

        Args:
            normalized_state: [B, 4] normalized state

        Returns:
            [B, 4] embedded state (UNCHANGED for Euclidean)
        """
        # Pure Euclidean: no embedding transformation needed
        return normalized_state

    def get_loss_weights(self) -> torch.Tensor:
        """
        Get per-dimension loss weights for Pendulum Cartesian (4D state).

        State: (x, y, vx, vy) - all Euclidean
        - x → weight = x_limit
        - y → weight = y_limit
        - vx → weight = vx_limit
        - vy → weight = vy_limit

        Returns:
            torch.Tensor: 4D weights
        """
        weights = [
            self.x_limit,   # x position
            self.y_limit,   # y position
            self.vx_limit,  # x velocity
            self.vy_limit   # y velocity
        ]
        return torch.tensor(weights, dtype=torch.float32)

    def __repr__(self) -> str:
        return f"PendulumCartesianSystem(ℝ⁴, limits=[{self.x_limit}, {self.y_limit}, {self.vx_limit:.1f}, {self.vy_limit:.1f}], goal=(0, 1))"
