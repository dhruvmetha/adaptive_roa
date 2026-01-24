"""
Mountain Car system for Latent Conditional Flow Matching
"""
import torch
import numpy as np
import json
from pathlib import Path
from adaptive_roa.systems.base import DynamicalSystem, ManifoldComponent
from adaptive_roa.utils.env_config import get_data_dir
from typing import List, Dict, Tuple


class MountainCarSystem(DynamicalSystem):
    """
    Mountain Car system with ℝ² manifold structure (pure Euclidean)

    State representation: (position, velocity) where:
    - position ∈ ℝ (car position on hill, normalized to [-1, 1])
    - velocity ∈ ℝ (car velocity, normalized to [-1, 1])

    Goal: Navigate the car to reach the goal region centered at x = π/6 ≈ 0.524
    """

    def __init__(self, dataset_dir: str = None):
        """
        Initialize Mountain Car system

        Args:
            dataset_dir: Path to dataset directory containing dataset_description.json.
                        If None, uses default path.
        """
        if dataset_dir is None:
            dataset_dir = f"{get_data_dir()}/mountain_car_power_0p001"

        dataset_dir = Path(dataset_dir)
        json_path = dataset_dir / "dataset_description.json"

        # Load bounds from JSON - no fallback, error if not found
        if not json_path.exists():
            raise FileNotFoundError(
                f"Cannot load bounds: dataset_description.json not found at {json_path}. "
                f"Achieved bounds are required for consistent normalization."
            )
        self._load_bounds_from_json(json_path)

        # Goal parameters (from dataset description)
        self.goal_center = np.pi / 6  # ≈ 0.524 rad
        self.position_threshold = 0.05
        self.velocity_threshold = 0.02

        super().__init__()

    def _load_bounds_from_json(self, json_path: Path):
        """Load actual data bounds from dataset_description.json"""
        with open(json_path) as f:
            dataset_info = json.load(f)

        bounds = dataset_info['achieved_bounds']

        # Support both old ('position'/'velocity') and new ('x'/'x_dot') key names
        pos_key = 'position' if 'position' in bounds else 'x'
        vel_key = 'velocity' if 'velocity' in bounds else 'x_dot'

        self.position_limit = max(abs(bounds[pos_key]['min']), abs(bounds[pos_key]['max']))
        self.velocity_limit = max(abs(bounds[vel_key]['min']), abs(bounds[vel_key]['max']))

        # Store the full dataset info for reference
        self.dataset_info = dataset_info
        self.achieved_bounds = bounds

        # Print in state vector order: [position, velocity]
        print(f"  [0] Position: [{bounds[pos_key]['min']:.3f}, {bounds[pos_key]['max']:.3f}] -> limit: ±{self.position_limit:.3f}")
        print(f"  [1] Velocity: [{bounds[vel_key]['min']:.3f}, {bounds[vel_key]['max']:.3f}] -> limit: ±{self.velocity_limit:.3f}")

    def _compute_bounds_from_trajectories(self, trajectories_dir: Path):
        """Compute bounds from trajectory files as fallback when JSON is not accessible"""
        from adaptive_roa.utils.bounds_from_trajectories import compute_mountain_car_bounds

        bounds_info = compute_mountain_car_bounds(trajectories_dir, max_files=1000)

        self.position_limit = bounds_info['position_limit']
        self.velocity_limit = bounds_info['velocity_limit']
        self.achieved_bounds = bounds_info['achieved_bounds']
        self.dataset_info = None  # Not available when computing from trajectories

    def define_manifold_structure(self) -> List[ManifoldComponent]:
        """
        Define ℝ² manifold structure (pure Euclidean):
        - Real component for position
        - Real component for velocity
        """
        return [
            ManifoldComponent("Real", 1, "position"),  # position ∈ ℝ
            ManifoldComponent("Real", 1, "velocity")   # velocity ∈ ℝ
        ]

    def define_state_bounds(self) -> Dict[str, Tuple[float, float]]:
        """
        Define state bounds for normalization using actual data bounds
        """
        return {
            "position": (-self.position_limit, self.position_limit),
            "velocity": (-self.velocity_limit, self.velocity_limit)
        }

    def attractors(self) -> List[List[float]]:
        """
        Mountain Car attractor positions in ℝ² space
        Target state: car at goal position (π/6) with zero velocity

        Returns:
            List of [position, velocity] attractor positions
        """
        return [
            [self.goal_center, 0.0],  # Goal region: x ≈ 0.524, v = 0
        ]

    def is_in_attractor(self, state, radius: float = None):
        """
        Check if states are within the goal attractor region

        Success condition: |position - π/6| < 0.05 AND |velocity| < 0.02

        Args:
            state: States [B, 2] as (position, velocity) - numpy array or torch tensor
            radius: Ignored (uses fixed thresholds from dataset)

        Returns:
            Boolean tensor [B] indicating attractor membership
        """
        # Convert to torch tensor if needed
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()

        if state.dim() == 1:
            state = state.unsqueeze(0)

        position, velocity = state[:, 0], state[:, 1]

        # Success criteria from dataset description
        position_ok = torch.abs(position - self.goal_center) < self.position_threshold
        velocity_ok = torch.abs(velocity) < self.velocity_threshold

        result = position_ok & velocity_ok

        if len(result) == 1:
            return result.item()

        return result

    def classify_attractor(self, state: torch.Tensor, radius: float = None) -> torch.Tensor:
        """
        Classify Mountain Car states into categories based on goal achievement

        Two-way classification:
        1. SUCCESS (label=1): In goal region (|pos - π/6| < 0.05 AND |vel| < 0.02)
        2. FAILURE (label=0): Outside goal region

        Args:
            state: States [B, 2] as (position, velocity)
            radius: Ignored (uses fixed thresholds)

        Returns:
            Integer tensor [B] with:
                1: State in goal attractor (SUCCESS)
                0: State outside goal region (FAILURE/SEPARATRIX)
        """
        # Convert to torch tensor if needed
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()

        if state.dim() == 1:
            state = state.unsqueeze(0)

        position, velocity = state[:, 0], state[:, 1]

        # SUCCESS: Check goal region constraints
        position_ok = torch.abs(position - self.goal_center) < self.position_threshold
        velocity_ok = torch.abs(velocity) < self.velocity_threshold

        in_attractor = position_ok & velocity_ok

        # Binary classification: SUCCESS (1) or FAILURE (-1)
        labels = torch.ones_like(in_attractor, dtype=torch.long) * -1
        labels[in_attractor] = 1

        return labels

    # ===================================================================
    # NORMALIZATION & EMBEDDING FOR FLOW MATCHING
    # ===================================================================

    def normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        """
        Normalize raw state coordinates (position, velocity) → (pos_norm, vel_norm)

        Normalizes both quantities to roughly [-1, 1] using symmetric bounds.

        Args:
            state: [B, 2] raw mountain car state

        Returns:
            [B, 2] normalized state
        """
        # Extract components
        position = state[:, 0]
        velocity = state[:, 1]

        # Normalize both dimensions to [-1, 1] range using symmetric bounds
        position_norm = position / self.position_limit
        velocity_norm = velocity / self.velocity_limit

        return torch.stack([position_norm, velocity_norm], dim=1)

    def denormalize_state(self, normalized_state: torch.Tensor) -> torch.Tensor:
        """
        Denormalize state back to raw coordinates.

        Args:
            normalized_state: [B, 2] normalized state (pos_norm, vel_norm)

        Returns:
            [B, 2] raw state (position, velocity)
        """
        position_norm = normalized_state[:, 0]
        velocity_norm = normalized_state[:, 1]

        # Denormalize using system bounds
        position = position_norm * self.position_limit
        velocity = velocity_norm * self.velocity_limit

        return torch.stack([position, velocity], dim=1)

    def embed_state_for_model(self, normalized_state: torch.Tensor) -> torch.Tensor:
        """
        Embed normalized state for neural network input.

        For pure Euclidean manifold (ℝ²), embedding is IDENTITY (no change).
        Unlike CartPole's SO2 angle (which needs sin/cos), Euclidean dimensions
        stay as-is.

        Args:
            normalized_state: [B, 2] normalized state

        Returns:
            [B, 2] embedded state (UNCHANGED for Euclidean)
        """
        # Pure Euclidean: no embedding transformation needed
        return normalized_state

    def get_loss_weights(self) -> torch.Tensor:
        """
        Get per-dimension loss weights for Mountain Car (2D state).

        State: (position, velocity) - both Euclidean
        - position → weight = position_limit
        - velocity → weight = velocity_limit

        Returns:
            torch.Tensor: 2D weights
        """
        weights = [
            self.position_limit,  # position
            self.velocity_limit   # velocity
        ]
        return torch.tensor(weights, dtype=torch.float32)

    def __repr__(self) -> str:
        return f"MountainCarSystem(ℝ², limits=[{self.position_limit}, {self.velocity_limit}], goal={self.goal_center:.3f})"
