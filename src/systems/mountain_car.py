"""
Mountain Car system for Latent Conditional Flow Matching
"""
import torch
import numpy as np
import pickle
from pathlib import Path
from src.systems.base import DynamicalSystem, ManifoldComponent
from typing import List, Dict, Tuple


class MountainCarSystem(DynamicalSystem):
    """
    Mountain Car system with ℝ² manifold structure (pure Euclidean)

    State representation: (position, velocity) where:
    - position ∈ ℝ (car position on hill, normalized to [-1, 1])
    - velocity ∈ ℝ (car velocity, normalized to [-1, 1])

    Goal: Navigate the car to reach the goal region centered at x = π/6 ≈ 0.524
    """

    def __init__(self,
                 bounds_file: str = "/common/users/dm1487/arcmg_datasets/mountain_car/mountain_car_data_bounds.pkl",
                 use_dynamic_bounds: bool = True):
        """
        Initialize Mountain Car system

        Args:
            bounds_file: Path to pickle file containing actual data bounds
            use_dynamic_bounds: If True, load bounds from file; if False, use fallback defaults
        """

        print(f"Loading Mountain Car bounds from: {bounds_file}")
        print(f"Use dynamic bounds: {use_dynamic_bounds}")
        print(f"Path exists: {Path(bounds_file).exists()}")

        if use_dynamic_bounds and Path(bounds_file).exists():
            self._load_bounds_from_file(bounds_file)
            print(f"Loaded Mountain Car bounds from: {bounds_file}")
        else:
            # Fallback to default bounds if file not found
            self._use_default_bounds()
            if use_dynamic_bounds:
                print(f"Warning: Bounds file not found at {bounds_file}, using defaults")

        # Goal parameters (from dataset description)
        self.goal_center = np.pi / 6  # ≈ 0.524 rad
        self.position_threshold = 0.05
        self.velocity_threshold = 0.02

        super().__init__()

    def _load_bounds_from_file(self, bounds_file: str):
        """Load actual data bounds from pickle file"""
        with open(bounds_file, 'rb') as f:
            bounds_data = pickle.load(f)

        bounds = bounds_data['bounds']
        self.position_limit = max(abs(bounds['position']['min']), abs(bounds['position']['max']))
        self.velocity_limit = max(abs(bounds['velocity']['min']), abs(bounds['velocity']['max']))

        # Store the actual bounds for reference
        self.actual_bounds = bounds_data

        # Print in state vector order: [position, velocity]
        print(f"  [0] Position: [{bounds['position']['min']:.3f}, {bounds['position']['max']:.3f}] -> limit: ±{self.position_limit:.3f}")
        print(f"  [1] Velocity: [{bounds['velocity']['min']:.3f}, {bounds['velocity']['max']:.3f}] -> limit: ±{self.velocity_limit:.3f}")

    def _use_default_bounds(self):
        """Use default fallback bounds (from dataset description)"""
        self.position_limit = 2.0  # Position range: [-2, 1] → symmetric limit ±2
        self.velocity_limit = 0.1  # Velocity range: [-0.1, 0.1] → symmetric limit ±0.1
        self.actual_bounds = None
        print("Using default Mountain Car bounds (fallback mode)")
        print(f"  Position limit: ±{self.position_limit}")
        print(f"  Velocity limit: ±{self.velocity_limit}")

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

    def __repr__(self) -> str:
        return f"MountainCarSystem(ℝ², limits=[{self.position_limit}, {self.velocity_limit}], goal={self.goal_center:.3f})"
