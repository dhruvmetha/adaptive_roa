"""
Pendulum Cartesian system for Latent Conditional Flow Matching

State representation: (x, y, ẋ, ẏ) in Cartesian coordinates
- Position (x, y) is constrained to unit circle: x² + y² = 1
- All dimensions are Euclidean (no circular SO2 components)
"""
import torch
import numpy as np
import pickle
from pathlib import Path
from src.systems.base import DynamicalSystem, ManifoldComponent
from typing import List, Dict, Tuple


class PendulumCartesianSystem(DynamicalSystem):
    """
    Pendulum in Cartesian coordinates with ℝ⁴ manifold structure

    State representation: (x, y, ẋ, ẏ) where:
    - x, y ∈ ℝ (position on unit circle, constrained: x² + y² = 1)
    - ẋ, ẏ ∈ ℝ (velocity components, normalized)

    Unlike the angular pendulum (S¹×ℝ), this uses pure Euclidean manifold.
    """

    def __init__(self,
                 bounds_file: str = "/common/users/dm1487/arcmg_datasets/pendulum_cartesian/pendulum_cartesian_data_bounds.pkl",
                 use_dynamic_bounds: bool = True):
        """
        Initialize Pendulum Cartesian system

        Args:
            bounds_file: Path to pickle file containing actual data bounds
            use_dynamic_bounds: If True, load bounds from file; if False, use fallback defaults
        """
        if use_dynamic_bounds and Path(bounds_file).exists():
            self._load_bounds_from_file(bounds_file)
            print(f"Loaded Pendulum Cartesian bounds from: {bounds_file}")
        else:
            # Fallback to default bounds if file not found
            self._use_default_bounds()
            if use_dynamic_bounds:
                print(f"Warning: Bounds file not found at {bounds_file}, using defaults")

        super().__init__()

    def _load_bounds_from_file(self, bounds_file: str):
        """Load actual data bounds from pickle file"""
        with open(bounds_file, 'rb') as f:
            bounds_data = pickle.load(f)

        bounds = bounds_data['bounds']

        # Per-dimension limits (symmetric)
        self.x_limit = max(abs(bounds['x']['min']), abs(bounds['x']['max']))
        self.y_limit = max(abs(bounds['y']['min']), abs(bounds['y']['max']))
        self.x_dot_limit = max(abs(bounds['x_dot']['min']), abs(bounds['x_dot']['max']))
        self.y_dot_limit = max(abs(bounds['y_dot']['min']), abs(bounds['y_dot']['max']))

        # Store the actual bounds for reference
        self.actual_bounds = bounds_data

        # Print in state vector order: [x, y, ẋ, ẏ]
        print(f"  [0] Position x: [{bounds['x']['min']:.3f}, {bounds['x']['max']:.3f}] -> limit: ±{self.x_limit:.3f}")
        print(f"  [1] Position y: [{bounds['y']['min']:.3f}, {bounds['y']['max']:.3f}] -> limit: ±{self.y_limit:.3f}")
        print(f"  [2] Velocity ẋ: [{bounds['x_dot']['min']:.3f}, {bounds['x_dot']['max']:.3f}] -> limit: ±{self.x_dot_limit:.3f}")
        print(f"  [3] Velocity ẏ: [{bounds['y_dot']['min']:.3f}, {bounds['y_dot']['max']:.3f}] -> limit: ±{self.y_dot_limit:.3f}")

        # Verify position constraint
        stats = bounds_data['statistics']
        print(f"  Position norm: {stats['position_norm_mean']:.6f} ± {stats['position_norm_std']:.6f} (should be ≈1.0)")

    def _use_default_bounds(self):
        """Use default fallback bounds"""
        # Position on unit circle
        self.x_limit = 1.0
        self.y_limit = 1.0
        # Velocities approximately ±2π
        self.x_dot_limit = 2 * np.pi
        self.y_dot_limit = 2 * np.pi
        self.actual_bounds = None
        print("Using default Pendulum Cartesian bounds (fallback mode)")

    def define_manifold_structure(self) -> List[ManifoldComponent]:
        """
        Define ℝ⁴ manifold structure (all Euclidean):
        - Real component for x position
        - Real component for y position
        - Real component for ẋ velocity
        - Real component for ẏ velocity

        Note: Unlike angular pendulum, NO SO2 components (already in Cartesian form)
        """
        return [
            ManifoldComponent("Real", 1, "position_x"),   # x ∈ ℝ
            ManifoldComponent("Real", 1, "position_y"),   # y ∈ ℝ
            ManifoldComponent("Real", 1, "velocity_x"),   # ẋ ∈ ℝ
            ManifoldComponent("Real", 1, "velocity_y")    # ẏ ∈ ℝ
        ]

    def define_state_bounds(self) -> Dict[str, Tuple[float, float]]:
        """
        Define state bounds for normalization using actual data bounds
        """
        return {
            "position_x": (-self.x_limit, self.x_limit),
            "position_y": (-self.y_limit, self.y_limit),
            "velocity_x": (-self.x_dot_limit, self.x_dot_limit),
            "velocity_y": (-self.y_dot_limit, self.y_dot_limit)
        }

    def attractors(self) -> List[List[float]]:
        """
        Pendulum attractor positions in Cartesian space

        For a pendulum:
        - Bottom position (stable): (x=0, y=-1) with zero velocity
        - Top position (unstable): (x=0, y=1) with zero velocity

        Returns:
            List of [x, y, ẋ, ẏ] attractor positions
        """
        return [
            [0.0, -1.0, 0.0, 0.0],   # Bottom equilibrium (stable, SUCCESS)
            [0.0, 1.0, 0.0, 0.0],    # Top equilibrium (unstable, FAILURE)
        ]

    def is_in_attractor(self, state: torch.Tensor, radius: float = 0.15) -> torch.Tensor:
        """
        Check if states are within attractor basins using Euclidean distance

        Args:
            state: States [B, 4] or [4] as (x, y, ẋ, ẏ) - can be numpy array or torch tensor
            radius: Attractor radius

        Returns:
            Boolean tensor [B] or bool indicating attractor membership
        """
        # Convert numpy to torch if needed
        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float32)

        # Handle single state [4] -> [1, 4]
        if state.ndim == 1:
            state = state.unsqueeze(0)
            single_state = True
        else:
            single_state = False

        attractors = torch.tensor(self.attractors(), device=state.device, dtype=state.dtype)

        # Compute Euclidean distance in 4D space
        # state: [B, 4], attractors: [2, 4]
        state_expanded = state.unsqueeze(1)  # [B, 1, 4]
        attractors_expanded = attractors.unsqueeze(0)  # [1, 2, 4]

        # Euclidean distance
        distances = torch.sqrt(((state_expanded - attractors_expanded) ** 2).sum(dim=2))  # [B, 2]

        # Check if any attractor is within radius
        result = (distances < radius).any(dim=1)  # [B]

        # Return scalar bool for single state
        if single_state:
            return result.item()
        return result

    def classify_attractor(self, state: torch.Tensor, radius: float = 0.15) -> torch.Tensor:
        """
        Classify which attractor (if any) each state belongs to

        Args:
            state: States [B, 4] or [4] as (x, y, ẋ, ẏ) - can be numpy array or torch tensor
            radius: Attractor radius

        Returns:
            Integer tensor [B] or int with:
                1: State in stable bottom attractor [0, -1, 0, 0] (SUCCESS)
               -1: State in unstable top attractor [0, 1, 0, 0] (FAILURE)
                0: State in none of the attractors (SEPARATRIX)
        """
        # Convert numpy to torch if needed
        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float32)

        # Handle single state [4] -> [1, 4]
        if state.ndim == 1:
            state = state.unsqueeze(0)
            single_state = True
        else:
            single_state = False

        attractors = torch.tensor(self.attractors(), device=state.device, dtype=state.dtype)

        # Compute Euclidean distance in 4D space
        state_expanded = state.unsqueeze(1)  # [B, 1, 4]
        attractors_expanded = attractors.unsqueeze(0)  # [1, 2, 4]

        # Euclidean distance
        distances = torch.sqrt(((state_expanded - attractors_expanded) ** 2).sum(dim=2))  # [B, 2]

        # Find closest attractor for each state
        min_distances, closest_attractor_idx = distances.min(dim=1)  # [B]

        # Initialize all as separatrix (0)
        labels = torch.zeros(state.shape[0], dtype=torch.long, device=state.device)

        # Mask for states within radius of an attractor
        within_radius = min_distances < radius

        # Classify based on which attractor they're closest to
        # Index 0: [0, -1, 0, 0] → label = 1 (stable bottom, success)
        # Index 1: [0, 1, 0, 0] → label = -1 (unstable top, failure)
        labels[within_radius & (closest_attractor_idx == 0)] = 1   # Stable bottom
        labels[within_radius & (closest_attractor_idx == 1)] = -1  # Unstable top

        # Return scalar int for single state
        if single_state:
            return labels.item()
        return labels

    # ===================================================================
    # NORMALIZATION & EMBEDDING FOR FLOW MATCHING
    # ===================================================================

    def normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        """
        Normalize pendulum cartesian state for flow matching

        Per-dimension normalization to approximately [-1, 1]:
        - x: normalized by x_limit (≈1.0)
        - y: normalized by y_limit (≈1.0)
        - ẋ: normalized by x_dot_limit (≈2π)
        - ẏ: normalized by y_dot_limit (≈2π)

        Args:
            state: [B, 4] raw state (x, y, ẋ, ẏ)

        Returns:
            [B, 4] normalized state without modifying input
        """
        # Create new tensor to avoid in-place modification
        normalized = state.clone()

        # Normalize each dimension by its limit
        normalized[:, 0] = state[:, 0] / self.x_limit
        normalized[:, 1] = state[:, 1] / self.y_limit
        normalized[:, 2] = state[:, 2] / self.x_dot_limit
        normalized[:, 3] = state[:, 3] / self.y_dot_limit

        # Clamp to [-1, 1] for safety
        normalized = torch.clamp(normalized, -1.0, 1.0)

        return normalized

    def denormalize_state(self, normalized_state: torch.Tensor) -> torch.Tensor:
        """
        Denormalize pendulum cartesian state back to raw coordinates

        Args:
            normalized_state: [B, 4] normalized state

        Returns:
            [B, 4] raw state (x, y, ẋ, ẏ) without modifying input
        """
        # Create new tensor to avoid in-place modification
        denormalized = normalized_state.clone()

        # Denormalize each dimension
        denormalized[:, 0] = normalized_state[:, 0] * self.x_limit
        denormalized[:, 1] = normalized_state[:, 1] * self.y_limit
        denormalized[:, 2] = normalized_state[:, 2] * self.x_dot_limit
        denormalized[:, 3] = normalized_state[:, 3] * self.y_dot_limit

        return denormalized

    def embed_state_for_model(self, state: torch.Tensor) -> torch.Tensor:
        """
        Embed pendulum cartesian state for model input

        Since all dimensions are Euclidean (no SO2), this is IDENTITY mapping.
        No sin/cos embedding needed - state is already in Cartesian form.

        Args:
            state: [B, 4] (x, y, ẋ, ẏ)

        Returns:
            [B, 4] same as input (identity embedding)
        """
        # Use base class implementation which handles Real components correctly
        return self.embed_state(state)
