"""
Humanoid system for Flow Matching - Get Up Task
"""
import torch
import numpy as np
import pickle
from pathlib import Path
from src.systems.base import DynamicalSystem, ManifoldComponent
from typing import List, Dict, Tuple


class HumanoidSystem(DynamicalSystem):
    """
    Humanoid system with Product manifold structure (get-up task)

    Manifold: ℝ³⁴ × S² × ℝ³⁰ (67-dimensional state)

    State representation:
    - Dimensions 0-33: Euclidean (34 dims) - positions, velocities, joint angles
    - Dimensions 34-36: Sphere (3 dims) - orientation as 3D unit vector (norm=1)
    - Dimensions 37-66: Euclidean (30 dims) - additional joint states

    Total: 34 + 3 + 30 = 67 dimensions
    """

    def __init__(self,
                 bounds_file: str = None,
                 use_dynamic_bounds: bool = False):
        """
        Initialize Humanoid system

        Args:
            bounds_file: Path to pickle file containing actual data bounds (optional)
            use_dynamic_bounds: If True, load bounds from file; if False, use defaults
        """
        if use_dynamic_bounds and bounds_file and Path(bounds_file).exists():
            self._load_bounds_from_file(bounds_file)
            print(f"Loaded Humanoid bounds from: {bounds_file}")
        else:
            self._use_default_bounds()
            if use_dynamic_bounds and bounds_file:
                print(f"Warning: Bounds file not found at {bounds_file}, using defaults")

        super().__init__()
        self.name = "humanoid"

    def _load_bounds_from_file(self, bounds_file: str):
        """Load actual data bounds from pickle file"""
        with open(bounds_file, 'rb') as f:
            bounds_data = pickle.load(f)

        # Store bounds for Euclidean dimensions (sphere dims have unit norm)
        limits = bounds_data.get('limits', {})
        self.euclidean_limit = limits.get('euclidean_limit', 20.0)
        self.dimension_bounds = bounds_data.get('bounds', {})

        print(f"Loaded Humanoid bounds")
        print(f"  Euclidean dimensions limit: ±{self.euclidean_limit:.3f}")
        print(f"  Sphere dimensions: unit norm (no normalization)")

    def _use_default_bounds(self):
        """Use default fallback bounds"""
        # Conservative default for Euclidean dimensions
        # Based on sampling: actual range is ~[-17, 20], so use 20 as limit
        self.euclidean_limit = 20.0
        self.dimension_bounds = None
        print(f"Using default Humanoid bounds:")
        print(f"  Euclidean (64 dims): ±{self.euclidean_limit}")
        print(f"  Sphere (3 dims): unit norm")

    def define_manifold_structure(self) -> List[ManifoldComponent]:
        """
        Define ℝ³⁴ × S² × ℝ³⁰ manifold structure

        Manifold composition:
        1. Euclidean(34): First 34 dimensions (indices 0-33)
        2. Sphere(3): Next 3 dimensions (indices 34-36) - 3D unit vector
        3. Euclidean(30): Last 30 dimensions (indices 37-66)

        Returns:
            List of manifold components
        """
        components = []

        # First Euclidean block: 34 dimensions (indices 0-33)
        for i in range(34):
            components.append(ManifoldComponent("Real", 1, f"euclidean1_{i}"))

        # Sphere block: 3 dimensions (indices 34-36) as a single 3D unit vector
        # This is S² (2-sphere embedded in ℝ³)
        components.append(ManifoldComponent("Sphere", 3, "orientation"))

        # Second Euclidean block: 30 dimensions (indices 37-66)
        for i in range(30):
            components.append(ManifoldComponent("Real", 1, f"euclidean2_{i}"))

        return components

    def define_state_bounds(self) -> Dict[str, Tuple[float, float]]:
        """
        Define state bounds for normalization

        Returns:
            Dictionary mapping component names to (min, max) bounds
        """
        bounds = {}

        # First Euclidean block (34 dims)
        for i in range(34):
            bounds[f"euclidean1_{i}"] = (-self.euclidean_limit, self.euclidean_limit)

        # Sphere block (3 dims) - unit norm, nominal range for each component
        bounds["orientation"] = (-1.0, 1.0)

        # Second Euclidean block (30 dims)
        for i in range(30):
            bounds[f"euclidean2_{i}"] = (-self.euclidean_limit, self.euclidean_limit)

        return bounds

    def attractors(self) -> List[List[float]]:
        """
        Humanoid attractor positions (successful get-up pose)

        For the get-up task, the attractor is the standing upright position.
        Using origin as placeholder - should be replaced with actual target pose.

        Returns:
            List of [state_0, ..., state_66] attractor positions
        """
        # Placeholder attractor: mostly zeros, with unit orientation
        attractor = [0.0] * 67
        # Set orientation to default unit vector pointing up [0, 0, 1]
        attractor[34] = 0.0
        attractor[35] = 0.0
        attractor[36] = 1.0

        return [attractor]

    def is_in_attractor(self, state, radius: float = 1.0):
        """
        Check if states are in attractor basin (successfully stood up)

        Uses Euclidean distance for Euclidean components and angular distance
        for Sphere components.

        Args:
            state: States [B, 67] or [67] - numpy array or torch tensor
            radius: Attractor radius threshold (default: 1.0)

        Returns:
            Boolean tensor [B] or scalar indicating attractor membership
        """
        # Convert to torch tensor if needed
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()

        if state.dim() == 1:
            state = state.unsqueeze(0)

        # Get attractor
        attractor_state = torch.tensor(self.attractors()[0], dtype=state.dtype, device=state.device)

        # Compute distance considering manifold structure
        # For simplicity, use Euclidean distance across all dimensions
        # (Sphere components are already constrained to unit norm)
        distance = torch.norm(state - attractor_state, dim=1)
        result = distance < radius

        # Return scalar if single state
        if len(result) == 1:
            return result.item()

        return result

    def classify_attractor(self, state: torch.Tensor, radius: float = 1.0) -> torch.Tensor:
        """
        Classify humanoid states into success/failure categories

        Binary classification:
        1. SUCCESS (label=1): In standing attractor (successfully stood up)
        2. FAILURE (label=-1): Not in standing attractor

        Args:
            state: States [B, 67] as full humanoid state
            radius: Attractor radius (default: 1.0)

        Returns:
            Integer tensor [B] with:
                 1: State in standing attractor (SUCCESS)
                -1: State not in attractor (FAILURE)
        """
        # Convert to torch tensor if needed
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()

        if state.dim() == 1:
            state = state.unsqueeze(0)

        # Check if in attractor
        in_attractor = self.is_in_attractor(state, radius=radius)
        if isinstance(in_attractor, bool):
            in_attractor = torch.tensor([in_attractor])

        # Binary classification: 1 for success, -1 for failure
        labels = torch.where(in_attractor,
                           torch.ones_like(in_attractor, dtype=torch.long),
                           -torch.ones_like(in_attractor, dtype=torch.long))

        return labels

    # ===================================================================
    # NORMALIZATION & EMBEDDING FOR FLOW MATCHING
    # ===================================================================

    def normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        """
        Normalize raw state coordinates

        - Euclidean dimensions: normalize to [-1, 1] using bounds
        - Sphere dimensions: keep as-is (already unit norm)

        Args:
            state: [B, 67] raw humanoid state

        Returns:
            [B, 67] normalized state
        """
        normalized = state.clone()

        # Normalize first Euclidean block (dims 0-33)
        normalized[:, :34] = state[:, :34] / self.euclidean_limit

        # Sphere block (dims 34-36): keep as-is (already unit norm)
        # normalized[:, 34:37] = state[:, 34:37]  # No change

        # Normalize second Euclidean block (dims 37-66)
        normalized[:, 37:] = state[:, 37:] / self.euclidean_limit

        return normalized

    def denormalize_state(self, normalized_state: torch.Tensor) -> torch.Tensor:
        """
        Denormalize state back to raw coordinates

        Args:
            normalized_state: [B, 67] normalized state

        Returns:
            [B, 67] raw state
        """
        denormalized = normalized_state.clone()

        # Denormalize first Euclidean block (dims 0-33)
        denormalized[:, :34] = normalized_state[:, :34] * self.euclidean_limit

        # Sphere block (dims 34-36): keep as-is (already unit norm)
        # No denormalization needed

        # Denormalize second Euclidean block (dims 37-66)
        denormalized[:, 37:] = normalized_state[:, 37:] * self.euclidean_limit

        return denormalized

    def embed_state_for_model(self, normalized_state: torch.Tensor) -> torch.Tensor:
        """
        Embed normalized state for neural network input

        For this manifold structure (ℝ³⁴ × S² × ℝ³⁰):
        - Euclidean components: pass through as-is
        - Sphere components: pass through as-is (already 3D continuous)

        The sphere is already embedded in ℝ³ as a unit vector,
        so no additional embedding (like sin/cos) is needed.

        Args:
            normalized_state: [B, 67] normalized state

        Returns:
            [B, 67] embedded state (identity transformation)
        """
        # No embedding needed - sphere is already in continuous 3D space
        return normalized_state

    def __repr__(self) -> str:
        return f"HumanoidSystem(ℝ³⁴ × S² × ℝ³⁰, euclidean_limit=±{self.euclidean_limit})"
