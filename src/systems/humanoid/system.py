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

    Success criteria (composite condition - ALL must be satisfied):
    - head_height >= 1.4m (dimension 21)
    - torso_vertical_z >= 0.9 (dimension 36) - torso nearly upright
    - horizontal_speed <= 0.2 m/s (sqrt(vx² + vy²), dimensions 37-38) - stable CoM
    """

    def __init__(self,
                 bounds_file: str = None,
                 use_dynamic_bounds: bool = False,
                 head_height_threshold: float = 1.4,
                 torso_z_threshold: float = 0.9,
                 speed_threshold: float = 0.2):
        """
        Initialize Humanoid system

        Args:
            bounds_file: Path to pickle file containing actual data bounds (optional)
            use_dynamic_bounds: If True, load bounds from file; if False, use defaults
            head_height_threshold: Minimum head height for success (default: 1.4m)
            torso_z_threshold: Minimum torso z-component for upright stance (default: 0.9)
            speed_threshold: Maximum horizontal speed for stability (default: 0.2 m/s)
        """
        # Store success thresholds
        self.head_height_threshold = head_height_threshold
        self.torso_z_threshold = torso_z_threshold
        self.speed_threshold = speed_threshold

        # Store bounds configuration (CRITICAL for checkpoint saving/loading)
        self.bounds_file = bounds_file
        self.use_dynamic_bounds = use_dynamic_bounds

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
        """Load actual data bounds from pickle file (per-dimension like CartPole)"""
        with open(bounds_file, 'rb') as f:
            bounds_data = pickle.load(f)

        # Store per-dimension bounds
        self.dimension_bounds = bounds_data.get('bounds', {})

        # Compute symmetric limits for Euclidean dimensions only
        # Sphere dimensions (34-36) have no bounds and no normalization
        self.dimension_limits = {}
        for i in range(67):
            if 34 <= i <= 36:
                # Sphere dimensions: no limit needed (no normalization)
                self.dimension_limits[i] = 1.0  # Placeholder, not used
            elif i in self.dimension_bounds:
                # Euclidean dimensions: compute symmetric limit
                min_val = self.dimension_bounds[i]['min']
                max_val = self.dimension_bounds[i]['max']
                self.dimension_limits[i] = max(abs(min_val), abs(max_val))
            else:
                # Default fallback for missing Euclidean dims
                raise ValueError(f"Dimension {i} not found in bounds file")
                self.dimension_limits[i] = 20.0

        print(f"✅ Loaded Humanoid bounds from: {bounds_file}")
        print(f"📊 Per-Dimension Normalization Limits:")
        print(f"")
        print(f"   Euclidean Block 1 (dims 0-33):")
        for i in range(min(5, 34)):
            if i in self.dimension_limits and i not in [34, 35, 36]:
                print(f"     [{i:2d}]: ±{self.dimension_limits[i]:7.3f}")
        if 34 > 5:
            print(f"     ... ({34-5} more dimensions)")
        print(f"")
        print(f"   Sphere (dims 34-36): NO NORMALIZATION (unit norm)")
        print(f"")
        print(f"   Euclidean Block 2 (dims 37-66):")
        for i in range(37, min(42, 67)):
            if i in self.dimension_limits:
                print(f"     [{i:2d}]: ±{self.dimension_limits[i]:7.3f}")
        if 67 > 42:
            print(f"     ... ({67-42} more dimensions)")
        print(f"")

    def _use_default_bounds(self):
        """Use default fallback bounds (per-dimension)"""
        # Conservative default for all dimensions
        # Based on sampling: actual range is ~[-17, 20], so use 20 as limit
        default_limit = 20.0

        self.dimension_bounds = None
        self.dimension_limits = {}
        for i in range(67):
            if 34 <= i <= 36:
                self.dimension_limits[i] = 1.0  # Placeholder for sphere (not used)
            else:
                self.dimension_limits[i] = default_limit

        print(f"Using default Humanoid bounds (per-dimension):")
        print(f"  Euclidean dimensions: ±{default_limit}")
        print(f"  Sphere (34-36): NO normalization (always unit norm)")

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
        Define state bounds for normalization (per-dimension)

        Returns:
            Dictionary mapping component names to (min, max) bounds
        """
        bounds = {}

        # First Euclidean block (34 dims) - per-dimension bounds
        for i in range(34):
            limit = self.dimension_limits[i]
            bounds[f"euclidean1_{i}"] = (-limit, limit)

        # Sphere block (3 dims) - unit norm, nominal range for each component
        bounds["orientation"] = (-1.0, 1.0)

        # Second Euclidean block (30 dims) - per-dimension bounds
        for i in range(30):
            limit = self.dimension_limits[37 + i]  # Dimensions 37-66
            bounds[f"euclidean2_{i}"] = (-limit, limit)

        return bounds

    def attractors(self) -> List[List[float]]:
        """
        Humanoid attractor positions (successful get-up pose)

        For the get-up task, success is defined by a composite condition:
        - head_height >= 1.4m (dimension 21)
        - torso_vertical_z >= 0.9 (dimension 36) - upright torso
        - horizontal_speed <= 0.2 m/s (dimensions 37-38) - stable CoM

        The attractor represents a standing pose meeting all criteria.

        Note: Success is determined by composite threshold, not distance to this pose.

        Returns:
            List of [state_0, ..., state_66] attractor positions
        """
        # Standing attractor: mostly zeros with successful standing configuration
        attractor = [0.0] * 67
        # Set head height to success threshold (dimension 21)
        attractor[21] = self.head_height_threshold  # Default: 1.4m
        # Set orientation to upward unit vector [0, 0, 1] (dimensions 34-36)
        attractor[34] = 0.0
        attractor[35] = 0.0
        attractor[36] = 1.0  # Torso vertical z = 1.0 (upright)
        # CoM velocities (dimensions 37-38) already zero for stable standing

        return [attractor]

    def is_in_attractor(self, state, radius: float = 1.0):
        """
        Check if humanoid successfully stood up using composite success criteria

        Success criteria from genMoPlan dataset (ALL must be satisfied):
        - head_height >= 1.4m (dimension 21): humanoid standing
        - torso_vertical_z >= 0.9 (dimension 36): torso nearly upright
        - horizontal_speed <= 0.2 m/s (dimensions 37-38): stable center of mass

        Args:
            state: States [B, 67] or [67] - numpy array or torch tensor
            radius: NOT USED - kept for API compatibility (success is threshold-based)

        Returns:
            Boolean tensor [B] or scalar indicating success (all criteria met)
        """
        
        # Convert to torch tensor if needed
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()

        if state.dim() == 1:
            state = state.unsqueeze(0)
            single_state = True
        else:
            single_state = False

        # Extract relevant state components
        head_height = state[:, 21]              # Dimension 21: head height
        torso_z = state[:, 36]                  # Dimension 36: torso vertical z-component
        com_vx = state[:, 37]                   # Dimension 37: CoM velocity x
        com_vy = state[:, 38]                   # Dimension 38: CoM velocity y
        com_speed = torch.sqrt(com_vx**2 + com_vy**2)  # Horizontal speed

        # Composite success condition: ALL three criteria must be met
        result = (head_height >= self.head_height_threshold) & \
                 (torso_z >= self.torso_z_threshold) & \
                 (com_speed <= self.speed_threshold)
        

        # Return scalar if single state
        if single_state:
            return result.item()
        
        return result

    def classify_attractor(self, state: torch.Tensor, radius: float = 1.0) -> torch.Tensor:
        """
        Classify humanoid states into success/failure categories

        Binary classification based on composite success criteria:
        1. SUCCESS (label=1):  ALL criteria met (head_height >= 1.4m AND torso_z >= 0.9 AND speed <= 0.2)
        2. FAILURE (label=-1): ANY criterion violated

        Args:
            state: States [B, 67] as full humanoid state
            radius: NOT USED - kept for API compatibility (success is threshold-based)

        Returns:
            Integer tensor [B] with:
                 1: State is success (all criteria satisfied)
                -1: State is failure (any criterion violated)
        """
        # Convert to torch tensor if needed
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()

        if state.dim() == 1:
            state = state.unsqueeze(0)

        is_standing = self.is_in_attractor(state, radius=radius)
        if isinstance(is_standing, bool):
            is_standing = torch.tensor([is_standing])
        elif not isinstance(is_standing, torch.Tensor):
            is_standing = torch.tensor(is_standing)

        # Binary classification: 1 for success, -1 for failure
        labels = torch.where(is_standing,
                           torch.ones_like(is_standing, dtype=torch.long),
                           -torch.ones_like(is_standing, dtype=torch.long))

        return labels

    # ===================================================================
    # NORMALIZATION & EMBEDDING FOR FLOW MATCHING
    # ===================================================================

    def normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        """
        Normalize raw state coordinates (per-dimension like CartPole)

        - Euclidean dimensions: normalize using per-dimension limits
        - Sphere dimensions: keep as-is (already unit norm)

        Args:
            state: [B, 67] raw humanoid state

        Returns:
            [B, 67] normalized state
        """
        normalized = state.clone()

        # Normalize each dimension individually
        for i in range(67):
            if 34 <= i <= 36:
                # Sphere dimensions: no normalization (already unit norm)
                continue
            else:
                # Euclidean dimensions: normalize by per-dimension limit
                normalized[:, i] = (state[:, i] - self.dimension_bounds[i]['min']) / (self.dimension_bounds[i]['max'] - self.dimension_bounds[i]['min'])

        return normalized

    def denormalize_state(self, normalized_state: torch.Tensor) -> torch.Tensor:
        """
        Denormalize state back to raw coordinates (per-dimension)

        Args:
            normalized_state: [B, 67] normalized state

        Returns:
            [B, 67] raw state
        """
        denormalized = normalized_state.clone()
        # Denormalize each dimension individually
        for i in range(67):
            if 34 <= i <= 36:
                # Sphere dimensions: no denormalization (already unit norm)
                continue
            else:
                # Euclidean dimensions: denormalize by per-dimension limit
                denormalized[:, i] = normalized_state[:, i] * (self.dimension_bounds[i]['max'] - self.dimension_bounds[i]['min']) + self.dimension_bounds[i]['min']

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
        return (f"HumanoidSystem(ℝ³⁴ × S² × ℝ³⁰, per-dimension bounds, 67D state, "
                f"success: h≥{self.head_height_threshold}m & tz≥{self.torso_z_threshold} & v≤{self.speed_threshold}m/s)")
