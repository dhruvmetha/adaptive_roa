"""
Pendulum system for Latent Conditional Flow Matching
"""
import torch
import json
import math
from pathlib import Path
from adaptive_roa.systems.base import DynamicalSystem, ManifoldComponent
from adaptive_roa.utils.env_config import get_data_dir, get_noise_regime
from typing import List, Dict, Tuple


class PendulumSystem(DynamicalSystem):
    """
    Pendulum system with S¹ × ℝ manifold structure

    State representation: (θ, θ̇) where:
    - θ ∈ S¹ (circular angle)
    - θ̇ ∈ ℝ (angular velocity, normalized to [-1, 1])
    """

    def __init__(self, dataset_dir: str = None):
        """
        Initialize Pendulum system

        Args:
            dataset_dir: Path to dataset directory containing dataset_description.json.
                        If None, uses default path.
        """
        if dataset_dir is None:
            dataset_dir = f"{get_data_dir()}/{get_noise_regime()}/pendulum_lqr_50k"

        dataset_dir = Path(dataset_dir)
        self.dataset_dir = str(dataset_dir)
        json_path = dataset_dir / "dataset_description.json"

        # Load bounds from JSON - no fallback, error if not found
        if not json_path.exists():
            raise FileNotFoundError(
                f"Cannot load bounds: dataset_description.json not found at {json_path}. "
                f"Achieved bounds are required for consistent normalization."
            )
        self._load_bounds_from_json(json_path)

        super().__init__()

    def _load_bounds_from_json(self, json_path: Path):
        """Load actual data bounds from dataset_description.json"""
        with open(json_path) as f:
            dataset_info = json.load(f)

        bounds = dataset_info['achieved_bounds']

        # Angle is always wrapped to [-π, π]
        self.angle_limit = math.pi
        self.angular_velocity_limit = max(abs(bounds['theta_dot']['min']), abs(bounds['theta_dot']['max']))

        # Store the full dataset info for reference
        self.dataset_info = dataset_info
        self.achieved_bounds = bounds

        # Print in state vector order: [θ, θ̇]
        print(f"  [0] Angle (θ): [{bounds['theta']['min']:.3f}, {bounds['theta']['max']:.3f}] -> WRAPPED to ±π")
        print(f"  [1] Angular velocity (θ̇): [{bounds['theta_dot']['min']:.3f}, {bounds['theta_dot']['max']:.3f}] -> limit: ±{self.angular_velocity_limit:.3f}")

    def _compute_bounds_from_trajectories(self, trajectories_dir: Path):
        """Compute bounds from trajectory files as fallback when JSON is not accessible"""
        from adaptive_roa.utils.bounds_from_trajectories import compute_pendulum_bounds

        bounds_info = compute_pendulum_bounds(trajectories_dir, max_files=1000)

        self.angle_limit = bounds_info['angle_limit']  # Always π
        self.angular_velocity_limit = bounds_info['angular_velocity_limit']
        self.achieved_bounds = bounds_info['achieved_bounds']
        self.dataset_info = None  # Not available when computing from trajectories

    def define_manifold_structure(self) -> List[ManifoldComponent]:
        """
        Define S¹ × ℝ manifold structure:
        - SO2 component for angle θ
        - Real component for angular velocity θ̇
        """
        return [
            ManifoldComponent("SO2", 1, "angle"),             # θ ∈ S¹
            ManifoldComponent("Real", 1, "angular_velocity")  # θ̇ ∈ ℝ (normalized)
        ]
    
    def define_state_bounds(self) -> Dict[str, Tuple[float, float]]:
        """
        Define state bounds for normalization using actual data bounds
        """
        return {
            "angle": (-self.angle_limit, self.angle_limit),
            "angular_velocity": (-self.angular_velocity_limit, self.angular_velocity_limit)
        }
    
    def attractors(self) -> List[List[float]]:
        """
        Pendulum attractor positions in S¹ × ℝ space
        
        Returns:
            List of [θ, θ̇] attractor positions
        """
        return [
            [0.0, 0.0],      # Bottom equilibrium (stable)
            [2.1, 0.0],      # Top-right equilibrium  
            [-2.1, 0.0],     # Top-left equilibrium
        ]
    
    def is_in_attractor(self, state: torch.Tensor, radius: float = 0.1) -> torch.Tensor:
        """
        Check if states are within attractor basins using circular distance

        Args:
            state: States [B, 2] as (θ, θ̇)
            radius: Attractor radius

        Returns:
            Boolean tensor [B] indicating attractor membership
        """
        attractors = torch.tensor(self.attractors(), device=state.device, dtype=state.dtype)

        # Compute circular distance for angle (θ) and Euclidean for velocity (θ̇)
        # state: [B, 2], attractors: [3, 2]
        state_expanded = state.unsqueeze(1)  # [B, 1, 2]
        attractors_expanded = attractors.unsqueeze(0)  # [1, 3, 2]

        # Angular difference with circular wrapping
        angle_diff = state_expanded[:, :, 0] - attractors_expanded[:, :, 0]  # [B, 3]
        # Wrap to [-π, π] using atan2(sin, cos)
        angle_diff = torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff))

        # Velocity difference (Euclidean)
        vel_diff = state_expanded[:, :, 1] - attractors_expanded[:, :, 1]  # [B, 3]

        # Combined distance: sqrt(angle_diff² + vel_diff²)
        distances = torch.sqrt(angle_diff**2 + vel_diff**2)  # [B, 3]

        # Check if any attractor is within radius
        return (distances < radius).any(dim=1)  # [B]

    def classify_attractor(self, state: torch.Tensor, radius: float = 0.1) -> torch.Tensor:
        """
        Classify which attractor (if any) each state belongs to

        Args:
            state: States [B, 2] as (θ, θ̇)
            radius: Attractor radius

        Returns:
            Integer tensor [B] with:
                1: State in the goal attractor [0.0, 0.0] (SUCCESS)
               -1: State in a saturation attractor [±2.1, 0.0] (FAILURE)
                0: State in none of the attractors (SEPARATRIX)

        Attractor geometry (verified 2026-07-15 against dataset_description.json; an earlier
        version of this docstring had it backwards):
          * theta = 0 is the *upright / inverted* equilibrium and is UNSTABLE -- it is what the
            LQR stabilises. The documented EOM is
                theta_ddot = (g/l) sin(theta) + u/I - (b/I) theta_dot
            and d(theta_ddot)/d(theta) at 0 = +(g/l) > 0, hence unstable. It is NOT a stable
            "bottom" equilibrium; reaching and holding it is the task.
          * +/-2.1 are NOT the "top". They are the *torque-saturation* equilibria: with the
            controller pinned at u = -u_sat, gravity balances control where
                sin(theta*) = (u_sat/I)/(g/l) = 0.866025  ->  theta* = 2.0944 rad = 120 deg
            (u_sat appears chosen to place this exactly at 120 deg). The pendulum sticks there
            because saturated torque cannot lift it further -- that is the failure mode.
        """
        attractors = torch.tensor(self.attractors(), device=state.device, dtype=state.dtype)

        # Compute circular distance for angle (θ) and Euclidean for velocity (θ̇)
        state_expanded = state.unsqueeze(1)  # [B, 1, 2]
        attractors_expanded = attractors.unsqueeze(0)  # [1, 3, 2]

        # Angular difference with circular wrapping
        angle_diff = state_expanded[:, :, 0] - attractors_expanded[:, :, 0]  # [B, 3]
        angle_diff = torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff))
        

        # Velocity difference (Euclidean)
        vel_diff = state_expanded[:, :, 1] - attractors_expanded[:, :, 1]  # [B, 3]

        # Combined distance: sqrt(angle_diff² + vel_diff²)
        distances = torch.sqrt(angle_diff**2 + vel_diff**2)  # [B, 3]

        # Find closest attractor for each state
        min_distances, closest_attractor_idx = distances.min(dim=1)  # [B]

        # Initialize all as separatrix (0)
        labels = torch.zeros(state.shape[0], dtype=torch.long, device=state.device)

        # Mask for states within radius of an attractor
        within_radius = min_distances < radius

        # Classify based on which attractor they're closest to
        # Index 0: [0.0, 0.0] → label = 1 (stable, success)
        # Index 1: [2.1, 0.0] → label = -1 (unstable, failure)
        # Index 2: [-2.1, 0.0] → label = -1 (unstable, failure)
        labels[within_radius & (closest_attractor_idx == 0)] = 1   # Stable bottom
        labels[within_radius & (closest_attractor_idx == 1)] = -1  # Unstable top-right
        labels[within_radius & (closest_attractor_idx == 2)] = -1  # Unstable top-left

        return labels

    # ===================================================================
    # NORMALIZATION & EMBEDDING FOR FLOW MATCHING
    # ===================================================================

    def normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        """
        Normalize pendulum state for flow matching

        Normalization:
        - θ: kept as-is in [-π, π] (natural S¹ range)
        - θ̇: normalized to [-1, 1] by dividing by max velocity (2π)

        Args:
            state: [B, 2] raw state (θ, θ̇)

        Returns:
            [B, 2] normalized state (θ, θ̇_norm) without modifying input
        """
        # Create new tensor to avoid in-place modification
        normalized = state.clone()

        # Keep angle as-is (already in [-π, π])
        # Normalize angular velocity to [-1, 1]
        normalized[:, 1] = torch.clamp(
            state[:, 1] / self.state_bounds["angular_velocity"][1],
            -1.0, 1.0
        )

        return normalized

    def denormalize_state(self, normalized_state: torch.Tensor) -> torch.Tensor:
        """
        Denormalize pendulum state back to raw coordinates

        Denormalization:
        - θ: kept as-is (already in [-π, π])
        - θ̇_norm: scaled back to [-2π, 2π] by multiplying by max velocity

        Args:
            normalized_state: [B, 2] normalized state (θ, θ̇_norm)

        Returns:
            [B, 2] raw state (θ, θ̇) without modifying input
        """
        # Create new tensor to avoid in-place modification
        denormalized = normalized_state.clone()

        # Keep angle as-is
        # Denormalize angular velocity from [-1, 1] to [-2π, 2π]
        denormalized[:, 1] = normalized_state[:, 1] * self.state_bounds["angular_velocity"][1]

        return denormalized

    def embed_state_for_model(self, state: torch.Tensor) -> torch.Tensor:
        """
        Embed pendulum state for model input using base class implementation

        Converts angle to sin/cos representation via ManifoldComponent.

        Args:
            state: [B, 2] (θ, θ̇)

        Returns:
            [B, 3] (sin θ, cos θ, θ̇)
        """
        return self.embed_state(state)

    def get_loss_weights(self) -> torch.Tensor:
        """
        Get per-dimension loss weights for Pendulum (2D state).

        State: (θ, θ̇)
        - θ: angle (SO2) → weight = 1.0 (circular)
        - θ̇: angular velocity → weight = angular_velocity_limit

        Returns:
            torch.Tensor: 2D weights
        """
        weights = [
            1.0,                       # θ (angle, circular)
            self.angular_velocity_limit  # θ̇ (angular velocity)
        ]
        return torch.tensor(weights, dtype=torch.float32)

    def __repr__(self) -> str:
        return f"PendulumSystem(S¹ × ℝ, limits=[π, {self.angular_velocity_limit}])"


class PendulumG1System(PendulumSystem):
    """Pendulum where SUCCESS means reaching the RL->LQR handoff region G1.

    For RL-swing-up-only datasets (trajectories truncated at first G1 entry),
    the success criterion is membership in G1 -- the union of two rotated
    ellipses on the LQR ROA arms -- rather than proximity to the upright
    attractor. The G1 geometry is read from the ``G1_handoff_region`` block
    of the dataset's ``dataset_description.json``.

    classify_attractor() returns 1 (success) for states inside G1 and
    -1 (failure) otherwise; there is no separatrix/invalid class (0).
    """

    def __init__(self, dataset_dir: str = None):
        super().__init__(dataset_dir=dataset_dir)

        spec = (self.dataset_info or {}).get("G1_handoff_region")
        if spec is None:
            raise ValueError(
                f"G1_handoff_region not found in dataset_description.json under "
                f"{self.dataset_dir}. PendulumG1System requires the G1 geometry."
            )

        angle = float(spec["orientation"]["angle_rad"])
        # Rotation by -angle maps (dtheta, dtheta_dot) into (u, v) ellipse axes
        self._g1_cos = math.cos(-angle)
        self._g1_sin = math.sin(-angle)
        self._g1_centers = torch.tensor(
            [spec["arm_centers"]["left"], spec["arm_centers"]["right"]],
            dtype=torch.float32,
        )  # [2, 2]
        self._g1_r_major = float(spec["radii"]["major_along_band"])
        self._g1_r_minor = float(spec["radii"]["minor_perpendicular"])

    def in_g1(self, state: torch.Tensor) -> torch.Tensor:
        """Boolean [B] mask: state inside the G1 handoff region."""
        centers = self._g1_centers.to(device=state.device, dtype=state.dtype)
        theta = torch.atan2(torch.sin(state[:, 0]), torch.cos(state[:, 0]))

        dtheta = theta.unsqueeze(1) - centers[:, 0].unsqueeze(0)  # [B, 2]
        dtheta = torch.atan2(torch.sin(dtheta), torch.cos(dtheta))
        dvel = state[:, 1].unsqueeze(1) - centers[:, 1].unsqueeze(0)  # [B, 2]

        u = self._g1_cos * dtheta - self._g1_sin * dvel
        v = self._g1_sin * dtheta + self._g1_cos * dvel
        score = (u / self._g1_r_major) ** 2 + (v / self._g1_r_minor) ** 2  # [B, 2]
        # 2e-4 tolerance absorbs rounding of the published G1 parameters vs the
        # generator's exact values (observed dataset boundary states score <= 1.0001)
        return (score < 1.0 + 2e-4).any(dim=1)

    def classify_attractor(self, state: torch.Tensor, radius: float = 0.1) -> torch.Tensor:
        """G1 membership: 1 = success (inside G1), -1 = failure (outside).

        ``radius`` is accepted for interface compatibility but unused --
        G1 membership is a fixed geometric test.
        """
        labels = torch.full(
            (state.shape[0],), -1, dtype=torch.long, device=state.device
        )
        labels[self.in_g1(state)] = 1
        return labels

    def __repr__(self) -> str:
        return (
            f"PendulumG1System(S¹ × ℝ, success=G1 membership, "
            f"centers={self._g1_centers.tolist()})"
        )
