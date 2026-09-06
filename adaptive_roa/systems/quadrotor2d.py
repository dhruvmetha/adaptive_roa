"""
Quadrotor 2D system for Latent Conditional Flow Matching

Manifold: ℝ² × S¹ × ℝ³ (6-dimensional state in XZ plane)
"""
import math
import torch
import numpy as np
import json
from pathlib import Path
from adaptive_roa.systems.base import DynamicalSystem, ManifoldComponent
from adaptive_roa.utils.env_config import get_data_dir, get_noise_regime
from typing import List, Dict, Tuple

_REQUIRED_BOUNDS_2D = ('x', 'z', 'theta', 'x_dot', 'z_dot', 'theta_dot')
_REQUIRED_BOUNDS_3D = ('x', 'y', 'z', 'x_dot', 'y_dot', 'z_dot', 'p', 'q', 'r')


def _resolve_achieved_bounds(dataset_info, json_path, det_dataset_name, required):
    """Return normalization bounds, falling back to the deterministic sibling.

    The `stochastic/` quadrotor datasets ship a different `dataset_description.json`
    schema from the `deterministic/` ones the system classes were written against:
    they document the noise mechanism, horizon and success criteria, but carry no
    `achieved_bounds` block. Their own descriptions state that
    `termination_thresholds` was taken from the deterministic dataset, i.e. they
    are the same plant over the same state space.

    So when `achieved_bounds` is missing we borrow the deterministic set's, rather
    than deriving bounds from the local `train.npz`. That choice is deliberate and
    load-bearing: bounds set the input normalization, so per-level bounds would
    give every noise level a DIFFERENT input scaling and silently destroy
    comparability across levels and against the deterministic baseline -- the
    comparison these datasets exist to support.

    Behaviour is unchanged for any dataset that already carries the key.
    """
    if 'achieved_bounds' in dataset_info:
        return dataset_info['achieved_bounds']

    fallback = Path(get_data_dir()) / "deterministic" / det_dataset_name / "dataset_description.json"
    if not fallback.exists():
        raise KeyError(
            f"{json_path} has no 'achieved_bounds' and the deterministic fallback "
            f"{fallback} does not exist. Normalization bounds are required."
        )
    with open(fallback) as f:
        det_info = json.load(f)
    if 'achieved_bounds' not in det_info:
        raise KeyError(f"Deterministic fallback {fallback} also lacks 'achieved_bounds'.")
    bounds = det_info['achieved_bounds']

    missing = [k for k in required if k not in bounds]
    if missing:
        raise KeyError(f"Deterministic fallback {fallback} is missing bounds for {missing}.")

    print(
        f"NOTE: {json_path} carries no 'achieved_bounds' (stochastic-tree schema); "
        f"borrowing normalization bounds from {fallback} so that every noise level "
        f"shares one input scaling."
    )
    return bounds


class Quadrotor2DSystem(DynamicalSystem):
    """
    Quadrotor 2D system with ℝ² × S¹ × ℝ³ manifold structure

    State representation: (x, z, θ, ẋ, ż, θ̇) where:
    - x ∈ ℝ (horizontal position)
    - z ∈ ℝ (vertical position / altitude)
    - θ ∈ S¹ (pitch angle, circular)
    - ẋ ∈ ℝ (horizontal velocity)
    - ż ∈ ℝ (vertical velocity)
    - θ̇ ∈ ℝ (pitch rate / angular velocity)

    Goal: Hover at (0, 1, 0, 0, 0, 0)
    """

    def __init__(self, dataset_dir: str = None):
        """
        Initialize Quadrotor 2D system

        Args:
            dataset_dir: Path to dataset directory containing dataset_description.json.
                        If None, uses default path from environment.
        """
        if dataset_dir is None:
            dataset_dir = f"{get_data_dir()}/{get_noise_regime()}/quadrotor2D_rl"

        dataset_dir = Path(dataset_dir)
        self.dataset_dir = str(dataset_dir)
        json_path = dataset_dir / "dataset_description.json"

        if not json_path.exists():
            raise FileNotFoundError(
                f"Cannot load bounds: dataset_description.json not found at {json_path}. "
                f"Achieved bounds are required for consistent normalization."
            )
        self._load_bounds_from_json(json_path)

        # Goal state: hover at (x=0, z=1, θ=0, ẋ=0, ż=0, θ̇=0)
        self.goal_state = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
        self.success_threshold = 0.3

        super().__init__()

    def _load_bounds_from_json(self, json_path: Path):
        """Load actual data bounds from dataset_description.json"""
        with open(json_path) as f:
            dataset_info = json.load(f)

        bounds = _resolve_achieved_bounds(
            dataset_info, json_path, "quadrotor2D_rl", _REQUIRED_BOUNDS_2D
        )

        # Position bounds
        self.x_limit = max(abs(bounds['x']['min']), abs(bounds['x']['max']))
        self.z_min = bounds['z']['min']
        self.z_max = bounds['z']['max']

        # Angle: circular, always [-π, π]
        self.angle_limit = np.pi

        # Velocity bounds
        self.x_dot_limit = max(abs(bounds['x_dot']['min']), abs(bounds['x_dot']['max']))
        self.z_dot_limit = max(abs(bounds['z_dot']['min']), abs(bounds['z_dot']['max']))
        self.theta_dot_limit = max(abs(bounds['theta_dot']['min']), abs(bounds['theta_dot']['max']))

        # Derived normalization for asymmetric z
        self.z_center = (self.z_min + self.z_max) / 2
        self.z_scale = (self.z_max - self.z_min) / 2

        # Store full info
        self.dataset_info = dataset_info
        self.achieved_bounds = bounds

        print(f"Quadrotor2D bounds loaded from {json_path}:")
        print(f"  [0] x (horizontal pos): [{bounds['x']['min']:.3f}, {bounds['x']['max']:.3f}] -> limit: ±{self.x_limit:.3f}")
        print(f"  [1] z (altitude): [{bounds['z']['min']:.3f}, {bounds['z']['max']:.3f}] -> center: {self.z_center:.3f}, scale: {self.z_scale:.3f}")
        print(f"  [2] θ (pitch angle): [{bounds['theta']['min']:.3f}, {bounds['theta']['max']:.3f}] -> WRAPPED to ±π")
        print(f"  [3] ẋ (horizontal vel): [{bounds['x_dot']['min']:.3f}, {bounds['x_dot']['max']:.3f}] -> limit: ±{self.x_dot_limit:.3f}")
        print(f"  [4] ż (vertical vel): [{bounds['z_dot']['min']:.3f}, {bounds['z_dot']['max']:.3f}] -> limit: ±{self.z_dot_limit:.3f}")
        print(f"  [5] θ̇ (pitch rate): [{bounds['theta_dot']['min']:.3f}, {bounds['theta_dot']['max']:.3f}] -> limit: ±{self.theta_dot_limit:.3f}")

    def define_manifold_structure(self) -> List[ManifoldComponent]:
        """
        Define ℝ² × S¹ × ℝ³ manifold structure:
        - Real components for position (x, z)
        - SO2 component for pitch angle θ
        - Real components for velocities (ẋ, ż, θ̇)
        """
        return [
            ManifoldComponent("Real", 2, "position"),       # (x, z) ∈ ℝ²
            ManifoldComponent("SO2", 1, "pitch_angle"),     # θ ∈ S¹
            ManifoldComponent("Real", 3, "velocity"),       # (ẋ, ż, θ̇) ∈ ℝ³
        ]

    def define_state_bounds(self) -> Dict[str, Tuple[float, float]]:
        """
        Define state bounds for normalization using actual data bounds
        """
        return {
            "position": (-self.x_limit, self.x_limit),
            "pitch_angle": (-self.angle_limit, self.angle_limit),
            "velocity": (-self.x_dot_limit, self.x_dot_limit),
        }

    def per_dim_bounds(self):
        """True per-axis support box: (x, z, theta, x_dot, z_dot, theta_dot).

        The component-level ``state_bounds`` collapses position to x's limit and
        velocity to x_dot's, which understates z (asymmetric, [z_min, z_max]) and
        theta_dot (an order of magnitude wider than x_dot). See
        DynamicalSystem.per_dim_bounds.
        """
        return [
            (-self.x_limit, self.x_limit),
            (self.z_min, self.z_max),
            (-math.pi, math.pi),
            (-self.x_dot_limit, self.x_dot_limit),
            (-self.z_dot_limit, self.z_dot_limit),
            (-self.theta_dot_limit, self.theta_dot_limit),
        ]

    def attractors(self) -> List[List[float]]:
        """
        Quadrotor 2D attractor: hover at (0, 1, 0, 0, 0, 0)

        Returns:
            List of [x, z, θ, ẋ, ż, θ̇] attractor positions
        """
        return [
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],  # Hover at z=1
        ]

    def is_in_attractor(self, state, radius: float = 0.3):
        """
        Check if states are within attractor basin (hovering at goal)

        Uses circular distance for pitch angle θ (index 2) and Euclidean
        distance for all other components.

        Args:
            state: States [B, 6] as (x, z, θ, ẋ, ż, θ̇) - numpy array or torch tensor
            radius: Attractor radius (distance threshold)

        Returns:
            Boolean tensor [B] indicating attractor membership
        """
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()

        if state.dim() == 1:
            state = state.unsqueeze(0)

        goal = torch.tensor(self.goal_state, device=state.device, dtype=state.dtype)

        # Euclidean components: x, z, ẋ, ż, θ̇ (indices 0, 1, 3, 4, 5)
        euclidean_indices = [0, 1, 3, 4, 5]
        euclidean_diff = state[:, euclidean_indices] - goal[euclidean_indices].unsqueeze(0)

        # Circular component: θ (index 2) - wrap difference to [-π, π]
        angle_diff = state[:, 2] - goal[2]
        angle_diff = torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff))

        # Combined distance: sqrt(sum of squared Euclidean diffs + squared angle diff)
        dist = torch.sqrt(
            torch.sum(euclidean_diff**2, dim=1) + angle_diff**2
        )
        result = dist < radius

        if len(result) == 1:
            return result.item()

        return result

    def classify_attractor(self, state: torch.Tensor, radius: float = 0.3) -> torch.Tensor:
        """
        Classify Quadrotor 2D states into three categories based on termination conditions

        Three-way classification:
        1. SUCCESS (label=1): Within radius of goal state
        2. FAILURE (label=-1): Exceeded termination thresholds
        3. SEPARATRIX (label=0): Between attractor and failure

        Termination thresholds from dataset_description.json:
        - |x| > 1.0 m
        - z < 0.1 or z > 1.5 m
        - |ẋ| > 1.0 m/s
        - |ż| > 1.0 m/s
        - |θ̇| > 8.0 rad/s
        - θ: no termination (infinite bounds)

        Args:
            state: States [B, 6] as (x, z, θ, ẋ, ż, θ̇)
            radius: Attractor radius (default 0.2)

        Returns:
            Integer tensor [B] with:
                 1: State in goal attractor (SUCCESS)
                -1: State exceeded termination thresholds (FAILURE)
                 0: State between attractor and failure (SEPARATRIX)
        """
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()

        if state.dim() == 1:
            state = state.unsqueeze(0)

        x, z, theta, x_dot, z_dot, theta_dot = (
            state[:, 0], state[:, 1], state[:, 2],
            state[:, 3], state[:, 4], state[:, 5]
        )

        # SUCCESS: Distance from goal < radius (with circular handling for θ)
        goal = torch.tensor(self.goal_state, device=state.device, dtype=state.dtype)

        # Euclidean components: x, z, ẋ, ż, θ̇ (indices 0, 1, 3, 4, 5)
        euclidean_indices = [0, 1, 3, 4, 5]
        euclidean_diff = state[:, euclidean_indices] - goal[euclidean_indices].unsqueeze(0)

        # Circular component: θ (index 2) - wrap difference to [-π, π]
        angle_diff = state[:, 2] - goal[2]
        angle_diff = torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff))

        # Combined distance
        dist = torch.sqrt(torch.sum(euclidean_diff**2, dim=1) + angle_diff**2)
        in_attractor = dist < radius

        # FAILURE: Exceeded termination thresholds (with small margin for overshoot)
        x_failed = torch.abs(x) > 0.9
        z_low_failed = z < 0.2
        z_high_failed = z > 1.4
        x_dot_failed = torch.abs(x_dot) > 0.9
        z_dot_failed = torch.abs(z_dot) > 0.9
        theta_dot_failed = torch.abs(theta_dot) > 7.5
        # theta has no termination threshold

        exceeded_thresholds = (x_failed | z_low_failed | z_high_failed |
                               x_dot_failed | z_dot_failed | theta_dot_failed)

        # Three-way classification
        labels = torch.zeros_like(in_attractor, dtype=torch.long)
        labels[in_attractor] = 1
        labels[exceeded_thresholds] = -1

        return labels

    # ===================================================================
    # NORMALIZATION & EMBEDDING FOR FLOW MATCHING
    # ===================================================================

    def normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        """
        Normalize raw state coordinates (x, z, θ, ẋ, ż, θ̇) → normalized

        Normalizes Euclidean quantities to approximately [-1, 1].
        z uses center/scale normalization due to asymmetric bounds.
        Angle remains unchanged (already in natural [-π, π] range).

        Args:
            state: [B, 6] raw state (θ already wrapped to [-π, π])

        Returns:
            [B, 6] normalized state
        """
        x = state[:, 0]
        z = state[:, 1]
        theta = state[:, 2]  # Keep as-is (circular)
        x_dot = state[:, 3]
        z_dot = state[:, 4]
        theta_dot = state[:, 5]

        x_norm = x / self.x_limit
        z_norm = (z - self.z_center) / self.z_scale
        x_dot_norm = x_dot / self.x_dot_limit
        z_dot_norm = z_dot / self.z_dot_limit
        theta_dot_norm = theta_dot / self.theta_dot_limit

        return torch.stack([x_norm, z_norm, theta, x_dot_norm, z_dot_norm, theta_dot_norm], dim=1)

    def denormalize_state(self, normalized_state: torch.Tensor) -> torch.Tensor:
        """
        Denormalize state back to raw coordinates.

        Args:
            normalized_state: [B, 6] normalized state

        Returns:
            [B, 6] raw state
        """
        x_norm = normalized_state[:, 0]
        z_norm = normalized_state[:, 1]
        theta = normalized_state[:, 2]  # Already in natural coordinates
        x_dot_norm = normalized_state[:, 3]
        z_dot_norm = normalized_state[:, 4]
        theta_dot_norm = normalized_state[:, 5]

        x = x_norm * self.x_limit
        z = z_norm * self.z_scale + self.z_center
        x_dot = x_dot_norm * self.x_dot_limit
        z_dot = z_dot_norm * self.z_dot_limit
        theta_dot = theta_dot_norm * self.theta_dot_limit

        return torch.stack([x, z, theta, x_dot, z_dot, theta_dot], dim=1)

    def embed_state_for_model(self, normalized_state: torch.Tensor) -> torch.Tensor:
        """
        Embed normalized state → (x_norm, z_norm, sin(θ), cos(θ), ẋ_norm, ż_norm, θ̇_norm)

        Converts circular angle to sin/cos representation for neural network input.

        Args:
            normalized_state: [B, 6] normalized state

        Returns:
            [B, 7] embedded state
        """
        x_norm = normalized_state[:, 0]
        z_norm = normalized_state[:, 1]
        theta = normalized_state[:, 2]
        x_dot_norm = normalized_state[:, 3]
        z_dot_norm = normalized_state[:, 4]
        theta_dot_norm = normalized_state[:, 5]

        sin_theta = torch.sin(theta)
        cos_theta = torch.cos(theta)

        return torch.stack([x_norm, z_norm, sin_theta, cos_theta,
                            x_dot_norm, z_dot_norm, theta_dot_norm], dim=1)

    def get_loss_weights(self) -> torch.Tensor:
        """
        Get per-dimension loss weights for Quadrotor 2D (6D state).

        State: (x, z, θ, ẋ, ż, θ̇)
        - x: position → weight = x_limit
        - z: altitude → weight = z_scale (half-range of asymmetric bounds)
        - θ: pitch angle (SO2) → weight = 1.0 (circular)
        - ẋ: horizontal velocity → weight = x_dot_limit
        - ż: vertical velocity → weight = z_dot_limit
        - θ̇: pitch rate → weight = theta_dot_limit

        Returns:
            torch.Tensor: 6D weights
        """
        weights = [
            self.x_limit,           # x (horizontal position)
            self.z_scale,            # z (altitude, asymmetric)
            1.0,                     # θ (pitch angle, circular)
            self.x_dot_limit,        # ẋ (horizontal velocity)
            self.z_dot_limit,        # ż (vertical velocity)
            self.theta_dot_limit,    # θ̇ (pitch rate)
        ]
        return torch.tensor(weights, dtype=torch.float32)

    def __repr__(self) -> str:
        return (f"Quadrotor2DSystem(ℝ² × S¹ × ℝ³, "
                f"limits=[±{self.x_limit:.2f}, {self.z_min:.2f}-{self.z_max:.2f}, ±π, "
                f"±{self.x_dot_limit:.2f}, ±{self.z_dot_limit:.2f}, ±{self.theta_dot_limit:.2f}])")
