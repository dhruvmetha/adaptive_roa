"""
CartPole system for Latent Conditional Flow Matching
"""
import torch
import numpy as np
import json
from pathlib import Path
from adaptive_roa.systems.base import DynamicalSystem, ManifoldComponent
from adaptive_roa.utils.env_config import get_data_dir, get_noise_regime
from typing import List, Dict, Tuple


class CartPoleSystem(DynamicalSystem):
    """
    CartPole system with ℝ² × S¹ × ℝ manifold structure

    State representation: (x, θ, ẋ, θ̇) where:
    - x ∈ ℝ (cart position, normalized to [-1, 1])
    - θ ∈ S¹ (pole angle, circular)
    - ẋ ∈ ℝ (cart velocity, normalized to [-1, 1])
    - θ̇ ∈ ℝ (pole angular velocity, normalized to [-1, 1])
    """

    def __init__(self,
                 dataset_dir: str = None,
                 success_tolerance: List[float] = None,
                 goal: List[float] | None = None,
                 angle_limit: float | None = None):
        """
        Initialize CartPole system

        Args:
            dataset_dir: Path to dataset directory containing dataset_description.json.
                        If None, uses default path from environment.
            goal: Target state [x, theta, x_dot, theta_dot]. Defaults to the origin,
                  which is the LQR set's stabilisation point. The safe_explorer_ppo
                  policy parks the cart at x = 0.7 (cartpole_stab.yaml sets
                  stabilization_goal [0.7, 0]), so that controller MUST pass
                  [0.7, 0, 0, 0]. Leaving it at the origin is not a small error: RL
                  successes sit at ||state|| ~ 0.72, so every predicted endpoint is
                  scored a failure, p_hat collapses to ~0 and sAUROC pins at 0.500.
            angle_limit: |theta| normalisation bound. Defaults to pi (the wrapped
                  range). The RL policy terminates at |theta| >= 0.48, so passing pi
                  there would compress the entire usable angular range into +-0.153
                  of the normalised space.
        """
        if dataset_dir is None:
            dataset_dir = f"{get_data_dir()}/{get_noise_regime()}/cartpole_pybullet"

        dataset_dir = Path(dataset_dir)
        self.dataset_dir = str(dataset_dir)
        self.goal = [0.0, 0.0, 0.0, 0.0] if goal is None else [float(g) for g in goal]
        if len(self.goal) != 4:
            raise ValueError(f"cartpole goal must be 4-D [x, theta, x_dot, theta_dot], got {self.goal}")
        self._angle_limit_override = None if angle_limit is None else float(angle_limit)
        json_path = dataset_dir / "dataset_description.json"

        # Load bounds from JSON - no fallback, error if not found
        if not json_path.exists():
            raise FileNotFoundError(
                f"Cannot load bounds: dataset_description.json not found at {json_path}. "
                f"Achieved bounds are required for consistent normalization."
            )
        self._load_bounds_from_json(json_path)
        if success_tolerance is not None:
            if len(success_tolerance) != 4:
                raise ValueError(
                    "CartPole success_tolerance must contain four canonical "
                    "coordinates: [x, theta, x_dot, theta_dot]"
                )
            self.success_rule = {
                "kind": "per_channel_box_entry",
                "tol": [float(value) for value in success_tolerance],
                "source": "CartPoleSystem.success_tolerance override",
            }

        super().__init__()

    @staticmethod
    def _bounds_from_schema(dataset_info: dict) -> dict:
        """Normalise either dataset_description schema into achieved_bounds form.

        Two schemas ship in the wild and they disagree about more than layout:

        OLD (``stochastic/cartpole/noisy_action``, ``deterministic``) carries an
        explicit ``achieved_bounds`` block of per-channel {min, max} measured off
        the collected trajectories -- x +/-6.048, x_dot +/-7.338,
        theta_dot +/-8.571 on sigma_020.0.

        NEW (``stochastic/cartpole/noisy_torque``) has no ``achieved_bounds``.
        It declares ``termination_thresholds`` instead, and its own note says
        why the numbers moved: "The previously shipped stochastic set relaxed
        x_dot/theta_dot to 20.0; this restores the deterministic 5.0." States at
        or beyond a threshold were dropped at collection, so the thresholds ARE
        the achieved bounds and are tighter than the old family's.

        Falling back to the old numbers on new data would inflate the
        normalisation scales by ~1.5x on x_dot and ~1.7x on theta_dot, which
        silently rescales every embedding the kNN/MLP length models measure
        distance in. Hence: read the schema, never assume.
        """
        if "achieved_bounds" in dataset_info:
            return dataset_info["achieved_bounds"]

        thr = dataset_info.get("termination_thresholds") or {}
        samp = ((dataset_info.get("sampling") or {}).get("train") or {}).get("bounds") or {}

        def limit(key, default=None):
            for src in (thr, samp):
                v = src.get(key)
                if isinstance(v, (int, float)):
                    return float(v)
            if default is not None:
                return float(default)
            raise KeyError(
                f"dataset_description.json has neither achieved_bounds nor a numeric "
                f"bound for {key!r} under termination_thresholds/sampling.train.bounds")

        # theta is "inf (periodic)" in the new schema -- it wraps to +/-pi anyway.
        return {
            "x":         {"min": -limit("x"),         "max": limit("x")},
            "x_dot":     {"min": -limit("x_dot"),     "max": limit("x_dot")},
            "theta":     {"min": -np.pi,              "max": np.pi},
            "theta_dot": {"min": -limit("theta_dot"), "max": limit("theta_dot")},
        }

    def _load_bounds_from_json(self, json_path: Path):
        """Load actual data bounds from dataset_description.json"""
        with open(json_path) as f:
            dataset_info = json.load(f)

        bounds = self._bounds_from_schema(dataset_info)
        self.cart_limit = max(abs(bounds['x']['min']), abs(bounds['x']['max']))
        self.velocity_limit = max(abs(bounds['x_dot']['min']), abs(bounds['x_dot']['max']))
        # For angle, we'll wrap to [-π, π] in preprocessing, so use π as limit
        # Wrapping still maps theta into [-pi, pi]; the LIMIT is how much of that
        # range the normaliser spends. A policy that terminates well inside pi wants
        # its own bound (safe_explorer_ppo: 0.48) or it throws away resolution.
        self.angle_limit = np.pi if self._angle_limit_override is None \
            else self._angle_limit_override
        self.angular_velocity_limit = max(abs(bounds['theta_dot']['min']), abs(bounds['theta_dot']['max']))

        # Store the full dataset info for reference
        self.dataset_info = dataset_info
        self.achieved_bounds = bounds
        self.success_rule = (
            dataset_info.get("success_rule")
            or dataset_info.get("collection", {}).get("success_rule")
        )

        # Print in state vector order: [x, θ, ẋ, θ̇]
        print(f"  [0] Cart position (x): [{bounds['x']['min']:.3f}, {bounds['x']['max']:.3f}] -> limit: ±{self.cart_limit:.3f}")
        print(f"  [1] Pole angle (θ): [{bounds['theta']['min']:.3f}, {bounds['theta']['max']:.3f}] -> WRAPPED to ±π")
        print(f"  [2] Cart velocity (ẋ): [{bounds['x_dot']['min']:.3f}, {bounds['x_dot']['max']:.3f}] -> limit: ±{self.velocity_limit:.3f}")
        print(f"  [3] Angular velocity (θ̇): [{bounds['theta_dot']['min']:.3f}, {bounds['theta_dot']['max']:.3f}] -> limit: ±{self.angular_velocity_limit:.3f}")

    def _compute_bounds_from_trajectories(self, trajectories_dir: Path):
        """Compute bounds from trajectory files as fallback when JSON is not accessible"""
        from adaptive_roa.utils.bounds_from_trajectories import compute_cartpole_bounds

        bounds_info = compute_cartpole_bounds(trajectories_dir, max_files=1000)

        self.cart_limit = bounds_info['cart_limit']
        self.velocity_limit = bounds_info['velocity_limit']
        self.angle_limit = bounds_info['angle_limit']  # Always π
        self.angular_velocity_limit = bounds_info['angular_velocity_limit']
        self.achieved_bounds = bounds_info['achieved_bounds']
        self.dataset_info = None  # Not available when computing from trajectories
    
    def define_manifold_structure(self) -> List[ManifoldComponent]:
        """
        Define ℝ² × S¹ × ℝ manifold structure:
        - Real components for cart position and velocity
        - SO2 component for pole angle θ
        - Real component for pole angular velocity θ̇
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
            "pole_angle": (-self.angle_limit, self.angle_limit),  # Now uses actual data range
            "pole_angular_velocity": (-self.angular_velocity_limit, self.angular_velocity_limit)
        }
    
    def attractors(self) -> List[List[float]]:
        """
        CartPole attractor positions in ℝ² × S¹ × ℝ space
        Target state: cart centered, pole upright at 0°, no velocities
        
        Returns:
            List of [x, θ, ẋ, θ̇] attractor positions
        """
        return [list(self.goal)]
    
    def is_in_attractor(self, state, radius: float = 1.0):
        """
        Check if states are within attractor basins (balanced CartPole)

        Uses circular distance for pole angle θ (index 1) and Euclidean
        distance for all other components.

        Args:
            state: States [B, 4] as (x, θ, ẋ, θ̇) - numpy array or torch tensor
            radius: Attractor radius (distance threshold)

        Returns:
            Boolean tensor [B] indicating attractor membership
        """
        # Convert to torch tensor if needed
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()

        if state.dim() == 1:
            state = state.unsqueeze(0)

        # The stochastic datasets use direct entry into an axis-aligned box in
        # the unwrapped environment state. Their canonicalized copies retain
        # that rule verbatim; do not silently substitute the deterministic
        # CartPole L2 ball.
        if self.success_rule and self.success_rule.get("kind") == "per_channel_box_entry":
            tolerance = torch.as_tensor(
                self.success_rule["tol"], dtype=state.dtype, device=state.device
            )
            if tolerance.shape != (state.shape[1],):
                raise ValueError(
                    "CartPole per_channel_box_entry tolerance has shape "
                    f"{tuple(tolerance.shape)}, expected {(state.shape[1],)}"
                )
            result = torch.all(torch.abs(state) < tolerance, dim=1)
            if len(result) == 1:
                return result.item()
            return result

        # Distances are measured from self.goal, NOT the origin.
        g = torch.as_tensor(self.goal, dtype=state.dtype, device=state.device)
        euclidean_diff = state[:, [0, 2, 3]] - g[[0, 2, 3]]

        # Circular component: θ (index 1) - wrap the DIFFERENCE to [-π, π]
        dth = state[:, 1] - g[1]
        angle_diff = torch.atan2(torch.sin(dth), torch.cos(dth))

        # Combined distance: sqrt(sum of squared Euclidean diffs + squared angle diff)
        dist = torch.sqrt(torch.sum(euclidean_diff**2, dim=1) + angle_diff**2)
        result = dist < radius

        if len(result) == 1:
            return result.item()

        return result

    def classify_attractor(self, state: torch.Tensor, radius: float = 0.1) -> torch.Tensor:
        """
        Classify CartPole states into three categories based on termination conditions

        Three-way classification:
        1. SUCCESS (label=1): In upright balanced attractor [0,0,0,0]
        2. FAILURE (label=-1): Exceeded termination thresholds (system failed)
        3. SEPARATRIX (label=0): Between attractor and failure (uncertain region)

        Termination thresholds from PyBullet dataset:
        - |x| > 6.0 m (cart position)
        - |ẋ| > 5.0 m/s (cart velocity)
        - |θ̇| > 5.0 rad/s (angular velocity)
        - θ: no termination threshold (can flip fully)

        Args:
            state: States [B, 4] as (x, θ, ẋ, θ̇)
            radius: Attractor radius (default 0.1)

        Returns:
            Integer tensor [B] with:
                 1: State in upright balanced attractor (SUCCESS)
                -1: State exceeded termination thresholds (FAILURE)
                 0: State between attractor and failure (SEPARATRIX)
        """
        # Convert to torch tensor if needed
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()

        if state.dim() == 1:
            state = state.unsqueeze(0)

        x, theta, x_dot, theta_dot = state[:, 0], state[:, 1], state[:, 2], state[:, 3]

        if self.success_rule and self.success_rule.get("kind") == "per_channel_box_entry":
            tolerance = torch.as_tensor(
                self.success_rule["tol"], dtype=state.dtype, device=state.device
            )
            if tolerance.shape != (state.shape[1],):
                raise ValueError(
                    "CartPole per_channel_box_entry tolerance has shape "
                    f"{tuple(tolerance.shape)}, expected {(state.shape[1],)}"
                )
            in_attractor = torch.all(torch.abs(state) < tolerance, dim=1)
        else:
            # SUCCESS: distance from self.goal < radius (circular handling for θ)
            g = torch.as_tensor(self.goal, dtype=state.dtype, device=state.device)
            euclidean_diff = state[:, [0, 2, 3]] - g[[0, 2, 3]]

            # Circular component: θ (index 1) - wrap the DIFFERENCE to [-π, π]
            dth = theta - g[1]
            angle_diff = torch.atan2(torch.sin(dth), torch.cos(dth))

            # Combined distance
            dist = torch.sqrt(torch.sum(euclidean_diff**2, dim=1) + angle_diff**2)
            in_attractor = dist < radius



        # FAILURE: Check if exceeded termination thresholds
        # From dataset_description.json termination thresholds
        # Derived from the LOADED bounds, not hardcoded. The old literals (5.9,
        # 4.9, 4.9) were the LQR set's +-6/+-5/+-5 minus a margin; against the
        # safe_explorer_ppo set, which terminates at +-3 on all three channels,
        # they could never fire. Both datasets store only states strictly inside
        # their box, so testing at the limit is inert on stored data and correct
        # for predicted states that fall outside it.
        x_failed = torch.abs(x) >= self.cart_limit
        x_dot_failed = torch.abs(x_dot) >= self.velocity_limit
        theta_dot_failed = torch.abs(theta_dot) >= self.angular_velocity_limit
        # Note: theta has no termination threshold (inf)

        exceeded_thresholds = x_failed | x_dot_failed | theta_dot_failed

        # Three-way classification:
        # - If in attractor → SUCCESS (1)
        # - Else if exceeded thresholds → FAILURE (-1)
        # - Else → SEPARATRIX (0) - between attractor and failure
        labels = torch.zeros_like(in_attractor, dtype=torch.long)  # Initialize as separatrix (0)
        labels[in_attractor] = 1                                   # Mark successes
        labels[exceeded_thresholds] = -1                          # Mark failures

        return labels

    # ===================================================================
    # NORMALIZATION & EMBEDDING FOR FLOW MATCHING
    # ===================================================================

    def normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        """
        Normalize raw state coordinates (x, theta, x_dot, theta_dot) → (x_norm, theta, x_dot_norm, theta_dot_norm)

        Normalizes linear quantities to [-1, 1] using symmetric bounds.
        Angle remains unchanged (already in natural [-π, π] range).

        Args:
            state: [B, 4] raw cartpole state (theta already wrapped to [-π, π])

        Returns:
            [B, 4] normalized state (theta unchanged)
        """
        # Extract components
        x = state[:, 0]
        theta = state[:, 1]  # Keep as-is (already wrapped)
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

    def get_loss_weights(self) -> torch.Tensor:
        """
        Get per-dimension loss weights for CartPole (4D state).

        State: (x, θ, ẋ, θ̇)
        - x: cart position → weight = cart_limit
        - θ: pole angle (SO2) → weight = 1.0 (circular)
        - ẋ: cart velocity → weight = velocity_limit
        - θ̇: angular velocity → weight = angular_velocity_limit

        Returns:
            torch.Tensor: 4D weights
        """
        weights = [
            self.cart_limit,           # x (cart position)
            1.0,                       # θ (pole angle, circular)
            self.velocity_limit,       # ẋ (cart velocity)
            self.angular_velocity_limit  # θ̇ (angular velocity)
        ]
        return torch.tensor(weights, dtype=torch.float32)

    def __repr__(self) -> str:
        return f"CartPoleSystemLCFM(ℝ² × S¹ × ℝ, limits=[{self.cart_limit}, {self.velocity_limit}, π, {self.angular_velocity_limit}])"
