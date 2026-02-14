"""
Quadrotor 3D Latent Conditional Flow Matching implementation using Facebook Flow Matching library

Manifold: ℝ³ × SO(3) × ℝ⁶
- Representation dimension: 13D (3 position + 4 quaternion + 6 velocity)
- Tangent dimension: 12D (3 position + 3 rotation + 6 velocity)

State components:
- Position: ℝ³ (x, y, z) - 3D representation, 3D tangent
- Orientation: SO(3) via unit quaternion (qw, qx, qy, qz) - 4D representation, 3D tangent
- Linear velocity: ℝ³ (ẋ, ẏ, ż) - 3D representation, 3D tangent
- Angular velocity: ℝ³ (p, q, r) - 3D representation, 3D tangent

The model outputs 12D tangent velocities directly. FB FM's Product manifold with
(SO3(), 4, 3) handles conversion between representation (13D) and tangent (12D) spaces.
"""
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
import lightning.pytorch as pl
from torchmetrics import MeanMetric

from flow_matching.path import GeodesicProbPath
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.solver import RiemannianODESolver
from flow_matching.utils import ModelWrapper
from flow_matching.utils.manifolds import Product, Euclidean, SO3

from adaptive_roa.flow_matching.base.flow_matcher import BaseFlowMatcher
from adaptive_roa.systems.base import DynamicalSystem


class Quadrotor3DLatentConditionalFlowMatcher(BaseFlowMatcher):
    """
    Quadrotor 3D Latent Conditional Flow Matching using Facebook FM:
    - Uses GeodesicProbPath for geodesic interpolation on ℝ³ × SO(3) × ℝ⁶
    - Uses RiemannianODESolver for manifold-aware ODE integration
    - Neural net takes embedded state x_t (13D), time t, latent z, and start condition
    - Predicts velocity directly in 12D tangent space

    Manifold structure (Product manifold):
    - Euclidean(3): Position (x, y, z) - state indices 0-2, tangent indices 0-2
    - SO3(4, 3): Quaternion (qw, qx, qy, qz) - state indices 3-6, tangent indices 3-5
    - Euclidean(6): Velocities (ẋ, ẏ, ż, p, q, r) - state indices 7-12, tangent indices 6-11

    Dimension summary:
    - State/representation: 13D (3 + 4 + 6)
    - Tangent/velocity: 12D (3 + 3 + 6)

    The Product manifold handles conversion via logmap (13D→12D) and expmap (12D→13D).
    """

    def __init__(self,
                 system: DynamicalSystem,
                 model: nn.Module,
                 optimizer,
                 scheduler,
                 model_config: Optional[dict] = None,
                 latent_dim: int = 4,
                 mae_val_frequency: int = 10,
                 use_loss_weights: bool = False,
                 use_manifold: bool = True,
                 use_log_loss_weights: bool = False,
                 quat_loss_weight: float = 1.0,
                 clamp_noise: bool = True,
                 zero_latent: bool = False,
                 val_error_log_file: Optional[str] = None,
                 noise_scale: float = 1.0):
        """
        Initialize Quadrotor 3D latent conditional flow matcher with FB FM integration

        Args:
            system: DynamicalSystem (Quadrotor3D with ℝ³×SO(3)×ℝ⁶ structure)
            model: Quadrotor3DUNet model
            optimizer: Optimizer instance
            scheduler: Learning rate scheduler
            model_config: Configuration dict
            latent_dim: Dimension of latent space
            mae_val_frequency: Compute MAE validation every N epochs
            use_loss_weights: If True, weight loss by normalization limits
            use_manifold: If True, use SO(3) manifold for quaternion;
                         If False, use pure Euclidean R^13 with post-hoc quaternion projection
            use_log_loss_weights: If True and use_loss_weights=True, use 1+log(limit) instead of limit
                                  for more balanced weight ratios across dimensions
            quat_loss_weight: Scalar weight applied to quaternion loss dimensions when use_loss_weights=True.
                             Default 1.0 (no extra scaling). Values > 1 increase quaternion loss importance.
            clamp_noise: If True, clamp noise to [-1, 1] to prevent ODE divergence
            zero_latent: If True, use zero latent vectors instead of random sampling
            val_error_log_file: Path to text file for logging validation errors
            noise_scale: Scale factor for noise in sample_noisy_input (0-1, default 1.0)
        """
        # Store use_manifold BEFORE calling super().__init__ because it calls _create_manifold()
        self.use_manifold = use_manifold
        self.use_log_loss_weights = use_log_loss_weights
        self.quat_loss_weight = quat_loss_weight

        super().__init__(system, model, optimizer, scheduler, model_config, latent_dim, mae_val_frequency, use_loss_weights, clamp_noise, zero_latent, val_error_log_file, noise_scale)

        # Override loss weights for Euclidean mode or log weights
        # (13D tangent for Euclidean, 12D for manifold mode)
        if use_loss_weights:
            if use_log_loss_weights or not use_manifold:
                loss_weights = self._get_euclidean_loss_weights()
                self.register_buffer('loss_weights', loss_weights)
                weight_type = "1+log(limit)" if use_log_loss_weights else "limit"
                dim_str = "13D" if not use_manifold else "12D"
                print(f"📊 Loss weights ({dim_str}, {weight_type}): {loss_weights.tolist()}")

            # Apply quat_loss_weight to quaternion/rotation dimensions
            if quat_loss_weight != 1.0 and self.loss_weights is not None:
                if use_manifold:
                    # Manifold mode: 12D tangent (3 pos + 3 rot + 6 vel)
                    # Rotation tangent dimensions are indices 3:6
                    self.loss_weights[3:6] *= quat_loss_weight
                else:
                    # Euclidean mode: 13D tangent (3 pos + 4 quat + 6 vel)
                    # Quaternion dimensions are indices 3:7
                    self.loss_weights[3:7] *= quat_loss_weight
                print(f"📊 Applied quat_loss_weight={quat_loss_weight} → updated weights: {self.loss_weights.tolist()}")

        manifold_str = "ℝ³ × SO(3) × ℝ⁶" if use_manifold else "ℝ¹³ (Euclidean)"
        tangent_str = "12D (3 pos + 3 rot + 6 vel)" if use_manifold else "13D (all Euclidean)"

        print("✅ Initialized Quadrotor3D LCFM with Facebook Flow Matching:")
        print(f"   - Manifold: {manifold_str}")
        print(f"   - State dimension: 13D (3 pos + 4 quat + 6 vel)")
        print(f"   - Tangent dimension: {tangent_str}")
        print(f"   - Path: GeodesicProbPath with CondOTScheduler")
        print(f"   - Latent dim: {latent_dim}")
        print(f"   - MAE validation frequency: every {mae_val_frequency} epochs")
        print(f"   - Use manifold: {use_manifold}")

    def _create_manifold(self):
        """
        Create manifold for Quadrotor 3D.

        If use_manifold=True:
            Product manifold ℝ³ × SO(3) × ℝ⁶
            - Euclidean(3): Position (x, y, z) - 3D
            - SO3(4, 3): Quaternion representation - 4D state, 3D tangent
            - Euclidean(6): Linear + Angular velocity - 6D
            Total: 13D state, 12D tangent (3 + 3 + 6)

        If use_manifold=False:
            Pure Euclidean ℝ¹³ (all dimensions treated as Euclidean)
            Total: 13D state, 13D tangent
            Note: Quaternion projection done post-hoc in predict_endpoint
        """
        if self.use_manifold:
            return Product(input_dim=13, manifolds=[
                (Euclidean(), 3),      # Position (x, y, z)
                (SO3(), 4, 3),         # Quaternion (qw, qx, qy, qz) - 4D representation, 3D tangent
                (Euclidean(), 6)       # Velocities (ẋ, ẏ, ż, p, q, r)
            ])
        else:
            # Pure Euclidean mode: all 13D treated as Euclidean
            return Euclidean()

    def _create_distance_manifold(self):
        """
        Create manifold for distance computation (always true system manifold).

        Always returns ℝ³ × SO(3) × ℝ⁶ regardless of use_manifold setting,
        ensuring proper geodesic distances for quaternion components.

        Returns:
            Product manifold with SO(3) for proper orientation distances
        """
        return Product(input_dim=13, manifolds=[
            (Euclidean(), 3),      # Position (x, y, z)
            (SO3(), 4, 3),         # Quaternion (qw, qx, qy, qz) - 4D representation, 3D tangent
            (Euclidean(), 6)       # Velocities (ẋ, ẏ, ż, p, q, r)
        ])

    def _get_euclidean_loss_weights(self) -> torch.Tensor:
        """
        Get loss weights for tangent space dimensions.

        When use_manifold=False (Euclidean mode), tangent space is 13D:
        - Position (3D): weights from position limits
        - Quaternion (4D): weight = 1.0 for each component (unit quaternion range [-1, 1])
        - Linear velocity (3D): weights from velocity limits
        - Angular velocity (3D): weights from angular velocity limits

        When use_manifold=True (manifold mode), tangent space is 12D:
        - Position (3D): weights from position limits
        - Rotation tangent (3D): weight = 1.0 (SO(3) tangent is bounded rotation vectors)
        - Linear velocity (3D): weights from velocity limits
        - Angular velocity (3D): weights from angular velocity limits

        If use_log_loss_weights=True, applies 1 + log(limit) transformation to compress
        the weight range (e.g., from 39:1 to ~4.7:1 for angular velocity vs position).

        Returns:
            torch.Tensor: 12D or 13D weights depending on manifold mode
        """
        system = self.system

        # Position weights (3D)
        z_scale = (system.z_max - system.z_min) / 2
        position_weights = [system.x_limit, system.y_limit, z_scale]

        # Linear velocity weights (3D)
        linear_vel_weights = [system.x_dot_limit, system.y_dot_limit, system.z_dot_limit]

        # Angular velocity weights (3D)
        angular_vel_weights = [system.p_limit, system.q_limit, system.r_limit]

        if self.use_manifold:
            # Manifold mode: 12D tangent space (3 pos + 3 rot + 6 vel)
            # Rotation tangent space of SO(3) is 3D (angular velocity-like)
            rotation_weights = [1.0, 1.0, 1.0]  # Bounded rotation vectors
            weights = position_weights + rotation_weights + linear_vel_weights + angular_vel_weights
        else:
            # Euclidean mode: 13D tangent space (3 pos + 4 quat + 6 vel)
            quaternion_weights = [1.0, 1.0, 1.0, 1.0]  # Unit quaternion range [-1, 1]
            weights = position_weights + quaternion_weights + linear_vel_weights + angular_vel_weights

        weights_tensor = torch.tensor(weights, dtype=torch.float32)

        # Apply log transformation if enabled: 1 + log(limit)
        # This compresses the weight range while maintaining relative ordering
        # e.g., angular velocity (limit=39) goes from weight=39 to weight=1+log(39)≈4.67
        if self.use_log_loss_weights:
            weights_tensor = 1.0 + torch.log(weights_tensor.clamp(min=1e-6))

        return weights_tensor

    def _project_quaternion(self, state: torch.Tensor) -> torch.Tensor:
        """
        Project quaternion components (indices 3-6) to unit norm.

        Used when use_manifold=False to ensure valid quaternion after Euclidean integration.
        Also canonicalizes to ensure qw >= 0.

        Args:
            state: State tensor [B, 13]

        Returns:
            State with normalized quaternion [B, 13]
        """
        result = state.clone()
        quat = result[:, 3:7]

        # Normalize to unit quaternion
        quat_norm = quat.norm(dim=1, keepdim=True).clamp(min=1e-8)
        quat_normalized = quat / quat_norm

        # Canonicalize: ensure qw >= 0
        sign = torch.sign(quat_normalized[:, 0:1])
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        quat_normalized = quat_normalized * sign

        result[:, 3:7] = quat_normalized
        return result

    def _get_start_states(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract start states from batch"""
        return batch["start_state"]

    def _get_end_states(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract end states from batch"""
        return batch["end_state"]

    def _get_dimension_name(self, dim_idx: int) -> str:
        """Get human-readable dimension name for Quadrotor 3D"""
        names = [
            "x", "y", "z",           # Position
            "qw", "qx", "qy", "qz",  # Quaternion
            "x_dot", "y_dot", "z_dot",  # Linear velocity
            "p", "q", "r"            # Angular velocity
        ]
        return names[dim_idx] if 0 <= dim_idx < len(names) else f"dim_{dim_idx}"

    def get_manifold_component_names(self) -> list:
        """
        Get names for manifold distance components.

        For Quadrotor3D with ℝ³ × SO(3) × ℝ⁶:
        - Euclidean(3) returns 3 distances (x, y, z)
        - SO3 returns 1 distance (geodesic angle)
        - Euclidean(6) returns 6 distances (velocities)

        Total: 10 components (not 13, because SO3.dist returns single geodesic angle)

        Returns:
            List of 10 component names
        """
        return [
            # Euclidean position (3 values)
            "pos_x", "pos_y", "pos_z",
            # SO3 geodesic angle (1 value - angular distance between quaternions)
            "orientation_geodesic",
            # Euclidean velocities (6 values)
            "vel_x", "vel_y", "vel_z", "ang_p", "ang_q", "ang_r"
        ]

    def get_euclidean_groups(self) -> dict:
        """
        Define groups of state dimensions for Euclidean (L2) distance computation.

        For Quadrotor3D (13D state):
        - Position: indices 0-2 (x, y, z)
        - Quaternion: indices 3-6 (qw, qx, qy, qz) - note: L2 norm is NOT geodesic
        - Linear velocity: indices 7-9 (ẋ, ẏ, ż)
        - Angular velocity: indices 10-12 (p, q, r)

        Returns:
            Dictionary mapping group names to dimension indices
        """
        return {
            "position_L2": [0, 1, 2],           # Position (x, y, z)
            "quaternion_L2": [3, 4, 5, 6],      # Quaternion (for comparison, but geodesic is better)
            "linear_velocity_L2": [7, 8, 9],    # Linear velocity (ẋ, ẏ, ż)
            "angular_velocity_L2": [10, 11, 12], # Angular velocity (p, q, r)
            "full_state_L2": list(range(13)),   # Full state L2 norm
        }

    def sample_noisy_input(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Sample noisy input uniformly in ℝ³ × SO(3) × ℝ⁶ space

        If self.noise_scale != 1.0, scales the noise to reduce variance.
        If self.clamp_noise is True, clamps position and velocity to [-1, 1]
        to prevent ODE divergence. Quaternion is always normalized to unit sphere.

        Args:
            batch_size: Number of samples
            device: Device to create tensors on

        Returns:
            Noisy states [batch_size, 13]
        """
        # Position: Gaussian noise, scale and clamp if enabled
        position = torch.randn(batch_size, 3, device=device)
        if self.noise_scale != 1.0:
            position = position * self.noise_scale
        if self.clamp_noise:
            position = torch.clamp(position, -1.0, 1.0)

        # Quaternion: uniform sampling on SO(3) via normalized Gaussian
        # Sample 4D Gaussian and normalize to get uniform distribution on unit quaternion sphere
        # Note: noise_scale applied before normalization affects the distribution, but
        # normalization ensures unit quaternion. No clamping needed.
        quat = torch.randn(batch_size, 4, device=device)
        if self.noise_scale != 1.0:
            quat = quat * self.noise_scale
        quat = quat / torch.norm(quat, dim=1, keepdim=True).clamp(min=1e-8)
        # Canonicalize: ensure qw >= 0
        sign = torch.sign(quat[:, 0:1])
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        quat = quat * sign

        # Velocities: Gaussian noise, scale and clamp if enabled
        velocities = torch.randn(batch_size, 6, device=device)
        if self.noise_scale != 1.0:
            velocities = velocities * self.noise_scale
        if self.clamp_noise:
            velocities = torch.clamp(velocities, -1.0, 1.0)

        noisy_input = torch.cat([position, quat, velocities], dim=1)

        # Project onto manifold
        noisy_input = self.manifold.projx(noisy_input)

        return noisy_input

    def normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        """Delegate to system for normalization"""
        return self.system.normalize_state(state)

    def denormalize_state(self, normalized_state: torch.Tensor) -> torch.Tensor:
        """Delegate to system for denormalization"""
        return self.system.denormalize_state(normalized_state)

    def embed_state_for_model(self, normalized_state: torch.Tensor) -> torch.Tensor:
        """Delegate to system for embedding"""
        return self.system.embed_state_for_model(normalized_state)

    # NOTE: compute_flow_loss is inherited from BaseFlowMatcher
    # The base class handles the flow matching loss correctly for Product manifolds
    # with mixed state/tangent dimensions. Model outputs 12D tangent velocity,
    # which matches path_sample.dx_t (also 12D from logmap).

    def predict_endpoint(self,
                        start_states: torch.Tensor,
                        num_steps: int = 100,
                        latent: Optional[torch.Tensor] = None,
                        method: str = "euler_riemannian") -> torch.Tensor:
        """
        Predict endpoints from start states.

        Overrides base class to add quaternion projection when use_manifold=False.

        Args:
            start_states: Start states [B, state_dim] in raw coordinates
            num_steps: Number of integration steps for ODE solving
            latent: Optional latent vectors [B, latent_dim]. If None, will sample.
            method: Integration method ("euler_riemannian", "euler", "rk4", "midpoint")

        Returns:
            Predicted endpoints [B, state_dim] in raw coordinates
        """
        # Call parent implementation
        endpoints = super().predict_endpoint(start_states, num_steps, latent, method)

        # IMPORTANT: Always return a valid quaternion representative.
        #
        # - Training data is canonicalized (qw >= 0).
        # - Depending on the SO(3) implementation / integration, we may return q or -q
        #   (same rotation). Downstream code (e.g. Euclidean goal-distance checks)
        #   is sign-sensitive, so we canonicalize here.
        if hasattr(self.system, "project_to_manifold"):
            endpoints = self.system.project_to_manifold(endpoints)
        else:
            endpoints = self._project_quaternion(endpoints)

        return endpoints

    def predict_endpoints_batch(self,
                               start_states: torch.Tensor,
                               num_steps: int = 100,
                               num_samples: int = 1) -> torch.Tensor:
        """
        Predict multiple endpoint samples per start state (for stochastic models).

        Args:
            start_states: Start states [B, 13] in raw coordinates
            num_steps: Number of integration steps
            num_samples: Number of samples per start state

        Returns:
            Predicted endpoints [B*num_samples, 13] in raw coordinates
        """
        if num_samples == 1:
            raw_endpoints = self.predict_endpoint(start_states, num_steps)
            return raw_endpoints

        batch_size = start_states.shape[0]
        all_endpoints = []

        for _ in range(num_samples):
            # Sample different latent vectors for each sample
            endpoints_raw = self.predict_endpoint(start_states, num_steps, latent=None)
            all_endpoints.append(endpoints_raw)

        # Concatenate all samples: [B*num_samples, 13]
        return torch.cat(all_endpoints, dim=0)

    # ===================================================================
    # CHECKPOINT LOADING FOR INFERENCE
    # ===================================================================

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path: str, device: Optional[str] = None):
        """
        Load a trained Quadrotor3D LCFM model from checkpoint for inference.

        Args:
            checkpoint_path: Path to Lightning checkpoint file (.ckpt) OR training folder
                           - If .ckpt file: loads that checkpoint directly
                           - If folder: searches for best checkpoint in folder/version_0/checkpoints/
            device: Device to load model on ("cuda", "cpu", or None for auto)

        Returns:
            Loaded model ready for inference
        """
        import torch
        from pathlib import Path
        from adaptive_roa.systems.quadrotor3d import Quadrotor3DSystem
        from adaptive_roa.flow_matching.base.checkpoint_utils import instantiate_model_from_config

        # Determine device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        checkpoint_path = Path(checkpoint_path)

        # Check if it's a folder or a .ckpt file
        if checkpoint_path.is_dir():
            print(f"📁 Folder provided: {checkpoint_path}")
            print(f"🔍 Searching for checkpoint in folder...")

            # Try v2 adaptive layout first, then legacy standalone layout
            checkpoint_dir = checkpoint_path / "checkpoints"
            if not checkpoint_dir.exists():
                checkpoint_dir = checkpoint_path / "version_0" / "checkpoints"
            if not checkpoint_dir.exists():
                raise FileNotFoundError(f"No checkpoints directory found in {checkpoint_path}")

            # Find all .ckpt files (exclude last.ckpt)
            checkpoints = [p for p in checkpoint_dir.glob("*.ckpt") if p.name != "last.ckpt"]

            if not checkpoints:
                raise FileNotFoundError(f"No .ckpt files found in {checkpoint_dir}")

            # Parse validation loss from filename
            best_checkpoint = None
            best_val_loss = float('inf')

            for ckpt in checkpoints:
                try:
                    if "val_loss" in ckpt.stem:
                        loss_str = ckpt.stem.split("val_loss")[1]
                        val_loss = float(loss_str)
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            best_checkpoint = ckpt
                except (ValueError, IndexError):
                    continue

            if best_checkpoint is None:
                checkpoint_path = max(checkpoints, key=lambda p: p.stat().st_mtime)
                print(f"   ⚠️  Could not parse val_loss, using most recent checkpoint")
            else:
                checkpoint_path = best_checkpoint
                print(f"   ✓ Found best checkpoint (val_loss={best_val_loss:.4f})")

            print(f"   📄 Using: {checkpoint_path.name}")

        print(f"🤖 Loading Quadrotor3D LCFM checkpoint: {checkpoint_path}")
        print(f"📍 Device: {device}")

        # Verify checkpoint exists
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Find training directory and load Hydra config
        from adaptive_roa.flow_matching.base.checkpoint_utils import find_training_dir, load_hydra_config
        training_dir = find_training_dir(checkpoint_path)
        print(f"🗂️  Training directory: {training_dir}")
        hydra_config = load_hydra_config(training_dir)

        # Load Lightning checkpoint
        print(f"📦 Loading Lightning checkpoint...")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        hparams = checkpoint.get("hyper_parameters", {})
        print("✅ Lightning checkpoint loaded")

        def _require_hparam(key: str):
            if key not in hparams:
                raise KeyError(
                    f"Missing required checkpoint hyper_parameter '{key}'. "
                    f"Checkpoint: {checkpoint_path}"
                )
            return hparams[key]

        latent_dim = int(_require_hparam("latent_dim"))
        use_manifold = bool(_require_hparam("use_manifold"))
        use_loss_weights = bool(_require_hparam("use_loss_weights"))
        use_log_loss_weights = bool(_require_hparam("use_log_loss_weights"))
        quat_loss_weight = float(_require_hparam("quat_loss_weight"))
        clamp_noise = bool(_require_hparam("clamp_noise"))
        zero_latent = bool(_require_hparam("zero_latent"))
        mae_val_frequency = int(_require_hparam("mae_val_frequency"))
        noise_scale = float(_require_hparam("noise_scale"))
        val_error_log_file = hparams.get("val_error_log_file")

        # Determine output dimension based on manifold mode
        # - use_manifold=True: SO(3) manifold → 12D tangent (3 pos + 3 rot + 6 vel)
        # - use_manifold=False: Euclidean R^13 → 13D tangent (same as state dim)
        model_output_dim = 12 if use_manifold else 13

        # Extract model config
        config_source = None
        if "model_config" in hparams:
            model_config = hparams["model_config"]
            config_source = "checkpoint (model_config)"
        elif "config" in hparams:
            model_config = hparams["config"]
            config_source = "checkpoint (config)"
        else:
            raise KeyError(
                f"Checkpoint missing both 'model_config' and 'config' hyper_parameters. "
                f"Checkpoint: {checkpoint_path}"
            )

        model_config["latent_dim"] = latent_dim

        print(f"Config source: {config_source}")
        print(f"Final config - latent_dim: {latent_dim}")
        print(f"Model config keys: {list(model_config.keys())}")
        print(
            "Runtime config - "
            f"use_manifold: {use_manifold}, "
            f"use_loss_weights: {use_loss_weights}, "
            f"use_log_loss_weights: {use_log_loss_weights}, "
            f"quat_loss_weight: {quat_loss_weight}, "
            f"clamp_noise: {clamp_noise}, "
            f"zero_latent: {zero_latent}, "
            f"mae_val_frequency: {mae_val_frequency}, "
            f"noise_scale: {noise_scale}"
        )

        # Initialize system and model
        system = hparams.get("system")
        if system is None:
            print("Creating new Quadrotor3D system (not found in hparams)")
            dataset_dir = hparams.get("system_dataset_dir")
            if not dataset_dir:
                raise KeyError(
                    "Checkpoint is missing 'system_dataset_dir' hyper_parameter required for "
                    "strict Quadrotor3D restoration."
                )
            print(f"   dataset_dir: {dataset_dir}")
            system = Quadrotor3DSystem(dataset_dir=dataset_dir)
        else:
            print("Restored Quadrotor3D system from checkpoint")

        # Dynamically instantiate model from config
        model = instantiate_model_from_config(model_config, latent_dim)

        # Create flow matcher instance
        flow_matcher = cls(
            system=system,
            model=model,
            optimizer=None,
            scheduler=None,
            model_config=model_config,
            latent_dim=latent_dim,
            mae_val_frequency=mae_val_frequency,
            use_loss_weights=use_loss_weights,
            use_manifold=use_manifold,
            use_log_loss_weights=use_log_loss_weights,
            quat_loss_weight=quat_loss_weight,
            clamp_noise=clamp_noise,
            zero_latent=zero_latent,
            val_error_log_file=val_error_log_file,
            noise_scale=noise_scale,
        )

        # Load model weights
        print("🔄 Loading model state dict...")
        state_dict = checkpoint["state_dict"]
        model_state_dict = {k.replace("model.", ""): v for k, v in state_dict.items() if k.startswith("model.")}

        if not model_state_dict:
            raise ValueError("No model weights found in checkpoint! Keys: " + str(list(state_dict.keys())[:10]))

        flow_matcher.model.load_state_dict(model_state_dict)

        # Move to device and set eval mode
        flow_matcher = flow_matcher.to(device)
        flow_matcher.eval()

        # Attach Hydra training config
        flow_matcher.training_config = hydra_config

        # Success summary
        manifold_str = "R3 x SO(3) x R6" if use_manifold else "R13 (Euclidean)"
        model_info = model.get_model_info() if hasattr(model, 'get_model_info') else {}
        print(f"\nModel loaded successfully!")
        print(f"   Checkpoint: {checkpoint_path.name}")
        print(f"   Config sources: {'Hydra + Lightning' if hydra_config else 'Lightning only'}")
        print(f"   System: {type(system).__name__}")
        print(f"   Latent dim: {latent_dim}")
        print(f"   Use manifold: {use_manifold} ({manifold_str})")
        print(f"   Model type: {model_info.get('model_type', 'unknown')}")
        print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   Device: {device}")

        return flow_matcher
