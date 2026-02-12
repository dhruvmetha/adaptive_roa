"""
Latent Conditional Flow Matching implementation using Facebook Flow Matching library

REFACTORED VERSION - Uses GeodesicProbPath and RiemannianODESolver
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
from flow_matching.utils.manifolds import Product, FlatTorus, Euclidean

from adaptive_roa.flow_matching.base.flow_matcher import BaseFlowMatcher
from adaptive_roa.systems.base import DynamicalSystem

class PendulumLatentConditionalFlowMatcher(BaseFlowMatcher):
    """
    Pendulum Latent Conditional Flow Matching using Facebook FM:
    - Uses GeodesicProbPath for geodesic interpolation on S¹×ℝ
    - Uses RiemannianODESolver for manifold-aware ODE integration
    - Neural net takes embedded x_t, time t, latent z, and start state condition
    - Predicts velocity in S¹ × ℝ tangent space

    KEY CHANGES FROM ORIGINAL:
    - ✅ Removed: manual interpolate_s1_x_r() → uses GeodesicProbPath
    - ✅ Removed: manual compute_target_velocity_s1_x_r() → automatic via path.sample()
    - ✅ Added: RiemannianODESolver for inference
    - ✅ Kept: latent variable z, conditioning on start state, all model logic
    """

    def __init__(self,
                 system: DynamicalSystem,
                 model: nn.Module,
                 optimizer,
                 scheduler,
                 model_config: Optional[dict] = None,
                 latent_dim: int = 2,
                 mae_val_frequency: int = 10,
                 use_loss_weights: bool = False,
                 use_manifold: bool = True,
                 use_log_loss_weights: bool = False,
                 clamp_noise: bool = True,
                 zero_latent: bool = False,
                 val_error_log_file: Optional[str] = None,
                 noise_scale: float = 1.0):
        """
        Initialize latent conditional flow matcher with FB FM integration

        Args:
            system: DynamicalSystem (pendulum with S¹ × ℝ structure)
            model: LatentConditionalUNet1D model
            optimizer: Optimizer instance
            scheduler: Learning rate scheduler
            model_config: Configuration dict
            latent_dim: Dimension of latent space
            mae_val_frequency: Compute MAE validation every N epochs
            use_loss_weights: If True, weight loss by normalization limits
            use_manifold: If True, use S¹ manifold for angle; if False, use pure Euclidean ℝ²
            use_log_loss_weights: If True and use_loss_weights=True, use 1+log(limit) for balanced weights
            clamp_noise: If True, clamp noise to [-1, 1] to prevent ODE divergence
            zero_latent: If True, use zero latent vectors instead of random sampling
            val_error_log_file: Path to text file for logging validation errors
            noise_scale: Scale factor for noise in sample_noisy_input (0-1, default 1.0)
        """
        # Store use_manifold BEFORE calling super().__init__ because it calls _create_manifold()
        self.use_manifold = use_manifold
        self.use_log_loss_weights = use_log_loss_weights

        super().__init__(system, model, optimizer, scheduler, model_config, latent_dim, mae_val_frequency, use_loss_weights, clamp_noise, zero_latent, val_error_log_file, noise_scale)

        # Override loss weights for Euclidean mode or log weights
        if use_loss_weights:
            if use_log_loss_weights or not use_manifold:
                loss_weights_2d = self._get_euclidean_loss_weights()
                self.register_buffer('loss_weights', loss_weights_2d)
                weight_type = "1+log(limit)" if use_log_loss_weights else "limit"
                print(f"📊 Loss weights (2D, {weight_type}): {loss_weights_2d.tolist()}")

        manifold_str = "S¹×ℝ (FlatTorus × Euclidean)" if use_manifold else "ℝ² (Euclidean)"

        print("✅ Initialized Pendulum LCFM with Facebook Flow Matching:")
        print(f"   - Manifold: {manifold_str}")
        print(f"   - Path: GeodesicProbPath with CondOTScheduler")
        print(f"   - Latent dim: {latent_dim}")
        print(f"   - MAE validation frequency: every {mae_val_frequency} epochs")
        print(f"   - Use manifold: {use_manifold}")

    def _create_manifold(self):
        """
        Create manifold for Pendulum.

        If use_manifold=True:
            Product manifold S¹×ℝ
            - FlatTorus(1): Angle (θ) - circular
            - Euclidean(1): Angular velocity (θ̇)

        If use_manifold=False:
            Pure Euclidean ℝ² (all dimensions treated as Euclidean)
        """
        if self.use_manifold:
            return Product(input_dim=2, manifolds=[(FlatTorus(), 1), (Euclidean(), 1)])
        else:
            return Euclidean()

    def _create_distance_manifold(self):
        """
        Create manifold for distance computation (always true system manifold).

        Always returns S¹×ℝ regardless of use_manifold setting,
        ensuring proper geodesic distances for the angle component.

        Returns:
            Product manifold with FlatTorus for proper angle distances
        """
        return Product(input_dim=2, manifolds=[(FlatTorus(), 1), (Euclidean(), 1)])

    def _get_euclidean_loss_weights(self) -> torch.Tensor:
        """
        Get 2D loss weights for Euclidean mode or log-weighted mode.

        Components:
        - Angle: weight = π (angle range)
        - Angular velocity: weight from angular_velocity_limit

        If use_log_loss_weights=True, applies 1 + log(limit) transformation.
        """
        import math

        angle_limit = math.pi
        angular_velocity_limit = self.system.angular_velocity_limit

        if self.use_log_loss_weights:
            weights = torch.tensor([
                1.0 + math.log(angle_limit),
                1.0 + math.log(angular_velocity_limit),
            ], dtype=torch.float32)
        else:
            weights = torch.tensor([
                angle_limit,
                angular_velocity_limit,
            ], dtype=torch.float32)

        return weights

    def _get_start_states(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract start states from batch"""
        return batch["start_state"]

    def _get_end_states(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract end states from batch"""
        return batch["end_state"]

    def _get_dimension_name(self, dim_idx: int) -> str:
        """Get human-readable dimension name for pendulum"""
        names = ["angle", "angular_velocity"]
        return names[dim_idx] if 0 <= dim_idx < len(names) else f"dim_{dim_idx}"

    def get_manifold_component_names(self) -> list:
        """
        Get names for manifold distance components.

        For Pendulum with S¹×ℝ:
        - FlatTorus(1) returns 1 distance (geodesic angle distance)
        - Euclidean(1) returns 1 distance (angular velocity)

        Total: 2 components

        Returns:
            List of 2 component names
        """
        return ["angle_geodesic", "angular_velocity"]

    def get_euclidean_groups(self) -> dict:
        """
        Define groups for Euclidean (L2) distance computation.

        For Pendulum (2D state):
        - Angle: index 0 (theta) - note: L2 is NOT proper circular distance
        - Angular velocity: index 1 (theta_dot)

        Returns:
            Dictionary mapping group names to dimension indices
        """
        return {
            "angle_L2": [0],           # Angle (for comparison, geodesic is better)
            "angular_velocity_L2": [1], # Angular velocity
            "full_state_L2": [0, 1],   # Full state L2 norm
        }

    def normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        """Delegate to system for normalization"""
        return self.system.normalize_state(state)

    def denormalize_state(self, normalized_state: torch.Tensor) -> torch.Tensor:
        """Delegate to system for denormalization"""
        return self.system.denormalize_state(normalized_state)

    def embed_state_for_model(self, state: torch.Tensor) -> torch.Tensor:
        """Delegate to system for embedding"""
        return self.system.embed_state_for_model(state)

    def sample_noisy_input(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Sample Gaussian noise in normalized space for S¹ × ℝ manifold.

        Returns noise directly in normalized space (no further normalization needed).
        Circular dimension (θ) is projected onto manifold.
        If self.noise_scale != 1.0, scales the noise to reduce variance.
        If self.clamp_noise is True, clamps to [-1, 1] to prevent ODE divergence.

        Args:
            batch_size: Number of samples
            device: Device to create tensors on

        Returns:
            Noisy states [batch_size, 2] in normalized space
        """
        noisy_input = torch.randn(batch_size, 2, device=device)
        if self.noise_scale != 1.0:
            noisy_input = noisy_input * self.noise_scale
        if self.clamp_noise:
            noisy_input = torch.clamp(noisy_input, -1.0, 1.0)
        noisy_input = self.manifold.projx(noisy_input)
        return noisy_input

    def _wrap_angle(self, state: torch.Tensor) -> torch.Tensor:
        """
        Wrap angle component (index 0) to [-π, π].

        Used when use_manifold=False to ensure valid angle after Euclidean integration.

        Args:
            state: State tensor [B, 2] as (θ, θ̇)

        Returns:
            State with wrapped angle [B, 2]
        """
        result = state.clone()
        # Wrap angle to [-π, π] using atan2(sin, cos)
        result[:, 0] = torch.atan2(torch.sin(state[:, 0]), torch.cos(state[:, 0]))
        return result

    def predict_endpoint(self,
                        start_states: torch.Tensor,
                        num_steps: int = 100,
                        latent: Optional[torch.Tensor] = None,
                        method: str = "euler") -> torch.Tensor:
        """
        Predict endpoints from start states.

        Overrides base class to add angle wrapping when use_manifold=False.

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

        # When use_manifold=False, wrap angle to [-π, π] after Euclidean integration
        if not self.use_manifold:
            endpoints = self._wrap_angle(endpoints)

        return endpoints

    # ===================================================================
    # REMOVED METHODS (now in base class or handled by Facebook FM):
    # ===================================================================
    # ✅ sample_latent() → moved to BaseFlowMatcher
    # ✅ forward() → moved to BaseFlowMatcher
    # ✅ compute_flow_loss() → moved to BaseFlowMatcher (unified implementation)
    # ✅ compute_endpoint_mae_per_dim() → moved to BaseFlowMatcher
    # ✅ validation_step() → moved to BaseFlowMatcher
    # ✅ on_validation_epoch_end() → moved to BaseFlowMatcher
    # ❌ interpolate_s1_x_r() → replaced by self.path.sample()
    # ❌ compute_target_velocity_s1_x_r() → automatic in path_sample.dx_t

    # ===================================================================
    # CHECKPOINT LOADING FOR INFERENCE
    # ===================================================================

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path: str, device: Optional[str] = None):
        """
        Load a trained Pendulum LCFM model from checkpoint for inference.

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
        from adaptive_roa.systems.pendulum import PendulumSystem
        from adaptive_roa.model.pendulum_unet import PendulumUNet

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

            # Parse validation loss from filename: "epoch{epoch:02d}-val_loss{val_loss:.4f}.ckpt"
            # Find checkpoint with lowest validation loss (best model)
            best_checkpoint = None
            best_val_loss = float('inf')

            for ckpt in checkpoints:
                # Extract val_loss from filename
                try:
                    # Example: "epoch42-val_loss0.4519.ckpt"
                    if "val_loss" in ckpt.stem:
                        loss_str = ckpt.stem.split("val_loss")[1]
                        val_loss = float(loss_str)
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            best_checkpoint = ckpt
                except (ValueError, IndexError):
                    continue

            if best_checkpoint is None:
                # Fallback: use most recent checkpoint
                checkpoint_path = max(checkpoints, key=lambda p: p.stat().st_mtime)
                print(f"   ⚠️  Could not parse val_loss, using most recent checkpoint")
            else:
                checkpoint_path = best_checkpoint
                print(f"   ✓ Found best checkpoint (val_loss={best_val_loss:.4f})")

            print(f"   📄 Using: {checkpoint_path.name}")

        print(f"🤖 Loading Pendulum LCFM checkpoint: {checkpoint_path}")
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
        clamp_noise = bool(_require_hparam("clamp_noise"))
        zero_latent = bool(_require_hparam("zero_latent"))
        mae_val_frequency = int(_require_hparam("mae_val_frequency"))
        noise_scale = float(_require_hparam("noise_scale"))
        val_error_log_file = hparams.get("val_error_log_file")

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

        # Remove _target_ key if present (not needed for reconstruction)
        if isinstance(model_config, dict) and "_target_" in model_config:
            model_config = {k: v for k, v in model_config.items() if k != "_target_"}

        model_config["latent_dim"] = latent_dim

        print(f"📋 Config source: {config_source}")
        print(f"📋 Final config - latent_dim: {latent_dim}")
        print(f"📋 Model config keys: {list(model_config.keys())}")
        print(
            "📋 Runtime config - "
            f"use_manifold: {use_manifold}, "
            f"use_loss_weights: {use_loss_weights}, "
            f"use_log_loss_weights: {use_log_loss_weights}, "
            f"clamp_noise: {clamp_noise}, "
            f"zero_latent: {zero_latent}, "
            f"mae_val_frequency: {mae_val_frequency}, "
            f"noise_scale: {noise_scale}"
        )

        # Initialize system and model
        system = hparams.get("system")
        if system is None:
            print("🔧 Creating new Pendulum system (not found in hparams)")
            dataset_dir = hparams.get("system_dataset_dir")
            if not dataset_dir:
                raise KeyError(
                    "Checkpoint is missing 'system_dataset_dir' hyper_parameter required for "
                    "strict Pendulum restoration."
                )
            print(f"   dataset_dir: {dataset_dir}")
            system = PendulumSystem(dataset_dir=dataset_dir)
        else:
            print("✅ Restored Pendulum system from checkpoint")

        # Create model architecture
        model = PendulumUNet(
            embedded_dim=model_config.get('embedded_dim', 3),
            latent_dim=model_config.get('latent_dim', latent_dim),
            condition_dim=model_config.get('condition_dim', 3),
            time_emb_dim=model_config.get('time_emb_dim', 64),
            hidden_dims=model_config.get('hidden_dims', [256, 512, 256]),
            output_dim=model_config.get('output_dim', 2),
            use_input_embeddings=model_config.get('use_input_embeddings', False),
            input_emb_dim=model_config.get('input_emb_dim', 64)
        )

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
        print(f"\n✅ Model loaded successfully!")
        print(f"   Checkpoint: {checkpoint_path.name}")
        print(f"   Config sources: {'Hydra + Lightning' if hydra_config else 'Lightning only'}")
        print(f"   System: {type(system).__name__}")
        print(f"   Latent dim: {latent_dim}")
        print(f"   Use manifold: {use_manifold}")
        print(f"   Model architecture: {model_config.get('hidden_dims', 'unknown')}")
        print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   Device: {device}")

        return flow_matcher
# ============================================================================
# NOTE: VelocityModelWrapper moved to BaseFlowMatcher
# (LatentConditionalVelocityWrapper)
# ============================================================================

