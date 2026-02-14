"""
Quadrotor 2D Latent Conditional Flow Matching implementation using Facebook Flow Matching library

Manifold: ℝ² × S¹ × ℝ³ (position × pitch angle × velocities)
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


class Quadrotor2DLatentConditionalFlowMatcher(BaseFlowMatcher):
    """
    Quadrotor 2D Latent Conditional Flow Matching using Facebook FM:
    - Uses GeodesicProbPath for geodesic interpolation on ℝ²×S¹×ℝ³
    - Uses RiemannianODESolver for manifold-aware ODE integration
    - Neural net takes embedded x_t, time t, latent z, and start state condition
    - Predicts velocity in ℝ²×S¹×ℝ³ tangent space (x, z, θ, ẋ, ż, θ̇)
    """

    def __init__(self,
                 system: DynamicalSystem,
                 model: nn.Module,
                 optimizer,
                 scheduler,
                 model_config: Optional[dict] = None,
                 latent_dim: int = 0,
                 mae_val_frequency: int = 10,
                 use_loss_weights: bool = False,
                 use_manifold: bool = True,
                 use_log_loss_weights: bool = False,
                 clamp_noise: bool = True,
                 zero_latent: bool = False,
                 val_error_log_file: Optional[str] = None,
                 noise_scale: float = 1.0):
        """
        Initialize Quadrotor 2D latent conditional flow matcher with FB FM integration

        Args:
            system: DynamicalSystem (Quadrotor2D with ℝ²×S¹×ℝ³ structure)
            model: Quadrotor2DUNet model
            optimizer: Optimizer instance
            scheduler: Learning rate scheduler
            model_config: Configuration dict
            latent_dim: Dimension of latent space
            mae_val_frequency: Compute MAE validation every N epochs
            use_loss_weights: If True, weight loss by normalization limits
            use_manifold: If True, use S¹ manifold for angle; if False, use pure Euclidean ℝ⁶
            use_log_loss_weights: If True and use_loss_weights=True, use 1+log(limit) for balanced weights
            clamp_noise: If True, clamp noise to [-1, 1] to prevent ODE divergence
            zero_latent: If True, use zero latent vectors instead of random sampling
            val_error_log_file: Path to text file for logging validation errors
            noise_scale: Scale factor for noise in sample_noisy_input (0-1, default 1.0)
        """
        self.use_manifold = use_manifold
        self.use_log_loss_weights = use_log_loss_weights

        super().__init__(system, model, optimizer, scheduler, model_config, latent_dim, mae_val_frequency, use_loss_weights, clamp_noise, zero_latent, val_error_log_file, noise_scale)

        # Override loss weights for Euclidean mode or log weights
        if use_loss_weights:
            if use_log_loss_weights or not use_manifold:
                loss_weights_6d = self._get_euclidean_loss_weights()
                self.register_buffer('loss_weights', loss_weights_6d)
                weight_type = "1+log(limit)" if use_log_loss_weights else "limit"
                print(f"Loss weights (6D, {weight_type}): {loss_weights_6d.tolist()}")

        manifold_str = "ℝ²×S¹×ℝ³ (Euclidean × FlatTorus × Euclidean)" if use_manifold else "ℝ⁶ (Euclidean)"

        print("Initialized Quadrotor2D LCFM with Facebook Flow Matching:")
        print(f"   - Manifold: {manifold_str}")
        print(f"   - Path: GeodesicProbPath with CondOTScheduler")
        print(f"   - Latent dim: {latent_dim}")
        print(f"   - MAE validation frequency: every {mae_val_frequency} epochs")
        print(f"   - Use manifold: {use_manifold}")

    def _create_manifold(self):
        """
        Create manifold for Quadrotor 2D.

        If use_manifold=True:
            Product manifold ℝ²×S¹×ℝ³
            - Euclidean(2): Position (x, z)
            - FlatTorus(1): Pitch angle (θ) - circular
            - Euclidean(3): Velocities (ẋ, ż, θ̇)

        If use_manifold=False:
            Pure Euclidean ℝ⁶ (all dimensions treated as Euclidean)
        """
        if self.use_manifold:
            return Product(input_dim=6, manifolds=[(Euclidean(), 2), (FlatTorus(), 1), (Euclidean(), 3)])
        else:
            return Euclidean()

    def _create_distance_manifold(self):
        """
        Create manifold for distance computation (always true system manifold).

        Always returns ℝ²×S¹×ℝ³ regardless of use_manifold setting,
        ensuring proper geodesic distances for the pitch angle component.

        Returns:
            Product manifold with FlatTorus for proper angle distances
        """
        return Product(input_dim=6, manifolds=[(Euclidean(), 2), (FlatTorus(), 1), (Euclidean(), 3)])

    def _get_euclidean_loss_weights(self) -> torch.Tensor:
        """
        Get 6D loss weights for Euclidean mode or log-weighted mode.

        Components:
        - x position: weight from x_limit
        - z position: weight from z_scale (asymmetric)
        - Pitch angle: weight = π (angle range)
        - x velocity: weight from x_dot_limit
        - z velocity: weight from z_dot_limit
        - Pitch rate: weight from theta_dot_limit

        If use_log_loss_weights=True, applies 1 + log(limit) transformation.
        """
        import math

        x_limit = self.system.x_limit
        z_scale = self.system.z_scale
        angle_limit = math.pi
        x_dot_limit = self.system.x_dot_limit
        z_dot_limit = self.system.z_dot_limit
        theta_dot_limit = self.system.theta_dot_limit

        limits = [x_limit, z_scale, angle_limit, x_dot_limit, z_dot_limit, theta_dot_limit]

        if self.use_log_loss_weights:
            weights = [1.0 + math.log(limit) if limit > 1 else 1.0 for limit in limits]
        else:
            weights = limits

        return torch.tensor(weights, dtype=torch.float32)

    def _get_start_states(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract start states from batch"""
        return batch["start_state"]

    def _get_end_states(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract end states from batch"""
        return batch["end_state"]

    def _get_dimension_name(self, dim_idx: int) -> str:
        """Get human-readable dimension name for Quadrotor 2D"""
        names = ["x_position", "z_position", "pitch_angle", "x_velocity", "z_velocity", "pitch_rate"]
        return names[dim_idx] if 0 <= dim_idx < len(names) else f"dim_{dim_idx}"

    def get_manifold_component_names(self) -> list:
        """
        Get names for manifold distance components.

        For Quadrotor 2D with ℝ²×S¹×ℝ³:
        - Euclidean(2) returns 2 distances (x, z)
        - FlatTorus(1) returns 1 distance (geodesic angle distance)
        - Euclidean(3) returns 3 distances (ẋ, ż, θ̇)

        Total: 6 components

        Returns:
            List of 6 component names
        """
        return ["x_position", "z_position", "pitch_angle_geodesic",
                "x_velocity", "z_velocity", "pitch_rate"]

    def get_euclidean_groups(self) -> dict:
        """
        Define groups for Euclidean (L2) distance computation.

        For Quadrotor 2D (6D state):
        - Position: indices 0, 1 (x, z)
        - Pitch angle: index 2 (θ)
        - Velocity: indices 3, 4, 5 (ẋ, ż, θ̇)

        Returns:
            Dictionary mapping group names to dimension indices
        """
        return {
            "x_position_L2": [0],
            "z_position_L2": [1],
            "pitch_angle_L2": [2],
            "x_velocity_L2": [3],
            "z_velocity_L2": [4],
            "pitch_rate_L2": [5],
            "position_L2": [0, 1],
            "velocity_L2": [3, 4, 5],
            "full_state_L2": [0, 1, 2, 3, 4, 5],
        }

    def sample_noisy_input(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Sample Gaussian noise in normalized space for ℝ²×S¹×ℝ³ manifold.

        Returns noise directly in normalized space (no further normalization needed).
        Circular dimension (θ) is projected onto manifold.

        Args:
            batch_size: Number of samples
            device: Device to create tensors on

        Returns:
            Noisy states [batch_size, 6] in normalized space
        """
        noisy_input = torch.randn(batch_size, 6, device=device)
        if self.noise_scale != 1.0:
            noisy_input = noisy_input * self.noise_scale
        if self.clamp_noise:
            noisy_input = torch.clamp(noisy_input, -1.0, 1.0)
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

    def _wrap_angle(self, state: torch.Tensor) -> torch.Tensor:
        """
        Wrap pitch angle component (index 2) to [-π, π].

        Used when use_manifold=False to ensure valid angle after Euclidean integration.

        Args:
            state: State tensor [B, 6] as (x, z, θ, ẋ, ż, θ̇)

        Returns:
            State with wrapped angle [B, 6]
        """
        result = state.clone()
        # Wrap pitch angle (index 2) to [-π, π] using atan2(sin, cos)
        result[:, 2] = torch.atan2(torch.sin(state[:, 2]), torch.cos(state[:, 2]))
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

        # When use_manifold=False, wrap pitch angle to [-π, π] after Euclidean integration
        if not self.use_manifold:
            endpoints = self._wrap_angle(endpoints)

        return endpoints

    def predict_endpoints_batch(self,
                               start_states: torch.Tensor,
                               num_steps: int = 100,
                               num_samples: int = 1) -> torch.Tensor:
        """
        Predict multiple endpoint samples per start state (for stochastic models).

        Args:
            start_states: Start states [B, 6] in raw coordinates
            num_steps: Number of integration steps
            num_samples: Number of samples per start state

        Returns:
            Predicted endpoints [B*num_samples, 6] in raw coordinates
        """
        if num_samples == 1:
            raw_endpoints = self.predict_endpoint(start_states, num_steps)
            return raw_endpoints

        all_endpoints = []
        for _ in range(num_samples):
            endpoints_raw = self.predict_endpoint(start_states, num_steps, latent=None)
            all_endpoints.append(endpoints_raw)

        return torch.cat(all_endpoints, dim=0)

    # ===================================================================
    # CHECKPOINT LOADING FOR INFERENCE
    # ===================================================================

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path: str, device: Optional[str] = None):
        """
        Load a trained Quadrotor 2D LCFM model from checkpoint for inference.

        Args:
            checkpoint_path: Path to Lightning checkpoint file (.ckpt) OR training folder
            device: Device to load model on ("cuda", "cpu", or None for auto)

        Returns:
            Loaded model ready for inference
        """
        import torch
        from pathlib import Path
        from adaptive_roa.systems.quadrotor2d import Quadrotor2DSystem
        from adaptive_roa.flow_matching.base.checkpoint_utils import instantiate_model_from_config

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        checkpoint_path = Path(checkpoint_path)

        # Check if it's a folder or a .ckpt file
        if checkpoint_path.is_dir():
            print(f"Folder provided: {checkpoint_path}")
            print(f"Searching for checkpoint in folder...")

            # Try v2 adaptive layout first, then legacy standalone layout
            checkpoint_dir = checkpoint_path / "checkpoints"
            if not checkpoint_dir.exists():
                checkpoint_dir = checkpoint_path / "version_0" / "checkpoints"
            if not checkpoint_dir.exists():
                raise FileNotFoundError(f"No checkpoints directory found in {checkpoint_path}")

            checkpoints = [p for p in checkpoint_dir.glob("*.ckpt") if p.name != "last.ckpt"]
            if not checkpoints:
                raise FileNotFoundError(f"No .ckpt files found in {checkpoint_dir}")

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
                print(f"   Could not parse val_loss, using most recent checkpoint")
            else:
                checkpoint_path = best_checkpoint
                print(f"   Found best checkpoint (val_loss={best_val_loss:.4f})")

            print(f"   Using: {checkpoint_path.name}")

        print(f"Loading Quadrotor2D LCFM checkpoint: {checkpoint_path}")
        print(f"Device: {device}")

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Find training directory and load Hydra config
        from adaptive_roa.flow_matching.base.checkpoint_utils import find_training_dir, load_hydra_config
        training_dir = find_training_dir(checkpoint_path)
        print(f"Training directory: {training_dir}")
        hydra_config = load_hydra_config(training_dir)

        # Load Lightning checkpoint
        print(f"Loading Lightning checkpoint...")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        hparams = checkpoint.get("hyper_parameters", {})
        print("Lightning checkpoint loaded")

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
            f"clamp_noise: {clamp_noise}, "
            f"zero_latent: {zero_latent}, "
            f"mae_val_frequency: {mae_val_frequency}, "
            f"noise_scale: {noise_scale}"
        )

        # Initialize system and model
        system = hparams.get("system")
        if system is None:
            print("Creating new Quadrotor2D system (not found in hparams)")
            dataset_dir = hparams.get("system_dataset_dir")
            if not dataset_dir:
                raise KeyError(
                    "Checkpoint is missing 'system_dataset_dir' hyper_parameter required for "
                    "strict Quadrotor2D restoration. Re-train with updated code."
                )
            print(f"   dataset_dir: {dataset_dir}")
            system = Quadrotor2DSystem(dataset_dir=dataset_dir)
        else:
            print("Restored Quadrotor2D system from checkpoint")

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
            clamp_noise=clamp_noise,
            zero_latent=zero_latent,
            val_error_log_file=val_error_log_file,
            noise_scale=noise_scale,
        )

        # Load model weights
        print("Loading model state dict...")
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

        print(f"\nModel loaded successfully!")
        print(f"   Checkpoint: {checkpoint_path.name}")
        print(f"   Config sources: {'Hydra + Lightning' if hydra_config else 'Lightning only'}")
        print(f"   System: {type(system).__name__}")
        print(f"   Latent dim: {latent_dim}")
        model_info = model.get_model_info() if hasattr(model, 'get_model_info') else {}
        print(f"   Model type: {model_info.get('model_type', 'unknown')}")
        print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   Device: {device}")

        return flow_matcher
