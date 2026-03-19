"""
Pendulum Trajectory Flow Matcher.

Concrete implementation of TrajectoryFlowMatcherBase for the pendulum system.
Uses S^1 x R manifold (FlatTorus x Euclidean) — same as the endpoint-level
PendulumLatentConditionalFlowMatcher.

Predicts full trajectory sequences [B, T, 2] and extracts the final timestep
to satisfy the predict_endpoint() interface used by the adaptive engine.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
from pathlib import Path

from flow_matching.utils.manifolds import Product, FlatTorus, Euclidean

from adaptive_roa.flow_matching.base.trajectory_flow_matcher import TrajectoryFlowMatcherBase
from adaptive_roa.systems.base import DynamicalSystem


class PendulumTrajectoryFlowMatcher(TrajectoryFlowMatcherBase):
    """
    Pendulum trajectory-level flow matching.

    Manifold: Product(FlatTorus(1), Euclidean(1)) — S^1 x R
    State: [theta, theta_dot] (2D)
    Embedding: [sin(theta), cos(theta), theta_dot_norm] (3D)
    Output: velocity in tangent space (2D)

    Training: learns to denoise full trajectories [B, T, 2]
    Inference: generates trajectory, returns final timestep [B, 2]
    """

    def __init__(
        self,
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
        noise_scale: float = 1.0,
        sequence_length: int = 32,
        history_length: int = 1,
    ):
        self.use_manifold = use_manifold
        self.use_log_loss_weights = use_log_loss_weights

        super().__init__(
            system=system,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            model_config=model_config,
            latent_dim=latent_dim,
            mae_val_frequency=mae_val_frequency,
            use_loss_weights=use_loss_weights,
            clamp_noise=clamp_noise,
            zero_latent=zero_latent,
            val_error_log_file=val_error_log_file,
            noise_scale=noise_scale,
            sequence_length=sequence_length,
            history_length=history_length,
        )

        # Override loss weights for log mode
        if use_loss_weights and use_log_loss_weights:
            import math
            angle_limit = math.pi
            angular_velocity_limit = self.system.angular_velocity_limit
            weights = torch.tensor([
                1.0 + math.log(angle_limit),
                1.0 + math.log(angular_velocity_limit),
            ], dtype=torch.float32)
            self.register_buffer('loss_weights', weights)

        manifold_str = "S^1 x R (FlatTorus x Euclidean)" if use_manifold else "R^2 (Euclidean)"
        print(f"Initialized PendulumTrajectoryFlowMatcher:")
        print(f"  Manifold: {manifold_str}")
        print(f"  Sequence length: {sequence_length}")
        print(f"  History length: {history_length}")
        print(f"  Latent dim: {latent_dim}")

    # ===================================================================
    # Abstract method implementations
    # ===================================================================

    def _create_manifold(self):
        if self.use_manifold:
            return Product(input_dim=2, manifolds=[(FlatTorus(), 1), (Euclidean(), 1)])
        else:
            return Euclidean()

    def _create_distance_manifold(self):
        """Always use true manifold for distances."""
        return Product(input_dim=2, manifolds=[(FlatTorus(), 1), (Euclidean(), 1)])

    def sample_noisy_input(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample noise for single endpoint [B, 2] (used by base class validation)."""
        noisy_input = torch.randn(batch_size, 2, device=device)
        if self.noise_scale != 1.0:
            noisy_input = noisy_input * self.noise_scale
        if self.clamp_noise:
            noisy_input = torch.clamp(noisy_input, -1.0, 1.0)
        noisy_input = self.manifold.projx(noisy_input)
        return noisy_input

    def _get_start_states(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        return batch["start_state"]

    def _get_end_states(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        return batch["end_state"]

    def _get_dimension_name(self, dim_idx: int) -> str:
        names = ["angle", "angular_velocity"]
        return names[dim_idx] if 0 <= dim_idx < len(names) else f"dim_{dim_idx}"

    def get_manifold_component_names(self) -> list:
        return ["angle_geodesic", "angular_velocity"]

    def get_euclidean_groups(self) -> dict:
        return {
            "angle_L2": [0],
            "angular_velocity_L2": [1],
            "full_state_L2": [0, 1],
        }

    def normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        return self.system.normalize_state(state)

    def denormalize_state(self, normalized_state: torch.Tensor) -> torch.Tensor:
        return self.system.denormalize_state(normalized_state)

    def embed_state_for_model(self, state: torch.Tensor) -> torch.Tensor:
        return self.system.embed_state_for_model(state)

    def _wrap_angle(self, state: torch.Tensor) -> torch.Tensor:
        """Wrap angle component (index 0) to [-pi, pi]."""
        result = state.clone()
        result[..., 0] = torch.atan2(torch.sin(state[..., 0]), torch.cos(state[..., 0]))
        return result

    def predict_endpoint(
        self,
        start_states: torch.Tensor,
        num_steps: int = 100,
        latent: Optional[torch.Tensor] = None,
        method: str = "euler",
    ) -> torch.Tensor:
        """Override to add angle wrapping when use_manifold=False."""
        endpoints = super().predict_endpoint(start_states, num_steps, latent, method)
        if not self.use_manifold:
            endpoints = self._wrap_angle(endpoints)
        return endpoints

    # ===================================================================
    # Checkpoint loading
    # ===================================================================

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path: str, device: Optional[str] = None):
        """
        Load a trained PendulumTrajectoryFlowMatcher from checkpoint.

        Args:
            checkpoint_path: Path to .ckpt file or training folder
            device: Device to load on

        Returns:
            Loaded model ready for inference
        """
        from adaptive_roa.systems.pendulum import PendulumSystem
        from adaptive_roa.flow_matching.base.checkpoint_utils import (
            instantiate_model_from_config, find_training_dir, load_hydra_config
        )

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        checkpoint_path = Path(checkpoint_path)

        if checkpoint_path.is_dir():
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

            checkpoint_path = best_checkpoint or max(checkpoints, key=lambda p: p.stat().st_mtime)

        print(f"Loading PendulumTrajectoryFlowMatcher: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        hparams = checkpoint.get("hyper_parameters", {})

        training_dir = find_training_dir(checkpoint_path)
        hydra_config = load_hydra_config(training_dir)

        def _hp(key, default=None):
            if key in hparams:
                return hparams[key]
            if default is not None:
                return default
            raise KeyError(f"Missing hyper_parameter '{key}'")

        latent_dim = int(_hp("latent_dim"))
        use_manifold = bool(_hp("use_manifold", True))
        use_loss_weights = bool(_hp("use_loss_weights", False))
        use_log_loss_weights = bool(_hp("use_log_loss_weights", False))
        clamp_noise = bool(_hp("clamp_noise", True))
        zero_latent = bool(_hp("zero_latent", False))
        mae_val_frequency = int(_hp("mae_val_frequency", 10))
        noise_scale = float(_hp("noise_scale", 1.0))
        sequence_length = int(_hp("sequence_length", 32))
        history_length = int(_hp("history_length", 1))
        val_error_log_file = hparams.get("val_error_log_file")

        model_config = hparams.get("model_config") or hparams.get("config")
        if model_config is None:
            raise KeyError("Checkpoint missing model config")
        model_config["latent_dim"] = latent_dim

        system = hparams.get("system")
        if system is None:
            dataset_dir = hparams.get("system_dataset_dir")
            if not dataset_dir:
                raise KeyError("Missing system_dataset_dir")
            system = PendulumSystem(dataset_dir=dataset_dir)

        model = instantiate_model_from_config(model_config, latent_dim)

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
            sequence_length=sequence_length,
            history_length=history_length,
        )

        state_dict = checkpoint["state_dict"]
        model_state_dict = {
            k.replace("model.", ""): v
            for k, v in state_dict.items()
            if k.startswith("model.")
        }
        flow_matcher.model.load_state_dict(model_state_dict)
        flow_matcher = flow_matcher.to(device)
        flow_matcher.eval()
        flow_matcher.training_config = hydra_config

        print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} params")
        return flow_matcher
