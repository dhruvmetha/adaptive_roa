"""HumanoidStandUpReach Latent Conditional Flow Matching (Facebook FM).

Manifold: ℝ³⁴ × S² × ℝ³⁰ (67-D state). The sphere block (dims 34:37) uses the FB FM
ambient-3 tangent (singularity-free), so model output_dim = 67 in both manifold modes.
"""
from typing import Dict, Optional

import torch
import torch.nn as nn

from flow_matching.utils.manifolds import Product, Euclidean, Sphere

from adaptive_roa.flow_matching.base.flow_matcher import BaseFlowMatcher
from adaptive_roa.systems.base import DynamicalSystem

SPHERE_START, SPHERE_END = 34, 37
_DIM_NAMES = (
    [f"joint_{i}" for i in range(21)] + ["head_height"]
    + [f"extremity_{i}" for i in range(12)]
    + ["torso_vx", "torso_vy", "torso_vz"]
    + ["com_vx", "com_vy", "com_vz"]
    + [f"vel_{i}" for i in range(27)]
)


class HumanoidStandUpReachLatentConditionalFlowMatcher(BaseFlowMatcher):
    def __init__(self, system: DynamicalSystem, model: nn.Module, optimizer, scheduler,
                 model_config: Optional[dict] = None, latent_dim: int = 8,
                 mae_val_frequency: int = 10, use_loss_weights: bool = False,
                 use_manifold: bool = True, use_log_loss_weights: bool = False,
                 clamp_noise: bool = True,
                 zero_latent: bool = False, val_error_log_file: Optional[str] = None,
                 noise_scale: float = 1.0):
        # Must be set before super().__init__ (it calls _create_manifold()).
        self.use_manifold = use_manifold
        self.use_log_loss_weights = use_log_loss_weights
        super().__init__(system, model, optimizer, scheduler, model_config, latent_dim,
                         mae_val_frequency, use_loss_weights, clamp_noise, zero_latent,
                         val_error_log_file, noise_scale)
        print(f"✅ HumanoidStandUpReach LCFM | manifold={use_manifold} | latent={latent_dim}")

    # ---- manifolds ---------------------------------------------------------
    def _create_manifold(self):
        if self.use_manifold:
            return Product(input_dim=67, manifolds=[
                (Euclidean(), 34, 34), (Sphere(), 3, 3), (Euclidean(), 30, 30)])
        return Euclidean()

    def _create_distance_manifold(self):
        # Always the S²-aware product (65 distance components), regardless of use_manifold.
        return Product(input_dim=67, manifolds=[
            (Euclidean(), 34, 34), (Sphere(), 3, 3), (Euclidean(), 30, 30)])

    def get_manifold_component_names(self) -> list:
        # Euclidean(34) -> 34 per-dim + Sphere -> 1 geodesic + Euclidean(30) -> 30 = 65
        names = [f"e1_{i}" for i in range(34)] + ["torso_vertical_geo"] + [f"e2_{i}" for i in range(30)]
        return names

    # ---- noise -------------------------------------------------------------
    def sample_noisy_input(self, batch_size: int, device: torch.device) -> torch.Tensor:
        euclid1 = torch.randn(batch_size, 34, device=device)
        sphere = torch.randn(batch_size, 3, device=device)
        euclid2 = torch.randn(batch_size, 30, device=device)
        if self.noise_scale != 1.0:
            # noise_scale is a no-op on the sphere block: unit-normalization below
            # cancels it (mirrors Quad3D). Only the Euclidean blocks are scaled.
            euclid1 *= self.noise_scale; sphere *= self.noise_scale; euclid2 *= self.noise_scale
        if self.clamp_noise:
            euclid1 = torch.clamp(euclid1, -1.0, 1.0)
            euclid2 = torch.clamp(euclid2, -1.0, 1.0)
        sphere = sphere / sphere.norm(dim=1, keepdim=True).clamp(min=1e-8)
        noisy = torch.cat([euclid1, sphere, euclid2], dim=1)
        return self.manifold.projx(noisy)

    # ---- batch accessors / delegation -------------------------------------
    def _get_start_states(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        return batch["start_state"]

    def _get_end_states(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        return batch["end_state"]

    def _get_dimension_name(self, dim_idx: int) -> str:
        return _DIM_NAMES[dim_idx] if 0 <= dim_idx < len(_DIM_NAMES) else f"dim_{dim_idx}"

    def normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        return self.system.normalize_state(state)

    def denormalize_state(self, normalized_state: torch.Tensor) -> torch.Tensor:
        return self.system.denormalize_state(normalized_state)

    def embed_state_for_model(self, normalized_state: torch.Tensor) -> torch.Tensor:
        return self.system.embed_state_for_model(normalized_state)

    # ---- prediction --------------------------------------------------------
    def predict_endpoint(self, start_states: torch.Tensor, num_steps: int = 100,
                         latent: Optional[torch.Tensor] = None,
                         method: Optional[str] = None) -> torch.Tensor:
        # Auto-select integrator so use_manifold is a genuine toggle: Riemannian
        # for the S²-aware product, plain euler for the flat Euclidean() path.
        if method is None:
            method = "euler_riemannian" if self.use_manifold else "euler"
        endpoints = super().predict_endpoint(start_states, num_steps, latent, method)
        return self.system.project_to_manifold(endpoints)

    def predict_endpoints_batch(self, start_states: torch.Tensor, num_steps: int = 100,
                                num_samples: int = 1) -> torch.Tensor:
        if num_samples == 1:
            return self.predict_endpoint(start_states, num_steps)
        return torch.cat([self.predict_endpoint(start_states, num_steps, latent=None)
                          for _ in range(num_samples)], dim=0)

    # ---- checkpoint loading ------------------------------------------------
    @classmethod
    def load_from_checkpoint(cls, checkpoint_path: str, device: Optional[str] = None):
        """Reconstruct a trained HumanoidStandUpReach LCFM from a checkpoint.

        The base ``save_hyperparameters(ignore=[model, optimizer, scheduler, system])`` drops
        the un-serializable constructor objects, so Lightning's default loader can't rebuild
        the module. Mirroring the Quadrotor3D loader, we rebuild the system from the saved
        ``system_dataset_dir`` and the model from ``model_config``, then load the ``model.*``
        weights. Accepts either a ``.ckpt`` file or a training/epoch folder.
        """
        from pathlib import Path
        from adaptive_roa.systems.humanoid_standup_reach import HumanoidStandUpReachSystem
        from adaptive_roa.flow_matching.base.checkpoint_utils import (
            instantiate_model_from_config, find_training_dir, load_hydra_config,
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
            ckpts = [p for p in checkpoint_dir.glob("best*.ckpt")] or \
                    [p for p in checkpoint_dir.glob("*.ckpt") if p.name != "last.ckpt"]
            if not ckpts:
                raise FileNotFoundError(f"No .ckpt files found in {checkpoint_dir}")
            checkpoint_path = max(ckpts, key=lambda p: p.stat().st_mtime)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        hparams = checkpoint.get("hyper_parameters", {})

        def _req(key):
            if key not in hparams:
                raise KeyError(f"Missing required checkpoint hyper_parameter '{key}' in {checkpoint_path}")
            return hparams[key]

        latent_dim = int(_req("latent_dim"))
        model_config = dict(_req("model_config"))
        model_config["latent_dim"] = latent_dim

        system = hparams.get("system")
        if system is None:
            system = HumanoidStandUpReachSystem(dataset_dir=_req("system_dataset_dir"))

        model = instantiate_model_from_config(model_config, latent_dim)

        flow_matcher = cls(
            system=system, model=model, optimizer=None, scheduler=None,
            model_config=model_config, latent_dim=latent_dim,
            mae_val_frequency=int(_req("mae_val_frequency")),
            use_loss_weights=bool(_req("use_loss_weights")),
            use_manifold=bool(_req("use_manifold")),
            use_log_loss_weights=bool(_req("use_log_loss_weights")),
            clamp_noise=bool(_req("clamp_noise")),
            zero_latent=bool(_req("zero_latent")),
            val_error_log_file=hparams.get("val_error_log_file"),
            noise_scale=float(_req("noise_scale")),
        )

        state_dict = checkpoint["state_dict"]
        model_state_dict = {k[len("model."):]: v for k, v in state_dict.items() if k.startswith("model.")}
        if not model_state_dict:
            raise ValueError("No 'model.*' weights in checkpoint: " + str(list(state_dict.keys())[:10]))
        flow_matcher.model.load_state_dict(model_state_dict)

        flow_matcher = flow_matcher.to(device)
        flow_matcher.eval()
        try:
            flow_matcher.training_config = load_hydra_config(find_training_dir(checkpoint_path))
        except Exception:
            flow_matcher.training_config = None
        return flow_matcher
