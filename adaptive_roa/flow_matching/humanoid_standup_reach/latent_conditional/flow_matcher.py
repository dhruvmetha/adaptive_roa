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
                 use_manifold: bool = True, clamp_noise: bool = True,
                 zero_latent: bool = False, val_error_log_file: Optional[str] = None,
                 noise_scale: float = 1.0):
        # Must be set before super().__init__ (it calls _create_manifold()).
        self.use_manifold = use_manifold
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
