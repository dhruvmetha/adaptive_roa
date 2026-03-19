"""
Quadrotor 3D Trajectory Flow Matcher.

Manifold: R^3 x SO(3) x R^6 (position x orientation x velocities)
State: [x, y, z, qw, qx, qy, qz, x_dot, y_dot, z_dot, p, q, r] (13D)
Embedding: same 13D (quaternion stays as-is in embedding)
Output: velocity in tangent space (12D manifold / 13D Euclidean)

NOTE: Quadrotor3D has mixed state/tangent dimensions due to SO(3).
The manifold.projx handles quaternion normalization per-timestep.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional

from flow_matching.utils.manifolds import Product, Euclidean, SO3

from adaptive_roa.flow_matching.base.trajectory_flow_matcher import TrajectoryFlowMatcherBase
from adaptive_roa.systems.base import DynamicalSystem


class Quadrotor3DTrajectoryFlowMatcher(TrajectoryFlowMatcherBase):

    def __init__(self, system, model, optimizer, scheduler,
                 model_config=None, latent_dim=4, mae_val_frequency=10,
                 use_loss_weights=False, use_manifold=True, use_log_loss_weights=False,
                 quat_loss_weight=1.0, clamp_noise=True, zero_latent=False,
                 val_error_log_file=None, noise_scale=1.0,
                 sequence_length=32, history_length=1):
        self.use_manifold = use_manifold
        self.use_log_loss_weights = use_log_loss_weights
        self.quat_loss_weight = quat_loss_weight
        super().__init__(
            system=system, model=model, optimizer=optimizer, scheduler=scheduler,
            model_config=model_config, latent_dim=latent_dim,
            mae_val_frequency=mae_val_frequency, use_loss_weights=use_loss_weights,
            clamp_noise=clamp_noise, zero_latent=zero_latent,
            val_error_log_file=val_error_log_file, noise_scale=noise_scale,
            sequence_length=sequence_length, history_length=history_length,
        )
        print(f"Initialized Quadrotor3DTrajectoryFlowMatcher: seq_len={sequence_length}, history={history_length}")

    def _create_manifold(self):
        if self.use_manifold:
            return Product(input_dim=13, manifolds=[
                (Euclidean(), 3), (SO3(), 4, 3), (Euclidean(), 6)])
        return Euclidean()

    def _create_distance_manifold(self):
        return Product(input_dim=13, manifolds=[
            (Euclidean(), 3), (SO3(), 4, 3), (Euclidean(), 6)])

    def sample_noisy_input(self, batch_size, device):
        """Sample noise with proper SO(3) quaternion sampling."""
        position = torch.randn(batch_size, 3, device=device)
        if self.noise_scale != 1.0:
            position = position * self.noise_scale
        if self.clamp_noise:
            position = torch.clamp(position, -1.0, 1.0)

        # Uniform SO(3) via normalized Gaussian
        quat = torch.randn(batch_size, 4, device=device)
        quat = quat / torch.norm(quat, dim=1, keepdim=True).clamp(min=1e-8)
        sign = torch.sign(quat[:, 0:1])
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        quat = quat * sign

        velocities = torch.randn(batch_size, 6, device=device)
        if self.noise_scale != 1.0:
            velocities = velocities * self.noise_scale
        if self.clamp_noise:
            velocities = torch.clamp(velocities, -1.0, 1.0)

        noisy = torch.cat([position, quat, velocities], dim=1)
        return self.manifold.projx(noisy)

    def sample_noisy_trajectory_input(self, batch_size, device):
        """Override: sample per-timestep noise with proper quaternion handling."""
        B, T, D = batch_size, self.sequence_length, 13
        # Sample per-timestep
        all_noise = []
        for _ in range(T):
            all_noise.append(self.sample_noisy_input(B, device))
        return torch.stack(all_noise, dim=1)  # [B, T, 13]

    def _get_start_states(self, batch):
        return batch["start_state"]

    def _get_end_states(self, batch):
        return batch["end_state"]

    def _get_dimension_name(self, dim_idx):
        names = ["x", "y", "z", "qw", "qx", "qy", "qz",
                 "x_dot", "y_dot", "z_dot", "p", "q", "r"]
        return names[dim_idx] if 0 <= dim_idx < len(names) else f"dim_{dim_idx}"

    def get_manifold_component_names(self):
        return ["pos_x", "pos_y", "pos_z", "orientation_geodesic",
                "vel_x", "vel_y", "vel_z", "ang_p", "ang_q", "ang_r"]

    def get_euclidean_groups(self):
        return {
            "position_L2": [0, 1, 2],
            "quaternion_L2": [3, 4, 5, 6],
            "linear_velocity_L2": [7, 8, 9],
            "angular_velocity_L2": [10, 11, 12],
            "full_state_L2": list(range(13)),
        }

    def normalize_state(self, state):
        return self.system.normalize_state(state)

    def denormalize_state(self, state):
        return self.system.denormalize_state(state)

    def embed_state_for_model(self, state):
        return self.system.embed_state_for_model(state)

    def predict_endpoint(self, start_states, num_steps=100, latent=None, method="euler"):
        endpoints = super().predict_endpoint(start_states, num_steps, latent, method)
        # Always project quaternion and canonicalize
        if hasattr(self.system, "project_to_manifold"):
            endpoints = self.system.project_to_manifold(endpoints)
        else:
            result = endpoints.clone()
            quat = result[:, 3:7]
            quat = quat / quat.norm(dim=1, keepdim=True).clamp(min=1e-8)
            sign = torch.sign(quat[:, 0:1])
            sign = torch.where(sign == 0, torch.ones_like(sign), sign)
            result[:, 3:7] = quat * sign
            endpoints = result
        return endpoints

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, device=None):
        from pathlib import Path
        from adaptive_roa.systems.quadrotor3d import Quadrotor3DSystem
        from adaptive_roa.flow_matching.base.checkpoint_utils import (
            instantiate_model_from_config, find_training_dir, load_hydra_config)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        checkpoint_path = Path(checkpoint_path)
        if checkpoint_path.is_dir():
            ckpt_dir = checkpoint_path / "checkpoints"
            if not ckpt_dir.exists():
                ckpt_dir = checkpoint_path / "version_0" / "checkpoints"
            ckpts = [p for p in ckpt_dir.glob("*.ckpt") if p.name != "last.ckpt"]
            best, best_loss = None, float('inf')
            for c in ckpts:
                try:
                    if "val_loss" in c.stem:
                        vl = float(c.stem.split("val_loss")[1])
                        if vl < best_loss:
                            best_loss, best = vl, c
                except (ValueError, IndexError):
                    pass
            checkpoint_path = best or max(ckpts, key=lambda p: p.stat().st_mtime)

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        hp = checkpoint.get("hyper_parameters", {})
        hydra_config = load_hydra_config(find_training_dir(checkpoint_path))

        def _hp(key, default=None):
            if key in hp:
                return hp[key]
            if default is not None:
                return default
            raise KeyError(f"Missing hyper_parameter '{key}' in checkpoint")

        mc = hp.get("model_config") or hp.get("config")
        if mc is None:
            raise KeyError("Checkpoint missing model config")
        mc["latent_dim"] = int(_hp("latent_dim"))

        system = hp.get("system")
        if system is None:
            system = Quadrotor3DSystem(dataset_dir=_hp("system_dataset_dir"))

        model = instantiate_model_from_config(mc, int(_hp("latent_dim")))
        fm = cls(
            system=system, model=model, optimizer=None, scheduler=None,
            model_config=mc, latent_dim=int(_hp("latent_dim")),
            mae_val_frequency=int(hp.get("mae_val_frequency", 10)),
            use_loss_weights=bool(hp.get("use_loss_weights", False)),
            use_manifold=bool(hp.get("use_manifold", True)),
            use_log_loss_weights=bool(hp.get("use_log_loss_weights", False)),
            quat_loss_weight=float(hp.get("quat_loss_weight", 1.0)),
            clamp_noise=bool(hp.get("clamp_noise", True)),
            zero_latent=bool(hp.get("zero_latent", False)),
            noise_scale=float(hp.get("noise_scale", 1.0)),
            sequence_length=int(hp.get("sequence_length", 32)),
            history_length=int(hp.get("history_length", 1)),
        )
        sd = {k.replace("model.", ""): v for k, v in checkpoint["state_dict"].items() if k.startswith("model.")}
        fm.model.load_state_dict(sd)
        fm = fm.to(device)
        fm.eval()
        fm.training_config = hydra_config
        return fm
