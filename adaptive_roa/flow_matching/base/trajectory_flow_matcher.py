"""
Trajectory-level flow matching base class.

Extends BaseFlowMatcher to operate on [B, T, D] trajectory tensors instead of
[B, D] endpoint tensors. Overrides compute_flow_loss and predict_endpoint to:
- Train a velocity field over full trajectory sequences with history pinning
- Predict endpoints by generating trajectories and extracting the final timestep

Key design decisions:
- Manual ODE stepping (not RiemannianODESolver) because manifold operations
  (projx, proju) must be applied per-timestep after reshaping [B, T, D] → [B*T, D]
- History pinning: first `history_length` timesteps are conditioned on the start state
  and their velocities are zeroed during both training and inference
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from abc import abstractmethod

from adaptive_roa.flow_matching.base.flow_matcher import BaseFlowMatcher


def apply_conditioning(x: torch.Tensor, conditions: dict) -> None:
    """
    Apply conditioning to trajectory tensor.

    Args:
        x: Tensor [B, T, D] to apply conditioning to
        conditions: Dict mapping timestep index → value tensor [B, D]
    """
    if conditions is None:
        return
    for t, val in conditions.items():
        x[:, t] = val.clone()


class TrajectoryFlowMatcherBase(BaseFlowMatcher):
    """
    Base class for trajectory-level flow matching.

    Adds trajectory-specific parameters and overrides the core training/inference
    methods to work with [B, T, D] tensors.

    Subclasses must implement all abstract methods from BaseFlowMatcher plus
    the trajectory-specific sample_noisy_trajectory_input method.
    """

    def __init__(
        self,
        system,
        model: nn.Module,
        optimizer: Any,
        scheduler: Any,
        model_config=None,
        latent_dim: int = 2,
        mae_val_frequency: int = 10,
        use_loss_weights: bool = False,
        clamp_noise: bool = True,
        zero_latent: bool = False,
        val_error_log_file: Optional[str] = None,
        noise_scale: float = 1.0,
        # Trajectory-specific parameters
        sequence_length: int = 32,
        history_length: int = 1,
    ):
        self.sequence_length = sequence_length
        self.history_length = history_length

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
        )

        self.save_hyperparameters(ignore=["model", "optimizer", "scheduler", "system"])

    def sample_noisy_trajectory_input(
        self, batch_size: int, device: torch.device
    ) -> torch.Tensor:
        """
        Sample noise for full trajectory [B, T, D].

        Default implementation: sample per-timestep noise and optionally clamp/project.
        Subclasses can override for system-specific behavior.

        Args:
            batch_size: Number of samples
            device: Device

        Returns:
            Noisy trajectory [B, T, state_dim]
        """
        noise = torch.randn(batch_size, self.sequence_length, self.system.state_dim, device=device)
        if self.noise_scale != 1.0:
            noise = noise * self.noise_scale
        if self.clamp_noise:
            noise = torch.clamp(noise, -1.0, 1.0)
        # Project each timestep onto manifold
        B, T, D = noise.shape
        noise_flat = noise.reshape(B * T, D)
        noise_flat = self.manifold.projx(noise_flat)
        noise = noise_flat.reshape(B, T, D)
        return noise

    def _normalize_trajectory(self, trajectory: torch.Tensor) -> torch.Tensor:
        """Normalize trajectory per-timestep: [B, T, D] → normalize → [B, T, D]."""
        B, T, D = trajectory.shape
        flat = trajectory.reshape(B * T, D)
        flat_norm = self.normalize_state(flat)
        return flat_norm.reshape(B, T, D)

    def _denormalize_trajectory(self, trajectory: torch.Tensor) -> torch.Tensor:
        """Denormalize trajectory per-timestep: [B, T, D] → denormalize → [B, T, D]."""
        B, T, D = trajectory.shape
        flat = trajectory.reshape(B * T, D)
        flat_denorm = self.denormalize_state(flat)
        return flat_denorm.reshape(B, T, D)

    def _embed_trajectory(self, trajectory: torch.Tensor) -> torch.Tensor:
        """Embed trajectory per-timestep: [B, T, D] → embed → [B, T, embed_dim]."""
        B, T, D = trajectory.shape
        flat = trajectory.reshape(B * T, D)
        flat_emb = self.embed_state_for_model(flat)
        embed_dim = flat_emb.shape[-1]
        return flat_emb.reshape(B, T, embed_dim)

    def _projx_trajectory(self, trajectory: torch.Tensor) -> torch.Tensor:
        """Project trajectory per-timestep onto manifold."""
        B, T, D = trajectory.shape
        flat = trajectory.reshape(B * T, D)
        flat_proj = self.manifold.projx(flat)
        return flat_proj.reshape(B, T, D)

    def compute_flow_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute trajectory-level flow matching loss.

        Operates on [B, T, D] trajectory tensors with history pinning.

        Args:
            batch: Dictionary with 'trajectory' [B, T, D], 'start_state' [B, D]

        Returns:
            Scalar loss
        """
        trajectory = batch['trajectory']  # [B, T, D]
        start_states = self._get_start_states(batch)  # [B, D]

        batch_size = trajectory.shape[0]
        device = self.device

        # Normalize full trajectory
        traj_normalized = self._normalize_trajectory(trajectory)

        # Sample noise [B, T, D]
        x_noisy = self.sample_noisy_trajectory_input(batch_size, device)

        # Pin history timesteps with normalized start state
        start_normalized = self.normalize_state(start_states)
        for t_idx in range(self.history_length):
            x_noisy[:, t_idx] = start_normalized.clone()

        # Project onto manifold per-timestep
        x_noisy = self._projx_trajectory(x_noisy)

        # Sample random flow times
        t = torch.rand(batch_size, device=device)

        # GeodesicProbPath interpolation
        # Reshape to [B*T, D] for path.sample, then back to [B, T, D]
        B, T, D = x_noisy.shape
        x0_flat = x_noisy.reshape(B * T, D)
        x1_flat = traj_normalized.reshape(B * T, D)
        t_expanded = t.unsqueeze(1).expand(B, T).reshape(B * T)

        path_sample = self.path.sample(x_0=x0_flat, x_1=x1_flat, t=t_expanded)

        x_t = path_sample.x_t.reshape(B, T, D)
        dx_t = path_sample.dx_t.reshape(B, T, D)

        # Embed interpolated trajectory for model input
        x_t_embedded = self._embed_trajectory(x_t)  # [B, T, embed_dim]

        # Embed start state as condition
        start_embedded = self.embed_state_for_model(start_normalized)  # [B, cond_dim]

        # Sample latent
        z = self.sample_latent(batch_size, device)

        # Predict velocity: model(x_t_embedded, t, z, condition) -> [B, T, tangent_dim]
        predicted_velocity = self.model(x_t_embedded, t, z, start_embedded)

        # Mask out history timesteps from loss (avoid in-place ops on computation graph)
        loss_mask = torch.ones(1, T, 1, device=device)
        loss_mask[:, :self.history_length, :] = 0.0

        # Compute MSE loss with history mask
        squared_error = (predicted_velocity - dx_t) ** 2 * loss_mask
        if self.use_loss_weights and self.loss_weights is not None:
            normalized_loss_weights = self.loss_weights / (self.loss_weights.mean() + 1e-12)
            # weights: [tangent_dim] -> broadcast over [B, T, tangent_dim]
            squared_error = normalized_loss_weights.unsqueeze(0).unsqueeze(0) * squared_error

        # Mean over non-masked positions only
        n_active = T - self.history_length
        loss = squared_error.sum() / (batch_size * n_active * predicted_velocity.shape[-1])

        return loss

    def predict_endpoint(
        self,
        start_states: torch.Tensor,
        num_steps: int = 100,
        latent: Optional[torch.Tensor] = None,
        method: str = "euler",
    ) -> torch.Tensor:
        """
        Predict endpoints by generating full trajectories and extracting final timestep.

        Uses manual ODE integration (Euler/midpoint) because manifold operations
        must be applied per-timestep.

        Args:
            start_states: [B, state_dim] in raw coordinates
            num_steps: Number of ODE integration steps
            latent: Optional [B, latent_dim]
            method: "euler" or "midpoint"

        Returns:
            Predicted endpoints [B, state_dim] in raw coordinates
        """
        batch_size = start_states.shape[0]
        device = start_states.device

        was_training = self.training
        self.eval()

        try:
            with torch.no_grad():
                # Normalize start states
                start_normalized = self.normalize_state(start_states)
                start_embedded = self.embed_state_for_model(start_normalized)

                # Sample latent
                if latent is None:
                    z = self.sample_latent(batch_size, device)
                else:
                    z = latent

                # Sample noise trajectory [B, T, D]
                x = self.sample_noisy_trajectory_input(batch_size, device)

                # Pin history
                for t_idx in range(self.history_length):
                    x[:, t_idx] = start_normalized.clone()

                # Project onto manifold
                x = self._projx_trajectory(x)

                # Manual ODE integration from t=0 to t=1
                dt = 1.0 / num_steps

                for step in range(num_steps):
                    t_val = step * dt
                    t_tensor = torch.full((batch_size,), t_val, device=device)

                    if method == "midpoint":
                        # Half step
                        x_embedded = self._embed_trajectory(x)
                        v_half = self.model(x_embedded, t_tensor, z, start_embedded)
                        v_half[:, :self.history_length, :] = 0.0

                        x_mid = x + 0.5 * dt * v_half
                        x_mid = self._projx_trajectory(x_mid)
                        for t_idx in range(self.history_length):
                            x_mid[:, t_idx] = start_normalized.clone()

                        # Full step from midpoint
                        t_mid = torch.full((batch_size,), t_val + 0.5 * dt, device=device)
                        x_mid_embedded = self._embed_trajectory(x_mid)
                        v = self.model(x_mid_embedded, t_mid, z, start_embedded)
                        v[:, :self.history_length, :] = 0.0

                        x = x + dt * v
                    else:
                        # Euler step
                        x_embedded = self._embed_trajectory(x)
                        v = self.model(x_embedded, t_tensor, z, start_embedded)
                        v[:, :self.history_length, :] = 0.0

                        x = x + dt * v

                    # Project onto manifold and re-pin history
                    x = self._projx_trajectory(x)
                    for t_idx in range(self.history_length):
                        x[:, t_idx] = start_normalized.clone()

                # Extract final timestep
                final_normalized = x[:, -1, :]  # [B, D]

                # Denormalize
                final_raw = self.denormalize_state(final_normalized)

                return final_raw

        finally:
            if was_training:
                self.train()

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        z: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through the model.

        For trajectory FM, x_t may be [B, T, embed_dim] (trajectory) or
        [B, embed_dim] (endpoint, for compatibility).

        Args:
            x_t: Embedded state(s)
            t: Time [B]
            z: Latent [B, latent_dim]
            condition: Condition [B, condition_dim]

        Returns:
            Predicted velocity
        """
        return self.model(x_t, t, z, condition)
