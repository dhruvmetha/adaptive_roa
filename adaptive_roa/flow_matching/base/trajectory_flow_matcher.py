"""
Trajectory-level flow matching base class.

Extends BaseFlowMatcher to operate on [B, T, D] trajectory tensors instead of
[B, D] endpoint tensors. Overrides compute_flow_loss and predict_endpoint to:
- Train a velocity field over full trajectory sequences with history pinning
- Predict endpoints by generating trajectories and extracting the final timestep

Uses FB FM's RiemannianODESolver for inference — Product manifold operations
(projx, proju, expmap) correctly handle [B, T, D] tensors because they operate
on the last dimension via [..., slice] indexing.

History pinning: first `history_length` timesteps are conditioned on the start
state. Their velocities are zeroed in the velocity wrapper, so they stay fixed
during ODE integration (no need for re-pinning inside the loop).
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional

from flow_matching.solver import RiemannianODESolver
from flow_matching.utils import ModelWrapper

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


class TrajectoryVelocityWrapper(ModelWrapper):
    """
    Wraps the trajectory model for FB FM's RiemannianODESolver.

    FB solver calls: velocity_model(x, t) → velocity
    Our model needs: model(x_embedded, t, z, condition) → velocity

    This wrapper bridges the gap by:
    - Embedding x [B, T, D] per-timestep via embed_fn
    - Passing fixed latent z and condition
    - Zeroing history timestep velocities
    """

    def __init__(
        self,
        model: nn.Module,
        latent: torch.Tensor,
        condition: torch.Tensor,
        embed_fn,
        history_length: int = 1,
    ):
        super().__init__(model)
        self.latent = latent
        self.condition = condition
        self.embed_fn = embed_fn
        self.history_length = history_length

    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras) -> torch.Tensor:
        """
        Forward pass compatible with RiemannianODESolver.

        Args:
            x: Current trajectory [B, T, state_dim]
            t: Time [B] or scalar

        Returns:
            Velocity [B, T, tangent_dim]
        """
        if t.dim() == 0:
            t = t.unsqueeze(0).expand(x.shape[0])

        # Embed per-timestep: [B, T, D] → [B*T, D] → embed → [B, T, embed_dim]
        B, T, D = x.shape
        x_flat = x.reshape(B * T, D)
        x_embedded_flat = self.embed_fn(x_flat)
        embed_dim = x_embedded_flat.shape[-1]
        x_embedded = x_embedded_flat.reshape(B, T, embed_dim)

        # Expand latent and condition to match batch size
        batch_size = x.shape[0]
        z = self.latent
        cond = self.condition
        if z.shape[0] == 1 and batch_size > 1:
            z = z.expand(batch_size, -1)
        if cond.shape[0] == 1 and batch_size > 1:
            cond = cond.expand(batch_size, -1)

        # Call model
        velocity = self.model(x_embedded, t, z, cond)

        # Zero history velocity — positions with zero velocity don't move
        velocity[:, :self.history_length, :] = 0.0

        return velocity


class TrajectoryFlowMatcherBase(BaseFlowMatcher):
    """
    Base class for trajectory-level flow matching.

    Adds trajectory-specific parameters and overrides the core training/inference
    methods to work with [B, T, D] tensors.

    Key differences from BaseFlowMatcher:
    - compute_flow_loss: operates on [B, T, D] trajectories with history masking
    - predict_endpoint: uses RiemannianODESolver on [B, T, D], returns [:, -1, :]
    - path.sample: called with [B, T, D] directly (t [B] broadcasts over T)

    Subclasses must implement all abstract methods from BaseFlowMatcher.
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

        Default implementation: sample per-timestep noise, clamp, and project
        onto manifold. projx operates on last dim so [B, T, D] works directly.

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
        # projx operates on last dim — handles [B, T, D] natively
        noise = self.manifold.projx(noise)
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

    def compute_flow_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute trajectory-level flow matching loss.

        Uses GeodesicProbPath.sample with [B, T, D] directly — t [B] broadcasts
        over the T dimension. History timesteps masked from loss.

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

        # Project onto manifold — works on [B, T, D] natively
        x_noisy = self.manifold.projx(x_noisy)

        # Sample random flow times [B]
        t = torch.rand(batch_size, device=device)

        # GeodesicProbPath requires t to be 1D [batch], so reshape [B,T,D] → [B*T,D]
        # with t expanded to [B*T] (same t for all timesteps within a trajectory)
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
        T = x_noisy.shape[1]
        loss_mask = torch.ones(1, T, 1, device=device)
        loss_mask[:, :self.history_length, :] = 0.0

        # Compute MSE loss with history mask
        squared_error = (predicted_velocity - dx_t) ** 2 * loss_mask
        if self.use_loss_weights and self.loss_weights is not None:
            normalized_loss_weights = self.loss_weights / (self.loss_weights.mean() + 1e-12)
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
        Predict endpoints via RiemannianODESolver on [B, T, D] trajectories.

        Uses FB FM's solver which correctly handles manifold operations (projx,
        proju, expmap) on [B, T, D] tensors. History is preserved by zeroing
        velocity in the TrajectoryVelocityWrapper.

        Args:
            start_states: [B, state_dim] in raw coordinates
            num_steps: Number of ODE integration steps
            latent: Optional [B, latent_dim]
            method: Integration method ("euler", "midpoint", "rk4")

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
                x_init = self.sample_noisy_trajectory_input(batch_size, device)

                # Pin history
                for t_idx in range(self.history_length):
                    x_init[:, t_idx] = start_normalized.clone()

                # Project onto manifold
                x_init = self.manifold.projx(x_init)

                # Create velocity wrapper for the solver
                velocity_model = TrajectoryVelocityWrapper(
                    model=self.model,
                    latent=z,
                    condition=start_embedded,
                    embed_fn=self.embed_state_for_model,
                    history_length=self.history_length,
                )

                # Use RiemannianODESolver — handles projx/proju on [B, T, D]
                solver = RiemannianODESolver(
                    manifold=self.manifold,
                    velocity_model=velocity_model,
                )

                final_trajectory = solver.sample(
                    x_init=x_init,
                    step_size=1.0 / num_steps,
                    method=method,
                    projx=True,
                    proju=True,
                    time_grid=torch.tensor([0.0, 1.0], device=device),
                )

                # Extract final timestep: [B, T, D] → [B, D]
                final_normalized = final_trajectory[:, -1, :]

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

        Args:
            x_t: Embedded state(s) [B, T, embed_dim] or [B, embed_dim]
            t: Time [B]
            z: Latent [B, latent_dim]
            condition: Condition [B, condition_dim]

        Returns:
            Predicted velocity
        """
        return self.model(x_t, t, z, condition)
