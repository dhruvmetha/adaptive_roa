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

        # Zero history velocity via mask (safe for autograd)
        if self.history_length > 0:
            mask = torch.ones(1, T, 1, device=x.device)
            mask[:, :self.history_length, :] = 0.0
            velocity = velocity * mask

        return velocity


class TrajectoryFlowMatcherBase(BaseFlowMatcher):
    """
    Base class for trajectory-level flow matching.

    Adds trajectory-specific parameters and overrides the core training/inference
    methods to work with [B, T, D] tensors.

    Key differences from BaseFlowMatcher:
    - compute_flow_loss: operates on [B, T, D] trajectories with history zeroing
    - predict_endpoint: uses RiemannianODESolver on [B, T, D], returns [:, -1, :]
    - path.sample: reshaped to [B*T, D] (FB FM requires t to be 1D)

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

        Matches local_dynamics' compute_loss flow:
        1. Sample noise, pin history, project BOTH noise and target onto manifold
        2. path.sample for geodesic interpolation
        3. projx on interpolated state, embed, model, proju on velocity
        4. Zero history velocity in prediction (model learns to output zero there)
        5. MSE loss on full trajectory including history (pred=0 vs target=nonzero)

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

        # Project BOTH noise and target onto manifold (matches local_dynamics)
        x_noisy = self.manifold.projx(x_noisy)
        traj_normalized = self.manifold.projx(traj_normalized)

        # Pin history AFTER projx so pinned values are not overwritten
        start_normalized = self.normalize_state(start_states)
        for t_idx in range(self.history_length):
            x_noisy[:, t_idx] = start_normalized.clone()

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
        # dx_t may have different dim than D for mixed state/tangent manifolds (e.g., SO3: state=4, tangent=3)
        tangent_dim = path_sample.dx_t.shape[-1]
        dx_t = path_sample.dx_t.reshape(B, T, tangent_dim)

        # Fix #1: projx on interpolated state before embedding (matches ManifoldEmbeddingLayer)
        x_t = self.manifold.projx(x_t)

        # Embed interpolated trajectory for model input
        x_t_embedded = self._embed_trajectory(x_t)  # [B, T, embed_dim]

        # Embed start state as condition
        start_embedded = self.embed_state_for_model(start_normalized)  # [B, cond_dim]

        # Sample latent
        z = self.sample_latent(batch_size, device)

        # Predict velocity: model(x_t_embedded, t, z, condition) -> [B, T, tangent_dim]
        predicted_velocity = self.model(x_t_embedded, t, z, start_embedded)

        # Fix #1: proju on model output (matches ManifoldEmbeddingLayer)
        # proju projects velocity to tangent space at x_t
        predicted_velocity = self.manifold.proju(x_t, predicted_velocity)

        # Fix #2: Zero history velocity in prediction, keep in loss
        # (matches local_dynamics: model is penalized for predicting nonzero at history)
        # Use a mask to zero prediction without in-place ops on computation graph
        history_mask = torch.ones(1, T, 1, device=device)
        history_mask[:, :self.history_length, :] = 0.0
        predicted_velocity = predicted_velocity * history_mask

        # Compute MSE loss on full trajectory (including history positions)
        squared_error = (predicted_velocity - dx_t) ** 2
        if self.use_loss_weights and self.loss_weights is not None:
            normalized_loss_weights = self.loss_weights / (self.loss_weights.mean() + 1e-12)
            squared_error = normalized_loss_weights.unsqueeze(0).unsqueeze(0) * squared_error

        loss = squared_error.mean()

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

                # Project onto manifold, then pin history (so pin is not overwritten)
                x_init = self.manifold.projx(x_init)
                for t_idx in range(self.history_length):
                    x_init[:, t_idx] = start_normalized.clone()

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

    def predict_trajectory(
        self,
        start_states: torch.Tensor,
        num_steps: int = 100,
        latent: Optional[torch.Tensor] = None,
        method: str = "euler",
    ) -> torch.Tensor:
        """
        Predict full trajectories [B, T, D] in raw coordinates.

        Same as predict_endpoint but returns the full trajectory instead of
        just the final timestep.

        Args:
            start_states: [B, state_dim] in raw coordinates
            num_steps: Number of ODE integration steps
            latent: Optional [B, latent_dim]
            method: Integration method

        Returns:
            Predicted trajectories [B, T, state_dim] in raw coordinates
        """
        batch_size = start_states.shape[0]
        device = start_states.device

        was_training = self.training
        self.eval()

        try:
            with torch.no_grad():
                start_normalized = self.normalize_state(start_states)
                start_embedded = self.embed_state_for_model(start_normalized)

                if latent is None:
                    z = self.sample_latent(batch_size, device)
                else:
                    z = latent

                x_init = self.sample_noisy_trajectory_input(batch_size, device)
                x_init = self.manifold.projx(x_init)
                for t_idx in range(self.history_length):
                    x_init[:, t_idx] = start_normalized.clone()

                velocity_model = TrajectoryVelocityWrapper(
                    model=self.model,
                    latent=z,
                    condition=start_embedded,
                    embed_fn=self.embed_state_for_model,
                    history_length=self.history_length,
                )

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

                # Denormalize full trajectory per-timestep
                return self._denormalize_trajectory(final_trajectory)

        finally:
            if was_training:
                self.train()

    def classify_trajectory(
        self,
        trajectory: torch.Tensor,
        attractor_radius: float = 0.1,
    ) -> torch.Tensor:
        """
        Classify each trajectory by checking reachability at every timestep.

        For each trajectory, iterates through timesteps:
        - If classify_state returns SUCCESS (1) at any step → SUCCESS
        - If classify_state returns FAILURE (-1) at any step → FAILURE
        - If neither by end → UNCERTAIN (0, separatrix)

        First definitive outcome wins (early exit per trajectory).

        Args:
            trajectory: [B, T, state_dim] in raw coordinates
            attractor_radius: Radius for attractor check

        Returns:
            Labels [B] with 1 (success), -1 (failure), 0 (uncertain)
        """
        B, T, D = trajectory.shape
        device = trajectory.device

        # Initialize all as uncertain (0)
        labels = torch.zeros(B, dtype=torch.long, device=device)
        resolved = torch.zeros(B, dtype=torch.bool, device=device)

        for t_idx in range(T):
            if resolved.all():
                break

            state_t = trajectory[:, t_idx, :]  # [B, D]

            # Check success: is_in_attractor returns [B] bool
            in_attractor = self.system.is_in_attractor(state_t, radius=attractor_radius)
            if isinstance(in_attractor, bool):
                in_attractor = torch.tensor([in_attractor], device=device).expand(B)
            if not isinstance(in_attractor, torch.Tensor):
                in_attractor = torch.tensor(in_attractor, device=device)
            in_attractor = in_attractor.bool()

            new_success = in_attractor & ~resolved
            labels[new_success] = 1
            resolved[new_success] = True

            # Check failure: classify_state returns [B] with -1 for failure
            if hasattr(self.system, 'classify_state'):
                state_labels = self.system.classify_state(state_t, attractor_radius)
                if not isinstance(state_labels, torch.Tensor):
                    state_labels = torch.tensor(state_labels, device=device)
                is_failed = (state_labels == -1)

                new_failure = is_failed & ~resolved
                labels[new_failure] = -1
                resolved[new_failure] = True

        return labels

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
