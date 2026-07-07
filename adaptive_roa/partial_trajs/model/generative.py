"""Generative (conditional flow-matching) T-step dynamics backend.

Models ``x_{t+T} | x_t`` with a conditional rectified-flow: a velocity network
(reusing the adaptive ``SimpleFlowMLP``) transports Gaussian noise to the target
next-state, conditioned on the manifold-embedded start state. Works in
normalized state space; the manifold is handled on the input embedding, matching
the adaptive setup. This is the backend for the stochastic (noisy) regime; for
deterministic data it collapses toward a point map.
"""
from __future__ import annotations

from typing import Dict, List

import lightning.pytorch as pl
import torch
import torch.nn.functional as F

from adaptive_roa.model.simple_flow_mlp import SimpleFlowMLP
from adaptive_roa.partial_trajs.model.base import DynamicsModel


class GenerativeDynamics(pl.LightningModule, DynamicsModel):
    def __init__(
        self,
        system,
        hidden_dims: List[int],
        latent_dim: int = 1,
        time_emb_dim: int = 64,
        num_integration_steps: int = 50,
        lr: float = 1e-3,
    ):
        super().__init__()
        self.system = system
        self.lr = lr
        self.latent_dim = latent_dim
        self.num_integration_steps = num_integration_steps
        state_dim = int(system.state_dim)
        cond_dim = self._embedded_dim(system)
        # x_t lives in raw (normalized) state space so velocity matches its dim;
        # the start state is embedded and used only as conditioning.
        self.net = SimpleFlowMLP(
            embedded_dim=state_dim,
            latent_dim=latent_dim,
            condition_dim=cond_dim,
            time_emb_dim=time_emb_dim,
            hidden_dims=list(hidden_dims),
            output_dim=state_dim,
            use_input_embeddings=False,
        )

    @staticmethod
    def _embedded_dim(system) -> int:
        dummy = torch.zeros(1, int(system.state_dim))
        return int(system.embed_state_for_model(system.normalize_state(dummy)).shape[-1])

    def _condition(self, x_start_raw: torch.Tensor) -> torch.Tensor:
        return self.system.embed_state_for_model(self.system.normalize_state(x_start_raw))

    def _zeros_latent(self, n: int, device) -> torch.Tensor:
        return torch.zeros(n, self.latent_dim, device=device)

    # --- training: conditional rectified flow in normalized space -----------
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx):
        cond = self._condition(batch["x_start"])
        x1 = self.system.normalize_state(batch["x_end"])
        x0 = torch.randn_like(x1)
        t = torch.rand(x1.shape[0], device=x1.device)
        x_t = (1.0 - t)[:, None] * x0 + t[:, None] * x1
        target_v = x1 - x0
        pred_v = self.net(x_t, t, self._zeros_latent(x1.shape[0], x1.device), cond)
        loss = F.mse_loss(pred_v, target_v)
        if self._trainer is not None:
            self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx):
        loss = self.training_step(batch, batch_idx)
        if self._trainer is not None:
            self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    # --- DynamicsModel interface --------------------------------------------
    @torch.no_grad()
    def _integrate(self, cond: torch.Tensor) -> torch.Tensor:
        """Euler-integrate the flow from noise to a next-state sample (normalized)."""
        n = cond.shape[0]
        x = torch.randn(n, int(self.system.state_dim), device=cond.device)
        z = self._zeros_latent(n, cond.device)
        dt = 1.0 / self.num_integration_steps
        for i in range(self.num_integration_steps):
            t = torch.full((n,), i * dt, device=cond.device)
            x = x + dt * self.net(x, t, z, cond)
        return x

    def sample(self, x: torch.Tensor, num_samples: int) -> torch.Tensor:
        cond = self._condition(x)
        draws = [self.system.denormalize_state(self._integrate(cond)) for _ in range(num_samples)]
        return torch.stack(draws, dim=0)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.sample(x, 1)[0]
