"""Deterministic T-step dynamics regressor (Lightning module).

A plain MLP mapping the (manifold-embedded) start state to the raw state ``T``
steps later. Backbone widths mirror the adaptive flow-MLP per-system
``mlp_hidden_dims``; the manifold is handled on the *input* embedding via
``system.embed_state_for_model`` (sin/cos for angles, etc.). Training is plain
MSE in normalized space, matching the adaptive runs' flags (no manifold loss,
no per-dim loss weights). Manifold-aware residual + loss weights are a
documented future option (spec §4).
"""
from __future__ import annotations

from typing import Dict, List

import lightning.pytorch as pl
import torch
import torch.nn.functional as F

from adaptive_roa.model.classifier_mlp import ClassifierMLP
from adaptive_roa.partial_trajs.model.base import DynamicsModel


class DynamicsRegressor(pl.LightningModule, DynamicsModel):
    def __init__(
        self,
        system,
        hidden_dims: List[int],
        lr: float = 1e-3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.system = system
        self.lr = lr
        embedded_dim = self._embedded_dim(system)
        self.net = ClassifierMLP(
            input_dim=embedded_dim,
            hidden_dims=list(hidden_dims),
            output_dim=int(system.state_dim),
            dropout=dropout,
        )

    @staticmethod
    def _embedded_dim(system) -> int:
        dummy = torch.zeros(1, int(system.state_dim))
        embedded = system.embed_state_for_model(system.normalize_state(dummy))
        return int(embedded.shape[-1])

    def _embed(self, x_raw: torch.Tensor) -> torch.Tensor:
        return self.system.embed_state_for_model(self.system.normalize_state(x_raw))

    def forward(self, x_raw: torch.Tensor) -> torch.Tensor:
        pred_norm = self.net(self._embed(x_raw))
        return self.system.denormalize_state(pred_norm)

    # DynamicsModel interface: raw state -> raw next state.
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    def _step_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        pred_norm = self.net(self._embed(batch["x_start"]))
        target_norm = self.system.normalize_state(batch["x_end"])
        return F.mse_loss(pred_norm, target_norm)

    def _maybe_log(self, name: str, value: torch.Tensor) -> None:
        # Skip logging when the module is used outside a Trainer (e.g. in tests).
        if self._trainer is not None:
            self.log(name, value, prog_bar=True)

    def training_step(self, batch, batch_idx):
        loss = self._step_loss(batch)
        self._maybe_log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._step_loss(batch)
        self._maybe_log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
