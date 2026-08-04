"""System-agnostic binary ROA classifier.

``ClassifierMLP`` is a plain MLP over embedded states (logits).
``ClassifierModule`` is the LightningModule that owns state preprocessing
(normalize + embed via the system) so that *one* code path does the embedding,
shared by training and by ``ClassifierProbabilityEstimator`` at inference time:
both call ``module(raw_states) -> logits``.

Binary outcome: logit -> sigmoid = p(success). Trained with
``binary_cross_entropy_with_logits`` and a ``pos_weight`` to counter class
imbalance (quad2d is ~8% success).
"""
from __future__ import annotations

from typing import Any, List

import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


_ACTIVATIONS = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU, "silu": nn.SiLU}


class ClassifierMLP(nn.Module):
    """MLP mapping an embedded state [B, input_dim] -> logits [B, output_dim].

    ``activation`` defaults to ReLU (every existing run and checkpoint). It is
    configurable because the HMC reference tier puts every arm on the SAME
    [50,50] tanh backbone -- ReLU's non-differentiability degrades leapfrog's
    local error to O(eps) -- and a hardcoded ReLU here silently kept the plain
    classifier on a different backbone than the tier claimed. Activations carry
    no parameters, so switching one does not change the state_dict layout and
    existing checkpoints still load.
    """

    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int = 1,
                 dropout: float = 0.0, activation: str = "relu"):
        super().__init__()
        act_cls = _ACTIVATIONS.get(str(activation).lower())
        if act_cls is None:
            raise ValueError(
                f"unknown activation {activation!r}; expected one of {sorted(_ACTIVATIONS)}"
            )
        dims = [int(input_dim)] + [int(h) for h in hidden_dims]
        layers: List[nn.Module] = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(a, b))
            layers.append(act_cls())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-1], int(output_dim)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ClassifierModule(pl.LightningModule):
    """Binary ROA classifier. ``forward`` takes RAW states and embeds internally."""

    def __init__(
        self,
        mlp: nn.Module,
        system: Any,
        pos_weight: float = 1.0,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
    ):
        super().__init__()
        self.mlp = mlp
        self.system = system  # plain attr (not an nn.Module); methods are device-agnostic
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.register_buffer("pos_weight", torch.tensor(float(pos_weight)))

    def forward(self, raw_states: torch.Tensor) -> torch.Tensor:
        normalized = self.system.normalize_state(raw_states)
        embedded = self.system.embed_state_for_model(normalized)
        return self.mlp(embedded)

    def _step(self, batch, stage: str):
        x = batch["inputs"]
        y = batch["label"].float().view(-1)
        logits = self(x).view(-1)
        loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=self.pos_weight)
        with torch.no_grad():
            preds = (torch.sigmoid(logits) > 0.5).float()
            acc = (preds == y).float().mean()
        self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log(f"{stage}_acc", acc, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=10, min_lr=1e-6)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "monitor": "val_loss"}}
