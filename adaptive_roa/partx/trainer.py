from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from adaptive_roa.partx.gp_classifier import GPClassifier
from adaptive_roa.partx.model_handle import GPModelHandle


def load_xy(path: str, state_dim: int):
    """Load (state, label) rows; map to binary success/failure, drop sep/invalid."""
    raw = np.loadtxt(path).reshape(-1, state_dim + 1)
    X, labels = raw[:, :state_dim], raw[:, -1]
    uniq = set(np.unique(labels).tolist())
    if uniq <= {0.0, 1.0}:                      # {0,1} scheme: 0=failure, 1=success
        keep = np.ones(len(labels), dtype=bool)
    else:                                        # signed scheme: keep only ±1
        keep = np.isin(labels, [1.0, -1.0])
    X = X[keep]
    y01 = (labels[keep] == 1.0).astype(np.int64) if not (uniq <= {0.0, 1.0}) \
        else (labels[keep] > 0.5).astype(np.int64)
    return X, y01


class GPPredictorTrainer:
    def __init__(self, cfg: Any, system: Any, system_name: str):
        self.cfg = cfg
        self.system = system
        self.system_name = system_name

    @property
    def _gp_cfg(self):
        pred = self.cfg.get("predictor")
        base = pred if pred is not None else self.cfg
        return base.get("gp", {})

    def fit(self, dataset_files: dict, output_dir: str, resume_checkpoint: str | None = None):
        gp_cfg = self._gp_cfg
        device = str(self.cfg.get("device", "cpu"))
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        X, y = load_xy(dataset_files["train"], int(self.system.state_dim))
        gp = GPClassifier(
            self.system,
            n_inducing=int(gp_cfg.get("n_inducing", 128)),
            kernel=str(gp_cfg.get("kernel", "matern52")),
            n_iters=int(gp_cfg.get("n_iters", 300)),
            lr=float(gp_cfg.get("lr", 0.1)),
            device=device,
        )
        gp.fit(X, y)
        ckpt_dir = Path(output_dir) / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(gp.state_dict(), ckpt_dir / "gp.pt")
        return GPModelHandle(gp, self.system).eval().to(device)
