from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import torch

from adaptive_roa.adaptive_v2.types import OutcomeProbabilities
from adaptive_roa.model.classifier_mlp import ClassifierMLP, ClassifierModule
from .base import ProbabilisticClassifier
from .registry import register_probabilistic_classifier


def clf_forward(module, system, states, device, bs=8192):
    out = []
    module.eval()
    with torch.no_grad():
        for i in range(0, len(states), bs):
            x = torch.as_tensor(states[i:i + bs], dtype=torch.float32, device=device)
            logits = module(x).squeeze(-1)
            out.append(torch.sigmoid(logits).double().cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def load_clf_module(epoch_dir, system, cfg, device):
    cls = cfg.get("classifier", {})
    dummy = torch.zeros(1, int(system.state_dim))
    input_dim = int(system.embed_state_for_model(system.normalize_state(dummy)).shape[-1])
    mlp = ClassifierMLP(
        input_dim=input_dim,
        hidden_dims=list(cls.get("hidden_dims", [256, 512, 256])),
        output_dim=1,
        dropout=float(cls.get("dropout", 0.0)),
    )
    module = ClassifierModule(
        mlp=mlp, system=system, pos_weight=torch.tensor(1.0), lr=1e-3, weight_decay=1e-5
    )
    best = glob.glob(str(Path(epoch_dir) / "checkpoints" / "best*.ckpt"))
    if not best:
        raise FileNotFoundError(f"no checkpoint in {epoch_dir}")
    ck = torch.load(best[0], map_location="cpu", weights_only=False)
    missing, _ = module.load_state_dict(ck["state_dict"], strict=False)
    bad = [k for k in missing if k.startswith("mlp.")]
    if bad:
        raise RuntimeError(f"checkpoint missing classifier weights {bad[:3]} in {best[0]}")
    module.eval().to(device)
    return module


@register_probabilistic_classifier
class ClassifierProbabilisticClassifier(ProbabilisticClassifier):
    predictor_type = "classifier"
    native_probs = ("p_success",)

    def __init__(self, module, system, device):
        self.module = module
        self.system = system
        self.device = device

    def predict(self, states: np.ndarray) -> OutcomeProbabilities:
        p_success = clf_forward(self.module, self.system, states, self.device)
        return OutcomeProbabilities(
            p_success=p_success,
            p_failure=1.0 - p_success,
            p_invalid=np.zeros_like(p_success),
        )

    @classmethod
    def load_from_run(cls, run_dir, epoch, cfg, system, device="cuda"):
        epoch_dir = str(Path(run_dir) / f"epoch_{epoch:03d}")
        module = load_clf_module(epoch_dir, system, cfg, device)
        return cls(module, system, device)
