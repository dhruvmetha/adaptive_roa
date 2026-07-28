"""Export wrappers for the Bayesian MLP outcome arms.

Each arm registers under its own name so a BNN run is never loaded as a plain
``ClassifierMLP`` -- the checkpoints are structurally different and the mis-load
would be silent.
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import torch

from adaptive_roa.adaptive_v2.types import OutcomeProbabilities
from adaptive_roa.predictors.bayesian_mlp import build_bayesian_mlp
from adaptive_roa.predictors.handles import OutcomeModelHandle
from .base import ProbabilisticClassifier
from .registry import register_probabilistic_classifier


class BNNProbabilisticClassifier(ProbabilisticClassifier):
    """Shared loader; subclasses only set ``predictor_name`` and the posterior."""

    predictor_type = "classifier"
    native_probs = ("p_success",)
    posterior_kind = ""

    def __init__(self, handle, system, device):
        self.handle = handle
        self.system = system
        self.device = device

    def predict(self, states: np.ndarray) -> OutcomeProbabilities:
        out = []
        with torch.no_grad():
            for i in range(0, len(states), 8192):
                x = torch.as_tensor(states[i:i + 8192], dtype=torch.float32, device=self.device)
                out.append(torch.sigmoid(self.handle(x).view(-1)).double().cpu().numpy())
        p_success = np.concatenate(out) if out else np.zeros(0)
        return OutcomeProbabilities(
            p_success=p_success,
            p_failure=1.0 - p_success,
            p_invalid=np.zeros_like(p_success),
        )

    @classmethod
    def load_from_run(cls, run_dir, epoch, cfg, system, device="cuda"):
        bnn = cfg.get("predictor", {}).get("bnn", {})
        dummy = torch.zeros(1, int(system.state_dim))
        input_dim = int(system.embed_state_for_model(system.normalize_state(dummy)).shape[-1])
        posterior = build_bayesian_mlp(
            input_dim=input_dim,
            hidden_dims=list(bnn.get("hidden_dims", [256, 512, 256])),
            output_dim=1,
            posterior=cls.posterior_kind,
            prior_sigma=float(bnn.get("prior_sigma", 1.0)),
            n_members=int(bnn.get("n_members", 5)),
            dropout=float(bnn.get("dropout", 0.0)),
            activation=str(bnn.get("activation", "relu")),
        )
        ckpt_dir = Path(run_dir) / f"epoch_{epoch:03d}" / "checkpoints"
        best = sorted(glob.glob(str(ckpt_dir / "best*.ckpt")))
        if not best:
            raise FileNotFoundError(f"no BNN checkpoint in {ckpt_dir}")
        ckpt = torch.load(best[0], map_location="cpu", weights_only=False)
        state = ckpt.get("state_dict", ckpt)
        # Lightning prefixes module attrs; strip "posterior." when present.
        state = {k[len("posterior."):] if k.startswith("posterior.") else k: v
                 for k, v in state.items()}
        posterior.load_state_dict(state, strict=False)

        # The Laplace GGN covariance is a non-persistent buffer, so the trainer
        # writes it alongside the checkpoint. Without it the arm would load as a
        # MAP point estimate and silently report zero epistemic uncertainty.
        cov_path = ckpt_dir / "laplace_cov.pt"
        if cls.posterior_kind == "laplace":
            if not cov_path.exists():
                raise FileNotFoundError(
                    f"{cov_path} missing; a Laplace arm without its covariance is "
                    "a MAP model and must not be reported as Bayesian."
                )
            posterior._cov = torch.load(cov_path, map_location="cpu", weights_only=True)

        handle = OutcomeModelHandle(
            posterior, system,
            n_marginal_samples=int(bnn.get("n_marginal_samples", 64)),
            seed=int(bnn.get("seed", 0)),
        ).eval().to(device)
        return cls(handle, system, device)


@register_probabilistic_classifier
class MFVIProbabilisticClassifier(BNNProbabilisticClassifier):
    predictor_name = "bnn_mfvi"
    posterior_kind = "mfvi"


@register_probabilistic_classifier
class EnsembleProbabilisticClassifier(BNNProbabilisticClassifier):
    predictor_name = "bnn_ensemble"
    posterior_kind = "ensemble"


@register_probabilistic_classifier
class LaplaceProbabilisticClassifier(BNNProbabilisticClassifier):
    predictor_name = "bnn_laplace"
    posterior_kind = "laplace"
