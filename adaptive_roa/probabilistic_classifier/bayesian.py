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
from adaptive_roa.predictors.bayesian_mlp import build_from_cfg, outcome_handle_from_cfg
from .base import ProbabilisticClassifier
from .registry import register_probabilistic_classifier

# Checkpoint entries that legitimately have no counterpart in the posterior's
# own state dict: `pos_weight` is the Lightning module's class-imbalance buffer,
# and the Laplace `_cov` is non-persistent (it round-trips via laplace_cov.pt).
_NON_PARAMETER_KEYS = {"pos_weight"}


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
        # Same builder the trainer uses, so the skeleton cannot drift from the
        # architecture the checkpoint was written with.
        posterior = build_from_cfg(bnn, system, cls.posterior_kind)
        ckpt_dir = Path(run_dir) / f"epoch_{epoch:03d}" / "checkpoints"
        best = sorted(glob.glob(str(ckpt_dir / "best*.ckpt")))
        if not best:
            raise FileNotFoundError(f"no BNN checkpoint in {ckpt_dir}")
        ckpt = torch.load(best[0], map_location="cpu", weights_only=False)
        state = ckpt.get("state_dict", ckpt)
        # Lightning prefixes module attrs; strip "posterior." when present.
        state = {k[len("posterior."):] if k.startswith("posterior.") else k: v
                 for k, v in state.items()}
        # strict=False is needed to tolerate `pos_weight` and the non-persistent
        # `_cov`, but on its own it also tolerates a WRONG skeleton: a changed
        # n_members or a laplace checkpoint loaded into an mfvi net leaves whole
        # sub-networks at random init and returns silently. Check the keys
        # ourselves so the failure is loud, as classifier.py:44-46 does.
        missing, unexpected = posterior.load_state_dict(state, strict=False)
        unexpected = [k for k in unexpected
                      if k not in _NON_PARAMETER_KEYS and not k.endswith("_cov")]
        if missing or unexpected:
            raise RuntimeError(
                f"checkpoint {best[0]} does not match the '{cls.posterior_kind}' "
                f"architecture built from cfg.predictor.bnn: "
                f"{len(missing)} missing key(s) {list(missing)[:5]}, "
                f"{len(unexpected)} unexpected key(s) {unexpected[:5]}. "
                f"Loading it would export a partly untrained network."
            )

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

        handle = outcome_handle_from_cfg(posterior, system, bnn).eval().to(device)
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
