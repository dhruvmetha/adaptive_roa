"""Export wrappers for the three GP arms.

`gp` and `gp_optdelta` are outcome-probability arms (a GP classifier's predictive
probability); `gp_reg` is a final-state arm whose probabilities come from
MC-classified endpoints. All three register under their own names so a GP run is
never loaded as some other arm's checkpoint.
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import torch

from adaptive_roa.adaptive_v2.types import OutcomeProbabilities
from adaptive_roa.partx.gp_classifier import GPClassifier
from adaptive_roa.partx.model_handle import GPModelHandle
from adaptive_roa.predictors.gp_final_state_handle import GPFinalStateHandle
from adaptive_roa.predictors.gp_regressor import GPRegressor
from .base import ProbabilisticClassifier
from .endpoint_mc import endpoint_mc_probabilities
from .registry import register_probabilistic_classifier


def _find_gp_checkpoint(ckpt_dir) -> Path:
    """Prefer the warm-startable name; accept the legacy one for older runs.

    The partx trainer used to write ``gp.pt``, which the engine's
    ``checkpoints/best*.ckpt`` glob never matched. Runs already on disk carry it.
    """
    ckpt_dir = Path(ckpt_dir)
    modern = sorted(glob.glob(str(ckpt_dir / "best*.ckpt")))
    if modern:
        return Path(modern[0])
    legacy = ckpt_dir / "gp.pt"
    if legacy.exists():
        return legacy
    raise FileNotFoundError(f"no GP checkpoint (best*.ckpt or gp.pt) in {ckpt_dir}")


@register_probabilistic_classifier
class GPProbabilisticClassifier(ProbabilisticClassifier):
    predictor_type = "classifier"
    predictor_name = "gp"
    native_probs = ("p_success",)

    def __init__(self, gp, system, device):
        self.gp = gp
        self.handle = GPModelHandle(gp, system)
        self.system = system
        self.device = device

    def predict(self, states: np.ndarray) -> OutcomeProbabilities:
        p_success = np.asarray(self.gp.p_success(np.asarray(states)), dtype=np.float64)
        return OutcomeProbabilities(
            p_success=p_success,
            p_failure=1.0 - p_success,
            p_invalid=np.zeros_like(p_success),
        )

    @classmethod
    def load_from_run(cls, run_dir, epoch, cfg, system, device="cuda"):
        ckpt = _find_gp_checkpoint(Path(run_dir) / f"epoch_{epoch:03d}" / "checkpoints")
        # GPClassifier.load_state_dict rebuilds from inducing_shape + kernel
        # (gp_classifier.py:107-114), so no architecture config is needed here.
        gp = GPClassifier(system, device=device)
        gp.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False))
        gp.to(device)
        return cls(gp, system, device)


@register_probabilistic_classifier
class GPOptDeltaProbabilisticClassifier(GPProbabilisticClassifier):
    predictor_name = "gp_optdelta"


@register_probabilistic_classifier
class GPRegProbabilisticClassifier(ProbabilisticClassifier):
    predictor_type = "generative"
    predictor_name = "gp_reg"
    native_probs = ("p_success", "p_failure", "p_invalid")

    def __init__(self, handle, system, device, attractor_radius, num_mc_samples):
        self.handle = handle
        self.gp = handle.gp
        self.system = system
        self.device = device
        self.attractor_radius = attractor_radius
        self.num_mc_samples = num_mc_samples

    def predict(self, states: np.ndarray) -> OutcomeProbabilities:
        # Same loop as the BNN final-state arms, batching and empty-input guard
        # included; see probabilistic_classifier/endpoint_mc.py.
        return endpoint_mc_probabilities(
            self.handle, self.system, np.asarray(states),
            self.attractor_radius, self.num_mc_samples,
        )

    @classmethod
    def load_from_run(cls, run_dir, epoch, cfg, system, device="cuda"):
        prob = cfg.get("probability", {})
        # No hardcoded radius fallback: a wrong radius silently relabels every
        # endpoint, which looks like a method result rather than a bug.
        if "attractor_radius" not in prob:
            raise KeyError(
                f"run config at {run_dir} has no probability.attractor_radius; "
                "refusing to guess a radius, which would relabel every endpoint."
            )
        ckpt = _find_gp_checkpoint(Path(run_dir) / f"epoch_{epoch:03d}" / "checkpoints")
        # GPRegressor's state_dict carries num_tasks, inducing_shape and kernel,
        # so the architecture is reconstructed without consulting the config.
        gp = GPRegressor(num_tasks=1, input_dim=1, device=device)
        gp.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False))
        gp.to(device)
        handle = GPFinalStateHandle(gp, system, device=device).eval().to(device)
        return cls(
            handle, system, device,
            attractor_radius=float(prob["attractor_radius"]),
            num_mc_samples=int(prob.get("num_mc_samples", 10)),
        )
