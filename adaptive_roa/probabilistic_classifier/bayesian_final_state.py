"""Export wrappers for the final-state predictor arms.

These mirror ``bayesian.py`` (the outcome-arm wrappers) with three differences:
``predictor_type`` is ``"generative"`` (endpoint data, not a binary label);
``native_probs`` includes ``p_invalid`` because MC-classified endpoints
genuinely produce an unresolved outcome; and ``predict`` draws K endpoint
samples per query and counts ``system.classify_attractor`` labels rather than
applying a sigmoid to a single logit.

Each arm registers under its own name so a run trained with one weight
posterior (mfvi/ensemble/laplace/deterministic) is never loaded with another
arm's skeleton -- the checkpoints are structurally different and the mis-load
would be silent.
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import torch

from adaptive_roa.adaptive_v2.types import OutcomeProbabilities
from adaptive_roa.predictors.bayesian_mlp import build_from_cfg
from adaptive_roa.predictors.final_state_handle import FinalStateModelHandle
from adaptive_roa.predictors.heads import FinalStateHead
from .base import ProbabilisticClassifier
from .endpoint_mc import endpoint_mc_probabilities
from .registry import register_probabilistic_classifier

# Checkpoint entries that legitimately have no counterpart in the posterior's
# own state dict. Unlike the outcome arms' Lightning module (which carries a
# `pos_weight` class-imbalance buffer alongside `self.posterior`),
# `_FinalStateModule` has no such extra buffer: `self.head` is a plain
# `FinalStateHead` (not an `nn.Module`) and `self.system` is a plain
# `DynamicalSystem` (also not an `nn.Module`), so neither contributes state-dict
# keys. This set stays empty; it exists so the shape of the check matches
# `bayesian.py` exactly. The Laplace `_cov` buffer is non-persistent (it
# round-trips via laplace_cov.pt) and is excluded by the `_cov` suffix check
# below, same as the sibling.
_NON_PARAMETER_KEYS: set[str] = set()


def _resolve_probability_cfg(cfg) -> tuple[float, int]:
    """(attractor_radius, num_mc_samples) from the run's ``probability`` block.

    This is the adaptive_v2 pipeline's own MC config (`configs/adaptive_v2/
    probability/endpoint_mc.yaml`), which every final-state predictor config
    pulls in via `defaults: - /probability: endpoint_mc`. It is a DIFFERENT
    schema from the legacy `conformal`/`evaluation` blocks `flow_matching.py`'s
    `resolve_radius_mc` falls back to -- that wrapper serves the older
    (pre-adaptive_v2) FM pipeline, not this one.
    """
    prob = cfg.get("probability", {}) or {}
    radius = prob.get("attractor_radius", cfg.get("attractor_radius", None))
    if radius is None:
        # No silent default: the radius decides EVERY endpoint's label, so a
        # wrong one moves p_success without any downstream signal. The old
        # fallback of 0.2 is wrong for pendulum and both quadrotors.
        raise ValueError(
            "no attractor_radius in the run config (looked at probability."
            "attractor_radius and the top-level attractor_radius). Guessing it "
            "would silently relabel every MC endpoint and change p_success."
        )
    # 10 matches configs/adaptive_v2/probability/endpoint_mc.yaml's own default,
    # which every final-state predictor config pulls in, so this fallback can
    # only fire for a run whose probability block was written by hand.
    num_mc_samples = int(prob.get("num_mc_samples", 10))
    return float(radius), num_mc_samples


class FinalStateProbabilisticClassifier(ProbabilisticClassifier):
    """Shared loader and MC predictor; subclasses set ``predictor_name`` and
    the weight-posterior kind used to build the checkpoint's skeleton."""

    predictor_type = "generative"
    native_probs = ("p_success", "p_failure", "p_invalid")
    posterior_kind = ""

    def __init__(self, handle, system, device, attractor_radius, num_mc_samples):
        self.handle = handle
        self.system = system
        self.device = device
        self.attractor_radius = attractor_radius
        self.num_mc_samples = num_mc_samples

    def predict(self, states: np.ndarray) -> OutcomeProbabilities:
        return endpoint_mc_probabilities(
            self.handle, self.system, states,
            self.attractor_radius, self.num_mc_samples,
        )

    @classmethod
    def load_from_run(cls, run_dir, epoch, cfg, system, device="cuda"):
        fs = cfg.get("predictor", {}).get("final_state", {})
        head = FinalStateHead(system, beta=float(fs.get("beta_nll", 0.5)))
        # Same builder the trainer uses (bayesian_mlp.build_from_cfg), so the
        # skeleton cannot drift from the architecture the checkpoint was
        # written with. It already branches on posterior_kind == "ensemble"
        # internally, so this one call is correct for all four arms.
        posterior = build_from_cfg(fs, system, cls.posterior_kind, output_dim=head.n_params)

        ckpt_dir = Path(run_dir) / f"epoch_{epoch:03d}" / "checkpoints"
        best = sorted(glob.glob(str(ckpt_dir / "best*.ckpt")))
        if not best:
            raise FileNotFoundError(f"no final-state checkpoint in {ckpt_dir}")
        ckpt = torch.load(best[0], map_location="cpu", weights_only=False)
        state = ckpt.get("state_dict", ckpt)
        # Lightning prefixes the trainer module's attribute name ("posterior.").
        # The ensemble's assembled checkpoint is instead a raw
        # `EnsemblePosterior.state_dict()` with no such prefix (see
        # final_state_trainer.py's `best-ensemble.ckpt`), so stripping is a
        # harmless no-op for it.
        state = {k[len("posterior."):] if k.startswith("posterior.") else k: v
                 for k, v in state.items()}
        # strict=False alone would also tolerate a WRONG skeleton -- a changed
        # n_members or a laplace checkpoint loaded into an mfvi net leaves
        # whole sub-networks at random init and returns silently. Check the
        # keys ourselves so the failure is loud, as bayesian.py does.
        missing, unexpected = posterior.load_state_dict(state, strict=False)
        unexpected = [k for k in unexpected
                      if k not in _NON_PARAMETER_KEYS and not k.endswith("_cov")]
        if missing or unexpected:
            raise RuntimeError(
                f"checkpoint {best[0]} does not match the '{cls.posterior_kind}' "
                f"architecture built from cfg.predictor.final_state: "
                f"{len(missing)} missing key(s) {list(missing)[:5]}, "
                f"{len(unexpected)} unexpected key(s) {unexpected[:5]}. "
                f"Loading it would export a partly untrained network."
            )

        # The Laplace GGN covariance is a non-persistent buffer, so the trainer
        # writes it alongside the checkpoint. Without it the arm would load as
        # a MAP point estimate and silently report zero epistemic uncertainty
        # while still being labelled Bayesian.
        if cls.posterior_kind == "laplace":
            cov_path = ckpt_dir / "laplace_cov.pt"
            if not cov_path.exists():
                raise FileNotFoundError(
                    f"{cov_path} missing; a Laplace arm without its covariance "
                    "is a MAP model and must not be reported as Bayesian."
                )
            posterior._cov = torch.load(cov_path, map_location="cpu", weights_only=True)
            # Sampling draws on a factor of the precision, not of the
            # covariance. Runs finished before that change shipped only the
            # covariance; the posterior falls back to it, so its absence is not
            # an error here.
            chol_path = ckpt_dir / "laplace_prec_chol.pt"
            if chol_path.exists():
                posterior._prec_chol = torch.load(
                    chol_path, map_location="cpu", weights_only=True)

        handle = FinalStateModelHandle(posterior, head, system).eval().to(device)
        radius, num_mc_samples = _resolve_probability_cfg(cfg)
        return cls(handle, system, device, radius, num_mc_samples)


@register_probabilistic_classifier
class DeterministicFinalStateProbabilisticClassifier(FinalStateProbabilisticClassifier):
    predictor_name = "mlp_det"
    posterior_kind = "deterministic"


@register_probabilistic_classifier
class MFVIFinalStateProbabilisticClassifier(FinalStateProbabilisticClassifier):
    predictor_name = "bnn_mfvi_reg"
    posterior_kind = "mfvi"


@register_probabilistic_classifier
class EnsembleFinalStateProbabilisticClassifier(FinalStateProbabilisticClassifier):
    predictor_name = "bnn_ensemble_reg"
    posterior_kind = "ensemble"


@register_probabilistic_classifier
class LaplaceFinalStateProbabilisticClassifier(FinalStateProbabilisticClassifier):
    predictor_name = "bnn_laplace_reg"
    posterior_kind = "laplace"
