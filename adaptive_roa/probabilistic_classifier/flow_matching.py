from __future__ import annotations

import glob
import importlib
from pathlib import Path

import numpy as np

from adaptive_roa.adaptive_v2.types import OutcomeProbabilities
from adaptive_roa.adaptive_v2.eval.mc_cache import (
    load_mc_cache,
    compute_mc_predictions,
)
from .base import ProbabilisticClassifier
from .registry import register_probabilistic_classifier

# system._target_ class name -> (FMModule, FMClass)
_FM_BY_SYSTEM = {
    "PendulumSystem": (
        "adaptive_roa.flow_matching.pendulum.latent_conditional.flow_matcher",
        "PendulumLatentConditionalFlowMatcher",
    ),
    "CartPoleSystem": (
        "adaptive_roa.flow_matching.cartpole.latent_conditional.flow_matcher",
        "CartPoleLatentConditionalFlowMatcher",
    ),
    "Quadrotor2DSystem": (
        "adaptive_roa.flow_matching.quadrotor_2d.latent_conditional.flow_matcher",
        "Quadrotor2DLatentConditionalFlowMatcher",
    ),
    "Quadrotor3DSystem": (
        "adaptive_roa.flow_matching.quadrotor_3d.latent_conditional.flow_matcher",
        "Quadrotor3DLatentConditionalFlowMatcher",
    ),
}


def resolve_fm_class(cfg):
    target = cfg.get("system", {}).get("_target_", "")
    cn = target.rsplit(".", 1)[-1]
    if cn not in _FM_BY_SYSTEM:
        raise KeyError(f"No flow-matcher class registered for system {cn!r}")
    mod, name = _FM_BY_SYSTEM[cn]
    return getattr(importlib.import_module(mod), name)


def resolve_radius_mc(run_dir, cfg):
    """Radius + MC-sample count: prefer a test mc_cache, else fall back to cfg."""
    caches = sorted(glob.glob(str(Path(run_dir) / "mc_cache" / "*_test.npz")))
    if caches:
        c = load_mc_cache(caches[0])
        return float(c.attractor_radius), int(c.num_mc_samples)
    conf = cfg.get("conformal", {})
    ev = cfg.get("evaluation", {})
    radius = float(conf.get("attractor_radius", ev.get("attractor_radius", 0.2)))
    nmc = int(conf.get("num_mc_samples", ev.get("num_samples", 20)))
    return radius, nmc


@register_probabilistic_classifier
class FMProbabilisticClassifier(ProbabilisticClassifier):
    predictor_type = "generative"
    predictor_name = "fm"
    native_probs = ("p_success", "p_failure", "p_invalid")

    def __init__(self, flow_matcher, system, device, attractor_radius, num_mc_samples):
        self.flow_matcher = flow_matcher
        self.system = system
        self.device = device
        self.attractor_radius = attractor_radius
        self.num_mc_samples = num_mc_samples

    def predict(self, states: np.ndarray) -> OutcomeProbabilities:
        cache = compute_mc_predictions(
            self.flow_matcher, self.system, states,
            num_mc_samples=self.num_mc_samples,
            attractor_radius=self.attractor_radius,
            batch_size=2048, device=self.device,
            verbose=False, refine_invalids=False,
        )
        ps, pf, pinv = cache.probabilities()
        return OutcomeProbabilities(p_success=ps, p_failure=pf, p_invalid=pinv)

    def predict_cached(self, run_dir, epoch, split, states):
        path = Path(run_dir) / "mc_cache" / f"epoch_{epoch:03d}_{split}.npz"
        if not path.exists():
            return None
        cache = load_mc_cache(str(path))
        # Row-alignment guard: a cache that does not match the export's own states
        # must not be trusted (it would silently misalign query_state with probs).
        # Check both the row count AND the actual state values (a same-length but
        # reordered/different cache would otherwise pass). Fall through to live
        # recompute on any mismatch.
        cached = np.asarray(cache.start_states, dtype=np.float32)
        query = np.asarray(states, dtype=np.float32)
        if cached.shape != query.shape or not np.allclose(cached, query, atol=1e-5, rtol=0.0):
            return None
        # Reclassify cached endpoints at the current eval radius (codebase standard:
        # full_roa.py / reevaluate.py do the same before using a loaded cache).
        cache = cache.reclassify(self.system, self.attractor_radius)
        ps, pf, pinv = cache.probabilities()
        return OutcomeProbabilities(p_success=ps, p_failure=pf, p_invalid=pinv)

    @classmethod
    def load_from_run(cls, run_dir, epoch, cfg, system, device="cuda"):
        fm_class = resolve_fm_class(cfg)
        ckpts = glob.glob(
            str(Path(run_dir) / f"epoch_{epoch:03d}" / "checkpoints" / "best*.ckpt")
        )
        if not ckpts:
            raise FileNotFoundError(
                f"no FM checkpoint in {run_dir}/epoch_{epoch:03d}"
            )
        fm = fm_class.load_from_checkpoint(ckpts[0], device=device)
        radius, nmc = resolve_radius_mc(run_dir, cfg)
        return cls(fm, system, device, radius, nmc)
