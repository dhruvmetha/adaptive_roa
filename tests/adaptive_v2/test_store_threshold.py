"""Offline threshold refit (A-W1) reproduces the engine's own fit at the run's w."""
import numpy as np
from omegaconf import OmegaConf

from adaptive_roa.adaptive_v2.eval.store_threshold import refit
from adaptive_roa.adaptive_v2.threshold.conformal_threshold import ConformalThresholdBackend

TCFG = {"predictor_type": "classifier", "decision_rule": "one_sided", "delta": 0.05, "w": 0.9,
        "alpha": 0.1, "attractor_radius": 0.2, "num_mc_samples": 10, "optimize_mode": "joint",
        "optimize_objective": "loss", "target_f1": 0.9, "lambda_grid_size": 30, "delta_grid_size": 20,
        "delta_min": 0.05, "delta_max": 0.45, "fixed_lambda_star": 0.5, "fixed_delta_star": 0.1,
        "use_p_invalid_veto": True, "verbose": False}


class _Logit:
    def eval(self):
        return self

    def to(self, d):
        return self

    def __call__(self, x):
        return x[:, :1] * 2.0


def test_refit_at_the_run_w_matches_the_engine_fit():
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, size=(300, 2)).astype(np.float32)
    y = np.where(X[:, 0] + 0.3 * rng.normal(size=300) > 0, 1, -1)
    b = ConformalThresholdBackend(OmegaConf.create(TCFG), system=None, device="cpu")
    b.bind_model(_Logit())
    st = b.optimize(X, y)
    f = b.last_fit
    lam, delta = refit(TCFG, 0.9, f["p_success"], f["p_failure"], f["p_invalid"], f["labels"])
    assert (lam, delta) == (st.lambda_star, st.delta_star)
    lam2, delta2 = refit(TCFG, 0.5, f["p_success"], f["p_failure"], f["p_invalid"], f["labels"])
    assert delta2 <= delta     # a cheaper miss buys less abstention
