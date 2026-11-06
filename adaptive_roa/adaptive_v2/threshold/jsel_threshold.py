"""det_v2 threshold opt backend (method section 4; eval/det_v2_rule.py).

Each round picks tau* = (tau_s*, tau_f*) on {0.01, ..., 0.99}^2 by minimizing J_sel
(or J_ver for ablation A1) on the threshold-fit rows: the held-out rows of the
acquired data, which predictions_thrfit.npz stores. The engine predicts those rows
at ``num_mc_samples`` draws (final-state counts; a classifier's probability as is)
and hands this backend the fractions (``fit_split = "acq"``).

The state carries tau* in ``tau_s_star`` / ``tau_f_star``; ``lambda_star`` /
``delta_star`` hold the converted values for logging only. ``predictor`` is None:
nothing downstream calibrates against this backend.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from adaptive_roa.adaptive_v2.eval import det_v2_rule
from adaptive_roa.adaptive_v2.types import ThresholdState


def to_lambda_delta(tau_s: float, tau_f: float) -> tuple[float, float]:
    return (1.0 + tau_s - tau_f) / 2.0, (tau_s + tau_f - 1.0) / 2.0


class JSelThresholdBackend:
    fit_split = "acq"

    def __init__(self, cfg: Any, system: Any, device: str):
        self.predictor_type = str(cfg.predictor_type)
        self.decision_rule = str(cfg.get("decision_rule", "two_sided"))   # logged only
        self.objective = str(cfg.get("objective", "j_sel"))
        if self.objective not in det_v2_rule.OBJECTIVES:
            raise ValueError(f"threshold.objective must be one of {det_v2_rule.OBJECTIVES}, got {self.objective!r}")
        self.w = float(cfg.get("w", 0.1))
        self.num_mc_samples = int(cfg.num_mc_samples)
        self.verbose = bool(cfg.get("verbose", True))
        self.optimize_mode = f"det_v2_{self.objective}"
        self.system = system
        self.device = device
        self.predictor = None
        self._last_fit: dict | None = None
        self.last_result: dict | None = None

    def bind_model(self, model_handle: Any) -> None:
        self._last_fit = None
        self.last_result = None

    @property
    def last_fit(self) -> dict | None:
        """States, labels and fit-K probabilities of the most recent fit (predictions_thrfit.npz)."""
        return self._last_fit

    def optimize(self, X_train: np.ndarray, y_train: np.ndarray) -> ThresholdState:
        raise RuntimeError("threshold=j_sel fits on predicted threshold-fit rows; the engine calls optimize_probs")

    def optimize_probs(self, states, y, p_success, p_failure, p_invalid,
                       num_mc_samples: int | None) -> ThresholdState:
        is_clf = self.predictor_type == "classifier"
        res = det_v2_rule.fit(p_success, None if is_clf else p_failure, y,
                              w=self.w, objective=self.objective, classifier=is_clf)
        ts, tf = res["tau_s"], res["tau_f"]
        lam, dl = to_lambda_delta(ts, tf)
        self.last_result = {**res, "num_mc_samples": num_mc_samples}
        self._last_fit = {
            "states": np.asarray(states), "labels": np.asarray(y),
            "p_success": np.asarray(p_success), "p_failure": np.asarray(p_failure),
            "p_invalid": np.asarray(p_invalid),
            "num_mc_samples": int(num_mc_samples) if num_mc_samples is not None else 1,
        }
        if self.verbose:
            print(f"[{self.objective}] {res['n']} threshold-fit rows ({res['n_success']} success), "
                  f"K={num_mc_samples if num_mc_samples is not None else 'classifier'}, w={self.w}")
            print(f"[{self.objective}] tau_s*={ts}, tau_f*={tf}: J={res['j']:.4f} MR={res['mr']:.4f} "
                  f"AR={res['ar']:.4f} (inv {res['ar_inv']:.4f}, out {res['ar_out']:.4f}) "
                  f"FCR={res['fcr']:.4f} REC={res['rec']:.4f}")
        return ThresholdState(lambda_star=float(lam), delta_star=float(dl),
                              tau_s_star=ts, tau_f_star=tf,
                              threshold_objective=self.objective, threshold_w=self.w)
