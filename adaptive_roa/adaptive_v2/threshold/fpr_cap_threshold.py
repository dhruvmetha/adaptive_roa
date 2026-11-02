"""FPR-capped threshold backend (det rerun, Idea 6; adaptive_v2/eval/fpr_cap.py).

Fits (tau_s, tau_f, tau_inv) each epoch on the held-out val split, not on the
threshold-fit pairs: those carry only 10 draws each and are trajectory states,
unlike test. The engine predicts the val split before the fit (``fit_split``) and
hands this backend the outcome counts over the first ``fit_mc_samples`` draws; a
classifier's probability is used as is.

``lambda_star`` / ``delta_star`` of the returned state hold the converted values
for logging; delta is negative whenever tau_s + tau_f < 1. No ConformalConfig is
built from them (its delta must lie in (0, 0.5)), and ``predictor`` is None, so
nothing downstream calibrates against this backend.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from adaptive_roa.adaptive_v2.eval import fpr_cap
from adaptive_roa.adaptive_v2.types import ThresholdState


class FprCapThresholdBackend:
    fit_split = "val"           # the engine predicts data_source.val_set_file before optimize_val

    def __init__(self, cfg: Any, system: Any, device: str):
        self.predictor_type = str(cfg.predictor_type)
        self.decision_rule = str(cfg.get("decision_rule", "two_sided"))   # logged; the tau rule is two-sided
        self.alpha_s = float(cfg.alpha_s)
        self.alpha_f = float(cfg.get("alpha_f", fpr_cap.ALPHA_F))
        self.tau_s_min = float(cfg.get("tau_s_min", fpr_cap.TAU_S_MIN))
        self.fit_mc_samples = int(cfg.get("fit_mc_samples", 20))
        if self.fit_mc_samples <= 0:
            raise ValueError(f"threshold.fit_mc_samples must be positive, got {self.fit_mc_samples}")
        self.verbose = bool(cfg.get("verbose", True))
        self.optimize_mode = "fpr_cap"
        self.system = system
        self.device = device
        self.predictor = None       # no ConformalPredictor: ranked/direct read tau through rule=tau
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
        raise RuntimeError("threshold=fpr_cap fits on the val split; the engine calls optimize_val")

    def optimize_val(self, states, y, p_success, p_failure, p_invalid,
                     num_mc_samples: int | None) -> ThresholdState:
        """Fit on the val split; num_mc_samples is the draws behind the fractions (None: classifier)."""
        res = fpr_cap.fit(p_success, p_failure, p_invalid, y, [self.alpha_s],
                          alpha_f=self.alpha_f, tau_s_min=self.tau_s_min)[self.alpha_s]
        ts, tf, ti = res["tau_s"], res["tau_f"], res["tau_inv"]
        lam, dl = fpr_cap.to_lambda_delta(ts, tf)
        self.last_result = {**res, "alpha_s": self.alpha_s, "alpha_f": self.alpha_f,
                            "fit_mc_samples": num_mc_samples}
        self._last_fit = {
            "states": np.asarray(states), "labels": np.asarray(y),
            "p_success": np.asarray(p_success), "p_failure": np.asarray(p_failure),
            "p_invalid": np.asarray(p_invalid),
            "num_mc_samples": int(num_mc_samples) if num_mc_samples is not None else 1,
        }
        if self.verbose:
            P, N = res["n_success"], res["n_failure"]
            print(f"[fpr_cap] val split: {P + N} states ({P} success), "
                  f"K={num_mc_samples if num_mc_samples is not None else 'classifier'}; "
                  f"alpha_s={self.alpha_s}, alpha_f={self.alpha_f}")
            print(f"[fpr_cap] tau_s={ts}, tau_f={tf}, tau_inv={ti} (lambda={lam:.4f}, delta={dl:+.4f}); "
                  f"val TP={res['tp']} FP={res['fp']} TN={res['tn']} FN={res['fn']}")
        return ThresholdState(lambda_star=float(lam), delta_star=float(dl),
                              tau_s=ts, tau_f=tf, tau_inv=ti, alpha_s=self.alpha_s)
