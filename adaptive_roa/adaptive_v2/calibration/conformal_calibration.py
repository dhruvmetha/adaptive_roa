from __future__ import annotations

from typing import Any

import numpy as np

from adaptive_roa.conformal import ConformalConfig, ConformalPredictor
from adaptive_roa.conformal.estimator_factory import build_probability_estimator
from adaptive_roa.adaptive_v2.types import ThresholdState


class ConformalCalibrationBackend:
    """Computes q_hat from a calibration set, decoupled from threshold optimization."""

    def __init__(self, cfg: Any, system: Any, device: str):
        self.delta = float(cfg.delta)
        self.w = float(cfg.w)
        self.alpha = float(cfg.alpha)
        self.alpha_eval = float(cfg.alpha_eval)
        self.decision_rule = str(cfg.decision_rule)
        self.attractor_radius = float(cfg.attractor_radius)
        self.num_mc_samples = int(cfg.num_mc_samples)
        self.verbose = bool(cfg.verbose)
        self.system = system
        self.device = device
        self._predictor: ConformalPredictor | None = None
        self._predictor_eval: ConformalPredictor | None = None

    def bind_model(self, model_handle: Any, predictor_type: str = "generative") -> None:
        conf = ConformalConfig(
            delta=self.delta,
            w=self.w,
            alpha=self.alpha,
            attractor_radius=self.attractor_radius,
            num_mc_samples=self.num_mc_samples,
            decision_rule=self.decision_rule,
        )
        conf_eval = ConformalConfig(
            delta=self.delta,
            w=self.w,
            alpha=self.alpha_eval,
            attractor_radius=self.attractor_radius,
            num_mc_samples=self.num_mc_samples,
            decision_rule=self.decision_rule,
        )
        estimator = build_probability_estimator(
            predictor_type, model_handle, self.system, conf, self.device
        )
        self._predictor = ConformalPredictor(
            flow_matcher=model_handle,
            system=self.system,
            config=conf,
            device=self.device,
            probability_estimator=estimator,
        )
        self._predictor_eval = ConformalPredictor(
            flow_matcher=model_handle,
            system=self.system,
            config=conf_eval,
            device=self.device,
            probability_estimator=estimator,
        )

    def calibrate(
        self,
        X_cal: np.ndarray,
        y_cal: np.ndarray,
        threshold_state: ThresholdState,
    ) -> float:
        """Return q_hat for acquisition using self.alpha."""
        if self._predictor is None:
            raise RuntimeError("CalibrationBackend used before bind_model")
        self._predictor.lambda_star = threshold_state.lambda_star
        self._predictor.delta_star = threshold_state.delta_star
        q_hat = self._predictor.calibrate_qhat(X_cal, y_cal, verbose=self.verbose)
        return float(q_hat)

    def calibrate_eval(
        self,
        X_cal_eval: np.ndarray,
        y_cal_eval: np.ndarray,
        threshold_state: ThresholdState,
    ) -> float:
        """Return q_hat for evaluation-time coverage using self.alpha_eval."""
        if self._predictor_eval is None:
            raise RuntimeError("CalibrationBackend used before bind_model")
        self._predictor_eval.lambda_star = threshold_state.lambda_star
        self._predictor_eval.delta_star = threshold_state.delta_star
        q_hat = self._predictor_eval.calibrate_qhat(X_cal_eval, y_cal_eval, verbose=self.verbose)
        return float(q_hat)
