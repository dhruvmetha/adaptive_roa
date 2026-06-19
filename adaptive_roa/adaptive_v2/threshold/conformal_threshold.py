"""Conformal threshold backend for adaptive v2."""

from __future__ import annotations

from typing import Any

import numpy as np

from adaptive_roa.conformal import ConformalConfig, ConformalPredictor
from adaptive_roa.conformal.estimator_factory import build_probability_estimator
from adaptive_roa.adaptive_v2.types import ThresholdState


class ConformalThresholdBackend:
    """Adapter around `ConformalPredictor` threshold and q-hat APIs."""

    def __init__(self, system: Any, cfg: Any, device: str):
        self.system = system
        self.cfg = cfg
        self.device = device
        self.predictor: ConformalPredictor | None = None

    def bind_model(self, model_handle: Any) -> None:
        conf = ConformalConfig.from_hydra(self.cfg)
        predictor_type = str(self.cfg.get("predictor", "generative"))
        estimator = build_probability_estimator(
            predictor_type, model_handle, self.system, conf, self.device
        )
        self.predictor = ConformalPredictor(
            flow_matcher=model_handle,
            system=self.system,
            config=conf,
            device=self.device,
            probability_estimator=estimator,
        )

    def optimize(self, X_train: np.ndarray, y_train: np.ndarray) -> ThresholdState:
        if self.predictor is None:
            raise RuntimeError("Threshold backend used before bind_model")

        self.predictor.optimize_thresholds(
            X_train,
            y_train,
            verbose=self.cfg.conformal.get("verbose", True),
        )

        return ThresholdState(
            lambda_star=float(self.predictor.lambda_star),
            delta_star=float(self.predictor.delta_star),
        )

    def calibrate_qhat(
        self,
        X_cal: np.ndarray,
        y_cal: np.ndarray,
        threshold_state: ThresholdState,
    ) -> float:
        if self.predictor is None:
            raise RuntimeError("Threshold backend used before bind_model")

        self.predictor.lambda_star = threshold_state.lambda_star
        self.predictor.delta_star = threshold_state.delta_star
        q_hat = self.predictor.calibrate_qhat(
            X_cal,
            y_cal,
            verbose=self.cfg.conformal.get("verbose", True),
        )
        return float(q_hat)
