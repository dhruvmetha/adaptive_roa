"""Conformal threshold backend for adaptive v2."""

from __future__ import annotations

from typing import Any

import numpy as np

from adaptive_roa.conformal import ConformalConfig, ConformalPredictor
from adaptive_roa.adaptive_v2.types import ThresholdState


class ConformalThresholdBackend:
    """Adapter around `ConformalPredictor` threshold and q-hat APIs."""

    def __init__(self, system: Any, cfg: Any, device: str):
        self.system = system
        self.cfg = cfg
        self.device = device
        self.predictor: ConformalPredictor | None = None

    def bind_model(self, model_handle: Any) -> None:
        conf = ConformalConfig(
            delta=self.cfg.conformal.get("delta", 0.05),
            w=self.cfg.conformal.get("w", 0.9),
            alpha=self.cfg.conformal.get("alpha_sampling", 0.1),
            num_mc_samples=self.cfg.conformal.get("num_mc_samples", 100),
            attractor_radius=self.cfg.conformal.get("attractor_radius", 0.2),
            optimize_mode=self.cfg.conformal.get("optimize_mode", "lambda"),
            decision_rule=self.cfg.conformal.get("decision_rule", "two_sided"),
            lambda_grid_size=self.cfg.conformal.get("lambda_grid_size", 100),
            delta_grid_size=self.cfg.conformal.get("delta_grid_size", 100),
            delta_min=self.cfg.conformal.get("delta_min", 0.01),
            delta_max=self.cfg.conformal.get("delta_max", 0.49),
        )
        self.predictor = ConformalPredictor(
            flow_matcher=model_handle,
            system=self.system,
            config=conf,
            device=self.device,
        )

    def optimize(self, X_train: np.ndarray, y_train: np.ndarray) -> ThresholdState:
        if self.predictor is None:
            raise RuntimeError("Threshold backend used before bind_model")

        threshold_mode = self.cfg.conformal.get("threshold_mode", "dynamic")
        if threshold_mode == "fixed":
            lambda_star = float(self.cfg.conformal.get("fixed_lambda_star", 0.5))
            delta_star = float(self.cfg.conformal.get("fixed_delta_star", 0.1))
            self.predictor.lambda_star = lambda_star
            self.predictor.delta_star = delta_star
        else:
            self.predictor.optimize_thresholds(
                X_train,
                y_train,
                verbose=self.cfg.conformal.get("verbose", True),
            )
            lambda_star = float(self.predictor.lambda_star)
            delta_star = float(self.predictor.delta_star)

        return ThresholdState(lambda_star=lambda_star, delta_star=delta_star)

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
