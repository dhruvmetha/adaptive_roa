"""Conformal threshold backend for adaptive v2."""

from __future__ import annotations

from typing import Any

import numpy as np

from adaptive_roa.conformal import ConformalConfig, ConformalPredictor
from adaptive_roa.conformal.estimator_factory import build_probability_estimator
from adaptive_roa.adaptive_v2.types import ThresholdState


class ConformalThresholdBackend:
    """Adapter around ConformalPredictor for threshold optimization only."""

    def __init__(self, cfg: Any, system: Any, device: str):
        self.predictor_type = str(cfg.predictor_type)
        self.decision_rule = str(cfg.decision_rule)
        self.optimize_mode = str(cfg.optimize_mode)
        self.optimize_objective = str(cfg.optimize_objective)
        self.target_f1 = float(cfg.target_f1)
        self.lambda_grid_size = int(cfg.lambda_grid_size)
        self.delta_grid_size = int(cfg.delta_grid_size)
        self.delta_min = float(cfg.delta_min)
        self.delta_max = float(cfg.delta_max)
        self.fixed_lambda_star = float(cfg.fixed_lambda_star)
        self.fixed_delta_star = float(cfg.fixed_delta_star)
        self.use_p_invalid_veto = bool(cfg.use_p_invalid_veto)
        self.verbose = bool(cfg.verbose)
        self.system = system
        self.device = device
        self.predictor: ConformalPredictor | None = None

    def _build_conformal_config(self) -> ConformalConfig:
        return ConformalConfig(
            delta=self.fixed_delta_star,
            w=0.9,
            alpha=0.1,
            attractor_radius=0.2,
            optimize_mode=self.optimize_mode,
            optimize_objective=self.optimize_objective,
            target_f1=self.target_f1,
            decision_rule=self.decision_rule,
            lambda_grid_size=self.lambda_grid_size,
            delta_grid_size=self.delta_grid_size,
            delta_min=self.delta_min,
            delta_max=self.delta_max,
            fixed_lambda_star=self.fixed_lambda_star,
            fixed_delta_star=self.fixed_delta_star,
            use_p_invalid_veto=self.use_p_invalid_veto,
        )

    def bind_model(self, model_handle: Any) -> None:
        conf = self._build_conformal_config()
        estimator = build_probability_estimator(
            self.predictor_type, model_handle, self.system, conf, self.device
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
        self.predictor.optimize_thresholds(X_train, y_train, verbose=self.verbose)
        return ThresholdState(
            lambda_star=float(self.predictor.lambda_star),
            delta_star=float(self.predictor.delta_star),
        )
