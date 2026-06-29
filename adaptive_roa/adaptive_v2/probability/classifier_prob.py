from __future__ import annotations

from typing import Any

import numpy as np

from adaptive_roa.conformal import ConformalConfig
from adaptive_roa.conformal.classifier_probability_estimator import ClassifierProbabilityEstimator
from adaptive_roa.adaptive_v2.types import OutcomeProbabilities


class ClassifierProbabilityBackend:
    """Probability backend using a discriminative classifier (single forward pass)."""

    def __init__(self, cfg: Any, system: Any, device: str):
        self.attractor_radius = float(cfg.attractor_radius)
        self.system = system
        self.device = device
        self.model_handle: Any = None
        self.estimator: ClassifierProbabilityEstimator | None = None

    def bind_model(self, model_handle: Any) -> None:
        self.model_handle = model_handle
        conformal_cfg = ConformalConfig(
            delta=0.05, w=0.9, alpha=0.1, attractor_radius=self.attractor_radius,
        )
        self.estimator = ClassifierProbabilityEstimator(
            model_handle, self.system, conformal_cfg, self.device
        )

    def estimate(self, start_states: np.ndarray) -> OutcomeProbabilities:
        if self.estimator is None:
            raise RuntimeError("Probability backend used before bind_model")
        p_success, p_failure, p_invalid = self.estimator.estimate(start_states)
        return OutcomeProbabilities(
            p_success=np.asarray(p_success),
            p_failure=np.asarray(p_failure),
            p_invalid=np.asarray(p_invalid),
        )
