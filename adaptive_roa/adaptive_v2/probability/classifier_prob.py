"""Classifier probability backend (forward-pass, no Monte Carlo)."""

from __future__ import annotations

from typing import Any

import numpy as np

from adaptive_roa.conformal import ConformalConfig
from adaptive_roa.conformal.classifier_probability_estimator import ClassifierProbabilityEstimator
from adaptive_roa.adaptive_v2.types import OutcomeProbabilities


class ClassifierProbabilityBackend:
    """Drop-in for EndpointMCProbabilityBackend, backed by a discriminative classifier."""

    def __init__(self, system: Any, cfg: Any, device: str):
        self.system = system
        self.cfg = cfg
        self.device = device
        self.model_handle: Any = None
        self.estimator: ClassifierProbabilityEstimator | None = None

    def bind_model(self, model_handle: Any) -> None:
        self.model_handle = model_handle
        conformal_cfg = ConformalConfig.from_hydra(self.cfg)
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
