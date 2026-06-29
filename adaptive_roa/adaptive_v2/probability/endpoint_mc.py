from __future__ import annotations

from typing import Any

import numpy as np

from adaptive_roa.conformal import ConformalConfig
from adaptive_roa.conformal.probability_estimator import ProbabilityEstimator
from adaptive_roa.adaptive_v2.types import OutcomeProbabilities


class EndpointMCProbabilityBackend:
    """Wraps ProbabilityEstimator (MC endpoint sampling) with the v2 interface."""

    def __init__(self, cfg: Any, system: Any, device: str):
        self.attractor_radius = float(cfg.attractor_radius)
        self.num_mc_samples = int(cfg.num_mc_samples)
        self.refine_invalids = bool(cfg.refine_invalids)
        self.refine_t_min = float(cfg.refine_t_min)
        self.refine_t_max = float(cfg.refine_t_max)
        self.refine_num_steps = int(cfg.refine_num_steps)
        self.refine_max_attempts = int(cfg.refine_max_attempts)
        self.trajectory_checking = bool(cfg.trajectory_checking)
        self.system = system
        self.device = device
        self.model_handle = None
        self.estimator: ProbabilityEstimator | None = None

    def _build_conformal_config(self) -> ConformalConfig:
        return ConformalConfig(
            attractor_radius=self.attractor_radius,
            num_mc_samples=self.num_mc_samples,
            refine_invalids=self.refine_invalids,
            refine_t_min=self.refine_t_min,
            refine_t_max=self.refine_t_max,
            refine_num_steps=self.refine_num_steps,
            refine_max_attempts=self.refine_max_attempts,
            trajectory_checking=self.trajectory_checking,
            delta=0.05,
            w=0.9,
            alpha=0.1,
        )

    def bind_model(self, model_handle: Any) -> None:
        self.model_handle = model_handle
        conformal_cfg = self._build_conformal_config()
        self.estimator = ProbabilityEstimator(model_handle, self.system, conformal_cfg, self.device)

    def estimate(self, start_states: np.ndarray) -> OutcomeProbabilities:
        if self.estimator is None:
            raise RuntimeError("Probability backend used before bind_model")
        p_success, p_failure, p_invalid = self.estimator.estimate(start_states)
        return OutcomeProbabilities(
            p_success=np.asarray(p_success),
            p_failure=np.asarray(p_failure),
            p_invalid=np.asarray(p_invalid),
        )
