"""Monte Carlo endpoint probability backend."""

from __future__ import annotations

from typing import Any

import numpy as np

from adaptive_roa.conformal import ConformalConfig
from adaptive_roa.conformal.probability_estimator import ProbabilityEstimator
from adaptive_roa.adaptive_v2.types import OutcomeProbabilities


class EndpointMCProbabilityBackend:
    """Wraps legacy `ProbabilityEstimator` with the v2 interface."""

    def __init__(self, system: Any, cfg: Any, device: str):
        self.system = system
        self.cfg = cfg
        self.device = device
        self.model_handle = None
        self.estimator: ProbabilityEstimator | None = None

    def bind_model(self, model_handle: Any) -> None:
        self.model_handle = model_handle
        conformal_cfg = ConformalConfig(
            delta=self.cfg.conformal.get("delta", 0.05),
            w=self.cfg.conformal.get("w", 0.9),
            alpha=self.cfg.conformal.get("alpha_sampling", 0.1),
            num_mc_samples=self.cfg.conformal.get("num_mc_samples", 100),
            mc_batch_size=self.cfg.conformal.get("mc_batch_size", 1024),
            attractor_radius=self.cfg.conformal.get("attractor_radius", 0.2),
            optimize_mode=self.cfg.conformal.get("optimize_mode", "lambda"),
            decision_rule=self.cfg.conformal.get("decision_rule", "two_sided"),
            lambda_grid_size=self.cfg.conformal.get("lambda_grid_size", 100),
            delta_grid_size=self.cfg.conformal.get("delta_grid_size", 100),
            delta_min=self.cfg.conformal.get("delta_min", 0.01),
            delta_max=self.cfg.conformal.get("delta_max", 0.49),
            refine_invalids=self.cfg.conformal.get("refine_invalids", False),
            refine_t_min=self.cfg.conformal.get("refine_t_min", 0.7),
            refine_t_max=self.cfg.conformal.get("refine_t_max", 0.9),
            refine_num_steps=self.cfg.conformal.get("refine_num_steps", 100),
            refine_max_attempts=self.cfg.conformal.get("refine_max_attempts", 5),
        )
        self.conformal_cfg = conformal_cfg
        self.estimator = ProbabilityEstimator(model_handle, self.system, conformal_cfg, self.device)

    def estimate(self, start_states: np.ndarray) -> OutcomeProbabilities:
        if self.estimator is None:
            raise RuntimeError("Probability backend used before bind_model")

        p_success, p_failure, p_invalid = self.estimator.estimate(
            start_states,
            refine_invalids=self.conformal_cfg.refine_invalids,
            refine_t_range=(self.conformal_cfg.refine_t_min, self.conformal_cfg.refine_t_max),
            refine_num_steps=self.conformal_cfg.refine_num_steps,
            refine_max_attempts=self.conformal_cfg.refine_max_attempts,
        )
        return OutcomeProbabilities(
            p_success=np.asarray(p_success),
            p_failure=np.asarray(p_failure),
            p_invalid=np.asarray(p_invalid),
        )
