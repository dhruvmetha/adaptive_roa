from __future__ import annotations

from typing import Any

import numpy as np

from adaptive_roa.adaptive_v2.types import OutcomeProbabilities


class GPProbabilityBackend:
    """Probability backend backed by a GP classifier's latent posterior."""

    def __init__(self, cfg: Any, system: Any, device: str):
        self.cfg = cfg
        self.system = system
        self.device = device
        self.model_handle = None

    def bind_model(self, model_handle: Any) -> None:
        self.model_handle = model_handle

    @property
    def gp(self):
        if self.model_handle is None:
            raise RuntimeError("GPProbabilityBackend used before bind_model")
        return self.model_handle.gp

    def estimate(self, start_states: np.ndarray) -> OutcomeProbabilities:
        p_s = np.asarray(self.gp.p_success(start_states), dtype=float)
        return OutcomeProbabilities(
            p_success=p_s, p_failure=1.0 - p_s, p_invalid=np.zeros_like(p_s)
        )

    def latent_posterior(self, start_states: np.ndarray):
        return self.gp.latent_posterior(start_states)
