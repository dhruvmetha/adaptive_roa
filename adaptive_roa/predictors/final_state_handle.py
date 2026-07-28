"""Model handle binding a final-state predictor to the endpoint-MC backend.

Satisfies the contract ``ProbabilityEstimator`` and ``compute_endpoint_prediction_error``
already rely on, so the Bayesian final-state arms route through the existing
conformal, threshold, and evaluation machinery unchanged.
"""
from __future__ import annotations

from typing import Any, List

import numpy as np
import torch


class _ManifoldDistanceShim:
    """Minimal ``distance_manifold`` stand-in exposing ``dist``.

    full_roa.py guards get_manifold_component_names() behind
    hasattr(model, "distance_manifold"), so supplying only one of the pair means
    the per-component error stats are silently skipped (or crash). This shim
    delegates to the head so both consumers agree.
    """

    def __init__(self, head):
        self._head = head

    def dist(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self._head.distance_per_component(x, y)


class FinalStateModelHandle:
    """Predicts a distribution over x_T and samples ONE endpoint per call.

    Determinism here is the OPPOSITE of ``OutcomeModelHandle``: the estimator
    calls ``predict_endpoint`` K times on the same batch and the spread across
    those calls IS the outcome probability. Every call therefore draws a fresh
    weight sample AND a fresh head sample. Seeding this handle collapses every
    arm to p in {0, 1}.
    """

    def __init__(self, posterior, head, system: Any, device: str = "cpu"):
        self.posterior = posterior
        self.head = head
        self.system = system
        self.device = device
        self.training = False
        self.distance_manifold = _ManifoldDistanceShim(head)

    def eval(self):
        self.posterior.eval()
        self.training = False
        return self

    def train(self, mode: bool = True):
        self.posterior.train(mode)
        self.training = bool(mode)
        return self

    def to(self, device):
        self.device = device
        self.posterior.to(device)
        return self

    def _params(self, states: torch.Tensor) -> torch.Tensor:
        embedded = self.system.embed_state_for_model(self.system.normalize_state(states))
        return self.posterior.forward_sample(embedded)

    def predict_endpoint(self, states) -> torch.Tensor:
        """[B, state_dim] raw -> [B, state_dim] raw. Fresh sample every call."""
        if torch.is_tensor(states):
            x = states.detach().to(dtype=torch.float32)
            out_device = states.device
        else:
            x = torch.as_tensor(np.asarray(states), dtype=torch.float32)
            out_device = self.device
        x = x.to(next(self.posterior.parameters()).device)
        with torch.no_grad():
            endpoints = self.head.sample(self._params(x))
        return endpoints.to(out_device)

    def get_manifold_component_names(self) -> List[str]:
        return list(self.head.component_names)

    def compute_manifold_distance_per_component(self, predicted, true) -> torch.Tensor:
        return self.head.distance_per_component(predicted, true)
