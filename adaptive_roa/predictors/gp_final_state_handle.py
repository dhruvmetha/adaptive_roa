"""Model handle binding the GP regressor to the endpoint-MC backend.

Mirrors ``adaptive_roa/predictors/final_state_handle.py``: fresh sample per call,
raw-coordinate output, and the same normalized/raw split between
``compute_manifold_distance_per_component`` (normalized, to match the flow
matcher) and ``distance_manifold.dist`` (raw, because full_roa passes raw for
both families).
"""
from __future__ import annotations

from typing import Any, List

import numpy as np
import torch

from adaptive_roa.predictors.embedding import EmbeddedStateDecoder
from adaptive_roa.predictors.heads import FinalStateHead


class _ManifoldDistanceShim:
    def __init__(self, head):
        self._head = head

    def dist(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self._head.distance_per_component(x, y)


class GPFinalStateHandle:
    def __init__(self, gp, system: Any, device: str = "cpu"):
        self.gp = gp
        self.system = system
        self.device = device
        self.training = False
        self.decoder = EmbeddedStateDecoder(system)
        # FinalStateHead is used ONLY for its system-derived distance and naming;
        # neither depends on distribution parameters, so this avoids duplicating
        # the per-component geodesic logic.
        self.head = FinalStateHead(system)
        self.distance_manifold = _ManifoldDistanceShim(self.head)

    def eval(self):
        self.gp.eval()
        self.training = False
        return self

    def to(self, device):
        self.device = device
        self.gp.to(device)
        return self

    def predict_endpoint(self, states) -> torch.Tensor:
        """[B, state_dim] raw -> [B, state_dim] raw. Fresh predictive draw per call."""
        if torch.is_tensor(states):
            x = states.detach().to(dtype=torch.float32)
            out_device = states.device
        else:
            x = torch.as_tensor(np.asarray(states), dtype=torch.float32)
            out_device = self.device
        feats = self.system.embed_state_for_model(self.system.normalize_state(x))
        # One predictive draw, through the likelihood so it carries observation noise.
        sample = self.gp.sample(feats, num_samples=1)[0]
        return self.decoder.decode(sample).to(out_device)

    def get_manifold_component_names(self) -> List[str]:
        return list(self.head.component_names)

    def compute_manifold_distance_per_component(self, predicted, true) -> torch.Tensor:
        # Normalized, matching flow_matcher.py:1156-1161 -- both feed the same
        # artifacts_v2.json field, compared positionally across families.
        return self.head.distance_per_component(
            self.system.normalize_state(predicted), self.system.normalize_state(true)
        )
