"""Part-X's scored readout (spec §5.4) in the evaluator's classifier contract.

evaluate_full_roa_classifier calls model(raw_states_tensor) -> logits and takes
sigmoid(logits) as p_success, the same contract adaptive_roa/partx/model_handle.py
satisfies.
"""
from __future__ import annotations

import numpy as np
import torch


class PiecewiseModel:
    def __init__(self, leaves, models: dict, root_low, root_high):
        self.leaves = list(leaves)
        self.models = models
        self.low = np.asarray(root_low, dtype=float)
        self.high = np.asarray(root_high, dtype=float)

    def p_success(self, X) -> np.ndarray:
        X = np.clip(np.atleast_2d(np.asarray(X, dtype=float)), self.low, self.high)
        p = np.full(len(X), np.nan)
        todo = np.ones(len(X), dtype=bool)
        for leaf in self.leaves:
            m = todo & leaf.contains(X)
            if m.any():
                p[m] = self.models[leaf.rid].p_success(X[m])
                todo &= ~m
        if todo.any():
            raise RuntimeError(f"readout: {int(todo.sum())} points fall in no leaf")
        return p


class PiecewiseHandle:
    def __init__(self, piecewise: PiecewiseModel):
        self.pw = piecewise

    def eval(self):
        return self

    def to(self, device):
        return self

    def __call__(self, states):
        X = states.detach().cpu().numpy() if torch.is_tensor(states) else np.asarray(states)
        p = np.clip(self.pw.p_success(X), 1e-12, 1.0 - 1e-12)
        device = states.device if torch.is_tensor(states) else "cpu"
        return torch.as_tensor(np.log(p / (1.0 - p)), dtype=torch.float32, device=device)
