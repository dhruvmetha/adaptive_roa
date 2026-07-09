from __future__ import annotations

import numpy as np
import torch


class GPModelHandle:
    """Engine-compatible wrapper around a fitted GPClassifier.

    The shared conformal/eval code paths for ``predictor_type == "classifier"``
    (``ClassifierProbabilityEstimator.estimate`` and
    ``full_roa.evaluate_full_roa_classifier``) call the bound model as
    ``logits = model(raw_states_tensor)`` and then take ``sigmoid(logits)`` as
    ``p_success``. ``__call__`` here reproduces that exact contract on top of
    the GP's own predictive probability (``GPClassifier.p_success``), so the
    GP predictor can be routed through the existing classifier-shaped
    machinery (``predictor.type: classifier`` in configs/adaptive_v2/predictor/gp.yaml)
    without any changes to that shared code.
    """

    def __init__(self, gp, system):
        self.gp = gp
        self.system = system

    def eval(self):
        self.gp.eval()
        return self

    def to(self, device):
        self.gp.to(device)
        return self

    def __call__(self, states):
        if torch.is_tensor(states):
            X_raw = states.detach().cpu().numpy()
        else:
            X_raw = np.asarray(states)
        p_success = np.clip(np.asarray(self.gp.p_success(X_raw), dtype=np.float64), 1e-12, 1 - 1e-12)
        logits = np.log(p_success / (1.0 - p_success))
        device = states.device if torch.is_tensor(states) else "cpu"
        return torch.as_tensor(logits, dtype=torch.float32, device=device)
