"""Duck-typed model handles binding predictors to the existing v2 backends.

``OutcomeModelHandle`` reproduces the contract that
``ClassifierProbabilityEstimator`` and ``full_roa.evaluate_full_roa_classifier``
already rely on -- ``model(raw_states) -> logits``, with ``sigmoid(logits)`` read
as p(success) -- exactly as ``adaptive_roa/partx/model_handle.py`` does for the
GP. Satisfying it means the Bayesian arms route through the shared conformal,
threshold, and evaluation machinery with no changes to that code.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

_EPS = 1e-12


class OutcomeModelHandle:
    """Marginalizes a weight posterior into a single deterministic logit vector.

    The marginalization is INTERNAL and seeded on purpose. The estimator calls
    this once per query, but threshold optimization, calibration, and evaluation
    each call it separately on overlapping states and must agree; a fresh draw
    per call would desynchronize them in a way that looks like a calibration bug
    rather than a sampling bug.
    """

    def __init__(self, posterior, system: Any, n_marginal_samples: int = 64,
                 seed: int = 0, device: str = "cpu"):
        self.posterior = posterior
        self.system = system
        self.n_marginal_samples = int(n_marginal_samples)
        self.seed = int(seed)
        self.device = device

    def eval(self):
        self.posterior.eval()
        return self

    def to(self, device):
        self.device = device
        self.posterior.to(device)
        return self

    def __call__(self, states) -> torch.Tensor:
        if torch.is_tensor(states):
            x = states.detach().to(dtype=torch.float32)
            out_device = states.device
        else:
            x = torch.as_tensor(np.asarray(states), dtype=torch.float32)
            out_device = self.device
        x = x.to(next(self.posterior.parameters()).device)

        embedded = self.system.embed_state_for_model(self.system.normalize_state(x))

        generator = torch.Generator().manual_seed(self.seed)
        with torch.no_grad():
            samples = self.posterior.forward_samples(
                embedded, S=self.n_marginal_samples, generator=generator
            )  # [S, B, 1]
            # Marginalize in PROBABILITY space, not logit space: the Bayesian
            # model average is E_q[p(y|x,w)], and averaging logits instead would
            # be a different (and systematically overconfident) estimator.
            p = torch.sigmoid(samples.view(samples.shape[0], samples.shape[1])).mean(dim=0)

        p = p.clamp(_EPS, 1.0 - _EPS)
        return torch.log(p / (1.0 - p)).to(out_device)
