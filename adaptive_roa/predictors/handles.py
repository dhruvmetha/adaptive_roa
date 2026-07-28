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
import torch.nn.functional as F


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

        # The generator must live on the same device the posteriors draw on:
        # VILinear draws on the parameter device, and Ensemble/Laplace now do
        # too (see posteriors.py). A CPU generator paired with CUDA parameters
        # (or vice versa) raises a device-mismatch RuntimeError.
        param_device = next(self.posterior.parameters()).device
        generator = torch.Generator(device=param_device).manual_seed(self.seed)
        with torch.no_grad():
            # Ensembles override this to enumerate all M members exactly;
            # everything else draws S samples. See Posterior.predictive_logit_samples.
            samples = self.posterior.predictive_logit_samples(
                embedded, S=self.n_marginal_samples, generator=generator
            )  # [S, B, 1]
            s = samples.view(samples.shape[0], samples.shape[1])

            # Marginalize in PROBABILITY space, not logit space: the Bayesian
            # model average is E_q[p(y|x,w)], and averaging logits instead would
            # be a different (and systematically overconfident) estimator.
            #
            # But compute that average in LOG space. The naive form --
            # p = sigmoid(s).mean(0); log(p / (1 - p)) -- breaks in float32:
            # sigmoid saturates to exactly 1.0 for s >~ 17, and the obvious
            # guard `p.clamp(eps, 1 - eps)` is a NO-OP on the upper side because
            # float32(1 - 1e-12) rounds to exactly 1.0. A posterior whose draws
            # all saturate then returns +inf logits, which propagate silently
            # into thresholds and calibration. (The same idiom is safe in
            # partx/model_handle.py only because that code runs in float64.)
            #
            # Identity used, for p_bar = (1/S) sum_i sigmoid(s_i):
            #   logit(p_bar) = log p_bar - log(1 - p_bar)
            #                = logsumexp_i logsigmoid(s_i)
            #                  - logsumexp_i logsigmoid(-s_i)
            # since 1 - sigmoid(s) = sigmoid(-s) and the two -log(S) terms
            # cancel. Exact, and finite for every finite input.
            logit_p = (torch.logsumexp(F.logsigmoid(s), dim=0)
                       - torch.logsumexp(F.logsigmoid(-s), dim=0))

        return logit_p.to(out_device)
