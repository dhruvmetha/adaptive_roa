"""Discriminative-classifier probability estimator.

Drop-in replacement for ``ProbabilityEstimator`` that returns outcome
probabilities from a single forward pass of a binary classifier instead of
Monte-Carlo rollouts of a generative flow matcher.

Contract (identical to ``conformal.probability_estimator.ProbabilityEstimator``):
    estimate(states, ...) -> (p_success, p_failure, p_invalid)
    each a ``[N]`` ``np.float64`` array.

For a binary classifier:
    p_success = sigmoid(logit)
    p_failure = 1 - p_success
    p_invalid = 0          # a discriminative classifier never produces an
                           # "unresolved" outcome; that only arises when
                           # classifying *generated* endpoints.

The bound ``model`` is expected to be callable as ``model(raw_states) -> logits``
(the ``ClassifierModule`` performs normalization + embedding internally), so this
estimator stays agnostic to the embedding details.
"""
from __future__ import annotations

from typing import Any, Tuple

import numpy as np
import torch


class ClassifierProbabilityEstimator:
    def __init__(self, model: Any, system: Any, config: Any, device: str = "cuda"):
        self.model = model
        self.system = system
        self.config = config
        self.device = device
        if hasattr(self.model, "eval"):
            self.model.eval()

    @torch.no_grad()
    def estimate(
        self,
        states,
        verbose: bool = True,
        refine_invalids: bool | None = None,
        refine_t_range: Tuple[float, float] | None = None,
        refine_num_steps: int | None = None,
        refine_max_attempts: int | None = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (p_success, p_failure, p_invalid), each [N] float64.

        Extra ``refine_*`` kwargs are accepted for signature parity with the
        MC estimator and ignored (no generation => nothing to refine).
        """
        if torch.is_tensor(states):
            x = states.to(device=self.device, dtype=torch.float32)
        else:
            x = torch.as_tensor(np.asarray(states), dtype=torch.float32, device=self.device)
        if x.dim() == 1:
            x = x.unsqueeze(0)

        n = x.shape[0]
        batch_size = getattr(self.config, "mc_batch_size", None) or 8192

        probs = []
        for start in range(0, n, batch_size):
            logits = self.model(x[start:start + batch_size])
            if logits.dim() > 1:
                logits = logits.squeeze(-1)
            probs.append(torch.sigmoid(logits))

        p_success = torch.cat(probs).to(torch.float64).cpu().numpy()
        p_failure = 1.0 - p_success
        p_invalid = np.zeros_like(p_success)
        return p_success, p_failure, p_invalid
