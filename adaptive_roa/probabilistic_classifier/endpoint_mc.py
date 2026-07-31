"""The endpoint-MC probability loop shared by every final-state export wrapper.

Both ``bayesian_final_state.FinalStateProbabilisticClassifier`` (mlp_det,
bnn_mfvi_reg, bnn_ensemble_reg, bnn_laplace_reg) and
``gaussian_process.GPRegProbabilisticClassifier`` turn a distribution over the
endpoint into outcome probabilities exactly the same way: draw K endpoints per
query and count ``system.classify_attractor`` labels. This module owns that loop
so the label mapping, the batching and the empty-input guard exist once. The GP
wrapper previously carried its own copy without the batching or the guard.
"""
from __future__ import annotations

import numpy as np
import torch

from adaptive_roa.adaptive_v2.types import OutcomeProbabilities

# MC classification batch size: matches the outcome arms' export batch size
# (bayesian.py / classifier.py); K forward passes run per batch regardless.
BATCH_SIZE = 8192


def endpoint_mc_probabilities(
    handle, system, states, attractor_radius: float, num_mc_samples: int,
    batch_size: int = BATCH_SIZE,
) -> OutcomeProbabilities:
    """K endpoint draws per query, counted into (p_success, p_failure, p_invalid).

    ``classify_attractor``'s label convention is 1 = success, -1 = failure and
    0 = invalid (unresolved), and it is applied here and nowhere else.

    Batching is not only about memory. ``GPRegressor.sample`` draws from a JOINT
    MVN over the query batch and its accuracy depends on that joint staying
    inside gpytorch's exact-Cholesky regime; the GP handle chunks internally, but
    keeping the export path's batch bounded here as well means every wrapper
    reaches the model with the same shaped call.
    """
    n = len(states)
    if n == 0:
        z = np.zeros(0)
        return OutcomeProbabilities(p_success=z, p_failure=z, p_invalid=z)

    k = int(num_mc_samples)
    success = np.zeros(n, dtype=np.int64)
    failure = np.zeros(n, dtype=np.int64)
    invalid = np.zeros(n, dtype=np.int64)
    with torch.no_grad():
        for i in range(0, n, batch_size):
            j = min(i + batch_size, n)
            batch = states[i:j]
            for _ in range(k):
                # Fresh weight sample AND fresh head sample every call -- the
                # spread across these K draws IS the outcome probability
                # (FinalStateModelHandle / GPFinalStateHandle's contract).
                endpoints = handle.predict_endpoint(batch)
                labels = system.classify_attractor(
                    endpoints, radius=attractor_radius
                ).cpu().numpy()
                success[i:j] += (labels == 1)
                failure[i:j] += (labels == -1)
                invalid[i:j] += (labels == 0)

    return OutcomeProbabilities(
        p_success=success.astype(np.float64) / k,
        p_failure=failure.astype(np.float64) / k,
        p_invalid=invalid.astype(np.float64) / k,
    )
