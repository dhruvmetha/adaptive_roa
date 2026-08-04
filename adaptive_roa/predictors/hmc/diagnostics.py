"""Convergence and fidelity diagnostics for the HMC reference tier.

R-hat is computed in FUNCTION space (per query point's predictive probability),
not weight space. Weight-space R-hat for a neural network is hopeless -- the
posterior is massively multimodal under permutation symmetry, so chains that
agree perfectly on predictions still look divergent in weights. Published BNN
reference work reports the function-space statistic for exactly this reason.
"""
from __future__ import annotations

import torch


def function_space_rhat(predictions: torch.Tensor) -> torch.Tensor:
    """Split-free Gelman-Rubin per point. ``predictions`` is [chains, draws, N]."""
    if predictions.shape[0] < 2:
        raise ValueError(
            f"R-hat needs at least 2 chains, got {predictions.shape[0]}"
        )
    m, n = predictions.shape[0], predictions.shape[1]
    chain_means = predictions.mean(dim=1)                       # [m, N]
    chain_vars = predictions.var(dim=1, unbiased=True)          # [m, N]
    w = chain_vars.mean(dim=0)                                  # within
    b = chain_means.var(dim=0, unbiased=True) * n               # between
    var_hat = (n - 1) / n * w + b / n
    return torch.sqrt((var_hat / w.clamp_min(1e-12)).clamp_min(0.0))


def agreement(p: torch.Tensor, q: torch.Tensor, threshold: float = 0.5) -> float:
    """Fraction of points where two predictives give the same hard label."""
    return float(((p >= threshold) == (q >= threshold)).float().mean().item())


def total_variation(p: torch.Tensor, q: torch.Tensor) -> float:
    """Mean total-variation distance between two Bernoulli predictives."""
    return float((p - q).abs().mean().item())


def hmc_vs_hmc_ceiling(predictions: torch.Tensor) -> dict:
    """How well HMC agrees with ITSELF, leave-one-chain-out.

    This is the ceiling an approximation competes against. Without it, an
    agreement of 0.71 invites comparison against 1.0 when the achievable maximum
    at this sample size may be 0.85.
    """
    if predictions.shape[0] < 2:
        raise ValueError(
            f"the HMC-vs-HMC ceiling needs at least 2 chains, got {predictions.shape[0]}"
        )
    agreements, tvs = [], []
    for i in range(predictions.shape[0]):
        held = predictions[i].mean(dim=0)
        rest = torch.cat([predictions[:i], predictions[i + 1:]], dim=0)
        rest_mean = rest.reshape(-1, rest.shape[-1]).mean(dim=0)
        agreements.append(agreement(held, rest_mean))
        tvs.append(total_variation(held, rest_mean))
    return {
        "agreement": float(sum(agreements) / len(agreements)),
        "total_variation": float(sum(tvs) / len(tvs)),
        "n_chains": int(predictions.shape[0]),
    }
