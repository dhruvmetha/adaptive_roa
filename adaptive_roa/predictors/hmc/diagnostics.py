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
    """Split-free Gelman-Rubin per point. ``predictions`` is [chains, draws, N].

    Limitation: this is classic location-based Gelman-Rubin, so it is
    structurally blind to scale mismatch between chains. Four chains sharing
    a mean but with per-chain variance differing 100-fold still report
    R-hat ~ 1.0 (verified empirically) -- a chain sampling a much wider
    region than its siblings (e.g. from a step-size mismatch) would pass
    this gate unflagged. Rank-normalized or folded R-hat closes that gap if
    scale mismatches ever need to be caught; this function does not attempt
    it.

    Edge case: at a query point where every chain has collapsed to an
    identical constant, both within-chain variance W and between-chain
    variance B are 0. The `clamp_min(1e-12)` guard on the denominator then
    makes this point's R-hat read 0.0, not the conventional 1.0 -- harmless
    against a >1.1 threshold (0.0 never trips it) but worth knowing if this
    tensor is inspected directly rather than just thresholded.
    """
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


def _require_probabilities(predictions: torch.Tensor) -> None:
    """Reject anything that is not a probability, loudly.

    ``agreement`` thresholds at 0.5 and ``total_variation`` is a distance in
    [0, 1]; neither means anything for an unbounded scalar. Feeding this the
    final-state head's raw location proxy produced ``total_variation = 3.89``,
    which a total-variation distance strictly cannot be, and an ``agreement``
    that thresholded a normalized cart position at 0.5. Both numbers went into
    the run artifact looking like diagnostics.

    Deliberately NOT clamped: a clamp turns an invalid input into a
    plausible-looking output, which is the exact failure mode that let those
    numbers ship. The caller must supply a probability or fix its summary.
    """
    lo = float(predictions.min())
    hi = float(predictions.max())
    if not (0.0 <= lo and hi <= 1.0):
        raise ValueError(
            f"the HMC-vs-HMC ceiling is defined on PROBABILITIES: agreement "
            f"thresholds at 0.5 and total_variation is a distance in [0, 1]. "
            f"Got values in [{lo:.4g}, {hi:.4g}]. Summarize the predictive as a "
            f"probability (for a final-state head, p(endpoint in attractor)) "
            f"rather than passing a raw network output."
        )


def hmc_vs_hmc_ceiling(predictions: torch.Tensor) -> dict:
    """How well HMC agrees with ITSELF, leave-one-chain-out.

    This is the ceiling an approximation competes against. Without it, an
    agreement of 0.71 invites comparison against 1.0 when the achievable maximum
    at this sample size may be 0.85.

    Reports both the mean and the worst (min agreement / max total variation)
    leave-one-out fold, not just the mean. Averaging across folds can dilute a
    single badly-diverged chain: its low-agreement fold gets blended away by
    the other, well-agreeing folds, inflating the reported ceiling exactly
    when the diagnostic most needs to flag trouble. A large gap between
    `agreement` and `agreement_min` means one chain disagrees with the rest;
    `agreement_min` (and `total_variation_max`) is the conservative bound to
    quote when this ceiling is used to judge an approximation's fidelity.
    """
    if predictions.shape[0] < 2:
        raise ValueError(
            f"the HMC-vs-HMC ceiling needs at least 2 chains, got {predictions.shape[0]}"
        )
    _require_probabilities(predictions)
    agreements, tvs = [], []
    for i in range(predictions.shape[0]):
        held = predictions[i].mean(dim=0)
        rest = torch.cat([predictions[:i], predictions[i + 1:]], dim=0)
        rest_mean = rest.reshape(-1, rest.shape[-1]).mean(dim=0)
        agreements.append(agreement(held, rest_mean))
        tvs.append(total_variation(held, rest_mean))
    return {
        "agreement": float(sum(agreements) / len(agreements)),
        "agreement_min": float(min(agreements)),
        "total_variation": float(sum(tvs) / len(tvs)),
        "total_variation_max": float(max(tvs)),
        "n_chains": int(predictions.shape[0]),
    }
