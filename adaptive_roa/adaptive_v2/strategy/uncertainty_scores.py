"""Split an ensemble's predictive uncertainty into aleatoric and epistemic parts.

For M members with per-member success probabilities p_1..p_M:

    H(p_bar)  =  E_m[H(p_m)]  +  I(y ; m | x)
    total        aleatoric       epistemic

Acquiring on the epistemic term targets what more data can fix; acquiring on the
total confuses that with irreducible outcome randomness, which is how entropy
acquisition degenerates under heavy process noise.

FINITE-SAMPLE WARNING. When each p_m is itself estimated from K Monte-Carlo
samples (flow matching), `epistemic_bald` is biased UPWARD by roughly
(1/2K)(1 - 1/M) -- ~0.021 nats at K=20, M=5. That bias is roughly flat across
the interior and vanishes only at p = 0 and 1, so it lifts every non-deterministic
state above every decided one regardless of whether members actually disagree.
It does NOT shrink as M grows. `epistemic_variance` with `k` set removes exactly
this term and is the safe default; pass `k=None` when p_m carries no sampling
noise (a classifier forward pass).
"""
from __future__ import annotations

import numpy as np
from scipy.special import xlogy

SCORE_MODES: tuple[str, ...] = ("total", "aleatoric", "epistemic_bald", "epistemic_var")


def binary_entropy(p: np.ndarray) -> np.ndarray:
    """Shannon entropy of Bernoulli(p) in nats; EXACTLY 0 at p = 0 and p = 1.

    xlogy(0, 0) is defined as 0, so the boundaries are exact rather than
    clipped. That exactness is load-bearing: the finite-K bias in
    epistemic_bald vanishes at deterministic states and is ~(1/2K)(1-1/M)
    across the interior, and a clipped boundary would blur that distinction.
    """
    p = np.asarray(p, dtype=np.float64)
    return -(xlogy(p, p) + xlogy(1.0 - p, 1.0 - p))


def _check(p_members: np.ndarray) -> np.ndarray:
    p = np.asarray(p_members, dtype=np.float64)
    if p.ndim != 2:
        raise ValueError(f"p_members must be [M, N], got shape {p.shape}")
    if p.shape[0] < 2:
        raise ValueError(f"need >= 2 members to separate epistemic, got {p.shape[0]}")
    return p


def total_uncertainty(p_members: np.ndarray) -> np.ndarray:
    """H of the ensemble marginal: the score the current entropy strategy uses."""
    return binary_entropy(_check(p_members).mean(axis=0))


def aleatoric_uncertainty(p_members: np.ndarray) -> np.ndarray:
    """Mean per-member entropy: the irreducible part. Used as a negative control."""
    return binary_entropy(_check(p_members)).mean(axis=0)


def epistemic_bald(p_members: np.ndarray) -> np.ndarray:
    """Mutual information between the label and the member index (BALD).

    Unbiased only when p_m carries no sampling noise. See the module warning.
    """
    p = _check(p_members)
    return binary_entropy(p.mean(axis=0)) - binary_entropy(p).mean(axis=0)


def epistemic_variance(p_members: np.ndarray, k: float | None) -> np.ndarray:
    """Between-member variance, debiased for each member's K-sample MC noise.

    Var_m[p_m] - mean_m[p_m(1-p_m)/(K-1)]. The subtracted term is the unbiased
    estimate of a proportion's sampling variance, so the result estimates true
    member disagreement and goes to 0 when members agree, at any K. Pass k=None
    when p_m is exact (no sampling), which skips the correction.
    """
    p = _check(p_members)
    var = p.var(axis=0, ddof=1)
    if k is None or not np.isfinite(k) or k <= 1:
        return var
    return var - (p * (1.0 - p) / (float(k) - 1.0)).mean(axis=0)


def score_by_mode(mode: str, p_members: np.ndarray, k: float | None) -> np.ndarray:
    """Dispatch to one score. `k` is ignored by every mode except epistemic_var."""
    if mode == "total":
        return total_uncertainty(p_members)
    if mode == "aleatoric":
        return aleatoric_uncertainty(p_members)
    if mode == "epistemic_bald":
        return epistemic_bald(p_members)
    if mode == "epistemic_var":
        return epistemic_variance(p_members, k)
    raise ValueError(f"unknown score mode {mode!r}; expected one of {SCORE_MODES}")
