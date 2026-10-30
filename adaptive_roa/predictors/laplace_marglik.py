"""Prior-precision selection for last-layer Laplace by the Laplace evidence.

The last-layer Laplace posteriors here share one feature-space precision across
the output rows of a block: every row's weights get covariance
``(G + delta I)^-1`` with ``G = sum_n lam_n phi_n phi_n^T``. Under the isotropic
prior ``N(0, delta^-1 I)`` over those ``O x D`` weights, the Laplace
approximation to the log evidence, keeping only the terms that depend on delta
(the MAP and therefore the data fit are held fixed), is

    f(delta) = -1/2 delta ||W||^2 + (O D / 2) log delta - (O / 2) sum_i log(g_i + delta)

with ``g_i`` the eigenvalues of ``G`` and ``W`` the block's MAP weights (bias
column included). This is the post-hoc evidence of Immer et al. (ICML 2021) and
Daxberger et al. (NeurIPS 2021) for an isotropic last-layer prior.

Why tune it at all: the trainers used to pin ``delta = 1 / prior_sigma^2 = 1``.
Against 1e5 to 1e7 intermediate-state rows the data term of ``G`` swamps that
constant in every direction the data spans, so the prior only acts on the
directions the data leaves empty, at a scale nobody chose. The evidence picks
the scale from the data instead.

The maximizer is unique. The derivative

    f'(delta) = -1/2 ||W||^2 + (O / 2) sum_i g_i / (delta (g_i + delta))

has a sum term that falls strictly from +inf to 0 as delta grows, so it crosses
zero exactly once whenever ``||W|| > 0``. At that point
``delta ||W||^2 = O sum_i g_i / (g_i + delta)``, MacKay's fixed point with the
effective number of parameters on the right. Solving ``f' = 0`` by bisection in
log delta is therefore exact up to the bracket, with no grid to tune.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

# Bracket for the bisection, in log10(delta). Wide on purpose: a root at either
# edge means the evidence is still rising there, and is reported as clipped.
_LOG10_LO = -12.0
_LOG10_HI = 12.0
_BISECTION_STEPS = 200


@dataclass
class PriorPrecisionFit:
    prior_precision: float
    log_evidence: float          # f(delta*), the delta-dependent part only
    effective_params: float      # O * sum_i g_i / (g_i + delta*)
    n_params: int                # O * D
    weight_sqnorm: float
    clipped: bool                # root fell on a bracket edge

    def as_dict(self) -> dict:
        return {
            "prior_precision": self.prior_precision,
            "log_evidence_delta_terms": self.log_evidence,
            "effective_params": self.effective_params,
            "n_params": self.n_params,
            "weight_sqnorm": self.weight_sqnorm,
            "clipped_at_bracket": self.clipped,
        }


def log_evidence(eigvals: torch.Tensor, weight_sqnorm: float, n_rows: int, delta: float) -> float:
    """The delta-dependent part of the Laplace log evidence, ``f(delta)``."""
    g = eigvals.double().clamp_min(0.0)
    d = g.numel()
    return float(
        -0.5 * delta * weight_sqnorm
        + 0.5 * n_rows * d * math.log(delta)
        - 0.5 * n_rows * torch.log(g + delta).sum()
    )


def _grad_sign(g: torch.Tensor, weight_sqnorm: float, n_rows: int, delta: float) -> float:
    return -0.5 * weight_sqnorm + 0.5 * n_rows * float((g / (delta * (g + delta))).sum())


def optimize_prior_precision(ggn: torch.Tensor, map_weights: torch.Tensor) -> PriorPrecisionFit:
    """Maximize the Laplace evidence over an isotropic last-layer prior precision.

    Args:
        ggn: ``[D, D]`` data term ``G`` of the block's precision (no prior added).
        map_weights: ``[O, D]`` MAP weights of the block, bias column included.
    """
    if ggn.ndim != 2 or ggn.shape[0] != ggn.shape[1]:
        raise ValueError(f"ggn must be square, got {tuple(ggn.shape)}")
    if map_weights.ndim != 2 or map_weights.shape[1] != ggn.shape[0]:
        raise ValueError(
            f"map_weights must be [O, {ggn.shape[0]}], got {tuple(map_weights.shape)}")
    g = torch.linalg.eigvalsh(0.5 * (ggn.double() + ggn.double().T)).clamp_min(0.0)
    n_rows = int(map_weights.shape[0])
    w2 = float(map_weights.double().pow(2).sum())
    if w2 <= 0.0:
        raise ValueError(
            "MAP weights are identically zero, so the evidence increases without "
            "bound in delta; there is no prior precision to select."
        )

    lo, hi = _LOG10_LO, _LOG10_HI
    clipped = False
    if _grad_sign(g, w2, n_rows, 10.0 ** lo) <= 0.0:
        root, clipped = lo, True
    elif _grad_sign(g, w2, n_rows, 10.0 ** hi) >= 0.0:
        root, clipped = hi, True
    else:
        for _ in range(_BISECTION_STEPS):
            mid = 0.5 * (lo + hi)
            if _grad_sign(g, w2, n_rows, 10.0 ** mid) > 0.0:
                lo = mid
            else:
                hi = mid
            if hi - lo < 1e-12:
                break
        root = 0.5 * (lo + hi)

    delta = 10.0 ** root
    eff = n_rows * float((g / (g + delta)).sum())
    return PriorPrecisionFit(
        prior_precision=delta,
        log_evidence=log_evidence(g, w2, n_rows, delta),
        effective_params=eff,
        n_params=n_rows * int(g.numel()),
        weight_sqnorm=w2,
        clipped=clipped,
    )
