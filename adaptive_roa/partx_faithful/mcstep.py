"""MCstep (Alg. 2), Classify (Alg. 3) and the continued-sampling indicator (eq. 4).

All three work on g, the robustness-oriented latent (g < 0 = falsifying).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from adaptive_roa.partx_faithful.sampling import lhs


def z_value(delta_c: float) -> float:
    return float(norm.ppf(1.0 - delta_c / 2.0))


@dataclass
class Quantiles:
    q_max: float        # paper Q-bar: mean over R of max_m (m + z s)
    var_q_max: float    # var over R of that max, divided by R
    q_min: float        # paper Q-underbar: mean over R of min_m (m - z s)
    var_q_min: float


def mc_step(model, low, high, R: int, M: int, delta_c: float, rng) -> Quantiles:
    """Alg. 2 with one GP per region (spec §4 row 3)."""
    z = z_value(delta_c)
    X = np.vstack([lhs(M, low, high, rng) for _ in range(R)])
    m, s = model.latent(X)
    m, s = m.reshape(R, M), s.reshape(R, M)
    q_max = np.max(m + z * s, axis=1)
    q_min = np.min(m - z * s, axis=1)
    return Quantiles(float(q_max.mean()), float(q_max.var() / R),
                     float(q_min.mean()), float(q_min.var() / R))


def classify(rtype: str, volume: float, min_volume: float, q: Quantiles, delta_c: float,
             compat_ci_var: bool = False) -> str:
    """Alg. 3, as in cpslab classification.py. compat_ci_var reproduces cpslab's Var-for-sd (spec §4 row 1)."""
    if volume <= min_volume:
        return "u"
    z = z_value(delta_c)
    w_max = q.var_q_max if compat_ci_var else np.sqrt(q.var_q_max)
    w_min = q.var_q_min if compat_ci_var else np.sqrt(q.var_q_min)
    upper = q.q_max + z * w_max       # confident upper bound on the region's max of g
    lower = q.q_min - z * w_min       # confident lower bound on the region's min of g
    if rtype == "+":
        return "r+" if lower <= 0.0 else "+"
    if rtype == "-":
        return "r-" if upper >= 0.0 else "-"
    if rtype == "r":
        if upper < 0.0:
            return "-"
        if lower > 0.0:
            return "+"
        return "r"
    if rtype in ("r+", "r-"):
        return rtype                  # cpslab leaves reclassified regions unchanged
    raise ValueError(f"cannot classify a region of type {rtype!r}")


def falsification_indicator(model, low, high, R: int, M: int, rng, times_volume: bool = False) -> float:
    """Paper eq. (4): mean over the region of P(g(x) < 0) = Phi(-m/s).

    times_volume reproduces cpslab's calculate_mc_integral, which multiplies
    by the volume instead of normalizing (spec §4 row 2).
    """
    X = np.vstack([lhs(M, low, high, rng) for _ in range(R)])
    m, s = model.latent(X)
    value = float(np.mean(norm.cdf(-m / s)))
    return value * float(np.prod(np.asarray(high) - np.asarray(low))) if times_volume else value
