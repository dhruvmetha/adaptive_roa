"""Falsification-volume estimates (spec §3 "Output"; cpslab fv_without_gp / fv_using_gp)."""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

from adaptive_roa.partx_faithful.sampling import lhs


def classified_falsification_volume(leaves, root_volume: float) -> float:
    """v(Theta^-)/v(S), the return value of Alg. 4."""
    return float(sum(r.volume() for r in leaves if r.rtype == "-") / root_volume)


def remaining_plus_violating_volume(leaves, root_volume: float) -> float:
    """The paper's "P-X" row in Tables 1-3: r, r+, r- and - leaves."""
    return float(sum(r.volume() for r in leaves if r.rtype in ("r", "r+", "r-", "-")) / root_volume)


def gp_quantile_falsification_volume(leaves, models: dict, R: int, M: int, quantiles, rng,
                                     root_volume: float) -> dict:
    """cpslab fv_using_gp: per leaf (skipping u and i), the share of R*M LHS points
    whose q-quantile of g is below 0, times the leaf volume, summed over leaves."""
    out = {float(q): 0.0 for q in quantiles}
    for r in leaves:
        model = models.get(r.rid)
        if r.rtype in ("u", "i") or model is None:
            continue
        X = np.vstack([lhs(M, r.low, r.high, rng) for _ in range(R)])
        m, s = model.latent(X)
        for q in out:
            out[q] += float(np.mean(norm.ppf(q, m, s) < 0.0)) * r.volume()
    return {q: v / root_volume for q, v in out.items()}
