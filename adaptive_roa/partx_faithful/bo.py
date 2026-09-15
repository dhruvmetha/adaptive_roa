"""SampleBO (Alg. 1): LHS top-up to n0, then nBO sequential expected-improvement steps.

EI minimizes g, so it hunts falsifiers (paper eq. 3). With binary outcomes
there is no observed best value, so f* is the plug-in minimum of the posterior
mean at the region's sampled points (spec §3, "EI on binary data"). The
"observed" mode (min of y) exists for the deterministic reference check.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

from adaptive_roa.partx_faithful.sampling import lhs


def expected_improvement(m, s, f_star: float) -> np.ndarray:
    s = np.maximum(np.asarray(s, dtype=float), 1e-12)
    imp = f_star - np.asarray(m, dtype=float)
    u = imp / s
    return imp * norm.cdf(u) + s * norm.pdf(u)


def sample_bo(region, store, sampler, fit_model, take, n0: int, n_bo: int, cand_cap: int, rng,
              f_star_mode: str = "plugin") -> dict:
    """Alg. 1 on one region. Always takes n_bo EI steps, as cpslab does (spec §4 row 4)."""
    new_ids: list = []
    shortfall = 0
    need = max(int(n0) - len(region.samples), 0)
    if need > 0:
        if sampler.n_available(region) == 0:
            return {"new_ids": new_ids, "shortfall": need + int(n_bo), "infeasible": True}
        handles = sampler.snap_design(lhs(need, region.low, region.high, rng), region)
        shortfall += need - len(handles)
        new_ids += take(handles)
    for t in range(int(n_bo)):
        Xc, hc = sampler.candidates(region, cand_cap, rng)
        if len(Xc) == 0 or not region.samples:
            return {"new_ids": new_ids, "shortfall": shortfall + int(n_bo) - t, "infeasible": True}
        model = fit_model(region)
        m, s = model.latent(Xc)
        if f_star_mode == "plugin":
            # f* uses only the posterior mean; computing it without the variance is identical
            # and saves the variance over every sample in the region.
            Xs = store.X(region.samples)
            means = model.latent_mean(Xs) if hasattr(model, "latent_mean") else model.latent(Xs)[0]
            f_star = float(np.min(means))
        elif f_star_mode == "observed":
            f_star = float(np.min(store.y(region.samples)))
        else:
            raise ValueError(f"unknown f_star_mode {f_star_mode!r}")
        j = int(np.argmax(expected_improvement(m, s, f_star)))
        new_ids += take(hc[j:j + 1])
    return {"new_ids": new_ids, "shortfall": shortfall, "infeasible": False}
