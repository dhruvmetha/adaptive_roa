from __future__ import annotations

import numpy as np
from scipy.stats import norm


def roa_volume_bound(tree, latent_fn, R=200, M=64, ci=0.9, seed=42):
    rng = np.random.default_rng(seed)
    leaves = tree.leaves()
    total_vol = sum(r.volume() for r in leaves) or 1.0
    per_region = []
    # point-estimate accumulator and R posterior-sample accumulators
    vol_point = 0.0
    vol_samples = np.zeros(R)
    for r in leaves:
        w = r.volume() / total_vol
        pts = r.sample_uniform(M, rng)
        m, s2 = latent_fn(pts)
        s = np.sqrt(np.maximum(s2, 0.0))
        p_in = norm.cdf(m / np.sqrt(1.0 + s2))          # integrated predictive
        frac_point = float(np.mean(p_in))
        vol_point += w * frac_point
        # posterior draws of the RoA indicator, averaged over MC points
        draws = m[None, :] + s[None, :] * rng.standard_normal((R, len(m)))
        frac_draws = np.mean(draws > 0.0, axis=1)       # [R]
        vol_samples += w * frac_draws
        per_region.append({"rid": r.rid, "class": r.region_class,
                            "weight": w, "frac_in": frac_point})
    lo = float(np.quantile(vol_samples, (1 - ci) / 2))
    hi = float(np.quantile(vol_samples, 1 - (1 - ci) / 2))
    return {"volume": float(vol_point), "ci_low": lo, "ci_high": hi,
            "per_region": per_region}
