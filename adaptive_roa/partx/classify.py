from __future__ import annotations

import numpy as np
from scipy.stats import norm


def z_from_alpha(alpha: float) -> float:
    return float(norm.ppf(1.0 - alpha))


def classify_region(m: np.ndarray, s2: np.ndarray, alpha: float = 0.05) -> str:
    """Classify a region from latent posteriors at MC points.

    '+' if the alpha-quantile of the lower confidence bound is > 0
        (robustly f>0 across the region),
    '-' if the (1-alpha)-quantile of the upper confidence bound is < 0,
    'r' otherwise (straddles the boundary -> subdivide).
    """
    c = z_from_alpha(alpha)
    s = np.sqrt(np.maximum(s2, 0.0))
    lcb, ucb = m - c * s, m + c * s
    if np.quantile(lcb, alpha) > 0.0:
        return "+"
    if np.quantile(ucb, 1.0 - alpha) < 0.0:
        return "-"
    return "r"
