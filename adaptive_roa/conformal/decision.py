"""Shared pieces of the lambda/delta decision rule.

Invalid threshold (det paper, ruled 2026-10-30): a state is flagged invalid when
p_invalid >= max(lambda, 1 - lambda) - delta. Under the two-sided rule (success if
p_s > lambda + delta, failure if p_f > 1 - lambda + delta) and p_s + p_f + p_inv = 1:

- p_inv >= t gives p_s <= 1 - t, so success is impossible exactly when t >= 1 - lambda - delta;
- it gives p_f <= 1 - t, so failure is impossible exactly when t >= lambda - delta.

So t = max(lambda, 1 - lambda) - delta is the smallest threshold at which every
invalid flag falls on a state the rule would otherwise call outcome-uncertain: the
check only separates model error from outcome uncertainty and never overrides a
confident success or failure. The old t = lambda - delta has that property only
for lambda >= 0.5.

Probabilities are MC fractions (n / K) and thresholds sums of grid values, so exact
ties are common and float rounding (0.24 + 0.21 = 0.44999999999999996) would decide
them. Both sides are rounded to TIE_DECIMALS before comparing, so ties compare as
ties.
"""
from __future__ import annotations

import numpy as np

TIE_DECIMALS = 9


def tie(x):
    """Round probabilities or thresholds so exact ties compare equal."""
    return np.round(np.asarray(x, dtype=float), TIE_DECIMALS)


def invalid_threshold(lambda_star: float, delta: float) -> float:
    return float(tie(max(lambda_star, 1.0 - lambda_star) - delta))
