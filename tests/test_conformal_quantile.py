"""q_hat is the split-conformal order statistic, not an interpolated quantile.

The calibrator used np.quantile at level min((1-alpha)(n+1)/n, 1) with linear
interpolation. Split conformal needs the k-th smallest score with
k = ceil((n+1)(1-alpha)), and an infinite q_hat when k > n (REVIEW_TODO R03).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from adaptive_roa.conformal import ConformalConfig
from adaptive_roa.conformal.calibrator import Calibrator, conformal_quantile


def test_order_statistic_at_alpha_0_1():
    scores = np.arange(1.0, 11.0)          # n = 10, k = ceil(9.9) = 10
    assert conformal_quantile(scores, 0.1) == 10.0


def test_order_statistic_at_alpha_0_2():
    scores = np.arange(1.0, 11.0)[::-1]    # unsorted input; k = ceil(8.8) = 9
    assert conformal_quantile(scores, 0.2) == 9.0


def test_rank_above_n_gives_infinite_q_hat():
    scores = np.arange(5.0)                # n = 5, k = ceil(5.4) = 6 > 5
    assert conformal_quantile(scores, 0.1) == math.inf


def test_empty_calibration_set_gives_infinite_q_hat():
    assert conformal_quantile(np.array([]), 0.1) == math.inf


def test_never_interpolates_between_scores():
    rng = np.random.default_rng(0)
    scores = rng.random(37)
    q = conformal_quantile(scores, 0.1)
    assert q in set(scores.tolist())


@pytest.mark.parametrize("alpha", [0.1, 0.05])
def test_marginal_coverage_holds_on_exchangeable_scores(alpha):
    rng = np.random.default_rng(1)
    n, trials = 40, 4000
    hits = 0
    for _ in range(trials):
        s = rng.random(n + 1)
        hits += s[-1] <= conformal_quantile(s[:-1], alpha)
    # Exact coverage is ceil((n+1)(1-alpha))/(n+1) >= 1 - alpha; allow MC noise.
    assert hits / trials >= 1 - alpha - 0.015


def test_calibrator_uses_the_order_statistic():
    cal = Calibrator(ConformalConfig(delta=0.05, w=0.9, alpha=0.1, attractor_radius=0.2,
                                     decision_rule="one_sided"))
    rng = np.random.default_rng(2)
    p = rng.random(25)
    y = np.where(rng.random(25) < 0.5, 1, -1)
    scores = cal.non_conformity_scores_batch(p, y, 0.5, 0.05, None)
    assert cal.calibrate(p, y, 0.5) == conformal_quantile(scores, 0.1)
