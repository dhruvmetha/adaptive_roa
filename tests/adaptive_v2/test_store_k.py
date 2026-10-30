"""Hypergeometric K subsampling (H04) matches drawing K' of the K labels directly."""
import numpy as np

from adaptive_roa.adaptive_v2.eval.store_k import subsample_counts


def test_subsample_matches_direct_draws_in_distribution():
    rng = np.random.default_rng(0)
    labels = np.array([1] * 37 + [-1] * 50 + [0] * 13)          # K = 100 at one state
    n = 20000
    s, f, i = subsample_counts(np.full(n, 37), np.full(n, 50), np.full(n, 13), 20, rng)
    direct = np.array([np.sum(rng.choice(labels, 20, replace=False) == 1) for _ in range(n)])
    assert abs(s.mean() - direct.mean()) < 0.05
    assert abs(s.var() - direct.var()) < 0.15
    assert np.all(s + f + i == 20) and np.all(i >= 0) and np.all(i <= 13)


def test_no_invalid_mass_stays_two_class():
    rng = np.random.default_rng(1)
    s, f, i = subsample_counts(np.array([60, 0, 100]), np.array([40, 100, 0]), np.zeros(3, int), 10, rng)
    assert np.all(i == 0) and np.all(s + f == 10)
    assert s[1] == 0 and s[2] == 10
