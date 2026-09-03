"""Stratified acquisition: keep the scored batch on the pool's marginal.

Diagnosed failure this exists to fix (docs/experiments/ensemble_epistemic/CARTPOLE.md):
on stochastic cartpole the flow matcher under-predicts success in the minority
basin, so its "uncertain" band (p_hat in [0.25, 0.75]) sits on states whose TRUE
p_success averages 0.79-0.85. Every uncertainty score therefore buys the success
basin rather than the separatrix -- measured mean true p of acquired points is
0.62-0.82 against a pool base rate of 0.240, with only 1-9% genuinely ambiguous.
That triples basin training mass and starves the failure bulk 3-4x, and the bulk
is ~70% of the eval grid, so resolution collapses.

Stratifying by the ensemble mean p_bar and giving each stratum a budget
proportional to its share of candidates makes the acquired batch inherit the
pool's marginal BY CONSTRUCTION, whatever the score does inside a stratum. The
score still picks which points within a region; it can no longer pick the region.

These tests pin the property that matters (marginal preserved), the property
that must survive (top-score within stratum), and the degenerate inputs that a
50k-candidate live run will actually hit.
"""

import numpy as np
import pytest

from adaptive_roa.adaptive_v2.strategy.stratified import stratified_top_score


def test_selection_preserves_the_candidate_marginal():
    """The whole point: strata shares in the batch match shares in the pool."""
    rng = np.random.default_rng(0)
    # skewed marginal, like the real pool: mostly low p_bar
    p_bar = np.clip(rng.beta(2, 6, size=5000), 0, 1)
    # adversarial score: monotonically increasing in p_bar, i.e. exactly the
    # pathology -- an unstratified top-N would take only the high-p_bar tail.
    score = p_bar + rng.normal(0, 0.01, size=p_bar.size)

    pos = stratified_top_score(score, p_bar, target_count=500, n_strata=10)
    assert len(pos) == 500
    edges = np.linspace(0.0, 1.0, 11)
    pool_share = np.histogram(p_bar, bins=edges)[0] / p_bar.size
    sel_share = np.histogram(p_bar[pos], bins=edges)[0] / len(pos)
    # every stratum within 2 percentage points of its pool share
    assert np.max(np.abs(pool_share - sel_share)) < 0.02


def test_unstratified_top_n_fails_that_same_test():
    """Guard the premise: plain greedy really does chase the tail here."""
    rng = np.random.default_rng(0)
    p_bar = np.clip(rng.beta(2, 6, size=5000), 0, 1)
    score = p_bar + rng.normal(0, 0.01, size=p_bar.size)
    pos = np.argsort(-score)[:500]
    edges = np.linspace(0.0, 1.0, 11)
    pool_share = np.histogram(p_bar, bins=edges)[0] / p_bar.size
    sel_share = np.histogram(p_bar[pos], bins=edges)[0] / len(pos)
    assert np.max(np.abs(pool_share - sel_share)) > 0.2, "premise broken"


def test_picks_the_top_scorers_inside_each_stratum():
    """Stratification constrains WHERE, not WHICH -- the score still ranks."""
    p_bar = np.repeat([0.05, 0.55], 100)          # two strata, 100 each
    score = np.concatenate([np.arange(100.0), np.arange(100.0)])
    pos = stratified_top_score(score, p_bar, target_count=20, n_strata=10)
    lo = [p for p in pos if p < 100]
    hi = [p for p in pos if p >= 100]
    assert len(lo) == 10 and len(hi) == 10          # equal shares, equal budgets
    assert set(lo) == set(range(90, 100))           # top 10 of stratum 0
    assert set(hi) == set(range(190, 200))          # top 10 of stratum 1


def test_budget_shortfall_is_redistributed_not_dropped():
    """A thin stratum must not silently shrink the batch.

    The live pool is heavily skewed, so high-p_bar strata routinely hold fewer
    candidates than their rounded budget. Returning fewer than target_count would
    quietly change the per-epoch acquisition budget and break matched-budget
    comparisons against the control.
    """
    p_bar = np.concatenate([np.full(980, 0.05), np.full(20, 0.95)])
    score = np.random.default_rng(1).random(1000)
    pos = stratified_top_score(score, p_bar, target_count=200, n_strata=10)
    assert len(pos) == 200
    assert len(set(pos)) == 200, "no duplicates"


def test_target_exceeding_candidates_returns_all():
    p_bar = np.linspace(0, 1, 50)
    score = np.random.default_rng(2).random(50)
    pos = stratified_top_score(score, p_bar, target_count=999, n_strata=10)
    assert sorted(pos) == list(range(50))


def test_nan_scores_are_ranked_last_not_crashed():
    """score_by_mode can emit NaN; decomposition.py already tolerates it."""
    p_bar = np.full(100, 0.3)
    score = np.arange(100.0)
    score[:10] = np.nan
    pos = stratified_top_score(score, p_bar, target_count=50, n_strata=10)
    assert len(pos) == 50
    assert not set(pos) & set(range(10)), "NaN-scored points must not be preferred"


def test_empty_input():
    pos = stratified_top_score(np.array([]), np.array([]), target_count=10, n_strata=10)
    assert len(pos) == 0


def test_single_stratum_reduces_to_plain_greedy():
    """With n_strata=1 the strategy must be exactly the arm it is compared against."""
    rng = np.random.default_rng(3)
    p_bar = rng.random(500)
    score = rng.random(500)
    pos = stratified_top_score(score, p_bar, target_count=50, n_strata=1)
    assert sorted(pos) == sorted(np.argsort(-score)[:50].tolist())
