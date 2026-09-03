"""Yield-aware acquisition: rank uncertainty per unit of budget, not per point.

The failure this targets (pendulum i100 sweep, measured 2026-08-15): the budget
is trajectories but the model trains on pairs, and on pendulum-high success
rollouts are 136 steps while failures run to the 1001-step cap. Every scored arm
bought the model's uncertain band -- which is the true-success region there --
so the arms accrued pairs at ~0.4x the control's rate (358k vs 652k) and lost at
matched trajectory count while WINNING at matched pair count.

These tests pin: the weight points the right way, alpha=0 is exactly the
unweighted arm, negative debiased scores are not promoted by a bigger weight,
and the length model degrades safely when a class has not been acquired yet.
"""

import numpy as np
import pytest

from adaptive_roa.adaptive_v2.strategy.yield_aware import (
    expected_length, yield_weighted_score, YieldAwareDecompositionAcquisitionStrategy,
)


def test_expected_length_interpolates_between_the_two_outcome_regimes():
    """p_bar=1 -> success length, p_bar=0 -> failure length."""
    np.testing.assert_allclose(expected_length(np.array([1.0]), 136.0, 1001.0), [136.0])
    np.testing.assert_allclose(expected_length(np.array([0.0]), 136.0, 1001.0), [1001.0])
    np.testing.assert_allclose(expected_length(np.array([0.5]), 136.0, 1001.0), [568.5])


def test_weighting_prefers_the_higher_yield_candidate_when_scores_tie():
    """The whole point: equal information, more pairs -> ranked first.

    On pendulum the low-p_bar candidate is the long (failure) rollout, so it
    buys ~7x the training pairs for the same slot in a 100-trajectory budget.
    """
    score = np.array([1.0, 1.0])
    p_bar = np.array([0.9, 0.1])            # short-success vs long-failure
    w = yield_weighted_score(score, p_bar, 136.0, 1001.0, alpha=1.0)
    assert w[1] > w[0]


def test_alpha_zero_reduces_exactly_to_the_unweighted_score():
    """Single-knob comparison against the plain epi_var arm must be exact."""
    rng = np.random.default_rng(0)
    score = rng.normal(size=500)
    p_bar = rng.random(500)
    out = yield_weighted_score(score, p_bar, 136.0, 1001.0, alpha=0.0)
    np.testing.assert_allclose(out, score)


def test_negative_debiased_scores_are_not_promoted_by_a_larger_weight():
    """epistemic_var is MC-debiased and can go negative for agreeing members.

    A naive product would rank a strongly-negative score with a big weight
    ABOVE a mildly-negative one, i.e. prefer the least informative candidate.
    """
    score = np.array([-0.5, -0.01])
    p_bar = np.array([0.0, 0.0])            # identical, large weight for both
    w = yield_weighted_score(score, p_bar, 136.0, 1001.0, alpha=1.0)
    assert w[1] > w[0], "less-negative score must still rank higher"
    assert np.all(w < 0), "weighting must not flip a negative score positive"


def test_nan_scores_survive_as_nan_and_rank_last_downstream():
    score = np.array([np.nan, 1.0])
    out = yield_weighted_score(score, np.array([0.5, 0.5]), 136.0, 1001.0)
    assert np.isnan(out[0]) and np.isfinite(out[1])
    key = np.where(np.isfinite(out), out, -np.inf)
    assert np.argsort(-key)[0] == 1


# ------------------------------------------------------------- length model
class _Src:
    def __init__(self, lengths, labels):
        self.lengths, self.labels = lengths, labels

    def get_trajectory_length(self, i):
        return self.lengths[i]

    def get_label(self, i):
        return self.labels[i]


class _Builder:
    def __init__(self, src, idx):
        self.data_source = src
        self.train_split = type("S", (), {"indices": idx})()


class _Pool:
    def __init__(self, builder):
        self.dataset_builder = builder


def _cfg(**kw):
    base = dict(score="epistemic_var", d2_ratio=1.0, n_candidates=100, alpha=1.0,
                verbose=False, prior_len_success=150.0, prior_len_failure=1000.0,
                min_per_class=5)
    base.update(kw)
    return type("C", (), {**base, "get": lambda self, k, d=None: base.get(k, d)})()


def test_length_stats_measured_from_acquired_trajectories():
    lengths = {i: (130 if i < 10 else 1001) for i in range(20)}
    labels = {i: (1 if i < 10 else 0) for i in range(20)}
    s = YieldAwareDecompositionAcquisitionStrategy(_cfg())
    ls, lf, ns, nf = s._length_stats(_Pool(_Builder(_Src(lengths, labels), list(range(20)))))
    assert (ls, lf, ns, nf) == (130.0, 1001.0, 10, 10)


def test_length_stats_fall_back_to_prior_for_an_unseen_class():
    """Epoch 0 can be lopsided; a class with no examples must not zero the weight."""
    lengths = {i: 1001 for i in range(20)}
    labels = {i: 0 for i in range(20)}          # no successes acquired yet
    s = YieldAwareDecompositionAcquisitionStrategy(_cfg())
    ls, lf, ns, nf = s._length_stats(_Pool(_Builder(_Src(lengths, labels), list(range(20)))))
    assert ls == 150.0 and lf == 1001.0 and ns == 0 and nf == 20


def test_length_stats_survive_a_pool_without_a_dataset_builder():
    s = YieldAwareDecompositionAcquisitionStrategy(_cfg())
    ls, lf, ns, nf = s._length_stats(object())
    assert (ls, lf, ns, nf) == (150.0, 1000.0, 0, 0)


def test_unknown_score_mode_is_rejected_at_construction():
    with pytest.raises(ValueError):
        YieldAwareDecompositionAcquisitionStrategy(_cfg(score="not_a_mode"))
