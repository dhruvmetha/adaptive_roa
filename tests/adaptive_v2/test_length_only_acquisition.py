"""Length-only acquisition: the control that isolates the yield weight.

``yield_aware`` = score x E[L] beats uniform at matched trajectory budget. This
arm keeps E[L] and drops the score, so if it reproduces the win the epistemic
component contributes nothing. These tests pin the properties that make it a
valid null: selection ignores the score entirely, ranking is monotone in E[L]
(hence in p_bar), and the score is still measured for diagnostics.
"""

import numpy as np
import pytest

from adaptive_roa.adaptive_v2.strategy.length_only import LengthOnlyAcquisitionStrategy


def _cfg(**kw):
    base = dict(score="epistemic_var", d2_ratio=1.0, n_candidates=100, verbose=False,
                prior_len_success=150.0, prior_len_failure=1000.0, min_per_class=5)
    base.update(kw)
    return type("C", (), {**base, "get": lambda self, k, d=None: base.get(k, d)})()


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
    """Pool whose candidates are just their indices; states are unused here."""

    def __init__(self, builder, n):
        self.dataset_builder = builder
        self._n = n

    def sample_candidates_without_marking(self, k, exclude=None):
        n = min(k, self._n)
        return np.zeros((n, 2)), list(range(n))


class _Backend:
    """Returns fixed per-member probabilities so p_bar is controlled exactly."""

    member_sample_size = 100

    def __init__(self, p_members):
        self._p = np.asarray(p_members, dtype=np.float64)

    def estimate_members(self, states, verbose=False):
        return self._p


def _pool(n=4):
    lengths = {i: (130 if i < 10 else 1001) for i in range(20)}
    labels = {i: (1 if i < 10 else 0) for i in range(20)}
    return _Pool(_Builder(_Src(lengths, labels), list(range(20))), n)


def test_selection_prefers_low_p_bar_because_failures_are_the_long_rollouts():
    """With L_failure >> L_success, E[L] is decreasing in p_bar."""
    # 2 members, 4 candidates with p_bar = [0.9, 0.1, 0.5, 0.99]
    p = np.array([[0.9, 0.1, 0.5, 0.99], [0.9, 0.1, 0.5, 0.99]])
    s = LengthOnlyAcquisitionStrategy(_cfg())
    r = s.select(_pool(4), _Backend(p), None, None, target_count=2)
    assert r.d2_indices == [1, 2], "must buy the two lowest-p_bar (longest) candidates"


def test_selection_ignores_the_uncertainty_score_entirely():
    """Two member sets with identical p_bar but wildly different disagreement
    must produce the SAME selection -- that is what makes this a clean null."""
    agree = np.array([[0.9, 0.1], [0.9, 0.1]])            # zero disagreement
    disagree = np.array([[1.0, 0.0], [0.8, 0.2]])          # same p_bar, high disagreement
    np.testing.assert_allclose(agree.mean(axis=0), disagree.mean(axis=0))

    s = LengthOnlyAcquisitionStrategy(_cfg())
    a = s.select(_pool(2), _Backend(agree), None, None, target_count=1)
    b = s.select(_pool(2), _Backend(disagree), None, None, target_count=1)
    assert a.d2_indices == b.d2_indices == [1]


def test_score_is_still_reported_so_the_confound_can_be_measured():
    """Selection ignores the score, but diagnostics must record it -- the ratio
    tells us whether length-ranking incidentally buys uncertain points."""
    p = np.array([[1.0, 0.0, 0.6], [0.8, 0.2, 0.4]])
    s = LengthOnlyAcquisitionStrategy(_cfg())
    r = s.select(_pool(3), _Backend(p), None, None, target_count=1)
    d = r.diagnostics
    assert d["selection_rule"] == "expected_length_only"
    assert d["score_mean_selected"] is not None
    assert d["score_selected_over_candidates"] is not None
    assert d["expected_pairs_selected"] > 0


def test_length_stats_measured_from_acquired_trajectories():
    s = LengthOnlyAcquisitionStrategy(_cfg())
    ls, lf, ns, nf = s._length_stats(_pool(4))
    assert (ls, lf, ns, nf) == (130.0, 1001.0, 10, 10)


def test_length_stats_fall_back_to_prior_for_an_unseen_class():
    lengths = {i: 1001 for i in range(20)}
    labels = {i: 0 for i in range(20)}
    s = LengthOnlyAcquisitionStrategy(_cfg())
    ls, lf, ns, nf = s._length_stats(_Pool(_Builder(_Src(lengths, labels), list(range(20))), 4))
    assert ls == 150.0 and lf == 1001.0 and ns == 0 and nf == 20


def test_length_stats_survive_a_pool_without_a_dataset_builder():
    s = LengthOnlyAcquisitionStrategy(_cfg())
    assert s._length_stats(object()) == (150.0, 1000.0, 0, 0)


def test_zero_target_count_skips_without_touching_the_backend():
    s = LengthOnlyAcquisitionStrategy(_cfg())
    r = s.select(_pool(4), None, None, None, target_count=0)
    assert r.d2_indices == [] and r.diagnostics["skipped_reason"] == "target_count_zero"


def test_unknown_score_mode_is_rejected_at_construction():
    with pytest.raises(ValueError):
        LengthOnlyAcquisitionStrategy(_cfg(score="not_a_mode"))


def test_backend_without_estimate_members_is_rejected_loudly():
    s = LengthOnlyAcquisitionStrategy(_cfg())
    with pytest.raises(RuntimeError, match="estimate_members"):
        s.select(_pool(4), object(), None, None, target_count=1)
