"""Learned length model for the yield weight.

yield_aware routes E[L] through the ensemble's p_bar, which is the miscalibrated
quantity: at epoch 0 it expected 82k pairs and bought 11.4k. Since 98.7% of
Var(L) on this pool is explained by the outcome label, predicting length from
the start state is the same problem as predicting the outcome -- and a kNN on
acquired truth does that far better than p_bar early on.

These tests pin: the regressor recovers length from local truth, it is used in
place of the p_bar mixture once support exists, it falls back cleanly before
then, and alpha=0 is still exactly the unweighted arm.
"""

import numpy as np
import pytest

from adaptive_roa.adaptive_v2.strategy.yield_knn import (
    knn_length, YieldKNNAcquisitionStrategy,
)


def _cfg(**kw):
    base = dict(score="epistemic_var", d2_ratio=1.0, n_candidates=100, alpha=1.0,
                verbose=False, knn_k=3, min_labeled=4, prior_len_success=150.0,
                prior_len_failure=1000.0, min_per_class=2)
    base.update(kw)
    return type("C", (), {**base, "get": lambda self, k, d=None: base.get(k, d)})()


# ------------------------------------------------------------- the regressor
def test_knn_length_recovers_the_local_length():
    """Query sitting on a labelled point returns that point's length."""
    X = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])
    L = np.array([100.0, 500.0, 1000.0])
    out = knn_length(X, L, np.array([[0.0, 0.0], [20.0, 0.0]]), k=1)
    np.testing.assert_allclose(out, [100.0, 1000.0])


def test_knn_length_interpolates_between_neighbours():
    X = np.array([[0.0, 0.0], [10.0, 0.0]])
    L = np.array([100.0, 1000.0])
    out = knn_length(X, L, np.array([[5.0, 0.0]]), k=2)
    assert 400.0 < out[0] < 700.0, "equidistant query -> roughly the mean"


def test_knn_length_k_larger_than_the_labelled_set_is_clamped():
    X = np.array([[0.0, 0.0], [1.0, 0.0]])
    L = np.array([100.0, 200.0])
    out = knn_length(X, L, np.array([[0.5, 0.0]]), k=50)
    assert np.isfinite(out).all()


# ------------------------------------------------------------------ the arm
class _Src:
    def __init__(self, starts, labels, lengths):
        self.starts, self.labels, self.lengths = starts, labels, lengths

    def get_start_state(self, i):
        return self.starts[i]

    def get_label(self, i):
        return self.labels[i]

    def get_trajectory_length(self, i):
        return self.lengths[i]


class _Pool:
    """Candidates are a fixed state array; `system=None` -> identity geometry."""

    def __init__(self, src, idx, cand):
        self.dataset_builder = type("B", (), {
            "data_source": src,
            "train_split": type("S", (), {"indices": idx})(),
        })()
        self._cand = cand
        self.system = None

    def sample_candidates_without_marking(self, k, exclude=None):
        n = min(k, len(self._cand))
        return self._cand[:n], list(range(n))


class _Backend:
    member_sample_size = 100

    def __init__(self, p):
        self._p = np.asarray(p, dtype=np.float64)

    def estimate_members(self, states, verbose=False):
        return self._p


def _pool(n_lab=8):
    # left half short successes, right half long failures
    starts = {i: np.array([0.0, 0.0]) if i < n_lab // 2 else np.array([5.0, 0.0])
              for i in range(n_lab)}
    labels = {i: (1 if i < n_lab // 2 else -1) for i in range(n_lab)}
    lengths = {i: (130 if i < n_lab // 2 else 1001) for i in range(n_lab)}
    cand = np.array([[0.0, 0.0], [5.0, 0.0]])          # one near each cluster
    return _Pool(_Src(starts, labels, lengths), list(range(n_lab)), cand)


def test_learned_length_drives_selection_toward_the_long_region():
    """Scores tie, so only the learned length can break it -- and it must pick
    the candidate sitting in the long-trajectory cluster."""
    p = np.array([[0.5, 0.5], [0.9, 0.1]])             # unequal p_bar, equal-ish score
    s = YieldKNNAcquisitionStrategy(_cfg())
    r = s.select(_pool(), _Backend(p), None, None, target_count=1)
    assert r.d2_indices == [1]
    assert r.diagnostics["knn_active"] is True


def test_labels_are_normalised_from_the_minus_one_convention():
    """get_label returns -1 for failure; a `y == 0` test would match nothing and
    silently zero the failure-length fallback (this bug disabled class_quota)."""
    s = YieldKNNAcquisitionStrategy(_cfg())
    _, y, L = s._acquired(_pool())
    assert set(np.unique(y)) == {0.0, 1.0}
    assert L[y == 0].mean() == 1001.0 and L[y == 1].mean() == 130.0


def test_falls_back_to_the_p_bar_mixture_below_min_labeled():
    """With too few acquired trajectories the kNN has no support; the arm must
    degrade to yield_aware rather than to a degenerate weight."""
    s = YieldKNNAcquisitionStrategy(_cfg(min_labeled=999))
    r = s.select(_pool(), _Backend(np.array([[0.5, 0.5], [0.9, 0.1]])),
                 None, None, target_count=1)
    assert r.diagnostics["knn_active"] is False
    assert r.diagnostics["len_hat_mean_selected"] == r.diagnostics["len_pbar_mean_selected"]


def test_diagnostics_expose_the_learned_vs_pbar_comparison():
    s = YieldKNNAcquisitionStrategy(_cfg())
    d = s.select(_pool(), _Backend(np.array([[0.5, 0.5], [0.9, 0.1]])),
                 None, None, target_count=1).diagnostics
    for key in ("len_hat_mean_selected", "len_pbar_mean_selected", "len_hat_over_pbar"):
        assert d[key] is not None
    assert d["selection_rule"].startswith("knn_length_weighted")


def test_alpha_zero_reduces_to_the_unweighted_score():
    """The weight must vanish exactly, so this stays a single-knob comparison."""
    s = YieldKNNAcquisitionStrategy(_cfg(alpha=0.0))
    p = np.array([[0.9, 0.1], [0.7, 0.3]])
    r = s.select(_pool(), _Backend(p), None, None, target_count=2)
    base = r.diagnostics["base_score_mean_selected"]
    assert np.isclose(r.diagnostics["score_mean_selected"], base)


def test_unknown_score_mode_is_rejected_at_construction():
    with pytest.raises(ValueError):
        YieldKNNAcquisitionStrategy(_cfg(score="not_a_mode"))


def test_backend_without_estimate_members_is_rejected_loudly():
    s = YieldKNNAcquisitionStrategy(_cfg())
    with pytest.raises(RuntimeError, match="estimate_members"):
        s.select(_pool(), object(), None, None, target_count=1)
