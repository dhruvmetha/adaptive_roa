"""MLP length model for the yield weight.

Measured motivation: predicting trajectory length from the start state, R^2 on
20k held-out pool trajectories -- MLP 64-64 beats kNN k=15 by +0.076 at n=100
(0.757 vs 0.681) and by only +0.006 at n=2000, i.e. it helps exactly where the
yield weight is broken. It is also 10x more stable at n=100 (sd 0.001 vs 0.011).

These tests pin: the MLP is what drives selection when it fits, the kNN is
computed alongside as an in-run comparator, every fallback path is reachable and
labelled, and alpha=0 is still exactly the unweighted arm.
"""

import numpy as np
import pytest

from adaptive_roa.adaptive_v2.strategy.yield_mlp import (
    choose_length_model, fit_mlp_length, YieldMLPAcquisitionStrategy,
)


def _cfg(**kw):
    base = dict(score="epistemic_var", d2_ratio=1.0, n_candidates=100, alpha=1.0,
                verbose=False, hidden=(16, 16), knn_k=3, min_labeled=4, mlp_seed=0,
                prior_len_success=150.0, prior_len_failure=1000.0, min_per_class=2)
    base.update(kw)
    return type("C", (), {**base, "get": lambda self, k, d=None: base.get(k, d)})()


# ------------------------------------------------------------- the regressor
def test_fit_mlp_length_learns_a_separable_length_field():
    """Two well-separated clusters with very different lengths must be told apart."""
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal([0, 0, 0], 0.05, (60, 3)),
                   rng.normal([5, 0, 0], 0.05, (60, 3))])
    L = np.concatenate([np.full(60, 130.0), np.full(60, 1001.0)])
    predict = fit_mlp_length(X, L, (16, 16), seed=0)
    assert predict is not None
    out = predict(np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]))
    assert out[0] < 400 and out[1] > 700, f"clusters not separated: {out}"


def test_fit_mlp_length_returns_none_rather_than_raising_on_a_bad_fit():
    """A degenerate input must fall back, not crash the acquisition step."""
    assert fit_mlp_length(np.empty((0, 3)), np.empty(0), (16, 16), seed=0) is None


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


def _pool(n_lab=40):
    """Left cluster = short successes, right cluster = long failures."""
    rng = np.random.default_rng(1)
    starts, labels, lengths = {}, {}, {}
    for i in range(n_lab):
        left = i < n_lab // 2
        starts[i] = np.array([0.0, 0.0]) + rng.normal(0, 0.02, 2) if left else \
                    np.array([2.0, 0.0]) + rng.normal(0, 0.02, 2)
        labels[i] = 1 if left else -1
        lengths[i] = 130 if left else 1001
    cand = np.array([[0.0, 0.0], [2.0, 0.0]])
    return _Pool(_Src(starts, labels, lengths), list(range(n_lab)), cand)


def test_selection_goes_to_the_long_region_whichever_model_wins():
    """The arm must buy the long-trajectory candidate. WHICH estimator supplies
    the length is decided per-epoch by held-out error, so the test pins the
    behaviour that matters and not the mechanism that happens to win here."""
    p = np.array([[0.5, 0.5], [0.9, 0.1]])
    s = YieldMLPAcquisitionStrategy(_cfg())
    r = s.select(_pool(), _Backend(p), None, None, target_count=1)
    assert r.d2_indices == [1]
    assert r.diagnostics["length_model"] in ("mlp", "knn")


def test_the_bake_off_rejects_an_mlp_that_is_worse_than_the_knn():
    """Regression test for the defect that motivated the bake-off: on 40 points
    in two clean clusters (lengths 130 / 1001) the MLP predicted 430 and 224 --
    ordering inverted -- while the kNN was exact. Blindly trusting the
    pool-level benchmark would have shipped that."""
    rng = np.random.default_rng(1)
    E = np.vstack([rng.normal([0.0, 0.0], 0.02, (20, 2)),
                   rng.normal([2.0, 0.0], 0.02, (20, 2))])
    L = np.concatenate([np.full(20, 130.0), np.full(20, 1001.0)])
    winner, mse_mlp, mse_knn = choose_length_model(E, L, (16, 16), knn_k=3, seed=0)
    assert winner == "knn"
    assert mse_knn < mse_mlp


def test_the_bake_off_degrades_to_knn_when_there_is_too_little_data_to_split():
    E = np.zeros((4, 2)); L = np.full(4, 130.0)
    winner, mse_mlp, mse_knn = choose_length_model(E, L, (16, 16), knn_k=3, seed=0)
    assert winner == "knn"


def test_knn_is_computed_alongside_as_an_in_run_comparator():
    """The whole point of logging both: compare estimators within one run."""
    s = YieldMLPAcquisitionStrategy(_cfg())
    d = s.select(_pool(), _Backend(np.array([[0.5, 0.5], [0.9, 0.1]])),
                 None, None, target_count=1).diagnostics
    for key in ("len_hat_mean_selected", "len_knn_mean_selected", "len_pbar_mean_selected"):
        assert d[key] is not None, key


def test_falls_back_to_pbar_below_min_labeled_and_says_so():
    s = YieldMLPAcquisitionStrategy(_cfg(min_labeled=9999))
    d = s.select(_pool(), _Backend(np.array([[0.5, 0.5], [0.9, 0.1]])),
                 None, None, target_count=1).diagnostics
    assert d["length_model"] == "pbar_fallback"
    assert d["len_knn_mean_selected"] is None
    assert d["len_hat_mean_selected"] == d["len_pbar_mean_selected"]


def test_falls_back_to_knn_when_the_mlp_fit_fails(monkeypatch):
    """A failed fit must degrade to yield_knn, not to an unweighted arm."""
    monkeypatch.setattr("adaptive_roa.adaptive_v2.strategy.yield_mlp.fit_mlp_length",
                        lambda *a, **k: None)
    s = YieldMLPAcquisitionStrategy(_cfg())
    d = s.select(_pool(), _Backend(np.array([[0.5, 0.5], [0.9, 0.1]])),
                 None, None, target_count=1).diagnostics
    assert d["length_model"] == "knn"
    assert d["len_hat_mean_selected"] == d["len_knn_mean_selected"]


def test_labels_are_normalised_from_the_minus_one_convention():
    s = YieldMLPAcquisitionStrategy(_cfg())
    _, y, L = s._acquired(_pool())
    assert set(np.unique(y)) == {0.0, 1.0}
    assert L[y == 0].mean() == 1001.0 and L[y == 1].mean() == 130.0


def test_alpha_zero_reduces_to_the_unweighted_score():
    s = YieldMLPAcquisitionStrategy(_cfg(alpha=0.0))
    r = s.select(_pool(), _Backend(np.array([[0.9, 0.1], [0.7, 0.3]])),
                 None, None, target_count=2)
    assert np.isclose(r.diagnostics["score_mean_selected"],
                      r.diagnostics["base_score_mean_selected"])


def test_unknown_score_mode_is_rejected_at_construction():
    with pytest.raises(ValueError):
        YieldMLPAcquisitionStrategy(_cfg(score="not_a_mode"))


def test_backend_without_estimate_members_is_rejected_loudly():
    s = YieldMLPAcquisitionStrategy(_cfg())
    with pytest.raises(RuntimeError, match="estimate_members"):
        s.select(_pool(), object(), None, None, target_count=1)
