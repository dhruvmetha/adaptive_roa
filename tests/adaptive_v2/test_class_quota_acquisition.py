"""Class-quota acquisition: composition bought from measured outcomes, not p_bar.

The failure this replaces (measured 2026-08-16): the yield-aware arm priced
yield through the ensemble's own p_bar and epoch 0 falsified it -- expected 82k
pairs, bought 11.4k, fewer than plain epi_var -- because p_bar calls
true-success states failures and the gap never closes (plateau ~0.3 over the
completed 20-epoch epi_var run).

These tests pin the properties the design argument rests on: the gate reads
ONLY acquired truth (an adversarially wrong p_bar cannot steer composition),
the quota is respected with backfill, the gate-off fallback is the plain arm's
selection, and the circular embedding does not tear theta at the seam.
"""

import numpy as np
import pytest

from adaptive_roa.adaptive_v2.strategy.class_quota import (
    ClassQuotaDecompositionAcquisitionStrategy, embed_states, knn_success_prob,
)


# ------------------------------------------------------------------ fakes
class _System:
    state_dim = 2

    def get_normalization_scales(self):
        return np.array([np.pi, 2.0])

    def get_circular_indices(self):
        return [0]


class _Backend:
    """5 members; p spread encodes disagreement so epi_var is non-degenerate."""
    member_sample_size = 20

    def __init__(self, p_bar, spread=0.1):
        self.system = _System()
        self.p_bar = np.asarray(p_bar, dtype=np.float64)
        self.spread = spread

    def estimate_members(self, states, verbose=False):
        offs = np.linspace(-self.spread, self.spread, 5)[:, None]
        return np.clip(self.p_bar[None, :] + offs, 0.0, 1.0)


class _Src:
    def __init__(self, starts, labels, lengths):
        self.starts, self.labels_, self.lengths = starts, labels, lengths

    def get_start_state(self, i):
        return self.starts[i]

    def get_label(self, i):
        return self.labels_[i]

    def get_trajectory_length(self, i):
        return self.lengths[i]


class _Pool:
    def __init__(self, cand_states, labeled_starts, labeled_labels, labeled_lengths):
        self.cand_states = np.asarray(cand_states, dtype=np.float64)
        src = _Src(labeled_starts, labeled_labels, labeled_lengths)
        split = type("S", (), {"indices": list(range(len(labeled_labels)))})()
        self.dataset_builder = type("B", (), {"data_source": src, "train_split": split})()

    def sample_candidates_without_marking(self, n, exclude=None):
        n = min(n, len(self.cand_states))
        return self.cand_states[:n], list(range(n))


def _cfg(**kw):
    base = dict(score="epistemic_var", d2_ratio=1.0, n_candidates=10_000,
                selection_rule="greedy", diversity_pool_multiplier=5,
                quota_fail="pool", knn_k=5, min_labeled_per_class=3, verbose=False)
    base.update(kw)
    return type("C", (), {**base, "get": lambda self, k, d=None: base.get(k, d)})()


def _two_region_problem(n_cand=200, n_lab=60, rng=None):
    """Truth: theta_dot > 0 fails, theta_dot < 0 succeeds. Labeled set is uniform."""
    rng = rng or np.random.default_rng(0)
    cand = np.column_stack([rng.uniform(-3, 3, n_cand), rng.uniform(-2, 2, n_cand)])
    lab = np.column_stack([rng.uniform(-3, 3, n_lab), rng.uniform(-2, 2, n_lab)])
    lab_y = {i: (0 if lab[i, 1] > 0 else 1) for i in range(n_lab)}
    lab_L = {i: (1001 if lab_y[i] == 0 else 136) for i in range(n_lab)}
    lab_s = {i: lab[i] for i in range(n_lab)}
    return cand, lab_s, lab_y, lab_L


# ------------------------------------------------------------------ gate
def test_gate_ignores_adversarial_p_bar_entirely():
    """The falsification, inverted: p_bar says the FAILURE region is uncertain-
    success and the success region is certain-failure. The quota must still buy
    predicted-failures from the true failure region, because the gate reads
    acquired truth, not members."""
    cand, lab_s, lab_y, lab_L = _two_region_problem()
    # adversarial p_bar: high where truth fails, low where truth succeeds
    p_bar = np.where(cand[:, 1] > 0, 0.9, 0.05)
    strat = ClassQuotaDecompositionAcquisitionStrategy(_cfg(quota_fail=0.6))
    res = strat.select(_Pool(cand, lab_s, lab_y, lab_L), _Backend(p_bar), None, None, 50)
    sel = np.asarray(res.d2_indices, dtype=int)
    true_fail = cand[sel, 1] > 0
    assert res.diagnostics["gate_on"] is True
    # 30 of 50 slots are the failure quota; kNN on 60 uniform labels resolves
    # this separable problem, so at least ~half the batch must be true failures
    assert true_fail.sum() >= 25, f"only {true_fail.sum()} true failures bought"


def test_quota_counts_are_respected_when_both_pots_are_deep():
    cand, lab_s, lab_y, lab_L = _two_region_problem(n_cand=400)
    p_bar = np.full(len(cand), 0.5)
    strat = ClassQuotaDecompositionAcquisitionStrategy(_cfg(quota_fail=0.6))
    res = strat.select(_Pool(cand, lab_s, lab_y, lab_L), _Backend(p_bar), None, None, 100)
    d = res.diagnostics
    assert len(res.d2_indices) == 100
    assert d["n_fail_quota"] == 60
    assert d["n_selected_pred_fail"] == 60


def test_pool_quota_tracks_the_measured_class_marginal():
    cand, lab_s, lab_y, lab_L = _two_region_problem()
    strat = ClassQuotaDecompositionAcquisitionStrategy(_cfg(quota_fail="pool"))
    res = strat.select(_Pool(cand, lab_s, lab_y, lab_L),
                       _Backend(np.full(len(cand), 0.5)), None, None, 50)
    measured_fail_frac = np.mean([lab_y[i] == 0 for i in range(len(lab_y))])
    assert res.diagnostics["quota_fail_target"] == pytest.approx(measured_fail_frac)


def test_backfill_fills_from_success_pot_when_failure_pot_is_shallow():
    """All labeled and all candidates are successes: failure pot is empty, the
    batch must still come back full rather than starved."""
    rng = np.random.default_rng(1)
    cand = np.column_stack([rng.uniform(-3, 3, 80), rng.uniform(-2, -0.5, 80)])
    lab = {i: np.array([rng.uniform(-3, 3), rng.uniform(-2, -0.5)]) for i in range(20)}
    lab_y = {i: 1 for i in range(15)}
    lab_y.update({i: 0 for i in range(15, 20)})       # a few failures far away
    for i in range(15, 20):
        lab[i] = np.array([0.0, 1.9])
    lab_L = {i: (136 if lab_y[i] == 1 else 1001) for i in range(20)}
    strat = ClassQuotaDecompositionAcquisitionStrategy(_cfg(quota_fail=0.9))
    res = strat.select(_Pool(cand, lab, lab_y, lab_L),
                       _Backend(np.full(80, 0.5)), None, None, 40)
    assert len(res.d2_indices) == 40


def test_gate_off_falls_back_to_plain_selection_of_target_count():
    """One class not yet seen min_labeled_per_class times -> plain arm."""
    cand, lab_s, lab_y, lab_L = _two_region_problem()
    lab_y = {i: 1 for i in lab_y}                     # no failures acquired yet
    strat = ClassQuotaDecompositionAcquisitionStrategy(_cfg())
    res = strat.select(_Pool(cand, lab_s, lab_y, lab_L),
                       _Backend(np.full(len(cand), 0.5)), None, None, 30)
    assert res.diagnostics["gate_on"] is False
    assert len(res.d2_indices) == 30


# ------------------------------------------------------------------ kNN + embed
def test_knn_resolves_a_separable_problem_near_certainty():
    rng = np.random.default_rng(2)
    X = rng.uniform(-1, 1, size=(200, 2))
    y = (X[:, 0] > 0).astype(float)
    q = np.array([[0.8, 0.0], [-0.8, 0.0]])
    p = knn_success_prob(X, y, q, k=15)
    assert p[0] > 0.9 and p[1] < 0.1


def test_embedding_does_not_tear_theta_at_the_seam():
    scales = np.array([np.pi, 2.0])
    mask = np.array([True, False])
    a = embed_states(np.array([[np.pi - 1e-6, 0.5]]), scales, mask)
    b = embed_states(np.array([[-np.pi + 1e-6, 0.5]]), scales, mask)
    np.testing.assert_allclose(a, b, atol=1e-5)


def test_expected_pairs_diagnostic_uses_measured_lengths():
    cand, lab_s, lab_y, lab_L = _two_region_problem(n_cand=400)
    strat = ClassQuotaDecompositionAcquisitionStrategy(_cfg(quota_fail=0.6))
    res = strat.select(_Pool(cand, lab_s, lab_y, lab_L),
                       _Backend(np.full(len(cand), 0.5)), None, None, 100)
    d = res.diagnostics
    assert d["len_failure_measured"] == pytest.approx(1001.0)
    assert d["len_success_measured"] == pytest.approx(136.0)
    # 60 predicted failures at ~1000 pairs each dominate: E[pairs] must be
    # in the tens of thousands, i.e. the arm prices itself like the control
    assert d["expected_pairs_selected"] > 45_000


def test_unknown_score_mode_is_rejected_at_construction():
    with pytest.raises(ValueError):
        ClassQuotaDecompositionAcquisitionStrategy(_cfg(score="not_a_mode"))


def test_bad_quota_is_rejected_at_construction():
    with pytest.raises(ValueError):
        ClassQuotaDecompositionAcquisitionStrategy(_cfg(quota_fail=1.5))


# --------------------------------------------------- label-convention regression
def test_acquired_truth_normalises_the_internal_minus_one_failure_label():
    """data_source.get_label returns {-1, +1}, NOT {0, 1}.

    `_load_shuffled_labels` builds labels with `label_mapping = {0: -1, 1: 1}`,
    so failure is -1. The first live run tested `y == 0` for failure, found 0 of
    57 real failures, disabled the gate, and bought 13,273 pairs instead of the
    predicted ~68k. It also silently broke `knn_success_prob`, which averages
    labels and would return values in [-1, 1] rather than probabilities.
    """
    import numpy as np
    from adaptive_roa.adaptive_v2.strategy.class_quota import (
        ClassQuotaDecompositionAcquisitionStrategy,
    )

    class _S:
        def get_start_state(self, i):
            return np.array([0.1 * i, 0.0])

        def get_label(self, i):
            return 1 if i < 43 else -1        # internal convention: -1 = failure

        def get_trajectory_length(self, i):
            return 130 if i < 43 else 1001

    class _B:
        def __init__(self):
            self.data_source = _S()
            self.train_split = type("T", (), {"indices": list(range(100))})()

    class _P:
        def __init__(self):
            self.dataset_builder = _B()

    s = ClassQuotaDecompositionAcquisitionStrategy(_cfg())
    _X, y, _L = s._acquired_truth(_P())
    assert set(np.unique(y)) <= {0.0, 1.0}, "labels must be normalised to {0,1}"
    assert int((y == 1).sum()) == 43 and int((y == 0).sum()) == 57
