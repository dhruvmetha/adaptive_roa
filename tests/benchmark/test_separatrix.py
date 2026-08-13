import numpy as np
import pytest
import torch

from adaptive_roa.benchmark.separatrix import separatrix_band, conditioned_metrics


def _two_blobs(n=60):
    # Two well-separated clusters: only points near x=0 should be flagged.
    #
    # NOTE on the 0.02 half-gap (the brief's Step 1 code uses 0.05): with
    # 0.05, the brief's OWN reference implementation (Step 3, run verbatim
    # -- an O(n^2) dense-matrix k-NN, not this module's cKDTree version)
    # fails its own test. Intra-cluster spacing at n=60 is
    # 0.95/29=0.032758; with k=3 a point straddling the gap needs its 3rd
    # nearest same-cluster neighbour to be FARTHER than the gap for the
    # opposite cluster to displace it into the k=3 set. At 0.05 the gap is
    # 0.10 and 3*spacing=0.098275 -- juuust under 0.10, so the two clusters
    # never displace each other into range and `band` comes back all-False
    # for every n the brief's tests use (60, 100), independent of which
    # k-NN implementation computes it (verified against the brief's exact
    # Step-3 code before changing anything here). 0.02 gives comfortable
    # margin (gap=0.04 vs 3*spacing=0.101) while keeping the fixture's
    # documented intent -- "only points near x=0 should be flagged" --
    # otherwise identical.
    left = np.linspace(-1.0, -0.02, n // 2).reshape(-1, 1)
    right = np.linspace(0.02, 1.0, n // 2).reshape(-1, 1)
    states = np.vstack([left, right])
    labels = np.array([0] * (n // 2) + [1] * (n // 2))
    return states, labels


# ---------------------------------------------------------------------------
# Brief's tests, verbatim.
# ---------------------------------------------------------------------------


def test_boundary_points_are_flagged_and_far_points_are_not():
    states, labels = _two_blobs()
    band = separatrix_band(states, labels, k=3)
    assert band[len(labels) // 2 - 1] and band[len(labels) // 2]   # straddle 0
    assert not band[0] and not band[-1]                            # extremes


def test_a_single_label_yields_an_empty_band():
    states = np.linspace(0, 1, 20).reshape(-1, 1)
    band = separatrix_band(states, np.zeros(20, dtype=int), k=3)
    assert not band.any()


def test_conditioned_metrics_split_the_population():
    states, labels = _two_blobs()
    probs = labels.astype(float)                # a perfect predictor
    m = conditioned_metrics(states, labels, probs, k=3)
    assert m["n_near"] + m["n_interior"] == len(labels)
    assert m["overall"] == pytest.approx(1.0)
    assert m["near_boundary"] == pytest.approx(1.0)


def test_boundary_errors_are_invisible_in_the_aggregate():
    # THE POINT of this module: an arm that fails only at the boundary still
    # scores well overall, so the aggregate alone cannot distinguish arms.
    states, labels = _two_blobs(n=100)
    probs = labels.astype(float)
    band = separatrix_band(states, labels, k=3)
    probs[band] = 1.0 - probs[band]             # wrong on every boundary point
    m = conditioned_metrics(states, labels, probs, k=3)
    assert m["overall"] > 0.85                  # aggregate still looks healthy
    assert m["near_boundary"] == pytest.approx(0.0)
    # Deliberately distinct from near_boundary (0.0): interior points were
    # never touched, so this also pins that "interior" and "near_boundary"
    # read from the CORRECT (complementary) mask rather than the same one --
    # a mutation that made "interior" reuse `band` instead of `~band` would
    # otherwise slip through unnoticed here (both keys would read 0.0).
    assert m["interior"] == pytest.approx(1.0)


def test_k_larger_than_the_population_is_rejected():
    states, labels = _two_blobs(n=10)
    with pytest.raises(ValueError, match="k"):
        separatrix_band(states, labels, k=50)


# ---------------------------------------------------------------------------
# Headline property, reinforced: overall must come from the FULL population,
# not from a mutant that quietly restricts it to interior or boundary only.
# ---------------------------------------------------------------------------


def test_overall_is_computed_over_the_full_population_independently_checked():
    states, labels = _two_blobs(n=100)
    probs = labels.astype(float)
    band = separatrix_band(states, labels, k=3)
    probs[band] = 1.0 - probs[band]
    m = conditioned_metrics(states, labels, probs, k=3)

    preds = probs >= 0.5
    expected_overall = float((preds == labels.astype(bool)).mean())
    assert m["overall"] == pytest.approx(expected_overall)
    assert m["overall"] == pytest.approx((100 - band.sum()) / 100.0)


def test_near_and_interior_masks_match_separatrix_band_exactly():
    states, labels = _two_blobs(n=80)
    probs = labels.astype(float)
    band = separatrix_band(states, labels, k=3)
    m = conditioned_metrics(states, labels, probs, k=3)
    assert m["n_near"] == int(band.sum())
    assert m["n_interior"] == int((~band).sum())
    assert m["n_near"] + m["n_interior"] == len(labels)
    assert m["n_interior"] == len(labels) - m["n_near"]


# ---------------------------------------------------------------------------
# Ties and degenerate inputs.
# ---------------------------------------------------------------------------


def test_population_of_two_opposite_labels_are_mutually_flagged():
    states = np.array([[0.0], [1.0]])
    labels = np.array([0, 1])
    band = separatrix_band(states, labels, k=1)
    assert band.tolist() == [True, True]


def test_population_of_two_same_label_is_empty():
    states = np.array([[0.0], [1.0]])
    labels = np.array([0, 0])
    band = separatrix_band(states, labels, k=1)
    assert not band.any()


def test_k_equal_to_population_of_two_is_rejected():
    states = np.array([[0.0], [1.0]])
    labels = np.array([0, 1])
    with pytest.raises(ValueError, match="k"):
        separatrix_band(states, labels, k=2)


def test_k_zero_is_rejected():
    states, labels = _two_blobs(n=10)
    with pytest.raises(ValueError, match="k"):
        separatrix_band(states, labels, k=0)


def test_duplicate_states_with_conflicting_labels_are_mutually_flagged():
    # Two points at the exact same location with different labels ARE the
    # boundary by definition; four more duplicates elsewhere, same label,
    # are not.
    states = np.array([[0.0], [0.0], [5.0], [5.0], [5.0], [5.0]])
    labels = np.array([0, 1, 1, 1, 1, 1])
    band = separatrix_band(states, labels, k=1)
    assert band[0] and band[1]
    assert not band[2:].any()


def test_more_duplicates_than_k_plus_one_does_not_crash():
    # 10 exactly-colocated points, split 5/5 by label, with k=3: more
    # duplicates than the k+1 neighbours queried, exercising the fallback
    # path where a point's own index may not appear among the tied
    # candidates the tree returns. Only shape/dtype are pinned -- which
    # specific ties scipy resolves is an implementation detail, not part of
    # this module's contract.
    n = 10
    states = np.zeros((n, 1))
    labels = np.array([0] * 5 + [1] * 5)
    band = separatrix_band(states, labels, k=3)
    assert band.shape == (n,)
    assert band.dtype == bool


def test_all_points_share_one_label_conditioned_metrics_has_no_near_boundary():
    states = np.linspace(0, 1, 20).reshape(-1, 1)
    labels = np.zeros(20, dtype=int)
    probs = np.full(20, 0.1)  # predicts failure, matching the shared label
    m = conditioned_metrics(states, labels, probs, k=3)
    assert m["n_near"] == 0
    assert m["n_interior"] == 20
    assert np.isnan(m["near_boundary"])
    assert m["interior"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Normalization: enforced via `system`, best-effort guarded without one.
# ---------------------------------------------------------------------------


class _FakeSystem:
    """Minimal stand-in for adaptive_roa.systems.*: scales each dimension."""

    def __init__(self, scale):
        self._scale = torch.as_tensor(scale, dtype=torch.float32)

    def normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        return state / self._scale


def test_system_normalization_is_actually_applied():
    # Raw states: one dimension straddles 0 at small scale (the true
    # boundary), the other is a huge-range decoy that would swamp distances
    # if left unnormalized. A system that scales the decoy dimension back
    # down must recover the same boundary as the single-dimension case.
    n = 60
    x = np.concatenate([
        np.linspace(-1.0, -0.05, n // 2),
        np.linspace(0.05, 1.0, n // 2),
    ])
    decoy = np.linspace(-1000.0, 1000.0, n)  # huge scale, uncorrelated w/ label
    raw_states = np.stack([x, decoy], axis=1)
    labels = np.array([0] * (n // 2) + [1] * (n // 2))

    system = _FakeSystem(scale=[1.0, 1000.0])  # normalizes decoy back to [-1, 1]
    band = separatrix_band(raw_states, labels, k=3, system=system)
    assert band[n // 2 - 1] and band[n // 2]
    assert not band[0] and not band[-1]


def test_without_system_unnormalized_scale_is_rejected():
    n = 60
    x = np.concatenate([
        np.linspace(-1.0, -0.05, n // 2),
        np.linspace(0.05, 1.0, n // 2),
    ])
    decoy = np.linspace(-1000.0, 1000.0, n)
    raw_states = np.stack([x, decoy], axis=1)
    labels = np.array([0] * (n // 2) + [1] * (n // 2))

    with pytest.raises(ValueError, match="unnormalized|normaliz"):
        separatrix_band(raw_states, labels, k=3)


def test_system_missing_normalize_state_is_rejected():
    states, labels = _two_blobs(n=10)
    states_2d = np.hstack([states, states])  # 2D so the scale guard would apply

    class _NotASystem:
        pass

    with pytest.raises(ValueError, match="normalize_state"):
        separatrix_band(states_2d, labels, k=3, system=_NotASystem())


# ---------------------------------------------------------------------------
# Label-convention and probability guards on conditioned_metrics.
# ---------------------------------------------------------------------------


def test_conditioned_metrics_rejects_pm1_label_convention():
    # This codebase's real ground-truth labels use {-1, 1} (failure,
    # success) elsewhere (adaptive_roa/adaptive/data_source.py). Silently
    # accepting them here via `.astype(bool)` would map -1 to True exactly
    # like 1, merging both classes into "success".
    states, _ = _two_blobs(n=10)
    labels_pm1 = np.array([-1] * 5 + [1] * 5)
    probs = np.full(10, 0.5)
    with pytest.raises(ValueError, match="0=failure/1=success|convention"):
        conditioned_metrics(states, labels_pm1, probs, k=3)


def test_conditioned_metrics_rejects_out_of_range_probs():
    states, labels = _two_blobs(n=10)
    probs = np.linspace(-0.2, 1.0, 10)  # one entry below 0
    with pytest.raises(ValueError):
        conditioned_metrics(states, labels, probs, k=3)


def test_conditioned_metrics_rejects_mismatched_probs_length():
    states, labels = _two_blobs(n=10)
    probs = np.full(9, 0.5)
    with pytest.raises(ValueError, match="probs"):
        conditioned_metrics(states, labels, probs, k=3)


def test_states_must_be_2d():
    labels = np.array([0, 1, 0, 1])
    with pytest.raises(ValueError, match="2D"):
        separatrix_band(np.array([0.0, 1.0, 2.0, 3.0]), labels, k=1)


def test_states_labels_row_mismatch_is_rejected():
    states, _ = _two_blobs(n=10)
    labels = np.zeros(9, dtype=int)
    with pytest.raises(ValueError, match="states"):
        separatrix_band(states, labels, k=1)
