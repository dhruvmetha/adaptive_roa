import math

import numpy as np
import pytest

from adaptive_roa.adaptive_v2.strategy.dispersion_score import (
    mean_pairwise_dispersion,
    normalized_distances,
    select_greedy,
    select_greedy_diverse,
    select_proportional,
)

# An arbitrary but self-consistent 2-D scale vector (theta circular, theta_dot
# real) used to exercise mean_pairwise_dispersion's arithmetic in isolation.
# NOTE: this is not what DynamicalSystem.get_normalization_scales() would
# return for a pendulum under the full-range convention (it would give 16.0
# for theta_dot bounded to [-8, 8], not 8.0) -- see test_normalization_scales.py
# for tests against the real convention. The tests below are still valid
# because they only rely on internal self-consistency of this vector.
SCALES = np.array([math.pi, 8.0], dtype=np.float64)
CIRCULAR = np.array([True, False])


def test_identical_endpoints_score_zero():
    endpoints = np.tile(np.array([[0.3, 1.5]]), (4, 1))[None, :, :]  # [1, 4, 2]
    scores = mean_pairwise_dispersion(endpoints, SCALES, CIRCULAR)
    assert scores.shape == (1,)
    assert scores[0] == pytest.approx(0.0, abs=1e-6)


def test_circular_dimension_wraps_across_pi():
    eps = 0.01
    endpoints = np.array([[[math.pi - eps, 0.0], [-math.pi + eps, 0.0]]])  # [1, 2, 2]
    scores = mean_pairwise_dispersion(endpoints, SCALES, CIRCULAR)
    # True separation is 2*eps, not 2*pi - 2*eps; normalized by pi.
    assert scores[0] == pytest.approx(2 * eps / math.pi, abs=1e-5)


def test_separation_equal_to_scale_costs_the_same_in_every_dimension():
    # This checks the metric's arithmetic in isolation (see the SCALES
    # comment above): a pairwise separation equal to a dimension's own scale
    # value normalizes to 1.0 in that dimension, so it costs the same
    # regardless of which dimension it's in.
    # theta separation of pi (wraps exactly to pi, the max) == SCALES[0].
    theta_spread = np.array([[[math.pi / 2, 0.0], [-math.pi / 2, 0.0]]])
    # theta_dot separation of 8.0 == SCALES[1].
    vel_spread = np.array([[[0.0, 4.0], [0.0, -4.0]]])
    s_theta = mean_pairwise_dispersion(theta_spread, SCALES, CIRCULAR)
    s_vel = mean_pairwise_dispersion(vel_spread, SCALES, CIRCULAR)
    assert s_theta[0] == pytest.approx(s_vel[0], rel=1e-5)


def test_two_modes_outscore_one_tight_blob():
    tight = np.array([[[0.0, 0.0], [0.01, 0.0], [0.0, 0.01], [0.01, 0.01]]])
    bimodal = np.array([[[0.0, 0.0], [0.01, 0.0], [1.5, 0.0], [1.51, 0.0]]])
    scores = mean_pairwise_dispersion(
        np.concatenate([tight, bimodal], axis=0), SCALES, CIRCULAR
    )
    assert scores[1] > scores[0]


def test_known_two_point_value():
    # Two endpoints 4.0 apart in theta_dot only: d = 4.0 / 8.0 = 0.5.
    # K=2 -> mean over the single pair = 0.5.
    endpoints = np.array([[[0.0, 0.0], [0.0, 4.0]]])
    scores = mean_pairwise_dispersion(endpoints, SCALES, CIRCULAR)
    assert scores[0] == pytest.approx(0.5, abs=1e-6)


def test_chunking_matches_unchunked():
    rng = np.random.default_rng(0)
    endpoints = rng.normal(size=(37, 6, 2))
    full = mean_pairwise_dispersion(endpoints, SCALES, CIRCULAR, chunk_size=1024)
    chunked = mean_pairwise_dispersion(endpoints, SCALES, CIRCULAR, chunk_size=5)
    np.testing.assert_allclose(full, chunked, rtol=1e-5)


def test_non_finite_candidate_scores_nan_without_poisoning_neighbours():
    endpoints = np.array(
        [
            [[0.0, 0.0], [0.0, 4.0]],
            [[0.0, np.nan], [0.0, 4.0]],
            [[0.0, 0.0], [np.inf, 4.0]],
            [[0.0, 0.0], [0.0, 4.0]],
        ]
    )
    scores = mean_pairwise_dispersion(endpoints, SCALES, CIRCULAR, chunk_size=2)
    assert np.isnan(scores[1])
    assert np.isnan(scores[2])
    assert scores[0] == pytest.approx(0.5, abs=1e-6)
    assert scores[3] == pytest.approx(0.5, abs=1e-6)


def test_requires_at_least_two_samples():
    with pytest.raises(ValueError, match="at least 2"):
        mean_pairwise_dispersion(np.zeros((3, 1, 2)), SCALES, CIRCULAR)


def test_rejects_non_positive_chunk_size():
    endpoints = np.zeros((3, 2, 2))
    with pytest.raises(ValueError, match="chunk_size"):
        mean_pairwise_dispersion(endpoints, SCALES, CIRCULAR, chunk_size=0)
    with pytest.raises(ValueError, match="chunk_size"):
        mean_pairwise_dispersion(endpoints, SCALES, CIRCULAR, chunk_size=-5)


def test_normalized_distances_wraps_and_scales():
    states = np.array([[math.pi - 0.01, 0.0], [0.0, 8.0]])
    reference = np.array([-math.pi + 0.01, 0.0])
    d = normalized_distances(states, reference, SCALES, CIRCULAR)
    assert d[0] == pytest.approx(0.02 / math.pi, abs=1e-5)
    # second: theta differs by pi - 0.01 (normalized ~0.997), vel by 8/8 = 1.0
    expected = math.hypot((math.pi - 0.01) / math.pi, 1.0)
    assert d[1] == pytest.approx(expected, rel=1e-5)


def test_greedy_takes_highest_scores():
    scores = np.array([0.1, 0.9, 0.5, 0.7])
    assert select_greedy(scores, 2).tolist() == [1, 3]


def test_greedy_skips_nan_scores():
    scores = np.array([0.1, np.nan, 0.5, np.nan, 0.7])
    picked = select_greedy(scores, 3).tolist()
    assert picked == [4, 2, 0]


def test_greedy_returns_all_when_fewer_than_requested():
    scores = np.array([0.1, np.nan, 0.5])
    assert sorted(select_greedy(scores, 10).tolist()) == [0, 2]


def test_greedy_diverse_spreads_further_than_greedy():
    # Four high-scoring states clustered together, plus three lower-scoring
    # states that are far away. Greedy takes only the cluster; diverse spreads.
    states = np.array(
        [
            [0.00, 0.0], [0.01, 0.0], [0.02, 0.0], [0.03, 0.0],
            [2.00, 0.0], [-2.00, 0.0], [0.00, 6.0],
        ]
    )
    scores = np.array([0.99, 0.98, 0.97, 0.96, 0.90, 0.89, 0.88])

    greedy = select_greedy(scores, 3)
    diverse = select_greedy_diverse(scores, states, SCALES, CIRCULAR, 3, pool_multiplier=3)

    def min_separation(idx):
        pts = states[idx]
        return min(
            normalized_distances(pts, pts[i], SCALES, CIRCULAR)[j]
            for i in range(len(pts))
            for j in range(len(pts))
            if i != j
        )

    assert min_separation(diverse) > min_separation(greedy)


def test_greedy_diverse_seeds_at_highest_score():
    states = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    scores = np.array([0.1, 0.9, 0.5, 0.4])
    picked = select_greedy_diverse(scores, states, SCALES, CIRCULAR, 2, pool_multiplier=4)
    assert picked[0] == 1


def test_greedy_diverse_returns_unique_indices():
    rng = np.random.default_rng(3)
    states = rng.normal(size=(40, 2))
    scores = rng.random(40)
    picked = select_greedy_diverse(scores, states, SCALES, CIRCULAR, 10)
    assert len(set(picked.tolist())) == 10


def test_greedy_diverse_skips_nan_scores():
    states = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    scores = np.array([0.9, np.nan, 0.5, 0.4])
    picked = select_greedy_diverse(scores, states, SCALES, CIRCULAR, 3, pool_multiplier=5)
    assert 1 not in picked.tolist()


def test_proportional_is_reproducible_under_seed():
    rng = np.random.default_rng(1)
    scores = rng.random(50)
    a = select_proportional(scores, 10, temperature=0.1, seed=7)
    b = select_proportional(scores, 10, temperature=0.1, seed=7)
    np.testing.assert_array_equal(a, b)


def test_proportional_differs_across_seeds():
    rng = np.random.default_rng(1)
    scores = rng.random(200)
    a = select_proportional(scores, 20, temperature=0.5, seed=1)
    b = select_proportional(scores, 20, temperature=0.5, seed=2)
    assert set(a.tolist()) != set(b.tolist())


def test_proportional_favours_high_scores_at_low_temperature():
    scores = np.concatenate([np.full(10, 1.0), np.full(90, 0.0)])
    picked = select_proportional(scores, 10, temperature=0.01, seed=5)
    assert set(picked.tolist()) == set(range(10))


def test_proportional_selects_unique_and_skips_nan():
    scores = np.array([0.9, np.nan, 0.5, 0.4, 0.8, np.nan, 0.2])
    picked = select_proportional(scores, 3, temperature=0.2, seed=0)
    assert len(set(picked.tolist())) == 3
    assert 1 not in picked.tolist()
    assert 5 not in picked.tolist()


def test_proportional_handles_all_equal_scores():
    scores = np.full(20, 0.42)
    picked = select_proportional(scores, 5, temperature=0.1, seed=0)
    assert len(set(picked.tolist())) == 5


# --- mode separation -------------------------------------------------------
# The score dispersion structurally cannot produce: it must rank two tight,
# well-separated modes ABOVE one wide unimodal blob, even when the blob has the
# larger mean pairwise distance. That is the defect the campaign exposed.

from adaptive_roa.adaptive_v2.strategy.dispersion_score import (
    mode_separation,
    idempotence_defect,
)


def _bimodal(gap, n=10, jitter=0.002, seed=0):
    """n endpoints split evenly between two tight clusters `gap` apart in theta_dot."""
    rng = np.random.default_rng(seed)
    a = np.column_stack([rng.normal(0, jitter, n // 2), rng.normal(0.0, jitter, n // 2)])
    b = np.column_stack([rng.normal(0, jitter, n // 2), rng.normal(gap, jitter, n // 2)])
    return np.vstack([a, b])


def _blob(width, n=10, seed=0):
    """n endpoints spread uniformly over `width` in theta_dot -- one wide mode."""
    rng = np.random.default_rng(seed)
    return np.column_stack([np.zeros(n), rng.uniform(-width / 2, width / 2, n)])


def test_two_tight_modes_outscore_a_wider_blob():
    # The blob is deliberately WIDER, so mean pairwise dispersion prefers it.
    modes, blob = _bimodal(gap=3.0)[None], _blob(width=6.0)[None]
    disp = mean_pairwise_dispersion(np.concatenate([modes, blob]), SCALES, CIRCULAR)
    sep = mode_separation(np.concatenate([modes, blob]), SCALES, CIRCULAR)
    assert disp[1] > disp[0], "precondition: blob must have larger raw dispersion"
    assert sep[0] > sep[1], "mode separation must prefer the bimodal cloud"


def test_unimodal_clouds_score_below_bimodal_floor():
    # Splitting any contiguous cloud leaves a gap comparable to the within-side
    # spread, so unimodal sits near 0.5 -- not 0. What matters is the margin to
    # the bimodal regime (>0.9), which is wide and width-independent.
    for width in (0.001, 1.0, 8.0):
        assert mode_separation(_blob(width=width)[None], SCALES, CIRCULAR)[0] < 0.6


def test_score_is_invariant_to_cloud_width():
    # THE property dispersion lacks: modality, not scale. A cloud 8000x wider
    # but equally unimodal must score the same, where dispersion differs ~100x.
    narrow = mode_separation(_blob(width=0.001)[None], SCALES, CIRCULAR)[0]
    wide = mode_separation(_blob(width=8.0)[None], SCALES, CIRCULAR)[0]
    assert abs(narrow - wide) < 0.05
    d_narrow = mean_pairwise_dispersion(_blob(width=0.001)[None], SCALES, CIRCULAR)[0]
    d_wide = mean_pairwise_dispersion(_blob(width=8.0)[None], SCALES, CIRCULAR)[0]
    assert d_wide > 100 * max(d_narrow, 1e-9), "precondition: dispersion IS width-sensitive"


def test_score_rises_with_mode_gap():
    clouds = np.concatenate([_bimodal(gap=g)[None] for g in (0.2, 1.0, 4.0)])
    sep = mode_separation(clouds, SCALES, CIRCULAR)
    assert sep[0] < sep[1] < sep[2]


def test_lone_outlier_is_penalised_by_balance_term():
    # 9 endpoints together, 1 far away: large gap but p=0.1 -> 4p(1-p)=0.36
    cloud = np.vstack([_blob(width=0.001, n=9), np.array([[0.0, 6.0]])])[None]
    balanced = _bimodal(gap=6.0)[None]
    sep = mode_separation(np.concatenate([cloud, balanced]), SCALES, CIRCULAR)
    assert sep[1] > sep[0]


def test_mode_separation_is_bounded_in_unit_interval():
    clouds = np.concatenate([_bimodal(gap=g, seed=s)[None]
                             for g in (0.1, 2.0, 9.0) for s in (0, 1)])
    sep = mode_separation(clouds, SCALES, CIRCULAR)
    assert np.all(sep >= 0.0) and np.all(sep <= 1.0)


def test_mode_separation_wraps_circular_dimension():
    # Two groups split by 2*eps ACROSS the +/-pi seam must score the same as two
    # groups split by 2*eps in the interior. Without wrapping the seam pair reads
    # as 2*pi apart. (Jitter is required: with zero within-group spread the ratio
    # saturates at 1.0 for any gap and the comparison is vacuous.)
    rng = np.random.default_rng(0)
    eps, j = 0.01, 0.002

    def two_groups(c_a, c_b):
        a = np.column_stack([rng.normal(c_a, j, 5), rng.normal(0, j, 5)])
        b = np.column_stack([rng.normal(c_b, j, 5), rng.normal(0, j, 5)])
        return np.vstack([a, b])[None]

    seam = mode_separation(two_groups(math.pi - eps, -math.pi + eps), SCALES, CIRCULAR)[0]
    interior = mode_separation(two_groups(-eps, eps), SCALES, CIRCULAR)[0]
    assert abs(seam - interior) < 0.1


def test_mode_separation_marks_non_finite_clouds_nan():
    clouds = np.concatenate([_bimodal(gap=3.0)[None], _bimodal(gap=3.0)[None]])
    clouds[1, 0, 1] = np.nan
    sep = mode_separation(clouds, SCALES, CIRCULAR)
    assert not np.isnan(sep[0]) and np.isnan(sep[1])


# --- idempotence defect ----------------------------------------------------

def test_idempotence_defect_zero_when_endpoints_are_fixed_points():
    cloud = _bimodal(gap=3.0)[None]
    d = idempotence_defect(cloud, cloud.copy(), SCALES, CIRCULAR)   # E(e) == e
    assert d[0] == pytest.approx(0.0, abs=1e-6)


def test_idempotence_defect_measures_mean_displacement():
    cloud = np.zeros((1, 4, 2))
    remapped = cloud.copy(); remapped[0, :, 1] = 4.0      # every endpoint moves 4.0 in theta_dot
    d = idempotence_defect(cloud, remapped, SCALES, CIRCULAR)
    assert d[0] == pytest.approx(4.0 / SCALES[1], abs=1e-6)


def test_idempotence_defect_marks_non_finite_nan():
    cloud = np.zeros((2, 4, 2)); remapped = cloud.copy()
    remapped[1, 0, 0] = np.inf
    d = idempotence_defect(cloud, remapped, SCALES, CIRCULAR)
    assert not np.isnan(d[0]) and np.isnan(d[1])


# --- k-agnostic mode separation -------------------------------------------
# The 2-partition deflates clouds with 3+ modes: the farthest-pair seeding merges
# two real modes into one side, inflating W and shrinking the apparent gap. On
# quadrotor2d ~30% of model-ambiguous candidates have 3+ clumps, so that is a
# systematic blind spot, not a corner case.

from adaptive_roa.adaptive_v2.strategy.dispersion_score import mode_separation_linkage


def _trimodal(gap, n=12, jitter=0.002, seed=0):
    """n endpoints in three tight clusters spaced `gap` apart in theta_dot."""
    rng = np.random.default_rng(seed)
    parts = [np.column_stack([rng.normal(0, jitter, n // 3),
                              rng.normal(c * gap, jitter, n // 3)]) for c in (-1, 0, 1)]
    return np.vstack(parts)


def test_linkage_scores_trimodal_high():
    assert mode_separation_linkage(_trimodal(gap=3.0)[None], SCALES, CIRCULAR)[0] > 0.8


def test_linkage_beats_two_partition_on_trimodal():
    # The point of the generalisation: the 2-way split cannot see three modes.
    cloud = _trimodal(gap=3.0)[None]
    assert (mode_separation_linkage(cloud, SCALES, CIRCULAR)[0]
            > mode_separation(cloud, SCALES, CIRCULAR)[0])


def test_linkage_still_scores_bimodal_high():
    assert mode_separation_linkage(_bimodal(gap=3.0)[None], SCALES, CIRCULAR)[0] > 0.8


def test_linkage_scores_unimodal_low():
    for width in (0.001, 1.0, 8.0):
        assert mode_separation_linkage(_blob(width=width)[None], SCALES, CIRCULAR)[0] < 0.6


def test_linkage_separates_unimodal_from_multimodal():
    blob = mode_separation_linkage(_blob(width=6.0)[None], SCALES, CIRCULAR)[0]
    bi = mode_separation_linkage(_bimodal(gap=3.0)[None], SCALES, CIRCULAR)[0]
    tri = mode_separation_linkage(_trimodal(gap=3.0)[None], SCALES, CIRCULAR)[0]
    assert min(bi, tri) > blob + 0.25


def test_linkage_penalises_lone_outlier():
    lone = np.vstack([_blob(width=0.001, n=11), np.array([[0.0, 6.0]])])[None]
    even = _bimodal(gap=6.0)[None]
    assert (mode_separation_linkage(even, SCALES, CIRCULAR)[0]
            > mode_separation_linkage(lone, SCALES, CIRCULAR)[0])


def test_linkage_marks_non_finite_nan():
    clouds = np.concatenate([_bimodal(gap=3.0)[None], _bimodal(gap=3.0)[None]])
    clouds[1, 0, 1] = np.nan
    s = mode_separation_linkage(clouds, SCALES, CIRCULAR)
    assert not np.isnan(s[0]) and np.isnan(s[1])
