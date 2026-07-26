import math

import numpy as np
import pytest

from adaptive_roa.adaptive_v2.strategy.dispersion_score import (
    mean_pairwise_dispersion,
    normalized_distances,
)

# A pendulum-like 2-D space: theta circular, theta_dot bounded to [-8, 8].
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


def test_range_normalization_equalizes_dimensions():
    # Full-width spread in theta (2*pi wide, but wraps -> pi apart max)
    theta_spread = np.array([[[math.pi / 2, 0.0], [-math.pi / 2, 0.0]]])
    # Full-width spread in theta_dot (16 wide -> 8.0 apart after halving)
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


def test_normalized_distances_wraps_and_scales():
    states = np.array([[math.pi - 0.01, 0.0], [0.0, 8.0]])
    reference = np.array([-math.pi + 0.01, 0.0])
    d = normalized_distances(states, reference, SCALES, CIRCULAR)
    assert d[0] == pytest.approx(0.02 / math.pi, abs=1e-5)
    # second: theta differs by pi - 0.01 (normalized ~0.997), vel by 8/8 = 1.0
    expected = math.hypot((math.pi - 0.01) / math.pi, 1.0)
    assert d[1] == pytest.approx(expected, rel=1e-5)
