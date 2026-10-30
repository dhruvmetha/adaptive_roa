"""Pendulum first-hitting recut: success and failure-ball membership, first entry."""
import numpy as np
import pytest

from adaptive_roa.det_rerun.failset import first_hit, pendulum_membership, wrap_angle

CENTERS = [(2.0957, 0.0), (-2.0957, 0.0)]


def _member(states):
    return pendulum_membership(np.asarray(states, dtype=float), success_radius=0.075,
                               failure_centers=CENTERS, failure_radius=0.05)


def test_wrap_angle_maps_to_minus_pi_pi():
    out = wrap_angle(np.array([3 * np.pi / 2, -3 * np.pi / 2, 0.1]))
    assert np.allclose(out, [-np.pi / 2, np.pi / 2, 0.1])


def test_success_ball_uses_raw_l2():
    in_s, in_f = _member([[0.05, 0.05], [0.06, 0.05]])
    assert in_s.tolist() == [True, False]
    assert not in_f.any()


def test_failure_ball_wraps_theta_difference():
    # 2.0957 + 2*pi is the same angle as the +2.0957 attractor.
    in_s, in_f = _member([[2.0957 + 2 * np.pi, 0.0], [-2.0957 + 0.04, 0.01], [2.0, 0.0]])
    assert in_f.tolist() == [True, True, False]
    assert not in_s.any()


def test_first_hit_success_first():
    assert first_hit(np.array([0, 0, 1, 0], bool), np.array([0, 0, 0, 1], bool)) == (2, 1)


def test_first_hit_failure_first_overrides_later_success():
    assert first_hit(np.array([0, 0, 0, 1], bool), np.array([0, 1, 0, 0], bool)) == (1, 0)


def test_first_hit_neither_is_timeout():
    assert first_hit(np.zeros(4, bool), np.zeros(4, bool)) == (None, -1)


def test_first_hit_both_on_one_row_raises():
    with pytest.raises(ValueError):
        first_hit(np.array([0, 1], bool), np.array([0, 1], bool))


def test_first_hit_at_row_zero():
    # A start already inside a set is resolved at row 0.
    assert first_hit(np.array([1, 0], bool), np.array([0, 0], bool)) == (0, 1)
