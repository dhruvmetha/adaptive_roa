import numpy as np
import pytest

from adaptive_roa.partx_faithful.mcstep import (Quantiles, classify, falsification_indicator,
                                                mc_step, z_value)


class _Const:
    def __init__(self, m, s):
        self.m, self.s = m, s

    def latent(self, X):
        n = len(X)
        return np.full(n, self.m), np.full(n, self.s)


def _q(q_max, q_min, var=0.0):
    return Quantiles(q_max=q_max, var_q_max=var, q_min=q_min, var_q_min=var)


def test_z_value():
    assert z_value(0.05) == pytest.approx(1.959964, abs=1e-6)


def test_mc_step_on_a_constant_model():
    q = mc_step(_Const(0.3, 0.1), np.zeros(2), np.ones(2), R=5, M=20, delta_c=0.05,
                rng=np.random.default_rng(0))
    z = z_value(0.05)
    assert q.q_max == pytest.approx(0.3 + z * 0.1)
    assert q.q_min == pytest.approx(0.3 - z * 0.1)
    assert q.var_q_max == pytest.approx(0.0) and q.var_q_min == pytest.approx(0.0)


@pytest.mark.parametrize("rtype,q,expected", [
    ("r", _q(-0.1, -0.5), "-"),
    ("r", _q(0.5, 0.1), "+"),
    ("r", _q(0.5, -0.5), "r"),
    ("+", _q(0.5, 0.1), "+"),
    ("+", _q(0.5, 0.0), "r+"),
    ("+", _q(0.5, -0.1), "r+"),
    ("-", _q(-0.1, -0.5), "-"),
    ("-", _q(0.0, -0.5), "r-"),
    ("r+", _q(-1.0, -2.0), "r+"),
    ("r-", _q(2.0, 1.0), "r-"),
])
def test_classify_truth_table(rtype, q, expected):
    assert classify(rtype, 1.0, 0.0, q, 0.05) == expected


def test_small_regions_become_u():
    assert classify("r", 1e-9, 1e-6, _q(5.0, 4.0), 0.05) == "u"


def test_terminal_types_cannot_be_classified():
    with pytest.raises(ValueError):
        classify("u", 1.0, 0.0, _q(1.0, 0.0), 0.05)


def test_compat_ci_var_uses_the_variance_instead_of_the_sd():
    # var 0.04 -> sd 0.2: z*0.2 = 0.39 > 0.3 keeps 'r'; z*0.04 = 0.078 < 0.3 gives '+'
    q = _q(1.0, 0.3, var=0.04)
    assert classify("r", 1.0, 0.0, q, 0.05) == "r"
    assert classify("r", 1.0, 0.0, q, 0.05, compat_ci_var=True) == "+"


def test_falsification_indicator():
    rng = np.random.default_rng(0)
    low, high = np.zeros(2), np.array([2.0, 3.0])
    assert falsification_indicator(_Const(0.0, 1.0), low, high, 2, 10, rng) == pytest.approx(0.5)
    assert falsification_indicator(_Const(0.0, 1.0), low, high, 2, 10, rng,
                                   times_volume=True) == pytest.approx(3.0)
