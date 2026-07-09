import numpy as np
from adaptive_roa.partx.classify import classify_region


def test_confident_positive():
    m = np.full(200, 3.0); s2 = np.full(200, 0.01)
    assert classify_region(m, s2, alpha=0.05) == "+"


def test_confident_negative():
    m = np.full(200, -3.0); s2 = np.full(200, 0.01)
    assert classify_region(m, s2, alpha=0.05) == "-"


def test_straddling_is_remaining():
    m = np.linspace(-3, 3, 200); s2 = np.full(200, 1.0)
    assert classify_region(m, s2, alpha=0.05) == "r"
