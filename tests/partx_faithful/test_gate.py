import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

_SCRIPTS = str(Path(__file__).resolve().parents[2] / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
gate = importlib.import_module("partx_faithful_gate")


def test_leaf_types_at_clips_and_assigns():
    lows = np.array([[0.0, 0.0], [0.5, 0.0]])
    highs = np.array([[0.5, 1.0], [1.0, 1.0]])
    t = gate.leaf_types_at(np.array([[0.2, 0.5], [0.8, 0.5], [9.0, 0.5]]), lows, highs,
                           np.array(["-", "+"]), np.zeros(2), np.ones(2))
    assert t.tolist() == ["-", "+", "+"]


def test_coverage_and_wrong_side():
    types = np.array(["+", "+", "-", "r"])
    p = np.array([0.9, 0.4, 0.1, 0.5])
    cov, wrong, n_c = gate.coverage_and_wrong_side(types, p)
    assert cov == pytest.approx(0.75) and n_c == 3 and wrong == pytest.approx(1 / 3)


def test_ambiguous_fraction():
    assert gate.ambiguous_fraction(np.array([0.1, 0.2, 0.5, 0.8, 0.9])) == pytest.approx(0.6)
