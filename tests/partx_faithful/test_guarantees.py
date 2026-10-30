"""Part-X guarantees per snapshot (tracker C21)."""
import math

import numpy as np

from adaptive_roa.partx_faithful.guarantees import leaf_delta, snapshot_guarantees
from adaptive_roa.partx_faithful.region import Region


def test_leaf_delta_matches_lemma_1():
    # p = 0.5, alpha_1 = 0.05, n = 100 at the root level
    d = leaf_delta(0.5, 100, depth=0, alpha=0.05, B=2)
    assert math.isclose(d, (math.log(2) + math.log(20) + 2 * math.log(100) + 1) / 100)
    # one level down alpha halves
    assert leaf_delta(0.5, 100, depth=1, alpha=0.05, B=2) > d
    assert leaf_delta(0.0, 100, 0, 0.05, 2) == math.inf


class _Model:
    def __init__(self, m, s):
        self.m, self.s = m, s

    def latent(self, X):
        return np.full(len(X), self.m), np.full(len(X), self.s)


def _leaf(rid, rtype, lo, hi):
    return Region(rid=rid, parent=0, low=np.array(lo, float), high=np.array(hi, float),
                  branch_dir=0, depth=1, rtype=rtype)


def test_computable_side_reports_probability_and_volume():
    leaves = [_leaf(1, "+", [0, 0], [1, 1]), _leaf(2, "-", [1, 0], [2, 1])]
    models = {1: _Model(3.0, 1.0), 2: _Model(-3.0, 1.0)}
    g = snapshot_guarantees(leaves, models, {1: 2000, 2: 2000}, root_volume=2.0, alpha=0.05, B=2,
                            R=2, M=10, rng=np.random.default_rng(0))
    sat = g["sides"]["satisfying"]
    assert sat["computable"] and 0 < sat["probability_at_least"] < 1
    assert sat["misclassified_volume_frac_at_most"] < 0.5
    assert g["sides"]["violating"]["computable"]
    assert g["unchecked_assumptions"]


def test_small_n_is_reported_as_vacuous():
    leaves = [_leaf(1, "+", [0, 0], [1, 1])]
    g = snapshot_guarantees(leaves, {1: _Model(3.0, 1.0)}, {1: 10}, 1.0, 0.05, 2, 2, 10,
                            np.random.default_rng(0))
    sat = g["sides"]["satisfying"]
    assert not sat["computable"] and "vacuous" in sat["reason"]
    assert not g["sides"]["violating"]["computable"]       # no '-' leaf at all
