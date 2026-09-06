import numpy as np
from adaptive_roa.systems.pendulum import PendulumSystem
from adaptive_roa.partx.tree import PartitionTree
from adaptive_roa.partx.acquisition import straddle_score, select_pool_indices


def test_straddle_prefers_uncertain_boundary():
    m = np.array([0.0, 5.0, 0.0])
    s2 = np.array([1.0, 1.0, 0.01])
    s = straddle_score(m, s2)
    assert np.argmax(s) == 0        # near-boundary + high variance wins


def test_select_only_from_unresolved_leaves():
    system = PendulumSystem()
    def latent_fn(X):
        return 1.0 - X[:, 0], np.full(len(X), 0.02)
    tree = PartitionTree(system, delta=0.1, m_class=128)
    for _ in range(3):
        tree.refine(latent_fn)
    rng = np.random.default_rng(1)
    cand = np.column_stack([rng.uniform(-np.pi, np.pi, 300), rng.uniform(-8, 8, 300)])
    idx = list(range(300))
    m, s2 = latent_fn(cand)
    sel, diag = select_pool_indices(tree, cand, idx, m, s2, target_count=10)

    target_count = 10
    assert len(sel) == target_count

    # Every selected pool index must map back to a candidate whose state lies
    # in an unresolved ("r" or "min") leaf of the tree.
    leaf_of = tree.assign(cand)
    unresolved = {i for i, r in enumerate(tree.leaves()) if r.region_class in ("r", "min")}
    for s_i in sel:
        row = idx.index(s_i)
        assert leaf_of[row] in unresolved


class _StubLeaf:
    def __init__(self, region_class, lo, hi):
        self.region_class = region_class
        self._lo, self._hi = np.asarray(lo, float), np.asarray(hi, float)

    def volume(self):
        return float(np.prod(self._hi - self._lo))

    def contains(self, X):
        X = np.asarray(X)
        return np.all((X >= self._lo) & (X <= self._hi), axis=1)


class _StubTree:
    def __init__(self, leaves):
        self._leaves = leaves

    def leaves(self):
        return list(self._leaves)

    def assign(self, X):
        out = np.full(len(np.asarray(X)), -1, dtype=int)
        for i, leaf in enumerate(self._leaves):
            out[leaf.contains(X)] = i
        return out.tolist()


def _linear_posterior(states):
    m = states[:, 0].astype(float)
    return m, np.full(len(states), 0.25)


def test_no_eligible_candidate_falls_back_instead_of_acquiring_nothing():
    # Every candidate sits outside the tree, so the unresolved-leaf filter keeps
    # none. Returning [] here is what let q3d_nd048_partx train on unchanged data
    # for 18 consecutive epochs while still writing normal-looking artifacts.
    tree = _StubTree([_StubLeaf("r", [-0.1, -0.1], [0.1, 0.1])])
    states = np.array([[5.0, 5.0], [6.0, 6.0], [7.0, 7.0]])
    m, s2 = _linear_posterior(states)
    selected, diag = select_pool_indices(
        tree, states, [10, 11, 12], m, s2, target_count=2
    )
    assert len(selected) == 2, "must still spend the budget"
    assert diag["fallback_used"] is True
    assert diag["n_eligible"] == 0
    assert diag["n_outside_tree"] == 3


def test_shortfall_tops_up_to_the_full_budget():
    # Only one candidate lies in an unresolved leaf, but the epoch's budget is 3.
    # Banking the shortfall would put the arm on a smaller budget than the arms
    # it is ranked against (quad2D corridor spent 500, then 29, 8, 4, 1).
    tree = _StubTree([_StubLeaf("r", [-0.5, -0.5], [0.5, 0.5]),
                      _StubLeaf("+", [9.5, 9.5], [10.5, 10.5])])
    states = np.array([[0.0, 0.0], [10.0, 10.0], [20.0, 20.0], [30.0, 30.0]])
    m, s2 = _linear_posterior(states)
    selected, diag = select_pool_indices(
        tree, states, [0, 1, 2, 3], m, s2, target_count=3
    )
    assert len(selected) == 3
    assert diag["n_topup_outside_unresolved"] == 2
    assert diag["fallback_used"] is False


def test_full_eligibility_needs_no_topup():
    tree = _StubTree([_StubLeaf("r", [-50.0, -50.0], [50.0, 50.0])])
    states = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    m, s2 = _linear_posterior(states)
    selected, diag = select_pool_indices(
        tree, states, [0, 1, 2], m, s2, target_count=2
    )
    assert len(selected) == 2
    assert diag["n_topup_outside_unresolved"] == 0
    assert diag["fallback_used"] is False
