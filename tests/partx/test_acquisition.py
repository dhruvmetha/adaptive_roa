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
