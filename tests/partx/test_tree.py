import numpy as np
from adaptive_roa.systems.pendulum import PendulumSystem
from adaptive_roa.partx.tree import PartitionTree, build_root


def test_root_covers_support():
    system = PendulumSystem()
    root = build_root(system)
    assert root.contains(np.array([[0.0, 0.0]]))[0]


def test_refine_subdivides_boundary():
    system = PendulumSystem()
    # Latent = signed distance to θ=1.0 line: boundary passes through the domain,
    # so the root must straddle and subdivide.
    def latent_fn(X):
        m = 1.0 - X[:, 0]          # f>0 for θ<1, f<0 for θ>1
        return m, np.full(len(X), 0.02)
    tree = PartitionTree(system, delta=0.1, m_class=128)
    tree.refine(latent_fn)
    assert len(tree.leaves()) > 1               # subdivided
    assert len(tree.remaining_leaves()) >= 1    # a boundary leaf remains
    # a clearly-positive point lands in a '+' leaf after enough refinement
    # Root normalized sides are [angle, angular_velocity] = [0.5, 1.0] after the
    # first split, so longest_dim() alternates dims each round (ties -> dim 0):
    # angle is only re-split on the 3rd subdivision, so 4 total refine() calls
    # are needed before a sub-box lies fully on one side of theta=1.0. This is
    # deterministic geometry (angular_velocity bound is 2x the angle bound),
    # not RNG-dependent -- verified stable across seeds 0-19.
    tree.refine(latent_fn); tree.refine(latent_fn); tree.refine(latent_fn)
    assert any(leaf.region_class == "+" for leaf in tree.leaves())
    assert any(leaf.region_class == "-" for leaf in tree.leaves())
