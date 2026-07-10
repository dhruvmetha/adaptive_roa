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


class _MultiDimComponent:
    def __init__(self, name, manifold_type, dim):
        self.name = name
        self.manifold_type = manifold_type
        self.dim = dim


class _FakeSystem:
    """Minimal system with multi-dimensional manifold components."""
    def __init__(self, components, bounds):
        self.manifold_components = components
        self.state_bounds = bounds


def test_build_root_expands_multidim_components():
    # A component with dim=k must contribute k raw state dims (all sharing the
    # component's scalar bounds), not a single dim. Mirrors quad2d/quad3d/humanoid,
    # where #components < state_dim.
    system = _FakeSystem(
        components=[
            _MultiDimComponent("position", "Real", 2),
            _MultiDimComponent("angle", "SO2", 1),
            _MultiDimComponent("velocity", "Real", 3),
        ],
        bounds={"position": (-1.0, 1.0), "velocity": (-5.0, 5.0)},
    )
    root = build_root(system)
    assert len(root.low) == 6           # 2 + 1 + 3, not 3 components
    # position dims share (-1, 1); SO2 dim is +-pi; velocity dims share (-5, 5)
    assert list(root.low) == [-1.0, -1.0, -np.pi, -5.0, -5.0, -5.0]
    assert list(root.high) == [1.0, 1.0, np.pi, 5.0, 5.0, 5.0]


def test_refine_respects_max_leaves_cap():
    system = PendulumSystem()
    # Everything straddles the boundary -> every leaf always wants to subdivide,
    # so without a cap the tree doubles each refine(). The cap must bound it.
    def latent_fn(X):
        return np.zeros(len(X)), np.full(len(X), 1.0)   # high variance, m=0 -> 'r'
    tree = PartitionTree(system, delta=1e-9, m_class=16, max_leaves=8)
    for _ in range(20):
        tree.refine(latent_fn)
    assert len(tree.leaves()) <= 8

    # Unbounded (default) keeps doubling well past 8 over the same schedule.
    tree_unbounded = PartitionTree(system, delta=1e-9, m_class=16)
    for _ in range(6):
        tree_unbounded.refine(latent_fn)
    assert len(tree_unbounded.leaves()) > 8
