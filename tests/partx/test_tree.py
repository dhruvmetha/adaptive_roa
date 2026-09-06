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


def test_build_root_expands_multidim_components_without_per_dim_bounds():
    # Duck-typed systems with no per_dim_bounds() keep the legacy expansion: a
    # component with dim=k contributes k raw dims sharing the component's scalar
    # bounds. Correct whenever every component is one-dimensional.
    system = _FakeSystem(
        components=[
            _MultiDimComponent("position", "Real", 2),
            _MultiDimComponent("angle", "SO2", 1),
            _MultiDimComponent("velocity", "Real", 3),
        ],
        bounds={"position": (-1.0, 1.0), "velocity": (-5.0, 5.0)},
    )
    root = build_root(system, pad_frac=0.0)
    assert len(root.low) == 6           # 2 + 1 + 3, not 3 components
    assert list(root.low) == [-1.0, -1.0, -np.pi, -5.0, -5.0, -5.0]
    assert list(root.high) == [1.0, 1.0, np.pi, 5.0, 5.0, 5.0]


class _PerDimSystem(_FakeSystem):
    def __init__(self, components, bounds, per_dim):
        super().__init__(components, bounds)
        self._per_dim = per_dim

    def per_dim_bounds(self):
        return self._per_dim


def test_build_root_prefers_per_dim_bounds():
    # When the system exposes true per-axis extents, build_root must use them
    # rather than the component-collapsed state_bounds. Collapsing is what gave
    # quadrotor2D's theta_dot x_dot's +-1.303 against a true +-13.365 and left
    # 91% of the state space outside the partition root, hence ineligible for
    # acquisition.
    system = _PerDimSystem(
        components=[_MultiDimComponent("velocity", "Real", 3)],
        bounds={"velocity": (-1.3, 1.3)},
        per_dim=[(-1.3, 1.3), (-1.4, 1.4), (-13.4, 13.4)],
    )
    root = build_root(system, pad_frac=0.0)
    assert list(root.low) == [-1.3, -1.4, -13.4]
    assert list(root.high) == [1.3, 1.4, 13.4]
    # the wide axis is now covered; under the collapsed bounds it was not
    assert root.contains(np.array([[0.0, 0.0, 9.0]]))[0]


def test_build_root_pads_closed_interval_boundary():
    # Grids that store a bound as a rounded decimal land just outside a box built
    # to exactly that bound: the quadrotor2D eval grid stores theta = -3.141593,
    # below -np.pi, which excluded 1/12 of the grid from a +-pi root.
    system = _PerDimSystem(
        components=[_MultiDimComponent("angle", "SO2", 1)],
        bounds={}, per_dim=[(-np.pi, np.pi)],
    )
    just_outside = np.array([[-3.141593]])
    assert not build_root(system, pad_frac=0.0).contains(just_outside)[0]
    assert build_root(system).contains(just_outside)[0]


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
