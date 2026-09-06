from __future__ import annotations

import numpy as np

from adaptive_roa.partx.classify import classify_region
from adaptive_roa.partx.region import Region


def build_root(system, pad_frac: float = 1e-6) -> Region:
    """Full state-space support box in raw coords, with per-dim normalization scale.

    Bounds come from ``system.per_dim_bounds()``, which gives one (low, high) per
    RAW state dimension. The older path read ``state_bounds[comp.name]``, keyed by
    manifold COMPONENT, so a component covering several raw dims applied one
    scalar range to all of them. On quadrotor2D that gave theta_dot x_dot's
    +-1.303 against a true +-13.365, and gave z x's symmetric +-1.011 against a
    true [0.089, 1.510]; only 9% of the eval grid fell inside the resulting box.
    Everything outside is assigned leaf -1 by PartitionTree.assign and is
    therefore permanently ineligible for acquisition, which is why partx arms
    under-spent their budget (quad2D ~41%) or never acquired at all (quad3D
    noisy_dynamics f_0.048: 0 trajectories across all 18 epochs).

    ``pad_frac`` widens the box by a relative epsilon. Region.contains uses
    closed-interval comparisons on floats, and grids that store a bound as a
    rounded decimal land just outside it: the quadrotor2D eval grid stores
    theta = -3.141593, which is below -math.pi, so the entire theta = -pi column
    (1/12 of the grid) was excluded from a box built to cover exactly +-pi.
    """
    fn = getattr(system, "per_dim_bounds", None)
    if callable(fn):
        pairs = fn()
    else:
        # Duck-typed systems that predate per_dim_bounds: expand components as
        # before. Correct whenever every component is one-dimensional.
        pairs = []
        for comp in system.manifold_components:
            dim = int(getattr(comp, "dim", 1))
            if comp.manifold_type == "SO2":
                lo, hi = -np.pi, np.pi
            else:
                b = system.state_bounds[comp.name]
                lo, hi = float(b[0]), float(b[1])
            pairs.extend([(lo, hi)] * dim)
    low = np.array([p[0] for p in pairs], dtype=float)
    high = np.array([p[1] for p in pairs], dtype=float)
    span = np.maximum(high - low, 1e-9)
    pad = pad_frac * span
    low, high = low - pad, high + pad
    norm_scale = np.maximum(high - low, 1e-9)
    return Region(low, high, norm_scale, region_class="r", rid=0)


class PartitionTree:
    def __init__(self, system, branching_factor=2, delta=0.05, alpha=0.05,
                 m_class=64, seed=42, max_leaves=None):
        self.system = system
        self.branching_factor = int(branching_factor)
        self.delta = float(delta)
        self.alpha = float(alpha)
        self.m_class = int(m_class)
        # Optional cap on the number of leaves. In high dimensions region
        # classification rarely resolves, so every remaining leaf subdivides
        # each epoch and the leaf count grows ~branching_factor^epochs -> OOM.
        # None = unbounded (default; preserves low-dim behavior exactly).
        self.max_leaves = int(max_leaves) if max_leaves else None
        self.rng = np.random.default_rng(seed)
        self._next_id = 1
        self.root = build_root(system)
        self._leaves = [self.root]

    def leaves(self):
        return list(self._leaves)

    def remaining_leaves(self):
        return [r for r in self._leaves if r.region_class in ("r", "min")]

    def _terminal(self, region: Region) -> bool:
        return bool(np.all(region._norm_sides() < self.delta))

    def refine(self, latent_fn) -> None:
        new_leaves = []
        n_current = len(self._leaves)
        for i, leaf in enumerate(self._leaves):
            pts = leaf.sample_uniform(self.m_class, self.rng)
            m, s2 = latent_fn(pts)
            leaf.region_class = classify_region(m, s2, self.alpha)
            # Would subdividing this leaf push the total leaf count over the cap?
            # Account for children replacing this leaf plus every not-yet-processed
            # leaf that will still contribute at least one leaf.
            remaining_unprocessed = n_current - i - 1
            projected = len(new_leaves) + self.branching_factor + remaining_unprocessed
            at_cap = self.max_leaves is not None and projected > self.max_leaves
            if leaf.region_class == "r" and not self._terminal(leaf) and not at_cap:
                children = leaf.subdivide(self.branching_factor)
                for c in children:
                    c.rid = self._next_id; self._next_id += 1
                new_leaves.extend(children)
            else:
                if leaf.region_class == "r":
                    leaf.region_class = "min"   # terminal / capped remaining
                new_leaves.append(leaf)
        self._leaves = new_leaves

    def assign(self, X: np.ndarray):
        X = np.asarray(X)
        out = np.full(len(X), -1, dtype=int)
        for i, leaf in enumerate(self._leaves):
            out[leaf.contains(X)] = i
        return out.tolist()
