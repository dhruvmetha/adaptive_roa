from __future__ import annotations

import numpy as np

from adaptive_roa.partx.classify import classify_region
from adaptive_roa.partx.region import Region


def build_root(system) -> Region:
    """Full state-space support box in raw coords, with per-dim normalization scale.

    Manifold components can be multi-dimensional (e.g. R^2 position, R^3 velocity,
    SO3 quaternion with dim=4). Each component is expanded into ``comp.dim`` raw
    state dimensions, all sharing the component's scalar bounds, so the region
    dimensionality matches the true state_dim (not the number of components).
    """
    lows, highs = [], []
    for comp in system.manifold_components:
        dim = int(getattr(comp, "dim", 1))
        if comp.manifold_type == "SO2":
            lo, hi = -np.pi, np.pi
        else:
            b = system.state_bounds[comp.name]
            lo, hi = float(b[0]), float(b[1])
        for _ in range(dim):
            lows.append(lo); highs.append(hi)
    low = np.array(lows, dtype=float)
    high = np.array(highs, dtype=float)
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
