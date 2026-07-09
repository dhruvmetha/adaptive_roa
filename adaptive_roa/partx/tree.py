from __future__ import annotations

import numpy as np

from adaptive_roa.partx.classify import classify_region
from adaptive_roa.partx.region import Region


def build_root(system) -> Region:
    """Full state-space support box in raw coords, with per-dim normalization scale."""
    lows, highs = [], []
    for comp in system.manifold_components:
        if comp.manifold_type == "SO2":
            lows.append(-np.pi); highs.append(np.pi)
        else:
            b = system.state_bounds[comp.name]
            lows.append(b[0]); highs.append(b[1])
    low = np.array(lows, dtype=float)
    high = np.array(highs, dtype=float)
    norm_scale = np.maximum(high - low, 1e-9)
    return Region(low, high, norm_scale, region_class="r", rid=0)


class PartitionTree:
    def __init__(self, system, branching_factor=2, delta=0.05, alpha=0.05,
                 m_class=64, seed=42):
        self.system = system
        self.branching_factor = int(branching_factor)
        self.delta = float(delta)
        self.alpha = float(alpha)
        self.m_class = int(m_class)
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
        for leaf in self._leaves:
            pts = leaf.sample_uniform(self.m_class, self.rng)
            m, s2 = latent_fn(pts)
            leaf.region_class = classify_region(m, s2, self.alpha)
            if leaf.region_class == "r" and not self._terminal(leaf):
                children = leaf.subdivide(self.branching_factor)
                for c in children:
                    c.rid = self._next_id; self._next_id += 1
                new_leaves.extend(children)
            else:
                if leaf.region_class == "r":
                    leaf.region_class = "min"   # terminal remaining
                new_leaves.append(leaf)
        self._leaves = new_leaves

    def assign(self, X: np.ndarray):
        X = np.asarray(X)
        out = np.full(len(X), -1, dtype=int)
        for i, leaf in enumerate(self._leaves):
            out[leaf.contains(X)] = i
        return out.tolist()
