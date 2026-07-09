from __future__ import annotations

import numpy as np


class Region:
    def __init__(self, low, high, norm_scale, region_class="r",
                 rid=0, parent_id=-1, depth=0):
        self.low = np.asarray(low, dtype=float)
        self.high = np.asarray(high, dtype=float)
        self.norm_scale = np.asarray(norm_scale, dtype=float)
        self.region_class = region_class
        self.rid = rid
        self.parent_id = parent_id
        self.depth = depth

    def contains(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        return np.all((X >= self.low) & (X <= self.high), axis=1)

    def _norm_sides(self) -> np.ndarray:
        return (self.high - self.low) / self.norm_scale

    def volume(self) -> float:
        return float(np.prod(self._norm_sides()))

    def longest_dim(self) -> int:
        return int(np.argmax(self._norm_sides()))

    def subdivide(self, branching_factor=2, split_value=None):
        d = self.longest_dim()
        edges = np.linspace(self.low[d], self.high[d], branching_factor + 1)
        if split_value is not None and branching_factor == 2:
            edges = np.array([self.low[d], split_value, self.high[d]])
        children = []
        for i in range(branching_factor):
            lo, hi = self.low.copy(), self.high.copy()
            lo[d], hi[d] = edges[i], edges[i + 1]
            children.append(Region(lo, hi, self.norm_scale, region_class="r",
                                    parent_id=self.rid, depth=self.depth + 1))
        return children

    def sample_uniform(self, n, rng):
        return rng.uniform(self.low, self.high, size=(n, self.low.shape[0]))
