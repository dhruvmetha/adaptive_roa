"""Region boxes and the Part-X region types (spec §3)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

RTYPES = ("r", "r+", "r-", "+", "-", "u", "i")
REMAINING = ("r", "r+", "r-")
CLASSIFIED = ("+", "-")


@dataclass
class Region:
    rid: int
    parent: int
    low: np.ndarray
    high: np.ndarray
    branch_dir: int
    depth: int
    rtype: str = "r"
    samples: list = field(default_factory=list)       # sample ids into the SampleStore
    children: list = field(default_factory=list)      # child region ids
    pool_candidates: np.ndarray | None = None          # pool indices inside the box (PoolSampler)

    def volume(self) -> float:
        return float(np.prod(self.high - self.low))

    def contains(self, X) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        return np.all((X >= self.low) & (X <= self.high), axis=1)

    def split_bounds(self, dim: int, B: int = 2) -> list:
        """B equal cuts along `dim` (cpslab branch_region with uniform=True)."""
        edges = self.low[dim] + (np.arange(B + 1) / B) * (self.high[dim] - self.low[dim])
        out = []
        for k in range(B):
            lo, hi = self.low.copy(), self.high.copy()
            lo[dim], hi[dim] = edges[k], edges[k + 1]
            out.append((lo, hi))
        return out


def child_masks(x: np.ndarray, bounds: list, dim: int) -> list:
    """Which child each coordinate x (along `dim`) belongs to.

    Child k takes lo_k <= x < hi_k, except that the first child has no lower
    limit and the last no upper limit. So every point lands in exactly one
    child, including points on an edge or just outside the parent box.
    """
    x = np.asarray(x, dtype=float)
    B = len(bounds)
    masks = []
    for k, (lo, hi) in enumerate(bounds):
        m = np.ones(len(x), dtype=bool)
        if k > 0:
            m &= x >= lo[dim]
        if k < B - 1:
            m &= x < hi[dim]
        masks.append(m)
    return masks
