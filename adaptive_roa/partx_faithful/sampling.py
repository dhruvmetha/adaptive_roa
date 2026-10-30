"""Sample store, LHS, and the two samplers Part-X draws evaluations from (spec §3, §6.3)."""
from __future__ import annotations

import numpy as np

from adaptive_roa.partx_faithful.region import child_masks


class SampleStore:
    """Append-only store of evaluated points. Sample ids are row numbers."""

    def __init__(self, dim: int):
        self._X = np.empty((64, dim))
        self._y = np.empty(64)
        self._pool = np.empty(64, dtype=np.int64)
        self.n = 0

    def _grow(self, need: int) -> None:
        cap = len(self._y)
        while cap < need:
            cap *= 2
        if cap == len(self._y):
            return
        X = np.empty((cap, self._X.shape[1]))
        X[:self.n] = self._X[:self.n]
        y = np.empty(cap)
        y[:self.n] = self._y[:self.n]
        p = np.empty(cap, dtype=np.int64)
        p[:self.n] = self._pool[:self.n]
        self._X, self._y, self._pool = X, y, p

    def add(self, X, y, pool_idx) -> list:
        X = np.asarray(X, dtype=float)
        if X.size == 0:
            return []
        X = np.atleast_2d(X)
        y = np.asarray(y, dtype=float).reshape(-1)
        p = np.asarray(pool_idx, dtype=np.int64).reshape(-1)
        k = len(X)
        self._grow(self.n + k)
        self._X[self.n:self.n + k] = X
        self._y[self.n:self.n + k] = y
        self._pool[self.n:self.n + k] = p
        ids = list(range(self.n, self.n + k))
        self.n += k
        return ids

    def X(self, ids) -> np.ndarray:
        return self._X[np.asarray(ids, dtype=np.int64)]

    def y(self, ids) -> np.ndarray:
        return self._y[np.asarray(ids, dtype=np.int64)]

    def pool_idx(self, ids) -> np.ndarray:
        return self._pool[np.asarray(ids, dtype=np.int64)]


def lhs(n: int, low, high, rng) -> np.ndarray:
    """Latin hypercube design of n points in the box [low, high]."""
    low = np.asarray(low, dtype=float)
    high = np.asarray(high, dtype=float)
    d = len(low)
    if n <= 0:
        return np.empty((0, d))
    strata = np.stack([rng.permutation(n) for _ in range(d)], axis=1)
    u = (strata + rng.random((n, d))) / n
    return low + u * (high - low)


class PoolSampler:
    """Part-X evaluations drawn from the pre-simulated pool (spec §3, "Pool sampling").

    starts [N, d] and labels [N] (+1 success, -1 failure) are indexed by pool
    index, and `used` marks indices already evaluated. Each region carries the
    pool indices inside its box (region.pool_candidates), which are split along
    with the region, so a query never rescans the whole pool. Distances are
    Euclidean after dividing by `scale`, the root box's side lengths.
    """

    def __init__(self, starts, labels, used, scale):
        self.starts = np.asarray(starts, dtype=float)
        self.labels = np.asarray(labels, dtype=float)
        self.used = np.asarray(used, dtype=bool).copy()
        self.scale = np.asarray(scale, dtype=float)

    def mark_used(self, pool_idx) -> None:
        self.used[np.asarray(pool_idx, dtype=np.int64)] = True

    def init_root(self, region) -> None:
        inside = np.all((self.starts >= region.low) & (self.starts <= region.high), axis=1)
        region.pool_candidates = np.flatnonzero(inside)

    def split(self, parent, children, dim) -> None:
        cand = parent.pool_candidates
        masks = child_masks(self.starts[cand, dim], [(c.low, c.high) for c in children], dim)
        for child, m in zip(children, masks):
            child.pool_candidates = cand[m]
        parent.pool_candidates = None

    def _available(self, region) -> np.ndarray:
        cand = region.pool_candidates
        return cand[~self.used[cand]]

    def n_available(self, region) -> int:
        return int(len(self._available(region)))

    def snap_design(self, design, region) -> np.ndarray:
        """For each design point in order, the nearest unused pool start in the box. No repeats."""
        avail = self._available(region)
        design = np.atleast_2d(np.asarray(design, dtype=float))
        if len(avail) == 0 or len(design) == 0:
            return np.empty(0, dtype=np.int64)
        Z = self.starts[avail] / self.scale
        taken = np.zeros(len(avail), dtype=bool)
        chosen = []
        for q in design / self.scale:
            d2 = np.sum((Z - q) ** 2, axis=1)
            d2[taken] = np.inf
            j = int(np.argmin(d2))
            if not np.isfinite(d2[j]):
                break
            taken[j] = True
            chosen.append(int(avail[j]))
        return np.asarray(chosen, dtype=np.int64)

    def candidates(self, region, cap, rng):
        avail = self._available(region)
        if len(avail) > cap:
            avail = np.sort(rng.choice(avail, size=int(cap), replace=False))
        return self.starts[avail], avail

    def take(self, handles):
        idx = np.asarray(handles, dtype=np.int64).reshape(-1)
        if len(np.unique(idx)) != len(idx):
            raise RuntimeError("PoolSampler.take: duplicate pool index")
        if self.used[idx].any():
            raise RuntimeError("PoolSampler.take: pool index already used")
        self.used[idx] = True
        return self.starts[idx], self.labels[idx], idx


class StatePoolSampler(PoolSampler):
    """PoolSampler over every non-terminal state of the pool trajectories (det paper D22c).

    Under deterministic first-hitting dynamics a query at row t of trajectory k
    returns k's outcome, so every stored state is a legitimate evaluation point.
    The flat arrays list each trajectory's candidate rows contiguously, in row
    order (row 0 only for a timeout). Taking a state marks its row and every later
    row of the same trajectory used, since their outcomes are then known. One
    design batch never takes two states of one trajectory. Part-X still gets one
    labelled point per evaluation, as published.
    """

    def __init__(self, states, labels, traj_of, traj_start, n_rows, scale):
        # float32 on purpose: quad3d's pool is about 33M states.
        self.starts = np.asarray(states, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=float)
        self.traj_of = np.asarray(traj_of, dtype=np.int64)
        self.traj_start = np.asarray(traj_start, dtype=np.int64)
        self.n_rows = np.asarray(n_rows, dtype=np.int64)
        self.used = np.zeros(len(self.starts), dtype=bool)
        self.scale = np.asarray(scale, dtype=np.float32)

    def mark_used(self, pool_idx) -> None:
        for i in np.asarray(pool_idx, dtype=np.int64).reshape(-1):
            t = self.traj_of[i]
            self.used[i:self.traj_start[t] + self.n_rows[t]] = True

    def snap_design(self, design, region) -> np.ndarray:
        avail = self._available(region)
        design = np.atleast_2d(np.asarray(design, dtype=np.float32))
        if len(avail) == 0 or len(design) == 0:
            return np.empty(0, dtype=np.int64)
        Z = self.starts[avail] / self.scale
        owner = self.traj_of[avail]
        taken = np.zeros(len(avail), dtype=bool)
        chosen = []
        for q in design / self.scale:
            d2 = np.sum((Z - q) ** 2, axis=1)
            d2[taken] = np.inf
            j = int(np.argmin(d2))
            if not np.isfinite(d2[j]):
                break
            taken |= owner == owner[j]
            chosen.append(int(avail[j]))
        return np.asarray(chosen, dtype=np.int64)

    def take(self, handles):
        idx = np.asarray(handles, dtype=np.int64).reshape(-1)
        if len(np.unique(idx)) != len(idx):
            raise RuntimeError("StatePoolSampler.take: duplicate state index")
        if len(np.unique(self.traj_of[idx])) != len(idx):
            raise RuntimeError("StatePoolSampler.take: two states of one trajectory in one batch")
        if self.used[idx].any():
            raise RuntimeError("StatePoolSampler.take: state already used")
        self.mark_used(idx)
        return self.starts[idx].astype(float), self.labels[idx], idx


def state_pool_arrays(states, offsets, rollout_ids, labels, timeout):
    """Flat candidate arrays for StatePoolSampler from an npz pool.

    states/offsets are the npz's; rollout_ids maps pool index k to its npz row;
    labels and timeout are per pool index. Returns (states, labels, traj_of,
    traj_start, n_rows), with candidate rows 0..len-2 (row 0 only for timeouts and
    length-1 trajectories, whose start is still a rollout).
    """
    rollout_ids = np.asarray(rollout_ids, dtype=np.int64)
    lengths = np.diff(np.asarray(offsets, dtype=np.int64))[rollout_ids]
    n_rows = np.where(np.asarray(timeout, dtype=bool), 1, np.maximum(lengths - 1, 1))
    traj_start = np.concatenate([[0], np.cumsum(n_rows)[:-1]])
    traj_of = np.repeat(np.arange(len(rollout_ids)), n_rows)
    row_of = np.arange(int(n_rows.sum())) - np.repeat(traj_start, n_rows)
    flat = np.asarray(offsets, dtype=np.int64)[rollout_ids][traj_of] + row_of
    return (np.asarray(states)[flat].astype(np.float32), np.asarray(labels, dtype=float)[traj_of],
            traj_of, traj_start, n_rows)


class ContinuousSampler:
    """Reference-check sampler (spec §8.2): evaluates fn anywhere in the box."""

    def __init__(self, fn):
        self.fn = fn

    def mark_used(self, pool_idx) -> None:
        pass

    def init_root(self, region) -> None:
        pass

    def split(self, parent, children, dim) -> None:
        pass

    def n_available(self, region) -> int:
        return int(np.iinfo(np.int64).max)

    def snap_design(self, design, region) -> np.ndarray:
        return np.atleast_2d(np.asarray(design, dtype=float))

    def candidates(self, region, cap, rng):
        U = region.low + rng.random((int(cap), len(region.low))) * (region.high - region.low)
        return U, U

    def take(self, handles):
        X = np.asarray(handles, dtype=float)
        if X.size == 0:
            return np.empty((0, 0)), np.empty(0), np.empty(0, dtype=np.int64)
        X = np.atleast_2d(X)
        y = np.array([float(self.fn(x)) for x in X])
        return X, y, np.full(len(X), -1, dtype=np.int64)
