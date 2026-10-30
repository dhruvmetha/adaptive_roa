"""Bookkeeping for acquisition over intermediate states ("suffix" candidate mode).

Under deterministic first-hitting dynamics, querying row t of trajectory tau
returns tau's suffix from t, so a pick of (tau, t) adds tau[t:] to training.
Rules (papers/det_paper/EXPERIMENT_TRACKER.md, D22 and C20):

- One stored suffix per trajectory: picking an earlier row of a trajectory
  replaces its suffix with the longer one, never a second copy.
- Rows at or after a trajectory's suffix start are labelled and leave the
  candidate set.
- A timeout trajectory offers only row 0: a suffix from a later row observes
  fewer steps than the horizon, so its outcome is not a fresh rollout.
- Every pick costs one rollout, including one later superseded by an earlier
  row of the same trajectory.
- The train/val assignment is fixed per trajectory at its first pick, so a
  replacement never moves a trajectory across the split.
- Acquisition candidates are a fresh random draw each call, at most one per
  trajectory. The uniform share takes the next trajectory starts in pool order,
  so a uniform run reaches the same data at a given budget whatever its step
  size.

This module holds only indices and rows; it never touches trajectory files.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np

Pick = Tuple[int, int, str]


class SuffixPoolState:
    def __init__(
        self,
        n_trajectories: int,
        val_ratio: float,
        seed: int,
        length_fn: Callable[[int], int],
        is_timeout_fn: Callable[[int], bool],
    ):
        if not 0.0 <= val_ratio < 1.0:
            raise ValueError(f"val_ratio must be in [0, 1), got {val_ratio}")
        self.n_trajectories = int(n_trajectories)
        self.val_ratio = float(val_ratio)
        self.seed = int(seed)
        self._length_fn = length_fn
        self._is_timeout_fn = is_timeout_fn
        self.suffix_start: Dict[int, int] = {}
        self.full = np.zeros(self.n_trajectories, dtype=bool)
        self.pick_log: List[Pick] = []
        self.traj_order: List[int] = []
        self.is_val: Dict[int, bool] = {}
        self._draws_since_pick = 0
        self._avail_cache: Dict[int, int] = {}

    # ------------------------------------------------------------------ rows
    def _untouched_rows(self, traj: int) -> int:
        """Rows 0..k-1 a fresh trajectory offers; the terminal row is excluded."""
        if traj not in self._avail_cache:
            if self._is_timeout_fn(traj):
                k = 1
            else:
                # A length-1 trajectory starts terminal; its start is still a
                # legitimate (uninformative) rollout, as in start mode.
                k = max(self._length_fn(traj) - 1, 1)
            self._avail_cache[traj] = k
        return self._avail_cache[traj]

    def available_rows(self, traj: int) -> int:
        if traj in self.suffix_start:
            return self.suffix_start[traj]
        return self._untouched_rows(traj)

    # ----------------------------------------------------------------- picks
    def _val_at(self, k: int) -> bool:
        # Exactly floor(n * r) of the first n distinct trajectories are val.
        r = self.val_ratio
        return math.floor((k + 1) * r + 1e-9) > math.floor(k * r + 1e-9)

    def record_picks(self, picks: Sequence[Tuple[int, int]], source: str) -> None:
        trajs = [t for t, _ in picks]
        if len(set(trajs)) != len(trajs):
            raise ValueError("one batch picked two states of the same trajectory")
        for traj, row in picks:
            traj, row = int(traj), int(row)
            if not 0 <= traj < self.n_trajectories:
                raise ValueError(f"trajectory {traj} outside the pool of {self.n_trajectories}")
            if not 0 <= row < self.available_rows(traj):
                raise ValueError(
                    f"row {row} of trajectory {traj} is not available "
                    f"(available rows 0..{self.available_rows(traj) - 1})")
            if traj not in self.is_val:
                self.is_val[traj] = self._val_at(len(self.traj_order))
                self.traj_order.append(traj)
            self.suffix_start[traj] = row
            if row == 0:
                self.full[traj] = True
            self.pick_log.append((traj, row, str(source)))
        self._draws_since_pick = 0

    @property
    def n_picks(self) -> int:
        return len(self.pick_log)

    def train_trajectories(self) -> List[int]:
        return [t for t in self.traj_order if not self.is_val[t]]

    def val_trajectories(self) -> List[int]:
        return [t for t in self.traj_order if self.is_val[t]]

    # ----------------------------------------------------------------- draws
    def _eligible(self, exclude_trajs: Iterable[int]) -> np.ndarray:
        mask = ~self.full
        ex = np.fromiter((int(t) for t in exclude_trajs), dtype=np.int64)
        if ex.size:
            mask = mask.copy()
            mask[ex] = False
        return np.flatnonzero(mask)

    def draw_candidates(self, n: int, exclude_trajs: Iterable[int] = ()) -> List[Tuple[int, int]]:
        """Random candidates, at most one per trajectory, seeded by the pick count.

        The seed is (seed, picks so far, draws since the last pick), so a resumed
        run that restores the pick log draws the same candidates.
        """
        if n <= 0:
            return []
        eligible = self._eligible(exclude_trajs)
        rng = np.random.default_rng([self.seed, self.n_picks, self._draws_since_pick])
        self._draws_since_pick += 1
        if eligible.size == 0:
            return []
        trajs = rng.choice(eligible, size=min(n, eligible.size), replace=False)
        return [(int(t), int(rng.integers(self.available_rows(int(t))))) for t in trajs]

    def next_uniform_starts(self, n: int, exclude_trajs: Iterable[int] = ()) -> List[Tuple[int, int]]:
        """Row 0 of the next trajectories in pool order whose start is unused."""
        if n <= 0:
            return []
        eligible = self._eligible(exclude_trajs)
        return [(int(t), 0) for t in eligible[:n]]

    # ----------------------------------------------------------------- state
    def to_dict(self) -> dict:
        return {
            "mode": "suffix",
            "n_trajectories": self.n_trajectories,
            "val_ratio": self.val_ratio,
            "seed": self.seed,
            "pick_log": [list(p) for p in self.pick_log],
        }

    def restore(self, state: dict) -> None:
        """Rebuild by replaying the pick log in order."""
        for key in ("n_trajectories", "val_ratio", "seed"):
            if state[key] != getattr(self, key):
                raise ValueError(f"saved {key}={state[key]} does not match this pool ({getattr(self, key)})")
        self.__init__(self.n_trajectories, self.val_ratio, self.seed,
                      self._length_fn, self._is_timeout_fn)
        for traj, row, source in state["pick_log"]:
            # Replay one pick at a time: the batch rule only constrains a call.
            self.record_picks([(traj, row)], source)
