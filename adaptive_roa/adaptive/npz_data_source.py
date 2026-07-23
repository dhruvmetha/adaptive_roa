"""Npz-backed trajectory data source for the stochastic pendulum datasets.

The stochastic datasets (noisy/pendulum/lqr/{level}) store the training pool
as a single flat ``train.npz`` instead of per-trajectory text files:

- ``states``    (sumT, 2)  concatenated rollout states
- ``offsets``   (N+1,)     rollout boundaries into ``states``
- ``starts``    (N, 2)     initial state per rollout
- ``labels``    (N,)       binary success per rollout
- ``start_ids`` (N,)       unique-start index per rollout
- ``seeds``     (N,)       per-rollout seed

``shuffled_indices_{v}.txt`` lists integer rollout row ids (contiguous blocks
of 10 per start state, block order = shuffled start order) rather than
filenames. Pool index k therefore maps to npz row ``rollout_ids[k]``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from adaptive_roa.adaptive.data_source import (
    TrajectoryDataSource,
    TrajectoryDataSourceConfig,
)


class NpzTrajectoryDataSource(TrajectoryDataSource):
    """Drop-in replacement for TrajectoryDataSource reading a flat npz pool.

    ``config.trajectories_dir`` is interpreted as the path to the ``train.npz``
    file (or a directory containing one).
    """

    def __init__(self, config: TrajectoryDataSourceConfig):
        self.config = config
        npz_path = Path(config.trajectories_dir)
        if npz_path.is_dir():
            npz_path = npz_path / "train.npz"
        with np.load(npz_path) as z:
            # float32 to match the text loader's dtype and halve memory
            self._states = z["states"].astype(np.float32)
            self._offsets = z["offsets"].astype(np.int64)
            self._starts = z["starts"].astype(np.float32)

        self._load_shuffled_indices(config.shuffled_indices_file)
        if config.shuffled_labels_file:
            self._load_shuffled_labels(config.shuffled_labels_file)
        else:
            self.labels = None

        self.start_states = self._starts[self.rollout_ids]
        self._traj_cache = {}
        self._length_cache = {}

        print("NpzTrajectoryDataSource initialized:")
        print(f"  Trajectories: {self.n_trajectories}")
        print(f"  Pool file: {npz_path}")
        if self.labels is not None:
            n_success = np.sum(self.labels == 1)
            n_failure = np.sum(self.labels == -1)
            print(f"  Labels: {n_success} success, {n_failure} failure")

    def _load_shuffled_indices(self, filepath: str):
        """Shuffled-indices file holds integer rollout row ids, not filenames."""
        self.rollout_ids = np.loadtxt(filepath, dtype=np.int64, ndmin=1)
        n_rollouts = len(self._offsets) - 1
        if self.rollout_ids.max() >= n_rollouts or self.rollout_ids.min() < 0:
            raise ValueError(
                f"shuffled_indices ids out of range for npz pool with {n_rollouts} rollouts"
            )
        self.n_trajectories = len(self.rollout_ids)

    def load_trajectory(self, idx: int) -> np.ndarray:
        r = self.rollout_ids[idx]
        return self._states[self._offsets[r]:self._offsets[r + 1]]

    def get_trajectory_length(self, idx: int) -> int:
        r = self.rollout_ids[idx]
        return int(self._offsets[r + 1] - self._offsets[r])

    def get_state_at(self, idx: int, row: int) -> np.ndarray:
        return self.load_trajectory(idx)[row]

    def get_start_state(self, idx: int) -> np.ndarray:
        return self.start_states[idx]

    def get_end_state(self, idx: int) -> np.ndarray:
        return self.load_trajectory(idx)[-1]
