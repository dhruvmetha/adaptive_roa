"""Intermediate trajectory states for partx_faithful_traj (spec 2026-09-15 §4).

The neural arms train on traj[0:-1] of every rollout with the rollout's label, and on x0
only for a timeout (TrajectoryDataSource.get_all_endpoint_pairs_from_trajectory). Part-X
already holds x0, so this supplies traj[1:-1], using the data source's own timeout rule so
the two arms can never disagree about which rollouts count.
"""
from __future__ import annotations

import numpy as np


class PoolTrajectoryStates:
    def __init__(self, data_source):
        self.ds = data_source
        self.dim = int(np.asarray(data_source.start_states).shape[1])

    def states(self, pool_idx: int):
        """(rows, timed_out): the states after x0 and before x_T; none for a timeout."""
        p = int(pool_idx)
        n_rows = int(self.ds.get_trajectory_length(p))
        if self.ds._drops_intermediates(p, n_rows):
            return np.empty((0, self.dim)), True
        traj = np.asarray(self.ds.load_trajectory(p), dtype=float)
        return traj[1:-1], False
