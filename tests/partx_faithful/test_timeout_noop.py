"""partx_faithful is a control arm for the timeout fix (f16314f).

The fix drops the intermediate rows of timeout trajectories, because pairing x_t
with x_T claims an outcome over T - t steps. It reaches those rows only through
`TrajectoryDataSource.get_all_endpoint_pairs_from_trajectory`.

Part-X never calls that. It samples START states and asks the pool for their
trajectory's outcome, so it consumes exactly `start_states` and `labels`. Both are
built per trajectory and neither depends on the intermediate rows, so
`timeout_intermediates=drop` cannot move this arm's inputs.

That makes partx_faithful a stronger control than a level with no timeouts: a cell
where nothing times out only shows the fix is inert where it was never going to act.
This asserts the inputs are identical on the pool where the fix bites hardest,
pendulum lqr med, where 59,401 of 100,000 trajectories time out and 89% of pool rows
are discarded. If a partx_faithful number moves between the pre-fix and post-fix
campaigns, the cause is the schedule or the seed, and this test is the evidence.
"""
import os

import numpy as np
import pytest

from adaptive_roa.adaptive.data_source import TrajectoryDataSourceConfig
from adaptive_roa.adaptive.npz_data_source import NpzTrajectoryDataSource
from adaptive_roa.utils.env_config import get_data_dir

_POOL = os.path.join(get_data_dir(), "stochastic/pendulum/gaussian_signal/lqr/med")


def _source(mode):
    return NpzTrajectoryDataSource(TrajectoryDataSourceConfig(
        trajectories_dir=_POOL,
        shuffled_indices_file=os.path.join(_POOL, "train_test_splits/shuffled_indices_0.txt"),
        shuffled_labels_file=os.path.join(_POOL, "train_test_splits/shuffled_labels_0.txt"),
        timeout_intermediates=mode))


@pytest.mark.skipif(not os.path.isdir(_POOL), reason="needs the pendulum lqr med pool")
def test_timeout_fix_does_not_move_what_partx_faithful_reads():
    keep, drop = _source("keep"), _source("drop")

    # The premise: this pool is where the fix does the most work. If the share ever
    # drops, the control is weaker than the docstring claims and should be re-sited.
    n_timeout = int(np.sum(np.asarray(keep.labels) != 1))
    assert n_timeout > 0.5 * len(keep.labels)

    assert np.array_equal(np.asarray(keep.start_states), np.asarray(drop.start_states))
    assert np.array_equal(np.asarray(keep.labels), np.asarray(drop.labels))

    # The line to cite: the label balance is untouched by a mode that discards 89%
    # of the pool's rows.
    balance = lambda ds: np.bincount((np.asarray(ds.labels) + 1).astype(int))  # noqa: E731
    assert np.array_equal(balance(keep), balance(drop))
