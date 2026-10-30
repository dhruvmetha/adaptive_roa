"""Part-X over the suffix state pool (det paper D22c, tracker C23)."""
import numpy as np
import pytest

from adaptive_roa.partx_faithful.region import Region
from adaptive_roa.partx_faithful.sampling import StatePoolSampler, state_pool_arrays


def _pool():
    # three trajectories: lengths 4, 1, 6; the third is a timeout
    lengths = [4, 1, 6]
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    states = np.arange(offsets[-1] * 2, dtype=np.float32).reshape(-1, 2) / 10.0
    rollout_ids = np.array([0, 1, 2])
    return state_pool_arrays(states, offsets, rollout_ids, labels=[1, -1, -1],
                             timeout=[False, False, True]), states, offsets


def test_flat_arrays_list_candidate_rows_per_trajectory():
    (S, y, traj_of, start, rows), states, offsets = _pool()
    np.testing.assert_array_equal(rows, [3, 1, 1])
    np.testing.assert_array_equal(start, [0, 3, 4])
    np.testing.assert_array_equal(traj_of, [0, 0, 0, 1, 2])
    np.testing.assert_allclose(S, states[[0, 1, 2, 4, 5]])
    np.testing.assert_array_equal(y, [1, 1, 1, -1, -1])


def _sampler():
    (S, y, traj_of, start, rows), _, _ = _pool()
    return StatePoolSampler(S, y, traj_of, start, rows, scale=np.ones(2))


def test_taking_a_state_retires_its_tail_only():
    sp = _sampler()
    sp.take([1])                                   # trajectory 0, row 1
    np.testing.assert_array_equal(sp.used, [False, True, True, False, False])


def test_one_batch_never_takes_two_states_of_one_trajectory():
    sp = _sampler()
    with pytest.raises(RuntimeError, match="two states of one trajectory"):
        sp.take([0, 2])


def test_snap_design_picks_at_most_one_state_per_trajectory():
    sp = _sampler()
    region = Region(rid=0, parent=-1, low=np.array([-1.0, -1.0]), high=np.array([10.0, 10.0]), branch_dir=0, depth=0)
    sp.init_root(region)
    S = sp.starts
    chosen = sp.snap_design(S[[0, 1, 2]], region)   # three points on trajectory 0
    assert len(set(sp.traj_of[chosen])) == len(chosen)
    assert sp.traj_of[chosen[0]] == 0


def test_used_states_are_never_offered():
    sp = _sampler()
    region = Region(rid=0, parent=-1, low=np.array([-1.0, -1.0]), high=np.array([10.0, 10.0]), branch_dir=0, depth=0)
    sp.init_root(region)
    sp.mark_used([0])
    assert sp.n_available(region) == 2
