"""The npz pool round-trips through NpzTrajectoryDataSource, timeouts stay out of pool orders."""
import numpy as np
import pytest

from adaptive_roa.adaptive.data_source import TrajectoryDataSourceConfig
from adaptive_roa.adaptive.npz_data_source import NpzTrajectoryDataSource
from adaptive_roa.det_rerun.pool_npz import parse_rows, write_pool_npz


def _pool(tmp_path, horizon=9):
    rng = np.random.default_rng(0)
    arrays = [rng.uniform(-3, 3, size=(int(rng.integers(2, 8)), 2)).astype(np.float32) for _ in range(12)]
    arrays.append(rng.uniform(-3, 3, size=(horizon + 1, 2)).astype(np.float32))  # a timeout, full length
    names = [f"sequence_{i}.txt" for i in range(13)]
    success = [i % 2 for i in range(12)] + [0]
    timeout = [False] * 12 + [True]
    desc = write_pool_npz(tmp_path, names, arrays, success, timeout, seeds=(5, 6), horizon_steps=horizon)
    return arrays, success, desc


def _load(tmp_path, seed=5, mode="drop"):
    return NpzTrajectoryDataSource(TrajectoryDataSourceConfig(
        trajectories_dir=str(tmp_path / "train.npz"),
        shuffled_indices_file=str(tmp_path / f"pool_order_{seed}.txt"),
        shuffled_labels_file=str(tmp_path / f"pool_labels_{seed}.txt"),
        timeout_intermediates=mode))


def test_round_trip_matches_arrays(tmp_path):
    arrays, success, desc = _pool(tmp_path)
    src = _load(tmp_path)
    ids = src.rollout_ids
    assert 12 not in ids.tolist() and len(ids) == 12      # the timeout row is never in the pool
    for k in range(len(ids)):
        assert np.array_equal(src.load_trajectory(k), arrays[ids[k]])
        assert src.get_label(k) == (1 if success[ids[k]] else -1)
    assert desc["n_pool"] == 12 and desc["n_timeout_rows_not_in_pool"] == 1


def test_seed_changes_order(tmp_path):
    _pool(tmp_path)
    assert not np.array_equal(_load(tmp_path, 5).rollout_ids, _load(tmp_path, 6).rollout_ids)


def test_rejects_rows_beyond_horizon(tmp_path):
    with pytest.raises(ValueError, match="horizon"):
        write_pool_npz(tmp_path, ["a"], [np.zeros((5, 2), np.float32)], [0], [False], seeds=(1,), horizon_steps=3)


def test_parse_rows_matches_text_loader_cast():
    text = "-1.34159,-3.76318\n2.0957,0.000325657\n"
    expect = np.array([[float(v) for v in ln.split(",")] for ln in text.split("\n") if ln], dtype=np.float32)
    assert np.array_equal(parse_rows(text), expect)
