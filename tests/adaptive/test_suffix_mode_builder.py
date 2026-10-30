"""candidate_mode='suffix' end to end on a tiny npz pool (det paper D22, C20)."""
from __future__ import annotations

import json

import numpy as np
import pytest
from omegaconf import OmegaConf

from adaptive_roa.adaptive.data_source import TrajectoryDataSourceConfig
from adaptive_roa.adaptive.dataset_builder import MAX_ROWS, AdaptiveDatasetBuilder
from adaptive_roa.adaptive.npz_data_source import NpzTrajectoryDataSource
from adaptive_roa.adaptive_v2.pool.trajectory_pool import TrajectoryPool

H = 12
N = 40


def _pool_files(tmp_path):
    root = tmp_path / "pool"
    (root / "train_test_splits").mkdir(parents=True)
    rng = np.random.default_rng(0)
    lengths = [int(rng.integers(3, 9)) for _ in range(N)]
    lengths[5] = H + 1                         # one timeout (label 0, full horizon)
    labels = [int(rng.random() < 0.4) for _ in range(N)]
    labels[5] = 0
    trajs = [rng.normal(size=(n, 2)).astype(np.float32) for n in lengths]
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    np.savez(root / "train.npz", states=np.vstack(trajs), offsets=offsets,
             starts=np.stack([t[0] for t in trajs]), labels=np.array(labels, dtype=np.uint8))
    ids = list(range(N))
    (root / "train_test_splits" / "order.txt").write_text("\n".join(map(str, ids)) + "\n")
    (root / "train_test_splits" / "labels.txt").write_text("\n".join(str(labels[i]) for i in ids) + "\n")
    (root / "train_description.json").write_text(json.dumps({"horizon_steps": H}))
    return root, trajs, labels


def _ds_cfg(root):
    return dict(
        trajectories_dir=str(root / "train.npz"),
        shuffled_indices_file=str(root / "train_test_splits" / "order.txt"),
        shuffled_labels_file=str(root / "train_test_splits" / "labels.txt"),
        timeout_intermediates="drop",
        horizon_steps=H,
        pool_format="npz",
    )


def _builder(tmp_path, root, seed=3):
    cfg = _ds_cfg(root)
    cfg.pop("pool_format")
    ds = NpzTrajectoryDataSource(TrajectoryDataSourceConfig(**cfg))
    return AdaptiveDatasetBuilder(ds, str(tmp_path / "out"), val_ratio=0.2, test_ratio=0,
                                  candidate_mode="suffix", seed=seed)


def _rows(path):
    a = np.loadtxt(path, ndmin=2)
    return a if a.size else np.zeros((0, 4))


def test_suffix_pairs_start_at_the_picked_row_and_split_by_trajectory(tmp_path):
    root, trajs, labels = _pool_files(tmp_path)
    b = _builder(tmp_path, root)
    b.get_initial_training_set(10)
    _, cids = b.sample_candidates_without_marking(6)
    b.add_to_training_balanced(cids, source="acquisition")
    files = b.build_all_datasets("endpoint")
    train, val = _rows(files["train"]), _rows(files["val"])
    st = b._suffix
    expected_train = sum(len(trajs[t]) - 1 - st.suffix_start[t] for t in st.train_trajectories()
                         if t != 5) + (1 if 5 in st.train_trajectories() and st.suffix_start[5] == 0 else 0)
    assert len(train) == expected_train
    tr = {tuple(np.round(r, 5)) for r in train[:, :2]}
    va = {tuple(np.round(r, 5)) for r in val[:, :2]}
    assert not tr & va
    assert len(st.val_trajectories()) == int(len(st.traj_order) * 0.2)


def test_classification_rows_match_endpoint_pairs(tmp_path):
    root, _, labels = _pool_files(tmp_path)
    b = _builder(tmp_path, root)
    b.get_initial_training_set(10)
    _, cids = b.sample_candidates_without_marking(6)
    b.add_to_training_balanced(cids)
    ep = _rows(b.build_all_datasets("endpoint")["train"])
    cl = np.loadtxt(b.build_all_datasets("classification")["train"], ndmin=2)
    assert len(cl) == len(ep)
    np.testing.assert_allclose(cl[:, :2], ep[:, :2], atol=1e-6)
    s, _e, y = b.get_training_data()
    np.testing.assert_array_equal(cl[:, 2], (y == 1).astype(float))


def test_timeout_trajectory_only_ever_offered_at_row_zero(tmp_path):
    root, _, _ = _pool_files(tmp_path)
    b = _builder(tmp_path, root)
    for _ in range(30):
        _, cids = b.sample_candidates_without_marking(N)
        assert all(b.candidate_to_traj_row(c)[1] == 0 for c in cids if c // MAX_ROWS == 5)


def test_pool_counts_every_pick_as_a_rollout(tmp_path):
    root, _, _ = _pool_files(tmp_path)
    pool = TrajectoryPool(OmegaConf.create(_ds_cfg(root)), str(tmp_path / "out"), val_ratio=0.2, test_ratio=0,
                          candidate_mode="suffix", seed=3)
    pool.initialize(8)
    traj, row = 20, 1
    pool.add_to_training_balanced([traj * MAX_ROWS + row])
    pool.add_to_training_balanced([traj * MAX_ROWS + 0])   # superseding pick
    assert pool.train_size == 10
    s, idx = pool.sample_uniform(3)
    assert [i // MAX_ROWS for i in idx] == [8, 9, 10]


def test_uniform_fill_skips_trajectories_picked_this_round(tmp_path):
    # D25: the top-up after a short direct round must not land on a trajectory
    # the same round already picked mid-flight.
    root, _, _ = _pool_files(tmp_path)
    pool = TrajectoryPool(OmegaConf.create(_ds_cfg(root)), str(tmp_path / "out"), val_ratio=0.2, test_ratio=0,
                          candidate_mode="suffix", seed=3)
    pool.initialize(8)
    _, idx = pool.sample_uniform(3, exclude={9 * MAX_ROWS + 2})
    assert [i // MAX_ROWS for i in idx] == [8, 10, 11]
    assert all(i % MAX_ROWS == 0 for i in idx)


def test_state_round_trip_rebuilds_identical_datasets(tmp_path):
    root, _, _ = _pool_files(tmp_path)
    a = _builder(tmp_path, root)
    a.get_initial_training_set(10)
    _, cids = a.sample_candidates_without_marking(7)
    a.add_to_training_balanced(cids)
    a.save_state(str(tmp_path / "state.json"))
    files_a = a.build_all_datasets("endpoint")
    rows_a = _rows(files_a["train"]).copy()
    b = _builder(tmp_path / "b", root)
    b.load_state(str(tmp_path / "state.json"))
    np.testing.assert_allclose(_rows(b.build_all_datasets("endpoint")["train"]), rows_a)
    assert b.sample_candidates_without_marking(5)[1] == a.sample_candidates_without_marking(5)[1]


def test_suffix_refuses_a_test_slice(tmp_path):
    root, _, _ = _pool_files(tmp_path)
    cfg = _ds_cfg(root)
    cfg.pop("pool_format")
    ds = NpzTrajectoryDataSource(TrajectoryDataSourceConfig(**cfg))
    with pytest.raises(ValueError, match="test_ratio=0"):
        AdaptiveDatasetBuilder(ds, str(tmp_path / "o"), val_ratio=0.2, test_ratio=0.1,
                               candidate_mode="suffix")
