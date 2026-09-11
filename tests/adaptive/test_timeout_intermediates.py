"""Timeout trajectories must not pair their intermediate states with x_T.

A trajectory that ends by success or at a wall inside the collector horizon T
reaches that terminal state from every x_t within T steps, so (x_t, x_end) is a
valid T-step pair. A trajectory cut at T because time ran out is different: the
T-step future of x_t is x_{t+T}, which was never recorded. Pairing x_t with x_T
claims an outcome over T - t steps only.

`timeout_intermediates="drop"` keeps only (x_0, x_T) for those trajectories.
`"keep"` (the default) is the historical behaviour and must stay byte-identical,
because running jobs and resumed runs depend on it.
"""

import json

import numpy as np
import pytest
from omegaconf import OmegaConf

from adaptive_roa.adaptive.data_source import (
    TrajectoryDataSource,
    TrajectoryDataSourceConfig,
    resolve_horizon_steps,
)
from adaptive_roa.adaptive.npz_data_source import NpzTrajectoryDataSource

H = 5  # horizon in control steps -> a timeout trajectory has H + 1 rows

# (n_rows, label): success early, timeout, wall failure early, success on the last step
POOL = [(4, 1), (H + 1, 0), (3, 0), (H + 1, 1)]


def _make_npz_pool(tmp_path, pool=POOL, description=None, timeout_flags=None):
    """Write a tiny npz pool with the layout the stochastic datasets ship."""
    root = tmp_path / "level"
    (root / "train_test_splits").mkdir(parents=True)
    rng = np.random.default_rng(0)
    trajs = [rng.normal(size=(n, 2)).astype(np.float32) for n, _ in pool]
    offsets = np.concatenate([[0], np.cumsum([len(t) for t in trajs])]).astype(np.int64)
    arrays = dict(
        states=np.vstack(trajs),
        offsets=offsets,
        starts=np.stack([t[0] for t in trajs]).astype(np.float64),
        labels=np.array([lab for _, lab in pool], dtype=np.uint8),
        seeds=np.arange(len(pool), dtype=np.int64),
    )
    if timeout_flags is not None:
        arrays["timeout"] = np.asarray(timeout_flags, dtype=np.uint8)
    np.savez(root / "train.npz", **arrays)
    ids = list(range(len(pool)))
    (root / "train_test_splits" / "shuffled_indices_0.txt").write_text(
        "\n".join(str(i) for i in ids) + "\n")
    (root / "train_test_splits" / "shuffled_labels_0.txt").write_text(
        "\n".join(str(pool[i][1]) for i in ids) + "\n")
    if description is None:
        description = {"horizon": {"steps": H}}
    if description:
        (root / "train_description.json").write_text(json.dumps(description))
    return root, trajs


def _npz_source(root, **kw):
    cfg = TrajectoryDataSourceConfig(
        trajectories_dir=str(root / "train.npz"),
        shuffled_indices_file=str(root / "train_test_splits" / "shuffled_indices_0.txt"),
        shuffled_labels_file=str(root / "train_test_splits" / "shuffled_labels_0.txt"),
        **kw,
    )
    return NpzTrajectoryDataSource(cfg)


def _n_rows(ds, idx):
    starts, ends = ds.get_all_endpoint_pairs_from_trajectory(idx, mode="train")
    assert len(starts) == len(ends)
    return len(starts)


def test_keep_is_the_default_and_pairs_every_row(tmp_path):
    root, _ = _make_npz_pool(tmp_path)
    ds = _npz_source(root)
    assert [_n_rows(ds, i) for i in range(len(POOL))] == [n - 1 for n, _ in POOL]


def test_drop_keeps_only_the_start_row_of_a_timeout(tmp_path):
    root, trajs = _make_npz_pool(tmp_path)
    ds = _npz_source(root, timeout_intermediates="drop")
    # success early, timeout, wall failure, success on the horizon step
    assert [_n_rows(ds, i) for i in range(len(POOL))] == [3, 1, 2, H]
    starts, ends = ds.get_all_endpoint_pairs_from_trajectory(1, mode="train")
    np.testing.assert_array_equal(starts, trajs[1][:1])
    np.testing.assert_array_equal(ends, trajs[1][-1:])


def test_drop_leaves_test_mode_alone(tmp_path):
    root, trajs = _make_npz_pool(tmp_path)
    ds = _npz_source(root, timeout_intermediates="drop")
    starts, ends = ds.get_all_endpoint_pairs_from_trajectory(1, mode="test")
    np.testing.assert_array_equal(starts, trajs[1][:1])
    np.testing.assert_array_equal(ends, trajs[1][-1:])


def test_drop_with_a_later_start_row_on_a_timeout_is_empty(tmp_path):
    """Intermediate candidate mode can start a trajectory at row > 0; a timeout
    has no valid pair there at all."""
    root, _ = _make_npz_pool(tmp_path)
    ds = _npz_source(root, timeout_intermediates="drop")
    starts, ends = ds.get_all_endpoint_pairs_from_trajectory(1, mode="train", start_row=2)
    assert starts.shape == (0, 2) and ends.shape == (0, 2)


def test_drop_classification_rows_and_labels(tmp_path):
    root, _ = _make_npz_pool(tmp_path)
    ds = _npz_source(root, timeout_intermediates="drop")
    out = tmp_path / "clf.txt"
    n = ds.save_classification_dataset(list(range(len(POOL))), str(out), mode="train")
    rows = np.loadtxt(out, ndmin=2)
    assert n == len(rows) == 3 + 1 + 2 + H
    np.testing.assert_array_equal(rows[:, -1], [1, 1, 1, 0, 0, 0] + [1] * H)


def test_drop_endpoint_rows_match_classification_rows(tmp_path):
    root, _ = _make_npz_pool(tmp_path)
    ds = _npz_source(root, timeout_intermediates="drop")
    n = ds.save_endpoint_dataset(list(range(len(POOL))), str(tmp_path / "ep.txt"), mode="train")
    assert n == 3 + 1 + 2 + H


def test_horizon_read_from_top_level_key(tmp_path):
    root, _ = _make_npz_pool(tmp_path, description={"horizon_steps": H})
    ds = _npz_source(root, timeout_intermediates="drop")
    assert ds.horizon_steps == H
    assert _n_rows(ds, 1) == 1


def test_explicit_horizon_is_used_without_a_description(tmp_path):
    root, _ = _make_npz_pool(tmp_path, description={})
    ds = _npz_source(root, timeout_intermediates="drop", horizon_steps=H)
    assert _n_rows(ds, 1) == 1


def test_explicit_horizon_that_contradicts_the_dataset_raises(tmp_path):
    root, _ = _make_npz_pool(tmp_path)
    with pytest.raises(ValueError, match="horizon"):
        _npz_source(root, timeout_intermediates="drop", horizon_steps=H + 2)


def test_drop_without_any_horizon_raises(tmp_path):
    root, _ = _make_npz_pool(tmp_path, description={})
    with pytest.raises(ValueError, match="horizon"):
        _npz_source(root, timeout_intermediates="drop")


def test_trajectory_longer_than_the_horizon_raises(tmp_path):
    """A too-small horizon would silently call every long success a non-timeout
    and every long failure a timeout; the pool itself contradicts it."""
    root, _ = _make_npz_pool(tmp_path, description={"horizon": {"steps": H - 1}})
    with pytest.raises(ValueError, match="rows"):
        _npz_source(root, timeout_intermediates="drop")


def test_npz_timeout_flag_that_agrees_is_accepted(tmp_path):
    # flag marks every trajectory cut at the horizon, including the success on the last step
    root, _ = _make_npz_pool(tmp_path, timeout_flags=[0, 1, 0, 1])
    ds = _npz_source(root, timeout_intermediates="drop")
    assert [_n_rows(ds, i) for i in range(len(POOL))] == [3, 1, 2, H]


def test_npz_timeout_flag_that_disagrees_raises(tmp_path):
    root, _ = _make_npz_pool(tmp_path, timeout_flags=[0, 0, 1, 0])
    with pytest.raises(ValueError, match="timeout"):
        _npz_source(root, timeout_intermediates="drop")


def test_unknown_mode_raises(tmp_path):
    root, _ = _make_npz_pool(tmp_path)
    with pytest.raises(ValueError, match="timeout_intermediates"):
        _npz_source(root, timeout_intermediates="truncate")


def test_text_pool_drop_reads_the_horizon_beside_the_trajectories_dir(tmp_path):
    root = tmp_path / "level"
    traj_dir = root / "trajectories"
    traj_dir.mkdir(parents=True)
    rng = np.random.default_rng(1)
    names = []
    for i, (n, _) in enumerate(POOL):
        np.savetxt(traj_dir / f"sequence_{i}.txt", rng.normal(size=(n, 2)),
                   delimiter=",", fmt="%.6f")
        names.append(f"sequence_{i}.txt")
    (root / "idx.txt").write_text("\n".join(names) + "\n")
    (root / "lab.txt").write_text("\n".join(str(lab) for _, lab in POOL) + "\n")
    (root / "dataset_description.json").write_text(json.dumps({"horizon": {"steps": H}}))
    cfg = TrajectoryDataSourceConfig(
        trajectories_dir=str(traj_dir),
        shuffled_indices_file=str(root / "idx.txt"),
        shuffled_labels_file=str(root / "lab.txt"),
        timeout_intermediates="drop",
    )
    ds = TrajectoryDataSource(cfg)
    assert [_n_rows(ds, i) for i in range(len(POOL))] == [3, 1, 2, H]


def test_resolve_horizon_prefers_one_consistent_value(tmp_path):
    root, _ = _make_npz_pool(tmp_path)
    (root / "dataset_description.json").write_text(json.dumps({"horizon_steps": H}))
    cfg = TrajectoryDataSourceConfig(trajectories_dir=str(root / "train.npz"),
                                     shuffled_indices_file="unused")
    assert resolve_horizon_steps(cfg) == H
    (root / "dataset_description.json").write_text(json.dumps({"horizon_steps": H + 1}))
    with pytest.raises(ValueError, match="horizon"):
        resolve_horizon_steps(cfg)


def test_trajectory_pool_passes_the_setting_through(tmp_path):
    from adaptive_roa.adaptive_v2.pool.trajectory_pool import TrajectoryPool

    root, _ = _make_npz_pool(tmp_path)
    ds_cfg = OmegaConf.create(dict(
        pool_format="npz",
        trajectories_dir=str(root / "train.npz"),
        shuffled_indices_file=str(root / "train_test_splits" / "shuffled_indices_0.txt"),
        shuffled_labels_file=str(root / "train_test_splits" / "shuffled_labels_0.txt"),
        timeout_intermediates="drop",
    ))
    pool = TrajectoryPool(ds_cfg, output_dir=str(tmp_path / "out"), val_ratio=0.2, test_ratio=0.0)
    ds = pool.dataset_builder.data_source
    assert ds.timeout_intermediates == "drop" and ds.horizon_steps == H
    assert _n_rows(ds, 1) == 1

    # A config written before the key existed (every resumed legacy run) keeps every row.
    legacy = OmegaConf.create({k: v for k, v in ds_cfg.items() if k != "timeout_intermediates"})
    pool = TrajectoryPool(legacy, output_dir=str(tmp_path / "out2"), val_ratio=0.2, test_ratio=0.0)
    assert _n_rows(pool.dataset_builder.data_source, 1) == H
