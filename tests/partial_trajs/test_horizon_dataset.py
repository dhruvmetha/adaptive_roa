"""Tests for the partial-trajectory HorizonDataset loader."""
from pathlib import Path

import numpy as np
import pytest

from adaptive_roa.partial_trajs.data.horizon_dataset import HorizonDataset

BASE = Path(
    "/common/users/shared/pracsys/genMoPlan/data_trajectories/partial_deterministic"
)
PENDULUM_T25 = BASE / "pendulum_lqr" / "pendulum_lqr_50k_T25"

pytestmark = pytest.mark.skipif(
    not PENDULUM_T25.exists(), reason="shared partial-trajectory dataset not available"
)


@pytest.fixture(scope="module")
def splits():
    train = HorizonDataset(PENDULUM_T25, split="train", val_fraction=0.2, seed=0)
    val = HorizonDataset(PENDULUM_T25, split="val", val_fraction=0.2, seed=0)
    return train, val


def test_split_is_trajectory_disjoint_and_covers_all_horizons(splits):
    train, val = splits
    assert set(train.traj_ids.tolist()).isdisjoint(val.traj_ids.tolist())
    total = len(np.load(PENDULUM_T25 / "train_splits" / "horizons.npy", mmap_mode="r"))
    assert len(train) + len(val) == total


def test_val_fraction_approximately_respected(splits):
    train, val = splits
    n_traj = len(train.traj_ids) + len(val.traj_ids)
    frac = len(val.traj_ids) / n_traj
    assert 0.15 < frac < 0.25


def test_item_matches_states_cache(splits):
    train, _ = splits
    for i in (0, len(train) // 2, len(train) - 1):
        item = train[i]
        rec = train.horizons[i]
        traj = train.states[str(int(rec["traj_id"]))]
        assert np.allclose(item["x_start"].numpy(), traj[int(rec["start"])])
        assert np.allclose(item["x_end"].numpy(), traj[int(rec["end"])])
        assert int(item["is_freeze"]) == int(rec["is_freeze"])
        assert int(item["label"]) == int(rec["label"])


def test_item_has_finite_scalar_motion(splits):
    train, _ = splits
    item = train[0]
    assert item["motion"].shape == ()
    assert np.isfinite(float(item["motion"]))


def test_split_is_deterministic_for_same_seed(splits):
    train, _ = splits
    again = HorizonDataset(PENDULUM_T25, split="train", val_fraction=0.2, seed=0)
    assert set(train.traj_ids.tolist()) == set(again.traj_ids.tolist())


def _gold_order_ids(n):
    """First ``n`` trajectory ids in gold/pool order (== ``train_pool.txt`` order),
    recovered as the first-occurrence order of ``traj_id`` blocks in horizons.npy."""
    hz = np.load(PENDULUM_T25 / "train_splits" / "horizons.npy", mmap_mode="r")
    tid = np.asarray(hz["traj_id"])
    _, first_idx = np.unique(tid, return_index=True)
    gold = tid[np.sort(first_idx)]
    return gold[:n]


def test_max_trajectories_caps_pool_to_first_n_gold_order():
    n = 50
    train = HorizonDataset(PENDULUM_T25, split="train", val_fraction=0.2, seed=0, max_trajectories=n)
    val = HorizonDataset(PENDULUM_T25, split="val", val_fraction=0.2, seed=0, max_trajectories=n)
    ids = set(train.traj_ids.tolist()) | set(val.traj_ids.tolist())
    assert len(ids) == n
    assert ids == set(_gold_order_ids(n).tolist())
    # train/val still trajectory-disjoint within the capped pool
    assert set(train.traj_ids.tolist()).isdisjoint(val.traj_ids.tolist())


def test_max_trajectories_none_uses_full_pool():
    train = HorizonDataset(PENDULUM_T25, split="train", val_fraction=0.2, seed=0, max_trajectories=None)
    val = HorizonDataset(PENDULUM_T25, split="val", val_fraction=0.2, seed=0, max_trajectories=None)
    ids = set(train.traj_ids.tolist()) | set(val.traj_ids.tolist())
    assert len(ids) == 5000


def test_max_trajectories_horizons_are_subset_of_full():
    capped = HorizonDataset(PENDULUM_T25, split="train", val_fraction=0.2, seed=0, max_trajectories=50)
    assert set(capped.traj_ids.tolist()).issubset(set(_gold_order_ids(50).tolist()))
