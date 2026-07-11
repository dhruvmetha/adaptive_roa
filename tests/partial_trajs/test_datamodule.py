"""Tests for the HorizonDataModule."""
from pathlib import Path

import pytest

from adaptive_roa.partial_trajs.data.datamodule import HorizonDataModule

BASE = Path(
    "/common/users/shared/pracsys/genMoPlan/data_trajectories/partial_deterministic"
)
PENDULUM_T25 = BASE / "pendulum_lqr" / "pendulum_lqr_50k_T25"

pytestmark = pytest.mark.skipif(
    not PENDULUM_T25.exists(), reason="shared partial-trajectory dataset not available"
)


@pytest.fixture(scope="module")
def dm():
    dm = HorizonDataModule(PENDULUM_T25, batch_size=64, val_fraction=0.2, seed=0)
    dm.setup()
    return dm


def test_dataloaders_are_nonempty(dm):
    assert len(dm.train_dataloader()) > 0
    assert len(dm.val_dataloader()) > 0


def test_batch_has_expected_keys_and_shapes(dm):
    batch = next(iter(dm.train_dataloader()))
    for key in ("x_start", "x_end", "is_freeze", "label", "motion"):
        assert key in batch
    assert batch["x_start"].shape == (64, 2)
    assert batch["x_end"].shape == (64, 2)
    assert batch["motion"].shape == (64,)


def test_train_and_val_trajectories_are_disjoint(dm):
    assert set(dm.train_ds.traj_ids.tolist()).isdisjoint(dm.val_ds.traj_ids.tolist())


def test_max_trajectories_is_plumbed_to_datasets():
    dm = HorizonDataModule(PENDULUM_T25, batch_size=64, val_fraction=0.2, seed=0, max_trajectories=40)
    dm.setup()
    ids = set(dm.train_ds.traj_ids.tolist()) | set(dm.val_ds.traj_ids.tolist())
    assert len(ids) == 40


def test_iid_horizons_is_plumbed_to_datasets():
    clustered = HorizonDataModule(PENDULUM_T25, batch_size=64, val_fraction=0.2, seed=0, max_trajectories=40)
    clustered.setup()
    k = len(clustered.train_ds) + len(clustered.val_ds)
    iid = HorizonDataModule(PENDULUM_T25, batch_size=64, val_fraction=0.2, seed=0, max_trajectories=40, iid_horizons=True)
    iid.setup()
    # same horizon budget, but spread over many more trajectories
    assert len(iid.train_ds) + len(iid.val_ds) == k
    ids = set(iid.train_ds.traj_ids.tolist()) | set(iid.val_ds.traj_ids.tolist())
    assert len(ids) > 40
