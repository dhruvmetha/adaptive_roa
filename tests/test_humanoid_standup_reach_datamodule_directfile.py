import numpy as np
import torch
import pytest
from adaptive_roa.data.humanoid_standup_reach_endpoint_data import HumanoidStandUpReachEndpointDataModule


def _write_pairs(path, n, seed):
    rng = np.random.default_rng(seed)
    starts = rng.normal(size=(n, 67)); ends = rng.normal(size=(n, 67))
    np.savetxt(path, np.hstack([starts, ends]), fmt="%.8f")  # space-separated, 134 cols (pool format)


def test_direct_file_mode_loads_start_end(tmp_path):
    f = tmp_path / "train.txt"; _write_pairs(f, 8, 0)
    dm = HumanoidStandUpReachEndpointDataModule(
        data_file=str(f), validation_file=str(f), test_file=str(f),
        batch_size=4, num_workers=0)
    dm.setup("fit")
    item = dm.train_dataset[0]
    assert item["start_state"].shape == (67,) and item["end_state"].shape == (67,)
    assert torch.allclose(item["start_state"][34:37].norm(), torch.tensor(1.0), atol=1e-5)  # sphere unit-norm


def test_datamodule_registered_in_trainer():
    from adaptive_roa.adaptive_v2.trainers.flow_matching_trainer import _DATAMODULES
    assert _DATAMODULES["humanoid_standup_reach"] is HumanoidStandUpReachEndpointDataModule
