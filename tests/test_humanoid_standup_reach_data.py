# tests/test_humanoid_standup_reach_data.py
import numpy as np
import torch
import pytest

from adaptive_roa.data.humanoid_standup_reach_endpoint_data import (
    HumanoidStandUpReachEndpointDataset,
)


def _make_traj(path, n_rows, seed):
    rng = np.random.default_rng(seed)
    rows = rng.normal(size=(n_rows, 67)).astype(np.float64)
    # make sphere block non-unit on purpose to test projection
    rows[:, 34:37] *= 5.0
    np.savetxt(path, rows, delimiter=", ", fmt="%.6f")  # comma+space like the dataset
    return rows


@pytest.fixture
def tiny_dataset(tmp_path):
    traj_dir = tmp_path / "trajectories"
    traj_dir.mkdir()
    raws = {}
    for i in range(3):
        raws[f"sequence_{i}.txt"] = _make_traj(traj_dir / f"sequence_{i}.txt", n_rows=5 + i, seed=i)
    idx = tmp_path / "shuffled_indices_0.txt"
    idx.write_text("\n".join(raws.keys()) + "\n")
    return str(idx), str(traj_dir), raws


def test_start_mode_returns_first_and_last_row(tiny_dataset):
    idx, traj_dir, raws = tiny_dataset
    ds = HumanoidStandUpReachEndpointDataset(shuffled_indices_file=idx, trajectories_dir=traj_dir, query_mode="start")
    item = ds[0]
    raw = raws["sequence_0.txt"]
    # end_state == final row (denormalized euclidean dims match raw exactly)
    assert torch.allclose(item["end_state"][:34], torch.tensor(raw[-1, :34], dtype=torch.float32), atol=1e-4)
    # start_state euclidean part == first row
    assert torch.allclose(item["start_state"][:34], torch.tensor(raw[0, :34], dtype=torch.float32), atol=1e-4)


def test_sphere_block_is_unit_norm(tiny_dataset):
    idx, traj_dir, _ = tiny_dataset
    ds = HumanoidStandUpReachEndpointDataset(shuffled_indices_file=idx, trajectories_dir=traj_dir, query_mode="start")
    item = ds[1]
    assert torch.allclose(item["start_state"][34:37].norm(), torch.tensor(1.0), atol=1e-5)
    assert torch.allclose(item["end_state"][34:37].norm(), torch.tensor(1.0), atol=1e-5)


def test_random_intermediate_query_is_non_terminal(tiny_dataset):
    idx, traj_dir, raws = tiny_dataset
    ds = HumanoidStandUpReachEndpointDataset(shuffled_indices_file=idx, trajectories_dir=traj_dir, query_mode="random_intermediate")
    raw = raws["sequence_2.txt"]  # 7 rows
    torch.manual_seed(0)
    queries = [ds[2]["start_state"][:34].numpy() for _ in range(20)]
    # every query must equal some non-terminal row (rows 0..n-2), never the final row
    final = raw[-1, :34]
    for q in queries:
        assert not np.allclose(q, final, atol=1e-4), "query must be non-terminal"
        assert any(np.allclose(q, raw[r, :34], atol=1e-4) for r in range(raw.shape[0] - 1))


def test_missing_file_raises_not_zero_fill(tmp_path):
    traj_dir = tmp_path / "trajectories"; traj_dir.mkdir()
    idx = tmp_path / "idx.txt"; idx.write_text("sequence_999.txt\n")
    ds = HumanoidStandUpReachEndpointDataset(shuffled_indices_file=str(idx), trajectories_dir=str(traj_dir), query_mode="start")
    with pytest.raises(Exception):
        _ = ds[0]
