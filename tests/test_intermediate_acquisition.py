# tests/test_intermediate_acquisition.py
import numpy as np
import pytest
from pathlib import Path
from adaptive_roa.adaptive.data_source import TrajectoryDataSource, TrajectoryDataSourceConfig
from adaptive_roa.adaptive.dataset_builder import AdaptiveDatasetBuilder


def _make_pool(tmp_path, lengths, seed=0):
    """Build a tiny trajectories dir + indices/labels; return a TrajectoryDataSource."""
    traj_dir = tmp_path / "trajectories"; traj_dir.mkdir()
    rng = np.random.default_rng(seed)
    names = []
    for i, T in enumerate(lengths):
        rows = rng.normal(size=(T, 4)).astype(np.float64)
        np.savetxt(traj_dir / f"sequence_{i}.txt", rows, delimiter=", ", fmt="%.6f")
        names.append(f"sequence_{i}.txt")
    idx = tmp_path / "idx.txt"; idx.write_text("\n".join(names) + "\n")
    lab = tmp_path / "lab.txt"; lab.write_text("\n".join("1" for _ in names) + "\n")
    cfg = TrajectoryDataSourceConfig(trajectories_dir=str(traj_dir), shuffled_indices_file=str(idx),
                                     shuffled_labels_file=str(lab))
    return TrajectoryDataSource(cfg)


def test_start_mode_unchanged(tmp_path):
    ds = _make_pool(tmp_path, [5, 6, 7])
    b = AdaptiveDatasetBuilder(ds, str(tmp_path / "out_s"), candidate_mode="start")
    states, ids = b.sample_candidates_without_marking(2)
    assert len(ids) == 2 and states.shape == (2, 4)
    # ids are trajectory indices; their states are row 0
    assert np.allclose(states[0], ds.get_start_state(ids[0]))


def test_intermediate_candidate_is_subtrajectory_start(tmp_path):
    ds = _make_pool(tmp_path, [5])  # one trajectory of length 5 → rows 0..4 → candidate rows 0..3 (non-terminal)
    b = AdaptiveDatasetBuilder(ds, str(tmp_path / "out_i"), candidate_mode="intermediate")
    states, ids = b.sample_candidates_without_marking(4)
    assert len(ids) == 4  # 4 non-terminal rows of the single trajectory
    traj = ds.load_trajectory(0)
    for k, cid in enumerate(ids):
        i, r = b.candidate_to_traj_row(cid)
        assert np.allclose(states[k], traj[r])


def test_marking_prevents_overlap_and_tail_expansion(tmp_path):
    ds = _make_pool(tmp_path, [6])  # rows 0..5, final=row5
    b = AdaptiveDatasetBuilder(ds, str(tmp_path / "out_m"), candidate_mode="intermediate", val_ratio=0.0)
    # add candidate (traj 0, row 3): tail rows 3,4 -> final(5)
    cid_3 = b.traj_row_to_candidate(0, 3)
    b.add_to_training_balanced([cid_3])
    starts, ends, labels = b.get_training_data()
    assert starts.shape[0] == 2  # rows 3,4
    traj = ds.load_trajectory(0)
    assert np.allclose(np.sort(starts[:, 0]), np.sort(traj[3:5, 0]))
    assert np.allclose(ends, np.tile(traj[5], (2, 1)))
    # now add (traj 0, row 1): only NEW rows 1,2 are added (3,4 already marked)
    cid_1 = b.traj_row_to_candidate(0, 1)
    b.add_to_training_balanced([cid_1])
    starts2, ends2, _ = b.get_training_data()
    assert starts2.shape[0] == 4  # rows 1,2,3,4 (no duplication of 3,4)


def test_marked_tail_unavailable(tmp_path):
    ds = _make_pool(tmp_path, [6])
    b = AdaptiveDatasetBuilder(ds, str(tmp_path / "out_a"), candidate_mode="intermediate", val_ratio=0.0)
    cid_2 = b.traj_row_to_candidate(0, 2)
    b.mark_indices_as_used([cid_2])  # marks rows 2,3,4,5-start... (>=2)
    _, ids = b.sample_candidates_without_marking(10)
    rows = sorted(b.candidate_to_traj_row(c)[1] for c in ids)
    assert all(r < 2 for r in rows)  # only rows 0,1 remain available
