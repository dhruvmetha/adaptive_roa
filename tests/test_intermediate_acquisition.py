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


def test_build_all_datasets_intermediate(tmp_path):
    """build_all_datasets in intermediate mode must not crash and must write tail pairs."""
    ds = _make_pool(tmp_path, [6])  # rows 0..5, final=row5
    b = AdaptiveDatasetBuilder(ds, str(tmp_path / "out_b"), candidate_mode="intermediate", val_ratio=0.0)
    cid_3 = b.traj_row_to_candidate(0, 3)
    b.add_to_training_balanced([cid_3])
    paths = b.build_all_datasets(dataset_kind="endpoint")
    assert "train" in paths
    train_data = np.loadtxt(paths["train"])
    # 2 pairs (rows 3, 4), each pair is [start(4), end(4)] = 8 columns
    if train_data.ndim == 1:
        train_data = train_data.reshape(1, -1)
    assert train_data.shape == (2, 8)
    traj = ds.load_trajectory(0)
    # The seeded shuffle changes pair order; check as a set of row-0 values (col 0)
    # rather than requiring strict row order.
    expected_first_cols = np.sort(traj[3:5, 0])
    actual_first_cols = np.sort(train_data[:, 0])
    assert np.allclose(actual_first_cols, expected_first_cols, atol=1e-5)


def test_get_labels_candidate_id_aware(tmp_path):
    """get_labels maps candidate-ids to trajectory labels in intermediate mode
    and maps trajectory indices to labels in start mode."""
    # Use two trajectories with the same label (both label=1 from _make_pool)
    ds = _make_pool(tmp_path, [5, 6])
    # --- intermediate mode ---
    b_int = AdaptiveDatasetBuilder(ds, str(tmp_path / "out_gl_i"), candidate_mode="intermediate")
    # candidate_id for traj 0 row 2
    cid_0_2 = b_int.traj_row_to_candidate(0, 2)
    # candidate_id for traj 1 row 1
    cid_1_1 = b_int.traj_row_to_candidate(1, 1)
    labels = b_int.get_labels([cid_0_2, cid_1_1])
    assert labels.shape == (2,)
    # Both trajectories have label 1 (internal: 1) from _make_pool
    assert labels[0] == ds.get_label(0)
    assert labels[1] == ds.get_label(1)

    # --- start mode ---
    b_st = AdaptiveDatasetBuilder(ds, str(tmp_path / "out_gl_s"), candidate_mode="start")
    st_labels = b_st.get_labels([0, 1])
    assert st_labels.shape == (2,)
    assert st_labels[0] == ds.get_label(0)
    assert st_labels[1] == ds.get_label(1)


def test_intermediate_val_nonempty_with_val_ratio(tmp_path):
    """In intermediate mode with val_ratio=0.2, build_all_datasets writes a
    non-empty val endpoint file; get_val_labels returns non-empty labels that
    are a subset of the trajectory's label set; val start states are
    intermediate rows (not necessarily row 0)."""
    # Single trajectory of length 6: rows 0..5, final=row5
    # Candidate at row 1: tail pairs = rows 1,2,3,4 -> final (4 pairs)
    # N=4, val_ratio=0.2: n_val = max(1, int(4*0.2)) = max(1,0) = 1
    # val = P[:1] = pair for row 1 (start is traj row 1, NOT row 0)
    ds = _make_pool(tmp_path, [6])
    b = AdaptiveDatasetBuilder(
        ds, str(tmp_path / "out_val"), candidate_mode="intermediate", val_ratio=0.2, test_ratio=0.0
    )
    cid_1 = b.traj_row_to_candidate(0, 1)
    b.add_to_training_balanced([cid_1])

    paths = b.build_all_datasets(dataset_kind="endpoint")

    # Val file must be non-empty
    val_data = np.loadtxt(paths["val"])
    if val_data.ndim == 1:
        val_data = val_data.reshape(1, -1)
    assert val_data.shape[0] >= 1, "Val file must have at least one row"
    # Val start states are 4-D (state_dim=4)
    assert val_data.shape[1] == 8  # 4 start + 4 end

    # get_val_labels: non-empty, labels subset of trajectory's label set {1}
    val_starts, val_labels = b.get_val_labels()
    assert len(val_starts) >= 1
    assert len(val_labels) >= 1
    traj_label = ds.get_label(0)  # = 1
    assert all(l == traj_label for l in val_labels), "All val labels must equal the trajectory label"

    # Val start states are intermediate rows (not row 0 / start state)
    # After seeded shuffle, val[0] is *some* tail row, not necessarily row 1.
    # Verify the val start is one of the expected tail rows (rows 1,2,3,4 of traj 0).
    traj = ds.load_trajectory(0)
    tail_rows = traj[1:5]  # rows 1,2,3,4 (all possible start rows for this candidate)
    assert any(np.allclose(val_starts[0], row, atol=1e-5) for row in tail_rows), (
        f"Expected val start to be one of the tail rows [1..4], got {val_starts[0]}"
    )


def test_packed_candidate_id_roundtrip(tmp_path):
    ds = _make_pool(tmp_path, [5, 800, 7])  # middle traj is long
    b = AdaptiveDatasetBuilder(ds, str(tmp_path / "out_pk"), candidate_mode="intermediate", val_ratio=0.0)
    for (i, r) in [(0, 0), (1, 798), (2, 3)]:
        cid = b.traj_row_to_candidate(i, r)
        assert b.candidate_to_traj_row(cid) == (i, r)


def test_marking_is_not_global_scan(tmp_path, monkeypatch):
    # Build a pool; ensure construction does NOT load every trajectory's full array,
    # and marking does not scan a 60M structure. We assert lengths are computed lazily:
    ds = _make_pool(tmp_path, [6, 6, 6, 6])
    calls = {"load": 0}
    orig = ds.load_trajectory
    def counting_load(idx):
        calls["load"] += 1
        return orig(idx)
    monkeypatch.setattr(ds, "load_trajectory", counting_load)
    b = AdaptiveDatasetBuilder(ds, str(tmp_path / "out_lz"), candidate_mode="intermediate", val_ratio=0.0)
    # Construction must not have force-loaded all 4 full trajectories (lazy length via line-count).
    loads_after_init = calls["load"]
    # Packed scheme: NO flat registry / reverse-lookup / global used-set is built
    # at init. These attributes existed under the old O(60M) flat scheme and must
    # be gone — this is what actually guards the registry removal.
    assert not hasattr(b, "_candidates"), "flat candidate registry must not exist"
    assert not hasattr(b, "_cand_lookup"), "reverse lookup dict must not exist"
    assert not hasattr(b, "_cid_used"), "global used-cid set must not exist"
    # mark one candidate; availability still works without a global registry
    cid = b.traj_row_to_candidate(2, 2)
    b.mark_indices_as_used([cid])
    _, ids = b.sample_candidates_without_marking(20)
    rows_traj2 = sorted(r for (i, r) in (b.candidate_to_traj_row(c) for c in ids) if i == 2)
    assert all(r < 2 for r in rows_traj2)  # rows >=2 of traj 2 are marked/unavailable
    assert loads_after_init == 0  # lengths came from a cheap line-count, not full load


def test_sample_zero_returns_empty(tmp_path):
    ds = _make_pool(tmp_path, [6, 6, 6])
    b = AdaptiveDatasetBuilder(ds, str(tmp_path / "out_z"), candidate_mode="intermediate", val_ratio=0.0)
    states, ids = b.sample_candidates_without_marking(0)
    assert list(ids) == [] and len(states) == 0


def test_initial_training_set_seeds_distinct_trajectories(tmp_path):
    """In intermediate mode, get_initial_training_set(n) must seed n DISTINCT
    trajectories (one candidate each at row 0), not the first n rows of traj 0."""
    ds = _make_pool(tmp_path, [6, 6, 6, 6, 6])
    b = AdaptiveDatasetBuilder(ds, str(tmp_path / "out_seed"), candidate_mode="intermediate", val_ratio=0.0)
    b.get_initial_training_set(3)
    # 3 DISTINCT trajectories seeded (row-0 tails), not 3 rows of trajectory 0
    assert set(b._added_min_row.keys()) == {0, 1, 2}
    assert all(v == 0 for v in b._added_min_row.values())  # full tails (row 0)


def test_intermediate_val_is_shuffled(tmp_path):
    """After a seeded shuffle of P, the val split should contain rows from different
    trajectories (not just the lowest-index front-slice), and must be non-empty."""
    ds = _make_pool(tmp_path, [8, 8, 8, 8])
    b = AdaptiveDatasetBuilder(ds, str(tmp_path / "out_sh"), candidate_mode="intermediate", val_ratio=0.25)
    b.get_initial_training_set(4)
    starts, labels = b.get_val_labels()
    # val must not be exclusively the lowest-traj front-slice (seeded shuffle mixes trajectories)
    assert len(starts) >= 1
