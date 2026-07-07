"""Guards + determinism for intermediate-mode dataset building."""
import numpy as np
import pytest

from adaptive_roa.adaptive.data_source import TrajectoryDataSource, TrajectoryDataSourceConfig
from adaptive_roa.adaptive.dataset_builder import AdaptiveDatasetBuilder


def _make_pool(tmp_path, lengths, seed=0):
    traj_dir = tmp_path / "trajectories"
    traj_dir.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    names = []
    for i, T in enumerate(lengths):
        rows = rng.normal(size=(T, 4)).astype(np.float64)
        np.savetxt(traj_dir / f"sequence_{i}.txt", rows, delimiter=", ", fmt="%.6f")
        names.append(f"sequence_{i}.txt")
    idx = tmp_path / "idx.txt"
    idx.write_text("\n".join(names) + "\n")
    lab = tmp_path / "lab.txt"
    lab.write_text("\n".join("1" for _ in names) + "\n")
    cfg = TrajectoryDataSourceConfig(
        trajectories_dir=str(traj_dir),
        shuffled_indices_file=str(idx),
        shuffled_labels_file=str(lab),
    )
    return TrajectoryDataSource(cfg)


def _build_with_all_trajectories(tmp_path, lengths, out, seed):
    ds = _make_pool(tmp_path, lengths)
    b = AdaptiveDatasetBuilder(
        ds, str(tmp_path / out), candidate_mode="intermediate",
        val_ratio=0.2, test_ratio=0.0, seed=seed,
    )
    # seed one candidate per trajectory at row 0 -> full tail expansion
    cids = [b.traj_row_to_candidate(i, 0) for i in range(len(lengths))]
    b.add_to_training_balanced(cids)
    return b


def test_intermediate_split_deterministic_across_instances(tmp_path):
    """Two builders with the same seed produce identical (train,val) splits."""
    lengths = [6, 7, 8, 5]
    b1 = _build_with_all_trajectories(tmp_path / "a", lengths, "out1", seed=123)
    b2 = _build_with_all_trajectories(tmp_path / "b", lengths, "out2", seed=123)
    t1, v1, _ = b1._intermediate_split()
    t2, v2, _ = b2._intermediate_split()
    assert len(t1) == len(t2) and len(v1) == len(v2)
    for (s1, e1, l1), (s2, e2, l2) in zip(t1, t2):
        assert np.allclose(s1, s2) and np.allclose(e1, e2) and l1 == l2
    for (s1, e1, l1), (s2, e2, l2) in zip(v1, v2):
        assert np.allclose(s1, s2) and np.allclose(e1, e2) and l1 == l2


def test_intermediate_split_seed_changes_shuffle(tmp_path):
    """A different seed yields a different val draw (shuffle is seed-driven)."""
    lengths = [6, 7, 8, 5]
    b1 = _build_with_all_trajectories(tmp_path / "a", lengths, "out1", seed=1)
    b2 = _build_with_all_trajectories(tmp_path / "b", lengths, "out2", seed=2)
    _, v1, _ = b1._intermediate_split()
    _, v2, _ = b2._intermediate_split()
    # Same number of pairs, but the specific val members should differ for these seeds.
    starts1 = np.array([s for s, _, _ in v1])
    starts2 = np.array([s for s, _, _ in v2])
    assert not (starts1.shape == starts2.shape and np.allclose(starts1, starts2))


def test_intermediate_classification_raises(tmp_path):
    """Intermediate mode must fail loud on classification (would emit wrong format)."""
    b = _build_with_all_trajectories(tmp_path, [6, 7], "out", seed=42)
    with pytest.raises(NotImplementedError, match="classification"):
        b.build_all_datasets(dataset_kind="classification")
