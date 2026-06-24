"""
Real-scale construction smoke test for AdaptiveDatasetBuilder in intermediate mode.

Guards against O(N)/eager regression from Task 1's packed-id / lazy-length rework:
  - Construction must be fast (no flat registry materialised, no trajectory loads)
  - Seeding a small initial set and sampling a handful of candidates must touch only
    those trajectories (lazy path), not all 150k
  - The builder must NOT have attributes from the old eager-registry implementation

Dataset-skip-guarded: tests are skipped if DATASET_DIR is absent (not available on CI).
"""
import time
import pytest
import numpy as np
from pathlib import Path
from typing import List

from adaptive_roa.adaptive.data_source import (
    TrajectoryDataSource,
    TrajectoryDataSourceConfig,
    MAX_ROWS,
)
from adaptive_roa.adaptive.dataset_builder import AdaptiveDatasetBuilder

# ---------------------------------------------------------------------------
# Dataset location
# ---------------------------------------------------------------------------
DATASET_DIR = "/common/users/shared/pracsys/genMoPlan/data_trajectories/humanoid_get_up_medium"

pytestmark = pytest.mark.skipif(
    not Path(DATASET_DIR).exists(),
    reason=f"Real dataset not found at {DATASET_DIR}; skipping scale smoke test",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_data_source() -> TrajectoryDataSource:
    """Build a TrajectoryDataSource over the real 150k humanoid train split."""
    cfg = TrajectoryDataSourceConfig(
        trajectories_dir=str(Path(DATASET_DIR) / "trajectories"),
        shuffled_indices_file=str(
            Path(DATASET_DIR) / "train_test_splits" / "shuffled_indices.txt"
        ),
        shuffled_labels_file=str(
            Path(DATASET_DIR) / "train_test_splits" / "shuffled_labels.txt"
        ),
    )
    return TrajectoryDataSource(cfg)


# ---------------------------------------------------------------------------
# The test
# ---------------------------------------------------------------------------

def test_intermediate_construction_stays_cheap(tmp_path, monkeypatch):
    """Construction of AdaptiveDatasetBuilder in intermediate mode must be O(1).

    Assertions:
    1. Builder construction completes in < 30 s (generous; should be ~instant).
    2. No flat registry attributes from the old eager implementation.
    3. Post-construction load_trajectory call count == 0 (no eager trajectory loads).
    4. Seeding 8 trajectories + sampling 16 candidates works correctly.
    5. Post-seed/sample distinct trajectory count is small (< 100), not 150k.
    """
    # -- Setup data source (outside the timer; indices + labels load is ~constant cost)
    ds = _make_data_source()

    # -- Instrument load_trajectory with a counter BEFORE building the builder
    load_call_count = [0]
    loaded_traj_indices: List[int] = []
    orig_load = ds.load_trajectory

    def counting_load(idx: int) -> np.ndarray:
        load_call_count[0] += 1
        loaded_traj_indices.append(idx)
        return orig_load(idx)

    monkeypatch.setattr(ds, "load_trajectory", counting_load)

    # -- Time builder construction only
    t0 = time.perf_counter()
    b = AdaptiveDatasetBuilder(
        ds,
        str(tmp_path / "out"),
        candidate_mode="intermediate",
    )
    elapsed = time.perf_counter() - t0

    # 1. Construction must be fast
    assert elapsed < 30.0, (
        f"AdaptiveDatasetBuilder construction took {elapsed:.2f}s (limit 30s); "
        f"likely O(N) eager work at pool scale."
    )

    # 2. No old flat-registry attributes (names from before Task 1 rework)
    assert not hasattr(b, "_candidates"), (
        "Builder has '_candidates' attribute — flat registry was materialised at init."
    )
    assert not hasattr(b, "_cid_used"), (
        "Builder has '_cid_used' attribute — flat registry was materialised at init."
    )
    assert not hasattr(b, "_cand_lookup"), (
        "Builder has '_cand_lookup' attribute — flat registry was materialised at init."
    )

    # 3. Zero trajectory loads at construction time
    assert load_call_count[0] == 0, (
        f"load_trajectory was called {load_call_count[0]} times during construction; "
        f"expected 0 (no eager trajectory loads)."
    )

    # -- Seed a small initial set (8 trajectories)
    init_ids = b.get_initial_training_set(8)
    assert len(init_ids) == 8, f"Expected 8 initial ids, got {len(init_ids)}"

    # -- Sample 16 candidates (lazy path at real 150k scale)
    states, cand_ids = b.sample_candidates_without_marking(16)

    # 4a. Correct number of candidates returned
    assert len(cand_ids) == 16, f"Expected 16 candidate ids, got {len(cand_ids)}"
    assert states.shape[0] == 16, (
        f"Expected states.shape[0]==16, got {states.shape}"
    )

    # 4b. Each candidate id decodes to a valid (traj, row) pair
    for cid in cand_ids:
        traj_idx, row = b.candidate_to_traj_row(cid)
        assert 0 <= traj_idx < ds.n_trajectories, (
            f"candidate_to_traj_row({cid}) → traj_idx={traj_idx} out of range "
            f"[0, {ds.n_trajectories})"
        )
        assert 0 <= row < MAX_ROWS, (
            f"candidate_to_traj_row({cid}) → row={row} out of range [0, {MAX_ROWS})"
        )

    # 5. Only a small number of distinct trajectories were touched during
    #    seeding + sampling (not 150k).  The initial 8 seeds call load via
    #    add_to_training_balanced → get_trajectory_length (line-count only, no
    #    load_trajectory call), so those won't show up.  The 16 sampled states
    #    resolve via get_state_at → load_trajectory, touching at most 16 distinct
    #    trajectories (likely far fewer due to caching).  Set a generous bound to
    #    guard only the 150k-regression case.
    distinct_touched = len(set(loaded_traj_indices))
    assert distinct_touched < 100, (
        f"{distinct_touched} distinct trajectories were loaded during init+seed+sample; "
        f"expected < 100 (lazy path)."
    )

    # Measure overall wall time (construction + seed + sample) for reporting
    print(
        f"\n[scale smoke] construction={elapsed:.3f}s, "
        f"distinct_trajectories_touched={distinct_touched}, "
        f"total_load_calls={load_call_count[0]}"
    )
