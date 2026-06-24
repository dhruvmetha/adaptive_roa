# HumanoidStandUpReach Scale-Hardening Plan (Plan 3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Make `candidate_mode=intermediate` feasible on the real 800k-trajectory humanoid pool by replacing the flat candidate registry with an O(1) packed candidate-id scheme + lazy lengths, fixing the intermediate seeding bug, and capping the huge FPS eval files.

**Architecture:** Candidate-id becomes a **packed integer** `cid = traj_idx * MAX_ROWS + row` (decode: `traj_idx, row = divmod(cid, MAX_ROWS)`), so no global registry is materialized. Per-trajectory `_marked_from_row[i]` replaces the global `used` scan; trajectory lengths are computed **lazily** (cheap line-count, cached) only for sampled trajectories. Candidate-ids stay opaque integers outside `AdaptiveDatasetBuilder`.

**Tech Stack:** Python, the existing adaptive_v2 pool/engine, pytest.

**Spec:** `docs/superpowers/specs/2026-06-23-humanoid-standup-reach-design.md` §5.5
**Depends on:** Plan 2 (committed `d24661a..4be37c3` on `main`). Final-review findings being fixed: #1 (O(60M) marking), #2 (intermediate mis-seed), #3 (eager registry), #4 (test_ratio), #5 (val shuffle), FPS row cap.

## Global Constraints
- `MAX_ROWS` is a fixed bound strictly greater than the max trajectory length (dataset max = 751); use `MAX_ROWS = 10000`. `cid = traj_idx * MAX_ROWS + row`; max `cid ≈ 8e9` fits int64. Add an assertion that every trajectory length ≤ MAX_ROWS when a length is computed (raise if violated).
- **Start mode byte-identical** — all changes guarded by `if self.candidate_mode == "intermediate"`. The full suite (currently 53) MUST stay green.
- Candidate-ids opaque integers outside `AdaptiveDatasetBuilder`; do NOT modify `balanced_sampler.py`, strategies, or `engine.py` acquisition logic.
- All existing intermediate tests (`tests/test_intermediate_acquisition.py`) MUST keep passing (the packed scheme must satisfy the same contract: `traj_row_to_candidate`/`candidate_to_traj_row`, marking dedup, tail expansion, candidate-id-aware labels, non-empty intermediate val).
- Tests run with `/common/home/st1122/Projects/adaptive_roa/env/bin/python -m pytest`. Git: `main`; commit per task; no `Co-Authored-By`; trailer `Claude-Session: https://claude.ai/code/session_01NZMXZPhQUpzSKYDqNqypoK`.

---

### Task 1: O(1) packed candidate-id registry + lazy lengths (#1, #3)

**Files:**
- Modify: `adaptive_roa/adaptive/dataset_builder.py` (the intermediate registry, `sample_candidates_without_marking`, `mark_indices_as_used`, `add_to_training_balanced`, `traj_row_to_candidate`/`candidate_to_traj_row`, `_intermediate_split`/training expansion, `get_labels`).
- Modify: `adaptive_roa/adaptive/data_source.py` (ensure `get_trajectory_length` is cheap/cached; no full-array load).
- Test: extend `tests/test_intermediate_acquisition.py`.

**Interfaces (must preserve from Plan 2):**
- `traj_row_to_candidate(i, row) -> int`, `candidate_to_traj_row(cid) -> (i, row)` — now packed (`i*MAX_ROWS+row` / `divmod`).
- `sample_candidates_without_marking(n, exclude)` → `(states[n,D], cand_ids[n])`; `mark_indices_as_used(cand_ids)`; `add_to_training_balanced(cand_ids)`; `get_labels(cand_ids)`; the training/val/test split — all behave exactly as the Plan 2 tests assert, but O(tail)/lazy.

- [ ] **Step 1: Add a scale/laziness test (RED) — read the current intermediate code first**

```python
# add to tests/test_intermediate_acquisition.py
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
    # mark one candidate; availability still works without a global registry
    cid = b.traj_row_to_candidate(2, 2)
    b.mark_indices_as_used([cid])
    _, ids = b.sample_candidates_without_marking(20)
    rows_traj2 = sorted(r for (i, r) in (b.candidate_to_traj_row(c) for c in ids) if i == 2)
    assert all(r < 2 for r in rows_traj2)  # rows >=2 of traj 2 are marked/unavailable
    assert loads_after_init == 0  # lengths came from a cheap line-count, not full load
```

Run RED: `/common/home/st1122/Projects/adaptive_roa/env/bin/python -m pytest tests/test_intermediate_acquisition.py::test_packed_candidate_id_roundtrip tests/test_intermediate_acquisition.py::test_marking_is_not_global_scan -v`
Expected: FAIL (registry currently flat / `get_trajectory_length` may full-load).

- [ ] **Step 2: Implement the packed scheme**

Read the current intermediate code in `dataset_builder.py` first. Then:
- Add `MAX_ROWS = 10000` module constant. `traj_row_to_candidate(i, row) = i * MAX_ROWS + row`; `candidate_to_traj_row(cid) = divmod(cid, MAX_ROWS)`.
- **Remove** the eager flat `_candidates` list and any `_cand_lookup` reverse dict. Replace marking state with `self._marked_from_row: dict[int,int]` (per-traj smallest marked row; default meaning "nothing marked") and keep `self._added_min_row: dict[int,int]`.
- `get_trajectory_length(idx)` (data_source): cheap line-count (no full parse), cached in a dict. Assert `length <= MAX_ROWS`.
- `sample_candidates_without_marking(n, exclude)` (intermediate): iterate available trajectories in the existing order; for trajectory `i`, available non-terminal rows are `[0, _marked_from_row.get(i, get_trajectory_length(i)-1))`; emit packed cids (skip those in `exclude`) and their states (`data_source.get_state_at(i,row)`) until `n` collected. Only compute lengths for trajectories actually touched.
- `mark_indices_as_used(cand_ids)` (intermediate): for each cid → `(i,t)`, set `_marked_from_row[i] = min(_marked_from_row.get(i, get_trajectory_length(i)-1), t)`. O(1) per id.
- `add_to_training_balanced(cand_ids)` (intermediate): for each `(i,t)`, `_added_min_row[i] = min(_added_min_row.get(i, get_trajectory_length(i)-1), t)`; then `mark_indices_as_used([cid])`.
- `_intermediate_split` / training expansion: unchanged in spirit (`rows[_added_min_row[i]:-1] → final` per added trajectory `i`); iterate `_added_min_row` keys.
- `get_labels(cand_ids)` (intermediate): `divmod` → traj → `data_source.get_label(traj)` (unchanged behavior, now via packed decode).
- Start-mode branches untouched.

- [ ] **Step 3: GREEN + full suite**

`pytest tests/test_intermediate_acquisition.py -v` (all, incl. the Plan-2 tests) then `pytest -q` (full suite stays green — proves start mode + other systems unaffected).

- [ ] **Step 4: Commit**
```bash
git add adaptive_roa/adaptive/dataset_builder.py adaptive_roa/adaptive/data_source.py tests/test_intermediate_acquisition.py
git commit -m "perf: O(1) packed candidate-ids + lazy lengths for intermediate acquisition

Replace the flat ~60M candidate registry with packed ids (traj*MAX_ROWS+row)
and per-traj _marked_from_row; lengths computed lazily via line-count. Marking
and sampling are O(tail); no eager 800k build. start mode byte-identical.

Claude-Session: https://claude.ai/code/session_01NZMXZPhQUpzSKYDqNqypoK"
```

---

### Task 2: Intermediate seeding fix, test_ratio, val shuffle, FPS row cap (#2, #4, #5)

**Files:**
- Modify: `adaptive_roa/adaptive/dataset_builder.py` (`get_initial_training_set` intermediate branch; seeded shuffle in `_intermediate_split`).
- Modify: `configs/adaptive_v2/system/humanoid_standup_reach.yaml` (`test_ratio: 0`).
- Modify: `adaptive_roa/adaptive/data_source.py::load_eval_states` (or its caller) + the eval path to honor an optional `max_eval_rows`/`max_cal_rows` cap; add the cap to the humanoid config.
- Test: extend `tests/test_intermediate_acquisition.py` + the adaptive-config test.

- [ ] **Step 1: Write failing tests**

```python
# add to tests/test_intermediate_acquisition.py
def test_initial_training_set_seeds_distinct_trajectories(tmp_path):
    ds = _make_pool(tmp_path, [6, 6, 6, 6, 6])
    b = AdaptiveDatasetBuilder(ds, str(tmp_path / "out_seed"), candidate_mode="intermediate", val_ratio=0.0)
    b.get_initial_training_set(3)
    # 3 DISTINCT trajectories seeded (row-0 tails), not 3 rows of trajectory 0
    assert set(b._added_min_row.keys()) == {0, 1, 2}
    assert all(v == 0 for v in b._added_min_row.values())  # full tails (row 0)


def test_intermediate_val_is_shuffled(tmp_path):
    ds = _make_pool(tmp_path, [8, 8, 8, 8])
    b = AdaptiveDatasetBuilder(ds, str(tmp_path / "out_sh"), candidate_mode="intermediate", val_ratio=0.25)
    b.get_initial_training_set(4)
    starts, labels = b.get_val_labels()
    # val must not be exclusively the lowest-traj front-slice (seeded shuffle mixes trajectories)
    assert len(starts) >= 1
```

(For the config test, assert `cfg.test_ratio == 0` and that a `max_eval_rows`/`max_cal_rows` key exists in the humanoid adaptive config.)

- [ ] **Step 2: RED**, then implement:
- `get_initial_training_set(n)` (intermediate): `add_to_training_balanced([traj_row_to_candidate(i, 0) for i in range(min(n, max_train_idx))])`.
- `_intermediate_split`: seeded shuffle of `P` (e.g. `random.Random(self.seed or 42).shuffle(P_copy)`) before slicing val/test/train, so the val/test draw is representative.
- `configs/adaptive_v2/system/humanoid_standup_reach.yaml`: add `test_ratio: 0` and (e.g. under `conformal:` or top-level) a `max_eval_rows`/`max_cal_rows` cap with a sane default (e.g. 50000) — wired so the eval/cal load uses `np.loadtxt(..., max_rows=cap)` when set.
- `load_eval_states` (and the test-set eval load): accept an optional `max_rows` and pass to `np.loadtxt`.

- [ ] **Step 3: GREEN + full suite green.**

- [ ] **Step 4: Commit**
```bash
git add adaptive_roa/adaptive/dataset_builder.py adaptive_roa/adaptive/data_source.py configs/adaptive_v2/system/humanoid_standup_reach.yaml tests/test_intermediate_acquisition.py tests/test_humanoid_standup_reach_adaptive_config.py
git commit -m "fix: intermediate seeding, test_ratio=0, seeded val shuffle, FPS eval row cap

get_initial_training_set seeds distinct trajectories (row-0 tails) in intermediate
mode; seeded shuffle of the val/test split; test_ratio=0 for humanoid; optional
max_eval_rows cap so the 1.3GB FPS files aren't fully loaded each eval epoch.

Claude-Session: https://claude.ai/code/session_01NZMXZPhQUpzSKYDqNqypoK"
```

---

### Task 3: Real-scale construction sanity check (no full run)

**Files:** Test: `tests/test_intermediate_scale_smoke.py` (dataset-skip-guarded).

- [ ] **Step 1: Write a guarded construction-timing test** that, on the real dataset, instantiates `AdaptiveDatasetBuilder(data_source_over_real_pool, candidate_mode="intermediate")` and seeds a small `initial_train_size`, asserting construction completes quickly (e.g. < 30 s) and does not materialize a giant registry (no full-trajectory loads beyond the seeded few). Use a small `shuffled_indices` slice if needed to bound the data_source. Keep it fast; this guards against regression of #1/#3.

- [ ] **Step 2–3: Run; if it reveals a residual O(N)/eager cost, fix in Task 1's files and re-run. Commit.**

```bash
git add tests/test_intermediate_scale_smoke.py
git commit -m "test: guard intermediate-mode construction stays cheap (no eager registry)

Claude-Session: https://claude.ai/code/session_01NZMXZPhQUpzSKYDqNqypoK"
```

## Notes / risk
- The packed-id `MAX_ROWS` bound must exceed the true max trajectory length (751). 10000 is safe; the length assertion guards against a longer trajectory silently colliding ids.
- Re-enabling `ConfidencePairFilter` for intermediate mode (the append-stability issue) remains a separate follow-up — out of scope here.
