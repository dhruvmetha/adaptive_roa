# HumanoidStandUpReach System — Design Spec

**Date:** 2026-06-23
**Status:** Approved (design); pending implementation plan
**Author:** brainstormed with the user

## 1. Goal

Add first-class support for the new **humanoid get-up** dataset to the adaptive-ROA
codebase by building a **fresh, self-contained system module set** named
`humanoid_standup_reach`, mirroring the mature `quadrotor_3d` (Quad3D) code path
file-for-file. The build covers the **full pipeline**: flow-matching training, ROA
evaluation, and the `adaptive_v2` conformal / adaptive-sampling loop.

The explicit motivation is to **avoid inheriting the defects** in the existing
`humanoid` scaffold (placeholder attractor, missing bounds pickle, prebuilt-file-only
data loader). That scaffold is **left untouched and unused** for now (deletion is
deferred, per the user).

## 2. Dataset (source of truth)

`/common/users/shared/pracsys/genMoPlan/data_trajectories/humanoid_get_up_medium/`

- **800,000 trajectories** from a teacher-student SAC controller in dm_control/MuJoCo
  "humanoid stand". Outcome is **binary**: success (93.85%, head_height ≥ 1.3 **and**
  ‖CoM velocity‖ ≤ 0.2) vs **timeout** (6.15%). **Zero explicit failures.**
- **State: 67-D**, `state_order =
  joint_angles(0–20) + head_height(21) + extremities(22–33) + torso_vertical(34–36) +
  com_velocity(37–39) + velocity(40–66)`.
- **Manifold**: predominantly Euclidean with one non-Euclidean block — `torso_vertical`
  (dims 34–36), a unit vector on **S²**. ⇒ ℝ³⁴ × S² × ℝ³⁰.
- **Layout matches `quadrotor3D_lqr`**: `trajectories/sequence_{i}.txt`,
  `train_test_splits/shuffled_indices_{0..9}.txt` (+ labels),
  `all_shuffled_indices.txt`/`all_shuffled_labels.txt`, `dataset_description.json`,
  `trajectory_labels.txt`.
- **Delimiter quirk**: trajectory rows are **comma+space (`, `)** separated (the eval
  files are comma-only). `np.loadtxt(path, delimiter=',')` handles both.
- **Pre-built held-out eval / calibration sets** in `train_test_splits/`, all in
  `state[67], final_state[67], label` (135-col) format, label `1=success / 0=timeout`:
  - **start_states** family: `eval_start_states.txt` (650k), `cal_set_start_states.txt`
    (150k), `test_set_start_states.txt` (500k) — one row per eval trajectory's prone
    **start** state.
  - **fps** family: `eval_fps.txt` (1.25M), `cal_set_fps.txt` (250k),
    `test_set_fps.txt` (1M) — farthest-point-sampled over the **non-terminal**
    eval-rollout states (intermediate states), each paired with its trajectory's final
    state + label.

## 3. Verified facts that shape the design

- **Predicted endpoints are classified via `system.classify_attractor(pred, radius)`**
  (`adaptive_v2/eval/full_roa.py:589`, `eval/mc_cache.py:144`). The `pred` passed in is
  **raw / denormalized** (Quad3D uses physical thresholds there). ⇒ humanoid success
  thresholds (head_height, CoM speed) apply directly to raw states.
- **`load_eval_states`** (`adaptive/data_source.py`) auto-infers `state_dim =
  (n_cols−1)//2`, so 135-col humanoid cal/test files load with **no code change**;
  default label map `{0:−1, 1:1}`.
- **Quad3D `_load_from_shuffled_indices` has a silent-zeros trap**: `np.loadtxt(path)`
  with the default whitespace delimiter **raises** on comma files (empirically confirmed
  on both Quad3D and humanoid trajectory files), and the bare `except` fills **all
  zeros**. We will **not** inherit this: use `delimiter=','` and let failures raise.
- **`humanoid_data_bounds.pkl` does not exist** ⇒ the old `HumanoidSystem.__init__`
  raises `FileNotFoundError`; the scaffold is currently un-instantiable. We load bounds
  from `dataset_description.json` instead (as `Quadrotor3DSystem` does).
- **FB FM `Sphere` manifold exists** (`flow_matching.utils.manifolds.Sphere`) with
  `projx`, `expmap`, `logmap`, `dist`. The S² block has **intrinsic tangent dim 2**
  (confirmed by the user), so inside `Product` it is entered as `(Sphere(), 3, 2)`
  (representation 3, tangent 2) — exactly analogous to Quad3D's `(SE3(), 7, 6)`. ⇒ the
  manifold **tangent space is 34 + 2 + 30 = 66**, and the model outputs **66-D** velocity
  when `use_manifold=True` (67-D when flat), mirroring Quad3D's 12-vs-13 split.
- **`adaptive_v2` is pool-based** (`TrajectoryDataSource` + `AdaptiveDatasetBuilder` over
  pre-collected trajectories) — **no live MuJoCo simulator** required.
- **`flow_matcher_local`** (trajectory FM) is referenced only by
  `prediction_mode/local.yaml`; the `adaptive_v2` default is `prediction_mode=global`,
  which needs only the **endpoint** flow matcher. ⇒ trajectory FM is **out of scope**.

## 4. Design decisions (locked with the user)

| # | Decision | Choice |
|---|---|---|
| 1 | Scope | **Full pipeline** (system + data + FM train/inference + ROA eval + adaptive_v2) |
| 2 | Code strategy | **Fresh parallel module set** `humanoid_standup_reach`; old `humanoid` scaffold left untouched |
| 3 | Manifold | ℝ³⁴ × S² × ℝ³⁰; flow matcher has `use_manifold` toggle (flat ℝ⁶⁷ when off); **distance/geometry always S²-aware** |
| 4 | Normalization | **Per-dimension → [−1,1]** for the 64 Euclidean dims (from JSON `per_dimension_min/max`); S² block passes through (unit-norm) |
| 5 | Success classify | **Binary**: `1` iff `head_height(21) ≥ 1.3` **and** `‖com_velocity(37:40)‖₂ ≤ 0.2`, else `−1`; on **raw** states; no separatrix; `radius` arg accepted but unused |
| 6 | Eval sets | **FPS primary** (`cal_set_fps`/`test_set_fps`), start_states as alternative |
| 7 | Training query | `query_mode` default **`random_intermediate`** (query = sampled non-terminal row → final row) |
| 8 | Adaptive pool source | Support **both**, pluggable: (A) on-the-fly intermediate sampling from train trajectories [default], (B) pre-generated `train_fps.txt` |
| 9 | Bounds source | `dataset_description.json` `achieved_bounds` (no pickle) |
| 10 | Sphere manifold | **Independently verified** beyond bundled tests (new test file) |
| 11 | Working branch | `main` (per the user, this session) |

## 5. Components

All new files; naming: class `HumanoidStandUpReach…`, dir/module `humanoid_standup_reach`.

### 5.1 `adaptive_roa/systems/humanoid_standup_reach.py` — `HumanoidStandUpReachSystem`
- `__init__(dataset_dir=None)` → defaults to `…/humanoid_get_up_medium`; calls
  `_load_bounds_from_json`.
- `_load_bounds_from_json`: read `achieved_bounds.per_dimension_min/max` → store 67-length
  min/max arrays; the S² dims (34–36) are excluded from normalization.
- `define_manifold_structure()` → `Real×34`, `Sphere(3, "torso_vertical")`, `Real×30`.
- `define_state_bounds()` → per-dim (min,max); orientation block `(−1,1)`.
- `normalize_state` / `denormalize_state`: per-dim affine map to/from [−1,1] for Euclidean
  dims; identity on dims 34–36. Exact inverses.
- `embed_state_for_model` → identity (sphere already continuous 3-D).
- `project_to_manifold(state)` → renormalize dims 34–36 to unit norm.
- `classify_attractor(state, radius=None)` → binary (see decision 5); operates on raw
  states; vectorized `[B]` int tensor in `{1, −1}`.
- `is_in_attractor(state, radius=None)` → boolean version of the same condition.
- `attractors()` → one nominal standing pose (head up, zero velocity,
  torso_vertical=[0,0,1]); **viz only**, not used in classification.
- `get_loss_weights()` → per-dim weights (used only if `use_loss_weights`).

### 5.2 `adaptive_roa/data/humanoid_standup_reach_endpoint_data.py`
`HumanoidStandUpReachEndpointDataset` + `HumanoidStandUpReachEndpointDataModule`.
- **Shuffled-indices mode** (primary): `train/val/test_indices_file` + `trajectories_dir`.
  Load each `sequence_{i}.txt` with `np.loadtxt(path, delimiter=',')`. **No zero-fill
  fallback — raise on failure.**
- **`query_mode`** (default `random_intermediate`):
  - `start` → query = row 0 (Quad3D behavior).
  - `random_intermediate` → query = uniformly sampled **non-terminal** row per
    `__getitem__`; target = last row. Stochastic coverage of intermediate states across
    epochs; matches the FPS eval.
  - `all_intermediate` → expand every non-terminal row into a (row_t → final) pair
    (larger dataset).
- `__getitem__` projects the S² block to exact unit norm.
- Optional direct-file mode (read a 135-col `*_states.txt`) for parity/debugging.
- Seeded RNG for reproducible intermediate sampling.

### 5.3 `adaptive_roa/flow_matching/humanoid_standup_reach/latent_conditional/`
`HumanoidStandUpReachLatentConditionalFlowMatcher(BaseFlowMatcher)` + `train.py`,
`inference.py`, `__init__.py`.
- `use_manifold: bool = True`:
  - on → `Product([(Euclidean(), 34, 34), (Sphere(), 3, 2), (Euclidean(), 30, 30)])`
    (tangent space 66) with `GeodesicProbPath` + `RiemannianODESolver`.
  - off → flat `Euclidean(67)` (tangent 67) + standard ODE.
- **`_create_distance_manifold()` ALWAYS returns the S²-aware Product** (validation MAE /
  distances are manifold-correct regardless of the toggle).
- `predict_endpoint` → integrate, then `project_to_manifold` (unit-norm sphere block).
- Model: `UniversalUNet`, input `142 = 67(embedded) + 67(condition) + 8(latent)`;
  **`output_dim = 66`** (manifold tangent; the flat `use_manifold=False` / 67-D case is
  reconciled in the flow matcher + loss-weight logic, as Quad3D does for 12-vs-13).
- `train.py` mirrors Quad3D: build system from `dataset_dir`, instantiate data module,
  flow matcher, optimizer/scheduler/trainer from Hydra.

### 5.4 Configs
- `configs/system/humanoid_standup_reach.yaml` → `_target_: …HumanoidStandUpReachSystem`,
  `dataset_dir`.
- `configs/model/humanoid_standup_reach_*.yaml` → UniversalUNet dims (142→66).
- `configs/train_humanoid_standup_reach.yaml` → data module in shuffled-indices mode
  (`shuffled_indices_{0,1,2}.txt`), `query_mode=random_intermediate`, `flow_matching`
  block incl. `use_manifold`.
- `configs/evaluate_humanoid_standup_reach_roa.yaml` → ROA eval (deterministic +
  probabilistic).
- `configs/adaptive_v2/system/humanoid_standup_reach.yaml` → `data_source` with
  `trajectories_dir`, `all_shuffled_indices`/labels, **`cal_set_file`/`test_set_file` →
  FPS sets** (start_states commented alt), `system`/`flow_matcher` targets,
  `prediction_mode=global`, `adaptive_v2.system_name`. Plus a run config mirroring
  `adaptive_vs_baselines_quadrotor3d.yaml` if needed for launching.

### 5.5 Adaptive intermediate-state pool
Extend `TrajectoryDataSource` / `AdaptiveDatasetBuilder` so candidates are
`(state_t → final_state, label)` over **intermediate** states (today they emit starts).
- **Option A (default):** on-the-fly intermediate sampling from the 150k train
  trajectories (FPS / stratified / random over collected rows). No new artifact;
  seed-reproducible.
- **Option B:** pre-generate a `train_fps.txt` pool via an FPS pass over train-split
  rollout states (mirrors `eval_fps.txt`). Reproducible, matches eval exactly; adds a
  generation script + large artifact + GPU step.
- Make the source pluggable; choose by practical tradeoff during implementation. The
  delimiter / no-zero-fill rule applies to trajectory loading here too.

## 6. Testing strategy (TDD)
- **System:** normalization round-trip (`denorm(norm(x))≈x`) per dim; `classify_attractor`
  at threshold boundaries (head_height = 1.3, CoM speed = 0.2); JSON bounds loaded
  correctly; S² block untouched by normalization.
- **Data module:** comma-delimiter parsing; correct start/final extraction; each
  `query_mode`; **assert batches are never all-zeros**; S² rows are unit-norm.
- **Flow matcher:** manifold on/off produce correct shapes (66 vs 67 tangent);
  `predict_endpoint` returns unit-norm sphere block; `_create_distance_manifold` is
  S²-aware even when `use_manifold=False`.
- **`tests/test_sphere_manifold.py` (independent Sphere verification):** `expmap`/`logmap`
  round-trip; `projx` idempotent + unit-norm preserving; tangent orthogonality
  (`⟨x, projx_tangent(v)⟩≈0`); `dist` symmetry and agreement with `arccos` geodesic; and
  confirm the **Sphere tangent_dim = 2** behavior inside `Product`. Record findings.

## 7. Out of scope
- Trajectory / `local` flow matcher (`flow_matcher_local`).
- Deleting the old `humanoid` scaffold (deferred).
- Any live MuJoCo simulator (pool is purely over pre-collected trajectories).

## 8. Open items to resolve during planning
- Sphere tangent dim **resolved = 2** ⇒ `Product` entry `(Sphere(), 3, 2)`, model
  `output_dim = 66`. **`Product` signature confirmed** from Quad3D:
  `Product(input_dim=<representation>, manifolds=[(Manifold(), repr_dim, tangent_dim), …])`
  — Quad3D uses `Product(input_dim=13, manifolds=[(SE3(), 7, 6), (Euclidean(), 6, 6)])`
  with model `output_dim=12`, and the Product does logmap (repr→tangent) / expmap
  (tangent→repr). Humanoid is the identical pattern (repr 67 → tangent 66), inherited for
  free by replicating Quad3D's flow matcher.
- `configs/model/system_dims` group wiring used by `adaptive_v2` (confirm path/name).
- Whether the adaptive pool ships Option A or B first (tradeoff call).
- Test directory location / harness convention in this repo.

## Appendix: branch history note
The design was first committed as `000eb2e` on branch `direct-classification`; the user
then moved to `main` and chose to continue the work there. This file is the `main` copy
(identical content plus the tangent-dim=2 resolution and this note).
