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
- **FB FM `Sphere` manifold** (`flow_matching.utils.manifolds.Sphere`): `projx`,
  `expmap`, `logmap`, `dist`. **Empirically verified**: the FB FM `Product` constructor
  **requires `state_dim == tangent_dim` for a `Sphere` block** — `(Sphere(), 3, 2)` raises
  `"Sphere manifold must have state_dim == tangent_dim"`; the valid entry is
  `(Sphere(), 3, 3)`. The sphere tangent is represented in **ambient 3-D** and projected
  onto the 2-DOF tangent plane via `proju` (still geometrically S²-aware; `expmap`
  preserves unit norm). ⇒ manifold **tangent space = 34 + 3 + 30 = 67**, so the model
  **`output_dim = 67`** in BOTH `use_manifold` modes (no toggle-dependent output dim —
  simpler than Quad3D's 12-vs-13, where SE3 genuinely reduces 7→6). Geometric note: S²
  has 2 intrinsic DOF, but this library *declares* it ambient-3.
- **`Product.dist` returns 65 components** for the humanoid distance manifold
  (Euclidean 34 → 34, Sphere → 1 geodesic, Euclidean 30 → 30), so
  `get_manifold_component_names()` must return **65** names.
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
  - on → `Product(input_dim=67, manifolds=[(Euclidean(), 34, 34), (Sphere(), 3, 3),
    (Euclidean(), 30, 30)])` (tangent 67) with `GeodesicProbPath` + `RiemannianODESolver`.
  - off → flat `Euclidean()` (tangent 67) + standard ODE.
- **`_create_distance_manifold()` ALWAYS returns the S²-aware Product** (validation MAE /
  distances are manifold-correct regardless of the toggle); `manifold.dist` → 65 comps.
- `get_manifold_component_names()` returns 65 names; `sample_noisy_input` mirrors Quad3D
  (Gaussian + clamp on the 64 Euclidean dims; unit-normalized Gaussian on the 3 sphere
  dims; then `manifold.projx`); `_get_start_states`/`_get_end_states` read
  `batch["start_state"]`/`["end_state"]`; `normalize/denormalize/embed` delegate to system.
- `predict_endpoint` overrides base to call `system.project_to_manifold` (unit-norm sphere
  block) after integration, as Quad3D does for quaternions.
- **Model: reuse `adaptive_roa.model.quadrotor3d_unet.Quadrotor3DUNet`** (it is generic —
  parameterized purely by dims, `forward(x_t, t, z, condition)` concatenates
  `[x_t, t_emb, z, condition]`). Config dims: `embedded_dim=67`, `condition_dim=67`,
  `latent_dim=8`, **`output_dim=67`**, `time_emb_dim=64`, `hidden_dims` (e.g.
  `[512,1024,1024,512]`). (Correction vs original spec note that said `UniversalUNet` —
  `UniversalUNet.forward(x,t,condition)` has no latent arg, so it is incompatible with the
  base FM's `self.model(x_t, t, z, condition)` call; Quad3D's UNet is the right template.)
- The base `BaseFlowMatcher` provides `forward`, `_prepare_model_inputs`,
  `compute_flow_loss`, `training_step`, `validation_step`, `configure_optimizers`,
  `predict_endpoint` — humanoid FM inherits these and only overrides the system-specific
  hooks listed above.
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
- **Flow matcher:** model `output_dim=67`; manifold on/off both produce 67-D tangent /
  67-D state; `predict_endpoint` returns unit-norm sphere block; `_create_distance_manifold`
  is S²-aware even when `use_manifold=False` and `manifold.dist` returns 65 components.
- **`tests/test_sphere_manifold.py` (independent Sphere verification):** `expmap`/`logmap`
  round-trip; `projx` idempotent + unit-norm preserving; tangent orthogonality
  (`⟨x, proju(x,v)⟩≈0`); `dist` symmetry and agreement with `arccos` geodesic; confirm
  `Product` requires `(Sphere(),3,3)` (rejects `(…,3,2)`) and `Sphere.dist` → 1 value /
  pair. Record findings.

## 7. Out of scope
- Trajectory / `local` flow matcher (`flow_matcher_local`).
- Deleting the old `humanoid` scaffold (deferred).
- Any live MuJoCo simulator (pool is purely over pre-collected trajectories).

## 8. Resolved during planning (was: open items)
- **Sphere tangent representation = ambient 3** (empirically: `Product` rejects
  `(Sphere(),3,2)`, requires `(Sphere(),3,3)`) ⇒ model **`output_dim = 67`** in both
  `use_manifold` modes; `manifold.dist` → 65 components. **`Product` signature confirmed**:
  `Product(input_dim=<representation>, manifolds=[(Manifold(), repr_dim, tangent_dim), …])`
  (Quad3D: `[(SE3(),7,6),(Euclidean(),6,6)]`, output 12). Humanoid:
  `[(Euclidean(),34,34),(Sphere(),3,3),(Euclidean(),30,30)]`, output 67.
- **Model: reuse `Quadrotor3DUNet`** (generic, latent-aware `forward(x_t,t,z,condition)`).
  `UniversalUNet` is incompatible (no latent arg).
- **Test convention**: `tests/` dir, `pytest.ini` (`testpaths=tests`, `addopts=-q`),
  `conftest.py` adds repo root to `sys.path`; tests import `adaptive_roa.*` and run under
  the repo conda env `/common/home/st1122/Projects/adaptive_roa/env`.
- **`adaptive_v2` model dims** live in `configs/adaptive_v2/model/system_dims/<name>.yaml`
  (Plan 2 adds `humanoid_standup_reach.yaml`).
- **`evaluate_roa.py` routing**: config provides `system.module` + `system.class`
  (the flow-matcher module/class), loaded via `importlib` (Plan 2).
- Whether the adaptive pool ships Option A or B first (tradeoff call).
- Test directory location / harness convention in this repo.

## Appendix: branch history note
The design was first committed as `000eb2e` on branch `direct-classification`; the user
then moved to `main` and chose to continue the work there. This file is the `main` copy
(identical content plus the tangent-dim=2 resolution and this note).
