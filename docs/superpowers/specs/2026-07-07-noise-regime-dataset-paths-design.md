# Noise-regime dataset paths (deterministic / noisy)

**Date:** 2026-07-07
**Status:** Approved

## Problem

The trajectory dataset directory was reorganized. What used to be

```
$DATA_DIR/<dataset_name>/...            # e.g. .../data_trajectories/pendulum_lqr_50k
```

now lives under a regime subdirectory:

```
$DATA_DIR/deterministic/<dataset_name>/...
$DATA_DIR/noisy/<variant_name>/...
```

Every current pipeline (adaptive_v2, flow-matching training, ROA evaluation, and
data-loading scripts) still builds paths as `<base>/<dataset_name>/...` and therefore
now points at a non-existent location. All current pipelines are logically operating on
the *deterministic* data.

The **noisy** data additionally has a different on-disk format (a single
`trajectories.npz` plus `dataset_description.json` / `extras.npz`, with **no**
`train_test_splits/`, `cal_set.txt`, `test_set.txt`, or `eval_states.txt`) and different
folder names (e.g. `pendulum_lqr_stoch-dyn-ctrl-high`). Full noisy support therefore
requires a new datamodule and split-generation logic — that is explicitly **out of
scope** here.

## Goal

1. Make every current pipeline resolve to the new `deterministic/` prefix, unchanged in
   behavior.
2. Introduce a single, explicit switch (`noise_regime`) so a future run can select
   `noisy` at one point, rather than editing scattered hardcoded paths.

## Non-goals ("later" work)

- Parsing the noisy `trajectories.npz` format.
- Generating splits / calibration / test / eval sets for noisy data.
- Noisy-specific system or datamodule configs.

The seam below ensures these become localized additions.

## Design

**Mechanism: explicit per-config variable** (chosen over baking the regime into
`get_data_dir()`), so the regime is visible at every path-construction site.

Path pattern everywhere becomes:

```
<base>/${noise_regime}/<dataset_name>/...
```

with `noise_regime` defaulting to `deterministic`.

### 1. Python choke point — `adaptive_roa/utils/env_config.py`

- Add `get_noise_regime()`: returns env var `NOISE_REGIME`, default `"deterministic"`.
- Leave `get_data_dir()` and `get_shared_data_base()` returning the **raw** base
  (regime is NOT baked in).
- Update each hardcoded fallback default (the `dataset_dir=None` branches) in:
  - `adaptive_roa/systems/{pendulum,cartpole,quadrotor2d,quadrotor3d,humanoid_standup_reach,pendulum_cartesian,mountain_car}.py`
  - `adaptive_roa/data/{cartpole,quadrotor2d,quadrotor3d,pendulum_cartesian,humanoid,mountain_car}_endpoint_data.py`

  from `f"{get_data_dir()}/pendulum_lqr_50k"` →
  `f"{get_data_dir()}/{get_noise_regime()}/pendulum_lqr_50k"`
  (and the `get_shared_data_base()` variants likewise).

### 2. adaptive_v2 system configs — `configs/adaptive_v2/system/`

- Add `noise_regime: deterministic` to `_base.yaml` (`# @package _global_`, shared by all
  5 systems, CLI-overridable).
- In each system config (`pendulum`, `cartpole_pybullet`, `humanoid_standup_reach`,
  `quadrotor2d`, `quadrotor3d`): introduce `dataset_name: <folder>` and a single base

  ```yaml
  dataset_root: ${data_dir}/${noise_regime}/${dataset_name}
  ```

  then route every `*_file` / `*_dir` and the `system.dataset_dir` through
  `${dataset_root}` (also normalizes pendulum, which is currently inline, and the
  `base_dir` systems onto one pattern).

### 3. Non-adaptive pipelines

Insert the `${noise_regime}` segment at each entry point that builds dataset paths,
adding a local `noise_regime: deterministic` where one is not already in scope:

- `configs/system/*.yaml` (cartpole, humanoid_standup_reach, mountain_car,
  pendulum_cartesian, pendulum, quadrotor2d, quadrotor3d)
- `configs/data/*.yaml` (cartpole, humanoid, pendulum_cartesian, quadrotor3d endpoint data)
- `configs/train_*.yaml`, `configs/evaluate_*.yaml`
- `configs/adaptive_v2/prediction_mode/local.yaml` (comment/paths if any)

These use a mix of `${data_dir:}`, `${shared_data_base:...}`, and `${shared_data}`
prefixes; insert the regime segment after whichever prefix is used.

### 4. Scripts — `scripts/`

Sweep only scripts that **read** `data_trajectories/<name>` (e.g. `run_adaptive.py`,
`plot_pendulum_roa_gt.py`, qualitative-video scripts). Insert the regime segment,
defaulting to deterministic. Leave pure output/plot-path references alone.

### Out of scope / false positives

- `configs/compare_*.yaml` reference experiment **output** dirs
  (`adaptive_roa_experiments/...`), not trajectory data — do not touch.

## Two knobs, one default

- Hydra runs: `noise_regime` config value (default `deterministic`), override with
  `noise_regime=noisy` on the CLI.
- Direct Python instantiation (no Hydra): env `NOISE_REGIME` (default `deterministic`).

Both independently default to `deterministic`; they cover different entry points.

## Verification

1. Grep: no live dataset path resolves to `${data_dir}/<name>` (or
   `${shared_data*}/<name>`) without a `${noise_regime}` / `deterministic` segment.
2. `--cfg job` smoke test on one adaptive_v2 system, one `train_*`, and one
   `evaluate_*` config; confirm interpolated paths contain `/deterministic/` and that
   the referenced files exist on disk.
3. Confirm `get_noise_regime()` picks up `NOISE_REGIME=noisy` and that a fallback default
   path then contains `/noisy/`.
