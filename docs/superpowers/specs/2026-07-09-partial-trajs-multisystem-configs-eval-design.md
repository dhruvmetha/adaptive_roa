# Partial-Trajectory Verifier: Multi-System Configs, Checkpointing, and Evaluation

**Date:** 2026-07-09
**Branch:** `feat/partial-trajs-multisystem`
**Status:** Approved

## Goal

Make the partial-trajectory T-step dynamics verifier (`adaptive_roa/partial_trajs/`)
runnable end-to-end across all five systems, with hyperparameters mirroring the
adaptive experiments. Three deliverables:

1. **Per-system + per-backend Hydra config groups** for training, using the same
   model sizes and hyperparameters as the adaptive experiments.
2. **A standalone evaluation script + config** that loads a trained checkpoint and
   reports ROA success-detection metrics plus a richly-stratified rollout
   final-state error (metric #2), writing `metrics.json` and `predictions.npz`.
3. **Checkpoint saving in training**, matching the adaptive `ClassifierTrainer`
   behavior (best-val checkpoint, last checkpoint, early stopping, CSV logs).

**Explicit non-goal:** no adaptive data acquisition. Only the training behavior /
configs and a standalone evaluation are in scope.

## Background (current state)

- `train.py` + `configs/partial_trajs/train_partial_trajs.yaml`: single flat config,
  defaults to pendulum; other systems require manual `system=`, `dataset_dir=`,
  `hidden_dims=` overrides. Only reports metric #1 (per-horizon T-step val error).
  `enable_checkpointing: true` uses Lightning's default checkpoint (no explicit
  `ModelCheckpoint`, no best-val selection, no early stopping).
- `verifier/evaluate_roa.py`: `load_eval_states` + `evaluate_roa` exist as library
  functions, tested only with stub models/systems. No CLI, no config, no checkpoint
  loading, no `K` wiring. `data/description.py::load_dataset_description` parses `K`,
  `state_dim`, etc. from `dataset_description.json` but nothing connects it to eval.
- Two backends: `deterministic` (`DynamicsRegressor`) and `generative`
  (`GenerativeDynamics`, conditional rectified flow). Both log `train_loss`/`val_loss`.
- Systems registry `systems/__init__.py::make_verifier_system` covers all five:
  `pendulum`, `cartpole`, `quadrotor2d`, `quadrotor3d`, `humanoid_standup_reach`.
  Pendulum + humanoid use the `_UnresolvedFailureMixin` (no failure set → −1 remapped
  to 0); cartpole/quad2d/quad3d use base classes with genuine OOB failure sets.

## Reference values (from the adaptive experiments)

Per-system model dims from `configs/adaptive_v2/model/system_dims/*.yaml`
(`mlp_hidden_dims`) and radii from `configs/adaptive_v2/system/*.yaml`
(`attractor_radius`):

| system (registry key)    | dataset folder                              | `hidden_dims`               | `attractor_radius` | gen `lr` | eval file          |
|--------------------------|---------------------------------------------|-----------------------------|--------------------|----------|--------------------|
| `pendulum`               | `pendulum_lqr/pendulum_lqr_50k_T{T}`        | `[256,512,256]`             | 0.1                | 1e-3     | `test_set.txt`     |
| `cartpole`               | `cartpole_pybullet/cartpole_pybullet_T{T}`  | `[256,512,1024,512,256]`    | 0.2                | 1e-3     | `test_set.txt`     |
| `quadrotor2d`            | `quadrotor2D_rl/quadrotor2D_rl_T{T}`        | `[256,512,1024,512,256]`    | 0.3                | 1e-3     | `test_set.txt`     |
| `quadrotor3d`            | `quadrotor3D_lqr/quadrotor3D_lqr_T{T}`      | `[512,1024,1024,512]`       | 0.3                | 5e-4     | `test_set.txt`     |
| `humanoid_standup_reach` | `humanoid_get_up_medium/humanoid_get_up_medium_T{T}` | `[512,1024,1024,512]` | 1.0            | 5e-4     | `test_set_fps.txt` |

Backend hyperparameters:

- **Generative (primary)** — flow-matching analog (`predictor/generative.yaml`,
  `system/_base.yaml`): `max_epochs 1000`, `batch_size 1024`,
  `num_integration_steps 100`, `latent_dim 0` (zero-latent), `time_emb_dim 64`,
  `lr 1e-3` (quad3d/humanoid override to `5e-4`).
- **Deterministic** — classifier analog (`predictor/classifier.yaml`): `max_epochs
  200`, `batch_size 1024`, `dropout 0.0`, `lr 1e-3` (all systems).

Checkpoint pattern (from `ClassifierTrainer.fit`): `ModelCheckpoint(monitor="val_loss",
mode="min", save_top_k=1, save_last=True, filename="best-{epoch:02d}-{val_loss:.4f}")`,
`EarlyStopping(monitor="val_loss", patience=20)`, `CSVLogger`, `gradient_clip_val=1.0`,
`log_every_n_steps=10`, `check_val_every_n_epoch=1`; reload best-val weights after fit.

## Design

### A. Config groups (Hydra composition)

Refactor `configs/partial_trajs/` into groups mirroring `adaptive_v2`:

```
configs/partial_trajs/
  train_partial_trajs.yaml       # defaults: [backend: generative, system: pendulum, _self_]
  evaluate_partial_trajs.yaml    # defaults: [system: pendulum, _self_]
  backend/generative.yaml        # @package _global_
  backend/deterministic.yaml     # @package _global_
  system/pendulum.yaml           # @package _global_
  system/cartpole.yaml
  system/quadrotor2d.yaml
  system/quadrotor3d.yaml
  system/humanoid_standup_reach.yaml
```

**Naming:** each `system/*.yaml` option file is named after the registry key it
selects, so `system=<key>` both selects the group file and matches the flat `system`
string the file sets (`system=humanoid_standup_reach`, `system=cartpole`, …). The
Hydra group is named `system`; because option files use `# @package _global_`, the
final composed config carries a top-level `system: <registry_key>` string, which is
exactly what `run()` passes to `make_verifier_system`.

**Critical constraint:** all group files use `# @package _global_` so composition
flattens into the top-level namespace. `train.py::run(cfg)` continues to read flat
keys (`system`, `dataset_dir`, `hidden_dims`, `lr`, `batch_size`, `max_epochs`, …).
This keeps `test_train_smoke.py` (which builds a flat `OmegaConf` and calls `run`)
green with **no test change**.

Each `system/*.yaml` sets: `system` (registry key), `horizon_T` (default 25,
composes the `_T{T}` dataset path via OmegaConf interpolation), `dataset_dir`,
`hidden_dims`, `attractor_radius`, `eval_split_file` (`test_set.txt`; humanoid →
`test_set_fps.txt`). quad3d/humanoid additionally override `lr: 5e-4` with a comment
noting the deterministic-classifier analog used 1e-3.

`dataset_dir` uses the existing `shared_data_base` resolver:
```yaml
dataset_dir: ${shared_data_base:/common/users/shared/pracsys/genMoPlan/data_trajectories}/partial_deterministic/pendulum_lqr/pendulum_lqr_50k_T${horizon_T}
```

Defaults order is `backend` then `system` then `_self_`, so per-system `lr` overrides
win over the backend default.

Usage:
```bash
python adaptive_roa/partial_trajs/train.py                       # pendulum, generative
python adaptive_roa/partial_trajs/train.py system=humanoid_standup_reach
python adaptive_roa/partial_trajs/train.py system=cartpole horizon_T=50 backend=deterministic
```

### B. Checkpointing in `train.py`

Extend the trainer construction in `run()`, gated so the smoke test path is unchanged:

- When `enable_checkpointing` is true (real runs): attach `ModelCheckpoint`
  (dir `<run_dir>/checkpoints`, monitor `val_loss`, `save_top_k=1`, `save_last=True`,
  `filename="best-{epoch:02d}-{val_loss:.4f}"`), `EarlyStopping(monitor="val_loss",
  patience=cfg.patience)`, a `CSVLogger`, and `gradient_clip_val=1.0`. After `fit`,
  reload the best-val checkpoint's weights into the in-memory module.
- When `enable_checkpointing` is false (smoke test): current behavior — no callbacks,
  `logger=False`. `EarlyStopping`/`CSVLogger` also gated on `enable_checkpointing`
  so the flat smoke cfg (which lacks these keys) is unaffected.

`<run_dir>` is obtained from `HydraConfig.get().runtime.output_dir` when running under
`@hydra.main`; falls back to `cfg.get("output_dir", ".")` otherwise.

New config keys (in `train_partial_trajs.yaml`): `patience: 20`,
`gradient_clip_val: 1.0`, `log_every_n_steps: 10`. All read via `cfg.get(...)`.

### C. Evaluation script + config

New `adaptive_roa/partial_trajs/verifier/evaluate.py` with `@hydra.main` on
`configs/partial_trajs/evaluate_partial_trajs.yaml`.

`evaluate_partial_trajs.yaml` composes the same `system:` group (→ `hidden_dims`,
`attractor_radius`, `dataset_dir`, `eval_split_file`) and adds:
- `run_dir` (training run dir containing `checkpoints/`) OR `checkpoint` (explicit
  `.ckpt`; if null, auto-pick `<run_dir>/checkpoints/best*.ckpt`, else `last.ckpt`).
- `backend` (must match training; default `generative`), plus the backend construction
  knobs (`hidden_dims`, `latent_dim`, `num_integration_steps`) — composed the same way
  as training so `build_model` reconstructs an identical architecture.
- `eval_split` (`test` | `cal` | `eval`; default `test`).
- `num_samples` (null ⇒ single deterministic rollout; `>1` ⇒ probabilistic).
- `radius` (default `${attractor_radius}` — the adaptive per-system radius).
- `max_eval_rows` (null ⇒ all), `export_predictions` (default true).
- Output dir: `<run_dir>/eval/<split>/` (configurable).

Flow:
1. Build the verifier system via `make_verifier_system(cfg.system, cfg.system_dataset_dir)`.
2. Build the model via `train.build_model(cfg, system)`; `torch.load` the checkpoint
   and `load_state_dict(strict=False)` (the models don't `save_hyperparameters`, so
   `load_from_checkpoint` cannot reconstruct constructor args — mirrors `ClassifierTrainer`).
3. `load_dataset_description(dataset_dir)` → `K = autoregressive_K`, `state_dim`.
4. Resolve the eval file: `<dataset_dir>/<eval_split_file>` for the chosen split
   (`test_set.txt` / `cal_set.txt` / `eval_states.txt`; humanoid `_fps` variants).
   `load_eval_states(path, state_dim)` → `(init, terminal, label)`.
5. Roll out (deterministic: `resolve_outcome` + `rollout_final_state`; probabilistic:
   `resolve_probabilistic`). Compute metrics + per-query arrays.
6. Write `metrics.json` and (if enabled) `predictions.npz`.

The existing `evaluate_roa` function and its 3 tests are left intact; the new driver
composes the lower-level rollout primitives + metrics helpers so it can emit both the
metric bundle and per-query arrays.

### D. Metrics

#### D.1 ROA success-detection (extends `roa_scores`, add-only keys)

`roa_scores(pred_labels, true_labels)` currently returns
`precision/recall/f1/accuracy/unresolved_frac`. **Add** (never rename existing keys,
so `test_evaluate.py` stays green):
- `tp`, `fp`, `fn`, `tn`, `n_total`
- `n_pred_success`, `n_pred_failure`, `n_pred_unresolved`
- `f1_resolved_only`, `precision_resolved_only`, `recall_resolved_only` — success
  detection restricted to the `pred != 0` (resolved) subset. This is the analog of the
  adaptive "confident-set" metrics; the plain `f1` remains the full-coverage /
  safety-aware flavor (unresolved counted as not-success).

#### D.2 Rollout final-state error (metric #2), stratified

New helper `stratified_error_report(err, gt_label, pred_label, terminal_class, motion)`
in `eval/metrics.py`. Each leaf reports `{mean, median, p90, max, n}`; only non-empty
buckets are emitted. Classes:
- **GT class** (dataset label): `gt_success` (label==1), `gt_nonsuccess` (label==0).
- **GT terminal class** (`system.classify_attractor(terminal_GT)`): `success` /
  `failure` / `unresolved`. Separates GT non-success into OOB failure vs
  timeout/unresolved for failure-set systems; always unresolved for pendulum/humanoid.
- **Predicted class** (verifier output): `pred_success` (+1) / `pred_failure` (−1) /
  `pred_unresolved` (0).

Structure in `metrics.json`:
```
rollout_final_state_error:
  overall:        {mean, median, p90, max, n}
  by_gt:          gt_success / gt_nonsuccess
  by_gt_terminal: success / failure / unresolved
  by_pred:        pred_success / pred_failure / pred_unresolved
  by_gt_x_pred:   gt_success__pred_success, gt_success__pred_unresolved,
                  gt_nonsuccess__pred_success (false alarms), … (up to 2×3 = 6 cells)
  by_motion:      low / high        # split at the median of manifold ‖final − init‖
```

`motion` is the manifold-aware distance `‖final − init‖` (reusing
`manifold_state_distance` with the system's circular indices).

#### D.3 Probabilistic (generative + `num_samples>1` only)

`p_success_mean` in `metrics.json`; per-query `p_success/p_failure/p_unresolved` in
`predictions.npz`.

#### D.4 `metrics.json` layout

```json
{
  "metadata": {
    "system": "...", "backend": "...", "dataset_dir": "...", "eval_file": "...",
    "horizon_T": 25, "K": 20, "radius": 0.1, "num_samples": null,
    "checkpoint": "...", "n_eval_rows": 48770, "in_sample": true, "seed": 0
  },
  "roa": { "f1": ..., "precision": ..., "recall": ..., "accuracy": ...,
           "tp": ..., "fp": ..., "fn": ..., "tn": ..., "n_total": ...,
           "n_pred_success": ..., "n_pred_failure": ..., "n_pred_unresolved": ...,
           "unresolved_frac": ..., "f1_resolved_only": ..., ... },
  "rollout_final_state_error": { ...stratified as above... },
  "p_success_mean": ...   // present only for probabilistic runs
}
```

#### D.5 `predictions.npz` fields

`init` (N, D), `pred_label` (N,), `true_label` (N,), `final_state` (N, D)
[rolled-out terminal], `terminal_class` (N,) [`classify_attractor(terminal_GT)`], and
for probabilistic runs `p_success` / `p_failure` / `p_unresolved` (N,).

## Testing

Preserve all 38 existing tests. Add:

- **`roa_scores` extension:** new keys present and correct on a hand-built confusion
  case (incl. unresolved rows); `f1_resolved_only` differs from `f1` when unresolved
  rows exist; existing keys unchanged.
- **`stratified_error_report`:** buckets partition the samples; `by_gt_x_pred` cells sum
  to `overall.n`; empty buckets omitted; per-leaf stats correct on a small fixture.
- **Config composition:** each `system/*.yaml` composes with each `backend/*.yaml` and
  yields the expected flat keys (`hidden_dims`, `attractor_radius`, `dataset_dir`
  ending in `_T25`, `eval_split_file`); humanoid → `test_set_fps.txt`; quad3d/humanoid
  generative `lr == 5e-4`. Uses Hydra `compose` (no dataset dependency).
- **Eval driver assembly:** with a stub model/system and a tiny temp eval file, the
  driver produces a `metrics.json` dict with the documented sections and a
  `predictions.npz` with the documented arrays. Reuses the `test_evaluate.py` stubs.
- **Checkpoint smoke (dataset-gated, like `test_train_smoke`):** a 1-epoch run with
  `enable_checkpointing=true` writes `checkpoints/best*.ckpt` and `last.ckpt`, and eval
  loads that checkpoint and returns a populated `metrics.json`.

## Risks / notes

- **Radius mismatch:** datasets were labeled with their own success criterion (pendulum
  L2<0.075) but eval uses the adaptive `attractor_radius` (pendulum 0.1) per the chosen
  design. Predicted resolutions therefore use a slightly larger goal region than the GT
  labels were built with; this is intentional (apples-to-apples with adaptive ROA) and
  recorded in `metadata.radius`.
- **Editable install:** work happens in-place on the main working directory (not a
  worktree) because `adaptive_roa` is `pip install -e .`; a worktree's edits would not
  be importable without reinstall.
- **quad3d/humanoid deterministic lr:** the per-system `lr: 5e-4` mirrors the generative
  FM runs. The deterministic-classifier analog used 1e-3; override on the CLI
  (`lr=1e-3`) when running those systems deterministically.
```
