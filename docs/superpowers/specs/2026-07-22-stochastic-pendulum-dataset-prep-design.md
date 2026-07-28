# Stochastic Pendulum Dataset Prep — Design

**Date:** 2026-07-22
**Status:** Approved

## Goal

Prepare the four stochastic pendulum LQR datasets for adaptive flow-matching
training, in a layout parallel to the deterministic `pendulum/lqr` dataset but
npz-backed (no per-trajectory text files).

## Source

`/common/users/shared/pracsys/genMoPlan/data_trajectories/noisy/pendulum_lqr_stoch-dyn-ctrl-{low,med,high,xhigh}/`

Each: `trajectories.npz` (per-grid-cell arrays: `states_{i}`, `starts_{i}`,
`offsets_{i}`, `seeds_{i}`, `terminated_{i}`, `truncated_{i}`, `cells_low/high`,
`cell_lengths`), `extras.npz` (actions), `dataset_description.json` (includes
`achieved_bounds`). 49,770 unique grid start states × 10 rollouts = 497,700
trajectories per level. Success = `terminated` flag; timeout = failure; zero
non-timeout failures in the source data. Source dirs are left untouched.

## Target layout

Per noise level ℓ ∈ {low, med, high, xhigh}:

```
noisy/pendulum/lqr/ℓ/
├─ train.npz                      # 100,000 training rollouts (10,000 starts × 10)
├─ eval.npz                       # 397,700 eval rollouts (39,770 starts × 10)
├─ eval_states.txt                # 39,770 rows: θ_s, θ̇_s, p_success (comma-sep)
├─ cal_set.txt                    # first 10,000 eval starts (shuffle order), same 3 cols
├─ test_set.txt                   # remaining 29,770 eval starts, same 3 cols
├─ train_test_splits/
│  ├─ shuffled_indices_0.txt      # training-pool order: rollout ids into train.npz
│  └─ shuffled_labels_0.txt      # aligned binary labels (1=success, 0=fail/timeout)
└─ dataset_description.json       # source description + prep provenance section
```

## Split

- One shuffle of the 49,770 unique start states, fixed recorded seed.
  First 10,000 starts → train (their 100k rollouts); remaining 39,770 → eval.
- **Same start split shared across all four noise levels.** Precondition
  (verified before writing): the grid start states are identical across levels.
- Cal/test: first 10,000 eval starts in shuffle order → `cal_set.txt`;
  remaining 29,770 → `test_set.txt`.
- All splits are at the start-state level: a start's 10 rollouts always stay
  in the same split (no train/eval or cal/test leakage).

## Training-pool ordering

`shuffled_indices_0.txt` is shuffled at the **start-state level**: rollout ids
grouped in contiguous blocks of 10, block order = shuffled start order. The
acquisition unit in the adaptive loop is a start state; acquiring one yields
all 10 rollouts (the outcome-distribution sample the probabilistic FM model
needs). Consequence: `initial_train_size`/`samples_per_epoch` are counted in
trajectories, so they should be scaled ×10 (or reinterpreted in starts) when
wiring up training — handled in the follow-up loader task, not here.

## npz internal format (train.npz and eval.npz)

Flat concatenated layout (no grid cells):

- `states` (ΣT, 2) — concatenated rollout states
- `offsets` (N+1,) — rollout boundaries into `states`
- `starts` (N, 2) — initial state per rollout
- `labels` (N,) — binary success per rollout
- `start_ids` (N,) — unique-start index per rollout (global 0..49,769 id)
- `seeds` (N,) — per-rollout seed

Rollout id in `shuffled_indices_0.txt` = row index into these arrays.

## Labels and probabilities

- Per-rollout binary label = `terminated` flag (1 success, 0 otherwise).
- `p_success` per eval start = mean of its 10 rollouts' labels.
- `eval_states.txt` / `cal_set.txt` / `test_set.txt` rows:
  `θ_s, θ̇_s, p_success` (comma-separated). Note: 3-column probabilistic
  format, unlike the deterministic 5-column cal/test files — conformal
  calibration code must handle this in the follow-up task.

## dataset_description.json

Copy of the source description plus a `prep` section: shuffle seed, split
sizes, file inventory/format, provenance (source dir, date).

## Verification

Per level, after writing:

- Counts: train 100,000 / eval 397,700 rollouts; eval_states 39,770 rows;
  cal 10,000 / test 29,770; cal ∪ test = eval_states exactly, no overlap.
- No start-id overlap between train and eval.
- Labels match source `terminated` flags; round-trip a few rollouts'
  states against the source npz.
- shuffled_indices blocks of 10 share a start_id; block order matches the
  recorded shuffle.
- Overall mean p_success ≈ source success rate (38.8–47.9% by level).

## Out of scope (follow-up task)

- npz-backed `TrajectoryDataSource` loader.
- Hydra `noise_level` knob in `configs/adaptive_v2/system/pendulum.yaml`
  and start-unit interpretation of pool sizes.
- Conformal calibration handling of probabilistic 3-column cal/test files.
