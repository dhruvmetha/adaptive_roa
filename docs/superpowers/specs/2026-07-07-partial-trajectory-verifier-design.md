# Partial-Trajectory T-Step Dynamics Verifier — Design Spec

**Date:** 2026-07-07
**Status:** Approved (design), pending implementation
**Package:** `adaptive_roa/partial_trajs/`

## 1. Motivation

The existing ROA pipeline trains models that map a start state directly to a
*resolved* outcome (endpoint / classification data). This requires every
training trajectory to have a known success/failure label. We want a pipeline
that can handle **partial trajectories with unresolved outcomes**: instead of
predicting the outcome directly, learn a **T-step forward dynamics model**
`f: x_t -> x_{t+T}` and apply it **autoregressively as a reachability
verifier** — roll forward from a query state until it is absorbed by the
success set or the failure set, and read off the outcome.

The datasets for this already exist at
`${data_dir}/partial_deterministic/<system>/<system>_T<T>` for
`system ∈ {pendulum_lqr, cartpole_pybullet, quadrotor2D_rl, quadrotor3D_lqr,
humanoid_get_up_medium}` and `T ∈ {10, 25, 50, 100}`.

### Primary goal (this spec)

**A — a drop-in ROA verifier.** Train the dynamics model, roll it out K times
from a query state, check absorption, and compare ROA-classification quality
(and data efficiency) against the existing endpoint / classifier baselines.

### Deferred (future phase, keep seams clean)

**C — uncertainty-driven adaptive data collection.** The verifier's
per-query uncertainty becomes an acquisition function feeding `adaptive_v2`.
Not built now, but the verifier exposes an uncertainty hook so it can be added
without redesign.

## 2. Dataset structure (as shipped)

Each `<system>_T<T>` directory contains:

- `train_splits/horizons.npy` — structured array, one row per horizon:
  `traj_id (i4), start (i4), end (i4), is_freeze (i1), label (i1),`
  plus a **motion field**: `abs_dtheta, abs_dthetadot (f4)` for pendulum,
  `jump_mag (f4)` for all other systems.
  `input = states_cache[traj_id][start]`, `target = states_cache[traj_id][end]`.
- `train_splits/states_cache.npz` — `float32` states keyed by `str(traj_id)`,
  for the training pool only.
- `_shared/` (symlinked into each variant) — `eval_states.txt`, `cal_set.txt`,
  `test_set.txt`, and the base `trajectories/`.
- `dataset_description.json` — authoritative metadata (see below).

### `dataset_description.json` schema (post-cleanup)

- `source_system`: `state_dim`, `state_order`, `manifold`, `angle_dims`,
  `quaternion_dims`, `goal_state`, `success_criterion`, `failure_criterion`,
  `dataset_outcome_counts`.
- `horizon_model`: `T_steps`, `T_seconds`, `resolution_horizon_l_steps` (l),
  `autoregressive_calls_K` (K ≈ l / T), `resolution_definition`, `freeze_rule`.
- `training_pool`: `source`, `size`, split note (train/val split by traj id at
  load time).
- `eval_sets`: `reused_as_is`, `in_sample`, `files`, and (humanoid only) `note`.
- `horizons`, `states_cache`, `trajectory_length_stats_states`.

### Key semantics

- **Freeze = freeze-on-resolution.** `is_freeze=1` means the horizon's target is
  an **absorbing resolved state**. A trajectory resolves at success **or**
  failure; both are absorbing. Pendulum and humanoid have **no failure
  criterion**, so for them resolution = success only.
- **Motion imbalance.** ~80% of horizons are low-motion near-attractor steps.
  The motion field is provided so training can, optionally, reweight toward
  high-motion transitions (see §4, option A — deferred).
- **Eval protocol differs by system, accepted as-is (option B).**
  Pendulum / cartpole / quad2D / quad3D ship **in-sample** eval
  (`in_sample: true`; eval spans the training pool). Humanoid ships
  **out-of-sample** eval (`in_sample: false`; native disjoint 150k/650k split).
  The pipeline consumes whatever each dataset ships; no held-out reconstruction.

### Cross-system inconsistencies to be aware of (intentional, not bugs)

- **Freeze convention:** pendulum/humanoid freeze only successful resolutions
  (freeze labels 100% success; horizon count decreases with T). cartpole/quads
  freeze all resolutions incl. failure (mixed labels; count ~constant with T).
  Consistent in *meaning* (target = resolved terminal state).
- **Motion field name:** `abs_dtheta/abs_dthetadot` (pendulum) vs `jump_mag`
  (others). The loader normalizes both to a single scalar "motion" column.
- **Resolution criteria source of truth:** the **`systems/` classes are the
  single authoritative source** for success/failure sets (`classify_attractor`).
  The dataset-description criteria were reconciled against the classes and the
  classes were updated to the agreed values (see §2b). The verifier calls
  `classify_attractor` directly.

## 2b. Resolution criteria (finalized) & unresolved labeling

Success/failure sets used by the verifier come from `system.classify_attractor`
(`+1` success, `-1` failure, `0` separatrix/unresolved). The base classes were
reviewed per-system against the dataset resolution rules and updated as follows:

| System | Success | Failure | Base change |
|---|---|---|---|
| pendulum | L2 `< 0.1` to `[0,0]` (circular θ) | top equilibria `[±2.1,0]` @ `0.1` | none |
| cartpole | L2 ball `< 0.1` | `\|x\|>5.9, \|ẋ\|>4.9, \|θ̇\|>4.9` | none |
| quad2D | L2 `< 0.3` | `\|x\|>0.9, z<0.2, z>1.4, \|ẋ\|>0.9, \|ż\|>0.9, \|θ̇\|>7.5` | **failure updated** |
| quad3D | Euclid `< 0.15` in **raw 13-D quaternion** state | `\|x\|>1.7, \|y\|>1.7, z<0.2, z>2.9, \|vel\|>2.9, \|rate\|>23.5` | **success (metric+radius) + failure updated** |
| humanoid | `head[21] ≥ 1.2 ∧ ‖CoM vel[37:40]‖ ≤ 0.3` | none (base returns binary `-1`) | **success thresholds relaxed** |

(quad3D success now uses raw 13-D Euclidean distance to the identity-quaternion
goal, assuming `qw ≥ 0`-canonicalized states — replacing the old Euler-space
`< 0.05` check.)

**Unresolved labeling via pure-relabel subclasses.** Pendulum and humanoid have
**no failure set** in the datasets, but their base `classify_attractor` returns
`-1` for non-success states (pendulum: top equilibria; humanoid: everything not
success). The other three systems already return `-1` only for a genuine
out-of-bounds failure set and `0` for the unresolved region, so they need no
change. The `partial_trajs` package therefore adds two thin subclasses that carry
**no criteria logic** — they call the parent and remap `-1 → 0`:

```python
class PartialTrajPendulumSystem(PendulumSystem):
    def classify_attractor(self, state, radius=...):
        labels = super().classify_attractor(state, radius)
        labels[labels == -1] = 0     # no failure set -> unresolved
        return labels
# (same pattern for HumanoidStandUpReachSystem)
```

Cartpole / quad2D / quad3D use their base class directly in the verifier.

**Reevaluation impact:** the base-class edits change `classify_attractor` for
quad2D / quad3D / humanoid, so existing ROA results for those three systems that
were scored via `classify_attractor` will shift and need re-running. Pendulum and
cartpole are unaffected.

**Legacy cleanup (done):** the obsolete `humanoid.py` FM variant (its
`HumanoidSystem`, `flow_matching/humanoid/`, `humanoid_endpoint_data.py`, and the
`train_humanoid` / `system/humanoid` / archived configs — 9 paths total) was
deleted; it was superseded by `humanoid_standup_reach` and had no active
consumers.

## 3. Architecture

New sibling package mirroring `flow_matching/` + `adaptive_v2/`. Dynamics model
is a pluggable backend behind a shared verifier; all manifold/orientation logic
is delegated to `systems/`. Success/failure resolution uses `classify_attractor`
on the base systems, with the two `partial_trajs` relabel-subclasses (§2b) for
pendulum and humanoid.

```
adaptive_roa/partial_trajs/
  data/
    horizon_dataset.py   # reads horizons.npy + states_cache.npz; traj-level
                         # train/val split; yields (x_start, x_end, is_freeze,
                         # label, motion); uniform sampling (B) default,
                         # motion-weighted/stratified (A) behind a flag (off).
    description.py       # parses dataset_description.json -> T, K, l, goal,
                         # success/failure criteria, manifold/angle/quaternion.
  model/
    base.py              # DynamicsModel interface:
                         #   predict(x)  -> x_next   [deterministic]
                         #   sample(x,n) -> x_next   [generative]
    regressor.py         # deterministic manifold-aware regressor (Lightning)
    generative.py        # generative backend reusing flow_matching/base stack
  verifier/
    rollout.py           # roll K steps; absorbing success/failure sets via
                         # system.classify_attractor; early-stop; label in
                         # {success, failure, unresolved}; det=1 rollout,
                         # gen=N rollouts -> p(success). Exposes uncertainty
                         # hook for future adaptive (C).
    evaluate_roa.py      # unified verifier eval -> ROA metrics + dynamics
                         # accuracy (#1) + rollout error (#2) into one report.
  train.py               # system-agnostic Hydra entrypoint
  __init__.py

configs/partial_trajs/
  model/                 # reuse adaptive_v2 model/{family,system_dims}
  data/                  # dataset path + description wiring
  trainer/
  verifier/
  train_partial_trajs.yaml
  evaluate_partial_trajs_roa.yaml
```

## 4. Dynamics model & training

### Data path

`HorizonDataset` memory-maps `horizons.npy` and loads `states_cache.npz`.
`__getitem__` returns `x_start = states[traj_id][start]`,
`x_end = states[traj_id][end]`, plus `is_freeze`, `label`, `motion`.
**Train/val split is by `traj_id`** — no horizon from a val trajectory appears
in train. `description.py` supplies `T`, manifold/angle/quaternion dims, goal,
and success/failure criteria so the loader/model stay system-agnostic.

### Backbones — match the adaptive experiments

The adaptive experiments' main method is the generative flow-matching endpoint
predictor (`configs/adaptive_v2/predictor/generative.yaml` ->
`FlowMatchingTrainer`), with a swappable velocity-net **family**
(`simple_mlp` / `adaln` / `dit` / `unet`) selected per-system in
`configs/adaptive_v2/model/{family,system_dims}`. The adaptive runs use plain
settings: `use_manifold: false`, `use_loss_weights: false`, `zero_latent: true`.

- **Generative backend** = the adaptive FM stack, re-pointed from (start ->
  final) to (`x_t` -> `x_{t+T}`) horizon pairs. Same families, same
  `FlowMatchingTrainer` machinery, same per-system dim configs, same flags.
- **Deterministic backend (default)** = the same backbone families as a direct
  regressor (reuse family modules; FM time/latent inputs collapse via the
  existing `zero_latent` path or a thin regression head).
- **Orientation/manifold** = delegated entirely to `systems/`
  (`embed_state_for_model`, `get_loss_weights`, quaternion/SO(3) handling,
  `classify_attractor`). No new manifold code.

Config reuses `configs/adaptive_v2/model/{family,system_dims}` by reference
rather than duplication.

### Target representation & sampling (default = B)

- **Default:** predict **absolute `x_{t+T}`**, **uniform sampling**, plain
  manifold-aware regression loss. Freeze horizons trained on like any other row
  (their target is simply the absorbing state). Simplest thing that fits the
  deterministic data.
- **Documented future options (wired but off):**
  - **A — motion-stratified / weighted sampling** using the motion field, with a
    tunable freeze fraction, to counter the ~80% low-motion imbalance.
  - **C(target) — predict the manifold-aware delta `x_{t+T} - x_t`** instead of
    absolute, to concentrate loss on real motion.
  Add these only if rollout accuracy (see §5) demands it.

## 5. Verifier & evaluation

### Rollout

`Verifier.rollout(x0, K)`: apply the dynamics model (`predict` deterministic /
`sample` generative); after each of K steps check the **absorbing sets** via
`system.classify_attractor` (success=+1, failure=-1, unresolved=0);
**early-stop** on +1 or -1; if K elapse with only 0s -> **unresolved**. The
`system` is the base class for cartpole/quad2D/quad3D and the §2b relabel
subclass for pendulum/humanoid (so their non-success states are 0, never an
absorbing failure). Criteria come from `classify_attractor` (single source of
truth, §2b).

### Outcome aggregation

- Deterministic -> one rollout -> label ∈ {success, failure, unresolved}.
- Generative -> N rollouts -> `p(success) = #success/N` plus resolved/unresolved
  fractions — the same probabilistic shape the conformal stack consumes.

### Metrics (one report / CSV, per epoch and per T)

1. **ROA classification metrics** — F1, accuracy, separatrix/unresolved %, over
   each dataset's shipped `eval_states.txt` / `cal_set.txt` / `test_set.txt`
   (in-sample where that is what ships). Reuses the `full_roa`-style machinery
   and (generative case) the conformal calibration path. Drops directly into the
   existing baseline-comparison plots (Classification / Conformal / final-state
   predictor / NeuroMancer).
2. **Dynamics accuracy — per-horizon T-step error (#1)** — manifold-aware
   distance between predicted `x_{t+T}` and the ground-truth horizon target,
   reported overall and **stratified by (freeze vs non-freeze)** and **by motion
   magnitude**. Computed on the val split (where `states_cache` gives the true
   target directly).
3. **Dynamics accuracy — rollout final-state error (#2)** — manifold-aware
   distance between the K-step rollout terminal and the trajectory's true
   terminal state. Diagnoses compounding drift and bridges #1 and the ROA F1.

### Adaptive seam (deferred)

`rollout.py` exposes a per-query uncertainty hook (rollout disagreement /
`p(success)` margin / unresolved flag) for a future `adaptive_v2` acquisition
strategy. No design change required to add C later.

## 6. Configs & entrypoints

- `adaptive_roa/partial_trajs/train.py` — system-agnostic Hydra entrypoint
  (`system=…`, `backend=deterministic|generative`, `horizon_T={10,25,50,100}`).
- `adaptive_roa/partial_trajs/verifier/evaluate_roa.py` — unified verifier eval
  producing metrics #1–#3 above.
- `configs/partial_trajs/` mirrors `adaptive_v2`, reusing `model/` groups; the
  two headline switches are `backend` and `horizon_T`.

## 7. Phasing (each phase independently runnable/verifiable)

1. **Data + deterministic model + #1.** `horizon_dataset.py`, `description.py`,
   `regressor.py`; train on **pendulum**; verify per-horizon T-step error (#1)
   on the val split.
2. **Verifier + ROA eval + #2.** `rollout.py`, `verifier/evaluate_roa.py` on
   pendulum; ROA metrics + rollout final-state error; compare to baselines.
3. **Generalize** to cartpole / quad2D / quad3D / humanoid (all system-agnostic;
   humanoid is the only out-of-sample eval).
4. **Generative backend** (`generative.py`) reusing the FM stack.
5. *(later)* **Adaptive loop C** via the uncertainty seam.

**First system:** pendulum — richest metadata, cheapest to iterate, reference
dataset.

## 8. Non-goals

- No adaptive/acquisition loop in this spec (C is deferred; only the seam).
- No held-out-eval reconstruction for the in-sample systems (option B accepted).
- No new manifold math — reuse `systems/`.
- No motion-weighted sampling or delta-target in the default (A / C(target)
  documented, wired off).
- Noisy-regime datasets are the eventual motivation for the generative backend
  but are out of scope for the deterministic-first phases.
