# Stochastic quadrotor **3D** — adaptive acquisition campaign

Written 2026-08-20, split out of the shared quadrotor README 2026-08-22.
**Scope: quad3D experiments on the `stochastic/` dataset tree only.** quad2D lives in
[`../quadrotor2d/`](../quadrotor2d/) and shares none of this file's numbers.
**Status: `noisy_dynamics` PAUSED, `corridor_sine_ambient f_0.30` IN FLIGHT.**
Header last refreshed **2026-08-23 15:5x**.

**PAUSED 2026-08-23 15:10 — `noisy_dynamics`.** All 26 `q3d_nd` jobs were cancelled on user
instruction to free the quota for the corridor family. Cancel is a pause, not a loss: every run
dir survives and `scripts/resume_adaptive.py` restarts each arm from its last completed epoch.
**109 f_0.048 artifacts are intact**; per-arm depths at the moment of pause are recorded in
[`q3d_nd_depths_at_pause.txt`](q3d_nd_depths_at_pause.txt) (`partx` and `clf_dir00` had already
finished 18/18). f_0.060 had reached 0 artifacts — its one running arm was 8% into the first eval.

**LAUNCHED 2026-08-23 15:10 — `corridor_sine_ambient f_0.30`**, 14 arms, jobs 239289-239306.
**One level at a time**, per user direction; f_0.30 was chosen over f_0.25 on `frac_mid`
(0.149 vs 0.124 — see the re-measured pool table below). The f_0.25 fleet was submitted and then
cancelled within ten minutes; its 14 run dirs are kept, and `q3d_cs025_partx` holds 4 real
artifacts from that window.

**Memory is split by evidence on this family, not the blanket 8G:** `epi_var_anch`, `epi_bald`,
`yield_a1` and `yield_mlp` each hit a host-RAM OOM at 8G on f_0.048 (pair count outgrows
trajectory count because acquisition selects long trajectories — `epi_bald` held 4,204,245 pairs
against `dir00_s42`'s 1,526,054 at the same 30,000 trajectories), so those four run at 64G on
corridor and the other ten at 8G.

**`partx` needs an explicit large-VRAM GPU.** On corridor it died at epoch 11 with a **CUDA** OOM
(not host RAM — MaxRSS was 6.2 GB against 8G, so `--mem` is the wrong dial). Generic
`--gres=gpu:1` gave it a 16 GB A4000; the f_0.048 run drew a 20 GB A4500 and survived all 18
epochs. The GP's variational backward grows with the training set, so the difference is card luck.
Resumed on `--gres=gpu:a100:1` with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

**Superseded launch (2026-08-22 10:21)** — `noisy_dynamics` f_0.048 and f_0.060, 14 arms each,
**jobs 226514-226541**, `--mem=8G`. The 2026-08-21 fleet (jobs 218355 + 218378-218458) is DEAD and
its 28 output directories were deleted (~19 GB). Two things were wrong with it: 20 of 28 arms
crashed at first acquisition (see the start-state section below), and it inherited quad2D's
budget, which is wrong for this system.

**Budget: `initial_train_size=10000`, `samples_per_epoch=5000`, `n_epochs=18` = 100,000
trajectories**, 12.5% of the 800k pool, with `acquisition.n_candidates=250000` (top-2%
selectivity; the inherited 50,000 would have been top-10% at this step size) and `K_acq=20`
unchanged. **Note the epoch count: quad3D caps at 14 × 18 = 252 per level, NOT 336.**

Measured per-epoch time on rlab7: epoch 0 2h11m, epoch 1 1h55m, so ~35 h for an 18-epoch arm.
rlab4 runs the full-ROA eval at 7.6 it/s against rlab7's 12.7, so arms placed there are closer to
45-50 h. Throughput on a given node varies by more than 2× with contention; do not project arm
duration from a single eval sample.

**Memory: use `--mem=8G`.** The inherited 40G is ~10× measured need (MaxRSS 3.7-4.4 GB on the FM
arms, whose pool is memory-mapped rather than resident) and cannot be placed on the iLab nodes
that actually have free GPUs (ilab1/2/3 had 6/9/12 GB free RAM against 10 idle cards). One
exception measured 2026-08-22: `partx` finished 18 epochs at **MaxRSS 7.6 GB**, just under the
cap, because the GP path holds the candidate pool resident. 8G works for it but has almost no
headroom — ask for 12G if a future `partx` arm is sized fresh.

**Scoring is live and incremental**, same machinery as quad2D:
`scripts/score_stoch_incremental.py` diffs disk against the committed CSV each monitoring cycle
and scores only the difference.

- **Data root** — `${data_dir}/stochastic/quadrotor3D/<family>/lqr/<level>`
- **Experiment dir** — `/common/users/shared/pracsys/adaptive_roa_experiments/quadrotor_stoch/`
  (run dirs prefixed `q3d_nd048_*` and `q3d_nd060_*`)
- **Config** — `configs/adaptive_v2/system/quadrotor3d_stoch.yaml`
- **Run log** — `runs_quad3d.jsonl` beside this README, 28 records, with a sibling
  `runs_quad3d.csv`. Append-only, same rules as quad2D: a resume is a NEW record. Append with
  `scripts/exp_log.py append --log docs/experiments/stochastic/quadrotor3d/runs_quad3d.jsonl …`.
  `epochs_on_disk` counts `artifacts_v2.json`, never `epoch_*` directories.
- **Figures and data** — `quad3d_noisy_dynamics_all_levels.{csv,png}` plus a `_clean.png`
  variant, with one panel per noise level.

---

## 1. What exists, and the two noise families

**These datasets are new and had never been run before 2026-08-20.** Neither system class could
load them (see §4), so the campaign starts from a fixed codebase, not from prior results.

| system | family | levels | controller | dim | n_traj |
|---|---|---|---|---|---|
| quad3D | `noisy_dynamics` | f_0.000, 0.032, 0.048, 0.060, 0.072 | LQR | 13 | 800k |
| quad3D | `corridor_sine_ambient` | f_0.25, f_0.30 | LQR | 13 | 800k |

**Bold = launched.** "Mid by disturbance magnitude" per level ordering.

**`noisy_dynamics`** — `F ~ U(-f, +f)` applied via `applyExternalForce` at the COM link, world
frame, zero-order hold redrawn every control step (100 Hz), **no torque**. **Uniform, not
Gaussian.** Flagged `"matched": false` in the dataset description: *"NOT a wind model: the force
is independent of the drone velocity, produces no moment, and is white in time."*

**`corridor_sine_ambient`** — `F_x = sigma(z)*(0.5 + 0.5*A*sin(2*pi*t/period + phi)) + N(0, ambient)`
with `A ~ U(0,1)` and `phi ~ U(-pi,pi)` drawn **once per rollout**. The corridor term is therefore
**coherent over the episode, not white** — given `(A, phi)` it is a deterministic function of
`(z, t)` — and it is **one-sided (+x)**, gated by altitude. Only the ambient term is per-step.

**The families are not comparable to each other**, to the other system, or to the
pendulum/cartpole `gaussian_signal` family: different distribution (uniform vs Gaussian),
injection point (external COM force vs actuator command), and temporal structure (white vs
per-rollout coherent). quad3D's corridor levels are also ~1× body weight against quad2D's 0.755
maximum, so even within one family the two systems are not on a common scale.

---

---

## 2. Measured pool structure (from the shipped `train.npz`)

**quad3D rows re-measured 2026-08-21** after dm1487 rewrote the corridor `train.npz` (08-20 00:02)
and both corridor `eval_success_prob.npz` (08-20 16:10). The corridor `frac_mid` values moved:
f_0.25 0.108 → **0.124**, f_0.30 0.130 → **0.149**. Trajectory-level stats are unchanged, so the
rewrite altered the eval grid, not the pool. quad2D rows and quad3D `noisy_dynamics` rows are
unchanged from the 2026-08-20 measurement (those files have not been touched since 08-15).

The length–outcome coupling runs the **cartpole** way — long == success — not the pendulum way:

| level | traj_succ | L_succ | L_fail | PAIR_succ | frac_mid |
|---|---:|---:|---:|---:|---:|
| 3D noisy_dynamics f_0.032 | 0.067 | 575.9 | 17.3 | 0.706 | 0.041 |
| 3D noisy_dynamics f_0.048 | 0.051 | 661.8 | 33.2 | 0.519 | **0.230** |
| 3D noisy_dynamics f_0.060 | 0.034 | 691.4 | 50.4 | 0.324 | **0.230** |
| 3D noisy_dynamics f_0.072 | 0.020 | 705.6 | 63.2 | 0.187 | 0.225 |
| 3D corridor f_0.25 | 0.052 | 563.0 | 16.9 | 0.647 | 0.124 |
| 3D corridor f_0.30 | 0.043 | 562.9 | 17.8 | 0.586 | 0.149 |

`frac_mid` = fraction of eval cells with `p` in (0.05, 0.95) — the separating power available to
KL/sAUROC. **Pendulum's is 0.286.** quad2D is THIN (max 0.103), so expect smaller gaps there and
judge everything against the 3-seed floor rather than nominal size. **quad3D `noisy_dynamics` at
f_0.048+ reaches 0.230**, comparable to the pendulum, and is where the real separating power is.

**Trajectory success is far rarer than either prior system** (0.020–0.079 vs cartpole's 0.100–0.175
and pendulum's 0.403). This is a new regime, not an interpolation of the two.

**Excluded from any probabilistic comparison:** `2D noisy_dynamics f_0.000`,
`2D corridor baseline`, `3D noisy_dynamics f_0.000` — all measure `frac_mid = 0.000`, i.e. no
ground-truth probability mass at all. They are the deterministic references for their families.

---

---

## 3. Campaign design

14 arms per level, the same arm table as quad2D and cartpole v2 so the campaigns are directly
comparable:

| arm | acquisition | d2_ratio | predictor |
|---|---|---|---|
| `dir00_s42/s43/s44` | direct (uniform) | 0 | fm_ensemble |
| `epi_var`, `epi_bald` | decomp_epi_var / _bald | 1.0 | fm_ensemble |
| `epi_var_anch` | decomp_epi_var | **0.5** | fm_ensemble |
| `yield_a1` | decomp_epi_var_yield (α=1) | 1.0 | fm_ensemble |
| `yield_mlp` | decomp_yield_mlp | 1.0 | fm_ensemble |
| `partx` | partx | 1.0 | **gp** |
| `clf_dir00` | direct | 0 | clf_ensemble |
| `clf_yield`, `clf_epi_var`, `clf_epi_bald` | as named | 1.0 | clf_ensemble |
| `clf_epi_var_anch` | decomp_epi_var | **0.5** | clf_ensemble |

Budget as above: 10,000 / 5,000 / **18 epochs** → 100,000 final trajectories. 5 ensemble members,
`filter_confident_pairs=false`.

**Why this budget and not quad2D's.** The minority class is what limits learning here: at
f_0.048 a 2,000-trajectory start holds ~102 successes against ~510 at 10,000. Wall-clock barely
moves with initial size because the epoch is eval-dominated (990,000 states × K=100 = 48,400
batches), so the larger start is close to free.

---

## 4. The blocker that had to be fixed first, and the pilot

**`achieved_bounds` did not exist in the stochastic descriptions.** The `stochastic/` quadrotor
datasets ship a *different* `dataset_description.json` schema from the `deterministic/` ones the
system classes were written against — documenting mechanism, horizon and success criteria, but
carrying no `achieved_bounds` block, which is where `Quadrotor2DSystem`/`Quadrotor3DSystem` read
their input-normalization bounds. Both classes raised `KeyError('achieved_bounds')` and died in
under a minute. Neither could ever have loaded a stochastic dataset.

Two changes resolved it:

1. **A backward-compatible fallback** (`_resolve_achieved_bounds` in
   `adaptive_roa/systems/quadrotor2d.py`, shared by `quadrotor3d.py`): if the key is missing,
   borrow the deterministic sibling's bounds and print a NOTE. Unchanged for any dataset that
   already has the key. It no longer fires for these datasets but remains a guard.
2. **Measured `achieved_bounds` written into all 15 descriptions** (2026-08-20), computed from
   each level's own `train.npz` over every stored step of every trajectory (18.4M–60.9M states
   each), matching the deterministic convention ("includes final out-of-bounds states").

**Writing the real bounds was the better call than borrowing**, and the measurement shows why:
the deterministic bounds **do not cover the stochastic data**. Noise widens the velocity envelope,
so normalizing by them would push real states outside [-1, 1] — 2D `x_dot` +7.00%,
2D `theta_dot` +6.93%, 3D `q` +5.08%.

**Caveat, recorded rather than resolved:** positions and angles are effectively identical across
levels (≤0.05% spread; quaternions and `theta` exactly 0%), but velocities spread **0.6–6.5%**
and that spread **grows monotonically with noise**. Normalization therefore varies in a way that
**correlates with the treatment variable**. Small, but not random. An envelope (per-system max
across levels) would remove the confound at ≤6.5% of dynamic range; the campaign is running with
**per-level** bounds. Originals backed up before writing.

---

## 5. Traps specific to this system

**Index files hold FILENAMES, not integers.** `train_test_splits/shuffled_indices_0.txt` stores
`sequence_<row>.txt` (the cartpole convention), so `np.loadtxt(..., dtype=int64)` raises. Parse
the integer out of the filename; verified `labels[perm] == shuffled_labels` at **1.000000** on all
15 levels before use. Pendulum stores bare integers — the two conventions coexist.

**The horizon is LOAD-BEARING, and more so on quad3D than anywhere else in the project.** Labels
are **bounded-time reach probabilities**, not asymptotic ones. From the quad3D description: given
unlimited time, success at f_0.072 is ~0.24 against f_0.000's ~0.25 — but at H=1000 it reads
**0.058**, and about 15% of f_0.072 rollouts would succeed with more time. **Most of the apparent
decline across that sweep is the deadline, not loss of stability.** A "noise hurts more"
conclusion drawn from those numbers would be largely an artifact. quad2D (H=1200) starts hitting
its cap from f=0.020 upward.

**quad3D's success criterion is dimensionally incoherent and mismatched.** Success is radius 0.05
(entry-cut) over the env's **12-D Euler** state, not the 13-D quaternion state the files store,
and the norm sums metres, m/s, radians and rad/s. Inherited from the deterministic set.
Against `attractor_radius=0.3` that is (0.3/0.05)¹² ≈ **2.2×10⁹** times the volume in 12-D — far
beyond quad2D's 11.4×. **Verify on a quad3D pilot that predicted `p_success` is not saturated
before launching that fleet.**

**Do NOT run quadrotor work on the existing Amarel clone.** Attempted 2026-08-20 to escape iLab's
12-GPU quota; A100s there admit a job in ~20 seconds (`sbatch --test-only` claimed a 21-hour wait
and was wrong by three orders of magnitude — always probe with a real job). But
`/home/st1122/Projects/adaptive_roa` on Amarel is **308 commits and 295 files behind origin, and
29 of those files are also locally modified** by someone else, so it cannot be brought current
safely. Three separate failures followed, each looking like the last one:

1. `ValueError: could not convert 'sequence_40536.txt' to int64` — its `npz_data_source.py`
   predates `4d71a63`, which taught the loader to parse filename-style indices.
2. `Could not find 'acquisition/decomp_yield_mlp'` — the entire yield-aware config and strategy
   family postdates that clone.
3. `IndexError: index 3 is out of bounds ... size 3` at `quadrotor2d.py:313` — a 3-column state
   reaching the 6-D `normalize_state`, from within the 29 locally-modified files. Not fixable
   without overwriting someone else's work.

All 12 Amarel attempts wrote **0 artifacts**, so nothing was lost, but ~45 minutes were.
**The diagnostic that would have prevented all of it is one command:**
`git log --oneline HEAD..origin/<branch> | wc -l`, run BEFORE launching. Patching file-by-file
treats each symptom as the last one; measure the gap first. If Amarel is wanted for quad3D, make a
**fresh clone** at the target branch and symlink the existing venv — do not repair this one.

**The three predictor config traps** (each killed a cartpole job in 8–80 s):

```
GP)  predictor=gp predictor.gp.n_iters=300     # NOT gp_reg -- partx needs GPProbabilityBackend
CLF) predictor=clf_ensemble  +predictor.lightning_trainer.enable_progress_bar=false   # '+' form
*)   predictor=fm_ensemble    predictor.lightning_trainer.enable_progress_bar=false   # no '+'
```

**Cost.** ~40 min/epoch on quad2D, **eval-dominated** — the ROA eval is 23,500 batches against
cartpole's 5,200, on a 489,789-cell grid. Training plus acquisition is a few minutes. 336 epochs
per level; at a 12-GPU iLab quota, both levels ≈ 37 h wall-clock.

**Job-level state hides step failure.** `sacct -X` reported `COMPLETED 0:0` for both pilot jobs
that had actually died — the `.0` step read `FAILED 1:0`. Always check the step.

---

---

## 6. Provenance

- Configs verified to resolve to the intended `stochastic/` paths with every required file
  present before launch.
- `achieved_bounds` written to all 15 descriptions; all re-parse with every pre-existing key
  intact; originals backed up.
- 28 run records in `runs_quad3d.jsonl` as `q3d_nd048_<arm>` / `q3d_nd060_<arm>`.

---

## 7. Standings

Scored by `scripts/score_stoch_incremental.py`, which re-projects the `p_success` already written
by each epoch's evaluation onto the continuous `eval_success_prob.npz` ground truth. Metrics are
KL, debiased Brier and sAUROC. The artifact `auc`/`brier` fields are never used: they score
against a 0.5-dichotomised label field rather than the continuous truth.

The 2·SD floor is the **FM** floor (3 uniform seeds, ep0 excluded) and licenses **FM-vs-FM**
statements only. `clf_*` arms are read against `clf_dir00`, their own uniform run, of which there
is one seed — so no classifier gap is claimable at any size. Part-X is a third model class again.

Data: `quad3d_noisy_dynamics_all_levels.csv`, figure `quad3d_noisy_dynamics_all_levels.png`.

### Two findings that survive independently of the tables

The per-level standings are generated below from the CSVs and change every cycle. These two
observations are about the data-generating process rather than the ranking, so they are recorded
here by hand.

**The epoch-0 models agree to 4 decimal places across every seed-42 arm** on `f_0.048` — 0.2986
KL for `epi_var`, `epi_var_anch`, `epi_bald`, `yield_a1`, `yield_mlp` and `dir00_s42` alike, with
`dir00_s43`/`s44` at 0.2932/0.2910. That is the expected signature of a correct pre-acquisition
epoch and a clean end-to-end check that the start-state fix did not perturb training.

**Part-X acquired nothing on `noisy_dynamics f_0.048`, and this is level-specific.**
`train_trajectories` reads 10,000 at every one of its 18 epochs there — not a single trajectory
added. On `corridor_sine_ambient f_0.30` the same arm behaves differently, going
10,000 → 15,000 → 20,000 → 25,000 → 30,000 over 11 epochs (**+20,000**), after a flat opening of
five epochs. So "Part-X does not spend its budget" is **false as a general statement** and true
only of `nd f_0.048`. An earlier revision of this section read the `nd` KL drift as "gets worse
with more data"; that is the opposite of what happened, since no data was added, and the sentence
has been struck rather than softened.

What the `nd f_0.048` series actually measures is **refit variance on a fixed 10,000-trajectory
set**. The model is re-fit each epoch on identical data and wanders non-monotonically over a
0.068 KL range:

```
ep0 .1601  .1680 .1813 .1707 .1735 .1681 .1660 .1890 .2280(worst)
     .2043 .1668 .2152 .1977 .1712 .1997 .2216 .1878 .2039(ep17)
```

Best at epoch 0, worst at epoch 8, no trend. That is a useful noise scale for the other arms: a
quad3D gap smaller than ~0.068 KL between two epochs of the same arm is not evidence of anything.
It also means Part-X on `nd f_0.048` is not a measurement of acquisition at all — nothing was
acquired. The corridor run, which does acquire, is the one to read for that question.

**`noisy_dynamics` is PAUSED** (2026-08-23 15:10, 26 jobs cancelled). Per-arm depths at the moment
of pause are in [`q3d_nd_depths_at_pause.txt`](q3d_nd_depths_at_pause.txt); `f_0.060` never
produced an artifact. Resume with `scripts/resume_adaptive.py --run-dir <dir>`.

---

## The start-state mismatch that blocked scored acquisition (RESOLVED 2026-08-22)

**Confirmed 2026-08-21 across all three predictor backends.** Every quad3D arm that scores
candidates dies at its first acquisition with

```
File "adaptive_roa/adaptive_v2/probability/ensemble_prob.py", line 62, in estimate_members
    embedded = self.system.embed_state_for_model(self.system.normalize_state(x))
File "adaptive_roa/systems/quadrotor3d.py", line 348, in normalize_state
    ang_vel[:, 2] / self.r_limit,
IndexError: index 2 is out of bounds for dimension 1 with size 2
```

It was seen first on the FM ensemble, then Part-X (GP), then the classifier ensemble, all through
the same `estimate_members` frame. The predictor is not the cause.

### The two "start states" are not the same state

`train.npz` holds two arrays the code both treats as start states:

| array | shape | what it is |
|---|---|---|
| `starts` | 800,000 x **12** | position, **Euler ZYX**, linear velocity, angular velocity |
| `states[offsets[r]]` | **13** | position, **quaternion (qw,qx,qy,qz)**, linear velocity, body rates |

Training pairs are built from `states` (`load_trajectory` -> `states[offsets[r]:offsets[r+1]]`),
so the model is trained on the 13-D row. `get_start_states` returns the 12-D `starts` array, and
that is the only thing the scored acquisition path ever feeds the model.

Measured over 5,000 rollouts of `noisy_dynamics/f_0.048`:

| block | agreement |
|---|---|
| position | max abs diff 1.19e-07 — identical |
| linear velocity | max abs diff 1.19e-07 — identical |
| orientation | Euler ZYX -> quaternion matches to 2.8e-03 on 99.7% of rollouts — convertible |
| **angular velocity** | **raw correlation -0.006**; after a Euler-rate -> body-rate transform, 0.337 |

The angular-velocity block is genuinely different data, not a different parameterisation of the
same numbers. `starts` is clipped to the +/-24 sampling bound; `states` row 0 reaches +/-37.7,
consistent with the dataset description's note that achieved rates reach 39.1 against a stated 24.
`starts` is the sampled initial condition; `states` row 0 is what the simulator recorded after the
first control step and external-force draw.

**So the crash is protective.** Padding the 12-D vector into 13-D — even converting Euler to
quaternion correctly — would hand the ensemble a state whose angular velocity is uncorrelated with
anything it saw in training, and every epistemic score would be computed on garbage silently. An
earlier revision of this section proposed exactly that (a 12-D branch in `normalize_state`); that
is wrong and would have produced a fleet of plausible, meaningless numbers.

**The fix is in the data source, not the system.** `NpzTrajectoryDataSource.start_states` should
read `states[offsets[rollout_ids]]` rather than `starts[rollout_ids]`, so acquisition scores the
same representation training uses. `normalize_state` is correct as written.

### Blast radius

10 of 14 arms per level, 20 of 28 across `f_0.048` + `f_0.060`.

| survives | dies |
|---|---|
| `dir00_s42`, `dir00_s43`, `dir00_s44`, `clf_dir00` | `epi_var`, `epi_var_anch`, `epi_bald`, `yield_a1`, `yield_mlp`, `partx`, `clf_yield`, `clf_epi_var`, `clf_epi_bald`, `clf_epi_var_anch` |

Uniform sampling never calls `estimate_members` — it passes only indices, and pairs are rebuilt
from `states` — which is exactly why the four `dir00`/`clf_dir00` arms are unaffected.

### quad2D was not affected

On `quadrotor2D/noisy_dynamics/rl/f_0.150`, `starts` and `states` are both 6-D and
`starts[r] == states[offsets[r]]` to 2.4e-07 across all columns. The mismatch is specific to
quad3D, where the collector stored the sampling representation (Euler) alongside the simulation
representation (quaternion). No quad2D result needs revisiting.

### Fix applied 2026-08-22

`adaptive_roa/adaptive/npz_data_source.py` now derives start states from `states`:

```python
self.start_states = self._states[self._offsets[self.rollout_ids]]
self._warn_if_starts_disagree()
```

`_starts` fed exactly one line, so this is the whole change. `_warn_if_starts_disagree` logs a
NOTE when the npz's `starts` array is not `states[offsets[r]]` — by width or by value — so a
dataset carrying this split announces itself on line one instead of being found by a crash.
`Quadrotor3DSystem.normalize_state` was NOT touched; it was correct as written.

Verified:

| check | result |
|---|---|
| quad3D `start_states` width | 12 -> **13** |
| equals `states[offsets[rollout_ids]]` | max abs diff **0.0** |
| `normalize_state` on 2000 candidates | OK, (2000, 13), all finite — this is the frame that raised |
| `embed_state_for_model` | OK, (2000, 13), matches `model_dims.condition_dim: 13` |
| **quad2D regression** | max abs diff vs pre-fix behaviour **0.0** — no-op |
| warning fires on quad3D | `NOTE: npz 'starts' is 12-D but 'states' is 13-D` |

The 28 quad3D run directories from the 08-21 launch were deleted (~19 GB); they carried the wrong
budget as well as this defect. Relaunch is at 10,000 / 5,000 / 18 epochs = 100,000 trajectories,
`n_candidates` 250,000, K 20, `--mem=8G`.

---

<!-- STANDINGS:BEGIN (generated by scripts/update_standings.py -- do not hand-edit) -->

### quad3D `noisy_dynamics f_0.048` (PAUSED) — 109/252 epochs, 2/14 arms complete

| metric | 2·SD floor (ep1–5) | 3-seed uniform control at ep5 |
|---|---|---|
| KL | **0.0032** | 0.1391 |
| Brier_deb | **0.0004** | 0.0236 |
| sAUROC | **0.0007** | 0.9495 |

| arm | ep | KL | Brier_deb | sAUROC | ΔKL | ΔBrier | ΔsAUROC | verdict |
|---|---|---|---|---|---|---|---|---|
| `clf_yield` | 5 | 0.0977 | 0.0273 | 0.9435 | -0.0414 | +0.0037 | -0.0060 | **mixed — better ×1, worse ×2** |
| `clf_epi_var` | 5 | 0.0980 | 0.0274 | 0.9421 | -0.0411 | +0.0038 | -0.0074 | **mixed — better ×1, worse ×2** |
| `clf_epi_bald` | 5 | 0.1005 | 0.0277 | 0.9384 | -0.0385 | +0.0041 | -0.0111 | **mixed — better ×1, worse ×2** |
| `epi_bald` | 4 | 0.1093 | 0.0204 | 0.9569 | -0.0297 | -0.0033 | +0.0074 | beats floor ×3 |
| `clf_dir00` | 5 | 0.1117 | 0.0311 | 0.9297 | -0.0273 | +0.0074 | -0.0199 | **mixed — better ×1, worse ×2** |
| `epi_var_anch` | 5 | 0.1121 | 0.0206 | 0.9555 | -0.0269 | -0.0030 | +0.0060 | beats floor ×3 |
| `epi_var` | 4 | 0.1146 | 0.0209 | 0.9555 | -0.0245 | -0.0027 | +0.0060 | beats floor ×3 |
| `yield_a1` | 3 | 0.1151 | 0.0200 | 0.9537 | -0.0239 | -0.0036 | +0.0042 | beats floor ×3 |
| `clf_epi_var_anch` | 3 | 0.1183 | 0.0318 | 0.9300 | -0.0208 | +0.0082 | -0.0195 | **mixed — better ×1, worse ×2** |
| `yield_mlp` | 2 | 0.1329 | 0.0225 | 0.9479 | -0.0062 | -0.0012 | -0.0016 | **mixed — better ×2, worse ×1** |
| `partx` | 5 | 0.1681 | 0.0480 | 0.8772 | +0.0291 | +0.0244 | -0.0723 | **worse than uniform ×3** |

> Compared at ep5, the control's deepest shared epoch. These arms are deeper than that and are read at ep5 rather than their own deepest: `clf_dir00`, `clf_epi_var`, `clf_yield`, `partx`.

> Level incomplete (2/14 arms at 18 epochs). Ordering can still move; treat as provisional.


### quad3D `noisy_dynamics f_0.060` (PAUSED) — 0/252 epochs, 0/14 arms complete

_No rows yet._


### quad3D `corridor_sine_ambient f_0.30` — 60/252 epochs, 0/14 arms complete

_Floor seeds share no epoch beyond ep0 yet._

<!-- STANDINGS:END -->

## Companion documents

- **Methods** — [`../METHODS.md`](../METHODS.md): what each arm, score and predictor does, the
  d1/d2 split, the metric definitions, and the standard of evidence. This file reports results
  and assumes those definitions.
- **Sibling systems** — [`../pendulum/`](../pendulum/), [`../cartpole/`](../cartpole/).
- **The other quadrotor** — [`../quadrotor2d/`](../quadrotor2d/) — 6-D, safe_explorer_ppo, 500k pool, 24-epoch budget.

**Why the two quadrotors live in separate directories.** They are different state spaces (6-D vs
13-D), different pools (500k vs 800k), different controllers (`safe_explorer_ppo` vs LQR) and,
since 2026-08-22, different budgets (24 epochs vs 18). Keeping one directory per system means a
figure, a CSV and a run log can never be read across systems by accident, and "how deep is this
level" stays a `wc -l` rather than a filtering problem.
