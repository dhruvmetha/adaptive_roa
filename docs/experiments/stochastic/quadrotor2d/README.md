# Stochastic quadrotor **2D** — adaptive acquisition campaign

Written 2026-08-20, split out of the shared quadrotor README 2026-08-22.
**Scope: quad2D experiments on the `stochastic/` dataset tree only.** quad3D lives in
[`../quadrotor3d/`](../quadrotor3d/) and shares none of this file's numbers.
**Status: COMPLETE.** Both levels finished — `noisy_dynamics f_0.150` and
`corridor_sine_ambient smooth`, 14 arms × 24 epochs each, 672/672 epochs, all scored.
Nothing is running on quad2D. Header last refreshed **2026-08-23 16:1x**.

Launched 2026-08-20 10:2x (28 jobs, 214675-214702), `initial_train_size=2000`,
`samples_per_epoch=500`, 24 epochs.

- **`noisy_dynamics f_0.150` is COMPLETE** — all 14 arms at 24/24, 336/336 epochs, all scored.
- **`corridor_sine_ambient smooth` is COMPLETE** — all 14 arms at 24/24, 336/336 epochs, all
  scored (finished 2026-08-23 12:13, last epoch `yield_a1` 23). All three uniform floor seeds
  `dir00_s42/s43/s44` reached 24/24, so this family has a real 3-seed floor and the
  FM-vs-classifier gap on it is testable rather than merely observed.

**Both quad2D levels are therefore done: 672/672 epochs, nothing running on this system.**
§7 standings were recomputed against the complete 336-row CSVs on 2026-08-23.

**Scoring is live and incremental.** `scripts/score_stoch_incremental.py` runs every monitoring
cycle: it diffs the epochs on disk against the committed CSV and scores only the difference, so
§7 tracks the campaign instead of lagging it. It re-projects the `p_success` each epoch already
wrote onto the continuous `eval_success_prob.npz` ground truth — no model inference, no retraining.

- **Data root** — `${data_dir}/stochastic/quadrotor2D/<family>/rl/<level>`
- **Experiment dir** — `/common/users/shared/pracsys/adaptive_roa_experiments/quadrotor_stoch/`
  (run dirs prefixed `q2d_nd_*` and `q2d_cs_*`)
- **Config** — `configs/adaptive_v2/system/quadrotor2d_stoch.yaml` (new, 2026-08-20)
- **Run log** — `runs_quad2d.jsonl` beside this README, 48 records, with a sibling
  `runs_quad2d.csv`. Append-only: one record per launch, never edited in place (a resume is a NEW
  record, not a status update on the old one). Append with
  `scripts/exp_log.py append --log docs/experiments/stochastic/quadrotor2d/runs_quad2d.jsonl …`;
  `--log` is required and has no default. Regenerate every CSV with `python scripts/runs_to_csv.py`
  and never hand-edit them.
- **Figures and data** — `quad2d_<family>_all_levels.{csv,png}`, plus a reduced
  `_clean.png` variant of each.

**Reading the run log.** It is append-only: a relocation or resume is a NEW record, never an edit
to the old one. So arms that moved hosts appear **more than once** — once per launch — and the
superseded record still reads `status: launched`, because it was, right up until it was cancelled.
**Counting rows is not counting runs.** Tell them apart with `cluster` + `launched_at` (newest
wins) and the `dir_state` column:

| `dir_state` | meaning |
|---|---|
| `local` | counted on this host; `epochs_on_disk` is real |
| `remote` | `output_dir` is on Amarel `/scratch`, unmeasurable from iLab — **not** zero progress |
| `missing` | local path absent: the run never wrote anything, e.g. cancelled before it started |

`epochs_on_disk` counts `artifacts_v2.json`, never `epoch_*` directories — a directory is created
before its epoch finishes, so counting directories overstates depth.

**Nothing runs on Amarel.** An earlier plan put 8 `cs` arms there; the clone turned out to be 308
commits behind and the arms were relaunched on iLab/westeros instead. Verified 2026-08-20 15:4x:
zero `q2d_*` jobs and zero `q2d_*` directories under `/scratch` on Amarel. Every arm writes to the
shared `/common` filesystem, so no rsync step stands between the runs and scoring. The campaign
runs on **iLab** (SLURM, 12-GPU quota) and **westeros** (tmux, one session per arm); an arm's live
host is in its newest run-log record.

---

## 1. What exists, and the two noise families

**These datasets are new and had never been run before 2026-08-20.** Neither system class could
load them (see §4), so the campaign starts from a fixed codebase, not from prior results.

| system | family | levels | controller | dim | n_traj |
|---|---|---|---|---|---|
| quad2D | `noisy_dynamics` | f_0.000, 0.070, 0.100, **0.150**, 0.200 | safe_explorer_ppo | 6 | 500k |
| quad2D | `corridor_sine_ambient` | baseline, **smooth**, sharp | safe_explorer_ppo | 6 | 500k |

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
| 2D noisy_dynamics f_0.070 | 0.079 | 424.4 | 8.9 | 0.802 | 0.054 |
| 2D noisy_dynamics f_0.100 | 0.071 | 473.5 | 12.7 | 0.739 | 0.085 |
| **2D noisy_dynamics f_0.150** | 0.048 | 515.8 | 23.6 | 0.522 | 0.103 |
| 2D noisy_dynamics f_0.200 | 0.024 | 480.1 | 27.2 | 0.306 | 0.090 |
| **2D corridor smooth** | 0.062 | 490.8 | 10.6 | 0.755 | 0.068 |
| 2D corridor sharp | 0.063 | 470.2 | 10.0 | 0.761 | 0.054 |

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

## 3. Campaign design (quad2D, complete)

14 arms per level, identical to the cartpole v2 design so the campaigns are directly comparable:

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

Budget: `initial_train_size=2000`, `samples_per_epoch=500`, **`n_epochs=24`** → 14,000 final
trajectories. 5 ensemble members, `filter_confident_pairs=false`.

**Note on `n_epochs`:** `quadrotor2d.yaml` justifies its default of 20 as a pool-exhaustion cap
(*"far beyond the ~16k available after the 0.2 val split"*). That was measured against the **old
deterministic** dataset. The stochastic pools hold 500k/800k trajectories, so 14,000 is ~2.8% of
the pool and the exhaustion argument does not bind. The epoch count here is a
convergence-vs-cost choice, not a pool limit.

---

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

### Pilot (2026-08-20, `q2d_pilot_{nd,cs}`, dir00 control, 3 epochs)

Run before committing the fleet. All gates clean: 6/6 epochs, zero magnitude flags,
`filter_diagnostics` null, both COMPLETED 0:0 at the **step** level.

| family | ep | sAUROC | KL | recal | RES |
|---|---|---:|---:|---:|---:|
| noisy_dynamics f_0.150 | 0 | 0.6473 | 0.2055 | 0.01723 | 0.00573 |
| | 1 | 0.7597 | 0.1663 | 0.01285 | 0.01011 |
| | 2 | **0.8715** | **0.1022** | 0.00906 | 0.01390 |
| corridor smooth | 0 | 0.8715 | 0.1435 | 0.01751 | 0.02490 |
| | 1 | 0.8783 | 0.1414 | 0.01627 | 0.02613 |
| | 2 | **0.8928** | **0.1361** | 0.01591 | 0.02649 |

**The two families behave completely differently and this must be pre-registered, not concluded
after the fact.** `noisy_dynamics` starts near-useless (0.647) and climbs steeply — +0.224 sAUROC,
KL halved, RES 2.4× in three epochs — and is nowhere near converged at epoch 2.
`corridor smooth` **starts where `nd` finishes** (0.8715 at epoch 0) and barely moves (+0.021
sAUROC over three epochs; `brier_debiased` actually *rises* slightly each epoch).

**Consequence:** on `corridor smooth` the control is nearly saturated by epoch 0, so there may be
little room for an acquisition arm to separate. **If that level returns a null, the likely cause
is a saturated control, not acquisition failing.** That is stated here in advance.

**On `attractor_radius`:** endpoint MAE moves very slowly (2D success MAE 0.4232 → 0.3992 over
three epochs) and will not cross the 0.3 radius within 24 epochs, so predicted `p_success` stays
compressed far below the true mean (0.0082 vs a true 0.045 at epoch 2). **The pilot shows this
does not block the campaign** — sAUROC and KL score the *ranking* and the *distribution*, not the
absolute level, and all three metrics improve monotonically. The compression lands in `recal`,
where it belongs. Radius left at 0.3; magnitudes are not comparable to an r=0.2 run.

---

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
  present, for all four (system, family, level) combinations, before launch.
- `achieved_bounds` written to all 15 descriptions; all re-parse with every pre-existing key
  intact; originals backed up.
- 28 run records appended to `runs.jsonl` as `q2d_{nd,cs}_<arm>`.
- Companions: `../pendulum/`, `../cartpole/`.

---

---

## 7. Standings

Scored 2026-08-23 16:0x by `scripts/score_stoch_incremental.py` (both levels complete, 672/672
epochs), which re-projects the
`p_success` already written by each epoch's evaluation onto the continuous
`eval_success_prob.npz` ground truth. It is incremental: every monitoring cycle it diffs the
epochs on disk against the committed CSV and scores only the difference, so these tables track
the campaign rather than lagging it. No model inference is involved and no run is re-trained.

Data and figures, one pair per family:

| family | CSV | figure |
|---|---|---|
| quad2D `noisy_dynamics f_0.150` | `quad2d_noisy_dynamics_all_levels.csv` | `quad2d_noisy_dynamics_all_levels.png` |
| quad2D `corridor_sine_ambient smooth` | `quad2d_corridor_sine_ambient_all_levels.csv` | `quad2d_corridor_sine_ambient_all_levels.png` |

**Three predictor families, three baselines.** The 2·SD floor is the **FM** floor (3 uniform
seeds, ep0 excluded) and it licenses **FM-vs-FM** statements only. `clf_*` arms are a different
model class and are read against `clf_dir00`, their own uniform run — of which there is **one
seed**, so no classifier gap is claimable at any size and the classifier tables below are
descriptive. Part-X is a third class again, and is reported against the FM control only because
it shares that control's training pool.

Metrics are KL, debiased Brier, and sAUROC. The artifact `auc`/`brier` fields are never used:
they score against a 0.5-dichotomised label field rather than the continuous truth.

<!-- STANDINGS:BEGIN (generated by scripts/update_standings.py -- do not hand-edit) -->

### quad2D `noisy_dynamics f_0.150` — 336/336 epochs, 14/14 arms complete

| metric | 2·SD floor (ep1–23) | 3-seed uniform control at ep23 |
|---|---|---|
| KL | **0.0184** | 0.0647 |
| Brier_deb | **0.0012** | 0.0127 |
| sAUROC | **0.0398** | 0.9220 |

| arm | ep | KL | Brier_deb | sAUROC | ΔKL | ΔBrier | ΔsAUROC | verdict |
|---|---|---|---|---|---|---|---|---|
| `clf_yield` | 23 | 0.0167 | 0.0039 | 0.9782 | -0.0480 | -0.0088 | +0.0562 | beats floor ×3 |
| `clf_epi_var` | 23 | 0.0182 | 0.0044 | 0.9769 | -0.0465 | -0.0083 | +0.0550 | beats floor ×3 |
| `clf_epi_bald` | 23 | 0.0183 | 0.0045 | 0.9768 | -0.0464 | -0.0083 | +0.0548 | beats floor ×3 |
| `epi_var_anch` | 23 | 0.0210 | 0.0043 | 0.9691 | -0.0437 | -0.0084 | +0.0471 | beats floor ×3 |
| `clf_epi_var_anch` | 23 | 0.0219 | 0.0051 | 0.9755 | -0.0428 | -0.0077 | +0.0535 | beats floor ×3 |
| `yield_a1` | 23 | 0.0229 | 0.0052 | 0.9695 | -0.0418 | -0.0076 | +0.0475 | beats floor ×3 |
| `yield_mlp` | 23 | 0.0244 | 0.0053 | 0.9671 | -0.0403 | -0.0074 | +0.0451 | beats floor ×3 |
| `epi_bald` | 23 | 0.0262 | 0.0059 | 0.9664 | -0.0384 | -0.0068 | +0.0444 | beats floor ×3 |
| `epi_var` | 23 | 0.0288 | 0.0063 | 0.9633 | -0.0359 | -0.0064 | +0.0413 | beats floor ×3 |
| `clf_dir00` | 23 | 0.0319 | 0.0076 | 0.9693 | -0.0328 | -0.0052 | +0.0473 | beats floor ×3 |
| `partx` | 23 | 0.0480 | 0.0123 | 0.9468 | -0.0167 | -0.0005 | +0.0248 | inside floor |


### quad2D `corridor_sine_ambient smooth` — 336/336 epochs, 14/14 arms complete

| metric | 2·SD floor (ep1–23) | 3-seed uniform control at ep23 |
|---|---|---|
| KL | **0.0040** | 0.0614 |
| Brier_deb | **0.0009** | 0.0154 |
| sAUROC | **0.0031** | 0.9623 |

| arm | ep | KL | Brier_deb | sAUROC | ΔKL | ΔBrier | ΔsAUROC | verdict |
|---|---|---|---|---|---|---|---|---|
| `yield_mlp` | 23 | 0.0214 | 0.0052 | 0.9851 | -0.0400 | -0.0102 | +0.0228 | beats floor ×3 |
| `epi_bald` | 23 | 0.0221 | 0.0055 | 0.9853 | -0.0394 | -0.0099 | +0.0229 | beats floor ×3 |
| `yield_a1` | 23 | 0.0224 | 0.0055 | 0.9843 | -0.0390 | -0.0099 | +0.0219 | beats floor ×3 |
| `epi_var` | 23 | 0.0241 | 0.0060 | 0.9840 | -0.0373 | -0.0094 | +0.0216 | beats floor ×3 |
| `epi_var_anch` | 23 | 0.0246 | 0.0059 | 0.9820 | -0.0368 | -0.0095 | +0.0197 | beats floor ×3 |
| `clf_epi_var` | 23 | 0.0629 | 0.0178 | 0.9846 | +0.0015 | +0.0024 | +0.0223 | **mixed — better ×1, worse ×1** |
| `clf_epi_bald` | 23 | 0.0635 | 0.0177 | 0.9846 | +0.0021 | +0.0023 | +0.0222 | **mixed — better ×1, worse ×1** |
| `clf_yield` | 23 | 0.0644 | 0.0181 | 0.9849 | +0.0029 | +0.0027 | +0.0226 | **mixed — better ×1, worse ×1** |
| `clf_epi_var_anch` | 23 | 0.0815 | 0.0216 | 0.9834 | +0.0200 | +0.0062 | +0.0210 | **mixed — better ×1, worse ×2** |
| `clf_dir00` | 23 | 0.0938 | 0.0249 | 0.9791 | +0.0324 | +0.0095 | +0.0168 | **mixed — better ×1, worse ×2** |
| `partx` | 23 | 0.1065 | 0.0299 | 0.9348 | +0.0451 | +0.0145 | -0.0275 | **worse than uniform ×3** |

<!-- STANDINGS:END -->

**Reading the two levels together.** On `nd` every acquisition arm and every classifier arm
beats the 3-seed floor on all three metrics; only `partx` fails to separate from uniform. On
`cs` the picture splits by predictor family: the five FM acquisition arms beat the floor ×3,
while all five classifier arms land **mixed** — better than uniform on sAUROC by 5–7× the floor,
but *worse* on debiased Brier, and for `clf_epi_var_anch` and `clf_dir00` worse on KL as well.
That is a ranking-versus-calibration split: on `cs` the classifiers order states well and
calibrate poorly. It does not appear on `nd`, where the classifier arms take the top three slots
outright. Any cross-level claim about "classifiers beat FM" is therefore false as stated — it
holds on `nd` and inverts on `cs` for the calibration metrics.

**`partx` is the one consistent negative.** Inside the floor on `nd` (no evidence it helps or
hurts) and worse than uniform on all three metrics on `cs`. Both readings still carry the budget
caveat below: `partx` does not spend its trajectory allowance, ending `nd` at 6,746 of 13,500 and
`cs` at 7,139, so these are not matched-budget comparisons.


## Companion documents

- **Methods** — [`../METHODS.md`](../METHODS.md): what each arm, score and predictor does, the
  d1/d2 split, the metric definitions, and the standard of evidence. This file reports results
  and assumes those definitions.
- **Sibling systems** — [`../pendulum/`](../pendulum/), [`../cartpole/`](../cartpole/).
- **The other quadrotor** — [`../quadrotor3d/`](../quadrotor3d/) — 13-D, LQR, 800k pool, 18-epoch budget.

**Why the two quadrotors live in separate directories.** They are different state spaces (6-D vs
13-D), different pools (500k vs 800k), different controllers (`safe_explorer_ppo` vs LQR) and,
since 2026-08-22, different budgets (24 epochs vs 18). Keeping one directory per system means a
figure, a CSV and a run log can never be read across systems by accident, and "how deep is this
level" stays a `wc -l` rather than a filtering problem.
