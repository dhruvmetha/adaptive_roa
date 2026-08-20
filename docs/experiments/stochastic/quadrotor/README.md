# Stochastic quadrotors — adaptive acquisition campaign

Written 2026-08-20. **Scope: experiments run on the `stochastic/` dataset tree only.**
**Status: IN FLIGHT.** quad2D launched 2026-08-20 10:2x (28 jobs, 214675–214702); quad3D not yet
launched.

- **Data root** — `${data_dir}/stochastic/quadrotor{2D,3D}/<family>/<controller>/<level>`
- **Experiment dir** — `/common/users/shared/pracsys/adaptive_roa_experiments/quadrotor_stoch/`
- **Configs** — `configs/adaptive_v2/system/quadrotor{2d,3d}_stoch.yaml` (new, 2026-08-20)
- **Run log** — 28 records in `docs/experiments/ensemble_epistemic/runs.jsonl`, `q2d_{nd,cs}_<arm>`

---

## 1. What exists, and the two noise families

**These datasets are new and had never been run before 2026-08-20.** Neither system class could
load them (see §4), so the campaign starts from a fixed codebase, not from prior results.

| system | family | levels | controller | dim | n_traj |
|---|---|---|---|---|---|
| quad2D | `noisy_dynamics` | f_0.000, 0.070, 0.100, **0.150**, 0.200 | safe_explorer_ppo | 6 | 500k |
| quad2D | `corridor_sine_ambient` | baseline, **smooth**, sharp | safe_explorer_ppo | 6 | 500k |
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

## 2. Measured pool structure (2026-08-20, from the shipped `train.npz`)

The length–outcome coupling runs the **cartpole** way — long == success — not the pendulum way:

| level | traj_succ | L_succ | L_fail | PAIR_succ | frac_mid |
|---|---:|---:|---:|---:|---:|
| 2D noisy_dynamics f_0.070 | 0.079 | 424.4 | 8.9 | 0.802 | 0.054 |
| 2D noisy_dynamics f_0.100 | 0.071 | 473.5 | 12.7 | 0.739 | 0.085 |
| **2D noisy_dynamics f_0.150** | 0.048 | 515.8 | 23.6 | 0.522 | 0.103 |
| 2D noisy_dynamics f_0.200 | 0.024 | 480.1 | 27.2 | 0.306 | 0.090 |
| **2D corridor smooth** | 0.062 | 490.8 | 10.6 | 0.755 | 0.068 |
| 2D corridor sharp | 0.063 | 470.2 | 10.0 | 0.761 | 0.054 |
| 3D noisy_dynamics f_0.032 | 0.067 | 575.9 | 17.3 | 0.706 | 0.041 |
| 3D noisy_dynamics f_0.048 | 0.051 | 661.8 | 33.2 | 0.519 | **0.230** |
| 3D noisy_dynamics f_0.060 | 0.034 | 691.4 | 50.4 | 0.324 | **0.230** |
| 3D noisy_dynamics f_0.072 | 0.020 | 705.6 | 63.2 | 0.187 | 0.225 |
| 3D corridor f_0.25 | 0.052 | 563.0 | 16.9 | 0.647 | 0.108 |
| 3D corridor f_0.30 | 0.043 | 562.9 | 17.8 | 0.586 | 0.130 |

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

## 3. Campaign design (quad2D, in flight)

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

## 6. Provenance

- Configs verified to resolve to the intended `stochastic/` paths with every required file
  present, for all four (system, family, level) combinations, before launch.
- `achieved_bounds` written to all 15 descriptions; all re-parse with every pre-existing key
  intact; originals backed up.
- 28 run records appended to `runs.jsonl` as `q2d_{nd,cs}_<arm>`.
- Companions: `../pendulum/`, `../cartpole/`.
