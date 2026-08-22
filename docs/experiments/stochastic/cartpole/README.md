# Stochastic cartpole — gaussian_signal adaptive acquisition campaign

Written 2026-08-19. **Scope: experiments run on the `stochastic/` dataset tree only.**

- **Data root** — `${data_dir}/stochastic/cartpole/gaussian_signal/lqr/{low,med,high}`
- **Experiment dir** — `/common/users/shared/pracsys/adaptive_roa_experiments/gaussian_torque/cp_{low,med,high}_v2_<arm>`
  (note the **`_v2`** — the v1 runs are invalid, see §1)
- **Status** — COMPLETE. 14 arms × 12 epochs × 3 levels = 504 epochs, all scored to full depth.

**Figure in this directory:** `gaussian_all_levels.png` — rows = low/med/high, columns = KL /
debiased Brier / sAUROC. Solid = FM, dotted = classifier, dash-dot = Part-X GP. Shaded band is the
3-seed FM non-adaptive range; the 2·SD floor is printed per panel.

**Not in scope:** the pendulum campaigns (`../pendulum/`) and anything under `noisy/` or
`noisy_torque/`. Different noise processes, pools, and ground-truth grids — **never pool or
compare them.**


**Methods:** see [`../METHODS.md`](../METHODS.md) for what each arm, score and predictor actually does, the d1/d2 split, the metric definitions, and the standard of evidence. This file reports results and assumes those definitions.

---

## 1. Directory naming traps

`gaussian_torque/` holds **three different cartpole families** plus the pendulum ones:

| prefix | dataset | status |
|---|---|---|
| `cp_{low,med,high}_v2_*` | `stochastic/cartpole/gaussian_signal/lqr/*` | ← **this doc** |
| `cp_med_*` (10 runs, no `_v2`) | same dataset | **INVALID — do not use** |
| `cp_*` (15 older runs) | `stochastic/cartpole/noisy_torque/lqr/med` | different dataset |
| `pen_low_*`, `pen_*`, `pen_high_*` | `stochastic/pendulum/gaussian_signal/lqr/*` | `../pendulum/` |

A glob like `cp_*` silently mixes all three cartpole families. A glob like `cp_med_*` matches both
the invalid v1 runs and the valid `cp_med_v2_*` ones.

**Why v1 is invalid.** Every v1 arm was launched with `acquisition.d2_ratio=0`, so all ten spent
their entire budget on uniform draws. All ten produced `n_d1=150, n_d2=0` per epoch, byte-identical
index sets, and identical MAE. They are ten copies of the control, not an acquisition comparison.
v1 also never exposed the `partx` predictor bug (§4) because `d2_ratio=0` meant Part-X's `select()`
never ran.

**Config repoint.** `configs/adaptive_v2/system/cartpole_stoch.yaml` was repointed to
`cartpole/gaussian_signal/lqr/${noise_level}` on **2026-08-18** (it previously read `noisy_torque`,
and before that `noisy_action/sigma_*`). All three v2 launches are 2026-08-19 — med 09:59, low
15:28, high 19:13 — so all are post-repoint. Verified from the runs' own logs, which print the
resolved dataset path. Any earlier cartpole result is on a different dataset.

---

## 2. Arms

14 per level, seed 42 unless noted. Settings: `initial_train_size=300`, `samples_per_epoch=150`,
`n_epochs=12`, 5 ensemble members, `filter_confident_pairs=false`.

| arm | acquisition | d2_ratio | predictor |
|---|---|---|---|
| `dir00_s42/s43/s44` | direct (uniform) | 0 | fm_ensemble |
| `epi_var` | decomp_epi_var | 1.0 | fm_ensemble |
| `epi_bald` | decomp_epi_bald | 1.0 | fm_ensemble |
| `epi_var_anch` | decomp_epi_var | **0.5** | fm_ensemble |
| `yield_a1` | decomp_epi_var_yield (α=1) | 1.0 | fm_ensemble |
| `yield_mlp` | decomp_yield_mlp | 1.0 | fm_ensemble |
| `partx` | partx | 1.0 | **gp** (not gp_reg — see §4) |
| `clf_dir00` | direct | 0 | clf_ensemble |
| `clf_yield` | decomp_yield_mlp | 1.0 | clf_ensemble |
| `clf_epi_var` | decomp_epi_var | 1.0 | clf_ensemble |
| `clf_epi_bald` | decomp_epi_bald | 1.0 | clf_ensemble |
| `clf_epi_var_anch` | decomp_epi_var | **0.5** | clf_ensemble |

This campaign carries **anchored (d2_ratio=0.5) arms in both predictor families**, which the
pendulum gaussian campaign does not.

---

## 3. Results

All three levels scored to full depth: **168 rows each = 14 arms × 12 epochs, verified.**
Scoring output: `CPV2_low/`, `CPV2_med/`, `CPV2_high/`.

Floors — 2·SD pooled over shared epochs, **epoch 0 excluded**:

| level | KL | debiased Brier | sAUROC | recal |
|---|---:|---:|---:|---:|
| low | 0.0091 | 0.0018 | 0.0028 | 0.00088 |
| med | 0.0059 | 0.0017 | 0.0025 | 0.00043 |
| high | 0.0099 | 0.0016 | 0.0067 | 0.00095 |

**Epoch-11 KL vs the 3-seed FM control mean, judged against those floors:**

| arm | low | med | high |
|---|---|---|---|
| dir00 (3-seed control) | .0313 | .0375 | .0393 |
| `epi_var` | **.0139 WIN** | **.0170 WIN** | **.0138 WIN** |
| `epi_bald` | **.0162 WIN** | **.0151 WIN** | **.0137 WIN** |
| `epi_var_anch` | **.0145 WIN** | **.0244 WIN** | **.0187 WIN** |
| `yield_mlp` | .0223 tie | **.0148 WIN** | **.0188 WIN** |
| `yield_a1` | .0247 tie | **.0151 WIN** | **.0208 WIN** |
| `partx` | .0259 tie | .0423 tie | .0527 **lose** |
| `clf_dir00` | .1120 | .0841 | .0416 |
| `clf_*` adaptive | .0966–.2632 | .1028–.1194 | .0489–.0747 |

**FM epistemic acquisition wins at every noise level.** `epi_var` and `epi_bald` clear the floor on
KL, debiased Brier, sAUROC **and** recal at all three levels — a 2.3–2.9× KL reduction over
uniform. This is the opposite of the pendulum gaussian result, where high noise was a flat null.

**Other findings**

- **Part-X (GP) degrades as noise rises**: tie → tie → **lose**. It is the only arm that gets worse
  with noise, and the reversal is sharp — on pendulum gaussian it was the *strongest* arm overall,
  below the FM band for all of med and high. Part-X also **loses on recal at every level** (.0075
  / .0087 / .0099 vs the control's .0064 / .0054 / .0045), so its ranking ability is not backed by
  calibration here. It is a **different model class** on the same axes: GP-vs-FM, not an
  acquisition comparison.
- **The yield arms need noise to work.** Both tie at low and win at med and high. At low noise the
  model is confident nearly everywhere, so there is little uncertainty to trade against rollout
  length and the yield weight has nothing to exploit.
- **Classifier arms lose on KL/Brier but win on sAUROC.** At med and high, four of five clf arms
  beat the FM control on sAUROC while losing badly on KL and Brier. They rank states correctly and
  are calibrated poorly. This is exactly the split that makes a threshold metric untrustworthy —
  and the reason **KL is the metric carrying this campaign.** Judge clf arms against `clf_dir00`,
  their own uniform run, never against the FM band.
- **Anchoring rescues the classifier family.** `clf_epi_var_anch` (d2_ratio=0.5) is the best clf
  arm at low by a factor of 2.2–2.7 on KL/Brier; the pure-scored clf arms diverge upward. Same
  pattern as pendulum low, where `clf_yield` got worse over epochs.
- Arms converge at different rates, so a vertical slice mid-run flatters early converging arms and
  penalises late ones. **Read the right-hand end of the figure.**

---

## 4. Methodology and traps

**Scoring.** `scripts/stoch_prob_metrics.py --spec <json> --data-root
${data_dir}/stochastic/cartpole/gaussian_signal/lqr --epochs all --out <dir>`.
Use **sAUROC, KL, and recal = UNC_debiased − RES**. Never the artifact AUC/Brier — it scores
against a 0.5-dichotomised truth.

**Verify row count == arms × epochs.** A silent grid rejection once produced 18 rows instead of 75.
Cached metrics also go stale. **Re-score against disk before plotting**; both plot scripts assert
`n_rows == 168` and refuse to draw otherwise.

**Three predictor families share each panel and are not interchangeable.** FM arms are judged
against the 3-seed FM band; classifier arms against their own single uniform run; Part-X is a GP.
Line style encodes the family. There is **no classifier floor** — those arms have one seed each.

**The three predictor config traps.** Each of these killed a v2 job within 80 seconds:

```
GP)  predictor=gp predictor.gp.n_iters=300     # NOT gp_reg
CLF) predictor=clf_ensemble  +predictor.lightning_trainer.enable_progress_bar=false   # '+' append form
*)   predictor=fm_ensemble    predictor.lightning_trainer.enable_progress_bar=false   # no '+'
```

`partx` with `gp_reg` raises `AttributeError: 'EndpointMCProbabilityBackend' object has no
attribute 'latent_posterior'` at `adaptive_roa/partx/strategy.py:30` — Part-X needs the
`GPProbabilityBackend`. The classifier config has no `lightning_trainer` key to override, so it
needs the `+` append form while fm_ensemble does not.

**The pool's length coupling is INVERTED vs pendulum.** Measured over the gaussian_signal pools:

| level | traj succ | mean len (succ) | mean len (fail) | ratio | PAIR succ |
|---|---:|---:|---:|---:|---:|
| low | 0.175 | 484.7 | 7.3 | 66.9 | 0.934 |
| med | 0.119 | 549.9 | 13.7 | 40.3 | 0.845 |
| high | 0.100 | 567.4 | 19.6 | 28.9 | 0.762 |

On pendulum, successes end at the goal (~136 steps) while failures run to the 1001-step cap, so
buying the uncertain band buys the SHORT rollouts and arms accrue pairs at ~0.4× the control's
rate. **Here it is reversed**: successes are the long rollouts and failures die almost immediately
(7–20 steps). Arms that select into the uncertain band buy pairs at **2.6–4.8× the control's
rate**. That is a large part of why every FM adaptive arm wins on cartpole while the same arms
lose at matched trajectory count on pendulum. The trajectory-vs-pair budget mismatch is a property
of the pool, not of the acquisition rule.

**`attractor_radius` is 0.2, but the data labels success as an L2 ball of radius 0.05** over the
full 4-D state. This is deliberate (decision 2026-08-17) and is **not cosmetic**:
`attractor_radius` labels the K sampled endpoints in `endpoint_mc_probabilities`, so it defines the
model's predicted p_success and feeds straight into sAUROC / KL / recal. A 4-D ball at r=0.2
against a labelling rule at r=0.05 is (0.2/0.05)⁴ = **256× the volume**, so predicted p_success is
heavily inflated. Every arm shares it, so the *ranking* should survive; the **magnitudes are not
comparable** to an r=0.05 run or to the pendulum numbers.

**Magnitude screen, not non-finite screen.** Flag any epoch whose max of
`success_mae`/`failure_mae`/`overall_mae` exceeds 1.0 **or** is non-finite. A non-finite-only screen
once cleared a cartpole arm carrying 9.4e+09. All 504 epochs here pass: worst per-arm MAE was
0.547–0.926 (low), 0.593–0.821 (high). At high noise the *control* seeds carry the largest errors
while adaptive arms sit lower — the reverse of low.

**A stale artifact mtime is not a hang.** Arms routinely go 25–35 min between artifacts inside MC
estimation or a 5,200-batch ROA eval. Confirm from the log tail
(`tail -c 1500 | tr '\r' '\n'`) before diagnosing. Per-epoch cadence also *grows* through a run —
`yield_a1` went from 13 min at epoch 1 to 27 min at epoch 11 as the training set accumulated.

**Job-level state hides task failure.** `sacct -X` can report COMPLETED / ExitCode 0:0 while the
python step exited 1. Check the `.0` step's exit code. All 14 high steps verified COMPLETED 0:0.

**A timed-out `find` returns 0**, which reads as catastrophic data loss. Gate on the exit code and
report "count unreliable" rather than the number.

**Infrastructure.** `/common/users` is krb5-authenticated. When the ticket expires, access flips to
`EACCES` while `/common/home` keeps working, and every tool touching the scratchpad dies —
including Bash, which cannot create its own task dir. Fix: `kinit st1122@CS.RUTGERS.EDU`.

---

## 5. Provenance

- **All 504 epochs have `filter_diagnostics` null.**
- **Collapsed-epoch screen: 0 hits across all 504 rows** (`scripts/screen_collapsed_epochs.py`,
  run 2026-08-19 over `CPV2_low/med/high`). Note `yield_a1` at med epoch 8: KL jumps .0191 → .0506
  and sAUROC dips .9735 → .9585 for one epoch, then recovers to .0246 at ep9 and finishes best of
  the series at .0151. Well clear of the screen's chance thresholds — a wobble, not a collapse, and
  it does not touch the epoch-11 verdict.
- **d2_ratio verified per arm at full depth** (1,800 acquisitions each): seeds + `clf_dir00` at
  1800 d1 / 0 d2; the eight adaptive arms at 0 / 1800; both anchored arms at exactly 900 / 900.
- **Acquisition divergence verified.** Jaccard of acquired indices vs `dir00_s42`: adaptive arms
  0.013–0.023 (near-disjoint), anchored arms 0.341–0.349 (the ≈0.5 signature). The uniform arms
  show Jaccard 1.000 against each other — **expected, not a defect**: d1 reads sequentially from
  the seed-independent `shuffled_indices_0.txt`, so every pure-uniform arm draws the identical
  index set and the three seeds differ only in training stochasticity. The 3-seed floor therefore
  bounds training noise, not sampling noise.
- Companion: `../pendulum/` for the pendulum gaussian_signal campaign.
