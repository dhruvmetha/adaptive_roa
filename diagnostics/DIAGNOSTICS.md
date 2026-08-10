# Ensemble epistemic acquisition — causal-chain diagnostics

Read-only audit of the completed campaign. No training was launched or resumed; no run
directory, `FINDINGS.md`, `LOG.md`, or `runs.jsonl` was modified. Everything here is
reproducible from `diagnostics/diagnostics.json`, written by `diagnostics/run_diagnostics.py`.

**The question.** Four causal links must *all* hold for epistemic acquisition to produce a
measurable gain over uniform sampling:

| | link | broken means |
|---|---|---|
| 1 | the acquisition score differentiates the candidate pool | every candidate scores alike; selection is arbitrary |
| 2 | different scores acquire different states | the arms are the same experiment under different names |
| 3 | acquired states shift the training distribution | selection happens but training data is unchanged |
| 4 | ensemble disagreement stays above the measurement floor | the epistemic signal is sampling noise |

The campaign measured a null. This audit asks **where** the chain breaks — a null with an
identified broken link is a different result from a null with four intact links.

---

## 1. Runs included and excluded

Inclusion is delegated to `contaminated_runs()` in `scripts/ensemble_verdicts.py`, the same
gate the headline verdicts use, so this audit and `FINDINGS.md` cannot disagree about which
runs are admissible.

- **Included: 50 runs** — 2 predictors x 5 noise levels x 5 arms, seed 42.
- **Excluded: 0 runs.** `contaminated_runs()` returned empty; no run in this experiment was
  flagged for the code-version contamination that affects the older probabilistic-predictor
  families.
- The `dir00` floor seeds (43/44) are not part of the 50 — they carry no `acquisition`
  block to diagnose (they sample uniformly) and enter only as the run-to-run floor.

Per-run completed-epoch counts are in `diagnostics.json :: meta.included`. **Depth is
uneven** — preemption on the Amarel `general` account truncated many arms. Counts range from
19 epochs (all `clf_det`) down to 1 scored epoch (`fm_med_epi_bald`). Every trend statistic
below **excludes epoch 0**, where all arms hold identical pre-acquisition data.

---
## 2. The four diagnostics

### Link 1 — does the acquisition score differentiate the pool?

**Method.** Every arm writes pool-level score statistics each epoch. Read
`<run>/epoch_*/artifacts_v2.json :: acquisition.diagnostics` and derive
`range = score_max - score_min`, `mean_over_max = score_mean / score_max`, and
`enrichment = score_mean_selected / score_mean` — the factor by which the selected batch
beats the pool average. These are true pool statistics over
`n_candidates_evaluated` points, not a proxy computed from the selected batch.

**T1 — Link 1: does the score differentiate the pool? (mean over epochs ≥ 1)**

| pred | level | arm | mean/max | enrichment | score_min | score_max | epochs |
|---|---|---|---|---|---|---|---|
| clf | det | `total` | 0.0157 | 42.99 | +0.00000 | 0.6770 | 18 |
| clf | det | `epi_var` | 0.0051 | 43.74 | +0.00000 | 0.2101 | 18 |
| clf | det | `epi_bald` | 0.0058 | 41.49 | +0.00000 | 0.4348 | 18 |
| clf | det | `aleat` | 0.0130 | 34.72 | +0.00000 | 0.6042 | 18 |
| clf | low | `total` | 0.0197 | 12.11 | +0.00000 | 0.6913 | 11 |
| clf | low | `epi_var` | 0.0041 | 14.07 | +0.00000 | 0.1275 | 10 |
| clf | low | `epi_bald` | 0.0034 | 17.70 | -0.00000 | 0.2599 | 13 |
| clf | low | `aleat` | 0.0151 | 15.23 | +0.00000 | 0.6917 | 12 |
| clf | med | `total` | 0.0378 | 12.20 | +0.00000 | 0.6931 | 11 |
| clf | med | `epi_var` | 0.0049 | 13.63 | +0.00000 | 0.0884 | 11 |
| clf | med | `epi_bald` | 0.0070 | 12.46 | +0.00000 | 0.1971 | 11 |
| clf | med | `aleat` | 0.0373 | 11.80 | +0.00000 | 0.6925 | 10 |
| clf | high | `total` | 0.2379 | 3.83 | +0.00025 | 0.6931 | 18 |
| clf | high | `epi_var` | 0.0166 | 8.47 | +0.00000 | 0.1279 | 18 |
| clf | high | `epi_bald` | 0.0276 | 6.01 | +0.00000 | 0.1637 | 18 |
| clf | high | `aleat` | 0.1948 | 4.27 | +0.00012 | 0.6917 | 18 |
| clf | xhigh | `total` | 0.4814 | 2.04 | +0.00837 | 0.6931 | 13 |
| clf | xhigh | `epi_var` | 0.0783 | 4.69 | +0.00000 | 0.0095 | 9 |
| clf | xhigh | `epi_bald` | 0.0408 | 4.42 | +0.00000 | 0.0237 | 13 |
| clf | xhigh | `aleat` | 0.4642 | 2.09 | +0.00292 | 0.6930 | 11 |
| fm | det | `total` | 0.2019 | 4.99 | +0.00000 | 0.6497 | 9 |
| fm | det | `epi_var` | 0.0274 | 16.05 | -0.01132 | 0.1831 | 18 |
| fm | det | `epi_bald` | 0.0639 | 7.47 | +0.00000 | 0.3916 | 18 |
| fm | det | `aleat` | 0.1571 | 5.98 | +0.00000 | 0.6864 | 18 |
| fm | low | `total` | 0.0494 | 6.92 | +0.00000 | 0.6931 | 11 |
| fm | low | `epi_var` | 0.0049 | 7.41 | -0.01241 | 0.1366 | 11 |
| fm | low | `epi_bald` | 0.0167 | 6.77 | -0.00000 | 0.3003 | 14 |
| fm | low | `aleat` | 0.0428 | 6.76 | +0.00000 | 0.6907 | 10 |
| fm | med | `total` | 0.0994 | 6.32 | +0.00000 | 0.6931 | 6 |
| fm | med | `epi_var` | 0.0085 | 8.91 | -0.01264 | 0.1441 | 5 |
| fm | med | `epi_bald` | 0.0612 | 4.63 | -0.00000 | 0.4092 | 1 |
| fm | med | `aleat` | 0.0814 | 6.51 | +0.00000 | 0.6913 | 9 |
| fm | high | `total` | 0.1709 | 4.43 | +0.00000 | 0.6931 | 18 |
| fm | high | `epi_var` | 0.0088 | 20.07 | -0.01269 | 0.0880 | 16 |
| fm | high | `epi_bald` | 0.0797 | 3.49 | -0.00000 | 0.2014 | 18 |
| fm | high | `aleat` | 0.1535 | 4.68 | +0.00000 | 0.6914 | 18 |
| fm | xhigh | `total` | 0.4902 | 1.89 | +0.00000 | 0.6931 | 16 |
| fm | xhigh | `epi_var` | 0.0176 | 18.14 | -0.01283 | 0.0967 | 12 |
| fm | xhigh | `epi_bald` | 0.1187 | 2.53 | -0.00000 | 0.2535 | 17 |
| fm | xhigh | `aleat` | 0.4361 | 2.01 | +0.00000 | 0.6919 | 17 |
**Pool size is not uniform across levels** (`n_candidates_evaluated`): the four noisy levels
evaluate exactly **50,000** candidates per epoch, but `det` evaluates only **18,100–19,800** —
there the pool is the entire remaining unlabeled dataset, and it drains as points are acquired.
`det` also acquires **100 states/epoch** against **1,000/epoch** at the noisy levels
(T2, `n acquired`). This 10x budget asymmetry is a property of the system configs, not of this
audit, and is not described in `REPORT.md`.

**Reading.** Enrichment exceeds 1 everywhere — from 1.89 (`fm_xhigh_total`) to 43.7
(`clf_det_epi_var`) — so the selected batch is always scored far above the pool average. The
score is doing work. But the epistemic arms are extremely right-skewed: `mean_over_max` runs
0.0034–0.0783 for `epi_var`/`epi_bald`, against 0.013–0.49 for `total`/`aleat`. Almost all of
the pool sits near zero epistemic score while a thin tail carries the signal.

On FM, `epi_var`'s `score_min` is not merely small but **exactly** `-p(1-p)/(K-1)` at p=0.5
(-0.0113 to -0.0128 across levels). The debiased estimator is `Var_m[p] - mean_m[p(1-p)/(K-1)]`,
so a minimum at exactly minus the correction term means **raw between-member variance is
identically zero** for part of the pool at every epoch: those members agreed perfectly.

**Verdict: INTACT** — the score separates candidates at every level and both predictors
(enrichment >> 1). The exact stated criterion (median debiased `epi_var`, fraction clipped to
zero) is **UNAVAILABLE**: only min/mean/max were persisted, never quantiles or a zero count.
The `score_min` evidence above establishes that the zero mass is non-empty but does not bound
its size.

---

### Link 2 — do different scores acquire different states?

**Method.** Read the acquired pool indices per epoch from
`<run>/epoch_*/artifacts_v2.json :: acquisition.d2_indices` (`d1_indices` for `dir00`, which
samples uniformly), accumulate them per arm, and compute pairwise Jaccard overlap both between
treatment arms and against `dir00`.

**T2 — Link 2: do different scores acquire different states? (cumulative Jaccard, final epoch)**

| pred | level | n acquired | treat-pair min | treat-pair max | max-overlap pair | vs `dir00` min | vs `dir00` max |
|---|---|---|---|---|---|---|---|
| clf | det | 1900 | 0.460 | 0.566 | `total + aleat` | 0.048 | 0.054 |
| clf | low | 10956–14000 | 0.583 | 0.718 | `epi_bald + aleat` | 0.103 | 0.144 |
| clf | med | 11000–14000 | 0.637 | 0.720 | `total + epi_var` | 0.117 | 0.120 |
| clf | high | 19000 | 0.338 | 0.605 | `total + aleat` | 0.171 | 0.173 |
| clf | xhigh | 10000–14000 | 0.086 | 0.597 | `total + aleat` | 0.113 | 0.128 |
| fm | det | 600–1900 | 0.089 | 0.196 | `epi_var + aleat` | 0.017 | 0.022 |
| fm | low | 11000–15000 | 0.293 | 0.373 | `total + aleat` | 0.128 | 0.203 |
| fm | med | 2000–10000 | 0.060 | 0.378 | `total + aleat` | 0.029 | 0.076 |
| fm | high | 17000–19000 | 0.286 | 0.655 | `total + aleat` | 0.170 | 0.188 |
| fm | xhigh | 13000–18000 | 0.174 | 0.554 | `total + aleat` | 0.143 | 0.167 |
**Reading.** Overlap with `dir00` is low everywhere (0.017–0.203): every treatment arm acquires
a set clearly distinct from uniform sampling. That is the strict form of the stated criterion,
and it holds at all ten cells.

The treatment arms are, however, far more like *each other* than like uniform. On the
classifier they overlap 0.46–0.72 at every level — at `clf_med`, `total` and `epi_var` share
72% of their cumulative acquisitions despite scoring by different quantities. **`total + aleat`
is the maximum-overlap pair in 8 of 10 cells.** That is the decomposition's own prediction:
when aleatoric uncertainty dominates the total, entropy-based and aleatoric acquisition select
nearly the same points, and the negative control stops being a control.

**Verdict: INTACT vs `dir00` at all 10 cells; DEGRADED between treatments** where the pairwise
maximum exceeds 0.5 — `clf` at all five levels (0.566–0.720) and `fm` at high (0.655) and
xhigh (0.554). `fm` det/low/med stay separated (0.196/0.373/0.378).

---

### Link 3 — do the acquired states shift the training distribution?

**Method.** Map each acquired pool index back to its initial state via
`<pool>/<level>/train.npz :: starts` permuted by
`train_test_splits/shuffled_indices_0.txt`, take its binary label from
`shuffled_labels_0.txt`, and look up the oracle success probability by nearest neighbour on the
158x315 rollout grid in `<level>/eval_success_prob.npz` (90 rollouts per cell). `det` is
excluded: the system is deterministic, so p is degenerate in {0,1} and carries no information
beyond the label. Pool root is
`/common/users/shared/pracsys/genMoPlan/data_trajectories/noisy/pendulum/lqr`.

**T3 — Link 3: do acquired states shift the training distribution? (mean over epochs ≥ 1)**

| pred | level | arm | batch mean true p | frac ambiguous (0.2–0.8) | cumulative label marginal |
|---|---|---|---|---|---|
| clf | low | `dir00` | 0.3926 | 0.0283 | 0.3917 |
| clf | low | `total` | 0.4222 | 0.1251 | 0.4202 |
| clf | low | `epi_var` | 0.4083 | 0.1379 | 0.4055 |
| clf | low | `epi_bald` | 0.4631 | 0.1177 | 0.4628 |
| clf | low | `aleat` | 0.4650 | 0.1203 | 0.4642 |
| clf | med | `dir00` | 0.3959 | 0.0521 | 0.3950 |
| clf | med | `total` | 0.4019 | 0.2396 | 0.4016 |
| clf | med | `epi_var` | 0.3598 | 0.2378 | 0.3589 |
| clf | med | `epi_bald` | 0.3272 | 0.2234 | 0.3259 |
| clf | med | `aleat` | 0.4146 | 0.2527 | 0.4137 |
| clf | high | `dir00` | 0.4059 | 0.1321 | 0.4031 |
| clf | high | `total` | 0.0187 | 0.0043 | 0.0173 |
| clf | high | `epi_var` | 0.0186 | 0.0092 | 0.0181 |
| clf | high | `epi_bald` | 0.0277 | 0.0302 | 0.0266 |
| clf | high | `aleat` | 0.0244 | 0.0042 | 0.0222 |
| clf | xhigh | `dir00` | 0.4837 | 0.2667 | 0.4783 |
| clf | xhigh | `total` | 0.1273 | 0.0374 | 0.1272 |
| clf | xhigh | `epi_var` | 0.2140 | 0.3130 | 0.2120 |
| clf | xhigh | `epi_bald` | 0.3280 | 0.4115 | 0.3288 |
| clf | xhigh | `aleat` | 0.1289 | 0.0427 | 0.1333 |
| fm | low | `dir00` | 0.3936 | 0.0282 | 0.3934 |
| fm | low | `total` | 0.6790 | 0.1078 | 0.6767 |
| fm | low | `epi_var` | 0.6895 | 0.0713 | 0.6880 |
| fm | low | `epi_bald` | 0.6863 | 0.0818 | 0.6849 |
| fm | low | `aleat` | 0.6674 | 0.1088 | 0.6643 |
| fm | med | `dir00` | 0.3958 | 0.0498 | 0.3920 |
| fm | med | `total` | 0.5783 | 0.2737 | 0.5737 |
| fm | med | `epi_var` | 0.6763 | 0.1746 | 0.6774 |
| fm | med | `epi_bald` | 0.9324 | 0.0600 | 0.9280 |
| fm | med | `aleat` | 0.5825 | 0.2404 | 0.5832 |
| fm | high | `dir00` | 0.4059 | 0.1321 | 0.4031 |
| fm | high | `total` | 0.4572 | 0.4456 | 0.4515 |
| fm | high | `epi_var` | 0.5308 | 0.2973 | 0.5261 |
| fm | high | `epi_bald` | 0.5208 | 0.2119 | 0.5174 |
| fm | high | `aleat` | 0.4704 | 0.4427 | 0.4633 |
| fm | xhigh | `dir00` | 0.4829 | 0.2665 | 0.4801 |
| fm | xhigh | `total` | 0.4676 | 0.7711 | 0.4607 |
| fm | xhigh | `epi_var` | 0.5097 | 0.3625 | 0.5112 |
| fm | xhigh | `epi_bald` | 0.5107 | 0.2802 | 0.5107 |
| fm | xhigh | `aleat` | 0.4708 | 0.7292 | 0.4624 |
**Reading.** Selection reaches the training set everywhere, and the two predictors move in
**opposite directions**.

On the classifier at `high`, every treatment arm collapses the acquired batch onto
near-certain-failure states: mean true p falls from 0.406 (`dir00`) to **0.019–0.028**, and the
cumulative label marginal follows to 0.017–0.027. `clf_xhigh` shows the same drag more weakly
(0.127–0.329 vs 0.484). This is the known classifier pathology, now measured at its source.

On flow matching the shift goes the other way — *toward* the ambiguous band. At `low` the
arms acquire batches with mean true p 0.667–0.690 against `dir00`'s 0.394; at `med`,
`epi_bald` reaches **0.932**. At `xhigh` the mean barely moves but the composition does:
`total` and `aleat` draw 73–77% of their batch from the ambiguous 0.2–0.8 band against
`dir00`'s 27%.

**Verdict: INTACT at all 8 available cells** — treatment marginals sit far outside the `dir00`
value in every case. **UNAVAILABLE for `det`** (both predictors): no oracle probability grid
exists for a deterministic system.

This link matters most for interpretation: FM acquisition is *not* inert. It substantially
rewrites the training distribution and still moves no metric.

---

### Link 4 — is ensemble disagreement above the measurement floor?

**Method.** FM members estimate p by Monte Carlo with `k_acq = 20` samples, so each member's p
carries binomial sampling noise. The debiased estimator already subtracts
`mean_m[p(1-p)/(K-1)]`; its magnitude is the floor below which between-member disagreement
cannot be resolved. Recover the floor empirically as `-score_min` from the `epi_var` arm (see
Link 1: raw variance hits exactly zero, so the minimum *is* minus the correction), and compare
it to `epistemic_mean` over the pool. Classifier members are exact (`member_sample_size: null`,
K=None), so their floor is identically zero.

**T4 — Link 4: is ensemble disagreement above the measurement floor? (`epi_var` arm)**

| pred | level | epochs | MC floor | mean debiased between-member var | signal / floor |
|---|---|---|---|---|---|
| clf | det | 18 | 0 (exact) | 0.00290 | n/a (no MC noise) |
| clf | low | 10 | 0 (exact) | 0.00166 | n/a (no MC noise) |
| clf | med | 11 | 0 (exact) | 0.00136 | n/a (no MC noise) |
| clf | high | 18 | 0 (exact) | 0.00572 | n/a (no MC noise) |
| clf | xhigh | 9 | 0 (exact) | 0.00180 | n/a (no MC noise) |
| fm | det | 18 | 0.01132 | 0.02252 | 1.99 |
| fm | low | 11 | 0.01241 | 0.00668 | 0.54 |
| fm | med | 5 | 0.01264 | 0.01081 | 0.86 |
| fm | high | 16 | 0.01269 | 0.01528 | 1.20 |
| fm | xhigh | 12 | 0.01283 | 0.02765 | 2.15 |
**Reading.** This is the audit's most consequential number. On flow matching at `low` and
`med`, mean debiased between-member variance is **below the sampling floor** — signal/floor
0.54 and 0.86. The epistemic quantity those arms acquired on was, on average, not
distinguishable from the noise introduced by estimating each member's p from 20 samples.
`fm_high` is marginal (1.20). Only `fm_det` (1.99) and `fm_xhigh` (2.15) clear the floor
comfortably.

The floor is a fixed property of the estimator, `(1/2K)(1 - 1/M) ~ 0.021` nats at K=20, M=5 for
BALD, and it **does not shrink as members are added** — only as K grows. Adding ensemble
members cannot fix this; raising `k_acq` can.

The classifier has no such problem — exact members, zero floor — but its epistemic magnitudes
are small in absolute terms (0.0014–0.0057 nats).

**Verdict: BROKEN for `fm` low (0.54) and med (0.86); MARGINAL for `fm` high (1.20); INTACT for
`fm` det (1.99) and xhigh (2.15); INTACT for `clf` at all five levels** (exact members, no MC
noise).

A stronger version of this check — per-member variance on the evaluation set rather than the
candidate pool — is **UNAVAILABLE**: `full_roa_per_point.npz` stores only the ensemble mean
(`p_success`, `p_failure`, `p_invalid`); per-member arrays were never written. Recovering them
means re-running inference for 5 members x ~19 epochs x ~40k points per run.

**Lineage.** `warm_start: false` in every run's `.hydra/config.yaml`, and
`engine.py:165` only passes a resume checkpoint when `warm_start` is set. Members are
reinitialised from scratch each epoch, so the "all members converge to one basin because they
share a warm start" failure mode does not apply here.

---
## 3. Cross-link summary

`INTACT` = link holds · `DEGRADED` = holds against the control but weakened between treatments
· `BROKEN` = fails the stated criterion · `UNAVAILABLE` = required data was never persisted.

| predictor | level | Link 1 score spread | Link 2 distinct sets | Link 3 distribution shift | Link 4 signal vs floor |
|---|---|---|---|---|---|
| clf | det | INTACT (enrich 34.7–43.7) | DEGRADED (0.566) | UNAVAILABLE (deterministic) | INTACT (exact members) |
| clf | low | INTACT (12.1–17.7) | DEGRADED (0.718) | INTACT (0.393 → 0.406–0.464) | INTACT (exact members) |
| clf | med | INTACT (11.8–13.6) | DEGRADED (0.720) | INTACT (0.396 → 0.326–0.415) | INTACT (exact members) |
| clf | high | INTACT (3.8–8.5) | DEGRADED (0.605) | INTACT (0.406 → **0.019–0.028**) | INTACT (exact members) |
| clf | xhigh | INTACT (2.0–4.7) | DEGRADED (0.597) | INTACT (0.484 → 0.127–0.329) | INTACT (exact members) |
| fm | det | INTACT (5.0–16.1) | INTACT (0.196) | UNAVAILABLE (deterministic) | INTACT (1.99) |
| fm | low | INTACT (6.8–7.4) | INTACT (0.373) | INTACT (0.394 → 0.667–0.690) | **BROKEN (0.54)** |
| fm | med | INTACT (4.6–8.9) | INTACT (0.378) | INTACT (0.396 → 0.578–0.932) | **BROKEN (0.86)** |
| fm | high | INTACT (3.5–20.1) | DEGRADED (0.655) | INTACT (0.406 → 0.451–0.531) | MARGINAL (1.20) |
| fm | xhigh | INTACT (1.9–18.1) | DEGRADED (0.554) | INTACT (comp. shift, 27% → 77% ambiguous) | INTACT (2.15) |

**What the chain says.** No cell has a fully broken chain, and Links 1 and 3 hold everywhere
they can be evaluated. The null is therefore **not** explained by "acquisition did nothing" —
the arms demonstrably score, select, and reshape the training set differently.

Two distinct partial failures account for the rest:

1. **`fm` low and med — the epistemic signal is below its own measurement floor.** Those arms
   ranked candidates on a quantity dominated by K=20 sampling noise. This is a fixable
   instrumentation defect, and it is the one mechanism identified here that could hide a real
   effect. `fm_high` sits close enough to the floor (1.20) to be suspect too.
2. **Everywhere else — the arms converge on the same states.** `total + aleat` is the
   maximum-overlap pair in 8 of 10 cells, exactly as the curvature argument predicts once
   aleatoric uncertainty dominates. An "epistemic" arm that acquires what the aleatoric control
   acquires cannot separate from it.

Neither mechanism rescues the headline result, but they bound it: **the campaign is a valid
test of epistemic acquisition only at `fm` det/xhigh and across the classifier**. At `fm`
low/med it is not a test of the hypothesis at all — it is a measurement of noise.

---

## 4. Figures

Generated by `diagnostics/make_figures.py` (read-only). Palette validated with the `dataviz`
skill's `validate_palette.js` — lightness band, chroma floor, CVD separation and normal-vision
floor all PASS on the light surface.

### `fig0_ground_truth_prob.png` — the target function

Oracle p(success) over the full evaluation state space at all five noise levels. The region of
attraction is a sharp band at `det` and blurs into a wide probabilistic gradient by `xhigh`.
This is what the models are trying to learn, and it is the reference for the maps below.

### `fig1_uncertainty_vs_epoch.png` — the decomposition over time

Total `H(p̄)`, aleatoric `E_m[H]` and epistemic `I(y;Θ)` over the candidate pool, per epoch,
for all ten level x predictor cells, with the FM Monte-Carlo floor drawn as a dashed line.
Two things are visible immediately: **aleatoric uncertainty tracks total almost exactly** at
every noisy level (the green epistemic trace hugs zero), and on FM the epistemic trace sits at
or under the dashed floor for most of `low`, `med` and `high` — the tabular Link 4 result in
trajectory form. Each panel carries its own y-scale; magnitudes differ ~10x across levels.

### `fig2_state_space_fm.png`, `fig2_state_space_clf.png` — the learned probability field

Predicted p(success) over the **full evaluation set** (39,770 points, not the test subset) for
every arm, beside the oracle probability map and the binary label map. Each arm is drawn at
its own deepest completed epoch, labelled per panel, because preemption left arms at different
depths.

The two predictors look completely different, and the difference matters.

**Flow matching** reproduces the **graded oracle probability band** — including its width —
rather than the sharply bimodal binary label field it was trained against, at every noisy
level. It is recovering calibrated aleatoric structure. And across `low`, `med`, `high` and
`xhigh` the five arms are **visually indistinguishable from one another and from `dir00`**.
The FM null is not a small effect buried in metric noise: the learned functions genuinely
coincide. `det` is the exception — there the arms differ visibly, and all render the sharp
boundary imperfectly.

**The classifier** matches that picture only at `det`, `low` and `med`. At `high` and `xhigh`
every arm — `dir00` included — **inflates the success region far beyond the oracle**, until the
failure basin collapses to a small blob, and the arms are clearly distinguishable from each
other. This is the known `clf` high-noise catastrophe, visible directly in the learned function.

One observation here resists easy explanation and is recorded rather than resolved: at
`clf_high` the treatment arms acquired batches that are ~98% failures (T3: mean true p
0.019–0.028) yet their learned fields over-predict *success* more than the balanced `dir00`
control does. Acquiring far-from-boundary, near-certain-failure states appears to starve the
model of boundary information rather than biasing it toward the majority label. Confirming that
mechanism needs per-epoch calibration data this audit did not have.

---
## 5. Discrepancies and gaps

### Contradicts or is absent from `REPORT.md` / `FINDINGS.md`

1. **`det` runs on a different budget than every other level.** The noisy levels evaluate a
   fixed 50,000-candidate pool and acquire 1,000 states/epoch (19,000 cumulative). `det`
   evaluates 18,100–19,800 — the whole remaining unlabeled dataset, draining as it goes — and
   acquires 100 states/epoch (1,900 cumulative). A 10x acquisition-budget and ~2.5x pool-size
   asymmetry sits underneath every `det`-vs-noisy comparison, and neither document mentions it.
   Source: `link1.*.n_candidates_evaluated`, `link2.*.n_cumulative`.

2. **Selection is not purely score-ranked.** Every arm runs
   `selection_rule: greedy_diverse` — take the top `5 x n_select` by score, then farthest-point
   sample in initial-state space. Only one fifth of the score ordering survives to selection,
   and the final choice is driven by geometric spread. This mechanically pushes different arms
   toward the same well-spread subset and is a plausible contributor to the high inter-arm
   Jaccard in Link 2. Neither document describes it. Source: `link1.*.selection_rule`.

3. **`warm_start: false` everywhere.** Members reinitialise from scratch each epoch. This
   *removes* a failure mode rather than adding one, but the campaign documents never state it,
   so the "shared warm start collapses member diversity" hypothesis was never live.
   Source: each run's `.hydra/config.yaml`; `engine.py:165`.

4. **The `total` ≈ `aleat` overlap is stronger and more general than reported.** `FINDINGS.md`
   records it at `high` (0.67) and `xhigh` (0.57). It is in fact the maximum-overlap pair in
   **8 of 10** level x predictor cells, including `clf_low` and `clf_det`, where noise is low
   and the curvature argument was not expected to bite.

5. **Nothing here contradicts the headline nulls.** The metric-level conclusions in
   `REPORT.md` and `FINDINGS.md` stand as written. What changes is their scope: at `fm` low and
   med the experiment did not test the hypothesis (Link 4 broken), so those two cells should
   not be counted as evidence against epistemic acquisition.

### Expected data that could not be located

| wanted | why unavailable |
|---|---|
| Median debiased `epi_var`, and the fraction of the pool clipped to exactly zero | Only `score_min` / `score_mean` / `score_max` are persisted per epoch — no quantiles, no zero count. This is the exact stated Link 1 criterion, so Link 1 is answered on enrichment evidence instead. |
| Per-member p on the evaluation set (the strong form of Link 4) | `full_roa_per_point.npz` stores only ensemble means: `start_states`, `p_success`, `p_failure`, `p_invalid`, `true_labels`, `lambda_star`, `delta`, `attractor_radius`. Recomputing needs inference over 5 members x ~19 epochs x ~40k points per run. |
| Oracle p(success) for `det` (Link 3) | No `eval_success_prob.npz` exists under `lqr/det`; the system is deterministic and p is degenerate in {0,1}. `det` Link 3 is UNAVAILABLE by construction, not by omission. |
| Matched-depth comparison at `fm_med` | Preemption left `fm_med_epi_bald` with a single scored epoch and `fm_med_epi_var` with five. The `fm_med` row of every table rests on very thin data and should not be read as a converged result. |

### Inferences made from file name or array shape

Stated explicitly, as required:

- **`eval_success_prob.npz` is the oracle.** Its `successes`/`trials` arrays (`trials` = 90 per
  cell) and a `p_success` equal to their ratio are consistent with a 90-rollout Monte-Carlo
  estimate on a 158x315 grid. No file documents this; it is inferred from field names and shapes.
- **`d2_indices` are the epoch's acquisitions, `d1_indices` the initial/uniform draw.** Inferred
  from the naming and from `dir00` — which acquires uniformly — carrying only `d1_indices`.
  Counts match `samples_per_epoch` at every level, which corroborates it.
- **The FM MC floor is read as `-score_min` of the `epi_var` arm.** This is exact *only if* raw
  between-member variance reaches zero somewhere in the pool. The measured minima match
  `-p(1-p)/(K-1)` at p=0.5 to five decimals at every level, which is strong evidence it does,
  but the zero itself was never recorded directly.
- **Nearest-neighbour lookup** maps eval points onto the oracle grid; the eval set is 39,770 of
  the 49,770 grid cells, so ~20% of cells have no eval point (the white speckle in fig0/fig2).

---

## Reproducing

```bash
python diagnostics/run_diagnostics.py   # writes diagnostics/diagnostics.json
python diagnostics/make_figures.py      # writes fig0/fig1/fig2 PNGs
```

Both are read-only over
`/common/users/shared/pracsys/adaptive_roa_experiments/ensemble_epistemic` and the pool root
`/common/users/shared/pracsys/genMoPlan/data_trajectories/noisy/pendulum/lqr`.
Constants: `n_members = 5`, `k_acq = 20` (FM), classifier members exact (`K = None`).
Every table cell above is a direct read or a mean-over-epochs of values in `diagnostics.json`;
the per-diagnostic method notes give the exact JSON path and source file for each.
