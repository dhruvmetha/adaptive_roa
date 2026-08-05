# Ensemble Epistemic Acquisition — current findings

**Status: campaign in progress.** Classifier arms at epochs ~9–13 of 19 and will stop at ~17
(walltime, see §6); deterministic complete at 19/19; flow-matching arms at epoch 4 of 19. This file holds only what is *currently defensible*.
`LOG.md` is the chronological record including superseded claims and their corrections.

Standard of evidence used throughout: a gap counts only if it exceeds **2×SD of three
genuinely-distinct seeds** of the same configuration (the run-to-run floor) at **two consecutive
epochs**. The floor is **pooled across epochs** (`2·sqrt(mean(var_e))`), not taken from a single
epoch: with three seeds each epoch's SD has two degrees of freedom, and the deterministic floor
was measured to vary **15.6×** between epochs — enough to flip a verdict depending on which slice
was used. Metrics are reported post-recalibration (`UNC − RES`) — the component recalibration
cannot repair — because raw Brier differences at high noise turned out to be almost entirely
calibration, which is routinely fixed post hoc.

---

## 1. Established

### The decomposition is mechanically correct
`H(p̄) = E_m[H(p_m)] + I(y;m|x)` holds to **exact float precision** on real model outputs (the
`total` arm computes the left side, `epi_bald`/`aleat` the two right-hand terms; residual 0.0e+00).
Epoch-0 identity confirmed on both predictors: before the first acquisition all five arms are
identical to within 6e-05.

### Separation is a monotone function of noise, with a threshold
Arms only diverge once aleatoric mass in the candidate pool exceeds roughly 0.1:

| level | aleatoric pool mass | do arms separate? |
|---|---|---|
| det, low, med | 0.005–0.034 | no |
| high | 0.17–0.21 | yes |
| xhigh | 0.34–0.41 | yes, most |

### Deterministic pendulum (complete, 19/19 epochs) — MARGINAL, metric-dependent
With the complete 18-epoch floor, "adaptive beats random" holds on **Brier only** (−1.3× to −1.8×)
and **fails on log score** (−0.3× to −0.4×, within noise). auc/auprc are saturated (0.9999) and
uninformative. Two proper scoring rules on identical predictions disagree, because log score's
seed floor is ~15× larger (0.0236 vs 0.0016).

Claim as: *a small Brier improvement that does not replicate on log score.*

**Unaffected:** the four adaptive arms are indistinguishable from each other on every metric —
which is the design's actual validation prediction for this level (aleatoric ≈ 0, so the score
choice should not matter). With aleatoric ≈ 0 there is nothing to separate, so the score choice
does not matter while adaptive-vs-random does. This is the design's validation case, passed.

### low noise is marginal; med noise is a null
Against **pooled** floors over 4 epochs, low sits at −1.0 to −1.4× (all four arms just over the
threshold, in the helping direction) and med at −0.4 to −1.1× (straddling it). Neither level shows
a score effect: the arms are within ~0.4× of each other at both levels.

*Superseded: med was previously reported as "adaptive helps (−1.5 to −3.1×)" against a
single-epoch floor. The pooled floor is 3.4× larger and the effect does not survive it.*

### high noise: three of four adaptive arms are worse than random
`epi_bald`, `epi_var` and `total` all exceed the pooled floor in the harmful direction at two
consecutive epochs (+3.1× to +12.0×). The negative control `aleat` does **not**: it reads −0.6×
at one epoch and +2.6× at the next, so it is inconsistent.

Only the *sign* is reportable here. Magnitudes swing between consecutive epochs (`epi_bald`
+12.0× → +4.1×) and **the ordering among arms is not stable** — both are excluded from the claim.

*Withdrawn: an earlier entry flagged `aleat` as the least harmful arm at high (an inversion
against xhigh) as an open question. With six more epochs it is now among the most harmful there
(+20.3×) and `total` the least (+1.6×). The inversion was shallow-epoch noise.*

**At high, only the sign has ever replicated.** Three finer patterns read there — the arm
ordering, a tail-depth mechanism, and this inversion — each dissolved with more data. Report
"adaptive is worse than random at high" and nothing beyond it.

### xhigh (classifier): total entropy does real damage; the epistemic split removes it
Re-verified 2026-08-04 14:25 against a **pooled** floor (4 epochs; recal 2×SD = 0.00095),
at two consecutive epochs:

| arm | ep7 | ep8 |
|---|---|---|
| `aleat` | +30.4× | +39.4× |
| `total` | +30.4× | +25.4× |
| `epi_bald` | −1.1× | −1.1× |
| `epi_var` | −0.3× | −0.8× |

(pooled over 7 epochs; `total` is the clean post-preemption relaunch, so every arm here is
first-run data)

Acquiring on the aleatoric component is **27–31× the pooled floor worse than random sampling**;
`total` measured 17–22× against a single-epoch floor before its relaunch and is being re-derived.
Acquiring on either epistemic component is **indistinguishable from random** (0.3–0.5×).
Parity is the expected ceiling, not a shortfall: the epistemic split restores a balanced,
representative sample, which is what random already provides. The value is in avoiding total
entropy's skew.

*(Being re-derived: `clf_xhigh_total` was preempted and relaunched clean, so no epoch is currently
shared by all five arms.)*

---

## 2. Mechanism (measured, not inferred)

Each epoch records the pool indices it acquired, so the **true** success probability of every
purchased point is recoverable.

### Harm tracks how far an arm drags the training marginal — across levels, not within them
Over 112 arm-epochs spanning all four stochastic levels: corr(fraction of ambiguous points
acquired, harm) = **−0.429**, corr(marginal skew |mean_p − 0.5|, harm) = **+0.406**. The effect is
strong at xhigh (r = −0.748) and **absent within high** (r = +0.259, wrong sign), where all arms
are compressed into mean_p 0.008–0.079 — too narrow a band to discriminate. So this explains the
dose-response *between* noise levels, and explains nothing about differences *between arms* at
high.

### Why total entropy fails at xhigh
| arm | fraction of acquired points genuinely ambiguous |
|---|---|
| `epi_bald` | **51–68%** |
| `epi_var` | 26–31% |
| `total` / `aleat` | **2–4%** |

Total entropy harvests near-certain failures and crushes the training marginal to p ≈ 0.12; the
epistemic score recovers ambiguous states by ~20× and restores a near-balanced marginal. This is
exactly the pathology the previous campaign diagnosed, and the decomposition repairs it.

### BALD is not "epistemic focus" — it is epistemic focus divided by p(1−p)
`BALD ≈ Var_m[p] / (2·p̄(1−p̄))`, because `H''(p) = −1/(p(1−p))`. For **identical** member
disagreement, BALD is amplified **34.9× at p = 0.01** versus p = 0.5. `epi_var` (debiased variance)
has no such factor and is flat.

So BALD systematically hunts extreme probabilities while total entropy, maximised at p = 0.5,
systematically hunts ambiguous ones. This is a property of the functional form, not a bug, and it
holds even with exact probabilities and no sampling.

**This is the real argument for `epi_var`, and it is not the one the design gave.** The design
justified `epi_var` by freedom from finite-K MC bias. That bias was measured on real flow matchers
at **97.3% of the raw BALD signal** (0.01946 against the analytic 0.02000) — yet it does *not*
reorder anything, because a near-constant offset preserves ranking. The advantage that actually
shows up is freedom from the curvature weighting.

---

## 3. The classifier half is largely a negative result about miscalibration

At high noise, from the same candidate pool:

| predictor | mean true p of acquired points | fraction ambiguous |
|---|---|---|
| flow matching | 0.52–0.59 | 21–80% |
| classifier | 0.02–0.03 | ~0% |

The classifier's probability estimates are distorted enough (7.7× worse Brier than FM before any
acquisition) that **every** score — including total entropy, which targets p = 0.5 by construction
— lands on states whose true p is ~0.02. The scores are computed on p̄ values that are simply
wrong, so no score can select correctly.

**Consequence:** "BALD is worse than non-adaptive at high" is true *as measured on this
classifier*, and is **not** evidence about BALD as a method. The flow-matching arms are the fair
test and are at epoch 4 of 19.

---

## 4. Not yet established

- **All flow-matching results.** No FM floor exists yet (both seed replicates still early). Two
  consecutive epochs show every adaptive arm beating the control, with `total` the only arm
  costing discriminative power — but no gap is yet distinguishable from noise.

  The FM floor replicates are verified config-identical to the main `dir00` arm apart from the
  seed (checked against the recorded Hydra overrides, not just the launch command). One asymmetry
  to carry forward when the floor lands: `dir00` and `s43` run on iLab while `s44` runs on Amarel,
  so cross-cluster float nondeterminism is folded into that floor. That inflates it, which is
  **conservative** — it makes significance harder to claim, not easier — but it means the FM floor
  is not directly comparable in magnitude to the classifier floors, which were measured
  within-cluster.
- **Whether the epistemic split helps a well-calibrated predictor at all.** This is the campaign's
  actual question and it is unanswered.
- **`epi_var` vs `epi_bald`.** On the classifier both are exact (members enumerated, K = None), so
  the comparison there is uninformative about the estimator question. Only FM at K = 20 tests it.

## 5. Cautions learned the hard way

- **Single-epoch reads lie.** On the only complete curve (deterministic), `aleat` was +10.7× the
  floor *worse* at epoch 4 and the *best* arm by epoch 18.
- **A floor of ~0 means a broken floor, not a precise one.** Ensemble members were seeded from a
  config key that exists in no config, so seed replicates trained bit-identical models
  (fixed in 4c7c561).
- **Requeued preemptions are invisible** to a `PREEMPTED` query and silently overwrite their own
  early epochs, because the engine has no resume logic (fixed in `check_preempted.sh`).

---

## 6. Known coverage limits (not defects)

- **Classifier arms stop at ~epoch 17–18, not 19.** Submitted with a 1-day walltime; 19 epochs
  needs ~25h at their measured ~1.28h/epoch. SLURM refuses to raise `TimeLimit` on a running job,
  and relaunching would discard 12h × ~15 arms. Matched-epoch analysis loses one or two epochs of
  depth and nothing else.
- **The FM floor will cap around epoch 15.** `fm_high_dir00_s44` runs on Amarel at ~4.5h/epoch
  (versus ~2.5h on iLab) against a 3-day limit, which is Amarel's maximum — so it cannot reach 19.
  The FM *arms* themselves are safe (iLab, 7-day limit, ~48h needed). Consequence: FM comparisons
  are floor-backed up to ~epoch 15 and unbacked beyond it. Any FM claim at epochs 16–19 must say so.
