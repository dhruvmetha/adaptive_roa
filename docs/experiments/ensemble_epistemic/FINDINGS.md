# Ensemble Epistemic Acquisition — current findings

**Status as of 2026-08-09.** This file holds only what is *currently defensible*; `LOG.md` is the
chronological record including superseded claims and their corrections.

| half | state |
|---|---|
| **[CLF]** | det complete 19/19; **high complete 19/19 on all five arms** (rescored, §1); low/med/xhigh from truncated curves (walltime, see §6) |
| **[FM]** | **high scored six times, floor halved, null (§4)**; **xhigh scored three times, null (§4b)**; det preliminary (§4b); low and med running, not yet scoreable |

Three [FM] levels now have verdicts. The [FM] half is no longer "not established" — that framing
survived into this document after it stopped being true and is corrected here.

Standard of evidence used throughout: a gap counts only if it exceeds **2×SD of three
genuinely-distinct seeds** of the same configuration (the run-to-run floor) at **two consecutive
epochs**. The floor is **pooled across epochs** (`2·sqrt(mean(var_e))`), not taken from a single
epoch: with three seeds each epoch's SD has two degrees of freedom, and the deterministic floor
was measured to vary **15.6×** between epochs — enough to flip a verdict depending on which slice
was used. ### What "post-recalibration" means here — NO recalibration was performed

Three distinct things, easy to conflate:

1. **The scored probabilities are raw and uncalibrated.** `full_roa_per_point.npz` stores
   `p_success` as the ensemble-averaged model output; `lambda_star` and `delta` sit beside it as
   separate fields. Nothing transforms `p`.
2. **The pipeline calibrates the DECISION RULE, not the probabilities.** `conformal_threshold` /
   `conformal_calibration` fit λ* and δ each epoch — the bands turning a probability into
   success / failure / uncertain. That is conformal coverage on decisions; it cannot repair a
   miscalibrated `p`.
3. **`post-recal` in this document is `UNC − RES` from the Murphy decomposition**
   (`Brier = REL − RES + UNC`) — what Brier *would* be if reliability error were driven to zero.
   It is an idealisation computed from the data. **No Platt scaler, isotonic fit or temperature
   was ever fitted.**

So "the epistemic split removes the harm once calibration is accounted for" means the residual
damage lives entirely in the component a *perfect* recalibrator would remove — not that
recalibration was applied and observed to help. A real recalibrator recovers only part of that
floor, so the achievable benefit lies between the raw and post-recal columns. This is why the
headline is stated as a **~10× harm reduction on raw metrics**, which assumes nothing about
recalibration.

---

## 1. Established — ALL CLASSIFIER (CLF)

> **Every result in this section is the ensemble CLASSIFIER (`predictor=clf_ensemble`).**
> Do not read any number below as a statement about flow matching — §4 shows [FM] behaves
> *differently* at the same noise levels, most sharply at xhigh where [CLF] breaks (+29× to +45×)
> and [FM] shows nothing. And do not read these as statements about the estimators in general:
> §3 shows this classifier is badly miscalibrated at high noise, so every score here is computed
> on distorted p̄.

### [CLF + FM] The decomposition is mechanically correct
`H(p̄) = E_m[H(p_m)] + I(y;m|x)` holds to **exact float precision** on real model outputs (the
`total` arm computes the left side, `epi_bald`/`aleat` the two right-hand terms; residual 0.0e+00).
Epoch-0 identity confirmed on both predictors: before the first acquisition all five arms are
identical to within 6e-05.

### [CLF] Separation is a monotone function of noise, with a threshold
Arms only diverge once aleatoric mass in the candidate pool exceeds roughly 0.1:

| level | aleatoric pool mass | do arms separate? |
|---|---|---|
| det, low, med | 0.005–0.034 | no |
| high | 0.17–0.21 | yes |
| xhigh | 0.34–0.41 | yes, most |

### [CLF] Deterministic pendulum (complete, 19/19 epochs) — MARGINAL, metric-dependent
With the complete 18-epoch floor, "adaptive beats random" holds on **Brier only** (−1.3× to −1.8×)
and **fails on log score** (−0.3× to −0.4×, within noise). auc/auprc are saturated (0.9999) and
uninformative. Two proper scoring rules on identical predictions disagree, because log score's
seed floor is ~15× larger (0.0236 vs 0.0016).

Claim as: *a small Brier improvement that does not replicate on log score.*

**Unaffected:** the four adaptive arms are indistinguishable from each other on every metric —
which is the design's actual validation prediction for this level (aleatoric ≈ 0, so the score
choice should not matter). With aleatoric ≈ 0 there is nothing to separate, so the score choice
does not matter while adaptive-vs-random does. This is the design's validation case, passed.

### [CLF] low: small but consistent help; med: consistent direction, marginal size
Full-trajectory check (every epoch, not a two-epoch window): **low** runs −1.1 to −3.3× from
epoch 2 onward with the sign stable for 3 of 4 arms — a small but consistent improvement, not a
marginal one. **med** hovers at −0.5 to −1.8× across nine epochs: direction never flips, magnitude
sits on the threshold, so it is a weak effect rather than a null. Neither level shows a score
effect — the arms track each other within ~0.4×.

*Superseded: med was previously reported as "adaptive helps (−1.5 to −3.1×)" against a
single-epoch floor. The pooled floor is 3.4× larger and the effect does not survive it.*

### [CLF] high noise: `total` is harmful; the other three do not survive full depth

**Rescored 2026-08-08 with all five arms complete at 19/19 epochs.** This supersedes every
earlier reading of this level, all of which came from truncated 12-15 epoch curves against
2-3 epoch floors.

```
clf high: pooled 2*SD = 0.00051 over 10 epochs
   epi_bald   ep17: +0.00013 (+0.3x)  ep18: +0.00107 (+2.1x)   -> not stable
   epi_var    ep17: -0.00014 (-0.3x)  ep18: +0.00180 (+3.5x)   -> not stable
   total      ep17: +0.00968 (+19.1x) ep18: +0.00705 (+13.9x)  -> DISTINGUISHABLE (sign stable 18/18 ep)
   aleat      ep17: +0.00138 (+2.7x)  ep18: +0.00268 (+5.3x)   -> DISTINGUISHABLE [!] sign flips
```

**Only `total` holds.** +19.1x and +13.9x the floor with the sign stable across **all 18 epochs** —
the strongest and best-supported single result in the campaign. Acquiring on total predictive
entropy actively damages a classifier at high noise, consistently from first epoch to last.

**`epi_bald` and `epi_var` are NOT stable** at full depth: both flip sign between the final two
epochs. **`aleat` clears the floor in the final window but is flagged for sign flips across the
trajectory**, so it is not reportable either.

Two things produced the change. The floor tightened to **0.00051** pooled over 10 shared epochs
(the truncated verdict used 2-3 epoch floors, i.e. a far noisier estimate of the same quantity),
and the full-trajectory sign check finally had 18 epochs to work with rather than a handful.

*Withdrawn: "three of four adaptive arms are worse than random." At 19 epochs only `total`
qualifies. Also withdrawn: every claim about arm ORDERING at this level — the least-harmful slot
rotated among all four arms across the run.*

**This is FINDINGS section 5 vindicated, at a second level.** On the deterministic curve `aleat`
went from +10.7x floor *worse* at epoch 4 to *best* by epoch 18; here `epi_bald`/`epi_var` look
harmful at shallow depth and dissolve at full depth. The two-consecutive-epoch rule is **not**
sufficient protection on its own — the quantity oscillates on that timescale, so two adjacent
samples can agree by luck even across independent metrics. Only the full-trajectory sign check
plus a deeply-pooled floor separates signal from oscillation.

### [CLF] xhigh: total entropy does real damage; the epistemic split cuts it ~10x
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

**Verified on all seven available metrics** (post-recal, sAUROC, brier_debiased, log_score,
skill_score, AURC, KL), each with its own 7-epoch pooled floor:

- `total` and `aleat` are distinguishably harmful on **every one**. On terminal data with a
  9-epoch pooled floor: `aleat` +44.8×/+31.7× and `total` +29.1×/+26.9× post-recal, with the
  matching sAUROC figures −51.8×/−32.8× and −27.1×/−23.1×, sign-stable 9/9 epochs.
- `epi_bald` cuts that by roughly **10×** on the raw metrics and **ties random sampling** on the
  calibration-free ones (0.7–1.5×, straddling the threshold). *An earlier claim that it slightly
  beats random did not survive the terminal data.*
- `epi_var` is clearly worse than `epi_bald` here — 9–18× on every raw metric. Both estimators
  are exact on the classifier (members enumerated, K=None), so this is a genuine ranking
  difference between the two epistemic scores, favouring the naive one.

Stated without requiring recalibration as the lens: **acquiring on the epistemic component
reduces the harm ~10×, and removes it entirely once calibration is accounted for.**
Parity is the expected ceiling, not a shortfall: the epistemic split restores a balanced,
representative sample, which is what random already provides. The value is in avoiding total
entropy's skew.

*(Being re-derived: `clf_xhigh_total` was preempted and relaunched clean, so no epoch is currently
shared by all five arms.)*

---

## 2. Mechanism (measured, not inferred)

Each epoch records the pool indices it acquired, so the **true** success probability of every
purchased point is recoverable.

### [CLF] Harm tracks how far an arm drags the training marginal — across levels, not within them
Over 112 arm-epochs spanning all four stochastic levels: corr(fraction of ambiguous points
acquired, harm) = **−0.429**, corr(marginal skew |mean_p − 0.5|, harm) = **+0.406**. The effect is
strong at xhigh (r = −0.748) and **absent within high** (r = +0.259, wrong sign), where all arms
are compressed into mean_p 0.008–0.079 — too narrow a band to discriminate. So this explains the
dose-response *between* noise levels, and explains nothing about differences *between arms* at
high.

### [CLF] Why total entropy fails at xhigh
| arm | fraction of acquired points genuinely ambiguous |
|---|---|
| `epi_bald` | **51–68%** |
| `epi_var` | 26–31% |
| `total` / `aleat` | **2–4%** |

Total entropy harvests near-certain failures and crushes the training marginal to p ≈ 0.12; the
epistemic score recovers ambiguous states by ~20× and restores a near-balanced marginal. This is
exactly the pathology the previous campaign diagnosed, and the decomposition repairs it.

### [predictor-independent] BALD is not "epistemic focus" — it is epistemic focus divided by p(1−p)
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

## 4. Flow matching — [FM] high is the campaign's best-powered null

### [FM] high noise: no arm is distinguishable from the non-adaptive control (6 passes, floor halved)

Post-restart runs (EarlyStopping `patience=30`), pooled floor from `dir00` / `dir00_s43` /
`dir00_s44`. **Scored six times between 2026-08-06 and 2026-08-09 as the floor deepened.** The
final pass is the one to quote:

```
FINAL (13 shared floor epochs; dir00 and s43 both complete at 19/19):
fm high: pooled 2*SD = 0.00102 over 13 epochs
   epi_bald   ep15: -0.00055 (-0.5x)  ep16: -0.00045 (-0.4x)   -> within noise (null)
   epi_var    ep15: -0.00059 (-0.6x)  ep16: -0.00056 (-0.6x)   -> within noise (null)
   total      ep15: -0.00053 (-0.5x)  ep16: -0.00052 (-0.5x)   -> within noise (null)
   aleat      ep15: -0.00060 (-0.6x)  ep16: -0.00049 (-0.5x)   -> within noise (null)
```

**Every pass, every arm, null.** The floor halved as data accumulated and the verdict never moved:

| pass | floor (2*SD) | shared epochs | window | result |
|---|---|---|---|---|
| 1 | 0.00198 | 2 | ep 4-5 | 4 nulls |
| 2 | 0.00163 | 3 | ep 5-6 | 4 nulls |
| 3 | 0.00179 | 4 | ep 6-7 | 4 nulls |
| 4 | 0.00161 | 5 | ep 7-8 | 4 nulls |
| 5 | 0.00111 | 11 | ep 14-15 | 4 nulls |
| **6** | **0.00102** | **13** | **ep 15-16** | **4 nulls** |

This is the campaign's best-powered null. Six independent windows walking from epoch 4 to epoch
16, against a floor that tightened by ~50%, and the four arms stay clustered between −0.00045 and
−0.00060 throughout — indistinguishable from each other and from the control at every depth.

**The four arms converge on the same number.** At the final pass they span −0.4x to −0.6x; earlier
(pass 5) they agreed to four decimal places. `epi_bald` (the arm the campaign exists to test),
`epi_var`, `total`, and `aleat` (the deliberately-useless negative control) are not merely all
null — they are numerically interchangeable. When the strategy designed *not* to work matches the
strategy designed to work, the ranking carries no information about acquisition quality.

**The uniform small negative offset is a property of the control, not an effect.** All four arms
sit slightly below `dir00` rather than scattered around zero. `dir00` is the one run doing
something structurally different (`d2_ratio=0`, no acquisition at all), so a shared offset against
it is what the pipeline produces, not what the scores produce.

**Power is not the limitation.** At a comparable floor (0.00051 over 10 epochs) and comparable
depth, [CLF] high resolved `total` at **+19.1x**, sign-stable 18/18. The instrument detects large
effects at this scale. It finds none here.

**Cross-cluster caveat (unchanged).** `dir00` and `s43` ran on iLab, `s44` on arrakis, so
cross-cluster float nondeterminism is folded into this floor. That *inflates* it, which is
conservative for claiming significance — but it means the FM floor is not directly comparable in
magnitude to the classifier floors, which were measured within-cluster. [FM] low will be the first
fully within-cluster FM floor.

### [FM] xhigh: also null — and this is the level where [CLF] broke

Scored 2026-08-07, shared epochs 7-8, floor from `dir00` / `dir00_s43` / `dir00_s44`:

```
fm xhigh: pooled 2*SD = 0.00139 over 2 epochs
   epi_bald   ep7: -0.00002 (-0.0x)  ep8: -0.00048 (-0.3x)   -> within noise (null)
   epi_var    ep7: -0.00014 (-0.1x)  ep8: -0.00073 (-0.5x)   -> within noise (null)
   total      ep7: -0.00022 (-0.2x)  ep8: -0.00069 (-0.5x)   -> within noise (null)
   aleat      ep7: -0.00033 (-0.2x)  ep8: -0.00066 (-0.5x)   -> within noise (null)
```

**This is a predictor dissociation, and it is the campaign's most informative comparison.**
xhigh is the ONLY level where anything ever cleared a floor by a wide margin: on [CLF], `total`
and `aleat` were **+29x and +45x** the floor on post-recal with sAUROC collapsing **-27x to -52x**
(section 1). On [FM] at the same noise level, with the same acquisition code and the same
decomposition, all four arms sit between 0.0x and 0.5x and are ordered indistinguishably from each
other and from the negative control.

**The harm does not reproduce on flow matching.** That is consistent with section 3's mechanism:
the classifier damage was diagnosed as *miscalibration* — arms dragging the training marginal away
from the evaluation distribution. A predictor that stays calibrated under heavy process noise has
no such failure mode for acquisition to exploit, so acquisition neither helps nor hurts it. The
[CLF] xhigh result should therefore be read as a fact about miscalibrated classifiers, NOT as a
general fact about entropy acquisition.

**Scope.** Two shared epochs (7-8), the minimum the rule permits; floor seeds are 3-5 epochs deep
against arms at 9-10. Widening the window is pending. Unlike [FM] high (four consistent passes),
this verdict has had one pass.

## 4c. Still not established

- **[FM] med, low, det.** Not launched — no FM runs exist at those levels.
- **`epi_var` vs `epi_bald`.** On the classifier both are exact (members enumerated, K = None), so
  the comparison there is uninformative about the estimator question. Only FM at K = 20 tests it —
  and at high noise both are equally null, so the estimator question remains open on the levels
  where an effect might exist at all.

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
