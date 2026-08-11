# Ensemble Epistemic Acquisition — current findings

**Status as of 2026-08-11.** This file holds only what is *currently defensible*; `LOG.md` is the
chronological record including superseded claims and their corrections.

| half | state |
|---|---|
| **[CLF]** | det complete 19/19; **high complete 19/19 on all five arms** (rescored, §1); low/med/xhigh from truncated curves (walltime, see §6) |
| **[FM]** | **all five levels now scored.** high null (7 passes, floor −52%) · xhigh null (5 passes, floor −45%, verdict stable) · med **null** (3 passes, floor −28%, verdict stable) · low **no verdict** (5 passes; the set of "distinguishable" arms churns none→two→three→none) · det **the metric decides the winner** — quote nothing from it |

**Headline — the constant across every FM level is that the SCORE does not matter.** The [FM]
sweep is complete: five noise levels, and at every one the four arms are numerically
interchangeable. Nothing survives at any level: null at high/xhigh, no verdict at med or low,
field-dependent at det. Where a persistent offset does appear (low, high), all four arms sit on
the same side of the control and track each other to within 5e-05 — a property of `dir00`
(`d2_ratio=0`, structurally unlike every arm), not of the scores.

**The negative control is never separable from the real arms, and twice it ranks first.** At
[FM] det on AUC, `aleat` is the only arm clearing the floor with a stable sign while `epi_var`
carries the wrong sign (§4d). At [FM] low it ranked first at pass 4 — though pass 5 then
withdrew every arm's verdict, which is itself the point: `aleat` acquires on the component that
by construction *cannot* be reduced by more data, and it is interleaved with the designed
estimators at every level and every pass. Whatever adaptive sampling does to flow matching,
decomposing entropy into aleatoric and epistemic parts does not change it.

**A caution this campaign earned the hard way.** [FM] low was scored five times as its floor
deepened, and the set of arms clearing the threshold went none → two → two → three → none while
the effect sizes never left −0.9× to −1.6×. Pass-to-pass promotions are not a result firming up;
they are a threshold being crossed by noise. Quote effect sizes and floor depth, never a bare
"DISTINGUISHABLE".

*Correction history for [FM] low, kept because the churn is itself the finding: the header read
"null at every level including low" (5-epoch floor) → "epi_var and total cross the threshold"
(6-epoch) → "the negative control ranks first" (7-epoch) → back to no verdict (9-epoch, current).
Five passes, four different headline claims, and the effect sizes never moved. See §4c.*

*Corrected 2026-08-10: the header previously read "**det harmful** (+2.3x to +26.5x, shallow
floor)". At a 9-epoch floor the det arms are mostly *helpful* and only one survives the stability
flag. The level is unsettled, not harmful. See §4d.*

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

**Rescored 2026-08-10 at greater depth; both levels firmed up, and `aleat` remains inseparable.**
med now clears its floor on **all four** arms with the sign stable 10/10 epochs (−1.2× to −1.9×
at ep9-10, floor 0.00051 over 13 epochs), promoting it from "weak effect" to distinguishable —
but `aleat` at −1.7× sits between `epi_bald` (−1.4×) and `epi_var` (−1.8×), so the promotion buys
no information about which score to use. At low (floor 0.00025 over 10 epochs) `aleat` is the
**largest** helpful arm at −3.0×, ahead of `epi_var` (−2.6×) and `epi_bald` (−2.4×).

`total` at low is the one arm that breaks the pattern, and it breaks it catastrophically:
−0.00070 (−2.8×) at ep9 becomes **+0.22484 (+896×)** at ep10 — a three-order-of-magnitude blowup
in one epoch, flagged "not stable". This is the same total-entropy failure that dominates [CLF]
high and xhigh (§1), arriving abruptly rather than gradually.

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

## 1b. One number for the whole campaign: does the score matter more than the seed?

Every pairwise verdict above asks "does arm X beat the control by more than 2×SD of three seeds?"
That framing hides a simpler question: **is the spread among the four acquisition arms even as
large as the spread among three seeds of one config?** If not, choosing an acquisition score
matters less than re-running with a different random seed.

Computed on `recal` (UNC−RES), per epoch, as `SD(epi_bald, epi_var, total, aleat) / SD(dir00,
dir00_s43, dir00_s44)`, then median over all shared epochs with epoch 0 excluded:

| predictor | low | med | high | xhigh |
|---|---|---|---|---|
| **[FM]** | 0.72 | 1.85 | 0.68 | 1.31 |
| **[CLF]** | 0.63 | 0.51 | **6.89** | **84.61** |

**Acquisition genuinely changes the outcome in exactly two cells of eight** — the classifier at
high and xhigh, at 6.9× and 84.6× the seed spread. Those are the same two cells where §1 finds
`total` and `aleat` harmful and §3 diagnoses miscalibration as the cause. **Everywhere else — all
five [FM] levels and [CLF] at low/med — the ratio sits between 0.5 and 1.9**, i.e. swapping the
acquisition score perturbs the result about as much as, or less than, changing the seed.

This is the campaign's headline in one statistic, and it is stronger than the pairwise verdicts
because it needs no threshold convention: it compares two measured spreads directly.

**Two caveats on individual cells.** `[FM] med`'s 1.85 rests on the campaign's shallowest floor
(3–4 shared epochs, §4d) and should not be read as evidence that acquisition matters there. And
`[CLF] low`'s per-epoch ratios contain a single 2700× outlier at ep10, from `total`'s +896×
blowup (§1); the median is used precisely because it is robust to that, but the mean would be
meaningless.

### The ratio is not an artefact of the metric

Recomputed on five metrics spanning rank-based, calibration-charging, and decomposition-based
families (2026-08-11):

| metric | fm/low | fm/med | fm/high | fm/xhigh | clf/low | clf/med | clf/high | clf/xhigh |
|---|---|---|---|---|---|---|---|---|
| recal (UNC−RES) | 0.72 | 1.85 | 0.68 | 1.31 | 0.63 | 0.51 | **6.89** | **84.61** |
| sAUROC | 0.45 | 0.96 | 1.08 | 2.31 | 0.78 | 0.46 | **8.41** | **61.11** |
| brier_debiased | 0.39 | 0.81 | 0.47 | 0.74 | 1.35 | 2.97 | **21.43** | **35.43** |
| log_score | 0.36 | 1.14 | 0.66 | 1.69 | 1.42 | 2.39 | **18.45** | **43.92** |
| skill_score | 0.39 | 0.81 | 0.47 | 0.74 | 1.35 | 2.97 | **21.43** | **35.43** |

**The two conclusions that matter are metric-independent.** Every one of the 20 [FM] cells lands
between 0.36 and 2.31 — no metric makes the acquisition score matter for flow matching. And
[CLF] high/xhigh are 6.9×–85× on *every* metric — no metric makes the classifier damage go away.
`recal`, the metric the verdicts use, is neither the most nor the least favourable choice.

**Where it IS metric-dependent: [CLF] low/med.** Those two cells read 0.46–0.78 on the rank-based
and decomposition metrics but 1.35–2.97 on Brier, log score and skill score. Brier and log score
charge for calibration, which is exactly the axis §3 identifies as the classifier's weak point,
so the arms separate more there. The honest statement for [CLF] low/med is therefore "comparable
to seed noise on rank-based metrics, up to ~3× it on calibration-charging ones" — not the flat
"below 1" that `recal` alone suggests.

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
test — and they have now run it. §4 reports [FM] high null across seven passes on a floor pooled
over 15 epochs, and §4b reports [FM] xhigh null where this classifier breaks worst. The
miscalibration hypothesis in this section is therefore not just an explanation offered after the
fact: it predicted that a calibrated predictor would show no such harm, and that prediction held.

---

## 4. Flow matching — three stable nulls, one churning level, one metric-dependent level

Every FM level has been rescored repeatedly as its floor deepened. **Whether a verdict held while
the floor tightened is the single most useful thing to know about it**, so it is tabulated first:

| level | passes | floor tightened | verdict across passes | trust |
|---|---|---|---|---|
| high (§4a) | 7 | −52% (0.00198 → 0.00095) | null throughout | highest |
| xhigh (§4b) | 5 | −45% (0.00139 → 0.00076) | null throughout | high |
| med (§4d) | 3 | −28% (0.00100 → 0.00072) | null throughout | good |
| **low (§4c)** | 5 | −16% (0.00061 → 0.00051) | **churned** none→2→2→3→none | **low** |
| det (§4d) | 3 | n/a (label metric) | ordering depends on the metric field | none |

**Read the campaign's conclusion off the top three rows, not the bottom two.** high, xhigh and med
each returned the same null as their floors tightened by 28–52%, across 15 scoring passes in
total. That is the trustworthy pattern: more data, same answer.

low and det are the levels where a verdict was available at some point and then stopped being
available. At low the effect sizes never left −0.9× to −1.6× while the set of arms clearing the
threshold changed at nearly every pass — a threshold recrossed by noise, not an effect resolving.
At det, three different label fields give three different arm orderings at the same depth.
**Neither supports a claim in either direction**, and in particular neither supports "adaptive
helps at low", which this document asserted for three passes before pass 5 withdrew it.

### 4a. [FM] high noise: no arm is distinguishable from the non-adaptive control (7 passes, floor halved)

Post-restart runs (EarlyStopping `patience=30`), pooled floor from `dir00` / `dir00_s43` /
`dir00_s44`. **Scored seven times between 2026-08-06 and 2026-08-10 as the floor deepened.** The
final pass is the one to quote:

```
FINAL (15 shared floor epochs; dir00 and s43 both complete at 19/19):
fm high: pooled 2*SD = 0.00095 over 15 epochs
   epi_bald   ep15: -0.00055 (-0.6x)  ep16: -0.00045 (-0.5x)   -> within noise (null)
   epi_var    ep15: -0.00059 (-0.6x)  ep16: -0.00056 (-0.6x)   -> within noise (null)
   total      ep15: -0.00053 (-0.6x)  ep16: -0.00052 (-0.5x)   -> within noise (null)
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
| 6 | 0.00102 | 13 | ep 15-16 | 4 nulls |
| **7** | **0.00095** | **15** | **ep 15-16** | **4 nulls** |

This is the campaign's best-powered null. Seven independent windows walking from epoch 4 to epoch
16, against a floor that tightened by ~52%, and the four arms stay clustered between −0.00045 and
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

### 4b. [FM] xhigh: also null — and this is the level where [CLF] broke

Floor from `dir00` / `dir00_s43` / `dir00_s44`. **Scored five times; the floor deepened from 2
shared epochs to 11 and the window walked from ep7-8 to ep11-12 without moving the verdict:**

```
PASS 5 (11 shared floor epochs, 2026-08-11) -- current:
fm xhigh: pooled 2*SD = 0.00076 over 11 epochs
   epi_bald   ep11: +0.00011 (+0.1x)  ep12: -0.00048 (-0.6x)   -> within noise (null)
   epi_var    ep11: -0.00013 (-0.2x)  ep12: -0.00039 (-0.5x)   -> within noise (null)
   total      ep11: +0.00103 (+1.4x)  ep12: -0.00050 (-0.7x)   -> not stable
   aleat      ep11: -0.00007 (-0.1x)  ep12: -0.00040 (-0.5x)   -> within noise (null)

PASS 4 (8 shared epochs, floor 0.00085, same ep11-12 window): identical verdict.
PASS 1 (2 shared epochs, floor 0.00139, window ep7-8): four nulls.
```

`total` is the only arm that ever leaves the band, and it does so in *opposite directions* on
adjacent epochs (+1.4x then −0.7x) — oscillation, not an effect.

**Like med (§4d), and unlike low (§4c), this verdict survives a deeper floor.** Passes 4 and 5
score the same window against floors differing by 11% and return the same three-null-plus-one-
oscillating result. `dir00` is complete at 19/19, so the floor is anchored by a full-depth seed.

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

**Scope.** Now 8 shared epochs with the window at ep11-12 — no longer the thin two-epoch read it
was on 2026-08-07. Four consistent passes against a floor that tightened from 0.00139 to 0.00085
(−39%). This is the campaign's second-best-powered null after [FM] high.

### 4c. [FM] low: no verdict — five passes, and the arms never separate from the floor

**This is the level the FM extension was launched to test.** [CLF] low is the only place in the
whole campaign where adaptive sampling genuinely beat uniform (−1.1× to −3.3×, sign stable for 3
of 4 arms, §1). The question was whether flow matching shows the same benefit there.

Scored five times as the floor deepened. **Pass 5 withdraws the verdict: at the deepest floor,
no arm clears the test.**

```
PASS 5 (9 shared floor epochs, window ep12-13, 2026-08-11) -- current:
fm low: pooled 2*SD = 0.00051 over 9 epochs
   epi_bald   ep12: -0.00062 (-1.2x)  ep13: -0.00044 (-0.9x)   -> not stable
   epi_var    ep12: -0.00058 (-1.1x)  ep13: -0.00046 (-0.9x)   -> not stable
   total      ep12: -0.00061 (-1.2x)  ep13: -0.00049 (-1.0x)   -> not stable
   aleat      ep12: -0.00063 (-1.2x)  ep13: -0.00045 (-0.9x)   -> not stable

PASS 4 (7 shared epochs, window ep11-12): epi_var, total AND aleat distinguishable; aleat largest
        at -1.6x; epi_bald the only arm failing stability.
PASS 3 (6 shared epochs, window ep10-11): epi_var + total distinguishable; aleat "not stable".
PASS 2 (window ep9-10): same two arms distinguishable, at -1.0x.
PASS 1 (5 shared floor epochs, window ep8-9): all four "not stable" or null.
```

**Five passes, and the set of "distinguishable" arms went none → two → two → three → none.**
Across all five the effect sizes never left −0.9× to −1.6×. The floor tightened monotonically
(0.00061 → 0.00057 → 0.00051) and the arms did *not* separate from it; they moved with it. This
is the diagnosis written up at pass 4, now confirmed by the pass that followed: **what changes
between passes is which arms happen to sit on the far side of a threshold, not what the arms do.**

*Superseded by pass 5: pass 4 was written up as "the negative control now passes the test
outright and ranks first". That statement was true of the 7-epoch floor and is false at 9. The
`aleat`-ranks-first observation survives only at [FM] det on AUC (§4d), where it is measured
independently.*

**What can be claimed at pass 5:** nothing about adaptive-vs-uniform at [FM] low. All four arms
sit persistently *below* the control by −0.9× to −1.2× — a consistent negative offset — but none
of it survives the two-consecutive-epoch rule against the current floor.

**The offset is almost certainly a property of the control, not an effect.** All four arms sit on
the same side of `dir00` rather than scattered around zero, and they track each other to within
5e-05 at ep12–13 (0.00058 to 0.00063). `dir00` is the one run doing something structurally
different — `d2_ratio=0`, no acquisition at all — so a shared offset against it is what the
pipeline produces, not what the scores produce. This is the identical argument §4a makes for
[FM] high, where the same uniform small negative offset appears and is also null.

**No ranking among the four arms carries information about acquisition quality.** `total` matches
`epi_var` to within 3e-05, so the epistemic split contributes nothing over plain total entropy —
the campaign's actual question — and the deliberately-useless `aleat` is interleaved with the
real arms at every pass, sometimes largest, sometimes smallest, never separable.

**[CLF] low remains the contrast.** There the effect is −1.1× to −3.3× and sign-stable, up to 3×
larger than anything seen here. Whatever benefit adaptive sampling delivers to the classifier at
low noise does not reproduce on flow matching.

**First fully within-cluster FM floor.** All three seeds ran on Amarel `gpu-redhat`, so unlike
[FM] high (iLab + arrakis) and xhigh this floor is not inflated by cross-cluster nondeterminism —
a *tighter* test than the FM floors above it, not a looser one.

**Scope.** 7 shared epochs, arms at 13–17, `s43` at 8 and still climbing. The verdict has moved
at every pass that added floor depth (pass 1 null → pass 2/3 two arms → pass 4 three arms
including the control), which is itself the warning: what changes between passes is *which arms
clear a threshold*, never the effect sizes, which have sat between −0.9× and −1.6× throughout.
That is the signature of a threshold being crossed by noise, not of an effect being resolved.

*Superseded: pass 1 was written up as "[CLF]'s one genuine benefit does NOT reproduce." With one
more shared floor epoch, two arms cross the threshold. The honest statement is a marginal effect
at the floor, not a null.*

## 4d. Weakly established or still open

### [FM] med: null, and the verdict survives a near-doubling of floor depth

Scored three times, with the floor depth nearly doubling. **The verdict has not moved: nothing
clears, at any depth.**

```
PASS 3 (7 shared floor epochs, window ep7-8, 2026-08-11) -- current:
fm med: pooled 2*SD = 0.00072 over 7 epochs
   epi_bald   ep7: -0.00051 (-0.7x)  ep8: -0.00070 (-1.0x)   -> within noise (null)
   epi_var    ep7: -0.00041 (-0.6x)  ep8: -0.00066 (-0.9x)   -> within noise (null)
   total      ep7: -0.00059 (-0.8x)  ep8: -0.00067 (-0.9x)   -> within noise (null)
   aleat      ep7: -0.00058 (-0.8x)  ep8: -0.00047 (-0.7x)   -> within noise (null)
```

| pass | floor | shared epochs | window | effect sizes | result |
|---|---|---|---|---|---|
| 1 | 0.00100 | 3 | ep4-5 | −0.2× to −0.7× | 4 null |
| 2 | 0.00089 | 4 | ep5-6 | −0.5× to −1.3× | 4 not stable |
| **3** | **0.00072** | **7** | **ep7-8** | **−0.6× to −1.0×** | **4 null** |

**This is the opposite of what happened at [FM] low, and the contrast is informative.** Both
levels show the same uniform small negative offset against the control, and at both the four arms
track each other tightly (med spans −0.00047 to −0.00070 at ep8, a 2.3e-04 range). But at low the
*verdict* churned none → two → two → three → none across five passes, while med has returned the
same "nothing clears" at 3, 4 and 7 shared epochs, with the floor tightening 28% along the way.
A verdict that survives a near-doubling of floor depth is worth more than one that flips with it.

**Scope.** 7 shared floor epochs, arms at 9–13, `dir00_s43` and `dir00_s44` at 8 are the binding
constraint. Per §6 med runs at 9.0–10.3 h/epoch against a 72h walltime cap, so it needs three
sequential jobs to finish and its floor will keep advancing slowly.

### [FM] det: the metric decides which arm wins — do not quote any ordering

Scored three times. Six of the seven det runs are complete at 19/19; only `dir00` lags (11/19),
which caps the floor at 10 shared epochs. Rescored 2026-08-11 on **all three available label
fields at the same depth**, and they do not agree with each other:

```
fm det, 10 shared floor epochs, window ep9-10 (2026-08-11)

  brier      floor 0.02567     auc        floor 0.00709     log_score  floor 0.06381
  epi_bald   -2.5x/-1.7x  [!]  epi_bald   -1.5x/-0.0x  ns   epi_bald   -2.3x/-1.7x  [!]
  epi_var    -3.3x/-2.2x  [!]  epi_var    +0.5x/+0.7x null  epi_var    -3.6x/-2.4x  [!]
  total      -3.0x/-1.8x  [!]  total      -0.8x/-0.1x null  total      -3.0x/-1.6x  [!]
  aleat      -1.7x/-0.5x  ns   aleat      -3.7x/-3.4x  OK   aleat      -0.9x/+0.5x null

  [!] = DISTINGUISHABLE but sign flips across the 10 epochs;  ns = not stable;
  OK = DISTINGUISHABLE, sign stable 10/10
```

**The ordering inverts with the metric.** On Brier and log score the three "real" arms lead and
`aleat` trails. On AUC the picture reverses: `aleat` is the *only* arm that clears the floor with
a stable sign (−3.4×), while `epi_var` carries the **wrong sign** (+0.7×, i.e. worse than the
control). AUC is rank-based and monotone-invariant; Brier and log score additionally charge for
calibration. So the arms differ from each other in *calibration*, while on pure ranking quality
the useless control comes out ahead.

**Do not quote [FM] det — not the sign, not the ordering, not the magnitude.** Three passes have
now given three different stories: 2026-08-09 read "all four DISTINGUISHABLE and harmful, sign
stable 4/4" at +2.3× to +26.5×; 2026-08-10 read mostly *helpful* with one survivor; 2026-08-11
gives an ordering that depends on which label field is chosen. Every arm that clears the floor on
Brier or log score also fails the sign-stability check on the same data.

Two structural reasons this level is hard, both worth keeping: the floor (0.02567) is **~27×
larger** than any stochastic-level floor because det is scored on the label metric rather than
post-recal, so effect sizes here are not comparable to the four stochastic levels; and `dir00` at
11/19 against arms at 19/19 means the comparison window cannot reach the depth where the other
levels settled.

This is §5's "single-epoch reads lie" caution generalised: on this level, *metric* reads lie too.

### Still open

- **[FM] det** at a settled floor — blocked on `dir00`, resumed and at 11/19 while five of the
  seven runs at that level are complete.
- **[FM] med** at a floor deeper than 3 epochs — blocked on `dir00_s43` at 2.
- **`epi_var` vs `epi_bald`.** On the classifier both are exact (members enumerated, K = None), so
  the comparison there is uninformative about the estimator question. Only FM at K = 20 tests it —
  and FM is now null at low, high and xhigh, with the two estimators indistinguishable at every
  one. The estimator question is therefore unanswerable from this campaign: there is no effect
  anywhere for the estimators to differ *on*.

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

- **No Amarel FM run reaches 19 epochs in a single job — every one needs 2–3 sequential jobs.**

  **Exact rates** (2026-08-11), from the five `en_fm_*` jobs that ran a full 72h continuously
  from epoch 0 before TIMEOUT. Elapsed ÷ epochs completed, so no dead time is included:

  | run | epochs in 72h | h/epoch | 19 epochs needs | 72h jobs |
  |---|---|---|---|---|
  | `fm_low_epi_var` | 15 | 4.80 | 91 h | 2 |
  | `fm_low_total` | 14 | 5.14 | 98 h | 2 |
  | `fm_low_dir00_s44` | 14 | 5.14 | 98 h | 2 |
  | `fm_low_aleat` | 14 | 5.14 | 98 h | 2 |
  | `fm_med_epi_var` | 8 | 9.00 | 171 h | 3 |
  | `fm_med_dir00` | 7 | 10.29 | 196 h | 3 |
  | `fm_med_dir00_s44` | 7 | 10.29 | 196 h | 3 |

  Seven clean 72h windows now: **low 4.80–5.14 h/epoch across four runs, med 9.00–10.29 across
  three.** The two levels do not overlap, so the ~2× med/low gap is a property of the level, not
  of node assignment.

  det (~6.3) and xhigh (~9.9) are estimates only — every run at those levels was resumed, so no
  clean 72h window exists to measure them from.

  **Measurement caveat, learned by getting it wrong.** An earlier version of this table came from
  `(last artifact mtime − first) / (n−1)` across a whole run directory. That is wrong in two
  directions at once: it **excludes** the epoch-0 training time before the first artifact
  (understating clean runs — 4.88 vs the true 5.14 for `fm_low_total`), and it **includes** the
  TIMEOUT→queue→resume dead time for any run that was resumed (overstating those — it reported a
  12.5h "epoch" for `fm_med_epi_var` where the job itself ran 4.9h). Per-epoch cost is only
  measurable inside a single continuous job.

  Per-epoch time also varies widely *between runs at the same level*: med spans 9.0–10.3 h/epoch
  on clean runs, and recent per-epoch gaps across med runs range from 2.8h to 20h once node speed,
  contention and early-stopping variation are included. Any claim that a configuration change
  made runs faster needs many runs, not one.

  Amarel's walltime maximum is 72h, so a TIMEOUT at 3-00:00 is the *expected* end-state for these
  runs, not a failure — `fm_med_dir00` and `fm_med_epi_var` both hit it at 7/19 and 8/19 with
  memory at 60% and no preemption. Each resume additionally pays the queue wait, currently 8h+
  against 262 pending jobs on `gpu-redhat`.

  **Consequence for the two unsettled levels.** med and xhigh are the *slowest* levels (~10h/epoch)
  and are the two whose floors are shallowest. Reaching a 19-epoch med floor would take ~8 days of
  compute plus two queue waits. Plan for med and xhigh to stay floor-capped in the low teens, and
  do not treat their shallow floors as a temporary state that more waiting will fix.
