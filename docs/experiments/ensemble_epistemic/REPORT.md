# Ensemble-Based Epistemic Acquisition — what we set out to prove, what was built, what the data says

**Status: 2026-08-10, campaign substantially complete.** Companion documents: `FINDINGS.md`
(current defensible claims, maintained continuously), `LOG.md` (chronological record including
superseded claims), `runs.jsonl` (every launch with its arguments and rationale).

---

## 1. The scientific question

### 1.1 The prior result that motivated it

An earlier campaign established that **entropy-based acquisition actively harms a classifier as
process noise grows**, and measured the mechanism:

| level | frac ambiguous (random → scored) | mean true p (random → scored) |
|---|---|---|
| low | 0.027 → 0.296 | 0.388 → 0.361 |
| med | 0.053 → 0.328 | 0.389 → 0.179–0.367 |
| high | 0.133 → **0.010** | 0.406 → **0.046** |
| xhigh | 0.269 → **0.080** | 0.481 → **0.138** |

At low/med, entropy finds genuinely ambiguous states — what you want. At high/xhigh it scores
*worse than random* on ambiguity and instead harvests near-deterministic **failure** states,
dragging the training marginal from ~41–48% success down to ~5–14%. Cost: +0.157 debiased Brier
at high noise.

### 1.2 The hypothesis

Predictive entropy conflates two things that a Bayesian treatment separates:

```
H(Y|x,D)  =  E_θ~p(θ|D)[ H(Y|x,θ) ]  +  I(Y;Θ|x,D)
  total          aleatoric                epistemic
             (irreducible outcome        (reducible by
              randomness)                 more data)
```

**Claim under test:** entropy acquisition fails under heavy noise because it chases the
*aleatoric* term — irreducible coin-flips that no amount of data fixes. Acquiring on the
*epistemic* term alone should target what more data can actually repair, and should therefore
survive at noise levels where total entropy collapses.

**Falsifiable predictions:**

1. At high/xhigh noise, `total` is harmful and the epistemic arms are not.
2. The epistemic arms beat the non-adaptive control wherever adaptive sampling works at all.
3. `aleat` (acquiring on the aleatoric term alone) is the worst arm — it is the negative control,
   deliberately targeting the useless component.

### 1.3 Secondary question — which epistemic estimator

Two estimators of the same epistemic quantity:

- **`epi_bald`** = `H(p̄) − E_m[H(p_m)]` — the exact mutual information (BALD).
- **`epi_var`** = `Var_m[p] − mean_m[p(1−p)/(K−1)]` — between-member variance, debiased for
  finite-K Monte-Carlo noise.

The design justified `epi_var` by its freedom from finite-K bias. Whether that matters, and
whether the two ever diverge in practice, was an open question the campaign was built to answer.

---

## 2. Experimental design

**Arms** (5 per level, identical except the acquisition score):

| arm | score | role |
|---|---|---|
| `dir00` | none (`d2_ratio=0`, pure uniform sampling) | **non-adaptive control** |
| `total` | `H(p̄)` | the strategy known to fail |
| `epi_var` | debiased `Var_m[p]` | epistemic, curvature-free |
| `epi_bald` | `I(Y;Θ|x)` | epistemic, exact MI |
| `aleat` | `E_m[H(p_m)]` | **negative control** |

**Grid:** 5 noise levels (`det`, `low`, `med`, `high`, `xhigh`) × 2 predictors (ensemble
classifier, ensemble flow matching) × 5 arms, 19 adaptive epochs, `d2_ratio=1.0` for adaptive
arms, `seed=42`, `n_members=5`, `k_acq=20` MC samples per member for FM.

**Floor seeds:** each level additionally runs `dir00` at seeds 43 and 44. These three
config-identical runs define the run-to-run noise floor. Without them no verdict is possible —
this turned out to be the single most important design decision (see §5.1).

### 2.1 Standard of evidence

A gap counts only if it exceeds **2×SD of three genuinely-distinct seeds** at **two consecutive
epochs**, with the floor **pooled across epochs** (`2·sqrt(mean(var_e))`) rather than taken from
one. Epoch 0 is excluded — all seeds hold identical pre-acquisition data there, so its spread is
structurally ~0 and including it deflates the floor.

Pooling is not fussiness: **the deterministic floor was measured to vary 15.6× between epochs.**
A single-epoch floor flips verdicts depending on which slice you happen to use.

A **full-trajectory sign check** was added later after the two-consecutive-epoch rule proved
insufficient — see §7.

### 2.2 Primary metric

`recal = UNC − RES` from the Murphy decomposition (`Brier = REL − RES + UNC`) — what Brier
*would* be if reliability error were driven to zero. **No recalibrator was ever fitted**; this is
an idealisation computed from the data. Reported effects are the arm's value **minus the
control's**, expressed in units of the floor.

`det` has no rollout ground truth (a deterministic system's `p` is degenerate {0,1}), so it is
scored on the label metrics in `artifacts_v2.json` instead.

---

## 3. What was implemented

| component | file | what it does |
|---|---|---|
| Score modes | `adaptive_roa/adaptive_v2/strategy/uncertainty_scores.py` | the four acquisition scores; input validation that raises on NaN/out-of-domain rather than silently clipping |
| FM ensemble trainer | `adaptive_roa/adaptive_v2/trainers/ensemble_flow_matching_trainer.py` | trains M flow matchers concurrently via `mp.spawn`, assembles them into one handle |
| CLF ensemble | `adaptive_roa/adaptive_v2/trainers/bayesian_mlp_trainer.py` | deep-ensemble posterior, per-member seeding |
| Probability backends | `configs/adaptive_v2/probability/` | expose per-member predictions so the decomposition can be computed |
| Verdict machinery | `scripts/ensemble_verdicts.py` | pooled floor, two-consecutive-epoch rule, full-trajectory sign check, restart-in-place contamination detector |
| Rescoring | `scripts/stoch_prob_metrics.py` | Murphy decomposition from stored per-point probabilities |
| Experiment log | `scripts/exp_log.py`, `runs.jsonl` | every launch with arguments and rationale |

**Correctness check that passed:** the decomposition identity
`H(p̄) = E_m[H(p_m)] + I(y;m|x)` holds to **exact float precision** on real model outputs
(residual 0.0e+00), and at epoch 0 — before any acquisition differentiates them — all five arms
agree to within 6e-05 on both predictors.

### 3.1 Infrastructure written because the campaign needed it

- `scripts/resume_adaptive.py` (pre-existing, initially overlooked): continues an interrupted run
  from its last completed epoch. **Also prevents corruption** — it deletes the partial epoch
  rather than half-overwriting a completed curve.
- `contaminated_runs()`: detects restart-in-place stitching via epoch-mtime inversion. A requeued
  job restarts at epoch 0 and overwrites in place, producing a directory that is two independent
  runs spliced together — every file valid, the curve meaningless.
- `check_preempted.sh`: queries `sacct`, because a preempted Amarel job on the `general` account
  may vanish from `squeue` entirely with no error state.
- `load_det(..., predictor)`: was hardcoded to `clf_det_*`, so FM det runs were **invisible to the
  scorer** and failed silently — printing no row rather than erroring.

---

## 4. Results

### 4.1 Classifier

| level | result | strength |
|---|---|---|
| det | adaptive beats random on **Brier only** (−1.3× to −1.8×); **fails on log score** (−0.3× to −0.4×) | complete 19/19, metric-dependent |
| low | **adaptive helps**, −1.1× to −3.3×, sign stable 3 of 4 arms | the campaign's one clear positive |
| med | −0.5× to −1.8×, direction never flips, magnitude on the threshold | weak |
| high | **only `total` is harmful**: +19.1×/+13.9×, sign stable **18/18 epochs** | complete 19/19, strongest single result |
| xhigh | `total` and `aleat` catastrophic: **+29× to +45×** post-recal, sAUROC −27× to −52× | the only large effect anywhere |

At `high`, an earlier reading claimed three of four arms were harmful. **At full 19-epoch depth
that collapses to one** — `epi_bald` and `epi_var` flip sign between the final epochs and read
"not stable". Every claim about arm *ordering* at that level was withdrawn: the least-harmful slot
rotated among all four arms across the run.

### 4.2 Flow matching

| level | floor | result |
|---|---|---|
| **high** | 0.00102 over **13 epochs**, scored **6×** as the floor halved | **null** — all four arms −0.4× to −0.6×, indistinguishable from each other at every depth |
| **xhigh** | 0.00089 over 7 epochs, scored 3× | **null** — three arms inside the floor; `total` makes one +1.2× excursion and immediately reverses |
| **low** | 0.00061 over 6 epochs | **marginal** — `epi_var` and `total` reach −1.0× "distinguishable" (sign stable 10/10), `epi_bald` and `aleat` "not stable" |
| **det** | 0.01333 over **4 epochs only** | all four arms harmful, +2.3× to +26.5× — **but see caveat** |
| **med** | — | not yet scoreable (`s43` preempted twice, 1 shared epoch) |

`[FM] high` is the best-powered result in the campaign: six independent windows walking from
epoch 4 to epoch 16, floor tightening from 0.00198 to 0.00102, verdict unchanged every time.

---

## 5. What the results say

### 5.1 The negative control is the whole story

**At every flow-matching level, `aleat` — the arm designed to be useless — lands inside the spread
of the arms it should underperform.**

| level | `epi_bald` | `epi_var` | `total` | `aleat` (control) |
|---|---|---|---|---|
| high | −0.4× | −0.6× | −0.5× | **−0.5×** |
| xhigh | −0.4× | −0.5× | −0.6× | **−0.5×** |
| low | −1.2× | −1.0× | −1.0× | **−1.1×** |
| det | +6.7× | +2.3× | +26.5× | **+6.5×** |

At `[FM] high` the four arms agree to within 1.5e-04; at one pass they agreed to four decimal
places. **When the strategy built to fail matches the strategy the campaign exists to test, the
ranking carries no information about acquisition quality.** This is why the FM nulls are read as
"no effect" rather than "underpowered" — an underpowered test scatters arms randomly; this one
places them on top of each other.

The `low` result deserves the same scepticism despite two arms registering "distinguishable":
they clear the floor by ~0%, and `aleat` has a *larger* magnitude than `epi_var`, failing only on
the stability flag.

### 5.2 Against the three predictions

1. **"`total` harmful, epistemic arms spared" — supported on the classifier only.** At `[CLF]`
   high, `total` is the sole arm that survives full-depth scrutiny at +19.1×. At xhigh, `total`
   and `aleat` are catastrophic while `epi_bald`/`epi_var` cut the harm ~10×. That is the
   predicted pattern. **It does not reproduce on flow matching at any level.**
2. **"Epistemic beats the control where adaptive works" — refuted.** `[CLF]` low is the one place
   adaptive genuinely wins, and there the four arms track each other within ~0.4× — the epistemic
   split contributes nothing. `[FM]` low is marginal at best, with `total` matching `epi_var`
   exactly.
3. **"`aleat` is worst" — refuted.** It is mid-pack everywhere on FM, and at `[CLF]` high it was
   *least* harmful at shallow depth before the ordering dissolved entirely.

### 5.3 The predictor dissociation, and what explains it

The sharpest comparison in the campaign is `xhigh`, run with identical acquisition code:

- `[CLF]`: `total` and `aleat` at **+29× to +45×** the floor.
- `[FM]`: every arm inside **±0.6×**, on a floor 3× tighter than the original test.

The classifier damage was independently diagnosed as **miscalibration** — the arms drag the
training marginal away from the evaluation distribution, and the scores are then computed on
distorted `p̄`. That account made a prediction: a predictor that stays calibrated under heavy
noise has no such failure mode for acquisition to exploit, so it should show neither harm nor
benefit. **Flow matching is that predictor, and the prediction held at both high and xhigh.**

So `[CLF] xhigh` should be read as *a fact about miscalibrated classifiers*, not a general fact
about entropy acquisition.

### 5.4 Why BALD behaves as it does

A mechanism result, predictor-independent and derived rather than fitted:

`BALD ≈ Var_m[p] / (2·p̄(1−p̄))`, because `H''(p) = −1/(p(1−p))`.

For **identical** member disagreement, BALD is amplified **34.9× at p = 0.01 versus p = 0.5**.
BALD does not hunt disagreement — it hunts disagreement *in near-certain regions*. This is a
property of the exact mutual information, not an estimator artifact: any faithful implementation
of `I(Y;Θ|x)` inherits it.

Notably the finite-K MC bias — measured at **97.3% of the raw BALD signal** at K=20 — is *not*
the operative problem, despite its size. A near-constant offset preserves ranking. The curvature
weighting is what changes which points get acquired, and it is the real argument for `epi_var`,
which the design did not anticipate.

### 5.5 The secondary question is unanswerable

`epi_var` vs `epi_bald` cannot be settled here. On the classifier both are exact (members
enumerated, K = None), so that comparison is uninformative by construction. On flow matching at
K=20 the comparison is live — but FM is null at low, high and xhigh, with the two estimators
indistinguishable at every one. **There is no effect anywhere for the estimators to differ on.**

---

## 6. Limits and caveats

- **One system.** Stochastic pendulum ROA. Nothing here generalises to other dynamics without
  re-running.
- **`det` rests on a 4-epoch floor.** Its `dir00` control was preempted and is rebuilding while
  five of seven runs at that level have finished all 19 epochs. Given `[CLF]` high showed a
  shallow floor supporting three "harmful" arms that collapsed to one at full depth, the det
  multipliers should be treated as provisional.
- **`[FM]` high and xhigh floors are cross-cluster** (iLab + arrakis / Amarel), which *inflates*
  them. Conservative for claiming significance, but not magnitude-comparable to the classifier
  floors. `[FM] low` is the first fully within-cluster FM floor.
- **`[CLF]` low/med/xhigh come from truncated curves** — those arms hit a 24h walltime at 12–15
  of 19 epochs. Only `[CLF]` det and high are complete.
- **No recalibrator was fitted.** "Post-recalibration" is `UNC − RES`, a theoretical floor. A real
  recalibrator recovers only part of it, so achievable benefit lies between the raw and
  post-recal columns.

---

## 7. Methodological lessons

**Shallow verdicts lie, repeatedly.** Three separate verdicts reversed when depth increased:
`aleat` on the deterministic curve went from +10.7× floor *worse* at epoch 4 to *best* by epoch
18; `[CLF] high` went from three harmful arms to one; `med` lost its effect entirely when the
pooled floor replaced a single-epoch one (3.4× larger). The two-consecutive-epoch rule is **not**
sufficient protection on its own — the quantity oscillates on that timescale, so two adjacent
samples can agree by luck *even across independent metrics*. The full-trajectory sign check was
added because of this and immediately caught cases the window rule passed.

**A run-to-run floor is not optional.** Seed-to-seed variation on identical configurations
exceeded every acquisition effect measured on flow matching. Without three seeds per level this
campaign would have reported a string of confident, meaningless rankings.

**Report the negative control's number, always.** It is the single most informative line in every
table here. A result where the useless arm performs like the real ones is not a weak positive —
it is evidence the measurement is not resolving what it claims to.
