# Scalar-outcome flow matching: is the calibration edge target or machinery?

Design and pre-registered attribution rule:
[`docs/superpowers/specs/2026-08-10-outcome-fm-ablation-design.md`](../superpowers/specs/2026-08-10-outcome-fm-ablation-design.md).

## Question

The stochastic-pendulum campaign found flow matching beats the discriminative
classifier on calibration by 6–43× in debiased Brier while tying on ranking. It
could not say *why*, because endpoint-FM and the classifier differ on two axes at
once: **what** they predict (full state vs binary outcome) and **how** they form a
predictive distribution (generative vs cross-entropy point estimate).

This ablation adds the missing cell — outcome-target flow matching — so the two
axes separate.

## Answer

**MACHINERY.** A scalar 0/1 FM predictor with the classifier's exact backbone,
trained on byte-identical data, reproduces endpoint-FM's calibration to within the
run-to-run noise floor at every level tested, while the classifier sits 12–73×
away. The edge comes from velocity regression against sharp anchors plus a
distributional readout — not from modelling the full endpoint state.

The pre-registered ranking prediction **held**: sAUROC spread across all three
arms is ≤ 0.0017 at every level.

## Results

All runs `d2_ratio: 0` (fixed acquisition), 3 seeds per arm where available,
seed **medians** (single runs spike), debiased Brier against M=90 ground-truth
rollouts.

| stage | verdict | tracks endpoint-FM | tracks classifier | within noise floor | clf/outcome | floor available |
|---|---|---|---|---|---|---|
| deterministic | plumbing OK — calibration degenerate by construction | — | — | — | — | n/a |
| low | MACHINERY | 5/5 | 0/5 | — | — | no (1-seed fm ref) |
| med | MACHINERY | 15/15 | 0/15 | 12/15 | 12–31× | **yes** |
| **high** | **MACHINERY** | **19/19** | **0/19** | **12/19** | **30–73×** | **yes** |
| xhigh | MACHINERY (directional) | 15/15 | 0/15 | — | 8–44× | no (1-seed fm ref) |

`high` is the most complete level: all three arms have 3 within-campaign seeds at
every epoch 0–18, so the floor exists everywhere and the floor test actually runs.

### high, epoch 18 (deepest matched)

| arm | debiased Brier | seed spread | mean bias | skill | sAUROC | mean p_invalid |
|---|---|---|---|---|---|---|
| endpoint FM (generative × state) | 0.00103 | 0.00010 | −0.0095 | 0.9945 | 0.9805 | **0.5650** |
| **outcome FM (generative × binary)** | **0.00093** | 0.00013 | −0.0021 | 0.9951 | 0.9816 | 0.0000 |
| classifier (discriminative × binary) | 0.05387 | **0.01469** | **+0.1375** | 0.7191 | 0.9811 | 0.0000 |

Noise floor (2×SD across fm seeds) = 0.00010; |outcome − endpointFM| = 0.00010,
exactly at the floor. sAUROC spread across arms 0.0011.

### Two observations that belong in any write-up

**Coverage is not equal.** Endpoint-FM abstains on 56.5% of the grid at `high`
(52.9% at `xhigh`); the outcome arm reaches the same — slightly better — Brier at
**zero** abstention. Equal calibration at unequal coverage is not an equal result.

**The classifier is unstable, not merely biased.** Its seed spread at `high` ep18
is 0.01469 — larger than the entire outcome-vs-endpointFM gap by two orders of
magnitude. Neither generative arm shows this.

## Readout validation

Both readouts required by the spec were computed. Checked on a *trained* model at
the hardest level (`high`, ep18, n=4000), where true p is interior and MC variance
is near maximum:

| quantity | value |
|---|---|
| mean\|MC − exact\| | 0.01244 |
| expected MC-only SE (K=100) | 0.01533 |
| non-monotone fraction | 0.0000% |
| exact vs the p the pipeline stored | 8.03e-06 |

The observed disagreement sits **below** the MC estimator's own standard error, so
the gap is entirely MC sampling noise and the exact readout adds no bias of its
own. Monotonicity held everywhere sampled, so bisection was valid and the
quadrature fallback never fired. Recomputing p from the checkpoint reproduces the
stored per-point p to 8e-6 — four orders below the effects being measured, and
confirmation that `forward_readout: exact` took effect rather than silently
defaulting. Stage 1 showed the same relationship (0.00134 vs 0.00177 expected SE).

## Caveats — none resolved by completion

1. **`pos_weight ≈ 1.5` on the classifier is a genuine confound.** Weighted BCE
   minimises to `w·p/(w·p + 1−p)`, a deliberate upward tilt. It predicts the right
   *sign* but roughly a fifth of the observed bias magnitude, so it contributes
   without explaining. **A `pos_weight: 1.0` control was never run**;
   `ClassifierTrainer._pos_weight` already honours the override if the machinery
   verdict is ever thought to hinge on it.
2. **Early stopping compares different quantities** — val BCE for one arm, val
   velocity MSE for the other. Inherent to comparing the two losses, not fixable
   by configuration.
3. **Capacity is +0.97%** in the outcome arm's favour (266,753 vs 264,193), all in
   the first layer. Cannot account for a 12–73× gap, but the arms are not
   bit-matched.
4. **Pendulum is 39–48% success.** These conclusions do **not** transfer to
   imbalanced systems (quad2d ≈ 8%) without re-examining the `pos_weight` axis.
5. **`low` and `xhigh` have no within-campaign noise floor** (their fm reference
   has 1 seed). Those verdicts are directional, not independently confirmatory.
   "Beats endpoint-FM" is not supportable at either level: a 1-seed reference has
   no spread to beat.

## Runs

Experiment root `${EXP}/outcome_fm`, 19 adaptive epochs × 3 seeds (42/43/44) per
level. All 13 SLURM jobs COMPLETED exit 0:0; zero errors and zero CPU fallbacks
campaign-wide.
