# Scalar-Outcome Flow Matching: separating target from machinery

## Goal

The stochastic-pendulum campaign found that flow matching beats the discriminative
classifier on **calibration** by 6–43× in debiased Brier while tying on **ranking**
(soft AUROC identical to 3–4 decimal places at every noise level). It did not
establish *why*, because endpoint-FM and the classifier differ on two axes at once:
what they predict, and how they form a predictive distribution.

This work adds the missing cell so the two axes can be separated.

## The factorial

The pipeline already holds three of four cells:

| | Outcome target | Final-state target |
|---|---|---|
| Cross-entropy point estimate | `classifier` | — |
| Gaussian NLL / Bayesian posterior | `bnn_*`, `gp` | `bnn_*_reg`, `gp_reg`, `hmc_reg`, `mlp_det` |
| **Flow matching** | **this work** | `EndpointMCProbabilityBackend` |

Flow matching appears only in the final-state column. Adding outcome-target FM
completes the row and makes the contrast one-factor-at-a-time in both directions:

- **vs `classifier`** — identical data, same backbone and hidden dims. Only the
  loss differs: velocity regression against ±1 anchors versus weighted
  cross-entropy. This isolates *machinery*.
- **vs endpoint FM** — same generative machinery, MC-sampled. Only the target
  differs: a scalar outcome versus a full state on the manifold. This isolates
  *target*.

## Attribution rule (fixed before running)

| Observation | Conclusion |
|---|---|
| outcome-FM calibration ≈ endpoint-FM | the 6–43× edge comes from flow-matching machinery |
| outcome-FM calibration ≈ classifier | the edge comes from modelling the full endpoint state |
| between the two | both contribute; the MC-vs-exact gap bounds how much is sampling noise |

Recorded prediction: **ranking should be unaffected.** Endpoint-FM and the
classifier tie on sAUROC to 3–4 dp, so outcome-FM should too. If it does not, the
ranking/calibration separation the campaign rests on is less clean than believed.

## Design

**Geometry.** Flow space is R¹ with sharp anchors `-1` = failure, `+1` = success.
Anchors at ±1 rather than {0,1} place the decision threshold at 0, the mean of the
`N(0,1)` source, so an uninformative model returns p = 0.5 and the threshold needs
no separate calibration.

**Training.** Standard conditional FM on a linear path. Per `(state x, label y)`:

```
x0 ~ N(0,1);  x1 = +1 if y else -1;  t ~ U(0,1)
xt = (1-t)*x0 + t*x1
loss = MSE( v(xt, t | x),  x1 - x0 )
```

**Backbone.** The same MLP the classifier uses, over the same embedded state, with
the same hidden dims `[256, 512, 256]`. Matching capacity is what makes the
machinery contrast clean; changing one arm's dims without the other breaks the
ablation.

Capacity is **near**-identical, not identical, and the exact figure is stated here
rather than glossed: the velocity net also takes `x_t` and a 9-dimensional time
embedding, so its input is 13 wide against the classifier's 3. That is
**266,753 vs 264,193 parameters, +0.97%**, all of it in the first layer. A 1%
parameter difference cannot account for a 6–43× calibration gap, so it does not
threaten the attribution — but the arms are not bit-matched on capacity and the
write-up should not claim they are.

**Readout.** In 1D the ODE is a deterministic map `Psi: x0 -> x1`, so
`p = P(Psi(x0) > 0)`. Both readouts are computed:

- `mc` — K-sample fraction, the same procedure endpoint-MC uses, so a gap against
  endpoint FM is attributable to the target and not the readout. Quantised to 1/K.
- `exact` — bracket the crossing on a 33-point grid over `x0 ∈ [-5, 5]`, then
  bisect. Continuous, no MC noise, and *cheaper* than MC (≈53 integrations per
  point versus 100 at K=100).

The pipeline runs on `exact`; both are exported for scoring. Their difference is
the estimate of sampling noise, which is what lets a residual gap be attributed to
the model rather than the estimator.

**Monotonicity is checked, not assumed.** `Psi` monotone is a property of the
learned field. Points whose grid shows more than one sign change fall back to
quadrature over the same evaluations, and the non-monotone fraction is reported so
a fallback cannot be mistaken for a clean bisection.

## Numerical requirements

The effects being measured are debiased-Brier gaps of 1e-4 to 1e-3, so the readout's
own error must sit well below that or a "result" can be an artefact of the estimator.
Both error sources were measured, not assumed:

| Source | Setting | Worst-case error in p | Evidence |
|---|---|---|---|
| ODE integration | `num_ode_steps: 50` | 3e-5 | max abs Δp vs a 200-step reference on an untrained (rough) field; 20 steps gives 1.1e-4, **not** safely below the signal |
| Bisection | `bisect_iters: 20` | 1.9e-6 | bracket 10/2²⁰ × dp/dx0 ≤ φ(0) = 0.399; n=14 gives 1.2e-4, same order as the signal |

Both are pinned by `test_readout_precision_is_below_the_effect_size`.

## Integration (Route B)

`predictor.type: classifier` names the **data path and eval branch, not the
machinery**. Outcome-FM consumes classification data and has no endpoint, so every
branch the tag disables — endpoint-error metrics, endpoint refinement at
`engine.py:311`, the eval skip at `:209` — is disabled correctly rather than as a
workaround. No engine edits, which matters because the `en_*` ensemble campaign is
running against that file.

Reusing the tag also means outcome-FM and the `classifier` arm see **byte-identical
training data at every adaptive epoch**, so the contrast is clean by construction
rather than by matching after the fact.

*Verified, not assumed:* with `d2_ratio: 0` the D1 selection in
`adaptive_roa/adaptive/dataset_builder.py:229-250` sorts the unused indices and takes
the first n — a deterministic prefix of the fixed on-disk shuffle
(`train_test_splits/shuffled_indices_0.txt`, shared across levels). No RNG is drawn,
so the data is identical across arms regardless of each run's `seed` setting.

`OutcomeFlowMatcher.forward(raw_states) -> logit(p_success)` makes the module a
drop-in for every discriminative code path (threshold optimisation, conformal
calibration, `ClassifierProbabilityEstimator`), which is what keeps Route B free of
engine surgery. Training calls `velocity()` instead.

**Readout consistency is enforced, not documented.** Calibration reaches `p` through
`forward`, evaluation through the backend. `OutcomeFMProbabilityBackend.bind_model`
raises if the two readouts disagree, because otherwise calibration would optimise
against a different quantity than evaluation reports — silently, with both numbers
looking reasonable.

### Files

| File | Contents |
|---|---|
| `adaptive_roa/model/outcome_flow_matcher.py` | `OutcomeVelocityMLP`, `OutcomeFlowMatcher` (training, flow map, both readouts) |
| `adaptive_roa/adaptive_v2/trainers/outcome_fm_trainer.py` | `OutcomeFMTrainer`, mirrors `ClassifierTrainer`'s contract |
| `adaptive_roa/adaptive_v2/probability/outcome_fm.py` | `OutcomeFMProbabilityBackend`, `estimate` + `estimate_both` |
| `configs/adaptive_v2/predictor/fm_outcome.yaml` | arm config, capacity matched to `classifier.yaml` |
| `configs/adaptive_v2/probability/outcome_fm.yaml` | readout and numerical settings |
| `configs/adaptive_v2/experiment/fm_outcome_baseline.yaml` | `d2_ratio: 0` fixed-dataset baseline |
| `tests/adaptive_v2/test_outcome_flow_matcher.py` | 13 tests |

## Staging

Deliberately one level at a time, ascending:

1. **deterministic pendulum** — correctness only. True `p ∈ {0,1}`, so calibration
   is degenerate and this stage answers *nothing* about the research question. It
   validates plumbing, monotonicity, and that the arm reaches classifier-level
   accuracy.
2. **noisy low** — first stage where true `p` is interior and calibration is
   measurable.
3. **med**, **high**, **xhigh** — the levels where the campaign's 6–43× gap and the
   classifier's skill-score collapse (0.44 at xhigh) were measured.

Each stage is scored against the existing `clf_{level}_dir00` and `fm_{level}_dir00`
arms at matched epochs before the next begins.

All runs use `d2_ratio: 0`. Fixed acquisition is required, not incidental: the
ablation varies target × machinery, so the data-selection rule must be held constant
or the two factors are confounded with a third.

## What differs between the arms (complete enumeration)

An attribution is only as good as the list of things that were *not* held fixed,
so the full list is here rather than left implicit:

| axis | classifier | outcome-FM | matched? |
|---|---|---|---|
| loss | weighted BCE | velocity MSE | **intended difference** |
| p readout | `sigmoid(logit)` | exact x0 threshold | **intended difference** |
| training rows | prefix of fixed shuffle | identical | yes (verified, no RNG) |
| backbone dims | `[256,512,256]` | same | yes |
| parameters | 264,193 | 266,753 | +0.97% |
| activation | relu | relu | yes |
| optimizer / lr / weight decay | AdamW / 1e-3 / 1e-5 | same | yes |
| max_epochs / patience | 200 / 20 | same | yes |
| **`pos_weight`** | **n_neg/n_pos ≈ 1.5** | **none** | **NO — confound** |
| early-stop criterion | val BCE | val velocity MSE | different *quantity* |

Two rows are not clean and both are stated rather than buried.

**`pos_weight` is a genuine confound for the machinery comparison.** BCE weighted
by `w` does not minimise to the posterior; its minimiser is `w·p/(w·p + 1−p)`, a
deliberate upward tilt. Measured on the non-adaptive `dir00` arms, the classifier
over-predicts success at every level (+0.026 / +0.038 / +0.138 at low / med /
high) while endpoint-FM is near-unbiased (−0.003 / −0.016 / −0.010). The tilt
predicts the right *sign* but roughly a fifth of the magnitude, because most
pendulum cells sit near 0 or 1 where the tilt is weak — so it contributes without
explaining. A `pos_weight: 1.0` classifier control would separate the two, and
`ClassifierTrainer._pos_weight` already honours that override, so it is a cheap
addition if the machinery verdict turns out to hinge on it.

**Early stopping compares different quantities.** Both arms stop on `val_loss`
with patience 20, but that is BCE for one and velocity MSE for the other. Neither
is a proper scoring rule for calibration, and they need not reach their optima at
the same point. This is inherent to comparing the two losses at all, not a defect
that can be configured away — it is a caveat on the result, not a bug to fix.

## Known asymmetry

`ClassifierModule` counters class imbalance with `pos_weight`; velocity regression
has no equivalent knob. On the pendulum (39–48% success) `pos_weight ≈ 1` and the
asymmetry is immaterial. It would **not** be immaterial on quad2d (~8% success), so
this ablation's conclusions do not transfer to imbalanced systems without
re-examining that axis.

## Validation status

- 13 unit tests pass, including the exact readout against the analytic `Phi(c)` for
  a constant field (5 values of c), MC/exact agreement, monotonicity detection,
  logit finiteness under saturation, and the precision guard.
- End-to-end recovery of a known field `p(x) = x`: mean abs error **0.013**, max
  **0.042**; MC and exact agree to 0.0004, confirming both readouts estimate the
  same quantity.
