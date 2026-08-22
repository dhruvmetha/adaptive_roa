# Methods — stochastic adaptive-ROA campaigns

What each arm actually does, taken from the configs and source rather than from
memory. Scope: the `stochastic/` campaigns documented in `pendulum/`, `cartpole/`
and `quadrotor/`. Definitions live here so those three docs can report results
without restating mechanics.

Every claim about a default is from `configs/adaptive_v2/{acquisition,predictor}/`
or `adaptive_roa/adaptive_v2/strategy/`. Where a number was *measured* rather than
configured, it says so.

---

## 1. The loop every arm shares

One "epoch" is an acquisition round, not a gradient epoch:

1. Train the predictor on the current training set.
2. Score `n_candidates` (50,000) pool states.
3. Select `samples_per_epoch` of them; reveal their true rollouts.
4. Add to the training set; evaluate on the full ROA grid; write
   `artifacts_v2.json`.

Arms differ **only in step 3**. Same pool, same seed, same budget, same eval — so
a difference between arms is a difference in what was bought.

### The d1/d2 split

`acquisition.d2_ratio` splits each epoch's budget between two streams
(`engine.py:212`):

```
n_d1 = samples_per_epoch * (1 - d2_ratio)     # uniform draw from the pool
n_d2 = samples_per_epoch - n_d1               # chosen by the arm's score
```

| d2_ratio | meaning | used by |
|---|---|---|
| `0` | pure uniform — the **control** | `dir00_*`, `clf_dir00` |
| `1.0` | pure scored | every adaptive arm |
| `0.5` | half uniform, half scored — the **anchored** variant | `*_anch` |

**d1 is not random per seed.** It reads sequentially from
`train_test_splits/shuffled_indices_0.txt`, which is fixed. Every pure-uniform arm
therefore draws a *byte-identical* index set regardless of seed, and the 3-seed
control band measures **training stochasticity only**, not sampling variability.
This is why the seeds show Jaccard 1.000 against each other in the gate checks —
expected, not a defect.

**Anchoring matters more than it looks.** Measured on pendulum i100 over 20 epochs,
`epi_var_anch`'s uniform half contributed **74.8% of all training pairs** from 50%
of the trajectory budget (654,657 vs 220,976). The scored half buys the
true-success region and starves on pairs; the uniform half is what keeps the
training set growing.

---

## 2. Predictors — what produces `p̄`

Every acquisition score is a functional of per-member success probabilities
`p_1 … p_M`. The predictor decides what those are.

### `fm_ensemble` — flow-matching ensemble (the main arm)

5 members, `EnsembleFlowMatchingTrainer`, probability backend
`ensemble_endpoint_mc`. Each member predicts an **endpoint distribution** from a
start state; `p_m` is the fraction of `K` sampled endpoints landing inside
`attractor_radius` of the goal.

So `p_m` is **itself a Monte-Carlo estimate** — which is what makes the finite-K
bias in §3 real rather than theoretical.

### `clf_ensemble` — deep-ensemble classifier

`BayesianMLPTrainer`, `posterior: ensemble`, hidden `[256, 512, 256]`. `p_m` is a
forward pass, **exact, no sampling noise** — so the debiasing term in
`epistemic_var` is skipped (`k=None`).

Uses `decision_rule: one_sided`, unlike the FM arms' `two_sided`.

### `gp` — Gaussian-process classifier (Part-X only)

`GPPredictorTrainer`, Matérn-5/2, 128 inducing points, 300 ELBO iterations, fixed
`λ*=0.5`, `δ*=0.1`. Exposes `latent_posterior`, which **Part-X requires**.

### `gp_reg` — GP regressor

`GPRegressorTrainer` with the `endpoint_mc` backend. **Not interchangeable with
`gp`:** it has no `latent_posterior`, so `partx` with `gp_reg` dies at
`adaptive_roa/partx/strategy.py:30`. This killed cartpole jobs in under 80 s.

> **The three predictor config traps.** Each of these cost a dead job:
> ```
> GP)  predictor=gp predictor.gp.n_iters=300                                   # never gp_reg
> CLF) predictor=clf_ensemble  +predictor.lightning_trainer.enable_progress_bar=false   # '+' form
> *)   predictor=fm_ensemble    predictor.lightning_trainer.enable_progress_bar=false   # no '+'
> ```
> The classifier config has no `lightning_trainer` key to override, so it needs the
> append form; `fm_ensemble` does not.

**Cross-family comparisons are invalid.** FM arms are judged against the 3-seed FM
band, classifier arms against `clf_dir00`, Part-X against neither — it is a
different model class. Plots encode family in line style for this reason. There is
**no classifier floor**: one seed per clf arm.

---

## 3. Uncertainty scores

For M members with success probabilities `p_1 … p_M`
(`strategy/uncertainty_scores.py`):

```
H(p̄)      =   E_m[H(p_m)]   +   I(y ; m | x)
total          aleatoric         epistemic
```

| `score` | formula | targets |
|---|---|---|
| `total` | `H(p̄)` | all uncertainty, reducible or not |
| `aleatoric` | `E_m[H(p_m)]` | irreducible outcome randomness |
| `epistemic_bald` | `H(p̄) − E_m[H(p_m)]` (mutual information) | member disagreement |
| `epistemic_var` | `Var_m[p_m] − mean_m[p_m(1−p_m)/(K−1)]` | member disagreement, **MC-debiased** |

`aleatoric` is a deliberate **negative control**: it targets what more data cannot
fix. If it is not the worst arm at high noise, the decomposition is not working.

### The finite-K bias — why `epistemic_var` is the default

When each `p_m` is MC-estimated (any FM arm), `epistemic_bald` is biased **upward**
by roughly `(1/2K)(1 − 1/M)` — about **0.021 nats at K=20, M=5**. The bias is
roughly flat across the interior and vanishes only at `p = 0` and `p = 1`, so it
**lifts every non-deterministic state above every decided one whether or not
members actually disagree**. It does *not* shrink as M grows.

`epistemic_var` subtracts exactly that term and is unbiased at any K. Pass
`k=None` when `p_m` is exact (classifier), which skips the correction.

`binary_entropy` is exact at the boundaries (`xlogy(0,0) = 0`, not clipped) —
load-bearing, because a clipped boundary would blur precisely the distinction the
bias analysis rests on. Out-of-domain input **raises** rather than returning NaN:
`select_greedy` skips non-finite scores, so a silent NaN would quietly drop
candidates and let a partially-non-adaptive arm report a full run.

---

## 4. Selection rule

All decomposition arms use `selection_rule: greedy_diverse`
(`strategy/dispersion_score.py:141`):

1. Take the top `5 × n_select` candidates by score.
2. Farthest-point-sample `n_select` from that shortlist in **initial-state**
   space, seeded at the highest scorer, with per-dimension normalization and
   circular-aware distance.

This stops a batch collapsing into one high-uncertainty pocket. Plain `greedy`
(top-N) is available but unused in these campaigns.

---

## 5. Acquisition strategies

### `direct` — uniform control
`DirectAcquisitionStrategy`, `d2_ratio=0`. Sequential draw from the fixed shuffle.
Three seeds (42/43/44) define the floor.

### `decomp_*` — the decomposition family
`DecompositionAcquisitionStrategy` with `score` set to one of the four above.
`decomp_epi_var`, `decomp_epi_bald`, `decomp_total`, `decomp_aleat`.

### The yield family — informativeness *per trajectory purchased*

Targets a measured failure, not a hypothesis:

> The budget is **trajectories**, but the model trains on **pairs**. On
> `noisy/pendulum/lqr/high`, success rollouts end at the goal (mean 136.5 steps)
> while failures run to the 1001-step cap. The pool is **40.3% success by
> trajectory but 8.4% by pair**. Every uncertainty score buys the model's uncertain
> band — on that system the true-success region (mean true `p` of acquired points
> 0.90–0.99) — hence the *short* rollouts. `epi_var` finished 20 epochs with 358k
> pairs against the control's 652k by epoch 9. **At matched pair count the arms beat
> uniform on KL by 1.6–2.6×; at matched trajectory count they lose.** Selection was
> never the problem; yield per selection was.

All three variants rank by `score(x) · E[L(x)]^α` and **reduce exactly to
`decomp_epi_var` at α=0**, so each is a single-knob change. They differ only in how
`E[L]` is estimated:

| arm | length estimator |
|---|---|
| `decomp_epi_var_yield` | two-point mixture `p̄·L_succ + (1−p̄)·L_fail`, both estimated online from **already-acquired** trajectories |
| `decomp_yield_knn` | distance-weighted kNN (k=15) on the **true lengths** of acquired trajectories |
| `decomp_yield_mlp` | MLP 64-64, refit each epoch, with a per-epoch held-out **bake-off against the kNN** — whichever wins is used, both MSEs logged |

`α` variants: `_a05` (0.5), default (1.0), `_a20` (2.0).

Lengths are never read from the *candidate* — its length is unknown before rollout.
Only `p̄` plus statistics of data already paid for.

**Why `yield_knn` exists:** the mixture routes through `p̄`, which is exactly what
is miscalibrated early. Measured at epoch 0: expected 82,002 pairs, bought 11,374
(7.2× over), because `p̄` said 0.21 where the truth was 0.99. Also measured: on that
pool `L_failure = 1001.0` with **sd 0.0** (every failure hits the cap) and
`L_success = 136.5` (sd 76.5), so **98.7% of Var(L) is explained by the outcome
label alone** — predicting length from the start state *is* predicting the outcome.

**Why `yield_mlp` runs a bake-off:** a unit test found the MLP inverting the length
ordering on 40 points in two clean clusters (430 vs 224 for true 130 vs 1001) —
sklearn's `early_stopping` carves 10% off an already tiny set. The per-epoch
bake-off bounds this arm below by `yield_knn`.

### `decomp_length_only` — the control that isolates the length term
Drops the score, keeps only `E[L]`. If it reproduces the yield arm's win, the
epistemic term contributes nothing and the honest description is "buy long
trajectories". Because `L_failure ≫ L_success` on pendulum, ranking by `E[L]` is
monotone *decreasing* in `p̄`, so this arm buys the predicted **failure** region —
the opposite pole from every scored arm. **Result: it does not reproduce the win.**

### `partx` — Part-X level-set estimation
`PartXAcquisitionStrategy` with the `gp` predictor. Partition tree
(branching factor 2, `δ=0.05`, `α=0.05`), UCB `β=1.96`, budget allocated
`per_region_volume`. A **different model class**, not an acquisition variant.

> `max_leaves: null` is fine in low dimensions. In ≥6-D, region classification
> rarely resolves and leaves grow ~2^epochs → OOM. Set e.g. 4000 for the
> quadrotors.

---

## 6. Metrics

Scored by `scripts/stoch_prob_metrics.py` against the **continuous** ground-truth
`p_success` grid.

| metric | what it measures | direction |
|---|---|---|
| **sAUROC** | ranking, against continuous truth (not a 0.5 dichotomy) | higher |
| **KL** | divergence from the true probability field | lower |
| **recal** = `UNC_debiased − RES` | calibration residual, from the Murphy decomposition `Brier = REL − RES + UNC` | lower |
| `brier_debiased` | finite-K sampling bias removed | lower |

**Never use the artifact-level `auc` / `brier` fields.** They score against
`p_success ≥ 0.5`, and **28.6% of the pendulum grid lies in [0.05, 0.95]** where
that dichotomy discards the signal.

**Where a win can show.** On pendulum i100 the oracle ceiling is sAUROC 0.9834
(0.9824 at the K=100 limit) against a control of 0.9785 — at most ~0.005 of room
against a floor of ~0.0015. **sAUROC and recal are nearly saturated there; KL has
~15× headroom.** Stated before scoring, not after.

---

## 7. Standard of evidence

**Floor** = `2 · sqrt(mean_e(var_e))` over three genuinely distinct control seeds,
pooled across epochs, **epoch 0 excluded** (it is the pre-acquisition model; its
spread reflects initialisation, not acquisition). A gap smaller than the floor is
not claimable.

Two windows are used, and they are not interchangeable:

- **full** (ep1–19) — conservative; inflated by the early epochs when every arm is
  still descending steeply.
- **converged** (ep12–19), judged on a pooled **ep15–19** mean — the registered
  standard for the i100 campaign. On pendulum i100: KL floor **0.01693** converged
  vs **0.03045** full.

Judging one epoch against the full-window floor understates every gap. Quote which
window is in use.

**Reference convention.** Gaps can be quoted against `dir00` alone (paired,
seed-42-vs-seed-42) or against the 3-seed control mean (unpaired, more
conservative). They differ by ~10% — `epi_var_yield` reads +1.87 vs +1.70 floor
units on KL. **Never mix the two in one table.**

### Two screens that gate scoring

**Collapsed epochs** (`scripts/screen_collapsed_epochs.py`) — flags sAUROC ≤0.60
**together with** RES ≤0.01. A collapsed epoch is a training failure, not a data
point: inside a floor pool it inflates the floor and manufactures false *nulls*;
inside an arm it manufactures a false *catastrophe*. Run before every scoring pass.

**Magnitude, not non-finiteness** — flag any epoch whose max of
`success_mae` / `failure_mae` / `overall_mae` exceeds **1.0** *or* is non-finite. A
non-finite-only screen once cleared a cartpole arm carrying `9.4e+09`.

> Classifier and Part-X arms report `endpoint_error` as 0.0 — they predict no
> endpoints. Including them makes the magnitude screen look clean while testing
> nothing. Screen FM arms separately.

---

## 8. Which methods ran where

| arm | acquisition | d2 | predictor | pendulum | cartpole | quad2D |
|---|---|---|---|---|---|---|
| `dir00_s42/s43/s44` | direct | 0 | fm_ensemble | ✓ | ✓ | ✓ |
| `epi_var` | decomp_epi_var | 1.0 | fm_ensemble | ✓ | ✓ | ✓ |
| `epi_bald` | decomp_epi_bald | 1.0 | fm_ensemble | ✓ | ✓ | ✓ |
| `epi_var_anch` | decomp_epi_var | 0.5 | fm_ensemble | i100 only | ✓ | ✓ |
| `yield_a1` | decomp_epi_var_yield | 1.0 | fm_ensemble | ✓ | ✓ | ✓ |
| `yield_mlp` | decomp_yield_mlp | 1.0 | fm_ensemble | ✓ | ✓ | ✓ |
| `partx` | partx | 1.0 | **gp** | ✓ | ✓ | ✓ |
| `clf_dir00` | direct | 0 | clf_ensemble | ✓ | ✓ | ✓ |
| `clf_yield` | decomp_yield_mlp | 1.0 | clf_ensemble | ✓ | ✓ | ✓ |
| `clf_epi_var` / `_bald` / `_var_anch` | as named | 1.0 / 1.0 / 0.5 | clf_ensemble | — | ✓ | ✓ |
| `total` | decomp_total | 1.0 | fm_ensemble | i100 only | — | — |
| `length_only`, `yield_knn`, `yield_a05`, `yield_a20`, `epi_var_qknn`, `qknn2` | see §5 | 1.0 | fm_ensemble | i100 only | — | — |

Budgets differ per system and are **not** comparable across them:

| campaign | init | step | epochs | final |
|---|---|---|---|---|
| pendulum i100 (`noisy/lqr/high`) | 100 | 100 | 20 | 2,100 |
| pendulum gaussian_signal | 100 | 100 | 20 | 2,100 |
| cartpole gaussian_signal v2 | 300 | 150 | 12 | 2,100 |
| quadrotor2D | 2,000 | 500 | 24 | 14,000 |

The three original campaigns all **end at 2,100 trajectories** despite different
schedules, so their final-epoch numbers sit at a matched trajectory budget. They
do *not* sit at a matched **pair** count — that is the whole subject of §5, and on
pendulum the same 2,100 trajectories bought 358k pairs for `epi_var` against 652k
for the control. Matching the trajectory budget is not matching the training data.

The quadrotors are 6.7x larger and not budget-comparable to any of them.

---

## 9. Results, in one line each

- **pendulum gaussian** — low/med: several arms beat the control on KL; **high is a
  null** and is reported as one.
- **pendulum i100** — at matched *trajectory* budget the original scored arms
  **lose**; at matched *pair* count four of five **win** on KL by 1.7–6.1×.
  `length_only` is catastrophic (−10.22 floor units), so the epistemic term is
  load-bearing. α peaks at 1.0.
- **cartpole gaussian** — `epi_var` and `epi_bald` win at **every** noise level
  (2.3–2.9× KL). Part-X degrades as noise rises. Classifier arms win on sAUROC while
  losing on KL — they rank well and calibrate badly.
- **quadrotor2D** — in flight.

Full numbers in `pendulum/`, `cartpole/`, `quadrotor/` and
`../ensemble_epistemic/YIELD_AWARE.md`.
