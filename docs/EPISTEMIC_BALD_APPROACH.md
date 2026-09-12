# Epistemic BALD acquisition, end to end

What the `decomp_epi_bald` arm does, why it exists, what data it runs on, and how the pieces
fit. Written from the configs and source rather than from memory; every claim about a default
is traceable to a file named in §9. Measured numbers say so explicitly.

Companion documents: `/common/users/shared/pracsys/genMoPlan/docs/stochastic/METHODS.md` (mechanics shared by every arm in
the stochastic campaigns), `docs/experiments/ensemble_epistemic/FINDINGS.md` (results and their
correction history), `docs/superpowers/specs/2026-08-04-ensemble-epistemic-acquisition-design.md`
(the original design).

---

## 1. The problem

The target is a scalar field. For a stochastic system with a fixed control law and a fixed
horizon, `p_success(x)` is the probability that a rollout starting at `x` reaches the goal set
before the horizon expires. It is a bounded-time reach probability, not an asymptotic region of
attraction, and on the quadrotors that distinction is large enough to matter (see the horizon
note in `configs/adaptive_v2/system/quadrotor3d_stoch.yaml`).

A model estimates that field. Two model families are in use: a flow matcher that predicts an
endpoint distribution from a start state, and an MLP classifier that predicts the success label
directly. Either way the model is fitted on rollouts, and rollouts cost simulation time.

The question this work answers is which rollouts to buy. Each adaptive epoch has a budget of
`samples_per_epoch` trajectories drawn from a pool of pre-simulated rollouts. Buying uniformly
at random is the control. Any acquisition rule has to beat it at matched budget.

**Why entropy acquisition is not the answer.** Scoring candidates by the predictive entropy
`H(p̄)` is the obvious rule and it degrades as process noise rises. On the earlier stochastic
pendulum campaign, at high noise the entropy score selected states that were genuinely
ambiguous 1.0% of the time against uniform sampling's 13.3%, and pulled the mean true `p` of
acquired points from 0.406 down to 0.046. Cost: +0.157 debiased Brier, roughly 5x the
run-to-run floor, with a monotone dose-response in the adaptive fraction.

The reason is that `H(p̄)` peaks where the outcome is a coin flip, and under heavy noise that is
exactly where more rollouts teach the model nothing. Entropy cannot tell "the outcome here is
random" from "the model does not know what happens here". The first is irreducible. Only the
second is worth paying for.

An ensemble can tell them apart, which is what BALD is doing here.

---

## 2. The decomposition

For M ensemble members with per-member success probabilities `p_1 … p_M` and marginal
`p̄ = mean_m p_m`:

```
H(p̄)        =    E_m[H(p_m)]      +    I(y ; m | x)
total             aleatoric             epistemic (BALD)
```

All three terms are in nats. `binary_entropy` uses `scipy.special.xlogy`, so `H` is exactly 0 at
`p = 0` and `p = 1` rather than clipped, which the bias analysis in §3 depends on.

| score mode | formula | what it buys |
|---|---|---|
| `total` | `H(p̄)` | anything uncertain, reducible or not |
| `aleatoric` | `E_m[H(p_m)]` | irreducible outcome randomness (negative control) |
| `epistemic_bald` | `H(p̄) − E_m[H(p_m)]` | states where the members disagree |
| `epistemic_var` | `Var_m[p_m] − mean_m[p_m(1−p_m)/(K−1)]` | member disagreement, MC-debiased |

All four live in `adaptive_roa/adaptive_v2/strategy/uncertainty_scores.py` and are dispatched by
`score_by_mode` at line 134. One strategy class reads them
(`strategy/decomposition.py:26`), so the four arms are the same code path with one word changed
in the config. That is what makes `aleatoric` a usable control instead of a different program.

The identity holds to float precision on real model outputs. `FINDINGS.md` §1 reports a residual
of 0.0e+00 between the `total` arm's score and the sum of the `epi_bald` and `aleat` arms'
scores, and epoch-0 agreement across all five arms to within 6e-05 on both predictors.

---

## 3. What BALD actually prefers, and its two biases

Two properties shape which states BALD picks, and neither is obvious from the formula.

**BALD is member disagreement divided by `p̄(1−p̄)`.** Because `H''(p) = −1/(p(1−p))`, a second
order expansion gives `BALD ≈ Var_m[p] / (2·p̄(1−p̄))`. For identical member disagreement the
score is 34.9x larger at `p = 0.01` than at `p = 0.5`. So BALD systematically hunts extreme
probabilities while total entropy hunts ambiguous ones. This holds even with exact probabilities
and no sampling, and it is the sharpest behavioural difference between `epi_bald` and `epi_var`
(`FINDINGS.md` §2).

**Finite-K bias when `p_m` is Monte Carlo estimated.** On the flow matching side each `p_m` is
the fraction of K sampled endpoints landing inside `attractor_radius` of the goal, so it carries
binomial noise, and naive BALD reads that noise as disagreement. The bias is about
`(1/2K)(1 − 1/M)`, which is 0.021 nats at K=20 and M=5. Measured on real flow matchers it came
out at 0.01946 against the analytic 0.02000. It is flat across the interior, vanishes only at
`p = 0` and `p = 1`, and does not shrink as M grows.

A near-constant offset does not reorder points inside the interior, so the practical effect is
narrower than it first looks: it lifts every non-deterministic state above every confidently
decided one whether or not the members actually disagree. Genuine disagreement (members at 0.2
versus 0.8) scores 0.209, so the signal to bias ratio is roughly 10:1 early in a run and
tightens as the ensemble converges.

`epistemic_variance` subtracts exactly this term and is unbiased at any K, which is why it is the
default of the family. On the classifier both estimators are exact: members are enumerated by one
forward pass each, `member_sample_size` is `None`, and the correction is skipped.

The design justified `epi_var` by the finite-K bias. The property that shows up in the results is
the curvature weighting, not the bias.

---

## 4. Data

Two separate datasets are in play per run, and conflating them is a common source of confusion.

### The pool: what a run can buy

`train.npz`, one flat archive per system and noise level, read by
`adaptive_roa/adaptive/npz_data_source.py`:

| key | shape | meaning |
|---|---|---|
| `states` | (sumT, D) | concatenated rollout states |
| `offsets` | (N+1,) | rollout boundaries into `states` |
| `starts` | (N, D') | sampled initial condition per rollout |
| `labels` | (N,) | binary success per rollout, {−1, +1} |
| `start_ids`, `seeds` | (N,) | unique-start index and per-rollout seed |

Pool index k maps to npz row `rollout_ids[k]` through
`train_test_splits/shuffled_indices_{variant}.txt`.

Start states are read from `states[offsets[r]]`, never from `starts`. On quadrotor3D the two are
different objects: `starts` is 12-D with Euler angles and clipped to the sampling bound, `states`
is 13-D with a quaternion and already reflects the first control step. Feeding `starts` to the
model killed all 20 scored-acquisition arms with an IndexError. On quadrotor2D they agree to
2.4e-07.

The training file the model sees depends on the predictor. Flow matching gets endpoint pairs
`(start, end)`; the classifier gets `(state, label)` rows. `AdaptiveDatasetBuilder` writes both.

### The ground truth: what a run is scored against

`eval_success_prob.npz` beside the pool, carrying `starts`, `successes`, `trials` and
`p_success` from repeated rollouts per cell (about 90 on the pendulum grid). `scripts/stoch_prob_metrics.py`
matches predictions to truth with a cKDTree nearest-neighbour query, so it is dimension
agnostic. `cal_set.txt` and `test_set.txt` split off calibration and evaluation states.

Scoring is against the **continuous** `p_success` field. The artifact-level `auc` and `brier`
fields score against a `p_success ≥ 0.5` dichotomy and throw away the signal exactly where the
interesting states are (28.6% of the pendulum grid lies in [0.05, 0.95]). Do not quote them.

### Systems and budgets currently in use

| system | dataset family | levels | init / step / epochs |
|---|---|---|---|
| pendulum | `stochastic/pendulum/gaussian_signal/lqr` | low, med, high | 1000 / 1000 / 19 |
| cartpole | `stochastic/cartpole/gaussian_signal/lqr` | low, med, high | 300 / 150 / 12 |
| quadrotor2D | `stochastic/quadrotor2D/{noisy_dynamics, corridor_sine_ambient}/rl` | per family | 2000 / 500 / 24 |
| quadrotor3D | `stochastic/quadrotor3D/{noisy_dynamics, corridor_sine_ambient}/lqr` | f_0.032 … f_0.072, f_0.25, f_0.30 | 10000 / 5000 / 18 |

Noise families are never pooled across systems, and quadrotor3D's two families are not on a
common scale with each other or with the 2D system.

### One structural fact about the pool that shapes every result

The budget is denominated in trajectories, the model trains on pairs, and the two are coupled to
the outcome label. On pendulum, successes stop on goal entry at roughly 150 steps while failures
run to the 800-step cap, so the pool is about 40% success by trajectory and 11% by pair. On
cartpole and both quadrotors the coupling inverts: successes are the long rollouts. Any score
that prefers one class therefore also changes how fast the training set grows. This is what the
yield-aware family (`docs/experiments/ensemble_epistemic/YIELD_AWARE.md`) exists to address, and
it is why a matched-trajectory comparison and a matched-pair comparison can disagree.

### A caveat about `attractor_radius`

`attractor_radius` labels each of the K sampled endpoints inside the probability backend, so it
defines the model's predicted `p_success` and feeds straight into every metric. Several system
configs deliberately keep a radius larger than the one the dataset used to label success:
pendulum 0.1 against a per-channel box of 0.05, cartpole 0.2 against 0.05 in 4-D (256x the
volume), quadrotor3D 0.3 against 0.05 in 12-D. Predicted `p_success` is inflated and there is a
floor under achievable KL. Every arm shares the bias, so rankings should survive, but magnitudes
are not comparable across runs scored at different radii.

---

## 5. How the method is structured

### The loop

One epoch is an acquisition round, not a gradient epoch (`adaptive_roa/adaptive_v2/engine.py`):

1. Train the predictor on the current training set.
2. Score `n_candidates` pool states (50,000 by default, raised to 250,000 on quadrotor3D).
3. Select `samples_per_epoch` of them and reveal their rollouts.
4. Add them to the training set, rebuild the dataset files.
5. Evaluate on the full ROA grid and write `artifacts_v2.json`.

Arms differ only in step 3. Same pool, same seed, same budget, same evaluation.

### The d1 / d2 split

`acquisition.d2_ratio` splits each epoch's budget (`engine.py:212`): `n_d1` states drawn
sequentially from the fixed shuffle, `n_d2` chosen by the arm's score. `epi_bald` runs at
`d2_ratio = 1.0`, fully scored. The `_anch` variants run at 0.5. Because d1 reads sequentially
from a fixed file, every pure-uniform arm draws a byte-identical index set regardless of seed, so
the three-seed control band measures training stochasticity only.

### Producing per-member probabilities

`adaptive_roa/adaptive_v2/probability/ensemble_prob.py` adds `estimate_members(states) -> [M, N]`
alongside the existing `estimate()`, which still returns the ensemble marginal so that threshold,
calibration and evaluation code is untouched.

- `EnsembleClassifierProbabilityBackend` (line 49) runs `forward_all_members` and a sigmoid. One
  exact forward pass per member, `member_sample_size = None`.
- `EnsembleEndpointMCProbabilityBackend` (line 68) loops M members and K=20 endpoint samples
  each, labelling every sample with `system.classify_attractor(pred, attractor_radius)`. That is
  `M · K` integrations per candidate batch, and it is why acquisition on the FM side is
  expensive.

`bind_model` refuses an ensemble with fewer than 2 members, because a one-member ensemble would
silently score 0 everywhere instead of failing.

### Scoring and selection

`DecompositionAcquisitionStrategy.select` reads `estimate_members`, calls `score_by_mode`, then
hands the scores to `select_greedy_diverse` (`strategy/dispersion_score.py:141`): take the top
`5 × n_select` by score, then farthest-point-sample `n_select` from that shortlist in
initial-state space, seeded at the highest scorer, with per-dimension normalisation and
circular-aware distance. Without this, a batch collapses into one high-uncertainty pocket.

The strategy records both components on every arm, not just the one being scored, so the
diagnostics show why an arm chose what it did.

**The arm is threshold-free.** `threshold_state` is passed in and deliberately unread, and the
engine skips fitting `lambda*` and `delta*` altogether in epochs where nothing else needs them
(`engine.py:225`). The training loop therefore never depends on a fitted decision boundary.

**Out-of-domain input raises rather than returning NaN.** `select_greedy` skips non-finite
scores, so a silent NaN would quietly drop candidates and let a partially non-adaptive arm report
a full run. `_validate_probabilities` checks NaN explicitly before the range check, since
`np.nanmin` would pass NaN straight through.

### The ensemble itself

`EnsembleFlowMatchingTrainer` spawns M processes with `torch.multiprocessing`, member m
round-robined onto device `m % n_visible_gpus`, each writing `checkpoints/member_{m}/`. No
communication. With one member per GPU the wall clock equals a single member, which is the whole
point at roughly 50h per member. The classifier ensemble trains members sequentially inside
`BayesianMLPTrainer`; an MLP member is cheap enough that this does not matter.

`EnsembleFlowMatcherHandle.predict_endpoint` round-robins members exactly rather than sampling
one at random, and `full_roa.py:805` pins the eval member by `sample_idx % n_members` for the
same reason: a seeded generator sampling members produced weights of
[.125, .281, .109, .234, .250] against an exact .2, enough to flip decisions near `lambda*`.

### Running it

```bash
python scripts/run_adaptive.py \
    system=cartpole_stoch noise_level=med \
    predictor=fm_ensemble acquisition=decomp_epi_bald \
    predictor.lightning_trainer.enable_progress_bar=false
```

For the classifier, swap `predictor=clf_ensemble` and note that its config has no
`lightning_trainer` key, so the progress-bar override needs the append form
(`+predictor.lightning_trainer.enable_progress_bar=false`). Getting this wrong has killed jobs.

K is 20 at acquisition and 100 at evaluation. M is 5 everywhere.

---

## 6. Arms and controls

| arm | acquisition config | d2_ratio | role |
|---|---|---|---|
| `dir00` × seeds 42/43/44 | `direct` | 0 | uniform control, defines the run-to-run floor |
| `total` | `decomp_total` | 1.0 | the method BALD is meant to replace |
| `epi_bald` | `decomp_epi_bald` | 1.0 | naive mutual information |
| `epi_var` | `decomp_epi_var` | 1.0 | MC-debiased variance, the family default |
| `aleat` | `decomp_aleat` | 1.0 | negative control |
| `*_anch` | same, `d2_ratio=0.5` | 0.5 | half uniform, half scored |
| `yield_*` | `decomp_epi_var_yield`, `decomp_yield_knn`, `decomp_yield_mlp` | 1.0 | `score · E[L]^α`, reduces to `epi_var` at α=0 |
| `partx` | `partx` with the `gp` predictor | 1.0 | a different model class, not an acquisition variant |

`aleat` is the load-bearing control. It targets the component that by construction cannot be
reduced by more data, so it should be the worst arm at high noise. If `epi_bald` beats `total`
while `aleat` loses to `total`, the split is real rather than a relabelling. Ensemble versions of
both the uniform and total-entropy arms are required because an ensemble marginal is better
calibrated than a single model purely from averaging.

The standard of evidence used throughout the campaigns: a gap counts only if it exceeds 2 SD of
three genuinely distinct seeds of the same configuration, pooled across epochs, at two
consecutive epochs, with the sign stable over the full trajectory.

---

## 7. What has been measured

Results split cleanly by predictor family, and cross-family comparisons are not valid.

**Flow matching, pendulum, noisy family (the original campaign).** Null at every noise level. All
four scores are numerically interchangeable, and where a persistent offset appears the arms track
each other to within 5e-05. The single statistic for this: the spread among the four acquisition
arms divided by the spread among three seeds of one config sits between 0.5 and 1.9 in six of
eight predictor by level cells. Choosing an acquisition score there matters about as much as
changing the seed.

**Classifier, pendulum, high and xhigh noise.** The two cells where acquisition genuinely changes
the outcome, at 6.9x and 84.6x the seed spread. At xhigh, `total` and `aleat` are distinguishably
harmful on all seven metrics (+29x to +45x the floor) while `epi_bald` cuts that by roughly 10x
and ties uniform sampling on the calibration-free metrics. The mechanism is visible in what each
arm buys: `epi_bald` acquires genuinely ambiguous states 51 to 68% of the time, `epi_var` 26 to
31%, and `total` and `aleat` 2 to 4%. Splitting the Murphy decomposition at epoch 9 shows all
four arms damaging calibration while only `total` and `aleat` also lose resolution.

That classifier result carries a caveat the campaign itself insists on. At high noise this
classifier is miscalibrated badly enough (7.7x worse Brier than flow matching before any
acquisition) that every score lands on states whose true `p` is about 0.02. "BALD beats total
entropy here" is a statement about this predictor, not about BALD as a method.

**Flow matching, gaussian_signal pendulum.** `epi_bald` wins on KL at low (.0079 against a
control of .0211, floor .0081), does not win at med, and high is a flat null.

**Flow matching, cartpole v2.** `epi_bald` clears the floor on KL, debiased Brier, sAUROC and
recal at all three noise levels, with epoch-11 KL of .0162 / .0151 / .0137 against controls of
.0313 / .0375 / .0393. A 2.3x to 2.9x KL reduction over uniform, and the opposite of the pendulum
result at the same nominal noise levels.

**Flow matching, quadrotor2D.** On `noisy_dynamics f_0.150`, `epi_bald` beats the floor on all
three metrics at epoch 23 (KL .0262 against a control of .0647, floor .0184), ranked fourth of
five FM arms, though the whole FM spread of .0078 sits inside the floor so no FM arm is
distinguishable from another. On `corridor_sine_ambient`, `epi_bald` is the second best arm and
again beats the floor on all three metrics.

**Quadrotor3D.** Still filling in. The `f_0.048` level was relaunched on 2026-08-22 with the
10k/5k/18 budget after the start-state fix and is too shallow to rank. Epoch-0 agreement to four
decimal places across every seed-42 arm confirms the fix did not perturb training.

Read together: BALD is a null on the pendulum, a clear win on cartpole and quadrotor2D, and
indistinguishable from `epi_var` and the yield arms wherever the family wins as a whole. The
consistent finding is that adaptive-versus-uniform separates on several systems while the choice
among epistemic scores mostly does not.

---

## 8. Traps worth knowing before running or reading one of these

- Cross-family comparisons are invalid. FM arms are judged against the three-seed FM band,
  classifier arms against `clf_dir00`, Part-X against neither. There is no classifier floor,
  since only one seed exists per classifier arm.
- Two consecutive epochs is not sufficient protection on its own. The metric oscillates on that
  timescale, so two adjacent samples can agree by luck. `FINDINGS.md` records four different
  headline claims for one level across six scoring passes while the effect sizes never moved.
  Quote effect sizes and floor depth, never a bare verdict.
- Never use the artifact-level `auc` and `brier` fields (§4).
- `post-recal` in these documents is `UNC − RES` from the Murphy decomposition. No Platt scaler,
  isotonic fit or temperature was ever fitted.
- Labels are `{−1, +1}`, so `.astype(bool)` on them is always wrong.
- `partx` needs `predictor=gp`, not `gp_reg`. The regressor has no `latent_posterior` and dies in
  under 80 seconds.

---

## 9. File map

| what | where |
|---|---|
| the four scores | `adaptive_roa/adaptive_v2/strategy/uncertainty_scores.py` |
| the acquisition strategy | `adaptive_roa/adaptive_v2/strategy/decomposition.py` |
| per-member probabilities | `adaptive_roa/adaptive_v2/probability/ensemble_prob.py` |
| ensemble FM training | `adaptive_roa/adaptive_v2/trainers/ensemble_flow_matching_trainer.py` |
| selection rule | `adaptive_roa/adaptive_v2/strategy/dispersion_score.py:141` |
| the epoch loop | `adaptive_roa/adaptive_v2/engine.py` |
| full-ROA evaluation | `adaptive_roa/adaptive_v2/eval/full_roa.py` |
| pool reader | `adaptive_roa/adaptive/npz_data_source.py` |
| arm config | `configs/adaptive_v2/acquisition/decomp_epi_bald.yaml` |
| predictor configs | `configs/adaptive_v2/predictor/{fm_ensemble,clf_ensemble}.yaml` |
| system configs | `configs/adaptive_v2/system/{pendulum,cartpole,quadrotor2d,quadrotor3d}_stoch.yaml` |
| scoring | `scripts/stoch_prob_metrics.py` |
| tests | `tests/adaptive_v2/test_uncertainty_scores.py`, `test_decomposition_strategy.py`, `test_ensemble_fm_handle.py` |
