# Adaptive ROA Method Deep Dive

This document is an implementation-faithful reference for the adaptive ROA pipeline, with emphasis on:

1. threshold optimization (\(\lambda^\*,\delta^\*\)),
2. adaptive sampling (D1/D2 with `direct` / `conformal` / `ranked`),
3. post-hoc `scripts/reevaluate.py` recalibration and formal conformal guarantees.

Primary code paths:

- `scripts/run_adaptive.py`
- `adaptive_roa/adaptive_v2/engine.py`
- `adaptive_roa/conformal/*.py`
- `adaptive_roa/adaptive/balanced_sampler.py`
- `adaptive_roa/adaptive_v2/eval/full_roa.py`
- `scripts/reevaluate.py`

---

## 1) Problem Statement

Given a start state \(x_0\), estimate terminal outcome class:

- \(+1\): success (in attractor basin),
- \(-1\): failure (constraint/termination failure basin),
- \(0\): separatrix/invalid/unknown (between confident success and failure).

Goal: improve ROA boundary quality with limited trajectory budget by prioritizing informative boundary-near points.

The method uses:

- stochastic endpoint modeling (flow matching),
- MC probability estimation \((\hat p_s,\hat p_f,\hat p_i)\),
- selective threshold classifier \((\lambda,\delta)\),
- optional conformal set calibration \(q_{\hat{}}\),
- adaptive pool sampling near uncertainty regions.

---

## 2) Notation

| Symbol | Meaning |
|---|---|
| \(x\) | initial/start state |
| \(X_T\) | random terminal state under learned endpoint model |
| \(h(X_T)\) | endpoint class in \(\{-1,0,+1\}\) |
| \(\hat p_s,\hat p_f,\hat p_i\) | estimated success/failure/invalid probabilities |
| \(\lambda\) | center threshold |
| \(\delta\) | abstention half-width |
| \(\lambda^\*,\delta^\*\) | optimized thresholds |
| \(s(x,y)\) | nonconformity score for candidate label \(y\) |
| \(q_{\hat{}}\) | calibrated conformal quantile |
| \(\Gamma(x)\) | conformal prediction set |
| \(\alpha\) | miscoverage level |
| \(K\) | MC samples per state |
| D1 | direct sampled trajectories in epoch |
| D2 | adaptively sampled trajectories in epoch |

---

## 3) Pipeline Overview (Per Epoch)

`AdaptiveEngine.run()` executes:

1. Train flow-matching endpoint predictor on current dataset.
2. Estimate MC probabilities on validation data.
3. Optimize \((\lambda^\*,\delta^\*)\) from validation data.
4. Sample D1 trajectories from pool and add them to training.
5. If conformal mode: calibrate \(q_{\hat{}}\) on D1.
6. Acquire D2 trajectories with selected strategy (`direct`, `conformal`, `ranked`).
7. Add retained D2 to training, rebuild datasets.
8. Optionally filter newly-added training pairs to keep uncertain/misclassified points.
9. Evaluate full ROA on held-out test data and save epoch artifacts.

---

## 4) Data Representation and Splits

### 4.1 Trajectory-to-endpoint-pair conversion

For selected trajectory \(\tau_j=(x_{j,0},\dots,x_{j,T_j})\), training pairs are:

\[
(x_{j,t}, x_{j,T_j}), \quad t=0,\dots,T_j-1.
\]

So one trajectory yields many endpoint supervision pairs.

### 4.2 Pool state management

Key pool mechanics:

- indices in `used_indices` are no longer available,
- D1/D2 are sampled at trajectory-index level,
- only retained candidates become used,
- discarded certain candidates remain available for later epochs.

### 4.3 Distinct dataset roles

Three different data roles are used and should be reported separately:

- train/val splits from selected pool trajectories (model fitting + threshold optimization),
- held-out calibration file `cal_set_file` (evaluation-time conformal calibration),
- held-out test file `test_set_file` (final metric reporting, including coverage).

---

## 5) Probabilistic Endpoint Estimation

For each start state \(x\), the model samples \(K\) terminal states:

\[
\hat x_T^{(k)} \sim p_\theta(x_T \mid x), \; k=1,\dots,K
\]

and classifies each with system-specific `classify_attractor`:

\[
h(\hat x_T^{(k)})\in\{-1,0,+1\}.
\]

Empirical probabilities:

\[
\hat p_s(x)=\frac{1}{K}\sum_k \mathbf{1}[h(\hat x_T^{(k)})=+1], \;
\hat p_f(x)=\frac{1}{K}\sum_k \mathbf{1}[h(\hat x_T^{(k)})=-1], \;
\hat p_i(x)=\frac{1}{K}\sum_k \mathbf{1}[h(\hat x_T^{(k)})=0].
\]

These probabilities feed threshold optimization, conformal scoring, and adaptive selection.

### 5.1 Optional invalid-endpoint refinement

When enabled (`refine_invalids`), invalid sampled endpoints can be re-integrated from random late time \(t_{\text{start}}\in[t_{\min},t_{\max}]\), up to `refine_max_attempts`.

This changes class counts, therefore changing \((\hat p_s,\hat p_f,\hat p_i)\), thresholds, and conformal calibration.

---

## 6) Threshold Optimization in Detail

Threshold optimization is done by `ConformalPredictor.optimize_thresholds()` with configurable objective/mode.

### 6.1 Decision rules used in optimization

One-sided:

- success if \(\hat p_s > \lambda+\delta\),
- failure if \(\hat p_s < \lambda-\delta\),
- unknown otherwise.

Two-sided:

- success if \(\hat p_s > \lambda+\delta\),
- failure if \((1-\hat p_f)<\lambda-\delta\),
- unknown otherwise.

Optional invalid veto:

- mark unknown if \(\hat p_i \ge \lambda-\delta\).

### 6.2 Loss-based objective (default)

\[
\mathcal{L}(\lambda,\delta)=
w\cdot \text{MisclassRate}_{\text{confident}}+
(1-w)\cdot \text{UnknownRate}.
\]

Implementation conventions:

- misclassification is computed only on confident (non-unknown) points,
- if no confident points exist for a candidate threshold, misclass rate is treated as \(0\),
- unknown rate is over all evaluation points.

### 6.3 Optimization modes

- `optimize_mode=lambda`: grid-search \(\lambda\), fixed \(\delta\),
- `optimize_mode=delta`: grid-search \(\delta\), fixed \(\lambda=0.5\),
- `optimize_mode=joint`: 2D grid over \((\lambda,\delta)\), valid combinations only.

### 6.4 F1-based objective mode

If `optimize_objective=f1`:

- grid-search \((\lambda,\delta)\) for target F1,
- among feasible candidates choose minimal separatrix percentage,
- if target F1 is unattainable, choose best available candidate.

### 6.5 Fixed threshold mode

If `threshold_mode=fixed`, optimization is skipped:

\[
\lambda^\*=\texttt{fixed\_lambda\_star},\quad
\delta^\*=\texttt{fixed\_delta\_star}.
\]

### 6.6 Data used for threshold fitting

In v2, thresholds are tuned on the val subset from currently selected trajectories, while FM training uses train subset.
This separation reduces direct threshold-overfit to FM train pairs.

---

## 7) Conformal Calibration and Prediction Sets

Given fixed \((\lambda^\*,\delta^\*)\), calibrate \(q_{\hat{}}\) on labeled calibration examples.

### 7.1 Nonconformity scores

Score \(s(x,y)\) measures incompatibility of candidate label \(y\) with estimated probabilities.

Implementation behavior:

- one-sided scores use band distances in \(\hat p_s\),
- two-sided scores use \(\hat p_s,\hat p_f\),
- in current two-sided implementation, success/failure candidate scores are single-margin forms; unknown uses max-style combined violation.

### 7.2 Calibration quantile

From calibration scores \(s_i=s(x_i,y_i)\):

\[
q_{\hat{}}=
\operatorname{Quantile}_{\min(1,(1-\alpha)(n+1)/n)}(\{s_i\}_{i=1}^n).
\]

### 7.3 Prediction set

\[
\Gamma(x)=\{y\in\{-1,0,+1\}: s(x,y)\le q_{\hat{}}\}.
\]

Interpretation:

- \(\{+1\}\): confident success,
- \(\{-1\}\): confident failure,
- \(\{0\}\): confident invalid/separatrix,
- multi-label (or empty): uncertain.

### 7.4 What coverage means

Coverage concerns set containment:

\[
Y\in\Gamma(X),
\]

not singleton accuracy.
Thus high coverage can coexist with high abstention/set size.

---

## 8) Adaptive Sampling in Detail

### 8.1 D1/D2 split

With `samples_per_epoch = N` and `d2_ratio = r`:

\[
N_{D1}=\lfloor N(1-r)\rfloor,\quad
N_{D2}=N-N_{D1}.
\]

- D1 points are directly sampled from available pool and added to train.
- D2 points are selected via strategy from remaining pool.

### 8.2 Why D1 exists

D1 serves both:

- exploration (non-adaptive baseline component),
- conformal calibration data for D2 in `sampling_mode=conformal`.

### 8.3 D2 strategy A: `conformal`

Procedure:

1. Calibrate \(q_{\hat{}}\) on D1 labels.
2. Batch-sample candidate states from pool.
3. Build prediction sets \(\Gamma(x)\).
4. Retain candidates with:
   - \(|\Gamma(x)|>1\), or
   - \(\Gamma(x)=\{0\}\).
5. Discard singleton confident \(\{+1\},\{-1\}\).
6. Continue until D2 target reached or pool exhausted.

Diagnostics tracked include:

- evaluated candidates,
- certain discarded,
- invalid added.

### 8.4 D2 strategy B: `direct`

No conformal quantile.
Use only threshold-band confidence from \((\lambda^\*,\delta^\*)\):

- retain uncertain band points,
- discard certain points.

### 8.5 D2 strategy C: `ranked`

No conformal quantile.

1. Sample many candidates (`n_ranked_candidates`).
2. Score each with unknown-label nonconformity \(s(x,0)\).
3. Select the lowest-score `target_count`.

This is a boundary-focused margin ranking approach.

### 8.6 Candidate lifecycle behavior

- within one acquisition call, already-sampled indices are excluded,
- discarded certain points remain in future pool,
- retained points are marked used and cannot be resampled.

### 8.7 Optional confidence filtering of training pairs

If enabled:

- keep all old train rows,
- among newly-added rows, keep only uncertain or confidently-misclassified rows.

This shifts optimization toward hard boundary examples.

---

## 9) Evaluation in Training vs Post-Hoc Re-evaluation

There are two conformal thresholds in practice:

- `q_hat` for training-time D2 acquisition,
- `q_hat_eval` for held-out evaluation reporting.

In training, `AdaptiveEngine` computes `q_hat_eval` from `cal_set_file` and evaluates on `test_set_file`.

`scripts/reevaluate.py` repeats that process post hoc for saved epoch checkpoints, with potentially different eval settings.

---

## 10) `scripts/reevaluate.py` in Detail

For each epoch:

1. Load epoch checkpoint.
2. Load \((\lambda^\*,\delta^\*)\) from epoch artifact.
3. Estimate MC probabilities on held-out calibration set.
4. Recompute \(q_{\hat{}}\) using requested `alpha_eval`.
5. Evaluate held-out test set with recomputed \(q_{\hat{}}\).
6. Save results under `evaluations/<eval_name>/epoch_xxx/artifacts_v2.json`.

### 10.1 Important defaults/options

- If no eval parameters are passed and no `--force`, script does not re-run evaluation and points to stored metrics.
- By default, it filters calibration points by conformal invalid threshold:
  \[
  \hat p_i < \lambda^\*-\delta^\*.
  \]
- `--no_filter_invalid_by_conformal` disables that default.
- `--invalid_threshold` allows explicit alternative filtering.
- If no calibration points remain after filtering, that epoch is skipped.
- `--refine_invalids` toggles endpoint refinement and must match desired evaluation protocol.

---

## 11) Formal Guarantee Interpretation

### 11.1 Split-conformal statement

Standard split-conformal target:

\[
\Pr(Y\in\Gamma(X))\ge 1-\alpha.
\]

In this pipeline, this corresponds to coverage in:

- `full_roa["qhat_prediction_sets"]["coverage"]`.

### 11.2 Conditions to state explicitly

For the standard statement, report assumptions:

- calibration and test samples are exchangeable,
- calibration set is held out from model fitting of \(q_{\hat{}}\),
- same scoring protocol is used during calibration and prediction.

### 11.3 Practical caveats in this implementation

1. Probabilities are finite-\(K\) MC estimates, not exact.
2. Optional calibration filtering by \(\hat p_i\) is data-dependent and should be disclosed.
3. Optional invalid refinement modifies score distribution; recalibration is required under the same refinement settings.
4. \((\lambda^\*,\delta^\*)\) are tuned before conformal calibration; calibration split must remain disjoint for clean interpretation.

### 11.4 Metric family distinction

The evaluator reports multiple families:

- `lambda_delta`: threshold point predictions,
- `qhat_prediction_sets`: conformal set predictions,
- `fixed_threshold`, `lambda_only`: baselines.

Formal conformal guarantees should be tied to `qhat_prediction_sets` coverage, not to `lambda_delta` metrics.

---

## 12) Suggested Paper Wording (Guarantee Paragraph)

Use wording close to:

"We calibrate a split-conformal nonconformity threshold \(q_{\hat{}}\) on a held-out calibration set and report set-coverage on a disjoint held-out test set. Under exchangeability between calibration and test samples, the resulting prediction sets satisfy the finite-sample marginal guarantee \(\Pr(Y\in\Gamma(X))\ge 1-\alpha\). Since nonconformity scores are computed from Monte Carlo endpoint probability estimates, this guarantee is interpreted for the implemented randomized predictor."

If invalid filtering is enabled, add:

"Calibration examples with high estimated invalid probability are excluded before quantile calibration; guarantees are therefore interpreted with respect to this calibration protocol."

---

## 13) Reproducibility Checklist

Report at minimum:

- system and decision rule,
- threshold mode (`dynamic` or `fixed`),
- optimization objective (`loss` or `f1`) and optimize mode (`lambda`/`delta`/`joint`),
- \(w\), lambda/delta grid ranges and grid sizes,
- D1/D2 settings (`samples_per_epoch`, `d2_ratio`, `sampling_mode`),
- whether confidence filtering is enabled,
- MC sample counts for thresholding and evaluation,
- attractor radius,
- \(\alpha_{\text{sampling}}\) and \(\alpha_{\text{eval}}\),
- reevaluation flags: invalid filtering on/off, refinement on/off and its parameters.

---

## 14) One-Line Mental Model

Train a stochastic endpoint predictor, tune selective thresholds, actively add uncertain boundary trajectories, then recalibrate conformal prediction sets on held-out data for finite-sample coverage reporting.
