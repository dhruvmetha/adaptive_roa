# R:SS Writeup Notes (Code-Accurate): Problem Statement + Method

This document is a **code-driven** description of the *exact* problem and adaptive-sampling method implemented in this repository, written in a style that can be reused directly when drafting an R:SS submission.

It is grounded in the following implementation modules:
- **Adaptive sampling**: `src/adaptive/` (notably `run_adaptive_*.py`, `balanced_sampler.py`, `dataset_builder.py`, `pipeline.py`)
- **Conformal prediction + uncertainty**: `src/conformal/` (notably `predictor.py`, `probability_estimator.py`, `lambda_optimizer.py`, `calibrator.py`, `config.py`)
- **System-specific labeling**: `src/systems/*` (`classify_attractor`)

---

## Problem Statement (What is the task?)

### Setting
We study controlled dynamical systems where an initial state $x_0 \in \mathcal{X}$ generates a trajectory and terminates at an endpoint $x_T$. In this repository, a trajectory is treated as a sequence of states saved to disk (see `TrajectoryDataSource` in `src/adaptive/data_source.py`), but the conceptual object is:

$$
x_T = S(x_0) \quad \text{(expensive simulator / rollout)}
$$

We define a **task label** $y(x_0)$ from the terminal state using a system-specific classifier:
- Implemented as `system.classify_attractor(state, radius=attractor_radius)` in `src/systems/*`.
- Labels are in the *internal* convention:
  - $+1$: **SUCCESS** (reaches target attractor / goal)
  - $-1$: **FAILURE** (fails / reaches failure set / does not reach goal)
  - $0$: **SEPARATRIX / UNKNOWN** (used by some systems as an explicit third region)

Concrete examples in this codebase:
- **Pendulum** (`src/systems/pendulum.py`): $\{1,-1,0\}$ (success bottom attractor, failure top attractors, else separatrix).
- **CartPole** (`src/systems/cartpole.py`): $\{1,-1,0\}$ (success near upright equilibrium, failure if termination thresholds exceeded, else separatrix).
- **MountainCar** (`src/systems/mountain_car.py`): $\{1,-1\}$ (goal reached vs not reached).

### Goal
The core scientific goal is **data-efficient region-of-attraction (ROA) estimation / classification**:

> Learn a decision rule that maps an initial state $x_0$ to SUCCESS vs FAILURE (and optionally SEPARATRIX), while minimizing expensive simulation usage and providing a calibrated notion of uncertainty near the ROA boundary.

Operationally, the project uses a learned probabilistic “successor/end-point” model to estimate:

$$
p_{\text{success}}(x_0) \approx \mathbb{P}(y(x_0)=1 \mid x_0)
$$

and then uses **conformal prediction** to transform these probabilities into **set-valued predictions** with nominal coverage:

$$
\mathbb{P}\big(y(x_0)\in \Gamma(x_0)\big) \ge 1-\alpha.
$$

### Why adaptive sampling?
Simulating (or even just *including* trajectories in training when using a pre-collected pool) is costly. The key observation is:

- Most of the state space is “easy” (clearly success or clearly failure).
- The value of new data is highest near the ROA boundary / separatrix, where the learned model is uncertain.

So the repository implements **adaptive acquisition policies** that preferentially request/retain data where uncertainty is high, while skipping confident regions.

---

## Method (What does the code actually do?)

The method has four core components:
1. **A probabilistic successor model** $p_\theta(x_T \mid x)$ learned via *latent conditional flow matching* (the “flow matcher”).
2. **Monte Carlo probability estimation** of success/failure/separatrix using sampled endpoints.
3. **Conformal calibration** to obtain a data-dependent uncertainty threshold $q_{\hat{}}$ with coverage target $1-\alpha$.
4. **Adaptive sampling**: each epoch, add (or simulate) only uncertain points from a candidate pool, plus a small always-add set for calibration/training stability.

### 1) Probabilistic successor model: flow matcher $p_\theta(x_T \mid x)$
The code assumes a trained model `flow_matcher` that supports:
- `flow_matcher.predict_endpoint(states)` which **samples** endpoints (stochastic via an internal latent draw) for each input state.

The adaptive sampling logic treats the flow matcher as a black-box sampler from a conditional distribution of endpoints given a state.

Training data for the flow matcher is constructed as **endpoint pairs**:
- Implemented in `TrajectoryDataSource.build_endpoint_dataset()` (`src/adaptive/data_source.py`).
- For each trajectory index, in `"train"` mode the dataset contains pairs $(x_t, x_T)$ for all $t\in\{0,\dots,T-1\}$. I.e., *every state along the rollout* is paired with the terminal state.
- The dataset is written as a text file of concatenated vectors `[start_state... end_state...]` for Lightning datamodules (see `src/data/*_endpoint_data.py`, invoked from `src/adaptive/run_adaptive_*.py`).

### 2) Monte Carlo estimation of $p_{\text{success}}, p_{\text{failure}}, p_{\text{separatrix}}$
For a candidate state batch $X=\{x_i\}_{i=1}^N$, the repo estimates class probabilities by repeated latent sampling:
- Implemented in `ProbabilityEstimator.estimate()` (`src/conformal/probability_estimator.py`).

Given `config.num_mc_samples = K`, for each $x_i$ it draws $K$ endpoint samples:
$$
\hat{x}_{T,i}^{(k)} \sim p_\theta(\cdot \mid x_i), \qquad k=1,\dots,K
$$
classifies them:
$$
\hat{y}_{i}^{(k)} = \texttt{system.classify\_attractor}(\hat{x}_{T,i}^{(k)}, \texttt{radius}=\texttt{attractor\_radius})
$$
and forms empirical probabilities:
$$
p_{\text{success}}(x_i)=\frac{1}{K}\sum_k \mathbf{1}[\hat{y}_i^{(k)}=1],\quad
p_{\text{failure}}(x_i)=\frac{1}{K}\sum_k \mathbf{1}[\hat{y}_i^{(k)}=-1],\quad
p_{\text{separatrix}}(x_i)=\frac{1}{K}\sum_k \mathbf{1}[\hat{y}_i^{(k)}=0].
$$

### 3) Decision boundary parameters $\lambda$ and $\delta$
The method uses a *decision boundary* $\lambda$ and an *unknown-band half width* $\delta$. These appear explicitly in:
- `ConformalConfig.delta` and the fitted outputs `lambda_star`, `delta_star` in `ConformalPredictor` (`src/conformal/config.py`, `src/conformal/predictor.py`).

Two decision rules are implemented (config: `ConformalConfig.decision_rule`):

- **One-sided** (`decision_rule="one_sided"`, used for Pendulum/MountainCar in typical configs):
  - predict SUCCESS if $p_{\text{success}} > \lambda+\delta$
  - predict FAILURE if $p_{\text{success}} < \lambda-\delta$
  - else UNKNOWN
  - Implemented in `apply_one_sided_rule()` (`src/conformal/lambda_optimizer.py`)

- **Two-sided** (`decision_rule="two_sided"`, used for CartPole):
  - predict SUCCESS if $p_{\text{success}} > \lambda+\delta$
  - predict FAILURE if $(1 - p_{\text{failure}}) < \lambda-\delta$ (equivalently $p_{\text{failure}} > 1-\lambda+\delta$)
  - else UNKNOWN
  - Implemented in `apply_two_sided_rule()` (`src/conformal/lambda_optimizer.py`) and mirrored in the adaptive selection logic (`src/adaptive/balanced_sampler.py`).

The two-sided rule is important in CartPole because a low $p_{\text{success}}$ does not necessarily imply failure (it may reflect high separatrix probability).

### 4) Optimizing $\lambda^\star$ or $\delta^\star$ by grid search
The repository learns a decision rule each epoch by optimizing either:
- $\lambda^\star$ with fixed $\delta$, or
- $\delta^\star$ with fixed $\lambda=0.5$.

This is controlled by `ConformalConfig.optimize_mode ∈ {"lambda","delta"}`.

Implementation: `LambdaOptimizer.optimize()` (`src/conformal/lambda_optimizer.py`).

**Loss (exactly as coded)**:

$$
\text{Loss} = w \cdot \text{MisclassRate} + (1-w)\cdot \text{UnknownRate}
$$

where:
- `w = ConformalConfig.w`
- `UnknownRate` is the fraction of points predicted UNKNOWN under the chosen rule
- `MisclassRate` is computed **only among confident (non-UNKNOWN) predictions**.

Search spaces:
- If `optimize_mode="lambda"`: grid $\lambda \in [\delta,\,1-\delta]$ of size `lambda_grid_size`.
- If `optimize_mode="delta"`: grid $\delta \in [\texttt{delta\_min},\,\texttt{delta\_max}]$ of size `delta_grid_size`, with $\lambda$ fixed at 0.5.

The result of each fit is stored in:
- `ConformalPredictor.lambda_star`
- `ConformalPredictor.delta_star`

### 5) Conformal calibration to compute $q_{\hat{}}$ (coverage)
After choosing $\lambda^\star,\delta^\star$, the repo calibrates a threshold $q_{\hat{}}$ to target coverage $1-\alpha$:
- `alpha = ConformalConfig.alpha`
- Implemented in `Calibrator.calibrate()` (`src/conformal/calibrator.py`).

**Nonconformity scores**:
- One-sided and two-sided variants are implemented:
  - `nonconformity_score_one_sided`
  - `nonconformity_score_two_sided`

Given calibration data $\{(x_i, y_i)\}_{i=1}^n$ and their estimated probabilities, compute scores $s_i = s(p(x_i), y_i; \lambda^\star, \delta^\star)$, then set:

$$
q_{\hat{}} = \text{Quantile}\left(\{s_i\},\, (1-\alpha)\frac{n+1}{n}\right).
$$

**Prediction set**:
For a new point $x$, the conformal prediction set is:

$$
\Gamma(x) = \{ y \in \{-1,0,1\} \;:\; s(p(x), y; \lambda^\star,\delta^\star) \le q_{\hat{}} \}.
$$

Implementation: `Calibrator.get_prediction_set()` and `Calibrator.get_prediction_sets_batch()` (`src/conformal/calibrator.py`).

**What counts as “uncertain” in code?**
The default uncertainty query used by `ConformalPredictor.select_uncertain()` is:

> A point is “uncertain” if its conformal prediction set is not a singleton, i.e. $|\Gamma(x)| > 1$.

This is implemented via `Calibrator.get_uncertain_mask()` and used by `ConformalPredictor.select_uncertain()` (`src/conformal/predictor.py`).

### 6) Adaptive sampling policy (two implemented variants)

There are two adaptive sampling implementations in this repo; both share the D1/D2 idea.

#### Variant A: Online / “simulate-on-demand” adaptive loop
Implemented as `AdaptiveSamplingPipeline` (`src/adaptive/pipeline.py`) with:
- `StateSampler` sampling continuous states uniformly from `system.define_state_bounds()` (`src/adaptive/sampler.py`)
- a `Simulator` interface (`src/adaptive/simulator.py`) that can either:
  - actually simulate (`CallbackSimulator`), or
  - look up from a file/pool (`FileBasedSimulator` / `PoolSimulator`)
- a `DataManager` that accumulates $(x_0, x_T, y)$ across epochs (`src/adaptive/data_manager.py`)

Per epoch (code order in `AdaptiveSamplingPipeline.run_epoch()`):
- sample `n_samples_per_epoch` states
- split into:
  - `D1` of size `d1_ratio * n_samples_per_epoch` (always simulated)
  - `D2` (candidate pool)
- simulate D1 → add to dataset → retrain flow matcher
- fit conformal predictor using D1 as calibration
- score D2 and **simulate only uncertain** points (`ConformalPredictor.select_uncertain`)
- evaluate on a fixed test set and optionally stop when F1 improvement < `convergence_threshold`.

#### Variant B: Offline / “pre-collected trajectory pool” adaptive loop (main experiments)
Implemented in:
- `src/adaptive/run_adaptive_cartpole.py`
- `src/adaptive/run_adaptive_pendulum.py`
- `src/adaptive/run_adaptive_mountain_car.py`

Here we assume we already have a large pool of trajectories on disk. The “adaptive” aspect is: **which trajectories do we include for training** (proxy for “simulations used”) and how quickly does model quality improve as that training set grows.

Key data infrastructure:
- `TrajectoryDataSource` loads:
  - `shuffled_indices_file`: mapping from index → trajectory filename
  - labels from `shuffled_labels_file` (preferred) or `roa_labels_file` (legacy)
  - see `src/adaptive/data_source.py`
- `AdaptiveDatasetBuilder` maintains the set of training indices and writes endpoint datasets for flow matcher training (`src/adaptive/dataset_builder.py`).

Per epoch (common structure across run scripts):
- train flow matcher on current dataset (optionally warm-start from prior epoch; config `warm_start`)
- fit conformal predictor on current per-trajectory labels
- evaluate on a “test” subset (implemented as a subset of training indices in this repo) and optionally on a full ROA label file
- select which new trajectories to add according to one of two sampling strategies:

**(B1) `sampling_strategy="fixed"`**
- Draw `samples_per_epoch` candidate trajectories sequentially from the shuffled pool (`AdaptiveDatasetBuilder.get_candidate_states`).
- Split into D1 and D2 using `d2_ratio`:
  - D1 size `int(samples_per_epoch * (1 - d2_ratio))` → always added
  - D2 → score with `ConformalPredictor.select_uncertain()` (this uses `q_hat`) and add only uncertain indices.
- Note: in this strategy, even “skipped” candidates are consumed because the sequential pointer advances.

**(B2) `sampling_strategy="balanced_uncertain"`**
Implements a fixed per-epoch *add budget*:
- `adaptive_data_max`: total number of trajectories to add per epoch
- `d2_ratio`: fraction of that budget intended to be uncertainty-filtered (D2) vs always-add (D1)

Implementation: `BalancedUncertainSampler` (`src/adaptive/balanced_sampler.py`).

Key property: it repeatedly samples candidate batches until it finds enough uncertain points (or hits `max_samples_per_epoch`). If not enough uncertain points exist, it back-fills with “certain” points to keep total additions fixed.

Important nuance (code-accurate):
- This balanced sampler’s **uncertainty test is threshold-band based** using $\lambda^\star\pm\delta^\star$ (and optionally $p_{\text{failure}}$ for two-sided), *not* conformal `q_hat`.
- Conformal `q_hat` is still computed and logged (and can be used elsewhere), but balanced selection uses the $\lambda,\delta$ region test implemented in `_classify_candidates()`.

---

## What gets logged / evaluated (for plots and paper figures)

The conformal predictor exposes metrics (see `ConformalPredictor.evaluate()` in `src/conformal/predictor.py`):
- **coverage**: fraction where $y \in \Gamma(x)$ (target $1-\alpha$)
- **unknown_rate**: fraction with $|\Gamma(x)|>1$ (non-singleton sets)
- **f1 / precision / recall**: computed **only on singleton predictions** (abstention is treated as abstention)

The adaptive pool scripts also write per-epoch artifacts under `output_dir/epoch_XXX/` including:
- `results.json` (epoch summary)
- `conformal_state.json` (λ*, δ*, q_hat, config)
- `full_roa_evaluation.json` (optional, system-specific helper in `run_adaptive_*.py`)

---

## Pointers for your R:SS draft (what to emphasize)

If you want the submission to read cleanly, the key high-level claim supported by this code is:
- A learned *probabilistic successor model* enables ROA classification.
- Conformal prediction turns the model’s MC uncertainty into calibrated abstention/selection.
- Adaptive sampling reduces the number of expensive simulations (or the amount of training data used from a pool) by focusing on uncertain regions.

For strict “method = what we actually run”, be explicit about:
- whether selection uses **conformal sets (`q_hat`)** (`sampling_strategy="fixed"`) or **threshold-band (`λ*±δ*`)** (`sampling_strategy="balanced_uncertain"`),
- and whether the candidate pool is **consumed** (fixed pointer) or **retained** (balanced with `used_indices`).

---

## Open choices / assumptions to confirm (useful clarification for the paper)

These are choices you may want to fix explicitly in the paper draft (they are configurable in code):
- Which mode is the “main” method: online simulator (`AdaptiveSamplingPipeline`) vs offline pool (`run_adaptive_*.py`)?
- Whether “uncertainty” for *selection* is defined via conformal `q_hat` sets or via the $\lambda^\star\pm\delta^\star$ band.
- Whether the paper frames the cost as “simulator calls” or “trajectories used from a large precomputed dataset”.

