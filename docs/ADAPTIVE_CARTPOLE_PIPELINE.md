# Adaptive Sampling Pipeline (CartPole PyBullet)

This report describes the **high-level pipeline** implemented by `scripts/run_adaptive_cartpole.py`: an iterative loop that (1) trains a stochastic **flow-matching endpoint model**, then (2) uses **Monte Carlo probability estimation + conformal prediction** to identify *uncertain* start states (near the separatrix) and add their trajectories to the training set.

It is written to help you reason about the system as a data/compute pipeline: **what goes in, what happens at each stage, and what artifacts come out**.

Related (more general) background: `docs/CONFORMAL_ADAPTIVE_SAMPLING.md`.

---

## 0) One-paragraph view

You start with a large, pre-collected pool of CartPole trajectories on disk. Each trajectory has a binary “success/failure” label (and the dynamics may also admit a “separatrix/invalid” outcome when you classify endpoints). The script maintains an ever-growing training subset of these trajectories. At each epoch it:

1. **Builds an endpoint dataset** (many `(start_state → end_state)` pairs per trajectory) from the current training subset.
2. **Trains a flow matcher** to model the conditional distribution of endpoints given the start state (stochastic via latent/noise).
3. **Runs MC sampling** through the flow matcher to estimate `p_success(x)`, `p_failure(x)`, `p_invalid(x)` for many candidate start states.
4. **Optimizes thresholds** `(λ*, δ*)` to trade off wrong predictions vs abstaining (“unknown”).
5. **Calibrates conformal q̂** (`q_hat`) on a fresh per-epoch calibration set (D1) to get prediction sets with coverage guarantees.
6. **Selects new training trajectories** (D2) concentrated in uncertain regions and repeats.

Separately, it evaluates on a held-out test set each epoch and writes rich metrics + plots to the run directory.

---

## 1) Core concepts and vocabulary

### 1.1 Data primitives

- **Trajectory**: a time series of states saved as a text file in `data_source.trajectories_dir`. A trajectory index `i` maps to a specific filename via `data_source.shuffled_indices_file`.
- **Start state**: the first state in a trajectory.
- **End state**: the final state in a trajectory.
- **Label conventions**:
  - Training pool labels (`shuffled_labels_file`) are stored as `{0, 1}` and mapped to internal `{-1, 1}` via `TrajectoryDataSourceConfig.label_mapping`:
    - `1 → 1` (SUCCESS)
    - `0 → -1` (FAILURE)
  - During MC endpoint classification, the system may also output `0` for “invalid/separatrix/unknown” depending on the endpoint classifier (`system.classify_attractor`).

### 1.2 Model primitive: flow matcher as a stochastic endpoint predictor

The trained flow matcher provides a callable:

```
end_state_sample = flow_matcher.predict_endpoint(start_state)
```

Because inference is stochastic (latent/noise), repeatedly calling `predict_endpoint()` for the same start state yields a distribution of possible endpoints.

This stochasticity is **essential**: it enables MC estimation of success/failure probabilities for a start state.

### 1.3 Probability estimation via MC + endpoint classification

For a start state `x`, we draw `K` endpoint samples:

```
x_T^(1), x_T^(2), ..., x_T^(K) ~ flow_matcher(x)
```

Each endpoint is classified into `{SUCCESS=1, FAILURE=-1, INVALID=0}` using:

```
label = system.classify_attractor(endpoint, radius=attractor_radius)
```

From those labels the pipeline computes:

- `p_success(x)  = #(label==1)  / K`
- `p_failure(x)  = #(label==-1) / K`
- `p_invalid(x)  = #(label==0)  / K`

Implementation: `adaptive_roa/conformal/probability_estimator.py`.

### 1.4 Thresholds (λ, δ) and decision rules

The pipeline tries to produce **confident** decisions (success/failure) while allowing an abstention region (“unknown”) near the boundary.

For CartPole, the default is **two-sided** decision logic (uses both `p_success` and `p_failure`), matching `conformal.decision_rule: two_sided` in `configs/adaptive_cartpole_pybullet.yaml`.

Two-sided thresholds (conceptually):

- CONFIDENT SUCCESS if `p_success > λ + δ`
- CONFIDENT FAILURE if `p_failure > 1 - (λ - δ)` (equivalently `(1 - p_failure) < λ - δ`)
- UNKNOWN otherwise

Implementation: `adaptive_roa/conformal/lambda_optimizer.py` (`apply_two_sided_rule()`).

### 1.5 Conformal prediction (q̂) and prediction sets

Instead of treating `(λ, δ)` as *the* final decision rule, conformal prediction builds **prediction sets** `C(x) ⊆ {-1, 0, 1}` such that the true label is contained with high probability (coverage).

The calibrator:

1. Defines a **nonconformity score** `s(x, y)` (how “strange” it is to claim label `y` at point `x` given the estimated probabilities).
2. Computes scores on a labeled calibration set and chooses `q_hat` as a quantile so that:
   - `P(true_label ∈ C(x)) ≥ 1 - α`
3. Defines:
   - `C(x) = { y : s(x, y) ≤ q_hat }`

Implementation: `adaptive_roa/conformal/calibrator.py`.

### 1.6 D1 vs D2

Per epoch, the script splits new data acquisition into:

- **D1 (calibration set)**: fresh labeled points used to calibrate `q_hat` for that epoch.
- **D2 (uncertain set)**: additional points selected to be “maximally informative” (uncertain / near separatrix).

Sizes are controlled by:

- `samples_per_epoch`: total D1 + D2 to add each epoch
- `d2_ratio`: fraction allocated to D2

---

## 2) Key modules (what each subsystem is responsible for)

### 2.1 `TrajectoryDataSource`: “index → trajectory file → start/end/label”

File: `adaptive_roa/adaptive/data_source.py`

Responsibilities:

- Load the shuffled filename list (`shuffled_indices_file`).
- Optionally load aligned labels (`shuffled_labels_file`).
- Load a trajectory by index.
- Extract the start state and end state.
- Build endpoint datasets and write them to disk.

Important behavior:

- For training datasets it expands each trajectory into **many** pairs `(state_t → final_state)` for all timesteps.
- For evaluation datasets it can optionally produce only `(state_0 → final_state)`.

### 2.2 `AdaptiveDatasetBuilder`: “which indices are in train / used?”

File: `adaptive_roa/adaptive/dataset_builder.py`

Responsibilities:

- Track `train_split` (list of indices in training).
- Track `used_indices` (set of indices consumed from the pool).
- Provide sampling from the available pool **without** marking indices as used (so you can evaluate candidates and only commit the ones you want).
- Build train/val/test endpoint dataset files from the current training split.

Important nuance:

- The “val” and “test” datasets used by Lightning training are **subsets of training indices** (with overlap). They are not held-out generalization sets; they are training diagnostics.

### 2.3 Flow matcher training: “fit p(endpoint | start)”

Entry: `train_flow_matcher()` in `scripts/run_adaptive_cartpole.py`.

Core pieces:

- `CartPoleEndpointDataModule` (`adaptive_roa/data/cartpole_endpoint_data.py`) reads 8-float lines: `[start(4) end(4)]`.
- The flow matcher implementation is instantiated via config:
  - `flow_matcher._target_ = adaptive_roa.flow_matching.cartpole.latent_conditional.flow_matcher.CartPoleLatentConditionalFlowMatcher`
  - It supports manifold-aware behavior (`use_manifold`) and uses Lightning for training.
- Checkpointing and early stopping are driven by `cfg.trainer.callbacks`.

Warm start behavior (`warm_start: true`):

- The script loads the previous epoch’s best checkpoint and copies **only model weights** (not optimizer state) into the new flow matcher before training.

### 2.4 `ConformalPredictor`: “threshold optimization + q̂ calibration”

File: `adaptive_roa/conformal/predictor.py`

The CartPole script uses the **two-phase** fitting interface:

1. `optimize_thresholds(X_train, y_train)` → sets `lambda_star`, `delta_star`
2. `calibrate_qhat(X_cal, y_cal)` → sets `q_hat`

This separation is deliberate:

- Threshold optimization uses the *current* training set.
- `q_hat` calibration uses **fresh per-epoch D1** points.

### 2.5 `UncertainSampler`: “find D2 points worth adding”

File: `adaptive_roa/adaptive/balanced_sampler.py`

In conformal mode, it repeatedly:

1. Pulls a batch of candidate start states from the available pool.
2. Computes `p_success/p_failure` via MC.
3. Constructs prediction sets `C(x)` using `(λ*, δ*, q_hat)`.
4. Keeps:
   - **uncertain** points: `|C(x)| > 1`
   - **invalid** points: `C(x) == {0}`
5. Discards:
   - **certain** points: `C(x) == {1}` or `C(x) == {-1}` (these remain in the pool for later epochs)

In non-conformal mode (`conformal.use_conformal: false`), it uses `(λ*, δ*)` directly and keeps points inside the “unknown band”.

---

## 3) End-to-end pipeline: from inputs to outputs

### 3.1 Inputs (what must exist before you run)

Configured in `configs/adaptive_cartpole_pybullet.yaml` under `data_source`:

1. **Trajectory pool**
   - `trajectories_dir`: directory of trajectory text files
   - `shuffled_indices_file`: line `i` gives the filename for trajectory `i`
   - `shuffled_labels_file`: line `i` gives the 0/1 label for trajectory `i`

2. **Held-out eval files** (CSV with start, end, label)
   - `cal_set_file`: held-out calibration set (for evaluation-time `q_hat_eval`)
   - `test_set_file`: held-out test set (for ROA evaluation)

3. **Environment paths**
   - `exp_dir` and `data_dir` are typically resolved from `.env` via custom OmegaConf resolvers.

### 3.2 Startup stage (once per run)

`scripts/run_adaptive_cartpole.py`:

1. Register resolvers (`exp_dir`, `data_dir`, etc.) before Hydra loads config.
2. `pl.seed_everything(seed, workers=True)` for reproducibility across Python/numpy/torch/DataLoader.
3. Create the run output directory `cfg.output_dir` (Hydra is configured to also use it as the run dir).
4. Initialize:
   - `TrajectoryDataSource`
   - `AdaptiveDatasetBuilder`
5. Create the initial training split:
   - `initial_train_size` indices starting from 0 in shuffled order
6. Build endpoint dataset files (`datasets/`):
   - `train_endpoint_dataset.txt`
   - `val_endpoint_dataset.txt`
   - `test_endpoint_dataset.txt`

### 3.3 The per-epoch loop (the heart of the pipeline)

Each epoch writes into `output_dir/epoch_###/`.

#### Step [1] Train the flow matcher

- Input: current training endpoint dataset files
- Output:
  - trained flow matcher model
  - Lightning logs + checkpoints in `epoch_###/checkpoints/`

Optional warm-start:
- if enabled and a prior best checkpoint exists, seed the new model’s weights from it.

#### Step [2] Optimize thresholds (λ*, δ*)

- Input: training trajectory start states + trajectory labels (`dataset_builder.get_train_labels()`)
- Compute: `p_success/p_failure` on training starts via MC
- Optimize:
  - `optimize_mode: lambda` → optimize `λ*` with fixed `δ`
  - `optimize_mode: delta` → optimize `δ*` with fixed `λ=0.5`
- Output: `lambda_star`, `delta_star`

If `conformal.threshold_mode: fixed`, this step is skipped and fixed values are used.

#### Step [3] Sample D1 (calibration points)

- Determine sizes:
  - `n_d1 = int(samples_per_epoch * (1 - d2_ratio))`
  - `n_d2 = samples_per_epoch - n_d1`
- Sample D1 indices from the remaining pool **sequentially** in shuffled order.
- Load true labels for those indices from `shuffled_labels_file`.
- Mark them used and add to training.

D1 is now part of the training set for the *next* epoch’s training, but it is used immediately (this epoch) to calibrate `q_hat`.

#### Step [4] Calibrate q_hat on D1 (optional)

If `conformal.use_conformal: true`:

- Use `(λ*, δ*)` from Step [2]
- Estimate probabilities on D1 (MC)
- Calibrate `q_hat` so prediction sets achieve `1 - α` coverage

If `use_conformal: false`, `q_hat` is not computed and the sampler uses `(λ*, δ*)` directly.

#### Step [5] Sample D2 (uncertain points)

- Repeatedly sample candidate batches from the remaining pool.
- For each batch, compute `p_success/p_failure`.
- Classify into:
  - uncertain (multi-label prediction set)
  - invalid (`{0}` prediction set)
  - certain (`{1}` or `{-1}`)
- Keep uncertain + invalid points until `n_d2` are found, or the pool is exhausted, or `max_samples_per_epoch` candidates have been evaluated.

Add the chosen D2 indices to training; do *not* mark discarded “certain” candidates as used (they stay available for later).

#### Step [6] Epoch sampling summary

The script prints counts for:

- D1 added
- D2 added (split into uncertain vs invalid)
- number of “certain” candidates discarded
- number of candidates evaluated during D2 search

#### Step [7] Quick evaluation on a training subset

- Uses `dataset_builder.get_test_labels()` which returns a slice of training indices (not held-out).
- Calls `conformal_predictor.evaluate()`, which measures:
  - coverage on that subset
  - confident rate vs unknown rate
  - precision/recall/f1 among confident predictions

Treat this as a training diagnostic, not a generalization result.

#### Step [8] Held-out evaluation (recommended metric)

This is the proper evaluation loop each epoch.

If conformal is enabled, it recalibrates a new **evaluation-time** `q_hat_eval`:

- Load `data_source.cal_set_file` (held-out)
- Estimate probabilities on it
- Calibrate `q_hat_eval` using the held-out calibration set

Then it evaluates on `data_source.test_set_file` (held-out) via `evaluate_full_roa_fast()`:

- Computes `p_success/p_failure/p_invalid` for each held-out start state using batched MC sampling.
- Computes metrics under three schemes:
  1. `λ*±δ*` thresholds (two-sided)
  2. “notebook-style” thresholds (`p_s > 0.6` / `p_f > 0.6`)
  3. conformal prediction sets using `q_hat_eval` (if available), including coverage and average set size
- Computes endpoint error statistics using the manifold geodesic distance, including hierarchical MC-error summaries.
- Writes JSON + NPZ + plots (details in §4).

#### Step [9] Rebuild datasets

Rewrites the endpoint dataset files from the updated training split, ready for the next epoch’s training.

### 3.4 End-of-run summary

After all epochs, the script:

- prints final training set size and remaining pool size
- prints a progression summary (initial → final) for key held-out metrics
- saves `final_results.json` aggregating epoch results

---

## 4) Outputs and artifacts (what you can inspect after a run)

All outputs live under `cfg.output_dir` (see `configs/adaptive_cartpole_pybullet.yaml`).

### 4.1 Directory layout (typical)

```
output_dir/
  .hydra/                          # Hydra config snapshots
  datasets/
    train_endpoint_dataset.txt
    val_endpoint_dataset.txt
    test_endpoint_dataset.txt
  dataset_builder_state.json       # indices + used set
  final_results.json               # run summary + epoch progression
  epoch_000/
    checkpoints/
      best-*.ckpt
      last.ckpt
    validation_errors.txt
    results.json                   # epoch summary (counts + metrics pointers)
    conformal_state.json           # λ*, δ*, q_hat and config summary
    full_roa_evaluation.json       # held-out metrics + error stats
    full_roa_evaluation_per_point.npz
    full_roa_evaluation_roa_projections.png
    full_roa_evaluation_roa_heatmap.png
  epoch_001/
    ...
```

### 4.2 What each artifact is for

- **`datasets/*.txt`**: the actual training data consumed by Lightning each epoch (endpoint pairs).
- **`epoch_###/checkpoints/`**: model checkpoints; used for warm-start and post-run analysis.
- **`epoch_###/results.json`**: per-epoch bookkeeping:
  - how many points added/discarded
  - `lambda_star`, `delta_star`, `q_hat` (training-time)
  - summary metrics on the training subset and held-out test set
- **`epoch_###/conformal_state.json`**: serialized conformal predictor state.
- **`epoch_###/full_roa_evaluation.json`**: held-out evaluation metrics, including:
  - threshold-based metrics (`λ±δ`, notebook thresholds)
  - q_hat conformal metrics (coverage, avg set size), if enabled
  - endpoint error summaries
- **`epoch_###/full_roa_evaluation_per_point.npz`**: per-point arrays for analysis:
  - `start_states`, `p_success`, `p_failure`, `p_invalid`, `true_labels`, etc.
- **`epoch_###/*png`**: ROA visualizations in 2D projections and probability heatmaps.

---

## 5) Key configuration knobs (how to think about tuning)

File: `configs/adaptive_cartpole_pybullet.yaml`

### 5.1 Compute / runtime knobs

- `conformal.num_mc_samples` and `conformal.num_mc_samples_eval`:
  - higher = lower variance probability estimates, but much slower
- `batch_size_sampling` and `val_batch_size`:
  - affects GPU memory and throughput
- `max_samples_per_epoch`:
  - safety cap on how many candidates you’ll evaluate while searching for D2
- `trainer.max_epochs`, early stopping patience:
  - training time per epoch for the flow matcher

### 5.2 “Where do we sample?” knobs

- `samples_per_epoch`: total new trajectories added each epoch
- `d2_ratio`: how aggressively you focus on uncertain sampling vs calibration
  - large `d2_ratio` → more uncertain points, fewer calibration points
  - small `d2_ratio` → stronger calibration each epoch, less exploitation
- `initial_train_size`: how much data you start with

### 5.3 “How do we classify uncertainty?” knobs

- `conformal.use_conformal`:
  - true: prediction sets with coverage guarantee
  - false: direct λ±δ band heuristic (no conformal guarantee)
- `conformal.alpha`:
  - smaller α → stronger coverage guarantee → larger prediction sets (more abstention)
- `conformal.w`:
  - larger w → prioritize avoiding wrong confident predictions
  - smaller w → tolerate errors to reduce unknown region
- `conformal.decision_rule`:
  - for CartPole, `two_sided` is the intended mode (uses `p_success` and `p_failure`)

### 5.4 Threshold optimization mode

- `conformal.threshold_mode`:
  - `dynamic`: run optimization each epoch
  - `fixed`: use `fixed_lambda_star` and `fixed_delta_star`
- `conformal.optimize_mode` (only in dynamic mode):
  - `lambda`: optimize λ* with fixed δ from config
  - `delta`: optimize δ* with fixed λ=0.5

---

## 6) How to run (and common Hydra overrides)

From the repo root:

```bash
python scripts/run_adaptive_cartpole.py
```

Useful overrides for quick experiments:

```bash
# Short debug run
python scripts/run_adaptive_cartpole.py n_epochs=1 samples_per_epoch=10 trainer.max_epochs=10 conformal.num_mc_samples=5 conformal.num_mc_samples_eval=5

# Disable conformal prediction (use λ±δ directly for uncertainty sampling)
python scripts/run_adaptive_cartpole.py conformal.use_conformal=false

# Use fixed thresholds (no optimization)
python scripts/run_adaptive_cartpole.py conformal.threshold_mode=fixed conformal.fixed_lambda_star=0.5 conformal.fixed_delta_star=0.1

# More aggressive uncertainty sampling
python scripts/run_adaptive_cartpole.py d2_ratio=0.8 samples_per_epoch=200
```

---

## 7) “What to trust” when interpreting results

- **Held-out metrics (Step [8]) are the main signal.** Step [7] uses a subset of training indices and is not a generalization test.
- There are *two* different q̂ values you might see:
  - `q_hat` (training-time): calibrated on D1 (which was sampled from the same pool used for training, but not used for threshold optimization)
  - `q_hat_eval` (evaluation-time): calibrated on `cal_set_file` (held-out), then used for evaluating on `test_set_file`
- The “invalid” category is defined as `p_invalid >= 0.5` (majority of MC endpoints were labeled invalid). This is a modeling + classifier artifact, not a label in the shuffled labels file.

---

## 8) Extending this pipeline to other systems (mental checklist)

To replicate the same pipeline for a new system you need:

1. A `system` with a reliable endpoint classifier:
   - `system.classify_attractor(end_states, radius=...) -> {-1,0,1}`
2. A flow matcher that can:
   - train on `(start_state, end_state)` pairs
   - sample endpoints stochastically at inference
3. A `TrajectoryDataSource` / dataset builder that:
   - knows how to read trajectories
   - writes endpoint datasets in a consistent format the model expects
4. A decision rule consistent with the domain:
   - `one_sided` if “failure is not success”
   - `two_sided` if success and failure should be treated distinctly (as with CartPole)

---

## 9) Code map (where to look)

- Pipeline entrypoint: `scripts/run_adaptive_cartpole.py`
- Data pool + endpoint dataset generation: `adaptive_roa/adaptive/data_source.py`
- Index bookkeeping + dataset building: `adaptive_roa/adaptive/dataset_builder.py`
- Uncertain sampling logic: `adaptive_roa/adaptive/balanced_sampler.py`
- Conformal prediction core: `adaptive_roa/conformal/`
- CartPole endpoint dataset reader: `adaptive_roa/data/cartpole_endpoint_data.py`
- CartPole flow matcher: `adaptive_roa/flow_matching/cartpole/latent_conditional/flow_matcher.py`



Each sample: p_success, p_failure, p_invalid


if (p_invalid >= 0.5):
    label = INVALID



