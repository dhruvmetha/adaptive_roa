# Conformal Prediction + Adaptive Sampling

This document describes the conformal prediction and adaptive sampling modules for efficient trajectory data collection in flow matching.

## Overview

**Goal**: Efficiently collect training data by focusing simulations on uncertain regions (near the separatrix) while maintaining statistical coverage guarantees.

**Key Components**:
- **Conformal Prediction**: Provides uncertainty quantification with coverage guarantees
- **Adaptive Sampling**: Uses uncertainty to focus simulations where they matter most

---

## Module Structure

```
adaptive_roa/
├── conformal/                      # Conformal prediction module
│   ├── __init__.py
│   ├── config.py                   # ConformalConfig dataclass
│   ├── probability_estimator.py    # MC sampling for p(success|x)
│   ├── lambda_optimizer.py         # Find optimal λ* or δ*
│   ├── calibrator.py               # Compute q_hat for coverage
│   └── predictor.py                # Main ConformalPredictor class
│
├── adaptive/                       # Adaptive sampling module
│   ├── __init__.py
│   ├── data_source.py              # TrajectoryDataSource: load trajectory pools
│   ├── dataset_builder.py          # AdaptiveDatasetBuilder: build datasets from indices
│   └── balanced_sampler.py         # UncertainSampler: q_hat-based uncertain sampling
│
scripts/
└── run_adaptive.py        # Main adaptive sampling script

configs/
└── adaptive_v2/system/cartpole_pybullet.yaml # CartPole PyBullet config
```

---

## Epoch Flow

The adaptive sampling pipeline follows a 6-step epoch flow:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           EPOCH FLOW                                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  [1] TRAIN FLOW MATCHER                                                  │
│      Input: current training data (endpoint pairs)                       │
│      Output: trained flow_matcher                                        │
│                                                                          │
│  [2] OPTIMIZE λ*/δ* (no q_hat yet)                                       │
│      Input: ALL training data (X_train, y_train) + flow_matcher          │
│      Do: estimate p(success), grid search to minimize loss               │
│      Output: λ*, δ*                                                      │
│      Method: conformal_predictor.optimize_thresholds()                   │
│                                                                          │
│  [3] SAMPLE D1 (Calibration Set)                                         │
│      n_d1 = (1 - d2_ratio) × samples_per_epoch                           │
│      Sample n_d1 trajectories sequentially from shuffled list            │
│      Get TRUE LABELS from shuffled_labels.txt                            │
│      Mark as USED, add to training                                       │
│      → D1 = n_d1 labeled points                                          │
│                                                                          │
│  [4] CALIBRATE q_hat USING D1                                            │
│      Input: D1 (labeled) + λ*, δ*                                        │
│      Do: estimate p(success) for D1, compute non-conformity scores       │
│      Output: q_hat = (1-α) quantile of scores                            │
│      Method: conformal_predictor.calibrate_qhat()                        │
│                                                                          │
│  [5] SAMPLE D2 (Uncertain) USING q_hat                                   │
│      n_d2_target = d2_ratio × samples_per_epoch                          │
│      Loop:                                                               │
│        - Sample batch from available pool                                │
│        - Estimate p(success), p(failure)                                 │
│        - Compute prediction sets using q_hat                             │
│        - Uncertain (|prediction_set| > 1) → keep, mark USED              │
│        - Certain (|prediction_set| = 1) → discard, stays AVAILABLE       │
│      Until: n_d2_target uncertain found or pool exhausted                │
│      → D2 = uncertain points added to training                           │
│      Method: UncertainSampler.sample()                                   │
│                                                                          │
│  [6] SUMMARY                                                             │
│      D1 + D2 added to training                                           │
│      Rebuild datasets for next epoch                                     │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Concepts

### Three-Way Classification

For each initial state x, we classify into:
- **SUCCESS** (label=1): Trajectory converges to attractor
- **FAILURE** (label=-1): Trajectory does not converge
- **UNKNOWN** (label=0): Uncertain, near separatrix

### Decision Rules

**One-Sided** (for pendulum, mountain car):
```
         p(success|x)
    0                                               1
    |←———— FAILURE ————→|←— UNKNOWN —→|←——— SUCCESS ————→|
    0                 λ-δ            λ+δ                 1
```

**Two-Sided** (for CartPole):
Uses both p(success) and p(failure) for classification:
- SUCCESS: p_s ≥ λ+δ AND p_f ≤ 1-λ+δ
- FAILURE: p_f ≥ 1-λ+δ AND p_s ≤ λ+δ
- UNKNOWN: neither confident

### Prediction Sets

A point is **uncertain** if its prediction set contains multiple labels:
- `|C(x)| > 1` → uncertain (needs more training data)
- `|C(x)| = 1` → certain (confident prediction)

The prediction set includes all labels y where:
```
s(p(x), y) ≤ q_hat
```

---

## Key Parameters

### Adaptive Sampling Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `samples_per_epoch` | 50 | Total samples to add per epoch (D1 + D2) |
| `d2_ratio` | 0.5 | Fraction for D2 (uncertain sampling). D1 = (1-d2_ratio) × samples_per_epoch |
| `initial_train_size` | 100 | Initial training trajectories |
| `n_epochs` | 10 | Number of adaptive sampling epochs |
| `batch_size_sampling` | 50 | Batch size when searching for uncertain points |
| `max_samples_per_epoch` | 50000 | Safety cap on candidates evaluated per epoch |

### Conformal Prediction Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `alpha (α)` | 0.1 | Significance level (0.1 = 90% coverage) |
| `delta (δ)` | 0.05 | Unknown region half-width (used when optimize_mode="lambda") |
| `w` | 1.0 | Loss weight: w × misclass + (1-w) × unknown |
| `num_mc_samples` | 10 | MC samples for probability estimation |
| `attractor_radius` | 0.1 | Radius for classify_attractor() |
| `optimize_mode` | "delta" | "lambda" (optimize λ with fixed δ) or "delta" (optimize δ with fixed λ=0.5) |
| `decision_rule` | "two_sided" | "one_sided" (p_s only) or "two_sided" (p_s and p_f) |
| `threshold_mode` | "dynamic" | "dynamic" (optimize λ/δ) or "fixed" (use fixed thresholds) |
| `fixed_lambda_star` | 0.5 | Fixed λ* (only used when threshold_mode="fixed") |
| `fixed_delta_star` | 0.1 | Fixed δ* (only used when threshold_mode="fixed") |

---

## Threshold Mode

Controls whether to optimize thresholds or use fixed values in Step 2:

**Dynamic Mode** (`threshold_mode: dynamic`):
- Runs `optimize_thresholds()` to find optimal λ*/δ* via grid search
- Uses `optimize_mode` to decide what to optimize:
  - `optimize_mode: lambda` → optimize λ* with fixed δ from config
  - `optimize_mode: delta` → optimize δ* with fixed λ=0.5

**Fixed Mode** (`threshold_mode: fixed`):
- Skips optimization entirely
- Uses `fixed_lambda_star` and `fixed_delta_star` from config
- Useful for reproducibility or when optimal thresholds are known

Both modes still calibrate q_hat using D1 samples in Step 4.

---

## Two-Phase Fitting

The conformal predictor uses a two-phase fitting approach:

### Phase 1: Optimize Thresholds
```python
lambda_star, delta_star, opt_info = conformal_predictor.optimize_thresholds(
    X_train, y_train, verbose=True
)
```
- Uses ALL training data for λ/δ optimization
- Does NOT calibrate q_hat yet

### Phase 2: Calibrate q_hat
```python
q_hat = conformal_predictor.calibrate_qhat(
    X_cal, y_cal, verbose=True
)
```
- Uses D1 (fresh calibration data) for q_hat calibration
- Requires λ*, δ* from Phase 1

This separation ensures:
1. λ/δ optimization uses maximum training data
2. q_hat calibration uses fresh data (D1) not seen during optimization

---

## State Tracking

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         STATE MANAGEMENT                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  AVAILABLE POOL          USED (in training)                              │
│  ───────────────         ─────────────────                               │
│  Can be sampled          Never sampled again                             │
│                                                                          │
│  Initial: all indices    Initial: first initial_train_size               │
│                                                                          │
│  D1 (calibration):                                                       │
│    sampled → USED, added to training                                     │
│                                                                          │
│  D2 (uncertain):                                                         │
│    uncertain → USED, added to training                                   │
│    certain   → stays AVAILABLE (not added)                               │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Non-Conformity Scores

### One-Sided (using p_success only)

```python
# For FAILURE (y=-1): High p makes FAILURE claim strange
s(p, y=-1) = max(0, p - (λ-δ))

# For SUCCESS (y=1): Low p makes SUCCESS claim strange
s(p, y=1) = max(0, (λ+δ) - p)

# For UNKNOWN (y=0): Outside [λ-δ, λ+δ] is strange
s(p, y=0) = max(0, (λ-δ) - p, p - (λ+δ))
```

### Two-Sided (using p_success and p_failure)

Let u = λ+δ (success threshold), v = 1-λ+δ (failure threshold):

```python
# For SUCCESS (y=1): Require p_s ≥ u and p_f ≤ v
s(p_s, p_f, y=1) = max(0, u - p_s, p_f - v)

# For FAILURE (y=-1): Require p_f ≥ v and p_s ≤ u
s(p_s, p_f, y=-1) = max(0, v - p_f, p_s - u)

# For UNKNOWN (y=0): Require p_s < u and p_f < v
s(p_s, p_f, y=0) = max(0, p_s - u, p_f - v)
```

---

## Usage Examples

### Basic Conformal Prediction

```python
from adaptive_roa.conformal import ConformalConfig, ConformalPredictor
from adaptive_roa.systems.cartpole import CartPoleSystem

# Load your trained flow matcher
flow_matcher = YourFlowMatcher.load_from_checkpoint("path/to/checkpoint")
system = CartPoleSystem()

# Create config
config = ConformalConfig(
    delta=0.05,
    alpha=0.1,
    num_mc_samples=10,
    attractor_radius=0.1,
    optimize_mode="delta",
    decision_rule="two_sided"
)

# Create predictor
predictor = ConformalPredictor(flow_matcher, system, config, device="cuda")

# Two-phase fitting
lambda_star, delta_star, _ = predictor.optimize_thresholds(X_train, y_train)
q_hat = predictor.calibrate_qhat(X_cal, y_cal)

# Predict on new data
prediction_sets, p_success, p_failure = predictor.predict(X_new)

# Select uncertain points
uncertain_mask, uncertain_indices, p_s, p_f = predictor.select_uncertain(X_candidates)
```

### UncertainSampler

```python
from adaptive_roa.adaptive.balanced_sampler import UncertainSampler

# Create sampler
sampler = UncertainSampler(
    dataset_builder=dataset_builder,
    target_count=n_d2_target,      # How many uncertain points to find
    batch_size=50,                  # Batch size when searching
    max_candidates=50000,           # Safety cap
)

# Sample uncertain points
result = sampler.sample(
    prob_estimator=prob_estimator,
    calibrator=conformal_predictor.calibrator,
    lambda_star=lambda_star,
    delta_star=delta_star,
    q_hat=q_hat,
    decision_rule="two_sided",
    exclude=set(d1_indices),        # Don't re-sample D1
    verbose=True
)

# Results
uncertain_indices = result.uncertain_indices
n_candidates_evaluated = result.n_candidates_evaluated
n_certain_discarded = result.n_certain_discarded
```

### Running the Pipeline

```bash
# Default config
python scripts/run_adaptive.py system=cartpole_pybullet

# Custom parameters
python scripts/run_adaptive.py system=cartpole_pybullet \
    initial_train_size=200 \
    n_epochs=5 \
    samples_per_epoch=100 \
    d2_ratio=0.75
```

---

## Configuration Reference

### configs/adaptive_v2/system/cartpole_pybullet.yaml

```yaml
name: adaptive_cartpole_pybullet
seed: 42
device: cuda:0

# Data Source
data_source:
  trajectories_dir: ${data_dir}/cartpole_pybullet/trajectories
  shuffled_indices_file: ${data_dir}/train_test_splits/shuffled_indices_0.txt
  shuffled_labels_file: ${data_dir}/train_test_splits/shuffled_labels_0.txt
  eval_states_file: ${data_dir}/cartpole_pybullet/eval_states.txt

# Dataset splits
val_ratio: 0.1
test_ratio: 0.1

# Adaptive Sampling
initial_train_size: 100
n_epochs: 10
samples_per_epoch: 50
warm_start: false

# D1/D2 Split
d2_ratio: 0.5              # D1 = 25, D2 = 25 (for samples_per_epoch=50)
batch_size_sampling: 50
max_samples_per_epoch: 50000

# Conformal Prediction
conformal:
  delta: 0.05
  w: 1.0
  alpha: 0.1
  num_mc_samples: 10
  attractor_radius: 0.1
  optimize_mode: delta       # "lambda" or "delta"
  decision_rule: two_sided   # "one_sided" or "two_sided"
  threshold_mode: dynamic    # "dynamic" or "fixed"
  lambda_grid_size: 100
  delta_grid_size: 100
  delta_min: 0.05
  delta_max: 0.45
```

---

## Expected Output

```
============================================================
EPOCH 1
============================================================

[1] Training flow matcher on 150 trajectories...
    Best val_loss: 0.0234

[2] Optimizing thresholds on training data...
    → λ* = 0.5000 (fixed)
    → δ* = 0.1200

[3] Sampling D1 (calibration set)...
    D1: 25 calibration points sampled and added to training

[4] Calibrating q_hat on D1...
    → q_hat = 0.0850

[5] Sampling D2 (uncertain) using q_hat...
    Batch 1: 50 evaluated, 12 uncertain, 38 certain. Progress: 12/25
    Batch 2: 50 evaluated, 15 uncertain, 35 certain. Progress: 25/25
    RESULT: 25 uncertain found, 100 evaluated, 73 certain discarded

[6] Summary of sampling this epoch...
    D1 (calibration): 25 points (added)
    D2 (uncertain):   25 points (added)
    Total added:      50
    Discarded (certain, stay in pool): 73

[7] Evaluating on test set...
    Coverage: 92.00% (target: 90%)
    Confident accuracy: 87.50%
```

---

## System Compatibility

Works with any system that implements:
- `define_manifold_structure()` → List of ManifoldComponents
- `define_state_bounds()` → Dict of bounds
- `classify_attractor(state, radius)` → Labels tensor

Currently tested with:
- CartPole (ℝ² × S¹ × ℝ)
- Pendulum (S¹ × ℝ)
- Mountain Car (ℝ²)
- Humanoid (ℝ³⁴ × S² × ℝ³⁰)
