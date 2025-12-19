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
src/
├── conformal/                      # Conformal prediction module
│   ├── __init__.py
│   ├── config.py                   # ConformalConfig dataclass
│   ├── probability_estimator.py    # MC sampling for p(success|x)
│   ├── lambda_optimizer.py         # Find optimal λ*
│   ├── calibrator.py               # Compute q_hat for coverage
│   └── predictor.py                # Main ConformalPredictor class
│
├── adaptive/                       # Adaptive sampling module
│   ├── __init__.py
│   ├── data_source.py              # TrajectoryDataSource: load trajectory pools
│   ├── dataset_builder.py          # AdaptiveDatasetBuilder: build datasets from indices
│   ├── data_manager.py             # Track trajectory + label data (legacy)
│   ├── sampler.py                  # Sample initial states
│   ├── simulator.py                # Interface to simulations
│   ├── pipeline.py                 # Main pipeline orchestrator
│   └── run_adaptive_cartpole.py    # CartPole PyBullet run script
│
configs/
├── conformal/
│   └── default.yaml                # Conformal prediction config
├── adaptive/
│   └── default.yaml                # Adaptive sampling config
├── adaptive_pipeline_cartpole.yaml # Full pipeline config (generic)
└── adaptive_cartpole_pybullet.yaml # CartPole PyBullet config
```

---

## Concepts

### Three-Way Classification

For each initial state x, we classify into:
- **SUCCESS** (label=1): Trajectory converges to attractor
- **FAILURE** (label=-1): Trajectory does not converge
- **UNKNOWN** (label=0): Uncertain, needs simulation

```
         p(success|x)
    0                                               1
    |←———— FAILURE ————→|←— UNKNOWN —→|←——— SUCCESS ————→|
    0                 λ-δ            λ+δ                 1
```

### Key Parameters

#### Conformal Prediction Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `delta (δ)` | 0.05 | Unknown region half-width |
| `w` | 0.9 | Misclassification weight in loss |
| `alpha (α)` | 0.1 | Significance level (0.1 = 90% coverage) |
| `num_mc_samples` | 100 | MC samples for probability estimation |
| `attractor_radius` | 0.2 | Radius for classify_attractor() |
| `calibration_ratio` | 0.3 | Fraction of training labels for calibration (rest for λ optimization) |
| `lambda_grid_size` | 100 | Grid search resolution |

#### Adaptive Sampling Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `initial_train_size` | 100 | Initial training trajectories (indices 0 to n-1) |
| `n_epochs` | 10 | Number of adaptive sampling epochs |
| `samples_per_epoch` | 50 | New candidate trajectories per epoch |
| `d1_ratio` | 0.5 | Fraction always added to training (rest is selective) |

#### Flow Matcher Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `val_ratio` | 0.1 | Fraction of training for FM validation (with overlap) |
| `test_ratio` | 0.1 | Fraction of training for FM test (with overlap) |
| `fm_training.max_epochs` | 500 | Max epochs per FM training |

### Coverage Guarantee

Conformal prediction ensures that prediction sets contain the true label with probability ≥ 1-α (e.g., 90% for α=0.1).

---

## Usage

### Basic Conformal Prediction

```python
from src.conformal import ConformalConfig, ConformalPredictor
from src.systems.cartpole import CartPoleSystem

# Load your trained flow matcher
flow_matcher = YourFlowMatcher.load_from_checkpoint("path/to/checkpoint")
system = CartPoleSystem()

# Create config
config = ConformalConfig(
    delta=0.05,
    alpha=0.1,
    num_mc_samples=100,
    attractor_radius=0.2
)

# Create predictor
predictor = ConformalPredictor(flow_matcher, system, config, device="cuda")

# Fit on labeled data
# X_train, y_train: for lambda optimization
# X_cal, y_cal: for calibration (coverage guarantee)
predictor.fit(X_train, y_train, X_cal, y_cal)

# Predict on new data
prediction_sets, p_success = predictor.predict(X_new)

# Select uncertain points (for adaptive sampling)
uncertain_mask, uncertain_indices, p_success = predictor.select_uncertain(X_candidates)

# Evaluate on test set
metrics = predictor.evaluate(X_test, y_test)
```

### Full Adaptive Sampling Pipeline

```python
from src.conformal import ConformalConfig
from src.adaptive import AdaptiveSamplingPipeline, AdaptivePipelineConfig
from src.adaptive.simulator import FileBasedSimulator
from src.systems.cartpole import CartPoleSystem

# Setup system and simulator
system = CartPoleSystem()
simulator = FileBasedSimulator(
    "path/to/trajectory_pool.txt",
    system,
    attractor_radius=0.2
)

# Configs
conformal_config = ConformalConfig(delta=0.05, alpha=0.1, num_mc_samples=100)
pipeline_config = AdaptivePipelineConfig(
    n_samples_per_epoch=50,
    d1_ratio=0.5,
    max_epochs=10,
    output_dir="outputs/adaptive_cartpole"
)

# Create pipeline
pipeline = AdaptiveSamplingPipeline(
    system=system,
    simulator=simulator,
    conformal_config=conformal_config,
    pipeline_config=pipeline_config,
    device="cuda"
)

# Initialize with existing data
pipeline.initialize_with_data(
    "path/to/initial_trajectories.txt",
    flow_matcher_factory=lambda: create_new_flow_matcher()
)

# Define training function
def train_flow_matcher(trajectory_file: str):
    # Your training code here
    # Returns trained flow matcher
    ...

# Run adaptive sampling
results = pipeline.run(trainer_fn=train_flow_matcher, n_epochs=10)
```

### Using Hydra Configuration

```bash
python run_adaptive_pipeline.py --config-name=adaptive_pipeline_cartpole
```

---

## Algorithm Details

### Epoch Data Flow (Pre-Collected Data Mode)

Each epoch:

1. **Train flow matcher** on current training set (endpoint pairs from trajectories)
2. **Fit conformal predictor**:
   - Get training labels (start_state, label) for current training trajectories
   - Split into optimization set (70%) and calibration set (30%) based on `calibration_ratio`
   - Optimize λ* on optimization set
   - Calibrate q_hat on calibration set
3. **Sample candidates** - get next `samples_per_epoch` trajectories sequentially
4. **Split** into D1 and D2 based on `d1_ratio`
5. **Add D1** to training (always added)
6. **Evaluate D2** - use conformal predictor to identify uncertain points
7. **Add uncertain D2** to training (skip confident predictions = savings!)
8. **Rebuild datasets** with new training trajectories
9. **Evaluate** on test set (subset of training, for monitoring)

### Data Types

| Data Type | Format | Used For |
|-----------|--------|----------|
| **Trajectory Data** | (start_state, end_state) pairs | Flow matcher training |
| **Classification Data** | (initial_state, label) pairs | Conformal predictor (λ optimization) |

Both derived from the same simulations:
```
Simulation: x₀ → (run dynamics) → x_T
    │
    ├─→ Trajectory pair: (x₀, x_T) → FM training
    │
    └─→ Classification: (x₀, classify(x_T)) → CP training
```

### Non-Conformity Scores

```python
# For FAILURE (y=-1): High p makes FAILURE claim strange
s(p, y=-1) = max(0, p - (λ-δ))

# For SUCCESS (y=1): Low p makes SUCCESS claim strange
s(p, y=1) = max(0, (λ+δ) - p)

# For UNKNOWN (y=0): Outside [λ-δ, λ+δ] is strange
s(p, y=0) = max(0, (λ-δ) - p, p - (λ+δ))
```

### Lambda Optimization

```python
Loss(λ) = w × MisclassificationRate + (1-w) × UnknownRate
```

Grid search over λ ∈ [δ, 1-δ] to find λ* that minimizes loss.

### Calibration

```python
# Compute scores for calibration set using TRUE labels
scores = [s(p[i], y_true[i]) for i in calibration_set]

# Find quantile for coverage guarantee
q_hat = quantile(scores, (1-α) × (n+1) / n)

# Prediction set includes labels with score ≤ q_hat
C(x) = {y : s(p(x), y) ≤ q_hat}
```

---

## Expected Output

```
============================================================
EPOCH 0
============================================================

[1] Sampling 50 new initial states...
    D1 (calibration): 25 points
    D2 (selection pool): 25 points

[2] Simulating D1 (25 simulations)...

[3] Retraining flow matcher...

[4] Fitting conformal predictor...
    → λ* = 0.4800
    → Calibration q_hat = 0.0700

[5] Evaluating D2 for uncertain points...
    Uncertain: 7 points
    Confident: 18 points (skipped)

[6] Simulating uncertain points (7 simulations)...

[7] Evaluating on test set...

────────────────────────────────────────
EPOCH 0 SUMMARY
────────────────────────────────────────
  Simulations: 32 (saved 18)
  Total data: 132
  λ* = 0.4800, q_hat = 0.0700
  Unknown rate: 35.00%
  F1 score: 0.7500
  Coverage: 92.00%
```

---

## Configuration Reference

### configs/adaptive_cartpole_pybullet.yaml (Full Example)

```yaml
name: adaptive_cartpole_pybullet
seed: 42
device: cuda:0

# Data Source
data_source:
  trajectories_dir: /path/to/trajectories
  shuffled_indices_file: /path/to/shuffled_indices.txt
  roa_labels_file: /path/to/roa_labels.txt

# Dataset splits (val/test are subsets of training with overlap)
val_ratio: 0.1              # 10% of training for FM validation
test_ratio: 0.1             # 10% of training for FM testing

# Adaptive Sampling
initial_train_size: 100     # Initial training trajectories
n_epochs: 10                # Number of adaptive sampling epochs
samples_per_epoch: 50       # New candidates per epoch
d1_ratio: 0.5               # Fraction always added (rest is selective)

# Conformal Prediction
conformal:
  delta: 0.05               # Unknown region half-width
  w: 0.9                    # Misclassification weight
  alpha: 0.1                # 90% coverage guarantee
  num_mc_samples: 100       # MC samples for probability estimation
  attractor_radius: 0.2     # For classify_attractor()
  calibration_ratio: 0.3    # Fraction for calibration (rest for λ optimization)

# Flow Matcher Training
fm_training:
  max_epochs: 500
```

---

## Pre-Collected Trajectory Data (Recommended)

For systems with pre-collected trajectory data (like CartPole PyBullet), use the trajectory data infrastructure:

### Data Structure Expected

```
data_directory/
├── trajectories/           # Directory with trajectory files
│   ├── sequence_0.txt      # Each file: one trajectory (states over time)
│   ├── sequence_1.txt
│   └── ...
├── shuffled_indices.txt    # Line i = filename for trajectory i
└── roa_labels.txt          # Line i = "x,θ,ẋ,θ̇,label" for trajectory i
```

### TrajectoryDataSource

Loads and manages the trajectory pool:

```python
from src.adaptive import TrajectoryDataSource, TrajectoryDataSourceConfig

config = TrajectoryDataSourceConfig(
    trajectories_dir="/path/to/trajectories",
    shuffled_indices_file="/path/to/shuffled_indices.txt",
    roa_labels_file="/path/to/roa_labels.txt",  # Optional but recommended
)
data_source = TrajectoryDataSource(config)

# Access individual trajectories
traj = data_source.load_trajectory(idx=0)  # [T, state_dim]
start, end = data_source.get_endpoint_pair(idx=0)
label = data_source.get_label(idx=0)  # -1 (failure) or 1 (success)
```

### AdaptiveDatasetBuilder

Manages training set and builds endpoint datasets for flow matcher.

**IMPORTANT**: Sampling is **SEQUENTIAL** from the `shuffled_indices.txt` ordering.
The file already provides randomization, so we maintain that ordering by taking
indices 0, 1, 2, ... in sequence.

**Val/Test for Flow Matcher**: Subsets of the current training set (with overlap).
This is acceptable since we only use them for early stopping and validation loss.

```python
from src.adaptive import AdaptiveDatasetBuilder

builder = AdaptiveDatasetBuilder(
    data_source=data_source,
    output_dir="outputs/datasets",
    val_ratio=0.1,   # 10% of training for FM validation (with overlap)
    test_ratio=0.1,  # 10% of training for FM test (with overlap)
)

# Get initial training set (sequential: indices 0 to n-1)
indices = builder.get_initial_training_set(n=100)

# Build dataset files for flow matcher
files = builder.build_all_datasets()
# Returns: {'train': '...txt', 'val': '...txt', 'test': '...txt'}
# val/test are subsets of training (first/last 10% respectively)

# Get data for conformal prediction
X_train, y_train = builder.get_train_labels()  # [N_traj, dim], [N_traj]

# Get next batch of candidates (sequential: indices n to n+k-1)
candidate_states, candidate_indices = builder.get_candidate_states(n=50)

# Add selected trajectories to training
builder.add_selected_to_training(uncertain_indices)
```

### Running CartPole PyBullet

```bash
python src/adaptive/run_adaptive_cartpole.py
```

Or with custom config:
```bash
python src/adaptive/run_adaptive_cartpole.py \
    initial_train_size=200 \
    n_epochs=5 \
    samples_per_epoch=100
```

---

## Simulator Options (Alternative)

### FileBasedSimulator
Looks up trajectories from a pre-computed file by matching start states.
```python
simulator = FileBasedSimulator("trajectories.txt", system, attractor_radius=0.2)
```

### CallbackSimulator
Calls an actual simulation function for truly adaptive sampling.
```python
def run_simulation(initial_states: np.ndarray) -> np.ndarray:
    # Your physics simulation here
    return end_states

simulator = CallbackSimulator(run_simulation, system, attractor_radius=0.2)
```

### PoolSimulator
Samples random trajectories from a pool (doesn't require exact state matching).
```python
simulator = PoolSimulator("trajectory_pool.txt", system, attractor_radius=0.2)
```

---

## Outputs

The pipeline saves to `output_dir`:
- `epoch_XXX.json`: Results for each epoch
- `final_results.json`: Summary with all metrics
- `final_trajectories.txt`: All accumulated trajectory data
- `current_trajectories.txt`: Temporary file for FM training

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
- Pendulum Cartesian (ℝ⁴)
- Humanoid (ℝ³⁴ × S² × ℝ³⁰)
