# Endpoint Dataset Generation Guide

## Overview

This guide explains how to generate endpoint datasets from trajectory files using shuffled indices for consistent, reproducible dataset construction.

## What is an Endpoint Dataset?

An endpoint dataset contains pairs of `(start_state, attractor_state)` where:
- **start_state**: Initial state from a trajectory
- **attractor_state**: Final state where the trajectory reaches an attractor basin

The shuffled builder uses pre-computed shuffled indices to ensure consistent ordering across runs, making it ideal for incremental dataset construction.

## Quick Start

### Basic Usage (Use Defaults)
```bash
python src/build_shuffled_endpoint_dataset.py
```
Uses default configuration:
- System: `pendulum`
- Trajectories: First 100 trajectories from shuffled indices
- Mode: `train` (all states → final state pairs)

### Change System
```bash
python src/build_shuffled_endpoint_dataset.py system=cartpole
```

### Incremental Dataset Building
```bash
# Build first 500 trajectories
python src/build_shuffled_endpoint_dataset.py start=0 end=500 increment=500

# Build next 500 trajectories
python src/build_shuffled_endpoint_dataset.py start=500 end=1000 increment=1000

# Build validation set
python src/build_shuffled_endpoint_dataset.py start=45000 end=47500 increment=2500 type=val

# Build test set (one pair per trajectory)
python src/build_shuffled_endpoint_dataset.py start=47500 end=50000 increment=2500 type=test
```

### Fixed Attractor Mode
```bash
# Use fixed success/failure endpoints instead of actual final states
python src/build_shuffled_endpoint_dataset.py \
    use_fixed_attractors=true \
    attractor_radius=0.1 \
    balance_dataset=true
```

### Custom Paths
```bash
python src/build_shuffled_endpoint_dataset.py \
    data_dirs=/path/to/trajectories \
    shuffled_idxs_file=/path/to/shuffled_indices.txt \
    dest_dir=/path/to/output
```

## Configuration Structure

### Main Config (`configs/build_shuffled_endpoint_dataset.yaml`)

```yaml
defaults:
  - system: pendulum              # System dynamics (pendulum or cartpole)

# Trajectory data paths
data_dirs: /path/to/trajectories
shuffled_idxs_file: /path/to/shuffled_indices.txt
dest_dir: /path/to/output

# Processing range
start: 0       # Start index in shuffled_indices.txt
end: 100       # End index (exclusive)
increment: 100 # Number used in output filename

# Dataset type: "train", "val", or "test"
type: train

# Fixed Attractor Mode (Optional)
use_fixed_attractors: false  # Use fixed success/failure endpoints
attractor_radius: 0.1        # Radius for success/failure classification
balance_dataset: false       # Balance success/failure counts
```

### System Config (`configs/system/`)

Available options:
- `pendulum`: Pendulum with S¹×ℝ manifold
- `cartpole`: CartPole with ℝ²×S¹×ℝ manifold

Example: `configs/system/pendulum.yaml`
```yaml
_target_: src.systems.pendulum.PendulumSystem
name: pendulum
attractor_radius: 0.01  # Default attractor detection radius
```

### Dataset Types

- **train**: Creates endpoint pairs for ALL states in trajectory → final state
  (produces N pairs per trajectory where N = trajectory length)
- **val**: Same as train mode
- **test**: Creates only FIRST state → final state
  (produces 1 pair per trajectory)

## Output Format

### Standard Mode (use_fixed_attractors=false)

#### Pendulum Endpoint Dataset
**Format**: Space-separated values, 4 columns per line
```
θ_start θ̇_start θ_end θ̇_end
θ_start θ̇_start θ_end θ̇_end
...
```

Example:
```
0.5 1.0 0.0 0.0
1.2 -0.5 2.1 0.0
...
```

#### CartPole Endpoint Dataset
**Format**: Space-separated values, 8 columns per line
```
x_start θ_start ẋ_start θ̇_start x_end θ_end ẋ_end θ̇_end
x_start θ_start ẋ_start θ̇_start x_end θ_end ẋ_end θ̇_end
...
```

### Fixed Attractor Mode (use_fixed_attractors=true)

In this mode, the endpoint is replaced with a **scalar value** (1 or -1) representing success/failure. The model learns to regress from the start state to this scalar target.

#### Pendulum with Scalar Attractor
**Format**: Space-separated values, 3 columns per line
```
θ_start θ̇_start attractor
θ_start θ̇_start attractor
...
```
- Attractor: `1` (success) or `-1` (failure)

Example:
```
0.5 1.0 1
1.2 -0.5 -1
...
```

#### CartPole with Scalar Attractor
**Format**: Space-separated values, 5 columns per line
```
x_s θ_s ẋ_s θ̇_s attractor
x_s θ_s ẋ_s θ̇_s attractor
...
```
- Attractor: `1` (success) or `-1` (failure)

Example:
```
0.5 0.2 1.0 0.5 1
-0.8 2.9 -0.3 -1.2 -1
...
```

## Input Format

### Trajectory Files
The script expects trajectory files in the `data_dirs` directory:
- **File format**: `.txt` files with CSV format
- **Content**: Each line contains state values separated by commas

**Pendulum trajectory** (2 columns):
```
θ, θ̇
θ, θ̇
...
```

**CartPole trajectory** (4 columns):
```
x, θ, ẋ, θ̇
x, θ, ẋ, θ̇
...
```

## Advanced Usage

### Override Configuration Parameters

```bash
# Override attractor detection radius
python src/build_shuffled_endpoint_dataset.py attractor_radius=0.05

# Override system attractor radius
python src/build_shuffled_endpoint_dataset.py system.attractor_radius=0.05

# Use custom paths
python src/build_shuffled_endpoint_dataset.py \
    data_dirs=/custom/path/trajectories \
    shuffled_idxs_file=/custom/path/shuffled_indices.txt \
    dest_dir=/custom/output
```

### View Final Configuration

```bash
python src/build_shuffled_endpoint_dataset.py --cfg job
```

### List All Available Options

```bash
python src/build_shuffled_endpoint_dataset.py --help
```

### Fixed Attractor Mode Explained

When `use_fixed_attractors=true`:
1. Script classifies each trajectory as success/failure based on final state
2. Endpoint is replaced with a **scalar value**: `1` (success) or `-1` (failure)
3. Model learns to regress from start_state → scalar attractor

**Key differences from standard mode:**
- **Standard mode**: Each state points to its actual trajectory endpoint (multi-dimensional)
  - Pendulum: `[θ_start, θ̇_start] → [θ_end, θ̇_end]` (2D → 2D)
  - CartPole: `[x_s, θ_s, ẋ_s, θ̇_s] → [x_e, θ_e, ẋ_e, θ̇_e]` (4D → 4D)

- **Fixed attractor mode**: Each state points to a scalar target
  - Pendulum: `[θ_start, θ̇_start] → 1 or -1` (2D → scalar)
  - CartPole: `[x_s, θ_s, ẋ_s, θ̇_s] → 1 or -1` (4D → scalar)

This effectively turns the regression problem into a classification-style task where the model outputs a single value indicating trajectory success/failure.

When `balance_dataset=true` (only in train/val mode):
- Ensures equal number of success/failure pairs in output dataset
- Randomly samples from success and failure classes to achieve balance

## Hydra Benefits

✅ **Modular**: Reusable config groups
✅ **Composable**: Mix and match system + dataset
✅ **Type-safe**: Hydra validates configuration
✅ **Override-friendly**: Easy command-line overrides
✅ **Self-documenting**: Clear config structure

## System-Specific Details

### Pendulum System
- **State space**: S¹×ℝ (angle × angular velocity)
- **State**: (θ, θ̇)
- **Attractors**:
  - [0.0, 0.0]: Bottom equilibrium (stable)
  - [2.1, 0.0]: Top-right equilibrium
  - [-2.1, 0.0]: Top-left equilibrium

### CartPole System
- **State space**: ℝ²×S¹×ℝ (position × angle × velocities)
- **State**: (x, θ, ẋ, θ̇)
- **Attractors**: System-specific equilibrium points

## Common Workflows

### Build Training Dataset Incrementally (Pendulum)
```bash
# Build first 10k trajectories
python src/build_shuffled_endpoint_dataset.py start=0 end=10000 increment=10000

# Build next 10k trajectories
python src/build_shuffled_endpoint_dataset.py start=10000 end=20000 increment=20000

# Continue until desired size...
python src/build_shuffled_endpoint_dataset.py start=40000 end=45000 increment=45000
```

### Build Complete Train/Val/Test Split (Pendulum 50k)
```bash
# Training set (45k trajectories)
python src/build_shuffled_endpoint_dataset.py start=0 end=45000 increment=45000 type=train

# Validation set (2.5k trajectories)
python src/build_shuffled_endpoint_dataset.py start=45000 end=47500 increment=2500 type=val

# Test set (2.5k trajectories, one pair per trajectory)
python src/build_shuffled_endpoint_dataset.py start=47500 end=50000 increment=2500 type=test
```

### Build CartPole Dataset
```bash
# Training set
python src/build_shuffled_endpoint_dataset.py \
    system=cartpole \
    start=0 end=1000 increment=1000 \
    type=train

# Validation set
python src/build_shuffled_endpoint_dataset.py \
    system=cartpole \
    start=1000 end=1200 increment=200 \
    type=val
```

### Build Dataset with Scalar Attractor
```bash
# Dataset with scalar attractor (1 or -1)
python src/build_shuffled_endpoint_dataset.py \
    use_fixed_attractors=true \
    attractor_radius=0.1 \
    start=0 end=10000 increment=10000

# Balanced dataset (equal success/failure counts)
python src/build_shuffled_endpoint_dataset.py \
    use_fixed_attractors=true \
    balance_dataset=true \
    attractor_radius=0.1 \
    start=0 end=10000 increment=10000
```

**Output format (Pendulum example):**
```
# Standard mode (4 columns): state → endpoint
0.5 1.0 0.0 0.0
1.2 -0.5 2.1 0.0

# Fixed attractor mode (3 columns): state → scalar
0.5 1.0 1     # Success trajectory
1.2 -0.5 -1   # Failure trajectory
```

**Use case**: Train a model to predict trajectory success/failure as a regression to ±1

## Troubleshooting

### Configuration Not Found
```
Error: Could not find 'my_config'
```
**Solution**: Ensure config files exist in the correct directory and use correct naming.

### Path Not Found
```
Error: Directory not found: /path/to/data
```
**Solution**: Verify `data_dirs` path exists or create custom dataset config.

### No Trajectory Files Found
```
Found 0 trajectory files
```
**Solution**: Ensure trajectory `.txt` files exist in `data_dirs`.

## Output Files

Generated datasets are saved as:
- **Filename**: `{increment}_endpoint_dataset.txt` (e.g., `10000_endpoint_dataset.txt`)
- **Location**: `{dest_dir}/{increment}_endpoint_dataset.txt`

For validation/test datasets:
- `validation_endpoint_dataset.txt`
- `test_endpoint_dataset.txt`

## Related Documentation

- [TRAINING_GUIDE.md](TRAINING_GUIDE.md): Training flow matching models
- [ROA_ANALYSIS_GUIDE.md](ROA_ANALYSIS_GUIDE.md): ROA evaluation and basin analysis
- [CLAUDE.md](CLAUDE.md): Complete project overview
