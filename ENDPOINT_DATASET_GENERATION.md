# Endpoint Dataset Generation Guide

## Overview

This guide explains how to generate endpoint datasets from trajectory files using the modular Hydra configuration system.

## What is an Endpoint Dataset?

An endpoint dataset contains pairs of `(start_state, attractor_state)` where:
- **start_state**: Initial state from a trajectory
- **attractor_state**: First state in the trajectory that reaches an attractor basin

## Quick Start

### Basic Usage (Use Defaults)
```bash
python src/build_endpoint_dataset.py
```
Uses default configuration:
- System: `pendulum`
- Dataset: `pendulum_lqr_5k`

### Change System
```bash
python src/build_endpoint_dataset.py system=cartpole
```

### Change Dataset
```bash
python src/build_endpoint_dataset.py dataset=pendulum_lqr_50k
```

### Combine Multiple Overrides
```bash
python src/build_endpoint_dataset.py system=cartpole dataset=cartpole_lqr
```

### Override Individual Parameters
```bash
python src/build_endpoint_dataset.py system.attractor_radius=0.05
```

### Custom Paths (Advanced)
```bash
python src/build_endpoint_dataset.py \
    data_dirs=/path/to/trajectories \
    dest_dir=/path/to/output
```

## Configuration Structure

### Config Groups

The configuration is modular using Hydra config groups:

#### 1. **System** (`configs/system/`)
Defines the dynamical system and attractor parameters.

Available options:
- `pendulum`: Pendulum with S¹×ℝ manifold
- `cartpole`: CartPole with ℝ²×S¹×ℝ manifold

Example: `configs/system/pendulum.yaml`
```yaml
_target_: src.systems.pendulum.PendulumSystem
name: pendulum
attractor_radius: 0.01  # Radius for attractor detection
```

#### 2. **Dataset** (`configs/dataset/`)
Defines input/output data paths.

Available options:
- `pendulum_lqr_5k`: Pendulum 5k trajectories
- `pendulum_lqr_50k`: Pendulum 50k trajectories
- `cartpole_lqr`: CartPole LQR trajectories

Example: `configs/dataset/pendulum_lqr_5k.yaml`
```yaml
data_dirs: /common/users/dm1487/arcmg_datasets/pendulum_lqr/pendulum_lqr5k/pendulum_lqr5k
dest_dir: /common/users/dm1487/arcmg_datasets/pendulum_lqr/pendulum_lqr5k/endpoint_dataset
```

### Main Config (`configs/build_endpoint_dataset.yaml`)

```yaml
defaults:
  - system: pendulum              # System dynamics
  - dataset: pendulum_lqr_5k      # Dataset paths
  - _self_                        # This file (lowest priority)
```

## Output Format

### Pendulum Endpoint Dataset
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

### CartPole Endpoint Dataset
**Format**: Space-separated values, 8 columns per line
```
x_start θ_start ẋ_start θ̇_start x_end θ_end ẋ_end θ̇_end
x_start θ_start ẋ_start θ̇_start x_end θ_end ẋ_end θ̇_end
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

### Create Custom Dataset Config

Create a new file in `configs/dataset/`:

```yaml
# configs/dataset/my_custom_dataset.yaml
# @package _global_

data_dirs: /path/to/my/trajectories
dest_dir: /path/to/my/output
```

Then use it:
```bash
python src/build_endpoint_dataset.py dataset=my_custom_dataset
```

### Override Attractor Detection Radius

```bash
python src/build_endpoint_dataset.py system.attractor_radius=0.05
```

### View Final Configuration

```bash
python src/build_endpoint_dataset.py --cfg job
```

### List All Available Options

```bash
python src/build_endpoint_dataset.py --help
```

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

### Generate Pendulum 5k Dataset
```bash
python src/build_endpoint_dataset.py
```

### Generate Pendulum 50k Dataset
```bash
python src/build_endpoint_dataset.py dataset=pendulum_lqr_50k
```

### Generate CartPole Dataset
```bash
python src/build_endpoint_dataset.py system=cartpole dataset=cartpole_lqr
```

### Custom Attractor Radius
```bash
python src/build_endpoint_dataset.py \
    system.attractor_radius=0.02 \
    dataset=pendulum_lqr_50k
```

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

## Related Documentation

- [TRAINING_GUIDE.md](TRAINING_GUIDE.md): Training flow matching models
- [EVALUATION_GUIDE.md](EVALUATION_GUIDE.md): ROA evaluation
- [CLAUDE.md](CLAUDE.md): Complete project overview
