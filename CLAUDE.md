# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a robotics research project for training neural network classifiers, reachability models, and flow matching models for AI Olympics. The codebase uses PyTorch Lightning with Hydra configuration management and features unified evaluation/visualization modules with advanced attractor basin analysis.

## Architecture

### Core Components

- **Training Scripts**: Training is now modular and organized by system and variant
  - Pendulum training: `src/flow_matching/pendulum/{variant}/train.py`
  - CartPole training: `src/flow_matching/cartpole/{variant}/train.py`
  - Evaluation: `src/flow_matching/evaluate_roa.py` (unified ROA evaluation)
  - Dataset builders: `src/build_endpoint_dataset.py`, `src/generate_cartpole_endpoints.py`

- **Data Modules**: Located in `src/data/`
  - `endpoint_data.py`: Handles pendulum endpoint prediction data
  - `cartpole_endpoint_data.py`: Handles CartPole endpoint prediction data with 4D states
  - `circular_endpoint_data.py`: Circular-aware endpoint data handling
  - `classification_data.py`: Handles 4D classification data with 80/10/10 train/val/test splits
  - `reachability_data.py`: Manages 8D reachability data from separate train/valid/test files

- **Models**: Located in `src/model/`
  - **Pendulum Models**:
    - `pendulum_unet.py`: Latent conditional UNet for S¹×ℝ manifold
  - **CartPole Models**:
    - `cartpole_unet.py`: Latent conditional UNet for ℝ²×S¹×ℝ manifold
  - **Generic/Legacy Models**:
    - `simple_mlp.py`: Multi-layer perceptron with configurable hidden layers
    - `unet1d.py`: Generic 1D U-Net architecture
    - `circular_unet1d.py`: Circular-aware U-Net for pendulum dynamics
    - `conditional_unet1d.py`: Conditional U-Net with latent variables
    - `universal_unet.py`: Universal U-Net for multiple systems

- **Flow Matching Framework**: Located in `src/flow_matching/`
  - `base/`: Abstract base classes and common functionality
    - `flow_matcher.py`: Base Lightning module for latent conditional FM (~500 lines of shared code)
    - `config.py`: Common configuration and state handling
    - `inference.py`: Generic inference utilities
  - `pendulum/`: Pendulum flow matching variants
    - `latent_conditional/`: Latent Conditional FM (S¹×ℝ manifold)
      - `flow_matcher.py`: Flow matching with Facebook FM library
      - `inference.py`: Inference wrapper
      - `train.py`: Training script for pendulum
  - `cartpole/`: CartPole flow matching variants
    - `latent_conditional/`: Latent Conditional FM (ℝ²×S¹×ℝ manifold)
      - `flow_matcher.py`: Flow matching with latent variables & conditioning
      - `train.py`: Training script for CartPole
  - `evaluate_roa.py`: Unified ROA evaluation for all flow matching variants

### Unified Evaluation & Visualization System

- **Systems Configuration**: Located in `src/systems/`
  - `base.py`: Base system interface (DynamicalSystem abstract class)
  - `pendulum.py`: Pendulum dynamics implementation (S¹×ℝ manifold)
  - `cartpole.py`: CartPole dynamics implementation (ℝ²×S¹×ℝ manifold)
  - `pendulum_config.py`: Legacy pendulum system parameters
  - `pendulum_universal.py`: Universal pendulum variant

- **Evaluation Framework**: Located in `src/evaluation/`
  - `metrics.py`: Centralized metrics computation (MSE, MAE, attractor accuracy, circular distances)

- **Visualization Framework**: Located in `src/visualization/`
  - `phase_space_plots.py`: Standard phase space plotting with consistent styling
  - `flow_visualizer.py`: Flow path and trajectory visualization
  - `attractor_analysis.py`: Advanced basin analysis and separatrix detection

### Attractor Basin Analysis (NEW FEATURE)

The `AttractorBasinAnalyzer` provides state space discretization and basin mapping:

- **State Space Discretization**: Create grids with configurable resolution (default 0.1)
- **Basin Classification**: Determine which attractor each state converges to
- **Separatrix Detection**: Identify boundary regions and points that don't converge to attractors
- **Comprehensive Visualization**: Color-coded basin maps, statistics, and multi-resolution analysis
- **Data Export**: Save analysis results, visualizations, and numerical data

### Configuration System

Uses Hydra for configuration management with YAML files in `configs/`:
- **Main Training Configs**:
  - `train_pendulum.yaml`: Pendulum latent conditional FM
  - `train_cartpole.yaml`: CartPole latent conditional FM
- **Evaluation Configs**:
  - `evaluate_pendulum_roa.yaml`: Pendulum ROA evaluation (latent conditional)
  - `evaluate_cartpole_roa.yaml`: CartPole ROA evaluation (latent conditional)
- **Data Configs**: `configs/data/`
  - `endpoint_data.yaml`: Pendulum endpoint data
  - `cartpole_endpoint_data.yaml`: CartPole endpoint data
  - `circular_endpoint_data.yaml`: Circular endpoint data
  - `classification_data.yaml`, `reachability_data.yaml`: Legacy classifiers
- **Model Configs**: `configs/model/`
  - `latent_conditional_unet.yaml`, `cartpole_latent_conditional_unet.yaml`
  - `simple_mlp.yaml`, `conditional_unet.yaml`
- **Trainer Configs**: `configs/trainer/`
  - `default.yaml`, `gpu.yaml`, `flow_matching.yaml`
- **Device Configs**: `configs/device/` (gpu0-gpu5, cpu)
- **Optimizer/Scheduler**: `configs/optimizer/`, `configs/scheduler/`

### Data Processing

- **Classification Data**: 4D inputs normalized to [0,1] range with bounds [-0.5, 0.5, -10, 10]
- **Reachability Data**: 8D inputs normalized to [0,1] range with bounds [-4π, 4π, -20, 20] repeated twice

## Quick Start

**For complete training and evaluation instructions, see [TRAINING_GUIDE.md](TRAINING_GUIDE.md).**

### Training Commands
```bash
# Pendulum (Latent Conditional)
python src/flow_matching/pendulum/latent_conditional/train.py

# CartPole (Latent Conditional - richer model)
python src/flow_matching/cartpole/latent_conditional/train.py
```

### Evaluation Commands
```bash
# ROA evaluation (deterministic)
python src/flow_matching/evaluate_roa.py --config-name=evaluate_cartpole_roa

# ROA evaluation (probabilistic with uncertainty)
python src/flow_matching/evaluate_roa.py \
    --config-name=evaluate_cartpole_roa \
    evaluation.probabilistic=true \
    evaluation.num_samples=20
```

## Development Commands

### Setup
```bash
# Activate the required conda environment
conda activate /common/users/dm1487/envs/arcmg

# Install the package in development mode
pip install -e .
```

### Training
```bash
# Pendulum (Latent Conditional)
python src/flow_matching/pendulum/latent_conditional/train.py

# CartPole (Latent Conditional - richer model)
python src/flow_matching/cartpole/latent_conditional/train.py

# Customize training parameters (Latent Conditional)
python src/flow_matching/pendulum/latent_conditional/train.py \
    flow_matching.latent_dim=4 \
    base_lr=5e-4 \
    batch_size=512
```

### Evaluation & Analysis
```bash
# ROA evaluation (deterministic)
python src/flow_matching/evaluate_roa.py --config-name=evaluate_cartpole_roa

# ROA evaluation (probabilistic with uncertainty quantification)
python src/flow_matching/evaluate_roa.py \
    --config-name=evaluate_cartpole_roa \
    evaluation.probabilistic=true \
    evaluation.num_samples=20
```

### Flow Matching Inference Usage
```python
# Load trained models
from src.flow_matching.pendulum.latent_conditional.flow_matcher import PendulumLatentConditionalFlowMatcher
from src.flow_matching.cartpole.latent_conditional.flow_matcher import CartPoleLatentConditionalFlowMatcher
import torch

# Pendulum (Latent Conditional)
pendulum_model = PendulumLatentConditionalFlowMatcher.load_from_checkpoint(
    "outputs/pendulum_latent_conditional_fm/2025-10-13_18-45-32"
)

# CartPole (Latent Conditional)
cartpole_model = CartPoleLatentConditionalFlowMatcher.load_from_checkpoint(
    "outputs/cartpole_latent_conditional_fm/2025-10-13_18-45-32"
)

# Predict endpoints (Pendulum)
start_states_pendulum = torch.tensor([[0.5, 1.0]])  # (θ, θ̇)
pendulum_endpoints = pendulum_model.predict_endpoint(start_states_pendulum, num_steps=100)

# Predict endpoints (CartPole)
start_states_cartpole = torch.tensor([[0.5, 0.1, 2.0, 1.0]])  # (x, θ, ẋ, θ̇)
cartpole_endpoints = cartpole_model.predict_endpoint(start_states_cartpole, num_steps=100)

# Multiple samples for uncertainty
endpoints_batch = cartpole_model.predict_endpoints_batch(start_states_cartpole, num_samples=20)
```

### Attractor Basin Analysis Usage
```python
# Works with latent-conditional flow matching outputs
from src.visualization.attractor_analysis import AttractorBasinAnalyzer
from src.systems.pendulum_config import PendulumConfig

# Initialize analyzer
config = PendulumConfig()
analyzer = AttractorBasinAnalyzer(config)

# Run analysis (automatically handles latent-conditional outputs)
results = analyzer.analyze_attractor_basins(
    inferencer,         # Works with standard OR circular
    resolution=0.1,     # Grid resolution (configurable)
    batch_size=1000     # Batch size for efficiency
)

# Save complete analysis
analyzer.save_analysis_results("output_dir", results)
```

### GPU Configuration
GPU selection is configured via Hydra config files in `configs/device/`:
- `gpu0.yaml` through `gpu5.yaml`: Configure specific GPU device
- `cpu.yaml`: CPU-only training
- Training scripts use `cfg.trainer.devices` from Hydra configs

### Outputs
Training outputs are saved to `outputs/` directory with timestamped subdirectories:
- `outputs/{experiment_name}/{timestamp}/`: Training run directory
  - `version_0/checkpoints/`: Model checkpoints
  - TensorBoard logs
  - Hydra configuration snapshots (.hydra/)

## Data Requirements

- **Flow Matching Endpoint Data**:
  - Pendulum: 4-column text files (θ_start, θ̇_start, θ_end, θ̇_end)
  - CartPole: 8-column text files (x_s, θ_s, ẋ_s, θ̇_s, x_e, θ_e, ẋ_e, θ̇_e)
  - Format: Space-separated values, one trajectory per line
- **Classification data**: Text files with space-separated values (4 inputs + 1 label per line)
- **Reachability data**: 8D state data in separate train/valid/test files

## Model Features

### Flow Matching Models
- **Latent Conditional**: Uses latent variables z ~ N(0,I) and conditioning on start states
- **Manifold-Aware**:
  - Pendulum: S¹×ℝ manifold with circular angle handling
  - CartPole: ℝ²×S¹×ℝ manifold with mixed Euclidean/circular structure
- **Facebook Flow Matching Library**: Uses geodesic probability paths and Riemannian ODE solvers
- **Endpoint Prediction**: Predict final states from initial conditions
- **Flow Path Generation**: Generate complete trajectories through phase space
- **Attractor Convergence**: Models trained to predict convergence to system attractors
- **Uncertainty Quantification**: Probabilistic predictions with multiple samples

### Evaluation Capabilities
- **Unified Metrics**: MSE, MAE, attractor accuracy across all model types
- **Phase Space Visualization**: Consistent plotting with π-based labels
- **Flow Path Analysis**: Single and multi-trajectory visualization
- **Basin Analysis**: State space discretization and attractor basin mapping
- **Separatrix Detection**: Identify boundary regions between attractor basins
- **Multi-Resolution Analysis**: Compare basin structure at different grid resolutions
- **Comprehensive Reports**: Automated generation of evaluation summaries

## Advanced Analysis Features

### Attractor Basin Analysis
The system provides advanced analysis of dynamical system behavior:

- **Configurable Resolution**: Grid discretization from coarse (0.2) to fine (0.05) resolution
- **Batch Processing**: Efficient evaluation of large state space grids
- **Basin Classification**: Automatic classification of state space regions by attractor
- **Separatrix Mapping**: Detection of points that don't converge to any attractor
- **Statistical Analysis**: Quantitative analysis of basin sizes and convergence properties
- **Multi-Format Output**: Visualizations, raw data, and text reports

### Visualization Outputs
Generated analysis includes:
- `attractor_basins.png`: Color-coded basin map
- `basin_statistics.png`: Statistical analysis plots
- `attractor_basins_grid_points.png`: High-resolution grid visualization
- `basin_analysis_data.npz`: Raw numerical data
- `basin_analysis_report.txt`: Comprehensive text summary
- `resolution_comparison.png`: Multi-resolution comparison plots

## Architecture Benefits

### Code Organization
- **Eliminated Redundancy**: Removed ~200+ lines of duplicate evaluation/plotting code
- **Centralized Configuration**: Single source for system parameters (attractors, bounds, etc.)
- **Modular Design**: Reusable components for metrics, visualization, and analysis
- **Consistent APIs**: Uniform interfaces across all evaluation functionality

### Extensibility
- **Easy Integration**: New models can use existing evaluation pipeline
- **Configurable Analysis**: Basin analysis resolution and parameters easily adjustable
- **Plugin Architecture**: New metrics and visualizations easily added
- **Cross-Model Compatibility**: Same evaluation tools work across all model types

### Research Capabilities
- **Deep Insights**: Basin analysis reveals model behavior patterns invisible before
- **Quantitative Analysis**: Automated statistical analysis of dynamical behavior
- **Comparative Studies**: Multi-resolution and cross-model comparison tools
- **Publication Ready**: Automated generation of high-quality visualizations and reports