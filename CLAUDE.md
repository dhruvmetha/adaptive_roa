# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a robotics research project for training neural network classifiers, reachability models, and flow matching models for AI Olympics. The codebase uses PyTorch Lightning with Hydra configuration management and features unified evaluation/visualization modules with advanced attractor basin analysis.

## Architecture

### Core Components

- **Training Scripts**: Located in `src/` directory
  - `train_classifier.py`: Trains binary classifiers on 4D input data
  - `train_reachability.py`: Trains reachability models on 8D state data  
  - `train_flow_matching.py`: Trains flow matching models for pendulum dynamics
  - `train_circular_flow_matching.py`: Trains circular-aware flow matching models
  - `evaluate_reachability.py`: Evaluates trained reachability models
  - `evaluate_flow_matching_refactored.py`: Unified evaluation pipeline for flow matching models

- **Data Modules**: Located in `src/data/`
  - `classification_data.py`: Handles 4D classification data with 80/10/10 train/val/test splits
  - `reachability_data.py`: Manages 8D reachability data from separate train/valid/test files
  - `endpoint_data.py`: Handles endpoint prediction data for flow matching
  - `circular_endpoint_data.py`: Circular-aware endpoint data handling

- **Models**: Located in `src/model/`
  - `simple_mlp.py`: Multi-layer perceptron with configurable hidden layers
  - `unet1d.py`: 1D U-Net architecture for flow matching
  - `circular_unet1d.py`: Circular-aware U-Net for pendulum dynamics

- **Lightning Modules**: Located in `src/module/`
  - `simple_mlp.py`: Lightning wrapper for MLP models
  - `flow_matching.py`: **[DEPRECATED]** Legacy flow matching module
  - `circular_flow_matching.py`: **[DEPRECATED]** Legacy circular flow matching module

- **Unified Flow Matching Framework**: Located in `src/flow_matching/`
  - `base/`: Abstract base classes and common functionality
    - `flow_matcher.py`: Base Lightning module for all flow matching variants
    - `inference.py`: Base inference class with shared prediction logic
    - `config.py`: Common configuration and state handling
  - `standard/`: Standard flow matching implementation
    - `flow_matcher.py`: Standard flow matching using torchcfm
    - `inference.py`: Standard flow matching inference
    - `train.py`: Standard training script
  - `circular/`: Circular-aware flow matching implementation  
    - `flow_matcher.py`: Circular flow matching with geodesic interpolation
    - `inference.py`: Circular flow matching inference
    - `train.py`: Circular training script
  - `utils/`: Shared utilities
    - `state_transformations.py`: State embedding/extraction utilities
    - `geometry.py`: Circular distance and interpolation utilities

- **Legacy Inference**: Located in `src/` directory **[DEPRECATED]**
  - `inference_flow_matching.py`: **[DEPRECATED]** Use `flow_matching.standard.inference` instead
  - `inference_circular_flow_matching.py`: **[DEPRECATED]** Use `flow_matching.circular.inference` instead

### Unified Evaluation & Visualization System

- **Systems Configuration**: Located in `src/systems/`
  - `pendulum_config.py`: Centralized pendulum system parameters (attractors, bounds, normalization)
  - `base.py`: Base system interface
  - `pendulum.py`: Pendulum dynamics implementation

- **Evaluation Framework**: Located in `src/evaluation/`
  - `metrics.py`: Centralized metrics computation (MSE, MAE, attractor accuracy, circular distances)
  - `evaluator.py`: Unified evaluation pipeline for all model types

- **Visualization Framework**: Located in `src/visualization/`
  - `phase_space_plots.py`: Standard phase space plotting with consistent styling
  - `flow_visualizer.py`: Flow path and trajectory visualization
  - `attractor_analysis.py`: **Advanced basin analysis and separatrix detection**

### Attractor Basin Analysis (NEW FEATURE)

The `AttractorBasinAnalyzer` provides state space discretization and basin mapping:

- **State Space Discretization**: Create grids with configurable resolution (default 0.1)
- **Basin Classification**: Determine which attractor each state converges to
- **Separatrix Detection**: Identify boundary regions and points that don't converge to attractors
- **Comprehensive Visualization**: Color-coded basin maps, statistics, and multi-resolution analysis
- **Data Export**: Save analysis results, visualizations, and numerical data

### Configuration System

Uses Hydra for configuration management with YAML files in `configs/`:
- Main configs: `train_classifier.yaml`, `train_reachability.yaml`, `evaluate_reachability.yaml`
- Data configs: `configs/data/classifier_data.yaml`, `configs/data/reachability_data.yaml`  
- Model configs: `configs/model/simple_mlp.yaml`
- Trainer configs: `configs/trainer/default.yaml`, `configs/trainer/gpu.yaml`

### Data Processing

- **Classification Data**: 4D inputs normalized to [0,1] range with bounds [-0.5, 0.5, -10, 10]
- **Reachability Data**: 8D inputs normalized to [0,1] range with bounds [-4π, 4π, -20, 20] repeated twice

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
# Train classifier (currently incomplete - model/trainer instantiation commented out)
python src/train_classifier.py

# Train reachability model  
python src/train_reachability.py

# Train flow matching models using unified framework
python src/flow_matching/standard/train.py      # Standard flow matching
python src/flow_matching/circular/train.py     # Circular flow matching

# Legacy training scripts (deprecated but still functional)
python src/train_flow_matching.py              # [DEPRECATED] Use standard/train.py
python src/train_circular_flow_matching.py     # [DEPRECATED] Use circular/train.py

# Evaluate reachability model
python src/evaluate_reachability.py

# Evaluate flow matching model with unified pipeline
python src/evaluate_flow_matching_refactored.py
```

### Evaluation & Analysis
```bash
# Run comprehensive flow matching evaluation (unified pipeline)
python src/evaluate_flow_matching_refactored.py

# Demonstrate unified flow matching framework
python src/demo_unified_flow_matching.py        # NEW: Works with both variants

# Demonstrate attractor basin analysis
python src/demo_attractor_analysis.py

# Legacy demos (deprecated but still functional)
python src/demo_flow_matching.py               # [DEPRECATED] Use unified demo
python src/demo_circular_flow_matching.py      # [DEPRECATED] Use unified demo
```

### Unified Flow Matching Usage
```python
# NEW: Unified framework usage
from src.flow_matching.standard import StandardFlowMatchingInference
from src.flow_matching.circular import CircularFlowMatchingInference
from src.evaluation.evaluator import FlowMatchingEvaluator

# Load any variant - automatic detection
standard_inferencer = StandardFlowMatchingInference("checkpoint.ckpt")
circular_inferencer = CircularFlowMatchingInference("checkpoint.ckpt")

# Unified evaluation with automatic variant detection
evaluator = FlowMatchingEvaluator(auto_detect_variant=True)
results = evaluator.evaluate_on_dataloader(inferencer, test_loader, data_module)
```

### Attractor Basin Analysis Usage
```python
# Works with both flow matching variants
from src.visualization.attractor_analysis import AttractorBasinAnalyzer
from src.systems.pendulum_config import PendulumConfig

# Initialize analyzer
config = PendulumConfig()
analyzer = AttractorBasinAnalyzer(config)

# Run analysis (automatically detects variant)
results = analyzer.analyze_attractor_basins(
    inferencer,         # Works with standard OR circular
    resolution=0.1,     # Grid resolution (configurable)
    batch_size=1000     # Batch size for efficiency
)

# Save complete analysis
analyzer.save_analysis_results("output_dir", results)
```

### Migration Guide
```python
# OLD (deprecated):
from src.inference_flow_matching import FlowMatchingInference
from src.inference_circular_flow_matching import CircularFlowMatchingInference

# NEW (recommended):
from src.flow_matching.standard import StandardFlowMatchingInference  
from src.flow_matching.circular import CircularFlowMatchingInference

# Benefits of new architecture:
# • Automatic variant detection
# • Shared base functionality  
# • Reduced code duplication (~230 lines eliminated)
# • Consistent APIs across variants
# • Easy to extend with new variants
```

### GPU Configuration
The scripts are configured to use GPU 1 via `os.environ["CUDA_VISIBLE_DEVICES"] = "1"` in each training script.

### Outputs
Training outputs are saved to `outputs/` directory with timestamped subdirectories containing:
- Training logs
- Model checkpoints (in `logs/vae_training/version_X/checkpoints/`)
- TensorBoard event files

## Data Requirements

- **Classification data**: Text files with space-separated values (4 inputs + 1 label per line)
- **Reachability training data**: Directory containing `train.txt` and `valid.txt`  
- **Reachability test data**: Single test file specified separately
- **Flow matching data**: Endpoint prediction datasets with start and end state pairs
- **Circular flow matching data**: Endpoint data with circular angle handling for pendulum dynamics

## Model Features

### SimpleMLP Models
- Configurable hidden layer architecture
- Binary classification with BCEWithLogitsLoss
- Comprehensive metrics: accuracy, precision, recall, F1, confusion matrix
- Detailed test metrics including TPR, FPR, TNR, FNR
- Lightning integration with automatic logging

### Flow Matching Models
- **1D U-Net Architecture**: Efficient neural ODE for continuous dynamics
- **Circular Flow Matching**: Specialized for pendulum dynamics with proper angle handling
- **Endpoint Prediction**: Predict final states from initial conditions
- **Flow Path Generation**: Generate complete trajectories through phase space
- **Attractor Convergence**: Models trained to predict convergence to system attractors

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