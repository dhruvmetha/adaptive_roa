# Pendulum Flow Matching System Documentation

## Overview

Pendulum system for Flow Matching with S¹ × ℝ manifold structure for learning nonlinear dynamics and attractor basin analysis.

**State representation:** (θ, θ̇)
- θ ∈ S¹: Pendulum angle (circular)
- θ̇ ∈ ℝ: Angular velocity (normalized)

## 1. Dataset Building

### Prerequisites
- Raw trajectory data in: `/common/users/dm1487/arcmg_datasets/pendulum_lqr/pendulum_lqr5k/pendulum_lqr5k/`
- For LCFM: Need endpoint pairs (start_state, attractor_state)

### Build Endpoint Dataset
```bash
# Activate environment
conda activate /common/users/dm1487/envs/arcmg

# Build endpoint dataset for pendulum
python src/build_endpoint_dataset.py --config-name=build_endpoint_dataset.yaml
```

**Configuration:** `configs/build_endpoint_dataset.yaml`
```yaml
system:
    _target_: src.systems.pendulum.Pendulum
    name: pendulum
    attractor_radius: 0.01

data_dirs: /common/users/dm1487/arcmg_datasets/pendulum_lqr/pendulum_lqr5k/pendulum_lqr5k
dest_dir: /common/users/dm1487/arcmg_datasets/pendulum_lqr/pendulum_lqr5k/endpoint_dataset
```

**Output Format:**
- Each line: `[θ_start, θ̇_start, θ_end, θ̇_end]`
- Comma-separated trajectory files processed
- Only convergent trajectories (reaching attractors) included

### Alternative: Pre-built Dataset
Use existing circular endpoint datasets:
```
/common/users/dm1487/arcmg_datasets/pendulum_lqr/incremental_endpoint_dataset/
├── 100_endpoint_dataset.txt
├── 500_endpoint_dataset.txt  
├── 1000_endpoint_dataset.txt
└── 2000_endpoint_dataset.txt
```

## 2. Training

### Standard Flow Matching
```bash
# Train standard pendulum flow matching
python src/flow_matching/standard/train.py --config-name=train_pendulum_flow_matching.yaml
```

### Circular Flow Matching (Recommended)
```bash
# Train circular-aware pendulum flow matching
python src/flow_matching/circular/train.py --config-name=train_circular_flow_matching.yaml
```

### Configuration Files

**Standard Flow Matching:** `configs/train_pendulum_flow_matching.yaml`
```yaml
defaults:
  - model: unet1d
  - data: endpoint_data
  - trainer: default
  - _self_

model:
  embedded_dim: 3        # (sin θ, cos θ, θ̇_norm)
  time_emb_dim: 64
  hidden_dims: [256, 512, 256]

data:
  data_file: /common/users/dm1487/arcmg_datasets/pendulum_lqr/incremental_endpoint_dataset/1000_endpoint_dataset.txt
  batch_size: 64

trainer:
  max_epochs: 100
  
learning_rate: 1e-4
```

**Circular Flow Matching:** `configs/train_circular_flow_matching.yaml`
```yaml
defaults:
  - model: circular_unet1d
  - data: circular_endpoint_data
  - trainer: default
  - _self_

model:
  embedded_dim: 3        # Embedding for S¹ × ℝ
  time_emb_dim: 64
  hidden_dims: [256, 512, 256]

data:
  data_file: /common/users/dm1487/arcmg_datasets/pendulum_lqr/incremental_endpoint_dataset/1000_endpoint_dataset.txt
  batch_size: 64
  
trainer:
  max_epochs: 100
  
learning_rate: 1e-4
```

### Training Architecture Differences

**Standard Flow Matching:**
- Input: 3D embedded state (sin θ, cos θ, θ̇_norm) + time
- Output: 2D velocity (dθ/dt, dθ̇/dt) 
- Interpolation: Linear in embedded space
- Loss: MSE on embedded velocity predictions

**Circular Flow Matching:**
- Input: 3D embedded state (sin θ, cos θ, θ̇_norm) + time
- Output: 2D velocity in tangent space
- Interpolation: Geodesic on S¹, linear on ℝ
- Loss: MSE on tangent space velocities
- **Advantage**: Respects S¹ × ℝ manifold structure

## 3. Inference

### Load Trained Models
```python
# Standard flow matching
from src.flow_matching.standard.inference import StandardFlowMatchingInference

inferencer = StandardFlowMatchingInference(
    "path/to/standard_checkpoint.ckpt"
)

# Circular flow matching (recommended)
from src.flow_matching.circular.inference import CircularFlowMatchingInference

inferencer = CircularFlowMatchingInference(
    "path/to/circular_checkpoint.ckpt"
)
```

### Generate Predictions
```python
import torch

# Define start states (raw coordinates)
start_states = torch.tensor([
    [0.1, 0.5],    # (θ, θ̇) - small angle, positive velocity
    [2.8, -1.0],   # Near inverted, negative velocity  
    [-1.5, 0.3]    # Moderate negative angle, positive velocity
])

# Generate endpoint predictions
endpoints = inferencer.predict_endpoints(start_states, num_samples=10)
print(f"Predicted endpoints: {endpoints.shape}")  # [3, 10, 2]

# Generate trajectory
trajectory = inferencer.generate_trajectory(
    start_state=start_states[0],
    num_steps=100
)
print(f"Trajectory length: {len(trajectory)}")  # List of 101 states
```

### Integration Methods

**Standard Integration:**
- Works in embedded space throughout
- May not preserve circular structure perfectly
- Faster but less geometrically accurate

**Circular Integration:**
- Proper geodesic integration on S¹
- Preserves manifold structure
- More geometrically faithful to pendulum dynamics

## 4. System Configuration

### Pendulum System Parameters
```python
# Located in src/systems/pendulum_config.py
class PendulumConfig:
    # State bounds for normalization
    angle_bounds = (-π, π)              # Natural S¹ range
    angular_velocity_bounds = (-2π, 2π) # Typical range for pendulum
    
    # Attractors in S¹ × ℝ space
    attractors = [
        [0.0, 0.0],    # Bottom equilibrium (stable)
        [2.1, 0.0],    # Top-right equilibrium  
        [-2.1, 0.0],   # Top-left equilibrium
    ]
    
    attractor_radius = 0.1  # Basin membership threshold
```

### Data Processing
- **Angle Wrapping**: θ values wrapped to [-π, π]
- **Velocity Normalization**: θ̇ normalized to [-1, 1] for training
- **Embedding**: State (θ, θ̇) → (sin θ, cos θ, θ̇_norm)

## 5. Evaluation & Visualization

### Unified Evaluation
```bash
# Comprehensive evaluation with visualizations
python src/evaluate_flow_matching_refactored.py checkpoint=path/to/model.ckpt
```

### Attractor Basin Analysis
```python
from src.visualization.attractor_analysis import AttractorBasinAnalyzer
from src.systems.pendulum_config import PendulumConfig

# Initialize basin analyzer
config = PendulumConfig()
analyzer = AttractorBasinAnalyzer(config)

# Analyze basins (works with both standard and circular models)
results = analyzer.analyze_attractor_basins(
    inferencer,
    resolution=0.1,        # Grid resolution in phase space
    batch_size=1000       # Batch size for efficiency
)

# Save complete analysis
analyzer.save_analysis_results("basin_analysis_output/", results)
```

**Outputs:**
- `attractor_basins.png`: Color-coded basin map
- `basin_statistics.png`: Statistical analysis
- `basin_analysis_data.npz`: Raw numerical data
- `basin_analysis_report.txt`: Text summary

### Advanced Visualizations
```python
from src.visualization.flow_visualizer import FlowVisualizer

visualizer = FlowVisualizer(config)

# Visualize flow field
visualizer.plot_flow_field(
    inferencer, 
    grid_resolution=0.2,
    save_path="flow_field.png"
)

# Plot sample trajectories
visualizer.plot_trajectories(
    inferencer,
    start_states=start_states,
    num_steps=100,
    save_path="trajectories.png"
)
```

## 6. Performance Comparison

### Expected Training Metrics

**Standard Flow Matching:**
- Initial loss: ~2.0-4.0
- Final loss: ~0.5-1.0
- Training time: ~30-60 min for 100 epochs

**Circular Flow Matching:**
- Initial loss: ~1.0-2.0
- Final loss: ~0.2-0.5  
- Training time: ~40-80 min for 100 epochs
- **Better geometric accuracy**

### Inference Quality Metrics
- **MSE**: Mean squared error in state predictions
- **Circular Distance**: Proper distance on S¹ manifold
- **Attractor Accuracy**: Percentage reaching correct basins
- **Flow Consistency**: Smoothness of generated trajectories

## 7. Key Files

### Core Components
```
src/
├── systems/
│   ├── pendulum.py                 # Base pendulum system
│   ├── pendulum_lcfm.py           # LCFM pendulum variant
│   └── pendulum_config.py         # Centralized configuration
├── flow_matching/
│   ├── standard/
│   │   ├── flow_matcher.py        # Standard training
│   │   ├── inference.py           # Standard inference
│   │   └── train.py               # Standard training script
│   ├── circular/
│   │   ├── flow_matcher.py        # Circular training
│   │   ├── inference.py           # Circular inference  
│   │   └── train.py               # Circular training script
│   └── utils/
│       └── geometry.py            # Circular geometry utilities
├── model/
│   ├── unet1d.py                  # Standard UNet
│   └── circular_unet1d.py         # Circular UNet
└── data/
    ├── endpoint_data.py           # Standard data loader
    └── circular_endpoint_data.py  # Circular data loader
```

### Legacy Components (Deprecated)
```
src/
├── train_flow_matching.py              # Use standard/train.py instead
├── train_circular_flow_matching.py     # Use circular/train.py instead
├── inference_flow_matching.py          # Use standard/inference.py instead
└── inference_circular_flow_matching.py # Use circular/inference.py instead
```

## 8. Advanced Features

### Multi-Resolution Basin Analysis
```python
# Compare basin structure at different resolutions
analyzer.compare_resolutions(
    inferencer,
    resolutions=[0.2, 0.1, 0.05],
    save_path="resolution_comparison.png"
)
```

### Separatrix Detection
```python
# Find boundary regions between basins
separatrix_points = analyzer.detect_separatrix(
    inferencer,
    resolution=0.05,
    threshold=0.1
)
```

### Custom Attractor Definitions
```python
# Define custom attractors for specialized analysis
custom_attractors = [
    [0.0, 0.0],     # Bottom
    [π/2, 0.0],     # Right
    [-π/2, 0.0],    # Left  
    [π, 0.0]        # Top
]

config.attractors = custom_attractors
analyzer = AttractorBasinAnalyzer(config)
```

## 9. Troubleshooting

### Common Issues

**Training Divergence:**
- Reduce learning rate to 1e-5
- Increase batch size to 128+
- Check data normalization

**Poor Basin Classification:**
- Increase attractor radius for easier convergence
- Use circular flow matching for better geometry
- Verify attractor definitions match system dynamics

**Integration Instability:**
- Reduce integration step size (increase num_steps)
- Use circular integration for S¹ components
- Check velocity magnitude bounds

### Performance Optimization
- Use GPU training: `trainer.accelerator=gpu`
- Increase num_workers: `data.num_workers=8`
- Batch size tuning: Start with 64, increase to 128-256

## 10. Research Applications

### Basin Analysis Studies
- Quantify attractor basin shapes and sizes
- Study separatrix structure and stability
- Compare different control strategies

### Dynamics Learning
- Learn nonlinear pendulum dynamics from data
- Generate realistic trajectory samples
- Predict long-term evolution from initial conditions

### Control Applications
- Trajectory planning in phase space
- Basin-aware control design  
- Robustness analysis under perturbations