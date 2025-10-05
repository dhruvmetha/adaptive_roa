# CartPole LCFM System Documentation

## Overview

CartPole Latent Conditional Flow Matching (LCFM) system for learning dynamics and endpoint prediction on the ℝ² × S¹ × ℝ manifold.

**State representation:** (x, ẋ, θ, θ̇)
- x ∈ ℝ: Cart position
- ẋ ∈ ℝ: Cart velocity
- θ ∈ S¹: Pole angle (circular)
- θ̇ ∈ ℝ: Pole angular velocity

## 1. Dataset Building

### Prerequisites
- Raw trajectory data in: `/common/users/shared/pracsys/genMoPlan/data_trajectories/cartpole/trajectories/`
- Shuffled indices file: `/common/users/shared/pracsys/genMoPlan/data_trajectories/cartpole/shuffled_indices.txt`
- Data bounds file: `/common/users/dm1487/arcmg_datasets/cartpole/cartpole_data_bounds.pkl`

### Build Endpoint Dataset
```bash
# Activate environment
conda activate /common/users/dm1487/envs/arcmg

# Build endpoint dataset (start_state, attractor_state pairs)
python src/build_shuffled_endpoint_dataset.py --config-name=build_cartpole_endpoint_dataset.yaml
```

**Configuration:** `configs/build_cartpole_endpoint_dataset.yaml`
```yaml
system:
    _target_: src.systems.cartpole_lcfm.CartPoleSystemLCFM

data_dirs: /common/users/shared/pracsys/genMoPlan/data_trajectories/cartpole/trajectories
shuffled_idxs_file: /common/users/shared/pracsys/genMoPlan/data_trajectories/cartpole/shuffled_indices.txt
dest_dir: /common/users/dm1487/arcmg_datasets/cartpole/incremental_endpoint_dataset

# Number of trajectories to process (increment as needed)
increment: 500
```

**Output:** 
- Dataset saved as: `{dest_dir}/{increment}_endpoint_dataset.txt`
- Format: Each line contains `[x_start, ẋ_start, θ_start, θ̇_start, x_end, ẋ_end, θ_end, θ̇_end]`

### Data Processing Details
1. **Trajectory Processing**: Reads comma-separated trajectory files
2. **Attractor Detection**: Identifies first state that reaches balanced CartPole configuration
3. **Endpoint Extraction**: Creates (start, attractor) pairs for all states before convergence
4. **Data Bounds**: Uses actual data bounds from pickle file:
   - Cart position: ±2.064
   - Cart velocity: ±128.146 
   - Pole angle: wrapped to ±π
   - Angular velocity: ±18.121

## 2. Training

### Configuration
**Main config:** `configs/train_cartpole_latent_conditional_flow_matching.yaml`
```yaml
defaults:
  - _self_
  - data: cartpole_endpoint_data
  - model: cartpole_latent_conditional_unet
  - optimizer: adamw
  - scheduler: reduce_lr_on_plateau
  - trainer: flow_matching
  - device: gpu3

seed: 42
num_workers: 4
batch_size: 256
check_val_every_n_epoch: 1
name: cartpole_latent_conditional_fm

# Flow matching specific config
flow_matching:
  latent_dim: 2  # Dimension of Gaussian latent variable
  noise_distribution: "uniform"
  noise_scale: 1.0
  noise_bounds: null  # Use default bounds from CartPole system
  num_integration_steps: 100
  sigma: 0.0
```

**Data config:** `configs/data/cartpole_endpoint_data.yaml`
```yaml
_target_: src.data.cartpole_endpoint_data.CartPoleEndpointDataModule
data_file: /common/users/dm1487/arcmg_datasets/cartpole/incremental_endpoint_dataset/1000_endpoint_dataset.txt
validation_file: /common/users/dm1487/arcmg_datasets/cartpole/incremental_endpoint_dataset/validation_endpoint_dataset.txt
batch_size: 128
num_workers: 4
pin_memory: true
```

**Trainer config:** `configs/trainer/flow_matching.yaml`
```yaml
_target_: lightning.pytorch.Trainer
max_epochs: 500
accelerator: ${device.accelerator}
devices: ${device.devices}
precision: 32
gradient_clip_val: 1.0
val_check_interval: 1.0
log_every_n_steps: 10

logger:
  _target_: lightning.pytorch.loggers.TensorBoardLogger
  name: ${name}
  save_dir: "${hydra:runtime.output_dir}"

callbacks:
  - _target_: lightning.pytorch.callbacks.ModelCheckpoint
    monitor: "val_loss"
    mode: "min"
    save_top_k: 3
    save_last: true
    every_n_epochs: 1
  - _target_: src.callbacks.validation_inference_callback.ValidationInferenceCallback
    inference_frequency: 10  # Run validation inference every 10 epochs
    num_integration_steps: 100
```

### Run Training
```bash
# Activate environment
conda activate /common/users/dm1487/envs/arcmg

# Train CartPole LCFM model
python src/flow_matching/cartpole_latent_conditional/train.py

# Alternative: using config override
python src/flow_matching/cartpole_latent_conditional/train.py data.data_file=/path/to/your/dataset.txt
```

### Training Architecture
- **Model**: CartPoleLatentConditionalUNet1D
  - Input: 5D embedded state (x_norm, ẋ_norm, sin θ, cos θ, θ̇_norm) + time + latent + condition
  - Output: 4D velocity in tangent space (dx/dt, dẋ/dt, dθ/dt, dθ̇/dt)
  - Hidden dims: [256, 512, 256]
  - Time embedding: 64D sinusoidal
  - Latent dim: 2D Gaussian

- **Flow Matching**: Interpolates in ℝ² × S¹ × ℝ space
  - ℝ components: Linear interpolation
  - S¹ component: Geodesic interpolation on circle
  - Target velocities: Normalized by system bounds for stable training

- **Loss Function**: MSE between predicted and target velocities
  - **Before fix**: Loss 7-60 (unnormalized velocities ±120)
  - **After fix**: Loss ~1.0 (normalized velocities ±3)

### Training Features

#### **Validation Inference (New)**
- **Automatic**: Runs every 10 epochs during training
- **Unified Methods**: Uses same inference pipeline as standalone scripts
- **TensorBoard Logging**: Complete metrics logged under `val_inference/` namespace
  - `val_inference/mse` - Overall endpoint prediction error
  - `val_inference/mae` - Mean absolute error
  - `val_inference/x_mse`, `val_inference/theta_mse` - Component-wise losses
  - `val_inference/position_contrib`, `val_inference/angle_contrib` - Error contributions

#### **Unified Architecture** 
- **Training**: Flow matching loss using interpolated x_t
- **Validation**: Direct endpoint prediction using model.predict_endpoint()
- **Standalone**: Same predict_endpoint() method used by inference scripts
- **Consistency**: All paths use identical `normalize → embed → model()` pipeline

### Training Outputs
- **Checkpoints**: `outputs/cartpole_latent_conditional_fm/{timestamp}/checkpoints/`
- **Logs**: TensorBoard logs in `outputs/cartpole_latent_conditional_fm/{timestamp}/`
- **Best model**: Selected by validation loss
- **Validation metrics**: Logged every 10 epochs with full inference evaluation

## 3. Inference

### Direct Model Inference (Training-time)
During training, the model has unified inference methods built-in:

```python
# Access the trained model directly
model = trainer.model  # CartPoleLatentConditionalFlowMatcher

# Define start states (raw coordinates)
start_states = torch.tensor([
    [0.5, 2.0, 0.1, 1.0],   # (x, ẋ, θ, θ̇)
    [-0.8, -3.0, -0.5, -2.0]
])

# **UNIFIED METHODS** - Same as validation callback uses
endpoints = model.predict_endpoint(start_states, num_steps=100)
print(f"Predicted endpoints shape: {endpoints.shape}")  # [2, 4]

# Multiple samples per start state
batch_endpoints = model.predict_endpoints_batch(start_states, num_samples=10)
print(f"Batch endpoints shape: {batch_endpoints.shape}")  # [20, 4]

# Generate full trajectory with history
final_states, trajectory = model.integrate_trajectory(start_states, num_steps=100)
print(f"Trajectory length: {len(trajectory)}")  # List of 101 states
```

### Standalone Inference (Post-training)
Load a trained model for standalone inference:

```python
from src.flow_matching.cartpole_latent_conditional.inference import CartPoleLatentConditionalFlowMatchingInference

# Load from timestamped training folder
inferencer = CartPoleLatentConditionalFlowMatchingInference(
    "outputs/cartpole_latent_conditional_fm/2024-01-15_14-30-45"
)

# Uses same unified methods internally
endpoints = inferencer.predict_endpoint(start_states, num_steps=100)
```

### Integration Method
- **Normalized Integration**: All integration happens in [-1,1]² × [-π,π] × [-1,1] space
- **SO(2) Integration**: Uses Theseus exponential maps for proper angular integration
- **Single Denormalization**: Only converts to raw coordinates at final output
- **Benefits**:
  - Numerical stability (bounded integration domain)
  - Consistent with training normalization
  - Proper manifold structure preservation

### Inference Pipeline
1. **Normalize** start states to [-1,1]² × [-π,π] × [-1,1]
2. **Integrate** using normalized-space integrator with model velocity predictions
3. **Denormalize** final results back to raw coordinates
4. **Output** final states in original physical units

## 4. Key Components

### Files Structure
```
src/
├── systems/
│   ├── cartpole.py                     # Base CartPole system
│   └── cartpole_lcfm.py               # LCFM-specific CartPole system
├── model/
│   └── cartpole_latent_conditional_unet1d.py  # Neural network architecture
├── flow_matching/cartpole_latent_conditional/
│   ├── flow_matcher.py                 # Training logic + unified inference methods
│   ├── inference.py                   # Standalone inference wrapper
│   └── train.py                       # Training script
├── callbacks/
│   └── validation_inference_callback.py  # Validation inference callback
├── data/
│   └── cartpole_endpoint_data.py      # Data loading module (with separate validation file)
└── configs/
    ├── train_cartpole_latent_conditional_flow_matching.yaml
    ├── data/cartpole_endpoint_data.yaml
    └── trainer/flow_matching.yaml     # Includes validation callback
```

### System Bounds (Auto-loaded from data)
```python
# Actual bounds from CartPole trajectory data:
cart_limit: 2.064          # Cart position ±2.064
velocity_limit: 128.146    # Cart velocity ±128.146  
angle_limit: π             # Pole angle ±π (wrapped)
angular_velocity_limit: 18.121  # Angular velocity ±18.121
```

### Attractors
CartPole balanced states (all equivalent after wrapping):
```python
attractors = [
    [0.0, 0.0, 0.0, 0.0],    # Cart centered, pole upright (0°)
    [0.0, 0.0, π, 0.0],      # Cart centered, pole upright (180°)
    [0.0, 0.0, -π, 0.0],     # Cart centered, pole upright (-180°)
]
```

## 5. Expected Performance

### Training Metrics
- **Initial Loss**: ~1.0-3.0 (after normalization fix)
- **Convergence**: Should decrease steadily over epochs
- **Final Loss**: ~0.1-0.5 for well-trained model

### Inference Quality
- **Trajectory Smoothness**: Integration produces physically reasonable paths
- **Attractor Convergence**: Generated endpoints should be in balanced regions
- **State Bounds**: All generated states within physical limits

## 6. Troubleshooting

### High Loss Values
- **Symptom**: Loss > 10 during training
- **Cause**: Velocity targets not normalized by system bounds
- **Fix**: Ensure flow_matcher.py divides velocity targets by respective bounds

### Integration Instability
- **Symptom**: States go out of bounds during integration
- **Cause**: Integration in raw coordinates with large velocity values
- **Fix**: Use normalized integrator with bounded [-1,1] space

### Dimension Mismatches
- **Symptom**: Model input/output dimension errors
- **Cause**: Inconsistent embedding dimensions
- **Fix**: Verify embedded_dim=5, output_dim=4, condition_dim=5

## 7. Extensions

### Larger Datasets
```bash
# Build larger dataset by increasing increment
python src/build_shuffled_endpoint_dataset.py --config-name=build_cartpole_endpoint_dataset.yaml increment=2000
```

### Different Latent Dimensions
```yaml
# In training config
model:
  latent_dim: 4  # Try different latent dimensions
```

### Custom Bounds
```python
# Modify CartPoleSystemLCFM constructor for different bounds
system = CartPoleSystemLCFM(
    bounds_file="/path/to/custom_bounds.pkl",
    use_dynamic_bounds=True
)
```