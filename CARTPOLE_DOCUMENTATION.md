# CartPole LCFM System Documentation

## Overview

CartPole Latent Conditional Flow Matching (LCFM) system for learning dynamics and endpoint prediction on the ℝ² × S¹ × ℝ manifold.

**State representation:** (x, θ, ẋ, θ̇)
- x ∈ ℝ: Cart position
- θ ∈ S¹: Pole angle (circular)
- ẋ ∈ ℝ: Cart velocity
- θ̇ ∈ ℝ: Pole angular velocity

**What does this system do?**
This system learns to predict where a CartPole system will end up (attractor) given a starting state, using a stochastic flow matching approach. The model integrates through phase space using manifold-aware ODEs to predict trajectories that converge to balanced (upright pole) configurations.

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
- Format: Each line contains `[x_start, θ_start, ẋ_start, θ̇_start, x_end, θ_end, ẋ_end, θ̇_end]`

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

## 3. Inference & Evaluation

### Overview of Inference Pipeline
The model predicts where CartPole states will converge by integrating forward through time using the learned velocity field. The process involves:

1. **Start with noisy input** sampled uniformly from state space
2. **Sample latent variable** z ~ N(0,I) for stochasticity
3. **Integrate forward** using RiemannianODESolver with manifold-aware geodesics
4. **Output prediction** of final attractor state

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
The inference uses Facebook Flow Matching's **RiemannianODESolver** for manifold-aware integration:

- **Normalized Integration**: All integration happens in normalized space
  - Cart position & velocities: [-1, 1]
  - Pole angle: [-π, π] (natural circular space)
- **Manifold Structure**: ℝ²×S¹×ℝ (Euclidean × FlatTorus × Euclidean)
  - Geodesic interpolation on S¹ component (proper angle wrapping)
  - Linear interpolation on ℝ components
- **Projection Operations**:
  - `projx=True`: Projects states back onto manifold (wraps angles to [-π, π])
  - `proju=True`: Projects velocities to tangent space
- **Benefits**:
  - Numerical stability (bounded integration domain)
  - Mathematically correct angular dynamics
  - Consistent with training normalization

### Inference Pipeline (Detailed)
1. **Normalize** start states: raw coordinates → normalized space
   ```
   x_norm = x / cart_limit
   θ (already wrapped to [-π, π])
   ẋ_norm = ẋ / velocity_limit
   θ̇_norm = θ̇ / angular_velocity_limit
   ```

2. **Embed** for neural network input: normalized → embedded
   ```
   [x_norm, θ, ẋ_norm, θ̇_norm] → [x_norm, sin(θ), cos(θ), ẋ_norm, θ̇_norm]
   ```

3. **Sample** random inputs and latent variables
   - Noisy input: uniform sampling from full state space
   - Latent z ~ N(0, I) provides stochasticity

4. **Integrate** using RiemannianODESolver
   - Start from noisy input (normalized)
   - Follow velocity field predicted by neural network
   - Apply manifold projection at each step
   - 100 integration steps from t=0 to t=1

5. **Denormalize** final output: normalized → raw coordinates
   ```
   x = x_norm × cart_limit
   θ (remains in [-π, π])
   ẋ = ẋ_norm × velocity_limit
   θ̇ = θ̇_norm × angular_velocity_limit
   ```

6. **Output** predicted endpoint in original physical units

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

## 5. ROA (Region of Attraction) Evaluation

The system can be evaluated on labeled ROA data to assess how well it predicts which states will successfully reach the balanced attractor.

### Running ROA Evaluation
```bash
# Evaluate with auto-detected checkpoint
python src/flow_matching/evaluate_roa.py --config-name=evaluate_cartpole_roa

# Evaluate with specific checkpoint
python src/flow_matching/evaluate_roa.py --config-name=evaluate_cartpole_roa \
    checkpoint.path=outputs/cartpole_latent_conditional_fm/2025-10-13_02-11-56

# Probabilistic evaluation (uncertainty quantification)
python src/flow_matching/evaluate_roa.py --config-name=evaluate_cartpole_roa \
    evaluation.probabilistic=true \
    evaluation.num_samples=20
```

### Evaluation Modes

#### Deterministic Mode (`probabilistic=false`)
- Single prediction per state
- Binary classification: Success (1) or Failure (0)
- Fast evaluation
- Metrics: accuracy, precision, recall, F1, confusion matrix

#### Probabilistic Mode (`probabilistic=true`)
- Multiple samples per state (default: 20)
- Three-way classification based on sample distribution:
  - **Success (1)**: ≥60% of samples reach stable attractor
  - **Failure (0)**: ≥60% of samples fail to reach attractor
  - **Separatrix (-1)**: <60% for both (uncertain/mixed predictions)
- Provides uncertainty quantification via entropy
- Separatrix states excluded from accuracy metrics
- Additional metrics: AUC, ROC curve, entropy distribution

### Classification Logic
The system classifies predicted endpoints using `is_in_attractor()`:
```python
# Check if endpoint is in balanced state (within radius)
in_attractor = (|x| < radius) ∧ (|θ| < radius) ∧ (|ẋ| < radius) ∧ (|θ̇| < radius)
```

Default radius: 0.3 (configurable via `evaluation.attractor_radius`)

### Output Files
Evaluation generates the following outputs in `cartpole_roa_evaluation/`:
- `confusion_matrix.png`: Visual confusion matrix
- `error_analysis.png`: State space visualization with errors highlighted
- `state_space_classification.png`: Success/failure/separatrix regions
- `probability_distributions.png`: Probability and entropy distributions (probabilistic mode)
- `roc_curve.png`: ROC curve with AUC (probabilistic mode)
- `results.npz`: Numerical results and metrics
- `predicted_endpoints.txt`: Start states and predicted endpoints

## 6. Expected Performance

### Training Metrics
- **Initial Loss**: ~1.0-3.0 (after normalization fix)
- **Convergence**: Should decrease steadily over epochs
- **Final Loss**: ~0.1-0.5 for well-trained model
- **Validation MAE**: Computed per-dimension every 10 epochs
  - Cart position: typically < 0.5
  - Pole angle: typically < 0.3 radians
  - Velocities: typically < 1.0

### ROA Evaluation Performance
- **Deterministic Accuracy**: Typically 85-95% on labeled ROA data
- **Probabilistic Separatrix Detection**: 5-15% of states identified as uncertain
- **AUC**: Typically > 0.90 in probabilistic mode

### Inference Quality
- **Trajectory Smoothness**: Integration produces physically reasonable paths
- **Attractor Convergence**: Generated endpoints should be in balanced regions
- **State Bounds**: All generated states within physical limits
- **Angle Wrapping**: Angles correctly wrapped to [-π, π] at all steps

## 7. Troubleshooting

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

## 8. Key Technical Details

### Manifold Structure
The CartPole system uses a **Product manifold** ℝ²×S¹×ℝ:
```python
Product(input_dim=4, manifolds=[
    (Euclidean(), 1),   # Cart position (x)
    (FlatTorus(), 1),   # Pole angle (θ) - circular
    (Euclidean(), 2)    # Cart velocity (ẋ) and angular velocity (θ̇)
])
```

This ensures:
- Proper geodesic interpolation on the circular angle component
- Linear interpolation on Euclidean components
- Correct distance computation respecting manifold geometry

### State Embedding
Raw state `[x, θ, ẋ, θ̇]` is transformed for neural network processing:

1. **Normalization** (coordinates → normalized):
   ```
   [x, θ, ẋ, θ̇] → [x/cart_limit, θ, ẋ/vel_limit, θ̇/ang_vel_limit]
   ```

2. **Embedding** (normalized → embedded for model):
   ```
   [x_norm, θ, ẋ_norm, θ̇_norm] → [x_norm, sin(θ), cos(θ), ẋ_norm, θ̇_norm]
   ```

The sin/cos embedding is crucial for:
- Handling angle discontinuity at ±π
- Providing smooth, continuous representation for neural network
- Allowing network to learn circular geometry

### Neural Network Architecture
**CartPoleLatentConditionalUNet1D** structure:
- **Input**: 5D embedded state + 64D time embedding + 2D latent + 5D condition
- **Hidden layers**: [256, 512, 1024, 512, 256] with SiLU activations
- **Output**: 4D velocity in tangent space `[dx/dt, dθ/dt, dẋ/dt, dθ̇/dt]`
- **Total parameters**: ~1-2M depending on configuration

### Flow Matching Training
Uses **GeodesicProbPath** with **CondOTScheduler**:
1. Sample random time t ~ Uniform(0, 1)
2. Interpolate between noisy input x₀ and data target x₁
   - Geodesic interpolation for angle (shortest path on circle)
   - Linear interpolation for Euclidean components
3. Compute target velocity `dx_t/dt` via automatic differentiation
4. Train network to predict this velocity
5. Loss: MSE between predicted and target velocity

### Stochasticity via Latent Variables
The latent variable z ~ N(0, I) provides:
- **Multimodal predictions**: Different trajectories from same start state
- **Uncertainty quantification**: Spread of predictions indicates confidence
- **Separatrix detection**: Mixed predictions reveal boundary regions

In probabilistic evaluation, multiple samples reveal:
- Confident success: All samples reach attractor
- Confident failure: All samples fail
- Uncertain (separatrix): Mixed outcomes across samples

## 9. Flow Matching Variants

The CartPole system has **two flow matching variants** with different approaches to learning dynamics:

### Variant Comparison

| Aspect | Latent Conditional FM | Gaussian-Perturbed FM |
|--------|----------------------|----------------------|
| **Initial Noise** | Uniform from full state space | Gaussian N(start_state, σ²I) |
| **Latent Variable** | Yes: z ~ N(0,I) (2D) | No (removed) |
| **Conditioning** | Conditioned on start state | No conditioning |
| **Model Input** | `f(x_t, t, z, condition)` | `f(x_t, t)` - simpler! |
| **Model Parameters** | ~2M | ~1.5M (25% fewer) |
| **Stochasticity Source** | Latent variable z | Gaussian perturbation |
| **Training Complexity** | Higher (more inputs) | Lower (simpler model) |
| **Inference Speed** | Slower (latent handling) | Faster (no latent) |
| **Use Case** | Rich multimodal predictions | Direct perturbation-based |

### Latent Conditional Flow Matching (Original)

**Location**: `src/flow_matching/cartpole_latent_conditional/`

**Key Features**:
- Latent variable z ~ N(0,I) provides stochasticity
- Conditioned on start state for context
- More expressive model architecture
- Initial noise uniformly sampled from state space

**Training**:
```bash
python src/flow_matching/cartpole_latent_conditional/train.py
```

**Configuration**: `configs/train_cartpole_lcfm.yaml`

**Model Parameters**:
- `latent_dim`: 2
- `condition_dim`: 5 (embedded start state)
- `embedded_dim`: 5
- Total: ~2M parameters

### Gaussian-Perturbed Flow Matching (New Variant)

**Location**: `src/flow_matching/cartpole_gaussian_perturbed/`

**Key Features**:
- NO latent variables
- NO conditioning on start state
- Initial states sampled from Gaussian centered at start: x₀ ~ N(start_state, σ²I)
- Simpler model with fewer parameters
- Stochasticity from explicit Gaussian perturbation

**Training**:
```bash
python src/flow_matching/cartpole_gaussian_perturbed/train.py

# Adjust Gaussian noise std
python src/flow_matching/cartpole_gaussian_perturbed/train.py \
    flow_matching.noise_std=0.2
```

**Configuration**: `configs/train_cartpole_gaussian_perturbed.yaml`

**Model Parameters**:
- `noise_std`: 0.1 (Gaussian perturbation standard deviation)
- `embedded_dim`: 5
- NO latent_dim (removed)
- NO condition_dim (removed)
- Total: ~1.5M parameters (25% fewer than latent conditional)

**Key Configuration Parameter**:
```yaml
flow_matching:
  noise_std: 0.1  # Standard deviation of Gaussian around start state
                  # Smaller = closer to start state
                  # Larger = more exploration
```

### When to Use Each Variant

**Use Latent Conditional FM when**:
- You want richer multimodal predictions
- You need explicit conditioning on start state
- You have sufficient computational resources
- You want to model complex, multi-path dynamics

**Use Gaussian-Perturbed FM when**:
- You want simpler, faster training/inference
- You prefer explicit initial distribution (interpretable)
- You have limited computational resources
- You want direct perturbation-based uncertainty

### Inference Examples

#### Latent Conditional Inference
```python
from src.flow_matching.cartpole_latent_conditional.flow_matcher_fb import CartPoleLatentConditionalFlowMatcher

# Load model
model = CartPoleLatentConditionalFlowMatcher.load_from_checkpoint(
    "outputs/cartpole_latent_conditional_fm/2025-10-17_12-30-45"
)

# Single prediction (samples latent internally)
start_states = torch.tensor([[0.5, 0.1, 2.0, 1.0]])  # [x, θ, ẋ, θ̇]
endpoint = model.predict_endpoint(start_states, num_steps=100)

# Multiple samples (different latent z each time)
endpoints = model.predict_endpoints_batch(start_states, num_samples=20)
```

#### Gaussian-Perturbed Inference
```python
from src.flow_matching.cartpole_gaussian_perturbed.inference import CartPoleGaussianPerturbedInference

# Load model
inferencer = CartPoleGaussianPerturbedInference(
    "outputs/cartpole_gaussian_perturbed_fm/2025-10-17_14-15-30"
)

# Single prediction (samples Gaussian noise internally)
start_states = torch.tensor([[0.5, 0.1, 2.0, 1.0]])  # [x, θ, ẋ, θ̇]
endpoint = inferencer.predict_endpoint(start_states, num_steps=100)

# Multiple samples (different Gaussian noise each time)
endpoints = inferencer.predict_endpoints_batch(start_states, num_samples=20)

# Uncertainty quantification
uncertainty = inferencer.compute_uncertainty(start_states, num_samples=20)
print(f"Mean: {uncertainty['mean']}")
print(f"Std: {uncertainty['std']}")

# Attractor convergence analysis
convergence = inferencer.check_attractor_convergence(start_states, num_samples=20)
print(f"Success proportion: {convergence['proportion_success']}")
```

### Training Performance Comparison

Both variants should achieve similar performance on endpoint prediction, but with different characteristics:

**Latent Conditional**:
- Training time: Baseline
- Final val_loss: ~0.1-0.5
- Endpoint MAE: ~0.2-0.4 per dimension
- ROA accuracy: ~85-95%

**Gaussian-Perturbed**:
- Training time: ~25% faster (simpler model)
- Final val_loss: ~0.1-0.5 (similar)
- Endpoint MAE: ~0.2-0.4 per dimension (similar)
- ROA accuracy: ~85-95% (similar)

The main difference is training/inference speed and model interpretability, not accuracy.

## 10. Extensions

### Larger Datasets
```bash
# Build larger dataset by increasing increment
python src/build_shuffled_endpoint_dataset.py --config-name=build_cartpole_endpoint_dataset.yaml increment=2000
```

### Different Latent Dimensions (Latent Conditional Only)
```yaml
# In training config
model:
  latent_dim: 4  # Try different latent dimensions
```

### Different Gaussian Noise Levels (Gaussian-Perturbed Only)
```yaml
# In training config
flow_matching:
  noise_std: 0.05  # Tighter around start state
  # or
  noise_std: 0.2   # More exploration
```

### Custom Bounds
```python
# Modify CartPoleSystemLCFM constructor for different bounds
system = CartPoleSystemLCFM(
    bounds_file="/path/to/custom_bounds.pkl",
    use_dynamic_bounds=True
)
```