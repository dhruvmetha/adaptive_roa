# 🚀 Flow Matching Training & Evaluation Guide

Complete guide for training and evaluating flow matching models for Pendulum and CartPole systems.

---

## 📋 Table of Contents

1. [Quick Start](#quick-start)
2. [System Overview](#system-overview)
3. [Training](#training)
4. [Evaluation](#evaluation)
5. [Configuration](#configuration)
6. [Troubleshooting](#troubleshooting)

---

## ⚡ Quick Start

### **Prerequisites**

```bash
# Activate environment
conda activate /common/users/dm1487/envs/arcmg

# Ensure you're in the project root
cd /common/home/dm1487/robotics_research/tripods/olympics-classifier
```

### **Train Pendulum (Latent Conditional)**

```bash
python src/flow_matching/pendulum/latent_conditional/train.py
```

### **Train CartPole (Latent Conditional)**

```bash
python src/flow_matching/cartpole/latent_conditional/train.py
```

## 🎯 System Overview

### **Available Systems**

| System | Manifold | State | Config |
|--------|----------|-------|--------|
| **Pendulum** | S¹×ℝ | (θ, θ̇) | `train_pendulum.yaml` |
| **CartPole** | ℝ²×S¹×ℝ | (x, θ, ẋ, θ̇) | `train_cartpole.yaml` |

### **Key Insight: Circular Coordinates**

Both systems have **circular angle coordinates** (θ ∈ S¹):
- **Pendulum**: θ is the pendulum angle (wraps at ±π)
- **CartPole**: θ is the **pole angle** (wraps at ±π)

This is why both use geodesic interpolation on the circular components!

---

## 🏋️ Training

### **1. Pendulum Training**

#### **Location**
```
src/flow_matching/pendulum/latent_conditional/train.py
```

#### **Configuration**
```
configs/train_pendulum.yaml
```

#### **Command**
```bash
python src/flow_matching/pendulum/latent_conditional/train.py
```

#### **Key Config Parameters**
```yaml
# Flow matching settings
flow_matching:
  latent_dim: 2              # Dimension of latent variable z
  mae_val_frequency: 10      # Run validation inference every 10 epochs

# Model architecture
model:
  _target_: src.model.pendulum_unet.PendulumUNet
  embedded_dim: 3            # (sin θ, cos θ, θ̇_norm)
  output_dim: 2              # (dθ, dθ̇)
  latent_dim: 2
  condition_dim: 3
  hidden_dims: [256, 512, 256]

# Training
batch_size: 256
base_lr: 1e-4
trainer:
  max_epochs: 500
```

#### **Customize Training**
```bash
# Change latent dimension
python src/flow_matching/pendulum/latent_conditional/train.py \
    flow_matching.latent_dim=4

# Change learning rate
python src/flow_matching/pendulum/latent_conditional/train.py \
    base_lr=5e-4

# Change model architecture
python src/flow_matching/pendulum/latent_conditional/train.py \
    model.hidden_dims=[512,1024,512]

# Use different GPU
python src/flow_matching/pendulum/latent_conditional/train.py \
    device=gpu2
```

---

### **2. CartPole Training (Latent Conditional)**

#### **Location**
```
src/flow_matching/cartpole/latent_conditional/train.py
```

#### **Configuration**
```
configs/train_cartpole.yaml
```

#### **Command**
```bash
python src/flow_matching/cartpole/latent_conditional/train.py
```

#### **Key Config Parameters**
```yaml
# Flow matching settings
flow_matching:
  latent_dim: 2              # Dimension of latent variable z
  mae_val_frequency: 10      # Run validation inference every 10 epochs

# Model architecture
model:
  _target_: src.model.cartpole_unet.CartPoleUNet
  embedded_dim: 5            # (x_norm, sin θ, cos θ, ẋ_norm, θ̇_norm)
  output_dim: 4              # (dx, dθ, dẋ, dθ̇)
  latent_dim: 2
  condition_dim: 5
  hidden_dims: [256, 512, 1024, 512, 256]

# Training
batch_size: 256
base_lr: 1e-4
trainer:
  max_epochs: 500
```

#### **Data Files**
The CartPole system loads bounds automatically from:
```
/common/users/dm1487/arcmg_datasets/cartpole/cartpole_data_bounds.pkl
```

Training data:
```yaml
data:
  data_file: /common/users/dm1487/arcmg_datasets/cartpole/incremental_endpoint_dataset/1000_endpoint_dataset.txt
  validation_file: /common/users/dm1487/arcmg_datasets/cartpole/incremental_endpoint_dataset/validation_endpoint_dataset.txt
```

---

## 📊 Training Outputs

### **Checkpoint Location**
```
outputs/<system_name>/<timestamp>/version_0/checkpoints/
```

Example:
```
outputs/cartpole_latent_conditional_fm/2025-10-13_18-45-32/version_0/checkpoints/
  ├── epoch=042-val_loss=0.1234.ckpt  ← Best model
  ├── epoch=089-val_loss=0.1456.ckpt
  └── last.ckpt                        ← Most recent
```

### **TensorBoard Logs**
```bash
# Launch TensorBoard
tensorboard --logdir outputs/

# View at: http://localhost:6006
```

### **Metrics Logged**
- `train_loss`: Training loss (flow matching MSE)
- `val_loss`: Validation loss
- `val_inference/mse`: Overall endpoint prediction error (every 10 epochs)
- `val_inference/mae`: Mean absolute error
- `val_inference/{dim}_mse`: Per-dimension errors
- `val_inference/{dim}_contrib`: Error contribution percentages

---

## 🔍 Evaluation

### **1. ROA (Region of Attraction) Evaluation**

Evaluates how well the model predicts which states reach stable attractors.

#### **Command (Auto-detect best checkpoint)**
```bash
# Pendulum
python src/flow_matching/evaluate_roa.py --config-name=evaluate_pendulum_roa

# CartPole
python src/flow_matching/evaluate_roa.py --config-name=evaluate_cartpole_roa
```

#### **Command (Specific checkpoint)**
```bash
python src/flow_matching/evaluate_roa.py \
    --config-name=evaluate_cartpole_roa \
    checkpoint.path=outputs/cartpole_latent_conditional_fm/2025-10-13_18-45-32
```

#### **Evaluation Modes**

**A. Deterministic Mode** (default)
```bash
python src/flow_matching/evaluate_roa.py --config-name=evaluate_cartpole_roa
```
- Single prediction per state
- Binary classification: Success (1) or Failure (0)
- Fast evaluation
- Metrics: accuracy, precision, recall, F1, confusion matrix

**B. Probabilistic Mode** (uncertainty quantification)
```bash
python src/flow_matching/evaluate_roa.py \
    --config-name=evaluate_cartpole_roa \
    evaluation.probabilistic=true \
    evaluation.num_samples=20
```
- Multiple samples per state (default: 20)
- Three-way classification:
  - **Success (1)**: ≥60% of samples reach stable attractor
  - **Failure (0)**: ≥60% of samples fail
  - **Separatrix (-1)**: <60% for both (uncertain)
- Provides uncertainty via entropy
- Additional metrics: AUC, ROC curve

#### **Output Files**
Saved to `<system>_roa_evaluation/`:
- `confusion_matrix.png`: Visual confusion matrix
- `error_analysis.png`: State space visualization
- `state_space_classification.png`: Success/failure/separatrix regions
- `probability_distributions.png`: Probability/entropy (probabilistic mode)
- `roc_curve.png`: ROC curve with AUC (probabilistic mode)
- `results.npz`: Numerical results
- `predicted_endpoints.txt`: Predictions

---

### **2. Custom Inference**

#### **Load Model for Inference**

**Pendulum:**
```python
from src.flow_matching.latent_conditional.flow_matcher_fb import PendulumLatentConditionalFlowMatcher

# Load from training folder (auto-detects best checkpoint)
model = PendulumLatentConditionalFlowMatcher.load_from_checkpoint(
    "outputs/pendulum_latent_conditional_fm/2025-10-13_18-45-32"
)

# Or load specific checkpoint
model = PendulumLatentConditionalFlowMatcher.load_from_checkpoint(
    "outputs/.../checkpoints/epoch=042-val_loss=0.1234.ckpt"
)
```

**CartPole (Latent Conditional):**
```python
from src.flow_matching.cartpole.latent_conditional.flow_matcher import CartPoleLatentConditionalFlowMatcher

model = CartPoleLatentConditionalFlowMatcher.load_from_checkpoint(
    "outputs/cartpole_latent_conditional_fm/2025-10-13_18-45-32"
)
```


#### **Predict Endpoints**

```python
import torch

# Define start states
start_states = torch.tensor([
    [0.5, 0.1, 2.0, 1.0],   # CartPole: (x, θ, ẋ, θ̇)
    [-0.8, -0.5, -3.0, -2.0]
])

# Single prediction per state
endpoints = model.predict_endpoint(start_states, num_steps=100)
print(f"Predicted endpoints: {endpoints}")

# Multiple samples per state (for uncertainty)
batch_endpoints = model.predict_endpoints_batch(
    start_states,
    num_samples=20
)
print(f"Shape: {batch_endpoints.shape}")  # [B*num_samples, state_dim]

# Compute mean and std
batch_size = start_states.shape[0]
endpoints_reshaped = batch_endpoints.reshape(batch_size, 20, -1)
mean_endpoint = endpoints_reshaped.mean(dim=1)
std_endpoint = endpoints_reshaped.std(dim=1)
print(f"Mean: {mean_endpoint}")
print(f"Std: {std_endpoint}")
```

#### **Generate Full Trajectory**

```python
# Get full integration trajectory
final_state, trajectory = model.integrate_trajectory(
    start_states,
    num_steps=100
)

print(f"Trajectory length: {len(trajectory)}")  # 101 states (including initial)
print(f"Final state: {final_state.shape}")     # [B, state_dim]

# Each trajectory[i] is a state at time step i
for i, state in enumerate(trajectory[::10]):  # Every 10th step
    print(f"t={i*10}: {state}")
```

---

## ⚙️ Configuration

### **Config File Structure**

All config files are in `configs/`:

```yaml
# System definition
system:
  _target_: src.systems.cartpole.CartPoleSystem
  bounds_file: /path/to/bounds.pkl

# Data loading
data:
  _target_: src.data.cartpole_endpoint_data.CartPoleEndpointDataModule
  data_file: /path/to/train.txt
  validation_file: /path/to/val.txt
  batch_size: 256

# Model architecture
model:
  _target_: src.model.cartpole_unet.CartPoleUNet
  embedded_dim: 5
  output_dim: 4
  hidden_dims: [256, 512, 1024, 512, 256]

# Flow matching
flow_matching:
  latent_dim: 2              # Only for latent conditional
  mae_val_frequency: 10

# Optimizer
optimizer:
  _target_: torch.optim.AdamW
  weight_decay: 1e-5

# Scheduler
scheduler:
  _target_: torch.optim.lr_scheduler.ReduceLROnPlateau
  mode: min
  factor: 0.5
  patience: 10

# Trainer
trainer:
  _target_: lightning.pytorch.Trainer
  max_epochs: 500
  accelerator: gpu
  devices: [1]
  gradient_clip_val: 1.0
  callbacks:
    - _target_: lightning.pytorch.callbacks.ModelCheckpoint
      monitor: val_loss
      mode: min
      save_top_k: 3
    - _target_: src.callbacks.validation_inference_callback.ValidationInferenceCallback
      inference_frequency: 10
```

### **Common Config Overrides**

```bash
# Change GPU
python train.py device=gpu2

# Change batch size
python train.py batch_size=512

# Change learning rate
python train.py base_lr=5e-4

# Change max epochs
python train.py trainer.max_epochs=1000

# Change model architecture
python train.py model.hidden_dims=[512,1024,2048,1024,512]

# Change validation frequency
python train.py flow_matching.mae_val_frequency=5

# Multiple overrides
python train.py \
    batch_size=512 \
    base_lr=5e-4 \
    trainer.max_epochs=200 \
    device=gpu3
```

---

## 🐛 Troubleshooting

### **Training Issues**

#### **High Loss / Not Learning**

**Symptom**: Loss > 10 or not decreasing

**Possible Causes**:
1. Velocity targets not normalized properly
2. Learning rate too high
3. Data quality issues

**Solutions**:
```bash
# Reduce learning rate
python train.py base_lr=5e-5

# Check data bounds are loaded correctly
# Look for this in training output:
# "Loaded CartPole bounds from: /path/to/bounds.pkl"

# Verify angle wrapping
# Angles should be in [-π, π] in your data files
```

#### **Integration Instability**

**Symptom**: States go out of bounds during inference

**Cause**: Integration in wrong coordinate space

**Solution**: The code should automatically use normalized integration ([-1, 1] space). If issues persist, check:
- `normalize_state()` is called before integration
- `denormalize_state()` is called after integration
- Manifold projection is enabled (`projx=True`)

#### **Dimension Mismatch Errors**

**Symptom**: Runtime errors about tensor dimensions

**Common Issues**:
```python
# Embedded dim doesn't match model input
# Pendulum: embedded_dim = 3  (sin θ, cos θ, θ̇)
# CartPole: embedded_dim = 5  (x, sin θ, cos θ, ẋ, θ̇)

# Output dim doesn't match state dim
# Pendulum: output_dim = 2   (dθ, dθ̇)
# CartPole: output_dim = 4   (dx, dθ, dẋ, dθ̇)

# Latent dim inconsistency
# Ensure model.latent_dim == flow_matching.latent_dim
```

### **Data Issues**

#### **Angles Not Wrapped**

**Symptom**: Errors or poor training performance

**Solution**: Ensure angles are wrapped to [-π, π]:
```python
import numpy as np
theta_wrapped = np.arctan2(np.sin(theta), np.cos(theta))
```

#### **Missing Bounds File**

**Symptom**: Warning about using default bounds

**Solution**: Provide correct path to bounds pickle file:
```bash
python train.py \
    system.bounds_file=/path/to/your/cartpole_data_bounds.pkl
```

### **Memory Issues**

#### **Out of Memory**

**Solutions**:
```bash
# Reduce batch size
python train.py batch_size=128

# Reduce model size
python train.py model.hidden_dims=[128,256,128]

# Use gradient accumulation
python train.py trainer.accumulate_grad_batches=2
```

---

## 📈 Expected Performance

### **Training Metrics**

**Pendulum:**
- Initial loss: ~1.0-3.0
- Final loss: ~0.1-0.5
- Training time: ~2-4 hours (500 epochs, GPU)
- Validation MAE: < 0.3 per dimension

**CartPole (Latent Conditional):**
- Initial loss: ~1.0-3.0
- Final loss: ~0.1-0.5
- Training time: ~4-6 hours (500 epochs, GPU)
- Validation MAE: < 0.5 per dimension

### **ROA Evaluation Performance**

**Deterministic Mode:**
- Accuracy: 85-95%
- Precision/Recall: 80-95%

**Probabilistic Mode:**
- Accuracy: 85-95% (excluding separatrix)
- Separatrix detection: 5-15% of states
- AUC: > 0.90

---

## 📚 Related Documentation

- **CARTPOLE_DOCUMENTATION.md**: Detailed CartPole system documentation
- **QUICK_REFERENCE.md**: Quick command reference
- **NEW_SYSTEM_IMPLEMENTATION_GUIDE.md**: How to add new systems
- **ROA_ANALYSIS_GUIDE.md**: Region of attraction analysis details
- **CLAUDE.md**: Project overview and architecture

---

## 🎓 Key Takeaways

1. **Both systems use circular coordinates** - Pendulum θ and CartPole pole angle θ
2. **CartPole uses the latent conditional model** - same training/eval workflow as Pendulum
3. **Training is automatic** - Just run the training script, validation inference runs every 10 epochs
4. **Evaluation supports uncertainty** - Use probabilistic mode for uncertainty quantification
5. **Configs are composable** - Easy to override any parameter from command line

---

**Ready to train!** 🚀

```bash
# Pendulum
python src/flow_matching/pendulum/latent_conditional/train.py

# CartPole (rich model)
python src/flow_matching/cartpole/latent_conditional/train.py
```
