# Unified Training Guide - Facebook Flow Matching

## ✅ Setup Complete!

You now have a **unified training script** that works for both Pendulum and CartPole systems using Facebook Flow Matching.

---

## 🚀 Quick Start

### **Train Pendulum**
```bash
python src/flow_matching/train_latent_conditional.py --config-name=train_pendulum_lcfm
```

### **Train CartPole**
```bash
python src/flow_matching/train_latent_conditional.py --config-name=train_cartpole_lcfm
```

**That's it!** Just change the config name to switch systems.

---

## 📁 File Structure

```
src/flow_matching/
  └── train_latent_conditional.py  ← ONE unified training script

configs/
  ├── train_pendulum_lcfm.yaml     ← Pendulum configuration
  └── train_cartpole_lcfm.yaml     ← CartPole configuration
```

---

## 🎯 How It Works

### **Approach: Hydra Target Instantiation**

The training script uses Hydra's `_target_` mechanism to instantiate all components from the config file.

**Training Script** (simple!):
```python
# All components instantiated from config
system = hydra.utils.instantiate(cfg.system)
model = hydra.utils.instantiate(cfg.model)
data_module = hydra.utils.instantiate(cfg.data)
flow_matcher = hydra.utils.instantiate(cfg.flow_matcher, ...)

# Train!
trainer.fit(flow_matcher, data_module)
```

**Config File** (explicit!):
```yaml
# Each component specifies its class with _target_
system:
  _target_: src.systems.pendulum_lcfm.PendulumSystemLCFM

flow_matcher:
  _target_: src.flow_matching.latent_conditional.flow_matcher_fb.LatentConditionalFlowMatcher

model:
  _target_: src.model.latent_conditional_unet1d.LatentConditionalUNet1D
  embedded_dim: 3
  latent_dim: 2
  hidden_dims: [256, 512, 256]
```

---

## 📋 Config File Comparison

### **Pendulum** (`train_pendulum_lcfm.yaml`)

```yaml
system:
  _target_: src.systems.pendulum_lcfm.PendulumSystemLCFM

flow_matcher:
  _target_: src.flow_matching.latent_conditional.flow_matcher_fb.LatentConditionalFlowMatcher

model:
  _target_: src.model.latent_conditional_unet1d.LatentConditionalUNet1D
  embedded_dim: 3    # (sin θ, cos θ, θ̇_norm)
  output_dim: 2      # (dθ, dθ̇)

data:
  _target_: src.data.endpoint_data.EndpointDataModule
  data_file: /path/to/pendulum_data.txt
```

### **CartPole** (`train_cartpole_lcfm.yaml`)

```yaml
system:
  _target_: src.systems.cartpole_lcfm.CartPoleSystemLCFM

flow_matcher:
  _target_: src.flow_matching.cartpole_latent_conditional.flow_matcher_fb.CartPoleLatentConditionalFlowMatcher

model:
  _target_: src.model.cartpole_latent_conditional_unet1d.CartPoleLatentConditionalUNet1D
  embedded_dim: 5    # (x_norm, sin θ, cos θ, ẋ_norm, θ̇_norm)
  output_dim: 4      # (dx, dθ, dẋ, dθ̇)

data:
  _target_: src.data.cartpole_endpoint_data.CartPoleEndpointDataModule
  data_file: /path/to/cartpole_data.txt
```

---

## 🔧 Customization

### **Override Parameters from Command Line**

```bash
# Change latent dimension
python src/flow_matching/train_latent_conditional.py \
    --config-name=train_pendulum_lcfm \
    flow_matching.latent_dim=4

# Change learning rate and batch size
python src/flow_matching/train_latent_conditional.py \
    --config-name=train_pendulum_lcfm \
    base_lr=5e-4 \
    batch_size=512

# Change model architecture
python src/flow_matching/train_latent_conditional.py \
    --config-name=train_pendulum_lcfm \
    model.hidden_dims=[512,1024,512]

# Change max epochs
python src/flow_matching/train_latent_conditional.py \
    --config-name=train_pendulum_lcfm \
    trainer.max_epochs=200
```

### **Create Custom Config**

Copy an existing config and modify:

```bash
# Copy Pendulum config
cp configs/train_pendulum_lcfm.yaml configs/my_custom_pendulum.yaml

# Edit the new config
vim configs/my_custom_pendulum.yaml

# Train with custom config
python src/flow_matching/train_latent_conditional.py \
    --config-name=my_custom_pendulum
```

---

## 📊 What Each Config Section Does

### **System**
Defines the dynamical system (Pendulum or CartPole):
```yaml
system:
  _target_: src.systems.pendulum_lcfm.PendulumSystemLCFM
```

### **Flow Matcher**
Specifies which Facebook FM flow matcher to use:
```yaml
flow_matcher:
  _target_: src.flow_matching.latent_conditional.flow_matcher_fb.LatentConditionalFlowMatcher
```
- Pendulum: `LatentConditionalFlowMatcher` (S¹×ℝ)
- CartPole: `CartPoleLatentConditionalFlowMatcher` (ℝ²×S¹×ℝ)

### **Model**
Neural network architecture:
```yaml
model:
  _target_: src.model.latent_conditional_unet1d.LatentConditionalUNet1D
  embedded_dim: 3              # Input dimension (embedded state)
  latent_dim: 2                # Latent variable dimension
  condition_dim: 3             # Conditioning dimension
  hidden_dims: [256, 512, 256] # Hidden layers
  output_dim: 2                # Output dimension (velocity)
```

### **Data**
Dataset and dataloader configuration:
```yaml
data:
  _target_: src.data.endpoint_data.EndpointDataModule
  data_file: /path/to/data.txt
  batch_size: 256
  num_workers: 4
  train_split: 0.8
  val_split: 0.1
```

### **Optimizer**
Training optimizer:
```yaml
optimizer:
  _target_: torch.optim.AdamW
  lr: 1e-4
  weight_decay: 1e-5
```

### **Scheduler**
Learning rate scheduler:
```yaml
scheduler:
  _target_: torch.optim.lr_scheduler.ReduceLROnPlateau
  mode: min
  factor: 0.5
  patience: 10
```

### **Trainer**
PyTorch Lightning trainer:
```yaml
trainer:
  _target_: lightning.pytorch.Trainer
  max_epochs: 100
  accelerator: gpu
  devices: [1]
  callbacks:
    - _target_: lightning.pytorch.callbacks.ModelCheckpoint
      monitor: val_loss
    - _target_: lightning.pytorch.callbacks.EarlyStopping
      patience: 20
```

---

## 🎯 Benefits of This Approach

| Feature | Benefit |
|---------|---------|
| **Single Script** | Only maintain one training script |
| **Config-Driven** | All settings in YAML, no code changes |
| **Easy Switching** | Change config name to switch systems |
| **Override Anywhere** | Command-line overrides for any parameter |
| **Self-Documenting** | Config shows exactly what classes are used |
| **Extensible** | Add new systems by creating new configs |

---

## 📈 Training Output

When you run training, you'll see:

```
================================================================================
🚀 Latent Conditional Flow Matching Training (Facebook FM)
================================================================================
📋 Config: pendulum_latent_conditional_fm
🎲 Seed: 42
================================================================================

📥 Instantiating components from config...

  🔧 System...
     ✅ PendulumSystemLCFM
        <system details>

  📊 Data module...
     ✅ EndpointDataModule
        Dataset: /path/to/data.txt
        Batch size: 256

  🏗️  Model...
     ✅ LatentConditionalUNet1D
        Architecture: [256, 512, 256]
        Latent dim: 2D
        Parameters: 123,456

  🌊 Flow matcher...
✅ Initialized with Facebook Flow Matching:
   - Manifold: PendulumManifold (S¹×ℝ)
   - Path: GeodesicProbPath with CondOTScheduler
   - Latent dim: 2
     ✅ LatentConditionalFlowMatcher

================================================================================
🚀 Starting Training
================================================================================

[Training begins...]
```

---

## 🔍 Adding a New System (Future)

To add a new system (e.g., Acrobot):

1. **Create the flow matcher** (with Facebook FM):
   ```python
   # src/flow_matching/acrobot_latent_conditional/flow_matcher_fb.py
   class AcrobotLatentConditionalFlowMatcher(BaseFlowMatcher):
       def __init__(self, ...):
           self.manifold = AcrobotManifold()  # S¹×S¹×ℝ²
           self.path = GeodesicProbPath(...)
   ```

2. **Create the config**:
   ```yaml
   # configs/train_acrobot_lcfm.yaml
   system:
     _target_: src.systems.acrobot_lcfm.AcrobotSystemLCFM

   flow_matcher:
     _target_: src.flow_matching.acrobot_latent_conditional.flow_matcher_fb.AcrobotLatentConditionalFlowMatcher

   model:
     _target_: src.model.acrobot_unet1d.AcrobotUNet1D
   ```

3. **Train**:
   ```bash
   python src/flow_matching/train_latent_conditional.py --config-name=train_acrobot_lcfm
   ```

**No changes to the training script needed!**

---

## ✅ Summary

You now have:
- ✅ **One unified training script** for all systems
- ✅ **Clean config files** with explicit `_target_:` paths
- ✅ **Easy system switching** via config name
- ✅ **Facebook Flow Matching** integrated automatically
- ✅ **Command-line overrides** for experimentation

**To switch systems, just change the config name. That's it!** 🎉
