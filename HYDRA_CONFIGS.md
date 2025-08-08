# Hydra Configuration Guide

## 🚀 Quick Usage

### **Training with Different Devices**
```bash
# Train on GPU 1 (default)
python src/flow_matching/circular/train.py

# Train on GPU 0  
python src/flow_matching/circular/train.py device=gpu0

# Train on CPU
python src/flow_matching/circular/train.py device=cpu

# Change other parameters
python src/flow_matching/circular/train.py device=gpu0 trainer.max_epochs=1000 data.batch_size=64
```

### **Demo Scripts with Device Control**
```bash
# Run attractor analysis on GPU 1 (default)
python src/demo_attractor_analysis.py

# Run on CPU
python src/demo_attractor_analysis.py device=cpu

# Use different checkpoint
python src/demo_attractor_analysis.py checkpoint_path="path/to/checkpoint.ckpt"
```

---

## 📂 Modular Configuration Structure

### **Main Training Configs** 
- `train_circular_flow_matching.yaml` - **Modular config using defaults**
- `train_flow_matching.yaml` - **Modular config using defaults**  
- `train_reachability.yaml` - Reachability model training
- `train_classifier.yaml` - Classifier training

### **Device Configurations** 🆕
- `device/gpu0.yaml` - Use GPU 0
- `device/gpu1.yaml` - Use GPU 1 (default for most scripts)
- `device/cpu.yaml` - Use CPU only

### **Data Configurations**
- `data/endpoint_data.yaml` - Standard flow matching data
- `data/circular_endpoint_data.yaml` - Circular flow matching data
- `data/reachability_data.yaml` - Reachability training data  
- `data/classifier_data.yaml` - Classification data

### **Model Configurations**
- `model/flow_matching_unet.yaml` - Standard U-Net for flow matching
- `model/circular_flow_matching_unet.yaml` - Circular U-Net for pendulum
- `model/simple_mlp.yaml` - Multi-layer perceptron

### **Module Configurations** 🆕
- `module/standard_flow_matching.yaml` - Standard flow matching Lightning module
- `module/circular_flow_matching.yaml` - Circular flow matching Lightning module

### **Optimizer & Scheduler Configurations** 🆕
- `optimizer/adamw.yaml` - AdamW optimizer settings
- `scheduler/reduce_lr_on_plateau.yaml` - Learning rate scheduler

### **Trainer Configurations**
- `trainer/flow_matching.yaml` - Flow matching trainer config
- `trainer/gpu.yaml` - General GPU trainer config
- `trainer/default.yaml` - Default trainer settings

### **Demo Configurations** 🆕  
- `demo_attractor_analysis.yaml` - Attractor analysis demo settings

---

## ⚙️ Device Configuration System

### **New Features** ✅
- **No more hardcoded `CUDA_VISIBLE_DEVICES`** in Python scripts
- **Flexible device selection** via Hydra overrides
- **Automatic device setup** based on config

### **Device Config Structure**
```yaml
# configs/device/gpu1.yaml
device_id: "1"           # Sets CUDA_VISIBLE_DEVICES=1  
accelerator: gpu
devices: [0]             # PyTorch Lightning uses index 0 after CUDA_VISIBLE_DEVICES
```

```yaml
# configs/device/cpu.yaml  
device_id: null          # No CUDA_VISIBLE_DEVICES set
accelerator: cpu
devices: auto
```

### **How It Works**
1. **Hydra** loads device config (e.g., `device=gpu1`)
2. **Training script** reads `cfg.device.device_id` and sets `CUDA_VISIBLE_DEVICES`
3. **PyTorch Lightning** uses `cfg.device.accelerator` and `cfg.device.devices`

---

## 📋 Comprehensive Configs

### **train_circular_flow_matching.yaml** - ⭐ **MODULAR DESIGN**
```yaml
defaults:
  - _self_
  - data: circular_endpoint_data
  - model: circular_flow_matching_unet
  - module: circular_flow_matching
  - optimizer: adamw
  - scheduler: reduce_lr_on_plateau
  - trainer: flow_matching
  - device: gpu1

seed: 42
num_workers: 4
batch_size: 128
```

### **train_flow_matching.yaml** - ⭐ **MODULAR DESIGN** 
```yaml
defaults:
  - _self_
  - data: endpoint_data
  - model: flow_matching_unet
  - module: standard_flow_matching
  - optimizer: adamw  
  - scheduler: reduce_lr_on_plateau
  - trainer: flow_matching
  - device: gpu1

seed: 42
num_workers: 4
batch_size: 256

# Override for standard flow matching
trainer:
  max_epochs: 200
```

### **Legacy Configs** (Simple)
- `train_reachability.yaml` - Uses defaults pattern with separate model/trainer configs
- `train_classifier.yaml` - Minimal config, relies on defaults

---

## 🔄 Configuration Patterns

### **Modern Modular Pattern** ✅ **RECOMMENDED**
```yaml
defaults:
  - _self_
  - data: circular_endpoint_data
  - model: circular_flow_matching_unet
  - module: circular_flow_matching
  - optimizer: adamw
  - scheduler: reduce_lr_on_plateau
  - trainer: flow_matching
  - device: gpu1

# Clean modular design:
# - Each component in separate config file
# - Easy to swap components (model=different_model)
# - Clear separation of concerns
# - Reusable across different experiments
```

### **Legacy Simple Pattern**
```yaml
defaults:
  - data: classifier_data
  - model: simple_mlp
  - trainer: gpu

# Simpler but less flexible
```

---

## 🛠️ Available Overrides

### **Device Selection**
```bash
device=gpu0              # Use GPU 0
device=gpu1              # Use GPU 1 (default) 
device=cpu               # Use CPU only
```

### **Training Parameters**
```bash
trainer.max_epochs=1000         # Change max epochs
data.batch_size=128            # Change batch size
optimizer.lr=0.01              # Change learning rate
trainer.precision=16           # Use mixed precision
```

### **Data Configuration**
```bash
data.data_file="path/to/data.txt"    # Change data file
data.num_workers=8                   # Change data loading workers
```

### **Model Architecture**
```bash  
model.hidden_dims=[128,256,512]      # Change model size
model.time_emb_dim=256               # Change time embedding dimension
```

---

## 📚 Usage Examples

### **Training Examples**
```bash
# Quick training on different GPU
python src/flow_matching/circular/train.py device=gpu0

# Longer training with larger model
python src/flow_matching/circular/train.py \
    trainer.max_epochs=1000 \
    model.hidden_dims=[128,256,512] \
    device=gpu0

# CPU training with smaller batch
python src/flow_matching/circular/train.py \
    device=cpu \
    data.batch_size=32 \
    trainer.max_epochs=100
```

### **Demo Examples**
```bash
# Run analysis on different device
python src/demo_attractor_analysis.py device=gpu0

# Custom checkpoint and resolution
python src/demo_attractor_analysis.py \
    checkpoint_path="outputs/my_model.ckpt" \
    analysis.resolutions=[0.15,0.08] \
    device=cpu
```

### **View Current Config**
```bash
# See full configuration that will be used
python src/flow_matching/circular/train.py --cfg job

# See config with overrides
python src/flow_matching/circular/train.py device=cpu --cfg job
```

---

## 🎯 Benefits

### **Flexibility**
- ✅ **Easy device switching** without editing code
- ✅ **Parameter tuning** via command line
- ✅ **Reproducible experiments** with config logging

### **Organization** 
- ✅ **Modular configuration** management
- ✅ **Reusable components** (swap model, optimizer, etc.)
- ✅ **Clear separation of concerns** (model != trainer != data)
- ✅ **Component reuse** across different experiments

### **Development**
- ✅ **No more hardcoded values** in Python scripts  
- ✅ **Quick experimentation** with different settings
- ✅ **Easy deployment** on different hardware
- ✅ **Component swapping** - change just the model, optimizer, etc.

---

## 🎯 Best Practices

### **Modular Design Benefits**
- **Single Responsibility**: Each config file handles one concern
- **Reusability**: Share components across experiments
- **Flexibility**: Easy to swap individual components
- **Maintainability**: Changes in one place affect all usages

### **Examples of Modularity**
```bash
# Use different model architecture  
python src/flow_matching/circular/train.py model=larger_circular_unet

# Use different optimizer
python src/flow_matching/circular/train.py optimizer=sgd

# Use different data source
python src/flow_matching/circular/train.py data=different_dataset

# Combine multiple changes
python src/flow_matching/circular/train.py \
    model=larger_model \
    optimizer=sgd \
    device=gpu0 \
    batch_size=64
```

---

*The new modular configs (`train_circular_flow_matching.yaml`, `train_flow_matching.yaml`) follow Hydra best practices with clean component separation and easy customization.*