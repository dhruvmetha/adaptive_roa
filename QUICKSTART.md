# Olympics Classifier - Quick Start Guide

## 🚀 Essential Commands

### Environment Setup
```bash
export PATH="/common/users/dm1487/envs/arcmg/bin:$PATH"
# or if conda is configured:
# conda activate /common/users/dm1487/envs/arcmg
```

---

## 📋 Main Scripts to Run

### 1. **Training Flow Matching Models**
```bash
# Train circular flow matching (GPU 1 default)
python src/flow_matching/circular/train.py

# Train on specific GPU  
python src/flow_matching/circular/train.py device=gpu0

# Train standard flow matching
python src/flow_matching/standard/train.py device=gpu1

# Train on CPU
python src/flow_matching/circular/train.py device=cpu
```

### 2. **Build Training Data**
```bash
# Generate endpoint datasets for training
python src/build_endpoint_dataset.py
```

### 3. **Evaluation & Analysis**
```bash
# Comprehensive flow matching evaluation
python src/evaluate_flow_matching_refactored.py

# Advanced attractor basin analysis  
python src/demo_attractor_analysis.py

# Unified framework demonstration
python src/demo_unified_flow_matching.py
```

### 4. **Other Training Scripts**
```bash
# Train classifier (currently incomplete)
python src/train_classifier.py

# Train reachability model
python src/train_reachability.py

# Evaluate reachability
python src/evaluate_reachability.py
```

---

## 🏗️ Architecture Overview

### **Modern Unified Framework** ✅
- **`src/flow_matching/`** - Clean, unified implementation
  - `circular/` - Circular flow matching (pendulum dynamics)
  - `standard/` - Standard flow matching  
  - `base/` - Shared functionality
  - `utils/` - Common utilities

### **Key Classes**
```python
# Modern unified framework (USE THESE)
from src.flow_matching.circular import CircularFlowMatchingInference
from src.flow_matching.standard import StandardFlowMatchingInference
from src.visualization.attractor_analysis import AttractorBasinAnalyzer
```

### **Evaluation & Visualization**
- **`src/evaluation/`** - Unified metrics and evaluation
- **`src/visualization/`** - Phase space plots, basin analysis  
- **`src/systems/`** - Pendulum configuration and dynamics

---

## 📊 Attractor Basin Analysis

**What it does:** Analyzes which regions of state space lead to which attractors

### Quick Usage
```python
from src.flow_matching.circular import CircularFlowMatchingInference
from src.visualization.attractor_analysis import AttractorBasinAnalyzer
from src.systems.pendulum_config import PendulumConfig

# Load trained model
inferencer = CircularFlowMatchingInference("path/to/checkpoint.ckpt")

# Run analysis
config = PendulumConfig()
analyzer = AttractorBasinAnalyzer(config)
results = analyzer.analyze_attractor_basins(inferencer, resolution=0.1)

# Save complete analysis
analyzer.save_analysis_results("output_dir", results)
```

### Generated Files
- `attractor_basins.png` - Basin visualization
- `basin_statistics.png` - Statistical analysis
- `basin_analysis_report.txt` - Detailed text report
- `basin_analysis_data.npz` - Raw numerical data

---

## 📁 Key Directories

| Directory | Purpose |
|-----------|---------|
| `configs/` | YAML configuration files |
| `src/data/` | Data loading modules |
| `src/model/` | Neural network architectures |
| `outputs/` | Training outputs & checkpoints |
| `results/` | Evaluation results |
| `attractor_analysis_results/` | Basin analysis outputs |

---

## ⚡ Most Common Workflows

### **1. Train a New Model**
```bash
# Generate training data
python src/build_endpoint_dataset.py

# Train circular flow matching (GPU 1 default)
python src/flow_matching/circular/train.py

# Train on different device
python src/flow_matching/circular/train.py device=gpu0
python src/flow_matching/circular/train.py device=cpu

# Evaluate the trained model
python src/evaluate_flow_matching_refactored.py
```

### **2. Analyze Existing Model**
```bash
# Run comprehensive attractor basin analysis (GPU 1 default)
python src/demo_attractor_analysis.py

# Run on different device
python src/demo_attractor_analysis.py device=gpu0
python src/demo_attractor_analysis.py device=cpu

# Use custom checkpoint
python src/demo_attractor_analysis.py checkpoint_path="path/to/model.ckpt"
```

### **3. Compare Models**
```bash
# Use unified evaluation framework
python src/evaluate_flow_matching_refactored.py
```

---

## 🔧 Configuration

### **GPU Setup** 
Scripts use GPU 1 by default: `os.environ["CUDA_VISIBLE_DEVICES"] = "1"`

### **Key Config Files**
- `configs/train_circular_flow_matching.yaml` - Circular training config
- `configs/data/endpoint_data.yaml` - Data configuration
- `configs/model/flow_matching_unet.yaml` - Model architecture

### **Checkpoint Locations**
Training outputs are saved to timestamped directories:
```
outputs/YYYY-MM-DD/HH-MM-SS/lightning_logs/version_X/checkpoints/
```

---

## ❗ Important Notes

### **Device Configuration** 🆕
- ✅ **Flexible GPU/CPU selection**: `device=gpu0`, `device=gpu1`, `device=cpu`
- ✅ **No hardcoded devices** - all configurable via Hydra
- ✅ **Parameter overrides**: `trainer.max_epochs=1000`, `data.batch_size=64`

### **API Changes**
- ✅ Use: `predict_endpoint()` (unified framework)  
- ❌ Old: `predict_endpoints()` (deprecated)

### **Imports**
```python
# ✅ CORRECT (unified framework)
from src.flow_matching.circular import CircularFlowMatchingInference
from src.flow_matching.standard import StandardFlowMatchingInference

# ❌ OLD (removed files)
# from src.inference_circular_flow_matching import CircularFlowMatchingInference
```

### **Typical Checkpoint Path**
Update this in demo scripts:
```python
checkpoint_path = "outputs/2025-07-26/03-56-35/lightning_logs/version_0/checkpoints/epoch=499-step=29500.ckpt"
```

---

## 🐛 Common Issues

1. **"Module not found"** → Check environment: `export PATH="/common/users/dm1487/envs/arcmg/bin:$PATH"`
2. **"Checkpoint not found"** → Update checkpoint path in demo scripts  
3. **CUDA errors** → Models use GPU 1, make sure it's available
4. **Import errors** → Use unified framework imports (see above)

---

## 📈 Next Steps

1. **Train new models:** Use `src/flow_matching/circular/train.py`
2. **Analyze behavior:** Use `src/demo_attractor_analysis.py` 
3. **Evaluate performance:** Use `src/evaluate_flow_matching_refactored.py`
4. **Research insights:** Check basin analysis results for dynamical behavior patterns

---

## 📚 Additional Documentation

- **`CLAUDE.md`** - Comprehensive technical details and architecture
- **`HYDRA_CONFIGS.md`** - Complete guide to Hydra configuration system
- **Quick Start** - This file for essential commands

---

*This guide covers the essential commands and workflows for quick reference when returning to the project.*