# Flow Matching for Dynamical Systems with Circular Coordinates

PyTorch Lightning implementation of Latent Conditional Flow Matching for systems with circular coordinates (Pendulum and CartPole), using Facebook's Flow Matching library for manifold-aware geodesic interpolation.

---

## 🎯 Overview

This project implements flow matching models for **endpoint prediction** on dynamical systems with circular state spaces:

- **Pendulum**: S¹ × ℝ manifold (angle θ ∈ S¹, angular velocity θ̇ ∈ ℝ)
- **CartPole**: ℝ² × S¹ × ℝ manifold (cart position x, pole angle θ ∈ S¹, velocities)

### Key Features

- ✅ **Geodesic interpolation** on circular manifolds via Facebook Flow Matching
- ✅ **Multiple model variants**: Latent Conditional (rich) vs Gaussian Perturbed (fast)
- ✅ **Automatic validation inference** during training (every 10 epochs)
- ✅ **ROA evaluation** with probabilistic uncertainty quantification
- ✅ **Production-ready**: PyTorch Lightning, Hydra configs, comprehensive logging

---

## 📚 Documentation

### **Quick Links**

| Document | Purpose |
|----------|---------|
| **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)** | **START HERE** - Complete training & evaluation guide |
| [CARTPOLE_DOCUMENTATION.md](CARTPOLE_DOCUMENTATION.md) | CartPole-specific details (dataset building, ROA, etc.) |
| [NEW_SYSTEM_IMPLEMENTATION_GUIDE.md](NEW_SYSTEM_IMPLEMENTATION_GUIDE.md) | How to add new systems |
| [ROA_ANALYSIS_GUIDE.md](ROA_ANALYSIS_GUIDE.md) | Region of attraction evaluation |
| [CLAUDE.md](CLAUDE.md) | Project overview for Claude Code AI assistant |

---

## ⚡ Quick Start

### **Prerequisites**

```bash
# Activate the environment
conda activate /common/users/dm1487/envs/arcmg

# Navigate to project directory
cd /common/home/dm1487/robotics_research/tripods/olympics-classifier
```

### **Training**

```bash
# Pendulum (Latent Conditional)
python src/flow_matching/latent_conditional/train.py

# CartPole (Latent Conditional - richer model)
python src/flow_matching/cartpole_latent_conditional/train.py

# CartPole (Gaussian Perturbed - faster, simpler)
python src/flow_matching/cartpole_gaussian_perturbed/train.py
```

### **Evaluation**

```bash
# ROA evaluation (deterministic)
python src/flow_matching/evaluate_roa.py --config-name=evaluate_cartpole_roa

# ROA evaluation (probabilistic with uncertainty)
python src/flow_matching/evaluate_roa.py \
    --config-name=evaluate_cartpole_roa \
    evaluation.probabilistic=true \
    evaluation.num_samples=20
```

### **Inference**

```python
from src.flow_matching.cartpole_latent_conditional.flow_matcher_fb import CartPoleLatentConditionalFlowMatcher
import torch

# Load trained model
model = CartPoleLatentConditionalFlowMatcher.load_from_checkpoint(
    "outputs/cartpole_latent_conditional_fm/2025-10-13_18-45-32"
)

# Predict endpoints
start_states = torch.tensor([[0.5, 0.1, 2.0, 1.0]])  # (x, θ, ẋ, θ̇)
endpoints = model.predict_endpoint(start_states, num_steps=100)
```

**See [TRAINING_GUIDE.md](TRAINING_GUIDE.md) for complete details.**

---

## 🏗️ Architecture

### **System Overview**

```
src/
├── systems/                    # System definitions
│   ├── base.py                # Abstract base class
│   ├── pendulum_lcfm.py       # Pendulum: S¹×ℝ
│   └── cartpole_lcfm.py       # CartPole: ℝ²×S¹×ℝ
│
├── flow_matching/             # Flow matching implementations
│   ├── base/                  # Base classes (shared)
│   │   ├── flow_matcher.py   # BaseFlowMatcher (220+ lines of shared code)
│   │   └── config.py         # Configuration
│   ├── latent_conditional/   # Pendulum Latent Conditional FM
│   │   ├── flow_matcher_fb.py
│   │   └── train.py
│   ├── cartpole_latent_conditional/  # CartPole Latent Conditional FM
│   │   ├── flow_matcher_fb.py
│   │   └── train.py
│   └── cartpole_gaussian_perturbed/  # CartPole Gaussian Perturbed FM
│       ├── flow_matcher.py
│       ├── inference.py
│       └── train.py
│
├── model/                     # Neural network architectures
│   ├── latent_conditional_unet1d.py        # Pendulum UNet
│   ├── cartpole_latent_conditional_unet1d.py  # CartPole UNet (latent)
│   └── cartpole_gaussian_perturbed_unet1d.py  # CartPole UNet (simple)
│
├── data/                      # Data loading
│   ├── endpoint_data.py       # Pendulum endpoint data
│   └── cartpole_endpoint_data.py  # CartPole endpoint data
│
└── evaluation/                # Evaluation tools
    └── ...

configs/                       # Hydra configuration files
├── train_pendulum_lcfm.yaml
├── train_cartpole_lcfm.yaml
├── train_cartpole_gaussian_perturbed.yaml
├── evaluate_pendulum_roa.yaml
└── evaluate_cartpole_roa.yaml
```

### **Model Variants**

| Variant | Systems | Latent | Conditioning | Params | Speed |
|---------|---------|--------|--------------|--------|-------|
| **Latent Conditional** | Pendulum, CartPole | ✅ z ~ N(0,I) | ✅ On start state | ~2M | Baseline |
| **Gaussian Perturbed** | CartPole only | ❌ None | ❌ None | ~1.5M | 25% faster |

**Latent Conditional**: Richer multimodal predictions, explicit conditioning
**Gaussian Perturbed**: Simpler architecture, explicit Gaussian noise, faster training

---

## 🎓 Key Concepts

### **Why Circular Coordinates Matter**

Both systems have angles that **wrap around**:
- Pendulum: θ = -π is the same as θ = +π
- CartPole: Pole angle θ wraps at ±π

**Standard Euclidean interpolation breaks this!**
- Bad: Linear interpolation from θ = -3.1 to θ = 3.1 goes through θ = 0 (wrong path!)
- Good: Geodesic interpolation wraps around correctly (shortest path on circle)

This is why we use:
1. **Facebook Flow Matching**: Automatic geodesic interpolation on manifolds
2. **Product Manifolds**: ℝ components + S¹ components treated correctly
3. **RiemannianODESolver**: Manifold-aware ODE integration

### **State Representations**

**Pendulum** (2D → 3D embedding):
```
Raw:      (θ, θ̇)                    ∈ S¹ × ℝ
Embedded: (sin θ, cos θ, θ̇_norm)    ∈ ℝ³
```

**CartPole** (4D → 5D embedding):
```
Raw:      (x, θ, ẋ, θ̇)                        ∈ ℝ² × S¹ × ℝ
Embedded: (x_norm, sin θ, cos θ, ẋ_norm, θ̇_norm) ∈ ℝ⁵
```

The sin/cos embedding allows neural networks to learn the circular geometry naturally.

---

## 📊 Performance

### **Training Times** (500 epochs, single GPU)
- Pendulum: ~2-4 hours
- CartPole (Latent Conditional): ~4-6 hours
- CartPole (Gaussian Perturbed): ~3-5 hours (25% faster)

### **Expected Metrics**
- Training loss: ~1.0-3.0 → ~0.1-0.5
- Validation MAE: < 0.3 (Pendulum), < 0.5 (CartPole) per dimension
- ROA accuracy: 85-95%
- ROA AUC (probabilistic): > 0.90

---

## 🔬 Research Features

### **Training**
- ✅ Latent variables for stochasticity
- ✅ Conditioning on start states
- ✅ Geodesic interpolation on manifolds
- ✅ Automatic validation inference (every 10 epochs)
- ✅ Per-dimension MAE with geodesic distance
- ✅ TensorBoard logging

### **Evaluation**
- ✅ Deterministic ROA evaluation (single prediction)
- ✅ Probabilistic ROA evaluation (multiple samples)
- ✅ Separatrix detection (uncertain regions)
- ✅ Uncertainty quantification via entropy
- ✅ State space visualization
- ✅ Attractor basin analysis

### **Inference**
- ✅ Single endpoint prediction
- ✅ Batch prediction with multiple samples
- ✅ Full trajectory generation
- ✅ Attractor convergence checking
- ✅ Uncertainty quantification

---

## 🛠️ Requirements

```bash
# Core dependencies
pytorch >= 2.0.0
pytorch-lightning >= 2.0.0
hydra-core >= 1.3.0
torchmetrics

# Facebook Flow Matching (included as submodule)
# Located at: flow_matching/

# Environment
conda activate /common/users/dm1487/envs/arcmg
```

---

## 📖 Usage Examples

### **Custom Training Parameters**

```bash
# Change latent dimension
python src/flow_matching/latent_conditional/train.py \
    flow_matching.latent_dim=4

# Change learning rate and batch size
python src/flow_matching/cartpole_latent_conditional/train.py \
    base_lr=5e-4 \
    batch_size=512

# Use different GPU
python src/flow_matching/cartpole_latent_conditional/train.py \
    device=gpu2

# Adjust model architecture
python src/flow_matching/latent_conditional/train.py \
    model.hidden_dims=[512,1024,2048,1024,512]
```

### **Batch Inference with Uncertainty**

```python
# Load model
model = CartPoleLatentConditionalFlowMatcher.load_from_checkpoint("path/to/checkpoint")

# Multiple samples for uncertainty quantification
start_states = torch.tensor([[0.5, 0.1, 2.0, 1.0]])
endpoints_batch = model.predict_endpoints_batch(
    start_states,
    num_samples=20  # 20 samples per start state
)

# Reshape to [batch, samples, dims]
endpoints_reshaped = endpoints_batch.reshape(1, 20, 4)
mean = endpoints_reshaped.mean(dim=1)
std = endpoints_reshaped.std(dim=1)

print(f"Mean endpoint: {mean}")
print(f"Uncertainty (std): {std}")
```

---

## 🤝 Contributing

This is a research codebase. To add a new dynamical system:

1. Define system in `src/systems/newsystem_lcfm.py`
2. Implement flow matcher in `src/flow_matching/newsystem_latent_conditional/`
3. Create model in `src/model/newsystem_latent_conditional_unet1d.py`
4. Add config in `configs/train_newsystem_lcfm.yaml`

See [NEW_SYSTEM_IMPLEMENTATION_GUIDE.md](NEW_SYSTEM_IMPLEMENTATION_GUIDE.md) for detailed instructions.

---

## 📄 License

Research code for AI Olympics project.

---

## 📞 Support

- **Training issues**: See [TRAINING_GUIDE.md](TRAINING_GUIDE.md) → Troubleshooting section
- **CartPole specific**: See [CARTPOLE_DOCUMENTATION.md](CARTPOLE_DOCUMENTATION.md)
- **ROA evaluation**: See [ROA_ANALYSIS_GUIDE.md](ROA_ANALYSIS_GUIDE.md)

---

## 🎯 Quick Reference

```bash
# Train
python src/flow_matching/latent_conditional/train.py                    # Pendulum
python src/flow_matching/cartpole_latent_conditional/train.py          # CartPole (rich)
python src/flow_matching/cartpole_gaussian_perturbed/train.py          # CartPole (fast)

# Evaluate
python src/flow_matching/evaluate_roa.py --config-name=evaluate_cartpole_roa

# Monitor
tensorboard --logdir outputs/
```

**For complete documentation, see [TRAINING_GUIDE.md](TRAINING_GUIDE.md)** 🚀
