# 🚀 Facebook Flow Matching - Quick Reference Card

## 📚 **Documentation Index**

| Guide | Purpose | When to Read |
|-------|---------|--------------|
| **QUICK_REFERENCE.md** (this file) | Quick lookup | Always |
| **UNIFIED_TRAINING_SUMMARY.md** | How to use unified training | First time user |
| **UNIFIED_TRAINING_GUIDE.md** | Detailed training documentation | For customization |
| **NEW_SYSTEM_IMPLEMENTATION_GUIDE.md** | Adding new systems | Implementing new system |
| **PENDULUM_REFACTOR_COMPARISON.md** | Before/After for Pendulum | Understanding changes |
| **FB_FM_INTEGRATION_GUIDE.md** | FB FM technical details | Deep dive |

---

## ⚡ **Quick Commands**

### **Training**
```bash
# Pendulum
python src/flow_matching/train_latent_conditional.py --config-name=train_pendulum_lcfm

# CartPole
python src/flow_matching/train_latent_conditional.py --config-name=train_cartpole_lcfm

# Override parameters
python src/flow_matching/train_latent_conditional.py \
    --config-name=train_pendulum_lcfm \
    flow_matching.latent_dim=4 \
    base_lr=5e-4 \
    trainer.max_epochs=200
```

### **Testing**
```bash
# Test Pendulum
python test_pendulum_fb_fm.py

# Test CartPole
python test_cartpole_fb_fm.py
```

---

## 📁 **File Locations**

### **Training**
- Unified script: `src/flow_matching/train_latent_conditional.py`
- Pendulum config: `configs/train_pendulum_lcfm.yaml`
- CartPole config: `configs/train_cartpole_lcfm.yaml`

### **Flow Matchers (Facebook FM)**
- Pendulum: `src/flow_matching/latent_conditional/flow_matcher_fb.py`
- CartPole: `src/flow_matching/cartpole_latent_conditional/flow_matcher_fb.py`

### **Manifolds**
- All manifolds: `src/utils/fb_manifolds.py`
  - `PendulumManifold` (S¹×ℝ)
  - `CartPoleManifold` (ℝ²×S¹×ℝ)

### **Systems**
- Pendulum: `src/systems/pendulum_lcfm.py`
- CartPole: `src/systems/cartpole_lcfm.py`

---

## 🔑 **Key Concepts**

### **State Spaces**
| System | Manifold | State | Embedded Dim |
|--------|----------|-------|--------------|
| Pendulum | S¹×ℝ | (θ, θ̇) | 3: (sin θ, cos θ, θ̇) |
| CartPole | ℝ²×S¹×ℝ | (x, θ, ẋ, θ̇) | 5: (x, sin θ, cos θ, ẋ, θ̇) |

### **Facebook FM Components**
1. **Manifold**: Defines geometry (`logmap`, `expmap`, `projx`)
2. **GeodesicProbPath**: Automatic interpolation + velocity
3. **RiemannianODESolver**: Manifold-aware ODE integration

### **What Changed from Manual Implementation**
```python
# OLD (Manual - ~105 lines)
x_t = interpolate_s1_x_r(noise, data, t)          # 40 lines
velocity = compute_target_velocity_s1_x_r(...)    # 65 lines

# NEW (Facebook FM - 1 line!)
path_sample = self.path.sample(noise, data, t)
# path_sample.x_t: interpolation (automatic)
# path_sample.dx_t: velocity (automatic via autodiff!)
```

---

## 🎯 **Common Tasks**

### **Switch Between Systems**
Just change the config name:
```bash
--config-name=train_pendulum_lcfm  # Pendulum
--config-name=train_cartpole_lcfm  # CartPole
```

### **Change Latent Dimension**
```bash
python train_latent_conditional.py \
    --config-name=train_pendulum_lcfm \
    flow_matching.latent_dim=4
```

### **Change Model Architecture**
```bash
python train_latent_conditional.py \
    --config-name=train_pendulum_lcfm \
    model.hidden_dims=[512,1024,512]
```

### **Change Learning Rate**
```bash
python train_latent_conditional.py \
    --config-name=train_pendulum_lcfm \
    base_lr=5e-4
```

---

## 🆕 **Adding a New System**

**7 Steps** (see NEW_SYSTEM_IMPLEMENTATION_GUIDE.md for details):

1. **Manifold** in `src/utils/fb_manifolds.py`
   ```python
   class NewSystemManifold(Manifold):
       def expmap(self, x, u): ...
       def logmap(self, x, y): ...
       def projx(self, x): ...
       def proju(self, x, u): ...
   ```

2. **System** in `src/systems/newsystem_lcfm.py`
   ```python
   class NewSystemLCFM:
       def normalize_state(self, state): ...
       def denormalize_state(self, state): ...
       def embed_state(self, state): ...
   ```

3. **Flow Matcher** in `src/flow_matching/newsystem_latent_conditional/flow_matcher_fb.py`
   ```python
   class NewSystemLatentConditionalFlowMatcher(BaseFlowMatcher):
       def __init__(self, ...):
           self.manifold = NewSystemManifold()
           self.path = GeodesicProbPath(...)
   ```

4. **Model** in `src/model/newsystem_latent_conditional_unet1d.py`

5. **Data Module** in `src/data/newsystem_endpoint_data.py`

6. **Config** in `configs/train_newsystem_lcfm.yaml`
   ```yaml
   system:
     _target_: src.systems.newsystem_lcfm.NewSystemLCFM
   flow_matcher:
     _target_: src.flow_matching.newsystem_latent_conditional.flow_matcher_fb.NewSystemLatentConditionalFlowMatcher
   ```

7. **Test** in `test_newsystem_fb_fm.py`

Then train:
```bash
python src/flow_matching/train_latent_conditional.py --config-name=train_newsystem_lcfm
```

---

## 🔍 **Debugging Checklist**

### **Training not working?**
- [ ] Check PyTorch version ≥ 2.0 (`torch.func` needed)
- [ ] Verify `embedded_dim` matches `system.embed_state()` output
- [ ] Verify `output_dim` matches state dimension
- [ ] Check data file paths in config
- [ ] Ensure angles in data are in [-π, π]

### **Loss is NaN?**
- [ ] Reduce learning rate
- [ ] Check state normalization bounds
- [ ] Verify manifold `logmap`/`expmap` work correctly
- [ ] Test manifold separately first

### **Model doesn't learn?**
- [ ] Verify data quality (endpoints make sense?)
- [ ] Check latent_dim consistency
- [ ] Try different initialization
- [ ] Increase model capacity (hidden_dims)

---

## 📊 **State Dimension Reference**

```python
# Formula
embedded_dim = (num_angles × 2) + num_linear_quantities

# Examples
Pendulum:        1 angle × 2 + 1 velocity = 3
CartPole:        1 angle × 2 + 3 linear = 5
Acrobot:         2 angles × 2 + 2 velocities = 6
Double Pendulum: 2 angles × 2 + 2 velocities = 6
Quadcopter:      3 angles × 2 + 9 linear = 15
```

---

## ✅ **What You Have**

- ✅ **Pendulum**: Fully implemented with FB FM
- ✅ **CartPole**: Fully implemented with FB FM
- ✅ **Unified Training**: One script for all systems
- ✅ **Tests**: Validation for both systems
- ✅ **Documentation**: Complete guides

---

## 🎉 **Benefits Summary**

| Aspect | Manual | Facebook FM |
|--------|--------|-------------|
| **Interpolation** | 40 lines | 1 line |
| **Velocity** | 65 lines (Theseus) | Automatic (autodiff) |
| **Inference** | Manual Euler only | 3 solvers (Euler/RK4/Midpoint) |
| **Code to maintain** | ~105 lines/system | ~0 lines (library) |
| **Correctness** | Manual verification | Guaranteed by autodiff |
| **Extensibility** | Copy 105 lines | Copy 5 lines (registry) |

---

## 📞 **Getting Help**

1. **Quick lookup**: This file
2. **How to train**: UNIFIED_TRAINING_SUMMARY.md
3. **Implementation details**: UNIFIED_TRAINING_GUIDE.md
4. **Adding new system**: NEW_SYSTEM_IMPLEMENTATION_GUIDE.md
5. **Technical deep dive**: FB_FM_INTEGRATION_GUIDE.md

---

**Ready to train!** 🚀

```bash
python src/flow_matching/train_latent_conditional.py --config-name=train_pendulum_lcfm
```
