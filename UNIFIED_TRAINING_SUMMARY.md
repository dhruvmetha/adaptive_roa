# ✅ Unified Training System - Quick Reference

## 🚀 **How to Train**

### **Pendulum:**
```bash
python src/flow_matching/train_latent_conditional.py --config-name=train_pendulum_lcfm
```

### **CartPole:**
```bash
python src/flow_matching/train_latent_conditional.py --config-name=train_cartpole_lcfm
```

---

## 📁 **What Changed**

### **Before (Separate Scripts):**
```
src/flow_matching/latent_conditional/train.py          ← Pendulum only
src/flow_matching/cartpole_latent_conditional/train.py ← CartPole only
```

### **After (Unified Script):**
```
src/flow_matching/train_latent_conditional.py          ← Works for BOTH!
```

---

## 🎯 **How It Works**

**ONE script** + **Different configs** = **Different systems**

```bash
# Same script, different config names
python train_latent_conditional.py --config-name=train_pendulum_lcfm
python train_latent_conditional.py --config-name=train_cartpole_lcfm
#                                                  ^^^^^^^^
#                                            ONLY THIS CHANGES!
```

---

## 📋 **Config Files**

All configuration is in YAML files:

```
configs/
  ├── train_pendulum_lcfm.yaml  ← Pendulum: S¹×ℝ manifold
  └── train_cartpole_lcfm.yaml  ← CartPole: ℝ²×S¹×ℝ manifold
```

Each config specifies:
- System class (`_target_: src.systems.pendulum_lcfm.PendulumSystemLCFM`)
- Flow matcher class (`_target_: src.flow_matching.latent_conditional.flow_matcher_fb.LatentConditionalFlowMatcher`)
- Model architecture
- Data paths
- Optimizer settings
- Trainer configuration

---

## 🔧 **Override Parameters**

```bash
# Change latent dimension
python train_latent_conditional.py \
    --config-name=train_pendulum_lcfm \
    flow_matching.latent_dim=4

# Change learning rate
python train_latent_conditional.py \
    --config-name=train_pendulum_lcfm \
    base_lr=5e-4

# Change architecture
python train_latent_conditional.py \
    --config-name=train_pendulum_lcfm \
    model.hidden_dims=[512,1024,512]
```

---

## ✅ **Benefits**

| Aspect | Before | After |
|--------|--------|-------|
| **Scripts to maintain** | 2+ (one per system) | 1 (unified) |
| **To switch systems** | Run different script | Change config name |
| **To override params** | Edit code | Command line flag |
| **Code duplication** | High (~100 lines) | Zero |
| **Extensibility** | Copy entire script | Copy config file |

---

## 📝 **Key Files**

- **Training Script:** `src/flow_matching/train_latent_conditional.py`
- **Pendulum Config:** `configs/train_pendulum_lcfm.yaml`
- **CartPole Config:** `configs/train_cartpole_lcfm.yaml`
- **Full Guide:** `UNIFIED_TRAINING_GUIDE.md`

---

## 🎉 **Summary**

**Before:**
- 2 separate training scripts
- Copy-paste code duplication
- Hard to maintain

**After:**
- 1 unified training script
- Config-driven system selection
- Easy to maintain and extend

**Usage:**
```bash
# Just change the config name!
python train_latent_conditional.py --config-name=train_pendulum_lcfm
python train_latent_conditional.py --config-name=train_cartpole_lcfm
```

That's it! 🚀
