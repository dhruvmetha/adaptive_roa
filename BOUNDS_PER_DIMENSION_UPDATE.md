# Per-Dimension Bounds Update

## Summary

Updated Humanoid system to use **per-dimension bounds** (like CartPole) instead of global bounds.

## What Changed

### Before (Global Bounds)
```python
# Single limit for all Euclidean dimensions
self.euclidean_limit = 20.0

# Normalize all Euclidean dims by same limit
normalized[:, :34] = state[:, :34] / self.euclidean_limit
normalized[:, 37:] = state[:, 37:] / self.euclidean_limit
```

### After (Per-Dimension Bounds)
```python
# Per-dimension limits (like CartPole)
self.dimension_limits = {
    0: 18.2,
    1: 21.5,
    2: 15.7,
    ...
    66: 19.3
}

# Normalize each dimension individually
for i in range(67):
    if 34 <= i <= 36:  # Sphere: no normalization
        continue
    else:
        normalized[:, i] = state[:, i] / self.dimension_limits[i]
```

## Modified Files

### 1. scripts/compute_humanoid_bounds.py
**Before**: Computed global `euclidean_limit` for all 64 Euclidean dimensions

**After**: Computes per-dimension bounds
```python
# Initialize bounds for ALL dimensions
dimension_bounds = {i: {'min': inf, 'max': -inf} for i in range(67)}

# Update all dimensions
for i in range(67):
    dimension_bounds[i]['min'] = min(...)
    dimension_bounds[i]['max'] = max(...)

# Save per-dimension bounds
bounds_data = {
    'bounds': {i: dimension_bounds[i] for i in range(67)},
    'ranges': {i: max - min for i in range(67)},
    'statistics': {...}
}
```

**Output format** (same as CartPole, but NO sphere bounds):
```python
{
    'bounds': {
        0: {'min': -18.2, 'max': 21.5},      # Euclidean block 1
        1: {'min': -15.7, 'max': 19.3},
        ...
        33: {'min': -16.3, 'max': 20.1},
        # 34-36: NO BOUNDS (sphere, always unit norm)
        37: {'min': -17.1, 'max': 20.8},     # Euclidean block 2
        ...
        66: {'min': -18.5, 'max': 21.2}
    },
    'statistics': {
        'euclidean_dimensions': 64,          # 34 + 30
        'sphere_dimensions': 3               # dims 34-36 (no bounds)
    }
}
```

### 2. src/systems/humanoid.py

**`_load_bounds_from_file()`**:
```python
# Load per-dimension bounds (only Euclidean dims)
self.dimension_bounds = bounds_data.get('bounds', {})

# Compute symmetric limits for Euclidean dimensions only
self.dimension_limits = {}
for i in range(67):
    if 34 <= i <= 36:
        # Sphere: no limit needed (no normalization)
        self.dimension_limits[i] = 1.0  # Placeholder, not used
    elif i in self.dimension_bounds:
        # Euclidean: compute symmetric limit
        min_val = self.dimension_bounds[i]['min']
        max_val = self.dimension_bounds[i]['max']
        self.dimension_limits[i] = max(abs(min_val), abs(max_val))
    else:
        # Default fallback
        self.dimension_limits[i] = 20.0
```

**`normalize_state()`**:
```python
# Normalize each dimension individually
for i in range(67):
    if 34 <= i <= 36:  # Sphere: no normalization
        continue
    else:
        normalized[:, i] = state[:, i] / self.dimension_limits[i]
```

**`denormalize_state()`**:
```python
# Denormalize each dimension individually
for i in range(67):
    if 34 <= i <= 36:  # Sphere: no denormalization
        continue
    else:
        denormalized[:, i] = normalized_state[:, i] * self.dimension_limits[i]
```

### 3. src/flow_matching/humanoid/latent_conditional/flow_matcher.py

**`sample_noisy_input()`**:
```python
# Sample each Euclidean dimension with its own limit
samples = []
for i in range(67):
    if 34 <= i <= 36:  # Sphere: sample separately
        continue
    else:
        limit = self.system.dimension_limits[i]
        sample = torch.rand(batch_size, 1, device=device) * (2 * limit) - limit
        samples.append(sample)

# Combine Euclidean + Sphere
euclidean1 = torch.cat(samples[:34], dim=1)
euclidean2 = torch.cat(samples[34:], dim=1)
sphere = torch.randn(batch_size, 3, device=device)
sphere = sphere / torch.norm(sphere, dim=1, keepdim=True)

return torch.cat([euclidean1, sphere, euclidean2], dim=1)
```

### 4. LLM_PROMPT_ADD_NEW_SYSTEM.txt

Updated instructions to use per-dimension bounds for all new systems:
- Section 3: Normalization rules (per-dimension)
- Section 4: Sampling on manifold (per-dimension limits)
- Section 8: Bounds computation (per-dimension structure)

## Why Per-Dimension?

### Advantages
✅ More precise normalization (each dim optimally scaled)
✅ Consistent with CartPole (easier to understand)
✅ Allows different scaling for different dimension types
✅ Better for dimensions with very different ranges

### Pattern Across Systems

**CartPole** (4D):
- `cart_limit = 2.5`
- `velocity_limit = 10.0`
- `angular_velocity_limit = 15.0`

**Humanoid** (67D, only Euclidean dims):
- `dimension_limits[0] = 18.2`    (Euclidean)
- `dimension_limits[1] = 21.5`    (Euclidean)
- ...
- `dimension_limits[33] = 20.1`   (Euclidean)
- `dimension_limits[34-36]` = N/A (Sphere, no normalization)
- `dimension_limits[37] = 17.8`   (Euclidean)
- ...
- `dimension_limits[66] = 19.3`   (Euclidean)

**Same pattern**, just different scale!

**Key difference**: Sphere dimensions (34-36) have NO bounds and NO normalization.

## Usage

### 1. Compute Bounds (Updated Script)
```bash
python scripts/compute_humanoid_bounds.py \
    --data_dir /common/users/shared/pracsys/genMoPlan/data_trajectories/humanoid_get_up/trajectories \
    --output /common/users/dm1487/arcmg_datasets/humanoid_get_up/humanoid_data_bounds.pkl
```

**Output**:
```
📊 Computed Bounds (Per-Dimension)
================================================================================
Files processed: 100
Total states: 31,600

Manifold Structure: ℝ³⁴ × S² × ℝ³⁰ (67-dimensional state)

Euclidean Block 1 (dims 0-33):
  [ 0] euclidean1_ 0: [ -18.198,   21.536]  range=  39.734
  [ 1] euclidean1_ 1: [ -15.782,   19.345]  range=  35.127
  ...

Sphere (dims 34-36) - 3D unit vector:
  NO BOUNDS COMPUTED (always unit norm, no normalization needed)

Euclidean Block 2 (dims 37-66):
  [37] euclidean2_ 0: [ -17.123,   20.876]  range=  37.999
  ...

Per-dimension bounds stored for Euclidean dims only
  64 Euclidean dimensions: keys [0-33, 37-66]
  3 Sphere dimensions [34-36]: NO bounds (always unit norm)
```

### 2. Training (No Changes Required)

Training workflow is **identical** - bounds are loaded automatically:

```bash
python src/flow_matching/humanoid/latent_conditional/train.py
```

The system will print:
```
✅ Loaded Humanoid bounds (per-dimension)
   64 Euclidean dimension limits loaded
   3 Sphere dimensions (34-36): NO normalization (always unit norm)
```

### 3. Verification

Check that normalization works correctly:
```python
from src.systems.humanoid import HumanoidSystem
import torch

system = HumanoidSystem(
    bounds_file="/path/to/humanoid_data_bounds.pkl",
    use_dynamic_bounds=True
)

# Test state
state = torch.randn(1, 67)
state[:, 34:37] = state[:, 34:37] / torch.norm(state[:, 34:37])  # Make sphere unit norm

# Normalize
normalized = system.normalize_state(state)

# Check ranges
print(f"Normalized ranges:")
print(f"  Dims 0-33: [{normalized[:, :34].min():.2f}, {normalized[:, :34].max():.2f}]")
print(f"  Dims 34-36 (sphere): [{normalized[:, 34:37].min():.2f}, {normalized[:, 34:37].max():.2f}]")
print(f"  Dims 37-66: [{normalized[:, 37:].min():.2f}, {normalized[:, 37:].max():.2f}]")

# Denormalize and verify
denormalized = system.denormalize_state(normalized)
assert torch.allclose(state, denormalized, atol=1e-5)
print("✅ Normalization/denormalization are inverses")
```

## Backward Compatibility

**Old bounds files** (with global `euclidean_limit`) will still work:
- If `dimension_limits` not found, falls back to default bounds
- Default: `dimension_limits = {i: 20.0 for i in range(67)}`

**New bounds files** (per-dimension) provide better normalization.

## For New Systems

Follow this pattern for **all new systems**:

1. **Compute per-dimension bounds** in `scripts/compute_<system>_bounds.py`
2. **Load per-dimension limits** in `<System>System._load_bounds_from_file()`
3. **Normalize per-dimension** in `normalize_state()` and `denormalize_state()`
4. **Sample per-dimension** in `sample_noisy_input()`

See updated `LLM_PROMPT_ADD_NEW_SYSTEM.txt` for complete instructions.
