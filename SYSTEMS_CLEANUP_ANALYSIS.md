# Systems Cleanup Analysis

## What's Actually Used by Flow Matchers & ROA Evaluation

### CartPole LCFM Flow Matcher (`cartpole_latent_conditional/flow_matcher_fb.py`)

**Required from `CartPoleSystemLCFM`:**
- `state_dim` (from base class) - Used for validation metrics
- `cart_limit` - For normalizing cart position
- `velocity_limit` - For normalizing cart velocity  
- `angular_velocity_limit` - For normalizing angular velocity
- `manifold_components` (from base class) - For validation logging

**Required for ROA Evaluation (`evaluate_roa.py`):**
- `is_in_attractor(state, radius)` → Returns Tensor[B] - Check attractor membership

**NOT USED:**
- `get_attractor_labels()` - Never called
- `is_balanced()` - Never called
- `attractors()` - Only used internally by `is_in_attractor()`

---

### Pendulum LCFM Flow Matcher (`latent_conditional/flow_matcher_fb.py`)

**Required from `PendulumSystemLCFM`:**
- `state_dim` (from base class) - Used for validation metrics
- `embed_state(state)` → Embeds (θ, θ̇) to (sin θ, cos θ, θ̇) - Used in training & inference
- `manifold_components` (from base class) - For validation logging

**Required for ROA Evaluation (if we add it):**
- `is_in_attractor(state, radius)` → Returns Tensor[B] - Check attractor membership

**NOT USED:**
- `get_attractor_labels()` - Never called
- `attractors()` - Only used internally by `is_in_attractor()`

---

## Current System Files

### ✅ **KEEP - Required for Flow Matching:**

1. **`base.py`** - Base class with:
   - `state_dim` property
   - `manifold_components` property
   - `define_manifold_structure()` abstract method
   - `define_state_bounds()` abstract method

2. **`cartpole_lcfm.py`** - CartPole LCFM system:
   - Defines ℝ²×S¹×ℝ manifold structure
   - Provides normalization bounds
   - **KEEP:** `__init__`, `define_manifold_structure()`, `define_state_bounds()`, `is_in_attractor()`
   - **REMOVE:** `get_attractor_labels()`, `is_balanced()`, `attractors()` (make private if needed by is_in_attractor)

3. **`pendulum_lcfm.py`** - Pendulum LCFM system:
   - Defines S¹×ℝ manifold structure
   - Provides `embed_state()` for (sin θ, cos θ, θ̇) conversion
   - **KEEP:** `define_manifold_structure()`, `define_state_bounds()`, `is_in_attractor()`
   - **REMOVE:** `get_attractor_labels()`, `attractors()` (make private)

### ❓ **LEGACY - Used by Old Tools:**

4. **`pendulum_config.py`** - Config-based approach
   - Used by: `evaluation/`, `visualization/`, and one spot in `latent_conditional/inference.py`
   - Could be replaced with `PendulumSystemLCFM` if we update those tools

5. **`pendulum_universal.py`** - Alternative pendulum implementation
   - Exported in `__init__.py` but mainly used by old evaluation tools

6. **`cartpole.py`** - Alternative CartPole implementation  
   - Exported in `__init__.py` but NOT used by flow matchers
   - May be used by old scripts

7. **`pendulum.py`** - Old pendulum with `BaseSystem`
   - Appears to be legacy

---

## Recommendations

### Minimal Required for Flow Matching + ROA:

**Keep these 3 files only:**
1. `base.py` - Base class
2. `cartpole_lcfm.py` - CartPole system (cleaned up)
3. `pendulum_lcfm.py` - Pendulum system (cleaned up)

### Clean Up Each System:

**CartPoleSystemLCFM - Remove:**
- `get_attractor_labels()` - unused
- `is_balanced()` - unused  
- Make `attractors()` private (`_attractors()`) since only used internally

**PendulumSystemLCFM - Remove:**
- `get_attractor_labels()` - unused
- Make `attractors()` private (`_attractors()`) since only used internally

### Legacy Files Decision:

**Option 1: Archive them** (move to `src/systems/legacy/`)
- `cartpole.py`
- `pendulum.py`
- `pendulum_universal.py`
- `pendulum_config.py`

**Option 2: Update old tools to use LCFM systems**
- Update `evaluation/`, `visualization/` to use `PendulumSystemLCFM`
- Then delete legacy files

**Option 3: Keep for now** (if unsure about dependencies)
- Leave them but document they're not used by flow matching

---

## Summary of Methods Actually Called

### CartPoleSystemLCFM (in flow_matcher_fb.py):
```python
self.system.state_dim                    # Line 83
self.system.cart_limit                   # Lines 104, 147, 169
self.system.velocity_limit               # Lines 107, 148, 170  
self.system.angular_velocity_limit       # Lines 113, 149, 171
self.system.manifold_components          # Lines 363, 383
self.system.is_in_attractor(endpoints, radius)  # evaluate_roa.py
```

### PendulumSystemLCFM (in flow_matcher_fb.py):
```python
self.system.state_dim                    # Line 82
self.system.embed_state(x)               # Lines 183, 186, 414, 424
self.system.manifold_components          # Lines 219, 323, 343
self.system.is_in_attractor(endpoints, radius)  # For ROA eval
```


