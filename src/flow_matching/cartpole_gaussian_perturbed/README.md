# CartPole Gaussian-Perturbed Flow Matching

Simplified flow matching variant **without latent variables or conditioning**.

## Key Features

- **NO latent variables** (removed z ~ N(0,I))
- **NO conditioning** on start state
- **Gaussian-perturbed initial states**: x₀ ~ N(start_state, σ²I)
- **Simpler model**: f(x_t, t) instead of f(x_t, t, z, condition)
- **25% fewer parameters** than latent conditional variant
- **Faster training/inference** due to simpler architecture

## Quick Start

### Training

```bash
# Default training
python src/flow_matching/cartpole_gaussian_perturbed/train.py

# Adjust Gaussian noise std
python src/flow_matching/cartpole_gaussian_perturbed/train.py \
    flow_matching.noise_std=0.2

# Change batch size and learning rate
python src/flow_matching/cartpole_gaussian_perturbed/train.py \
    batch_size=512 \
    base_lr=5e-5
```

### Inference

```python
from src.flow_matching.cartpole_gaussian_perturbed.inference import CartPoleGaussianPerturbedInference
import torch

# Load trained model
inferencer = CartPoleGaussianPerturbedInference(
    "outputs/cartpole_gaussian_perturbed_fm/2025-10-17_14-15-30"
)

# Single prediction
start_states = torch.tensor([[0.5, 0.1, 2.0, 1.0]])  # [x, θ, ẋ, θ̇]
endpoint = inferencer.predict_endpoint(start_states)

# Multiple samples for uncertainty
endpoints = inferencer.predict_endpoints_batch(start_states, num_samples=20)

# Uncertainty quantification
uncertainty = inferencer.compute_uncertainty(start_states, num_samples=20)
print(f"Mean prediction: {uncertainty['mean']}")
print(f"Standard deviation: {uncertainty['std']}")

# Attractor convergence
convergence = inferencer.check_attractor_convergence(start_states)
print(f"Success rate: {convergence['proportion_success']}")
```

## How It Works

### Training

1. **Sample perturbed input**: x₀ ~ N(start_state, σ²I)
2. **Interpolate geodesically**: From x₀ to target endpoint x₁
3. **Predict velocity**: Model predicts dx/dt at interpolated points
4. **Minimize loss**: MSE between predicted and target velocities

### Inference

1. **Normalize** start state to [-1, 1]² × [-π, π] × [-1, 1]
2. **Sample** perturbed initial state from Gaussian
3. **Integrate** forward using learned velocity field
4. **Denormalize** final state back to physical coordinates

## Configuration

Key parameters in `configs/train_cartpole_gaussian_perturbed.yaml`:

```yaml
flow_matching:
  noise_std: 0.1  # Gaussian noise standard deviation
                  # Smaller = closer to start state
                  # Larger = more exploration

model:
  embedded_dim: 5              # (x_norm, sin θ, cos θ, ẋ_norm, θ̇_norm)
  time_emb_dim: 64             # Time embedding dimension
  hidden_dims: [256, 512, 1024, 512, 256]
  output_dim: 4                # Velocity (dx, dθ, dẋ, dθ̇)
```

## Differences from Latent Conditional Variant

| Aspect | This Variant | Latent Conditional |
|--------|-------------|-------------------|
| Latent Variable | ❌ None | ✅ z ~ N(0,I) |
| Conditioning | ❌ None | ✅ On start state |
| Initial Noise | Gaussian around start | Uniform from state space |
| Model Input | (x_t, t) | (x_t, t, z, condition) |
| Parameters | ~1.5M | ~2M |
| Training Speed | Faster | Slower |
| Stochasticity | Gaussian perturbation | Latent variable |

## Files

- `flow_matcher.py`: Main flow matching module and model wrapper
- `train.py`: Training script
- `inference.py`: Inference wrapper with utilities
- `README.md`: This file

## References

- Main documentation: `/CARTPOLE_DOCUMENTATION.md`
- Configuration: `/configs/train_cartpole_gaussian_perturbed.yaml`
- System definition: `/src/systems/cartpole_lcfm.py`
- Model architecture: `/src/model/cartpole_gaussian_perturbed_unet1d.py`
