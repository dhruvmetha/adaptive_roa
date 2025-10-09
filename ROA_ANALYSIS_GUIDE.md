# RoA Analysis Guide for Latent Conditional Flow Matching

This guide explains how to perform Region of Attraction (RoA) analysis on trained LCFM models.

## Overview

The RoA analysis system has been adapted to work with the latent conditional flow matching model. It provides:

- **Deterministic Analysis**: Fast basin mapping with random latent sampling
- **Probabilistic Analysis**: Uncertainty-aware basin mapping with Monte Carlo sampling
- **Comprehensive Visualization**: Basin maps, uncertainty maps, standard deviation maps

## Quick Start

### 1. Train a Model First

```bash
python src/flow_matching/train_latent_conditional.py --config-name=train_pendulum_lcfm
```

### 2. Run RoA Analysis

**Basic deterministic analysis:**
```bash
python demo_roa_analysis.py
```

**Probabilistic analysis with uncertainty:**
```bash
python demo_roa_analysis.py --probabilistic --num-samples 100
```

**Both analyses:**
```bash
python demo_roa_analysis.py --both
```

**Custom resolution:**
```bash
python demo_roa_analysis.py --resolution 0.05  # Finer grid (slower)
```

**Specify checkpoint:**
```bash
python demo_roa_analysis.py --checkpoint outputs/pendulum_latent_conditional_fm/checkpoints/epoch99-val_loss0.0123.ckpt
```

## Analysis Types

### Deterministic Analysis

- **Fast**: Single endpoint prediction per grid point
- **Stochastic**: Each point samples a random latent variable
- **Good for**: Quick basin visualization, understanding general structure

**Output files:**
- `attractor_basins.png` - Color-coded basin map
- `basin_statistics.png` - Statistical plots
- `basin_analysis_data.npz` - Raw numerical data
- `basin_analysis_report.txt` - Text summary

### Probabilistic Analysis

- **Comprehensive**: Multiple samples per grid point
- **Uncertainty-aware**: Entropy, max probability, margin metrics
- **Standard Deviation**: Endpoint prediction variability
- **Good for**: Understanding model confidence, separatrix detection, uncertainty quantification

**Additional output files:**
- `uncertainty_entropy.png` - Entropy map (higher = more uncertain)
- `uncertainty_pmax.png` - Max probability map
- `probability_heatmap_pmax.png` - Probability heatmap with threshold
- `endpoint_std_combined.png` - Standard deviation maps (magnitude, θ, ω)
- `endpoint_std_magnitude.png` - Overall endpoint variability
- `endpoint_std_theta.png` - Angular component variability
- `endpoint_std_omega.png` - Velocity component variability

## Programmatic Usage

### Method 1: Using Demo Functions

```python
from demo_roa_analysis import load_trained_model, run_deterministic_roa_analysis

# Load model
flow_matcher = load_trained_model("path/to/checkpoint.ckpt")

# Run analysis
results = run_deterministic_roa_analysis(
    flow_matcher,
    resolution=0.1,
    batch_size=1000
)
```

### Method 2: Direct API

```python
import torch
from pathlib import Path
from src.flow_matching.latent_conditional import (
    LatentConditionalFlowMatcher,
    LatentConditionalFlowMatchingInference
)
from src.systems.pendulum_lcfm import PendulumSystemLCFM
from src.systems.pendulum_config import PendulumConfig
from src.model.latent_conditional_unet1d import LatentConditionalUNet1D
from src.visualization.attractor_analysis import AttractorBasinAnalyzer

# 1. Load model
system = PendulumSystemLCFM()
model = LatentConditionalUNet1D(
    embedded_dim=3, latent_dim=2, condition_dim=3,
    time_emb_dim=64, hidden_dims=[256, 512, 256], output_dim=2
)
flow_matcher = LatentConditionalFlowMatcher(
    system=system, model=model,
    optimizer=None, scheduler=None,
    config={}, latent_dim=2
)
checkpoint = torch.load("checkpoint.ckpt")
flow_matcher.load_state_dict(checkpoint['state_dict'])
flow_matcher.eval()

# 2. Create inference wrapper
inferencer = LatentConditionalFlowMatchingInference(
    flow_matcher=flow_matcher,
    num_integration_steps=100,
    integration_method="rk4"
)

# 3. Run analysis
config = PendulumConfig()
analyzer = AttractorBasinAnalyzer(config)

# Deterministic analysis
results_det = analyzer.analyze_attractor_basins(
    inferencer=inferencer,
    resolution=0.1,
    batch_size=1000,
    use_probabilistic=False
)

# Probabilistic analysis
results_prob = analyzer.analyze_attractor_basins(
    inferencer=inferencer,
    resolution=0.1,
    batch_size=500,
    use_probabilistic=True,
    num_samples=64,
    thresholds={
        'entropy': 0.9,   # High entropy → separatrix
        'pmax': 0.55,     # Low max prob → uncertain
        'margin': 0.15    # Small margin → ambiguous
    }
)

# 4. Save results
analyzer.save_analysis_results(Path("output_dir"), results_prob)
```

## Inference API Features

The `LatentConditionalFlowMatchingInference` wrapper provides:

### predict_endpoint()
```python
# Single prediction
endpoint = inferencer.predict_endpoint(
    torch.tensor([[0.5, 0.2]]),  # [θ, θ̇]
    num_steps=100,
    latent=None,  # Random latent
    method="rk4"
)

# Deterministic with fixed latent
latent = torch.randn(1, 2)
endpoint = inferencer.predict_endpoint(
    torch.tensor([[0.5, 0.2]]),
    latent=latent  # Fixed latent
)
```

### sample_endpoints()
```python
# Sample multiple endpoints per state
samples = inferencer.sample_endpoints(
    torch.tensor([[0.5, 0.2], [-0.3, 0.1]]),  # [B, 2]
    num_samples=10
)
# Returns: [B, num_samples, 2]
```

### predict_attractor_distribution()
```python
# Get probability distribution over attractors
probs = inferencer.predict_attractor_distribution(
    torch.tensor([[0.5, 0.2], [-0.3, 0.1]]),
    num_samples=64
)
# Returns: [B, num_attractors] probabilities
```

### batch_predict()
```python
# Batch prediction with uncertainty
states = [[0.5, 0.2], [-0.3, 0.1], [1.0, -0.5]]
means, stds = inferencer.batch_predict(
    states,
    return_std=True,
    num_samples=10
)
```

## Resolution Guidelines

| Resolution | Grid Points | Time (Deterministic) | Time (Probabilistic, N=64) |
|------------|-------------|----------------------|----------------------------|
| 0.2        | ~10,000     | ~30 sec              | ~5 min                     |
| 0.1        | ~40,000     | ~2 min               | ~20 min                    |
| 0.05       | ~160,000    | ~8 min               | ~80 min                    |
| 0.025      | ~640,000    | ~30 min              | ~5 hours                   |

*Times approximate on GPU, may vary based on hardware and model complexity*

## Understanding the Outputs

### Basin Map Colors
- **Red**: Downward attractor basin
- **Teal**: Right attractor basin
- **Blue**: Left attractor basin
- **Light Salmon**: Separatrix region (uncertain)
- **Gray**: No attractor (doesn't converge)

### Uncertainty Metrics
- **Entropy**: Shannon entropy of attractor distribution
  - High → equally likely to reach multiple attractors
  - Low → confidently predicts one attractor
- **Pmax**: Maximum probability across attractors
  - High → confident in one attractor
  - Low → uncertain which attractor
- **Margin**: Difference between top-2 probabilities
  - High → clear winner
  - Low → ambiguous between two attractors

### Standard Deviation Maps
- **Magnitude**: Overall prediction variability (Euclidean norm)
- **θ component**: Angular prediction variability
- **ω component**: Velocity prediction variability

High std → model is uncertain/stochastic in that region

## Troubleshooting

**Issue**: No checkpoint found
- **Solution**: Train model first or specify `--checkpoint` path

**Issue**: Out of memory
- **Solution**: Reduce `--batch-size` or increase `--resolution`

**Issue**: Probabilistic analysis too slow
- **Solution**: Reduce `--num-samples` or increase `--resolution`

**Issue**: Model architecture mismatch
- **Solution**: Ensure model architecture in demo script matches training config

## Advanced: Custom Analysis

For custom analyses, you can access the raw data:

```python
# Load saved analysis
import numpy as np
data = np.load("roa_analysis_probabilistic/basin_analysis_data.npz")

# Access components
grid_points = data['grid_points']      # [N, 2] state space grid
endpoints = data['endpoints']          # [N, 2] mean predicted endpoints
basin_labels = data['basin_labels']    # [N] attractor labels (0,1,2,3,4)
entropy = data['entropy']              # [N] uncertainty entropy
pmax = data['pmax']                    # [N] max attractor probability
endpoint_std = data['endpoint_std']    # [N] prediction std magnitude

# Reshape to grid
grid_shape = tuple(data['resolution'])
basin_grid = basin_labels.reshape(grid_shape)
entropy_grid = entropy.reshape(grid_shape)
```

## Tips

1. **Start with deterministic analysis** to quickly visualize basins
2. **Use probabilistic for final results** when you need uncertainty quantification
3. **Higher num_samples = better uncertainty estimates** but slower
4. **Fine resolution (0.05)** for publication-quality figures
5. **Coarse resolution (0.2)** for quick iterations during development
6. **Check uncertainty maps** to identify regions where model is unreliable
7. **Standard deviation maps** reveal where latent diversity is highest

## Integration with Existing Code

The inference wrapper is compatible with the existing `AttractorBasinAnalyzer`, so all existing evaluation tools work seamlessly:

```python
from src.evaluation.evaluator import FlowMatchingEvaluator

evaluator = FlowMatchingEvaluator()
# Works with the inference wrapper!
results = evaluator.evaluate_on_dataloader(inferencer, test_loader, data_module)
```
