# Trajectory Representation Learning with Masked Autoencoders

This module implements a **Masked Autoencoder (MAE)** for learning rich representations from robotics trajectory data. The approach is inspired by BERT (for text) and MAE (for images), but adapted for time-series trajectory data with manifold-aware state spaces.

## Overview

### What is it?

A self-supervised learning system that:
1. **Masks** random portions of a trajectory (e.g., 75% of time steps)
2. **Encodes** the visible portions with a Transformer encoder
3. **Reconstructs** the masked portions with a Transformer decoder
4. **Learns** rich trajectory representations in the encoder

The learned encoder can then be used for downstream tasks like:
- Trajectory classification (success/failure prediction)
- Trajectory clustering and analysis
- Transfer learning to new tasks
- Distillation into smaller models

### Key Features

- ✅ **Manifold-Aware**: Properly handles circular coordinates (angles) using sin/cos encoding
- ✅ **Variable-Length Sequences**: Handles trajectories of different lengths
- ✅ **Flexible Masking**: Supports random masking (BERT-style) and block masking (MAE-style)
- ✅ **Transformer Architecture**: Modern encoder-decoder with multi-head attention
- ✅ **Efficient Training**: Mixed-precision training, gradient clipping, cosine learning rate schedule
- ✅ **Easy Inference**: Extract representations with a single function call

## Architecture

```
Input Trajectory (T × 4)
    ↓
Manifold Embedding: (x, θ, ẋ, θ̇) → (x, sin θ, cos θ, ẋ, θ̇)
    ↓
Apply Masking (75% masked)
    ↓
Positional Encoding (time step information)
    ↓
Transformer Encoder (6 layers, 256 dim, 8 heads)
    ↓  [Learned Representation]
    ↓
Transformer Decoder (4 layers)
    ↓
Reconstruct Masked Positions
    ↓
Loss: MSE on (x, sin θ, cos θ, ẋ, θ̇)
```

### Model Specifications

**Default Configuration:**
- **Embed Dim**: 256
- **Encoder Layers**: 6
- **Decoder Layers**: 4
- **Attention Heads**: 8
- **MLP Ratio**: 4.0
- **Dropout**: 0.1
- **Total Parameters**: ~8-10M

**Masking:**
- **Mask Ratio**: 75% (configurable 15-95%)
- **Strategies**: Random or Block masking
- **Only masks real trajectory points** (respects padding)

## Installation

No additional dependencies needed beyond the main project requirements:
```bash
# Activate environment
conda activate /common/users/dm1487/envs/arcmg

# Already installed: torch, pytorch-lightning, hydra-core, numpy
```

## Quick Start

### 1. Training

Train a Trajectory MAE on CartPole DM Control data:

```bash
# Basic training
python src/representation/train.py

# Custom masking ratio
python src/representation/train.py model.mask_ratio=0.5

# Block masking strategy
python src/representation/train.py model.mask_strategy=block model.block_size=10

# Adjust model size
python src/representation/train.py \
    model.embed_dim=512 \
    model.encoder_depth=8 \
    model.decoder_depth=6

# Train on different GPU
python src/representation/train.py device=gpu1
```

**Training typically takes:**
- ~2-3 hours for 200 epochs on 1 GPU
- Checkpoints saved to `outputs/trajectory_mae_cartpole/`

### 2. Extracting Representations

```python
from src.representation.inference import TrajectoryMAEInference
import numpy as np

# Load trained model
inferencer = TrajectoryMAEInference(
    checkpoint_path="outputs/trajectory_mae_cartpole/version_0/checkpoints/last.ckpt",
    device="cuda"
)

# Load a trajectory
traj = np.loadtxt("path/to/sequence_100.txt", delimiter=',')
# Shape: (seq_len, 4) - columns: x, theta, x_dot, theta_dot

# Define normalization bounds (from dataset)
state_bounds = {
    'min': np.array([-2.4, -130.0, -10.0, -10.0]),
    'max': np.array([2.4, 130.0, 10.0, 10.0])
}

# Extract representation
representation = inferencer.extract_representation(
    states=traj,
    aggregate='mean',  # 'mean', 'max', or 'last'
    normalize=False,
    state_bounds=state_bounds
)

# representation.shape: (256,) - the learned embedding
```

### 3. Batch Processing

```python
from pathlib import Path

# Get trajectory files
traj_dir = Path("/path/to/trajectories")
trajectory_files = sorted(traj_dir.glob("sequence_*.txt"))

# Extract representations for all trajectories
representations = inferencer.extract_representations_from_files(
    trajectory_files,
    aggregate='mean',
    state_bounds=state_bounds,
    batch_size=32,
    show_progress=True
)

# representations.shape: (num_trajectories, 256)
```

### 4. Downstream Classification

```python
# Extract representations
X = representations  # (N, 256)
y = labels           # (N,) - success/failure labels

# Train a classifier
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
accuracy = (y_pred == y_test).mean()
```

## Configuration

Main config file: `configs/train_trajectory_mae.yaml`

### Key Parameters

**Data:**
```yaml
data:
  data_dir: /path/to/cartpole_dmcontrol
  batch_size: 64
  max_seq_len: 1001
  normalize: true
  state_bounds:
    min: [-2.4, -130.0, -10.0, -10.0]
    max: [2.4, 130.0, 10.0, 10.0]
```

**Model Architecture:**
```yaml
model:
  embed_dim: 256
  encoder_depth: 6
  decoder_depth: 4
  num_heads: 8
  dropout: 0.1
```

**Masking:**
```yaml
model:
  mask_ratio: 0.75        # Mask 75% of tokens
  mask_strategy: "random" # or "block"
  block_size: null        # Adaptive if null
```

**Training:**
```yaml
model:
  learning_rate: 1.0e-4
  weight_decay: 0.05
  warmup_epochs: 10
  max_epochs: 200

trainer:
  precision: 16-mixed  # Mixed precision
  gradient_clip_val: 1.0
```

## File Structure

```
src/representation/
├── __init__.py              # Module init
├── masking.py               # Random & block masking utilities
├── trajectory_data.py       # Dataset & data module
├── trajectory_mae.py        # Transformer MAE model
├── train_module.py          # PyTorch Lightning training module
├── train.py                 # Training script
├── inference.py             # Inference wrapper
├── demo.py                  # Demo scripts
└── README.md                # This file

configs/
└── train_trajectory_mae.yaml  # Main config file
```

## Advanced Usage

### Custom Aggregation

```python
# Mean pooling (default)
repr_mean = inferencer.extract_representation(traj, aggregate='mean')

# Max pooling (captures extremes)
repr_max = inferencer.extract_representation(traj, aggregate='max')

# Last time step (final state encoding)
repr_last = inferencer.extract_representation(traj, aggregate='last')
```

### Reconstruction Visualization

```python
# Reconstruct a masked trajectory
original, masked_idx, reconstructed = inferencer.reconstruct_trajectory(
    states=traj,
    mask_ratio=0.75,
    normalize=False,
    state_bounds=state_bounds
)

# original: (seq_len, 4) - original states
# masked_idx: (num_masked,) - which positions were masked
# reconstructed: (num_masked, 5) - predicted (x, sin θ, cos θ, ẋ, θ̇)
```

### Fine-tuning for Downstream Tasks

```python
# Load pre-trained model
model = TrajectoryMAELightningModule.load_from_checkpoint(checkpoint_path)

# Freeze encoder
for param in model.model.encoder.parameters():
    param.requires_grad = False

# Add task-specific head
class ClassifierHead(nn.Module):
    def __init__(self, embed_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, repr):
        return self.fc(repr)

# Fine-tune on your task
classifier = ClassifierHead(embed_dim=256, num_classes=2)
# ... training loop ...
```

## Monitoring Training

View training progress with TensorBoard:

```bash
tensorboard --logdir outputs/trajectory_mae_cartpole
```

**Key Metrics:**
- `train/loss`: Training reconstruction loss
- `val/loss`: Validation reconstruction loss
- `val/mae_*`: Per-dimension MAE (x, sin θ, cos θ, ẋ, θ̇)
- Learning rate schedule

## Tips & Best Practices

### Masking Ratio
- **15-30%**: Easier task, faster convergence, less expressive representations
- **50-60%**: Balanced difficulty
- **75-90%**: Harder task, better representations (like MAE paper)

### Masking Strategy
- **Random**: Better for short trajectories, uniform coverage
- **Block**: Better for long trajectories, forces temporal reasoning

### Aggregation Method
- **Mean**: Best for general trajectory embedding
- **Max**: Good for detecting extreme events
- **Last**: Good for endpoint-focused tasks

### Model Size
- **Small** (embed_dim=128, depth=4): Fast, good for small datasets
- **Medium** (embed_dim=256, depth=6): Default, balanced
- **Large** (embed_dim=512, depth=8): Better representations, needs more data

## Troubleshooting

**Issue: Training loss not decreasing**
- Check learning rate (try 1e-4 to 1e-3)
- Verify data normalization
- Ensure warmup_epochs > 0
- Try lower mask_ratio (e.g., 0.5)

**Issue: Validation loss much higher than training**
- Normal for MAE (different random masks each time)
- Add dropout or weight decay
- Check for data leakage

**Issue: Out of memory**
- Reduce batch_size
- Reduce embed_dim or encoder_depth
- Use gradient accumulation

**Issue: Representations not useful for downstream task**
- Try different aggregation methods
- Train longer (more epochs)
- Use higher mask_ratio
- Fine-tune instead of frozen features

## Citation

If you use this code, please cite:

```bibtex
@software{trajectory_mae_2024,
  title={Trajectory Masked Autoencoder for Robotics},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/yourrepo}
}
```

## Future Extensions

Possible improvements:
- [ ] Support for other robot systems (humanoid, mountain car, etc.)
- [ ] Contrastive learning objectives
- [ ] Temporal consistency losses
- [ ] Multi-task learning (reconstruction + classification)
- [ ] Distillation into smaller models
- [ ] Online/streaming inference

## Contact

For questions or issues, please open a GitHub issue or contact [your email].
