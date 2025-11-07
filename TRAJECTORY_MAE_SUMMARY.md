# Trajectory Masked Autoencoder (MAE) Implementation Summary

## What Was Built

I've implemented a complete **Masked Autoencoder system for learning trajectory representations** from your CartPole DM Control data. This is a self-supervised learning approach inspired by BERT (for text) and MAE (for images), adapted for robotics trajectories.

## Key Components

### 1. **Masking Utilities** (`src/representation/masking.py`)
- **Random masking**: BERT-style random token masking
- **Block masking**: MAE-style contiguous block masking
- **Configurable mask ratio**: 15% to 95% (default: 75%)
- Respects padding masks for variable-length sequences

### 2. **Data Module** (`src/representation/trajectory_data.py`)
- Loads CartPole DM Control trajectories from files
- Handles variable-length sequences (up to 1001 time steps)
- Automatic train/val/test split (80/10/10)
- Efficient batching with padding
- State normalization using dataset bounds

### 3. **Transformer MAE Model** (`src/representation/trajectory_mae.py`)
- **Manifold-aware embedding**: Converts (x, θ, ẋ, θ̇) → (x, sin θ, cos θ, ẋ, θ̇)
- **Sinusoidal positional encoding**: Encodes time step information
- **Transformer Encoder** (6 layers, 256 dim, 8 heads): Processes visible tokens
- **Transformer Decoder** (4 layers): Reconstructs masked tokens
- **~8-10M parameters**: Efficient yet powerful
- Special `get_encoder_representation()` method for extracting trajectory embeddings

### 4. **Training Module** (`src/representation/train_module.py`)
- PyTorch Lightning module for streamlined training
- **MSE loss** on manifold features (x, sin θ, cos θ, ẋ, θ̇)
- **AdamW optimizer** with cosine annealing + warmup
- Per-dimension MAE metrics logged during validation
- Mixed-precision training (16-bit) for speed

### 5. **Configuration** (`configs/train_trajectory_mae.yaml`)
- Hydra-based configuration system
- All hyperparameters configurable via CLI
- Integrates with existing codebase structure

### 6. **Training Script** (`src/representation/train.py`)
- Simple entry point: `python src/representation/train.py`
- Override any config: `python src/representation/train.py model.mask_ratio=0.5`
- Automatic checkpointing, early stopping, TensorBoard logging

### 7. **Inference Module** (`src/representation/inference.py`)
- Load trained checkpoints
- Extract representations with single function call
- Batch processing for multiple trajectories
- Multiple aggregation methods (mean, max, last)
- Trajectory reconstruction for debugging

### 8. **Demo Script** (`src/representation/demo.py`)
- Complete examples of:
  - Loading trained models
  - Extracting representations
  - Batch processing
  - Downstream classification tasks

## How It Works

```
┌─────────────────────────────────────────────────────────────┐
│  Input: CartPole Trajectory (T × 4)                         │
│  Format: (x, theta, x_dot, theta_dot) at each time step    │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  Step 1: Manifold Embedding                                 │
│  (x, θ, ẋ, θ̇) → (x, sin θ, cos θ, ẋ, θ̇)                   │
│  Handles circular angle coordinate properly                 │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  Step 2: Random Masking                                     │
│  Randomly mask 75% of time steps                            │
│  Only encoder sees visible 25%                              │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  Step 3: Transformer Encoder                                │
│  6-layer transformer processes visible tokens               │
│  Learns rich trajectory representation                      │
│  Output: 256-dim embedding per time step                    │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  Step 4: Transformer Decoder                                │
│  4-layer transformer reconstructs masked tokens             │
│  Uses mask tokens + encoder memory                          │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  Step 5: Reconstruction Loss                                │
│  MSE between predicted and true manifold features           │
│  Forces encoder to learn meaningful representations         │
└─────────────────────────────────────────────────────────────┘
```

## Usage Examples

### Training
```bash
# Basic training (uses all default settings)
python src/representation/train.py

# Adjust mask ratio
python src/representation/train.py model.mask_ratio=0.5

# Use block masking
python src/representation/train.py model.mask_strategy=block

# Larger model
python src/representation/train.py model.embed_dim=512 model.encoder_depth=8

# Different GPU
python src/representation/train.py device=gpu1
```

### Inference
```python
from src.representation.inference import TrajectoryMAEInference
import numpy as np

# Load model
inferencer = TrajectoryMAEInference(
    checkpoint_path="outputs/trajectory_mae_cartpole/version_0/checkpoints/last.ckpt"
)

# Load trajectory
traj = np.loadtxt("sequence_100.txt", delimiter=',')

# Extract 256-dim representation
state_bounds = {
    'min': np.array([-2.4, -130.0, -10.0, -10.0]),
    'max': np.array([2.4, 130.0, 10.0, 10.0])
}

representation = inferencer.extract_representation(
    states=traj,
    aggregate='mean',
    normalize=False,
    state_bounds=state_bounds
)

# representation.shape: (256,)
```

### Downstream Task (Classification)
```python
# Extract representations for all trajectories
representations = inferencer.extract_representations_from_files(
    trajectory_files=list_of_files,
    state_bounds=state_bounds,
    batch_size=32
)

# Train classifier
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf.fit(representations, labels)
```

## What You Can Do With This

### 1. **Representation Learning**
- Get 256-dim compact representation of entire trajectory
- Use for visualization (t-SNE, UMAP)
- Cluster similar trajectories
- Detect anomalies

### 2. **Downstream Tasks**
- **Classification**: Success/failure prediction
- **Regression**: Predict final state or reward
- **Retrieval**: Find similar trajectories
- **Generation**: Condition other models on learned representations

### 3. **Distillation**
- Pre-train large MAE (256-512 dim, 6-8 layers)
- Extract representations for all data
- Train small MLP/transformer to mimic representations
- Deploy efficient small model

### 4. **Transfer Learning**
- Pre-train on CartPole
- Fine-tune encoder on other tasks
- Use as feature extractor

## File Locations

```
src/representation/
├── __init__.py              # Module initialization
├── masking.py               # Masking utilities (~230 lines)
├── trajectory_data.py       # Data loading (~350 lines)
├── trajectory_mae.py        # MAE model (~400 lines)
├── train_module.py          # Training module (~320 lines)
├── train.py                 # Training script (~80 lines)
├── inference.py             # Inference wrapper (~280 lines)
├── demo.py                  # Demo scripts (~180 lines)
└── README.md                # Detailed documentation

configs/
└── train_trajectory_mae.yaml  # Configuration (~80 lines)
```

**Total new code**: ~1,920 lines

## Next Steps

### Immediate
1. **Train your first model**:
   ```bash
   python src/representation/train.py
   ```

2. **Monitor training**:
   ```bash
   tensorboard --logdir outputs/trajectory_mae_cartpole
   ```

3. **Extract representations**:
   ```bash
   python src/representation/demo.py
   ```

### Future Work

1. **Experiment with hyperparameters**:
   - Try different mask ratios (0.5, 0.75, 0.9)
   - Test block vs random masking
   - Adjust model size

2. **Downstream tasks**:
   - Train classifiers on learned representations
   - Compare to end-to-end models
   - Fine-tune for specific tasks

3. **Distillation**:
   - Train small MLP to match MAE representations
   - Deploy efficient model for real-time use

4. **Extend to other systems**:
   - Adapt for Pendulum, Humanoid, Mountain Car
   - Multi-task learning across systems

## Design Decisions

### Why Transformer?
- **Attention mechanism**: Learns which time steps are important
- **Permutation equivariant**: Can handle variable-length sequences
- **Scalable**: Works from simple to complex trajectories
- **Pre-training friendly**: Transfer learning potential

### Why Manifold Embedding?
- **Circular coordinates**: θ=0 and θ=2π are the same point
- **Better representations**: sin/cos preserves angular structure
- **Smooth gradients**: No discontinuities at ±π

### Why 75% Masking?
- **Following MAE paper**: 75% masking forces better representations
- **Prevents shortcuts**: Can't just copy neighbors
- **Configurable**: Easy to adjust based on your data

### Why Separate Encoder/Decoder?
- **Asymmetric design**: Encoder does heavy lifting, decoder is lightweight
- **Efficient**: Encoder only processes 25% of tokens
- **Flexible**: Encoder can be used independently for inference

## Technical Details

### Data Format
- **Input**: CSV files with columns `x,theta,x_dot,theta_dot`
- **Theta**: Unwrapped (can exceed ±π), automatically wrapped in model
- **Variable length**: 1-1001 time steps, padded during batching

### Normalization
- States normalized to [0,1] using dataset bounds
- Bounds: x∈[-2.4,2.4], θ∈[-130,130], ẋ∈[-10,10], θ̇∈[-10,10]

### Training
- **Optimizer**: AdamW (lr=1e-4, weight_decay=0.05, betas=(0.9,0.95))
- **Scheduler**: Cosine annealing with 10-epoch linear warmup
- **Batch size**: 64 (adjustable)
- **Precision**: Mixed (16-bit) for speed
- **Typical runtime**: 2-3 hours for 200 epochs on 1 GPU

### Model Architecture
```
Input: (B, T, 4) normalized states
  ↓
Manifold Embedding: (B, T, 5)
  ↓
Masking → Visible: (B, 0.25*T, 5)
  ↓
Linear Projection: (B, 0.25*T, 256)
  ↓
Positional Encoding: (B, 0.25*T, 256)
  ↓
Transformer Encoder: (B, 0.25*T, 256)
  [6 layers, 8 heads, 1024 FFN dim]
  ↓
Decoder Embedding: (B, 0.75*T, 256)
  ↓
Transformer Decoder: (B, 0.75*T, 256)
  [4 layers, 8 heads, 1024 FFN dim]
  ↓
Output Projection: (B, 0.75*T, 5)
  ↓
Loss: MSE with ground truth manifold features
```

## Questions?

See the detailed README at `src/representation/README.md` for:
- Advanced usage examples
- Troubleshooting guide
- Configuration options
- Best practices
- API reference

Enjoy learning trajectory representations! 🚀
