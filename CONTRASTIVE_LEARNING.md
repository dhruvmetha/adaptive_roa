# Contrastive Representation Learning for Dynamical Systems

Self-supervised learning from trajectory temporal structure using contrastive methods.

## Overview

This module implements **trajectory-based contrastive learning** to learn meaningful state representations without labels. The key idea: **temporally nearby states should have similar embeddings**, while distant states should be far apart in embedding space.

### Key Features

- ✅ **Self-Supervised**: No labels required—learns from trajectory temporal structure
- ✅ **System-Agnostic**: Works with any `DynamicalSystem` (Humanoid, CartPole, Pendulum, etc.)
- ✅ **Flexible Loss**: Supports InfoNCE (NT-Xent) and Triplet loss
- ✅ **Normalized Embeddings**: L2-normalized for stable metric learning
- ✅ **Downstream Integration**: Easy integration with classification and flow matching
- ✅ **Comprehensive Evaluation**: Visualization and diagnostic tools included

---

## Quick Start

### 1. Train Contrastive Encoder (Humanoid)

```bash
# Activate environment
conda activate /common/users/dm1487/envs/arcmg

# Train humanoid encoder
python src/contrastive/train_contrastive.py --config-name=train_humanoid_repr

# Train with custom parameters
python src/contrastive/train_contrastive.py --config-name=train_humanoid_repr \
    model.embedding_dim=256 \
    contrastive.temperature=0.1 \
    batch_size=512 \
    trainer.max_epochs=100
```

**Training Output:**
- Checkpoints: `outputs/humanoid_contrastive_repr/{timestamp}/checkpoints/`
- TensorBoard logs: `outputs/humanoid_contrastive_repr/{timestamp}/`
- Best model saved based on validation nearest-neighbor accuracy

### 2. Evaluate Embeddings

```bash
# Visualize learned embeddings
python src/contrastive/evaluate_embeddings.py \
    --checkpoint outputs/humanoid_contrastive_repr/.../checkpoints/best.ckpt \
    --system humanoid \
    --num_samples 5000 \
    --output_dir evaluation_results/
```

**Evaluation Outputs:**
- `temporal_vs_embedding_distance.png` - Correlation plot
- `embeddings_umap_by_trajectory.png` - UMAP visualization
- `embedding_norms.png` - Norm distribution
- `evaluation_report.txt` - Summary statistics

### 3. Use Pretrained Encoder (Downstream Task)

```python
from src.contrastive.encoder_loader import load_pretrained_encoder
from src.systems.humanoid import HumanoidSystem
import torch.nn as nn

# Load system and encoder
system = HumanoidSystem()
encoder = load_pretrained_encoder(
    "outputs/humanoid_contrastive_repr/.../checkpoints/best.ckpt",
    freeze=True  # Freeze for feature extraction
)

# Create classification head
classifier = nn.Linear(encoder.embedding_dim, 3)  # 3 classes

# Forward pass
raw_states = ...  # [B, 67]
norm_states = system.normalize_state(raw_states)
embeddings = encoder(norm_states)  # [B, embedding_dim]
logits = classifier(embeddings)  # [B, 3]
```

---

## Architecture

### Temporal Triplet Sampling

For each anchor state at timestep `t`:
- **Positive**: Random state within `±k` timesteps (same trajectory)
- **Negative**: State far away (>2k timesteps) OR from different trajectory

```
Trajectory:  [t-k] ... [t-2] [t-1] [t] [t+1] [t+2] ... [t+k]
             |-------- Positives --------^-------- Positives --------|

Far away:    [0] ... [t-2k]                      [t+2k] ... [end]
             |---- Negatives ----|              |---- Negatives ----|
```

### Encoder Architecture

```
Input: Normalized State [B, state_dim]
  ↓
MLP Layers:
  - Linear(state_dim → 512)
  - ReLU + Dropout
  - Linear(512 → 256)
  - ReLU + Dropout
  - Linear(256 → 128)
  - ReLU + Dropout
  - Linear(128 → embedding_dim)
  ↓
L2 Normalization
  ↓
Output: Embedding [B, embedding_dim] (unit norm)
```

### Loss Functions

#### InfoNCE Loss (Default)
```python
# Treats positive as target, all negatives as contrastive samples
loss = -log(exp(sim(anchor, pos) / τ) / Σ exp(sim(anchor, neg_i) / τ))
```
- More stable than triplet loss
- Temperature τ controls hardness (lower = harder)
- Default: `τ = 0.07`

#### Triplet Loss
```python
# Enforces margin between positive and negative distances
loss = max(0, d(anchor, pos) - d(anchor, neg) + margin)
```
- Simpler, more interpretable
- Margin controls separation (higher = more separation)
- Default: `margin = 0.5`

---

## Training Guide

### System-Specific Configs

#### Humanoid (67D state)
```bash
python src/contrastive/train_contrastive.py --config-name=train_humanoid_repr
```
- **Embedding dim**: 128
- **Hidden layers**: [512, 256, 128]
- **Trajectories**: 10,000 (humanoid_get_up)

#### CartPole (4D state)
```bash
python src/contrastive/train_contrastive.py --config-name=train_cartpole_repr
```
- **Embedding dim**: 64
- **Hidden layers**: [256, 128]
- **Trajectories**: 3,000 (cartpole_dmcontrol)

#### Pendulum (2D state)
```bash
python src/contrastive/train_contrastive.py --config-name=train_pendulum_repr
```
- **Embedding dim**: 32
- **Hidden layers**: [128, 64]
- **Trajectories**: 10,000 (pendulum_lqr_50k)

### Key Hyperparameters

Edit config files in `configs/train_{system}_repr.yaml`:

```yaml
# Model architecture
model:
  embedding_dim: 128              # Output embedding dimension
  hidden_channels: [512, 256, 128]  # MLP hidden layers
  dropout: 0.1                    # Regularization

# Contrastive learning
contrastive:
  loss_type: infonce              # "infonce" or "triplet"
  temperature: 0.07               # InfoNCE temperature (lower = harder)
  margin: 0.5                     # Triplet margin

# Data
data:
  temporal_window: 10             # ±k timesteps for positives
  num_negatives: 1                # Negatives per anchor
  max_trajectories_train: 5000    # Dataset size limit
  batch_size: 256                 # Training batch size

# Training
trainer:
  max_epochs: 50                  # Training epochs
  patience: 15                    # Early stopping patience
```

### Monitoring Training

```bash
# Launch TensorBoard
tensorboard --logdir outputs/humanoid_contrastive_repr/

# View metrics:
# - train_loss, val_loss: Contrastive loss
# - val_nn_acc: Nearest-neighbor accuracy (is positive closer than negative?)
# - val_sim_gap: Similarity gap (how much closer are positives?)
# - val_emb_norm: Embedding norm (should be ~1.0)
```

---

## Downstream Integration

### Option 1: Frozen Encoder (Feature Extraction)

Use pretrained encoder as fixed feature extractor:

```python
from src.contrastive.encoder_loader import create_encoder_wrapper
from src.systems.humanoid import HumanoidSystem

# Load encoder with normalization
system = HumanoidSystem()
encoder_wrapper = create_encoder_wrapper(
    "outputs/humanoid_contrastive_repr/.../checkpoints/best.ckpt",
    system=system,
    freeze=True  # Freeze encoder weights
)

# Use in classification
class Classifier(nn.Module):
    def __init__(self, encoder_wrapper, num_classes=3):
        super().__init__()
        self.encoder = encoder_wrapper
        self.head = nn.Linear(encoder_wrapper.encoder.embedding_dim, num_classes)

    def forward(self, raw_states):
        embeddings = self.encoder(raw_states)  # Handles normalization
        return self.head(embeddings)

model = Classifier(encoder_wrapper)
```

### Option 2: Fine-Tuning

Initialize encoder with pretrained weights, then train jointly:

```python
from src.contrastive.encoder_loader import load_pretrained_encoder
from src.systems.humanoid import HumanoidSystem

# Load encoder (trainable)
encoder = load_pretrained_encoder(
    "outputs/humanoid_contrastive_repr/.../checkpoints/best.ckpt",
    freeze=False  # Allow fine-tuning
)

# Create downstream model
class DownstreamModel(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(encoder.embedding_dim, 3)

    def forward(self, x):
        return self.classifier(self.encoder(x))

model = DownstreamModel(encoder)

# Optimizer with different learning rates
import torch.optim as optim
optimizer = optim.Adam([
    {'params': encoder.parameters(), 'lr': 1e-5},      # Low LR for pretrained
    {'params': model.classifier.parameters(), 'lr': 1e-3}  # High LR for new head
])
```

### Option 3: EncoderWrapper (Convenience)

Combines encoder + normalization in single module:

```python
from src.contrastive.encoder_loader import EncoderWrapper, load_pretrained_encoder
from src.systems.humanoid import HumanoidSystem

system = HumanoidSystem()
encoder = load_pretrained_encoder("path/to/checkpoint.ckpt", freeze=True)

wrapper = EncoderWrapper(encoder, system)

# Use with RAW states (normalization handled internally)
raw_states = torch.randn(32, 67)
embeddings = wrapper(raw_states)  # [32, embedding_dim]
```

---

## Evaluation & Diagnostics

### Metrics

The evaluation script (`src/contrastive/evaluate_embeddings.py`) computes:

1. **Temporal-Embedding Correlation**
   - Measures if nearby timesteps → similar embeddings
   - **Good**: Correlation > 0.5
   - **Excellent**: Correlation > 0.7

2. **Nearest-Neighbor Accuracy**
   - For each state, is temporal positive the nearest neighbor?
   - **Good**: Accuracy > 70%
   - **Excellent**: Accuracy > 85%

3. **Embedding Norm Distribution**
   - Should be tightly clustered around 1.0 (L2 normalized)
   - **Good**: Mean ~1.0, Std < 0.05

### Visualization

- **UMAP/t-SNE**: Visualize embedding space structure
  - Should show smooth trajectories
  - Similar trajectories should cluster together

- **Temporal Distance vs. Embedding Distance**:
  - Should show positive correlation
  - Nearby timesteps → small embedding distance

### Running Evaluation

```bash
# Basic evaluation
python src/contrastive/evaluate_embeddings.py \
    --checkpoint outputs/humanoid_contrastive_repr/.../checkpoints/best.ckpt \
    --system humanoid

# Custom parameters
python src/contrastive/evaluate_embeddings.py \
    --checkpoint path/to/checkpoint.ckpt \
    --system cartpole \
    --num_samples 10000 \
    --temporal_window 10 \
    --output_dir my_evaluation/ \
    --use_tsne  # Use t-SNE instead of UMAP
```

---

## Advanced Usage

### Custom Encoder Architecture

Define your own encoder in `configs/model/my_custom_encoder.yaml`:

```yaml
_target_: src.model.contrastive_encoder.ContrastiveEncoder
input_dim: ${system.state_dim}
embedding_dim: 256
hidden_channels: [1024, 512, 256, 128]
dropout: 0.2
normalize_output: true
```

Then use it:
```bash
python src/contrastive/train_contrastive.py \
    --config-name=train_humanoid_repr \
    model=my_custom_encoder
```

### Temporal Encoder (1D CNN)

For capturing local dynamics:

```python
from src.model.contrastive_encoder import TemporalContrastiveEncoder

encoder = TemporalContrastiveEncoder(
    input_dim=67,
    embedding_dim=128,
    conv_channels=[64, 128],
    kernel_size=3
)

# Use with sequences
state_sequence = torch.randn(32, 10, 67)  # [B, seq_len, state_dim]
embeddings = encoder(state_sequence)  # [32, 128]
```

### Hard Negative Mining

Increase `num_negatives` for harder training:

```yaml
# configs/data/humanoid_trajectory_contrastive.yaml
num_negatives: 5  # Sample 5 negatives per anchor
```

Larger batches also provide more in-batch negatives for InfoNCE.

---

## File Structure

```
src/
├── contrastive/
│   ├── __init__.py
│   ├── contrastive_learner.py      # Lightning module (training loop)
│   ├── encoder_loader.py           # Load pretrained encoders
│   ├── train_contrastive.py        # Training script
│   └── evaluate_embeddings.py      # Evaluation and visualization
├── data/
│   └── trajectory_contrastive_data.py  # Trajectory triplet dataset
└── model/
    └── contrastive_encoder.py      # Encoder architectures

configs/
├── train_humanoid_repr.yaml        # Humanoid training config
├── train_cartpole_repr.yaml        # CartPole training config
├── train_pendulum_repr.yaml        # Pendulum training config
├── data/
│   ├── humanoid_trajectory_contrastive.yaml
│   ├── cartpole_trajectory_contrastive.yaml
│   └── pendulum_trajectory_contrastive.yaml
└── model/
    └── contrastive_encoder.yaml    # Encoder model config
```

---

## Troubleshooting

### Issue: Low Nearest-Neighbor Accuracy (<50%)

**Possible causes:**
- Temporal window too large (positives not actually similar)
- Learning rate too high (encoder not converging)
- Insufficient training epochs

**Solutions:**
- Reduce `temporal_window` (try 5 instead of 10)
- Lower learning rate (`base_lr: 1e-4`)
- Train longer (`max_epochs: 100`)

### Issue: Embedding Norms Not ~1.0

**Possible cause:** Model bug (L2 normalization disabled)

**Solution:** Check `model.normalize_output: true` in config

### Issue: Poor Temporal-Embedding Correlation

**Possible causes:**
- Dataset trajectories too noisy/random
- Encoder architecture too small
- Loss not converging

**Solutions:**
- Increase model capacity (`embedding_dim`, `hidden_channels`)
- Try different loss (`infonce` ↔ `triplet`)
- Check training loss curve (should decrease)

### Issue: Out of Memory (OOM)

**Solutions:**
- Reduce `batch_size` (try 128 instead of 256)
- Disable trajectory caching: `cache_trajectories: false`
- Reduce `max_trajectories_train`
- Use smaller model (`embedding_dim`, `hidden_channels`)

---

## Expected Results

### Humanoid (67D → 128D)

After 50 epochs with InfoNCE (τ=0.07):
- **Val Loss**: ~0.5-1.0
- **Nearest-Neighbor Accuracy**: 75-85%
- **Temporal Correlation**: 0.5-0.7
- **Embedding Norm**: 1.0 ± 0.02

### CartPole (4D → 64D)

After 50 epochs:
- **Val Loss**: ~0.3-0.6
- **Nearest-Neighbor Accuracy**: 80-90%
- **Temporal Correlation**: 0.6-0.8
- **Embedding Norm**: 1.0 ± 0.01

### Pendulum (2D → 32D)

After 50 epochs:
- **Val Loss**: ~0.2-0.5
- **Nearest-Neighbor Accuracy**: 85-95%
- **Temporal Correlation**: 0.7-0.9
- **Embedding Norm**: 1.0 ± 0.01

---

## References

1. **SimCLR**: [A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709)
2. **InfoNCE Loss**: [Representation Learning with Contrastive Predictive Coding](https://arxiv.org/abs/1807.03748)
3. **Triplet Loss**: [FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/abs/1503.03832)
4. **Temporal Contrastive Learning**: [Time-Contrastive Networks](https://arxiv.org/abs/1704.06888)

---

## Citation

If you use this contrastive learning module, please cite:

```bibtex
@misc{olympics_contrastive,
  title={Trajectory Contrastive Learning for Dynamical Systems},
  author={AI Olympics Team},
  year={2025},
  howpublished={\url{https://github.com/your-repo/olympics-classifier}}
}
```
