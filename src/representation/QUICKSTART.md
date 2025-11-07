# Trajectory MAE Quick Start

## 30-Second Start

```bash
# 1. Train a model
python src/representation/train.py

# 2. View training progress
tensorboard --logdir outputs/trajectory_mae_cartpole

# 3. Use trained model
python src/representation/demo.py
```

## What You Get

After training, you'll have a model that converts any CartPole trajectory into a **256-dimensional representation vector**.

## Basic Usage

### Python API

```python
from src.representation.inference import TrajectoryMAEInference
import numpy as np

# Load model
model = TrajectoryMAEInference("outputs/trajectory_mae_cartpole/version_0/checkpoints/last.ckpt")

# Load trajectory
traj = np.loadtxt("sequence_100.txt", delimiter=',')  # (seq_len, 4)

# Get representation
state_bounds = {
    'min': np.array([-2.4, -130.0, -10.0, -10.0]),
    'max': np.array([2.4, 130.0, 10.0, 10.0])
}

embedding = model.extract_representation(
    states=traj,
    aggregate='mean',
    normalize=False,
    state_bounds=state_bounds
)

print(f"Trajectory embedding: {embedding.shape}")  # (256,)
```

## Training Variants

```bash
# Fast training (50% mask, easier task)
python src/representation/train.py model.mask_ratio=0.5

# Block masking (better for long sequences)
python src/representation/train.py model.mask_strategy=block

# Bigger model (better representations)
python src/representation/train.py model.embed_dim=512 model.encoder_depth=8

# Smaller model (faster, less memory)
python src/representation/train.py model.embed_dim=128 model.encoder_depth=4
```

## Common Workflows

### 1. Classification Task

```python
# Extract embeddings for all trajectories
embeddings = model.extract_representations_from_files(
    trajectory_files=my_files,
    state_bounds=state_bounds
)

# Train classifier
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression().fit(embeddings, labels)

# Predict
predictions = clf.predict(embeddings)
```

### 2. Trajectory Similarity

```python
# Get embeddings
emb1 = model.extract_representation(traj1, ...)
emb2 = model.extract_representation(traj2, ...)

# Compute similarity
from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity([emb1], [emb2])[0, 0]
```

### 3. Clustering

```python
# Get embeddings for dataset
embeddings = model.extract_representations_from_files(all_files, ...)

# Cluster
from sklearn.cluster import KMeans
clusters = KMeans(n_clusters=5).fit_predict(embeddings)
```

## Files You Need

**For training:**
- `src/representation/train.py` - Training script
- `configs/train_trajectory_mae.yaml` - Configuration

**For inference:**
- `src/representation/inference.py` - Inference wrapper
- Trained checkpoint (`.ckpt` file)

**For examples:**
- `src/representation/demo.py` - Complete examples
- `src/representation/README.md` - Full documentation

## Troubleshooting

**Error: "No module named 'src.representation'"**
```bash
# Make sure you're in the project root
cd /common/home/dm1487/robotics_research/tripods/olympics-classifier

# Install in development mode
pip install -e .
```

**Training is slow**
```bash
# Use smaller batch or model
python src/representation/train.py data.batch_size=32 model.embed_dim=128
```

**Out of memory**
```bash
# Reduce batch size
python src/representation/train.py data.batch_size=16
```

## Next Steps

1. **Read full docs**: `src/representation/README.md`
2. **Experiment**: Try different mask ratios, model sizes
3. **Evaluate**: Use embeddings for your downstream tasks
4. **Extend**: Adapt for other robot systems

That's it! You're ready to learn trajectory representations. 🚀
