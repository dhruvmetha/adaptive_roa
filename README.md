# EndpointCFM: Conditional Flow Matching for Endpoint Prediction

A Python package for training and using conditional flow matching models to predict final states in dynamical systems.

## Features

- **Easy Training**: Train models from trajectory data with a single function call
- **Parallel Inference**: Generate multiple endpoint predictions efficiently  
- **Flexible Architecture**: Configurable UNet architecture with FiLM conditioning
- **Multiple Integration Methods**: Support for Euler and RK4 ODE integration

## Installation

```bash
# Install in development mode
pip install -e .

# Or install from PyPI (when published)
pip install endpoint-cfm
```

## Quick Start

### 1. Training a Model

```python
from endpoint_cfm import EndpointCFM

# Initialize orchestrator
cfm = EndpointCFM()

# Train model from trajectory files
checkpoint = cfm.train(
    trajectory_files=[
        "path/to/trajectory1.txt",
        "path/to/trajectory2.txt", 
        # ... more trajectory files
    ],
    output_dir="./my_model",
    max_epochs=100,
    batch_size=1024,
    device="cuda"  # or "cpu"
)

print(f"Model trained! Checkpoint: {checkpoint}")
```

### 2. Using a Trained Model

```python
import numpy as np
from endpoint_cfm import EndpointCFM

# Load trained model
cfm = EndpointCFM()
cfm.load_model("./my_model/checkpoints/best_model.ckpt")

# Prepare start states (θ, θ̇ format)
start_states = np.array([
    [0.1, 0.5],   # θ=0.1, θ̇=0.5
    [1.2, -0.3],  # θ=1.2, θ̇=-0.3
    # ... more start states
])

# Get final states (single sample per start state)
final_states = cfm.get_final_states(
    start_states=start_states,
    num_samples=1
)
print(f"Final states shape: {final_states.shape}")  # [N, 1, 2]

# Get multiple samples per start state for uncertainty quantification
final_states_multi = cfm.get_final_states(
    start_states=start_states,
    num_samples=10  # 10 samples per start state
)
print(f"Multi-sample shape: {final_states_multi.shape}")  # [N, 10, 2]
```

## API Reference

### EndpointCFM Class

#### `train(trajectory_files, output_dir, **kwargs)`

Train a conditional flow matching model.

**Parameters:**
- `trajectory_files` (List[str]): Paths to trajectory data files
- `output_dir` (str): Directory to save model and dataset
- `max_epochs` (int): Maximum training epochs (default: 100)
- `batch_size` (int): Training batch size (default: 1024)
- `learning_rate` (float): Learning rate (default: 1e-3)
- `device` (str): Training device ("auto", "cpu", "cuda", default: "auto")
- `num_workers` (int): Data loading workers (default: 4)
- `validation_split` (float): Validation fraction (default: 0.1)
- `noise_distribution` (str): "uniform" or "gaussian" (default: "uniform")
- `hidden_dims` (List[int]): UNet hidden dimensions (default: [64, 128, 256])
- `time_emb_dim` (int): Time embedding dimension (default: 128)

**Returns:**
- `str`: Path to best checkpoint

#### `load_model(checkpoint_path, device=None)`

Load a trained model for inference.

**Parameters:**
- `checkpoint_path` (str): Path to model checkpoint
- `device` (str, optional): Device to load on (auto-detected if None)

#### `get_final_states(start_states, num_samples=1, **kwargs)`

Predict final states from start states.

**Parameters:**
- `start_states` (np.ndarray): Start states [N, 2] in (θ, θ̇) format
- `num_samples` (int): Samples per start state (default: 1)
- `num_steps` (int): ODE integration steps (default: 100)
- `method` (str): Integration method ("euler" or "rk4", default: "rk4")

**Returns:**
- `np.ndarray`: Final states [N, num_samples, 2]

## Command Line Interface

### Training

```bash
endpoint-cfm-train \
    --trajectory-files traj1.txt traj2.txt \
    --output-dir ./my_model \
    --max-epochs 100 \
    --batch-size 1024 \
    --device cuda
```

### Inference

```bash
endpoint-cfm-infer \
    --checkpoint ./my_model/checkpoints/best.ckpt \
    --start-states start_states.npy \
    --output final_states.npy \
    --num-samples 5
```

## Data Format

### Trajectory Files

Each trajectory file should contain numerical data with shape `[time_steps, features]`. The first two columns are interpreted as (θ, θ̇). Example:

```
# time_step  θ      θ̇     (other features...)
0.0         0.1    0.5    ...
0.01        0.105  0.48   ...
0.02        0.11   0.46   ...
...
```

### Start States

Start states should be provided as numpy arrays with shape `[N, 2]` where each row is `[θ, θ̇]`.

## Requirements

- Python >= 3.8
- PyTorch >= 1.10.0
- PyTorch Lightning >= 2.0.0
- NumPy >= 1.21.0
- See `setup.py` for full requirements

## License

MIT License - see LICENSE file for details.