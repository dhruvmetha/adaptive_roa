# Setup Guide 


## 1. Clone the Repository

```bash
cd /common/users/$USER
git clone -b adaptive_main https://github.com/dhruvmetha/adaptive_roa.git adaptive_cartpole
cd adaptive_cartpole
```

---

## 2. Create Conda Environment

### On Westeros/GPU machines:

```bash
# Use system conda
source /koko/system/anaconda/etc/profile.d/conda.sh

# Create environment from yaml (if available) or manually
conda create -p /common/users/$USER/miniforge3/envs/adaptive_roa python=3.10 -y
conda activate /common/users/$USER/miniforge3/envs/adaptive_roa

# Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install pytorch-lightning hydra-core omegaconf
pip install numpy scipy matplotlib tqdm
pip install flow-matching  # Facebook's flow matching library

# Install the project in development mode
pip install -e .
```

### On iLab machines:

```bash
# iLab uses a different conda path
source /common/users/$USER/miniconda3/etc/profile.d/conda.sh
# Or if using miniforge:
source /common/users/$USER/miniforge3/etc/profile.d/conda.sh

conda activate /common/users/$USER/miniforge3/envs/adaptive_roa
```

---

## 3. Set Up Environment Variables

Create a `.env` file in the project root:

```bash
cd /common/users/$USER/adaptive_cartpole
cat > .env << EOF
NET_ID=$USER
EOF
```

This file is used by Hydra configs to resolve paths like `${net_id}`.

---

## 4. Verify Data Access

The trajectory data is stored in a shared location:

```bash
# Check that you can access the shared data
ls /common/users/shared/pracsys/genMoPlan/data_trajectories/

# You should see directories like:
# - cartpole_pybullet/
# - mountain_car_power_0p0008/
# - pendulum_lqr_50k/
```

Each system directory contains:
- `trajectories/` - Individual trajectory files (sequence_*.txt)
- `train_test_splits/` - Shuffled indices and labels for training
- `eval_states.txt` - Full dataset with start states, end states, and labels for ROA evaluation

---

## 5. Verify GPU Access

```bash
# Check available GPUs
nvidia-smi

# Request a GPU interactively (if using SLURM)
srun -G 1 --pty /bin/bash
```

---

## 6. Run a Test

### Quick import test:

```bash
cd /common/users/$USER/adaptive_cartpole
export PYTHONPATH=$(pwd):$PYTHONPATH

python -c "
from src.adaptive.data_source import TrajectoryDataSource, TrajectoryDataSourceConfig
from src.systems.cartpole import CartPoleSystem
print('All imports successful!')
"
```

### Run adaptive sampling (CartPole):

```bash
# Set up environment
source /koko/system/anaconda/etc/profile.d/conda.sh
conda activate /common/users/$USER/miniforge3/envs/adaptive_roa
cd /common/users/$USER/adaptive_cartpole
export PYTHONPATH=$(pwd):$PYTHONPATH

# Run with default config
CUDA_VISIBLE_DEVICES=0 python scripts/run_adaptive.py system=cartpole_pybullet

# Run with custom parameters
CUDA_VISIBLE_DEVICES=0 python scripts/run_adaptive.py system=cartpole_pybullet \
    initial_train_size=1000 \
    n_epochs=1 \
    trainer.max_epochs=500 \
    batch_size=1024
```

### Run adaptive sampling (Mountain Car):

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_adaptive_mountain_car.py \
    initial_train_size=1000 \
    n_epochs=1 \
    trainer.max_epochs=500 \
    batch_size=512
```

### Run adaptive sampling (Pendulum):

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_adaptive.py system=pendulum \
    initial_train_size=500 \
    n_epochs=1 \
    trainer.max_epochs=500
```

### Run unified v2 smoke tests (2 adaptive epochs, 1 FM epoch each):

```bash
BASE=/tmp/adaptive_v2_smoke_e2
mkdir -p $BASE/{logs,out,hydra}

for SYS in pendulum cartpole_pybullet quadrotor2d quadrotor3d; do
  INIT=10
  if [ "$SYS" = "quadrotor3d" ]; then INIT=20; fi

  timeout 180 python scripts/run_adaptive.py \
    system=$SYS \
    +adaptive_v2.smoke_mode=true \
    n_epochs=2 \
    trainer.max_epochs=1 \
    +trainer.limit_train_batches=1 \
    +trainer.limit_val_batches=1 \
    sampling_mode=ranked \
    n_ranked_candidates=10 \
    batch_size_sampling=5 \
    max_samples_per_epoch=20 \
    initial_train_size=$INIT \
    output_dir=$BASE/out/$SYS \
    hydra.run.dir=$BASE/hydra/$SYS \
    > $BASE/logs/$SYS.log 2>&1

  echo "$SYS EXIT_CODE=$?"
done
```

Notes:
- `+adaptive_v2.smoke_mode=true` skips heavy evaluation stages and is intended for pipeline validation.
- `trainer.max_epochs=1` enforces one FM training epoch per adaptive epoch.
- If CUDA is requested but unavailable, v2 falls back to CPU.

---

## 7. Running Multiple Experiments

### Using different shuffle variants (for cross-validation):

```bash
# Run with shuffle variant 0 (default)
python scripts/run_adaptive.py system=cartpole_pybullet shuffle_variant=0

# Run with shuffle variant 3
python scripts/run_adaptive.py system=cartpole_pybullet shuffle_variant=3

# Run all 10 variants
for i in {0..9}; do
    CUDA_VISIBLE_DEVICES=$((i % 4)) python scripts/run_adaptive.py system=cartpole_pybullet \
        shuffle_variant=$i &
done
wait
```

### Running in background with logging:

```bash
mkdir -p logs

CUDA_VISIBLE_DEVICES=0 python scripts/run_adaptive.py system=cartpole_pybullet \
    > logs/cartpole_run.log 2>&1 &

# Monitor progress
tail -f logs/cartpole_run.log
```

---

## 8. Key Configuration Parameters

Edit `configs/adaptive_v2/system/cartpole_pybullet.yaml` (or pass as CLI args):

| Parameter | Description | Default |
|-----------|-------------|---------|
| `initial_train_size` | Initial training trajectories | 300 |
| `n_epochs` | Number of adaptive sampling epochs | 8 |
| `warm_start` | Continue training from previous checkpoint | false |
| `samples_per_epoch` | New samples per epoch (D1 + D2) | 100 |
| `d2_ratio` | Fraction for uncertainty-filtered sampling | 0.5 |
| `trainer.max_epochs` | Training epochs per round | 1000 |
| `batch_size` | Training batch size | 1024 |
| `optimizer.lr` | Learning rate | 1e-3 |
| `conformal.optimize_mode` | "lambda" or "delta" | delta |
| `sampling_mode` | "ranked", "conformal", or "direct" | ranked |

---

## 9. Output Structure

Runs output to `outputs/<system>/<timestamp>/`:

```
outputs/adaptive_cartpole_pybullet/2026-01-18_16-45-00/
├── .hydra/                    # Hydra config snapshots
├── datasets/                  # Built endpoint datasets
│   ├── train_endpoint_dataset.txt
│   ├── val_endpoint_dataset.txt
│   └── test_endpoint_dataset.txt
├── final_results.json         # Run-level summary
└── epoch_000/                 # Per-epoch outputs
    ├── checkpoints/           # Model checkpoints
    ├── full_roa_evaluation.json
    ├── artifacts_v2.json      # Canonical v2 epoch artifact (includes legacy_epoch_metrics + conformal_state)
    ├── results.json           # Legacy flat epoch metrics (enabled by default)
    ├── conformal_state.json   # Legacy conformal predictor state (enabled by default)
    └── version_0/             # Lightning logs
```

---

## 10. Troubleshooting

### ModuleNotFoundError: No module named 'src'

```bash
export PYTHONPATH=/common/users/$USER/adaptive_cartpole:$PYTHONPATH
```

### CUDA out of memory

Reduce batch size:
```bash
python scripts/run_adaptive.py system=cartpole_pybullet batch_size=256
```

### Hydra override errors

Use `+` prefix for new keys, no prefix for existing keys:
```bash
# Override existing key
python script.py batch_size=512

# Add new key (not in config)
python script.py +new_param=value
```

### conda: command not found

```bash
# On westeros
source /koko/system/anaconda/etc/profile.d/conda.sh

# On ilab
source /common/users/$USER/miniconda3/etc/profile.d/conda.sh
```

---

## 11. Quick Reference Commands

```bash
# Activate environment (westeros)
source /koko/system/anaconda/etc/profile.d/conda.sh
conda activate /common/users/$USER/miniforge3/envs/adaptive_roa
cd /common/users/$USER/adaptive_cartpole
export PYTHONPATH=$(pwd):$PYTHONPATH

# Check GPU status
nvidia-smi

# Run CartPole
CUDA_VISIBLE_DEVICES=4 python scripts/run_adaptive.py system=cartpole_pybullet

# Run Mountain Car
CUDA_VISIBLE_DEVICES=5 python scripts/run_adaptive_mountain_car.py

# Run Pendulum
CUDA_VISIBLE_DEVICES=6 python scripts/run_adaptive.py system=pendulum

# Monitor logs
tail -f logs/*.log

# Check running jobs
ps aux | grep python
```

---

## 12. Legacy Mountain Car

`mountain_car` remains legacy in this phase and is intentionally separate from unified v2.

Use:

```bash
python scripts/run_adaptive_mountain_car.py
```

Legacy v1 details for the four unified systems are archived in:
`docs/archive/LEGACY_ADAPTIVE_V1.md`

---
