#!/bin/bash
#SBATCH --job-name=local-fm
#SBATCH --output=slurm_logs/local_%x_%j.out
#SBATCH --error=slurm_logs/local_%x_%j.err
#SBATCH --gres=gpu:4500_ada:1
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

# Train trajectory (local) flow matching for RoA estimation
#
# Usage:
#   # Cartpole, non-adaptive (d2=0), 5000 initial trajectories
#   sbatch scripts/train_local.sh system=cartpole_pybullet d2_ratio=0.0 initial_train_size=5000
#
#   # Pendulum, non-adaptive
#   sbatch scripts/train_local.sh system=pendulum d2_ratio=0.0 initial_train_size=5000
#
#   # Cartpole, with different sequence length
#   sbatch scripts/train_local.sh system=cartpole_pybullet flow_matching.sequence_length=64
#
#   # Cartpole, trajectory checking off (endpoint-only eval)
#   sbatch scripts/train_local.sh system=cartpole_pybullet conformal.trajectory_checking=false
#
#   # Override transformer size
#   sbatch scripts/train_local.sh system=cartpole_pybullet model.hidden_dim=512 model.num_layers=8

set -e

PROJECT_DIR="/common/home/dm1487/robotics_research/tripods/olympics-classifier"
PYTHON="/common/users/dm1487/envs/arcmg/bin/python"

cd "$PROJECT_DIR"
mkdir -p slurm_logs

echo "=== Local Dynamics Training ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Args: $@"
echo ""

# prediction_mode=local is always set; all other args pass through
$PYTHON scripts/run_adaptive.py \
    prediction_mode=local \
    "$@"

echo ""
echo "Done: $(date)"
