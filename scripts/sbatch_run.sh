#!/bin/bash
#SBATCH --job-name=roa
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=48G
# Generic single-run launcher for the ROA classifier-vs-flowmatching study.
# All args pass through as Hydra overrides to run_adaptive.py.
#   sbatch -J <name> scripts/sbatch_run.sh system=pendulum predictor=classifier ...
set -e
PROJECT_DIR="/common/home/dm1487/robotics_research/tripods/olympics-classifier"
PYTHON="/common/users/dm1487/envs/arcmg/bin/python"
cd "$PROJECT_DIR"
mkdir -p slurm_logs
echo "=== ROA run ==="; echo "Date: $(date)"; echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "Args: $@"; echo ""
$PYTHON scripts/run_adaptive.py "$@"
echo ""; echo "Done: $(date)"
