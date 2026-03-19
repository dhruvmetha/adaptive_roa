#!/bin/bash
#SBATCH --job-name=pend-eval
#SBATCH --output=slurm_logs/pend_eval_%j.out
#SBATCH --error=slurm_logs/pend_eval_%j.err
#SBATCH --gres=gpu:4500_ada:1
#SBATCH --time=04:00:00
#SBATCH --mem=32G

# Evaluate d2=0.5 non-manifold cold-start pendulum run
#
# Usage:
#   sbatch scripts/run_pendulum_eval.sh

set -e

PROJECT_DIR="/common/home/dm1487/robotics_research/tripods/olympics-classifier"
PYTHON="/common/users/dm1487/envs/arcmg/bin/python"

cd "$PROJECT_DIR"
mkdir -p slurm_logs

TRAINING_DIR="/common/users/shared/pracsys/adaptive_roa_experiments/dhruv/adaptive_pendulum_dhruv/outputs/training_index_0_d2_ratio_0.5_warm_start_False_threshold_mode_dynamic_adapt_iter_20_alpha_0.1_sampling_mode_direct/2026-03-01_19-50-39"

echo "=== Evaluating d2=0.5 non-manifold ==="
echo "Training dir: $TRAINING_DIR"

$PYTHON scripts/reevaluate.py "$TRAINING_DIR" \
    --attractor_radius 0.075 \
    --alpha_eval 0.1 \
    --num_mc_samples 20 \
    --batch_size 100000 \
    --device cuda:0 \
    --force \
    --verbose

echo ""
echo "Done!"
