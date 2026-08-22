#!/bin/bash
# iLab resume template. Same environment guards as sbatch_ilab.sh, but it calls
# scripts/resume_adaptive.py, which reads the run's own .hydra/config.yaml and
# continues from the last completed epoch rather than restarting from zero.
#
# Usage:
#   sbatch -J q3d_nd048_yield_mlp --mem=64G scripts/sbatch_ilab_resume.sh \
#          --run-dir /path/to/run_dir
#
# Memory is the reason this exists as a separate submission rather than a
# requeue: a resumed arm needs a DIFFERENT --mem from its original launch when
# the original died of OOM, and the pair count that drives that memory grows
# with the training set, so the right value depends on how deep the arm already is.
#
#SBATCH --partition=unlimited
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=40G
#SBATCH --exclude=ilab4
#SBATCH --output=/common/home/st1122/Projects/adaptive_roa/slurm_logs/%x_%j.out
#SBATCH --error=/common/home/st1122/Projects/adaptive_roa/slurm_logs/%x_%j.err

set -euo pipefail
REPO=/common/home/st1122/Projects/adaptive_roa
cd "$REPO"

export PYTHONNOUSERSITE=1     # user-site numpy shadows the env's and hangs mp.spawn
export OMP_NUM_THREADS=1      # else pools multiply across members -> can't start new thread
export MKL_NUM_THREADS=1

echo "Node:  $(hostname)"
echo "GPU:   $(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd, -)"
echo "Start: $(date)"
echo "Args:  $*"

"$REPO/env/bin/python" scripts/resume_adaptive.py "$@"

echo "Done:  $(date)"
