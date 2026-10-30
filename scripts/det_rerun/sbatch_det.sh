#!/bin/bash
# iLab job for one det-paper rerun cell. Runs from the det-paper worktree, not the
# main checkout, so the job imports the code it was launched with.
#   sbatch -J <name> --gres=gpu:<type>:1 --mem=<mem> scripts/det_rerun/sbatch_det.sh <hydra overrides>
# No --cpus-per-task: iLab rejects it.
#SBATCH --partition=unlimited
#SBATCH --nodes=1
#SBATCH --output=/common/home/st1122/Projects/adaptive_roa/slurm_logs/det_rerun/%x_%j.out
#SBATCH --error=/common/home/st1122/Projects/adaptive_roa/slurm_logs/det_rerun/%x_%j.err
set -euo pipefail
WT=/common/home/st1122/Projects/adaptive_roa/.claude/worktrees/det-paper
cd "$WT"
# The editable install points at the main tree; a worktree run must put itself first.
export PYTHONPATH="$WT"
export PYTHONNOUSERSITE=1           # ~/.local numpy shadows the env's and hangs spawn children
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
# DataLoader workers clean their temp dirs on exit; on NFS that raises EBUSY tracebacks.
export TMPDIR="/tmp/st1122_${SLURM_JOB_ID:-$$}"; mkdir -p "$TMPDIR"
: "${ADAPTIVE_ROA_COMMIT:?launch.sh must export ADAPTIVE_ROA_COMMIT}"
echo "Node:   $(hostname)"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd, -)"
echo "Commit: $ADAPTIVE_ROA_COMMIT"
echo "Start:  $(date)"
echo "Args:   $*"
/common/home/st1122/Projects/adaptive_roa/env/bin/python scripts/run_adaptive.py "$@"
echo "Done:   $(date)"
