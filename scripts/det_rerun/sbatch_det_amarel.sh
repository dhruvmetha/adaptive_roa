#!/bin/bash
# Amarel job for one det-paper rerun cell, run from the commit's read-only snapshot
# (DET_CODE, made on Amarel by snapshot.sh). launch.sh <name> amarel submits it.
#   sbatch -J <name> --mem=<mem> --time=<wall> scripts/det_rerun/sbatch_det_amarel.sh <hydra overrides>
# `general` preempts without requeue: a preempted cell resumes from its last epoch
# (DET_SCRIPT=scripts/resume_adaptive.py, args --run-dir <dir>), never from 0.
#SBATCH --account=general
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --exclude=gpu029,gpu046
#SBATCH --output=/home/st1122/det_rerun_logs/%x_%j.out
#SBATCH --error=/home/st1122/det_rerun_logs/%x_%j.err
set -euo pipefail
: "${DET_CODE:?launch.sh must export DET_CODE}"
: "${ADAPTIVE_ROA_COMMIT:?launch.sh must export ADAPTIVE_ROA_COMMIT}"
cd "$DET_CODE"
export PYTHONPATH="$DET_CODE" PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1} MKL_NUM_THREADS=${OMP_NUM_THREADS:-1}
export TMPDIR="/tmp/st1122_${SLURM_JOB_ID:-$$}"; mkdir -p "$TMPDIR"
echo "Node:   $(hostname)"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | paste -sd, - || true)"
echo "Commit: $ADAPTIVE_ROA_COMMIT ($DET_CODE)"
echo "Start:  $(date)"
echo "Args:   $*"
/home/st1122/Projects/adaptive_roa/env/bin/python "${DET_SCRIPT:-scripts/run_adaptive.py}" "$@"
echo "Done:   $(date)"
