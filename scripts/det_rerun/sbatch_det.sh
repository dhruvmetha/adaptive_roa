#!/bin/bash
# iLab job for one det-paper rerun cell. Runs from the commit's read-only snapshot
# (DET_CODE, made by snapshot.sh), so the job imports the code it was launched with.
#   sbatch -J <name> --gres=gpu:<type>:1 --mem=<mem> scripts/det_rerun/sbatch_det.sh <hydra overrides>
# No --cpus-per-task: iLab rejects it.
#SBATCH --partition=unlimited
#SBATCH --nodes=1
#SBATCH --output=/common/home/st1122/Projects/adaptive_roa/slurm_logs/det_rerun/%x_%j.out
#SBATCH --error=/common/home/st1122/Projects/adaptive_roa/slurm_logs/det_rerun/%x_%j.err
set -euo pipefail
: "${DET_CODE:?launch.sh must export DET_CODE}"
WT=$DET_CODE
cd "$WT"
# The editable install points at the main tree; a snapshot run must put itself first.
export PYTHONPATH="$WT"
export PYTHONNOUSERSITE=1           # ~/.local numpy shadows the env's and hangs spawn children
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1} MKL_NUM_THREADS=${OMP_NUM_THREADS:-1}
# DataLoader workers clean their temp dirs on exit; on NFS that raises EBUSY tracebacks.
export TMPDIR="/tmp/st1122_${SLURM_JOB_ID:-$$}"; mkdir -p "$TMPDIR"
: "${ADAPTIVE_ROA_COMMIT:?launch.sh must export ADAPTIVE_ROA_COMMIT}"
echo "Node:   $(hostname)"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | paste -sd, - || true)"
echo "Commit: $ADAPTIVE_ROA_COMMIT ($WT)"
echo "Start:  $(date)"
echo "Args:   $*"
/common/home/st1122/Projects/adaptive_roa/env/bin/python "${DET_SCRIPT:-scripts/run_adaptive.py}" "$@"
echo "Done:   $(date)"
