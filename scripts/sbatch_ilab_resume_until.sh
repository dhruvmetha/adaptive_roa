#!/bin/bash
# Resume one adaptive run, checking its evaluation metric after every round.
# Submit from the repository root; override --gres/--time at sbatch time when
# performing a one-round smoke.
#SBATCH --partition=unlimited
#SBATCH --nodes=1
#SBATCH --gres=gpu:5
#SBATCH --mem=100G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"

PY="${ADAPTIVE_ROA_PYTHON:-python}"
if ! command -v "$PY" >/dev/null 2>&1; then
  echo "ERROR: python '$PY' not found; export ADAPTIVE_ROA_PYTHON before sbatch." >&2
  exit 1
fi

export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

echo "Node:   $(hostname)"
echo "Repo:   $REPO"
echo "Python: $PY"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd, -)"
echo "Start:  $(date)"
echo "Args:   $*"

"$PY" scripts/resume_adaptive_until.py "$@"

echo "Done:   $(date)"
