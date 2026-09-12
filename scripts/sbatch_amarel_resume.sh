#!/bin/bash
# Resume an interrupted adaptive run on Amarel.
#
# Usage: sbatch -J rs_<name> scripts/sbatch_amarel_resume.sh <run-dir> [extra resume args]
#        sbatch -J rs_<name> scripts/sbatch_amarel_resume.sh --run-dir <run-dir> [extra]
#
# Unlike sbatch_amarel.sh this calls resume_adaptive.py, which continues from the
# epoch after the last COMPLETED one instead of restarting at 0. That distinction
# matters: run_adaptive.py would overwrite the existing epoch dirs in place and
# stitch two runs into one arm.
#SBATCH --account=general
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=3-00:00:00
#SBATCH --output=/home/st1122/Projects/adaptive_roa/slurm_logs/%x_%j.out
#SBATCH --error=/home/st1122/Projects/adaptive_roa/slurm_logs/%x_%j.err
set -euo pipefail
REPO=/home/st1122/Projects/adaptive_roa
cd "$REPO"
echo "Node:  $(hostname)"
echo "GPU:   $(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd, -)"
echo "Start: $(date)"
echo "Args:  $*"
# Accept both the bare `<run-dir>` form above and the `--run-dir <dir>` form the
# campaign resume plan uses. Without this the latter expands to
# `--run-dir --run-dir <dir>` and argparse takes the flag as the value.
if [ "${1:-}" = "--run-dir" ]; then
  "$REPO/env/bin/python" scripts/resume_adaptive.py "$@"
else
  "$REPO/env/bin/python" scripts/resume_adaptive.py --run-dir "$@"
fi
echo "Done:  $(date)"
