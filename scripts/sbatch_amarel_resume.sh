#!/bin/bash
# Resume an interrupted adaptive run on Amarel.
#
# Usage: sbatch -J rs_<name> scripts/sbatch_amarel_resume.sh <run-dir> [extra resume args]
#
# Unlike sbatch_amarel.sh this calls resume_adaptive.py, which continues from the
# epoch after the last COMPLETED one instead of restarting at 0. That distinction
# matters: run_adaptive.py would overwrite the existing epoch dirs in place and
# stitch two runs into one arm.
#SBATCH --account=general
#SBATCH --partition=gpu-redhat
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/st1122/Projects/adaptive_roa/slurm_logs/%x_%j.out
#SBATCH --error=/home/st1122/Projects/adaptive_roa/slurm_logs/%x_%j.err
set -euo pipefail
REPO=/home/st1122/Projects/adaptive_roa
cd "$REPO"
echo "Node:  $(hostname)"
echo "GPU:   $(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd, -)"
echo "Start: $(date)"
echo "Args:  $*"
"$REPO/env/bin/python" scripts/resume_adaptive.py --run-dir "$@"
echo "Done:  $(date)"
