#!/bin/bash
# Amarel job template for adaptive_roa.
#
# Usage (from the Amarel clone):
#   sbatch -J adapt_pend_rl scripts/sbatch_amarel.sh system=pendulum_g1 controller=rl
#
# Everything after the script name is passed through to run_adaptive.py as Hydra
# overrides. Override sbatch settings on the command line as needed, e.g.
#   sbatch -J big --gres=gpu:2 --time=2-00:00:00 scripts/sbatch_amarel.sh ...
#
#SBATCH --account=general
#SBATCH --partition=gpu-redhat
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/st1122/Projects/adaptive_roa/slurm_logs/%x_%j.out
#SBATCH --error=/home/st1122/Projects/adaptive_roa/slurm_logs/%x_%j.err

# NOTE: partition gpu-redhat only. cgpu-redhat is Camden -- do not submit there.
# Max walltime on all Amarel GPU partitions is 3-00:00:00.

set -euo pipefail

REPO=/home/st1122/Projects/adaptive_roa
cd "$REPO"

echo "Node:  $(hostname)"
echo "GPU:   $(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd, -)"
echo "Start: $(date)"
echo "Args:  $*"

# Fail early and loudly if the dataset tree was never staged for this run,
# rather than letting Hydra resolve to an empty directory.
DATA_DIR=$(grep -E '^DATA_DIR=' "$REPO/.env" | cut -d= -f2-)
if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: DATA_DIR does not exist: $DATA_DIR" >&2
    exit 1
fi

"$REPO/env/bin/python" scripts/run_adaptive.py "$@"

echo "Done:  $(date)"
