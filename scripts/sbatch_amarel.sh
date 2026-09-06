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
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=3-00:00:00
# WALLTIME IS A DELIBERATE TRADEOFF, raised from 1 day on 2026-09-05.
#
# For: the 1-day default silently killed a 24-epoch quadrotor2d run at epoch 19
# (TIMEOUT, job 61221125) because nobody passed an override. 3 days is the pool
# maximum, so this cannot overshoot, and the adaptive arms genuinely need 25-30h.
#
# Against: this account is `general`, which preempts SILENTLY. A job that wedges
# at epoch 0 now occupies a slot for three days instead of one, and preemption
# gives no clean signal that it happened. Confirm death with sacct, not absence
# from squeue.
#
# The default is only safe while something is actually watching depth. If you
# submit from this script with no monitor running, pass --time explicitly and
# size it to the job instead of inheriting three days.
#SBATCH --output=/home/st1122/Projects/adaptive_roa/slurm_logs/%x_%j.out
#SBATCH --error=/home/st1122/Projects/adaptive_roa/slurm_logs/%x_%j.err

# NOTE: partition `gpu` only. `cgpu` is Camden -- do not submit there. The old
# `-redhat` suffixed names no longer exist and sbatch rejects them outright.
# Max walltime on all Amarel GPU partitions is 3-00:00:00.

set -euo pipefail

REPO=/home/st1122/Projects/adaptive_roa
cd "$REPO"

# Keep the interpreter on the env's own packages only. On iLab a second numpy in
# ~/.local shadowed the env's and made every torch.multiprocessing spawn child die
# with "CPU dispatcher tracer already initlized", hanging the ensemble FM trainer
# with idle GPUs and empty checkpoint dirs. Set here too so the same user-site
# drift cannot reappear on Amarel.
export PYTHONNOUSERSITE=1

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
