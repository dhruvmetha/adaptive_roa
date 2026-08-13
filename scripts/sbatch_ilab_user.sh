#!/bin/bash
# Portable iLab job template for adaptive_roa — same job shape as sbatch_ilab.sh,
# with nothing user-specific baked in.
#
# sbatch_ilab.sh hardcodes REPO and the log paths under /common/home/st1122/...
# Those resolve on the shared filesystem but are not writable by anyone else, so
# a job submitted by another user dies at WRITE time, not at submit time — it
# accepts, allocates, then vanishes. This file resolves both at runtime instead.
#
# Usage (from your own clone):
#   mkdir -p slurm_logs
#   ADAPTIVE_ROA_PYTHON=/path/to/env/bin/python \
#     sbatch -J my_job --gres=gpu:2 --mem=60G --time=5-00:00:00 \
#     scripts/sbatch_ilab_user.sh system=pendulum_stoch_tau tau=t030 ...
#
# Everything after the script name is passed through to run_adaptive.py as Hydra
# overrides.

#SBATCH --partition=unlimited
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=40G
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

# --output/--error are RELATIVE, so SLURM resolves them against the submission
# directory. That is what makes this file user-agnostic. The submitting shell
# must have created slurm_logs/ first — SLURM does not mkdir, and a missing
# directory kills the job at write time with no queue-visible error.
#
# Do NOT add --cpus-per-task: iLab rejects the job outright ("Please do not
# specify the number of CPUs. There is actually no limit."). Amarel requires it.
#
# --nodes=1 is load-bearing: the ensemble FM trainer uses torch.multiprocessing
# .spawn, which cannot cross nodes. A multi-node allocation would strand members.
#
# No --exclude here. sbatch_ilab.sh excludes ilab4 because its RTX PRO 5000
# (Blackwell, sm_120) needs a cu128 torch build; check your own env before
# copying that constraint. Never name a node outside this cluster in --exclude
# (e.g. westeros) — sbatch rejects the job with "Invalid node name specified".

set -euo pipefail

# SLURM copies the batch script to a spool directory, so BASH_SOURCE points at
# the copy, not at the clone. SLURM_SUBMIT_DIR is the submission cwd and is the
# only reliable handle on the repo.
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"

if [ ! -f "$REPO/scripts/run_adaptive.py" ]; then
  echo "ERROR: $REPO does not look like an adaptive_roa clone (no scripts/run_adaptive.py)." >&2
  echo "       Submit from the repo root so SLURM_SUBMIT_DIR points at it." >&2
  exit 1
fi

PY="${ADAPTIVE_ROA_PYTHON:-python}"
if ! command -v "$PY" >/dev/null 2>&1; then
  echo "ERROR: python '$PY' not found on the compute node." >&2
  echo "       Export ADAPTIVE_ROA_PYTHON to your env's interpreter before sbatch;" >&2
  echo "       sbatch forwards the submitting environment by default." >&2
  exit 1
fi

# Load-bearing for the ensemble FM trainer. Two numpy installs can coexist on
# this filesystem — the env's own and one in ~/.local/lib/python3.10 — and the
# user-site copy shadows the env's. A torch.multiprocessing spawn child re-imports
# the main module, numpy's C extension gets initialized from two different paths
# in one process, and the child dies with
#   RuntimeError: CPU dispatcher tracer already initlized
# The members then leave empty checkpoint dirs and the GPUs sit at 0% while the
# parent blocks in mp.spawn(join=True) — a silent hang, not a crash.
export PYTHONNOUSERSITE=1

# Cap per-process thread pools. Without this each torch process opens an OpenMP
# pool sized to the node's core count (64-96 here). An ensemble FM arm is 5 member
# processes plus their DataLoader workers, and several arms share a node, so the
# pools multiply into thousands of threads and members die with
# "RuntimeError: can't start new thread" — which surfaces as the misleading
# "DataLoader worker exited unexpectedly". GPU training needs almost no OMP width.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "Node:   $(hostname)"
echo "Repo:   $REPO"
echo "Python: $PY"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd, -)"
echo "Start:  $(date)"
echo "Args:   $*"

"$PY" scripts/run_adaptive.py "$@"

echo "Done:   $(date)"
