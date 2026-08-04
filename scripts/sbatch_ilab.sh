#!/bin/bash
# iLab job template for adaptive_roa.
#
# Usage (from the iLab clone, which is the primary working dir):
#   sbatch -J en_fm_high_total scripts/sbatch_ilab.sh system=pendulum_stoch ...
#
# Everything after the script name is passed through to run_adaptive.py as Hydra
# overrides. Override sbatch settings on the command line as needed, e.g.
#   sbatch -J fm --gres=gpu:4 --mem=80G scripts/sbatch_ilab.sh ...
#
#SBATCH --partition=unlimited
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=40G
#SBATCH --exclude=ilab4
#SBATCH --output=/common/home/st1122/Projects/adaptive_roa/slurm_logs/%x_%j.out
#SBATCH --error=/common/home/st1122/Projects/adaptive_roa/slurm_logs/%x_%j.err

# Do NOT add --cpus-per-task: iLab rejects the job outright ("Please do not
# specify the number of CPUs. There is actually no limit.").
#
# --nodes=1 is load-bearing: the ensemble FM trainer uses torch.multiprocessing
# .spawn, which cannot cross nodes. A multi-node allocation would strand members.
#
# ilab4 is excluded because its RTX PRO 5000 (Blackwell, sm_120) is not supported
# by the main env's torch build -- that hardware needs env_blackwell.
#
# westeros is NOT in this SLURM cluster, so it must not appear in --exclude:
# naming an unknown node makes sbatch reject the job with "Invalid node name
# specified". (Its GPU5 is faulted and poisons NVML box-wide, so keep using it
# for CPU work only -- but that is enforced by not submitting there, not here.)

set -euo pipefail

REPO=/common/home/st1122/Projects/adaptive_roa
cd "$REPO"

# Load-bearing for the ensemble FM trainer. There are TWO numpy 2.2.6 installs on
# this filesystem -- the env's own and one in ~/.local/lib/python3.10 -- and the
# user-site copy shadows the env's. A torch.multiprocessing spawn child re-imports
# the main module (multiprocessing/spawn.py:_fixup_main_from_path), numpy's C
# extension gets initialized from two different paths in one process, and the
# child dies with "RuntimeError: CPU dispatcher tracer already initlized".
# The members then leave empty checkpoint dirs and the GPUs sit at 0% while the
# parent blocks in mp.spawn(join=True) -- a silent hang, not a crash.
export PYTHONNOUSERSITE=1

echo "Node:  $(hostname)"
echo "GPU:   $(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd, -)"
echo "Start: $(date)"
echo "Args:  $*"

"$REPO/env/bin/python" scripts/run_adaptive.py "$@"

echo "Done:  $(date)"
