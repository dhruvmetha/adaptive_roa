#!/bin/bash
# The body of every det_v2 run: launch and resume, on iLab (GPU and CPU), Amarel and the
# direct boxes. It runs from the commit's read-only snapshot (DET_V2_CODE) and sets the
# runtime environment in one place, so no path can miss a variable:
#   umask 002                         group-writable outputs in the shared, setgid det_v2 root
#   PYTHONPATH=<snapshot>             the env's editable install points at a live checkout
#   PYTHONNOUSERSITE=1                ~/.local numpy shadows the env's and hangs spawn children
#   Q3D_CACHE_TRANSFORMED=1, Q2D_CACHE_TRANSFORMED=1
#   OMP_NUM_THREADS / MKL_NUM_THREADS from the launcher (1 on a card, more on CPU-only jobs)
#   TMPDIR=<TMP_ROOT>/<net id>_<job id or name>; DataLoader temp dirs on NFS raise EBUSY
#   ADAPTIVE_ROA_COMMIT               pinned by the launcher; a snapshot has no .git
#   job.sh <script> <args...>         e.g. scripts/run_adaptive.py +experiment=det_v2/runs/<name> output_dir=...
#                                     or   scripts/resume_adaptive.py --run-dir <run dir>
# SLURM options come from the sbatch command line (launch.sh, resume_*.sh).
#SBATCH --nodes=1
set -euo pipefail
umask 002
: "${DET_V2_CODE:?launch.sh must export DET_V2_CODE}"
: "${DET_V2_PYTHON:?launch.sh must export DET_V2_PYTHON}"
: "${ADAPTIVE_ROA_COMMIT:?launch.sh must export ADAPTIVE_ROA_COMMIT}"
cd "$DET_V2_CODE"
export PYTHONPATH="$DET_V2_CODE"
export PYTHONNOUSERSITE=1
export Q3D_CACHE_TRANSFORMED=1
export Q2D_CACHE_TRANSFORMED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export ADAPTIVE_ROA_COMMIT
export TMPDIR="${DET_V2_TMP_ROOT:-/tmp}/${DET_V2_NET_ID:-${USER:-user}}_${SLURM_JOB_ID:-${DET_V2_NAME:-$$}}"
mkdir -p "$TMPDIR"
echo "Node:   $(hostname)"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | paste -sd, - || true) (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset})"
echo "Commit: $ADAPTIVE_ROA_COMMIT ($DET_V2_CODE)"
echo "Start:  $(date)"
echo "Args:   $*"
rc=0
"$DET_V2_PYTHON" "$@" || rc=$?
echo "Done:   $(date) (exit $rc)"
exit $rc
