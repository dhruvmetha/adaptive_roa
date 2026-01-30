#!/usr/bin/env bash
set -euo pipefail

# Launch 6 quadrotor classification training jobs, each in its own tmux session.
#
# Default behavior (recommended on Slurm clusters):
# - Uses `srun` to request 1 GPU per session.
# - Runs the actual python command inside the allocation via `bash -lc`.
#
# Usage:
#   bash adaptive_roa/scripts/launch_quadrotor_classification_tmux.sh
#
# Optional env overrides:
#   USE_SLURM=0                        # run locally, uses CUDA_VISIBLE_DEVICES=0..5
#   CONDA_ACTIVATE=/path/to/activate   # default: /common/users/rm1838/miniforge3/bin/activate
#   CONDA_ENV=adaptive_roa             # conda env name
#   SRUN_ARGS="-G 1 --pty"             # additional args to srun (partition/account/time etc)
#   PROJECT_DIR=/abs/path/to/adaptive_roa

PROJECT_DIR="${PROJECT_DIR:-}"
if [[ -z "$PROJECT_DIR" ]]; then
  PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

LOG_DIR="${PROJECT_DIR}/logs"
mkdir -p "$LOG_DIR"

CONDA_ACTIVATE="${CONDA_ACTIVATE:-/common/users/rm1838/miniforge3/bin/activate}"
CONDA_ENV="${CONDA_ENV:-adaptive_roa}"
CONDA_BIN="${CONDA_BIN:-/common/users/rm1838/miniforge3/bin/conda}"

USE_SLURM="${USE_SLURM:-1}"
if [[ "$USE_SLURM" == "1" ]] && ! command -v srun >/dev/null 2>&1; then
  echo "WARN: USE_SLURM=1 but 'srun' not found. Falling back to local GPU mode (USE_SLURM=0)." >&2
  USE_SLURM="0"
fi

SRUN_ARGS="${SRUN_ARGS:--G 1 --pty}"

ensure_tmux_session() {
  local session="$1"
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "Session already exists, skipping: $session"
    return 1
  fi
  tmux new-session -d -s "$session"
  return 0
}

build_inner_cmd() {
  local session="$1"
  local train_cmd="$2"

  # Keep tmux session open at the end for easy inspection.
  cat <<EOF
set -euo pipefail
cd "$PROJECT_DIR"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/$session.log"
{
  echo "[\$(date)] starting: $session"

  # Robust conda activation (works in non-interactive shells).
  if [[ -x "$CONDA_BIN" ]]; then
    eval "\$($CONDA_BIN shell.bash hook)"
    conda activate "$CONDA_ENV"
  elif [[ -f "$CONDA_ACTIVATE" ]]; then
    # Fallback (some installs support: source bin/activate <env>)
    source "$CONDA_ACTIVATE" "$CONDA_ENV"
  else
    echo "WARN: Could not find conda activation at CONDA_BIN='$CONDA_BIN' or CONDA_ACTIVATE='$CONDA_ACTIVATE'"
  fi

  ${train_cmd}

  echo "[\$(date)] finished: $session"
} 2>&1 | tee -a "\$LOG_FILE"
echo "Done. Press Enter to close this tmux session."
read -r _
EOF
}

launch_one() {
  local session="$1"
  local train_cmd="$2"
  local local_gpu_id="$3"

  ensure_tmux_session "$session" || return 0

  local inner_cmd
  inner_cmd="$(build_inner_cmd "$session" "$train_cmd")"

  if [[ "$USE_SLURM" == "1" ]]; then
    # Run inside a single allocation (no manual "wait then send keys" steps).
    tmux send-keys -t "$session" \
      "srun ${SRUN_ARGS} --job-name=${session} bash -lc $(printf "%q" "$inner_cmd")" Enter
  else
    # Local mode: pin each session to a specific GPU id (0..5 by default).
    tmux send-keys -t "$session" \
      "CUDA_VISIBLE_DEVICES=${local_gpu_id} bash -lc $(printf "%q" "$inner_cmd")" Enter
  fi
}

# (session_name, system, train_size)
# NOTE: user requested "Train Size" 500/1000/2000 for both 2D and 3D.
# We keep max_epochs aligned with the session name as well (can be changed independently if desired).
launch_one "q2d_500"  "python src/classification/train_quadrotor2d.py name=q2d_500  trainer.max_epochs=500  data.max_samples=500"   0
launch_one "q2d_1000" "python src/classification/train_quadrotor2d.py name=q2d_1000 trainer.max_epochs=1000 data.max_samples=1000" 1
launch_one "q2d_2000" "python src/classification/train_quadrotor2d.py name=q2d_2000 trainer.max_epochs=2000 data.max_samples=2000" 2
launch_one "q3d_500"  "python src/classification/train_quadrotor3d.py name=q3d_500  trainer.max_epochs=500  data.max_samples=500"   3
launch_one "q3d_1000" "python src/classification/train_quadrotor3d.py name=q3d_1000 trainer.max_epochs=1000 data.max_samples=1000" 4
launch_one "q3d_2000" "python src/classification/train_quadrotor3d.py name=q3d_2000 trainer.max_epochs=2000 data.max_samples=2000" 5

echo
echo "Launched (or already running). Useful commands:"
echo "  tmux ls"
echo "  tmux attach -t q2d_500    # detach: Ctrl-b then d"
echo "  tail -f \"$LOG_DIR/q2d_500.log\""
echo

