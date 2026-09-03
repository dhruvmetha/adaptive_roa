#!/bin/bash
# One static CartPole training/evaluation cell. Submit from repository root.
# Usage: sbatch ... scripts/cartpole_static_job.sh <regime> <method> <budget> [smoke|production]
# regime: deterministic | low | med | high | baseline
# method: fm_endpoint | fm_outcome | mlp
#
# Stochastic regimes read stochastic/cartpole/gaussian_signal/<controller>/<regime>.
# Pick the controller with CARTPOLE_STATIC_CONTROLLER (default lqr). Levels are
# NOT shared: lqr has low|med|high, safe_explorer_ppo adds baseline (zero noise).
#SBATCH --partition=unlimited
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:?submit from the repository root}"
cd "$REPO"

REGIME="${1:?missing regime}"
METHOD="${2:?missing method}"
BUDGET="${3:?missing fitting budget}"
MODE="${4:-production}"
SEED="${CARTPOLE_STATIC_SEED:-42}"
CONTROLLER="${CARTPOLE_STATIC_CONTROLLER:-lqr}"
PY="${ADAPTIVE_ROA_PYTHON:-/common/users/dm1487/envs/arcmg/bin/python}"
DATA_BASE="/common/users/shared/pracsys/genMoPlan/data_trajectories"
CAMPAIGN_ROOT="${CARTPOLE_STATIC_ROOT:-/common/users/shared/pracsys/adaptive_roa_experiments/dhruv/cartpole_static_v2}"

export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

if [[ "$MODE" == "smoke" ]]; then
  FIXED_VAL=10
  CAMPAIGN_ROOT="$CAMPAIGN_ROOT/smoke"
else
  FIXED_VAL=500
fi
SELECTED=$((BUDGET + FIXED_VAL))

case "$REGIME" in
  deterministic)
    SYSTEM="cartpole_det_static"
    DATASET_ROOT="$DATA_BASE/deterministic/cartpole_pybullet"
    # No controller axis on the deterministic anchor; keep its historical layout.
    REGIME_TAG="$REGIME"
    ;;
  low|med|high|baseline)
    SYSTEM="cartpole_stoch_static"
    DATASET_ROOT="$DATA_BASE/stochastic/cartpole/gaussian_signal/$CONTROLLER/$REGIME"
    # Both controllers ship a level called low/med/high. Without the controller in
    # the path, an RL run silently lands on top of the matching LQR run.
    REGIME_TAG="${CONTROLLER}_${REGIME}"
    ;;
  *) echo "unknown regime: $REGIME" >&2; exit 2 ;;
esac

# baseline exists only under safe_explorer_ppo, so a wrong pairing resolves to a
# path that is not there. Fail at submit time instead of part-way through training.
if [[ ! -d "$DATASET_ROOT" ]]; then
  echo "dataset root does not exist: $DATASET_ROOT" >&2
  echo "  regime=$REGIME controller=$CONTROLLER" >&2
  exit 2
fi

RUN_DIR="$CAMPAIGN_ROOT/$REGIME_TAG/$METHOD/train_${BUDGET}/seed_${SEED}"
mkdir -p "$RUN_DIR"

COMMON=(
  "system=$SYSTEM"
  "seed=$SEED"
  "fit_train_size=$BUDGET"
  "fixed_val_size=$FIXED_VAL"
  "initial_train_size=$SELECTED"
  "output_dir=$RUN_DIR"
  "n_epochs=1"
  "samples_per_epoch=0"
  "warm_start=false"
  "acquisition.d2_ratio=0"
  "adaptive_v2.filter_confident_pairs=false"
  "num_workers=0"
)
if [[ "$SYSTEM" == "cartpole_stoch_static" ]]; then
  COMMON+=("noise_level=$REGIME" "controller=$CONTROLLER")
fi

case "$METHOD" in
  fm_endpoint) METHOD_ARGS=("predictor=generative") ;;
  fm_outcome) METHOD_ARGS=("+experiment=fm_outcome_baseline") ;;
  mlp) METHOD_ARGS=("predictor=classifier" "+predictor.classifier.pos_weight=1.0") ;;
  *) echo "unknown method: $METHOD" >&2; exit 2 ;;
esac

if [[ "$MODE" == "smoke" ]]; then
  COMMON+=("eval.max_eval_rows=128" "eval.num_mc_samples_eval=2")
  case "$METHOD" in
    fm_endpoint) COMMON+=("predictor.lightning_trainer.max_epochs=1") ;;
    fm_outcome) COMMON+=("predictor.outcome_fm.max_epochs=1") ;;
    mlp) COMMON+=("predictor.classifier.max_epochs=1") ;;
  esac
fi

START_EPOCH="$(date +%s)"
on_exit() {
  code=$?
  if ! grep -q '^status=' "$RUN_DIR/job_metadata.txt" 2>/dev/null; then
    {
      echo "end=$(date --iso-8601=seconds)"
      echo "elapsed_seconds=$(($(date +%s) - START_EPOCH))"
      echo "status=failed"
      echo "exit_code=$code"
    } >> "$RUN_DIR/job_metadata.txt"
  fi
}
trap on_exit EXIT

{
  echo "job_id=${SLURM_JOB_ID:-local}"
  echo "node=$(hostname)"
  echo "start=$(date --iso-8601=seconds)"
  echo "git_head=$(git rev-parse HEAD)"
  echo "git_dirty=$(test -n "$(git status --porcelain)" && echo true || echo false)"
  echo "mode=$MODE"
  echo "regime=$REGIME_TAG"
  echo "controller=$CONTROLLER"
  echo "method=$METHOD"
  echo "fit_train_size=$BUDGET"
  echo "fixed_val_size=$FIXED_VAL"
  echo "selected_size=$SELECTED"
  echo "gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
} > "$RUN_DIR/job_metadata.txt"

echo "Starting $REGIME_TAG / $METHOD / train=$BUDGET on $(hostname)"
"$PY" scripts/run_adaptive.py "${METHOD_ARGS[@]}" "${COMMON[@]}"
"$PY" scripts/export_cartpole_static_predictions.py \
  --run-dir "$RUN_DIR" --dataset-root "$DATASET_ROOT" --regime "$REGIME_TAG" \
  --method "$METHOD" --budget "$BUDGET" --seed "$SEED"

END_EPOCH="$(date +%s)"
{
  echo "end=$(date --iso-8601=seconds)"
  echo "elapsed_seconds=$((END_EPOCH - START_EPOCH))"
  echo "status=complete"
} >> "$RUN_DIR/job_metadata.txt"
echo "Completed in $((END_EPOCH - START_EPOCH)) seconds"
