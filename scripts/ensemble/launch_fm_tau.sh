#!/bin/bash
# Launch the flow-matching half of the ensemble-epistemic campaign on the tau
# stochastic-pendulum family (system=pendulum_stoch_tau).
#
# WHY THIS EXISTS ALONGSIDE launch_fm_ilab.sh
# -------------------------------------------
# That script targets system=pendulum_stoch ({DATA_DIR}/noisy/pendulum/lqr/*), which
# is group st1122 and unreadable to anyone else, and it hardcodes st1122's repo and
# log paths so it cannot be reused as-is. This one runs the same five arms against
# the tau datasets, resolves its own paths, and adds the seed replicates in the same
# submit loop.
#
# READ scripts/ensemble/probe_tau_aleatoric.py OUTPUT FIRST. Every tau level measured
# so far sits at or below the `noisy/low` ambiguity of the earlier family, where two
# campaigns found no separation between arms. Launching the full grid before a level
# clears the separation floor buys a null result at ~50 GPU-hours per level.
#
# BUDGET
# ------
# fm_ensemble spawns one process per member (5), round-robined over visible GPUs, so
# 2 GPUs per arm is right: a pendulum flow matcher pushes a mid-range card to ~25%,
# and members that share a card cost wall-clock on that card, not correctness. Seven
# jobs per level (5 arms + 2 floor seeds) = 14 GPUs.
#
# Usage:
#   scripts/ensemble/launch_fm_tau.sh t050              # 5 arms + 2 floor seeds
#   scripts/ensemble/launch_fm_tau.sh t050 --smoke      # one arm, on the smallest card
#   scripts/ensemble/launch_fm_tau.sh t050 --dry-run
#
# Env overrides: EXP_ROOT, ADAPTIVE_ROA_PYTHON, ILAB_HOST, TIME_LIMIT, MEM, GPUS.
set -u

TAU="${1:-t050}"
shift || true

SMOKE=0
DRY=0
for arg in "$@"; do
  case "$arg" in
    --smoke)   SMOKE=1 ;;
    --dry-run) DRY=1 ;;
    *) echo "unknown flag: $arg" >&2; exit 2 ;;
  esac
done

case "$TAU" in
  t000|t010|t015|t030|t050) ;;
  *)
    # Hydra parses `tau=0.30` as a float and renders "tau_0.3", which misses the
    # tau_0.30 directory. The config takes a token for exactly that reason, and an
    # unknown token raises InterpolationKeyError before any training starts.
    echo "ERROR: tau must be a TOKEN: t000 | t010 | t015 | t030 | t050 (got '$TAU')" >&2
    exit 2 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$SCRIPT_DIR/../.." && pwd)"

PY="${ADAPTIVE_ROA_PYTHON:-$(command -v python || true)}"
if [ -z "$PY" ] || [ ! -x "$PY" ]; then
  echo "ERROR: set ADAPTIVE_ROA_PYTHON to your env's interpreter (activate the env, or" >&2
  echo "       export ADAPTIVE_ROA_PYTHON=/path/to/env/bin/python)." >&2
  exit 1
fi
export ADAPTIVE_ROA_PYTHON="$PY"

# EXP_DIR is per-user and lives in .env, which is gitignored; read it from there
# rather than assuming a path. Never write under DATA_DIR.
EXP_ROOT="${EXP_ROOT:-$("$PY" -c '
import sys
sys.path.insert(0, "'"$R"'")
from adaptive_roa.utils.env_config import get_exp_dir
print(get_exp_dir())
' 2>/dev/null)}"
if [ -z "$EXP_ROOT" ]; then
  echo "ERROR: could not resolve EXP_DIR. Set EXP_ROOT explicitly." >&2
  exit 1
fi

E="$EXP_ROOT/ensemble_epistemic_tau/$TAU"
LOG="$R/docs/experiments/ensemble_epistemic/runs_tau.jsonl"
TIME_LIMIT="${TIME_LIMIT:-5-00:00:00}"
MEM="${MEM:-60G}"
GPUS="${GPUS:-2}"

# No card-type pin and no --exclude. One job lands on one node, so an arm gets one
# card type automatically — which is the constraint that matters, since an arm's M
# members train in lockstep per epoch and a mixed arm runs at its slowest card.
# Across the campaign, mixed types are fine: cross-hardware float nondeterminism
# inflates the seed floor, which makes significance HARDER to claim, not easier.
# Comparisons are at matched EPOCH, never matched walltime.
GRES="gpu:${GPUS}"
if [ "$SMOKE" = 1 ]; then
  # a4000 = 16 GB is the smallest card in `unlimited`. OOM there is a crash, not
  # noise, so prove one arm fits before fanning out.
  GRES="gpu:a4000:${GPUS}"
fi

arm_overrides() {
  case $1 in
    total)    echo "acquisition=decomp_total" ;;
    epi_var)  echo "acquisition=decomp_epi_var" ;;
    epi_bald) echo "acquisition=decomp_epi_bald" ;;
    aleat)    echo "acquisition=decomp_aleat" ;;
    dir00|dir00_s43|dir00_s44) echo "acquisition=direct acquisition.d2_ratio=0" ;;
  esac
}

arm_seed() {
  case $1 in
    dir00_s43) echo 43 ;;
    dir00_s44) echo 44 ;;
    *)         echo 42 ;;
  esac
}

# The four scored arms, the random control, then the two floor replicates.
#
# The floor seeds go in THIS loop, not a later one. The previous FM campaign died
# with no floor because its replicates were launched behind the arms and the
# allocation was reclaimed first — leaving 15 epochs of arm data that no verdict
# could be read off. A floor is not optional post-processing; it is the only thing
# that turns a gap into a result.
ARMS="epi_var total dir00 epi_bald aleat dir00_s43 dir00_s44"
if [ "$SMOKE" = 1 ]; then
  ARMS="epi_bald"
fi

mkdir -p "$R/slurm_logs"

# Submit locally when sbatch is here, otherwise hop to a login node. ilab1 and
# ilab4 login nodes hang mid key-exchange; ilab2/ilab3 work. (The ilab4 COMPUTE
# node is fine and is not excluded above — different thing entirely.)
SSH_HOST="${ILAB_HOST:-ilab2.cs.rutgers.edu}"
submit() {
  if command -v sbatch >/dev/null 2>&1; then
    ( cd "$R" && sbatch --parsable "$@" ) 2>/dev/null | tr -dc '0-9'
  else
    # 240s, not 120: iLab logins are slow enough that a 120s cap silently killed
    # every submission in a loop while the same sbatch typed by hand succeeded.
    timeout 240 ssh "$SSH_HOST" \
      "cd $R && ADAPTIVE_ROA_PYTHON=$PY sbatch --parsable $*" 2>/dev/null | tr -dc '0-9'
  fi
}

echo "repo:    $R"
echo "python:  $PY"
echo "tau:     $TAU"
echo "outputs: $E"
echo "gres:    $GRES   time: $TIME_LIMIT   mem: $MEM"
echo

for arm in $ARMS; do
  RID="fm_${TAU}_${arm}"
  if [ -e "$E/$RID" ]; then
    echo "SKIP $RID: output dir exists — delete it first, or the arm's curve gets"
    echo "     stitched from two runs. The engine has no resume logic: a relaunch"
    echo "     restarts at epoch 0 and silently overwrites its own early epochs."
    continue
  fi

  SEED=$(arm_seed "$arm")
  # num_workers=1 is a hard constraint, not tuning. `ulimit -u` on the iLab nodes
  # is 2000, and on Linux that limit counts THREADS, per user, per node, across all
  # your jobs. One arm is 5 member processes, each with train+val DataLoader workers
  # carrying several runtime threads: ~570 threads per arm measured. Four arms on
  # one node blow the limit 25-35 minutes in, surfacing as "can't start new thread"
  # or "BlockingIOError: [Errno 11] Resource temporarily unavailable". Capping
  # OMP/MKL does NOT help — those shrink each process's pool, not the process count.
  ARGS="system=pendulum_stoch_tau tau=$TAU $(arm_overrides "$arm") \
predictor=fm_ensemble seed=$SEED num_workers=1 \
predictor.lightning_trainer.enable_progress_bar=false \
output_dir=$E/$RID"

  CMD="-J en_${RID} --gres=$GRES --mem=$MEM --time=$TIME_LIMIT \
scripts/sbatch_ilab_user.sh $ARGS"

  if [ "$DRY" = 1 ]; then
    echo "DRY $RID:"
    echo "    sbatch $CMD"
    continue
  fi

  JID=$(submit $CMD)
  if [ -z "$JID" ]; then
    echo "FAILED to submit $RID"
    continue
  fi

  case "$arm" in
    dir00*) D2=0.0 ;;
    *)      D2=1.0 ;;
  esac
  "$PY" "$R/scripts/exp_log.py" append --log "$LOG" \
    --run-id "$RID" --system pendulum_stoch_tau --level "$TAU" --predictor fm \
    --arm "$arm" --cluster ilab --job-id "$JID" --output-dir "$E/$RID" \
    --score-mode "$arm" --seed "$SEED" --n-members 5 --k-acq 20 \
    --d2-ratio "$D2" >/dev/null
  echo "$RID -> $JID"
done

echo
echo "Track:   $PY $R/scripts/exp_log.py report --log $LOG"
echo "Preempt: $R/scripts/check_preempted.sh   # requeued preemptions are invisible"
echo "         to a PREEMPTED query and silently overwrite their own early epochs."
