#!/usr/bin/env bash
# Overnight classifier data-efficiency sweep: 4 systems x {adaptive ranked d2=1.0, non-adaptive d2=0.0}.
# Sequential (one A100) to avoid OOM/contention. Continues on failure so one crash != lost night.
# Schedules (init -> max, increment):  pendulum 50-500/50 | cartpole 300-1000/100 | quad2d 3000-12000/1000 | quad3d 10000-100000/10000
set -u
PY=/common/users/dm1487/envs/arcmg/bin/python
cd /common/home/dm1487/robotics_research/tripods/olympics-classifier
DATA=/common/users/shared/pracsys/genMoPlan/data_trajectories
LOG=/tmp/overnight_$(date +%H%M%S 2>/dev/null || echo run)
mkdir -p "$LOG"
SUMMARY="$LOG/SUMMARY.txt"
echo "started batch" > "$SUMMARY"

run () {  # name  system  init  incr  nepochs  d2  extra...
  local name=$1 system=$2 init=$3 incr=$4 nep=$5 d2=$6; shift 6
  local mode="ranked"; [ "$d2" = "0.0" ] && mode="direct"  # mode irrelevant when d2=0
  echo ">>> $name : system=$system init=$init incr=$incr n_epochs=$nep d2=$d2 mode=$mode" | tee -a "$SUMMARY"
  $PY scripts/run_adaptive.py system=$system predictor=classifier \
    sampling_mode=$mode d2_ratio=$d2 \
    initial_train_size=$init samples_per_epoch=$incr n_epochs=$nep \
    classifier.max_epochs=80 classifier.patience=15 \
    device=cuda:0 eval_every=1 "$@" > "$LOG/$name.log" 2>&1
  local ec=$?
  local nev=$(grep -c 'λ±δ:' "$LOG/$name.log" 2>/dev/null)
  echo "    $name EXIT=$ec evals=$nev last=[$(grep 'λ±δ:' "$LOG/$name.log" 2>/dev/null | tail -1 | sed 's/^ *//')]" | tee -a "$SUMMARY"
}

Q3D_POOL="data_source.shuffled_indices_file=$DATA/quadrotor3D_lqr/train_test_splits/all_shuffled_indices.txt data_source.shuffled_labels_file=$DATA/quadrotor3D_lqr/train_test_splits/all_shuffled_labels.txt"

# --- pendulum: 50 -> 500 step 50 (n_epochs=10) ---
run pendulum_adaptive     pendulum          50    50    10  1.0
run pendulum_nonadaptive  pendulum          50    50    10  0.0
# --- cartpole: 300 -> 1000 step 100 (n_epochs=8) ---
run cartpole_adaptive     cartpole_pybullet 300   100   8   1.0
run cartpole_nonadaptive  cartpole_pybullet 300   100   8   0.0
# --- quad2d: 3000 -> 12000 step 1000 (n_epochs=10) ---
run quad2d_adaptive       quadrotor2d       3000  1000  10  1.0
run quad2d_nonadaptive    quadrotor2d       3000  1000  10  0.0
# --- quad3d: 10000 -> 100000 step 10000 (n_epochs=10), full 800k pool ---
run quad3d_adaptive       quadrotor3d       10000 10000 10  1.0  $Q3D_POOL
run quad3d_nonadaptive    quadrotor3d       10000 10000 10  0.0  $Q3D_POOL

echo "BATCH COMPLETE" | tee -a "$SUMMARY"
echo "logs in $LOG"
