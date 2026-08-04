#!/bin/bash
# Launch the classifier half of the ensemble-epistemic campaign on Amarel.
# One GPU per arm: the 5 ensemble members train sequentially (an MLP is cheap),
# so giving each member its own card would waste 4 of them. Flow matching gets
# the one-member-per-GPU treatment later, where parallelism actually buys
# wall-clock.
set -u
REPO=/home/st1122/Projects/adaptive_roa
EXP=/scratch/st1122/adaptive_roa/experiments/ensemble_epistemic
LOG=/common/home/st1122/Projects/adaptive_roa/docs/experiments/ensemble_epistemic/runs.jsonl
LOCAL=/common/home/st1122/Projects/adaptive_roa

arm_overrides() {
  case $1 in
    total)    echo "acquisition=decomp_total" ;;
    epi_var)  echo "acquisition=decomp_epi_var" ;;
    epi_bald) echo "acquisition=decomp_epi_bald" ;;
    aleat)    echo "acquisition=decomp_aleat" ;;
    dir00)    echo "acquisition=direct acquisition.d2_ratio=0" ;;
  esac
}

sys_overrides() {   # deterministic pendulum has no noise_level knob
  case $1 in
    det) echo "system=pendulum" ;;
    *)   echo "system=pendulum_stoch noise_level=$1" ;;
  esac
}

for lvl in high xhigh med low det; do
  for arm in dir00 total epi_var epi_bald aleat; do
    RID="clf_${lvl}_${arm}"
    JID=$(timeout 90 ssh amarel.rutgers.edu "cd $REPO && sbatch --parsable --requeue \
      -J en_${RID} --time=1-00:00:00 scripts/sbatch_amarel.sh \
      $(sys_overrides $lvl) $(arm_overrides $arm) predictor=clf_ensemble \
      seed=42 output_dir=$EXP/${RID}" 2>/dev/null | tail -1)
    if [ -z "$JID" ]; then echo "FAILED to submit $RID"; continue; fi
    "$LOCAL/env/bin/python" "$LOCAL/scripts/exp_log.py" append --log "$LOG" \
      --run-id "$RID" --system "$( [ "$lvl" = det ] && echo pendulum || echo pendulum_stoch )" \
      --level "$lvl" --predictor clf --arm "$arm" --cluster amarel --job-id "$JID" \
      --output-dir "$EXP/${RID}" --score-mode "$arm" --n-members 5 \
      --d2-ratio "$( [ "$arm" = dir00 ] && echo 0.0 || echo 1.0 )" >/dev/null
    echo "$RID -> $JID"
  done
done
