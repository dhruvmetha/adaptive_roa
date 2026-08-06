#!/bin/bash
# Launch the flow-matching half of the ensemble-epistemic campaign on iLab.
#
# Two GPUs per arm, not five. The members train concurrently (mp.spawn), and a
# pendulum flow matcher only pushes an a4500 to ~25%, so 2-3 members share a card
# with little contention. Five arms therefore fit in 10 of the 12-GPU quota and all
# advance together -- which matters because the matched-epoch comparison is bounded
# by the SLOWEST arm, so uniform hardware buys more common depth than giving one arm
# faster cards. a4500 is requested explicitly for that uniformity: iLab's mixed pool
# (a4000/a5000/a6000/ada/a100) would otherwise scatter the arms across speeds.
#
# num_workers=1 (default 4) is not a tuning choice, it is a hard constraint.
# `ulimit -u` on the iLab nodes is 2000 -- and on Linux that limit counts THREADS,
# per user, per node, across all your jobs. One arm is 5 member processes, each
# with train+val DataLoader workers, each carrying several runtime threads: about
# 570 threads per arm measured. Four arms land on one 8-GPU node at 2 GPUs each and
# blow the limit, which surfaces as "can't start new thread" or
# "BlockingIOError: [Errno 11] Resource temporarily unavailable" 25-35 minutes in,
# and killed three arms at once. Capping OMP/MKL does NOT help: those shrink each
# process's pool, not the process count. num_workers=1 removes the multiplier.
set -u
R=/common/home/st1122/Projects/adaptive_roa
E=/common/users/shared/pracsys/adaptive_roa_experiments/ensemble_epistemic
LOG=$R/docs/experiments/ensemble_epistemic/runs.jsonl
SSH="ilab1.cs.rutgers.edu"

arm_overrides() {
  case $1 in
    total)    echo "acquisition=decomp_total" ;;
    epi_var)  echo "acquisition=decomp_epi_var" ;;
    epi_bald) echo "acquisition=decomp_epi_bald" ;;
    aleat)    echo "acquisition=decomp_aleat" ;;
    dir00)    echo "acquisition=direct acquisition.d2_ratio=0" ;;
  esac
}

LVL=${1:-high}
for arm in epi_var total dir00 epi_bald aleat; do
  RID="fm_${LVL}_${arm}"
  if [ -e "$E/$RID" ]; then
    echo "SKIP $RID: output dir exists -- delete it first, or the arm's curve gets"
    echo "     stitched from two runs."
    continue
  fi
  # 240s, not 120: iLab logins are slow enough that a 120s cap silently killed
  # every submission in this loop while the same sbatch typed by hand succeeded.
  JID=$(timeout 240 ssh "$SSH" "cd $R && sbatch --parsable -J en_${RID} \
    --gres=gpu:a4500:2 --mem=60G $R/scripts/sbatch_ilab.sh \
    system=pendulum_stoch noise_level=$LVL $(arm_overrides $arm) \
    predictor=fm_ensemble seed=42 num_workers=1 \
    predictor.lightning_trainer.enable_progress_bar=false \
    output_dir=$E/$RID" 2>/dev/null | tr -dc '0-9')
  if [ -z "$JID" ]; then echo "FAILED to submit $RID"; continue; fi
  "$R/env/bin/python" "$R/scripts/exp_log.py" append --log "$LOG" \
    --run-id "$RID" --system pendulum_stoch --level "$LVL" --predictor fm \
    --arm "$arm" --cluster ilab --job-id "$JID" --output-dir "$E/$RID" \
    --score-mode "$arm" --n-members 5 \
    --d2-ratio "$( [ "$arm" = dir00 ] && echo 0.0 || echo 1.0 )" >/dev/null
  echo "$RID -> $JID"
done
