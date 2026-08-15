#!/bin/bash
# Emit one line whenever the FM member-recompute fleet changes state, and one
# final line when it drains. Failure signatures are included deliberately: a
# monitor that only greps for progress stays silent through a crash, and silence
# is indistinguishable from "still running".
MEMBERS=/common/users/shared/pracsys/adaptive_roa_experiments/_uncertainty_maps/members
LOGS=/common/home/st1122/Projects/adaptive_roa/slurm_logs
prev=""
while true; do
  running=$(squeue -u st1122 -h -o "%j" 2>/dev/null | grep -c "^um_")
  files=$(ls "$MEMBERS"/fm_*/members_epoch_*.npz 2>/dev/null | wc -l)
  fails=$(grep -lE "Traceback|CUDA out of memory|Killed|slurmstepd" "$LOGS"/um_fm_*.err 2>/dev/null | wc -l)
  cur="fm member jobs: running=$running npz_written=$files logs_with_errors=$fails"
  if [ "$cur" != "$prev" ]; then
    echo "$cur"
    prev="$cur"
  fi
  if [ "$running" -eq 0 ]; then
    echo "ALL FM MEMBER JOBS FINISHED: npz_written=$files logs_with_errors=$fails"
    break
  fi
  sleep 120
done
