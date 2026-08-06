#!/bin/bash
# Amarel's `general` account is preemptible. This reports BOTH ways a preemption
# shows up, because they need different responses and only one is visible in the
# obvious place.
#
#   1. PREEMPTED and gone. Submitted without --requeue: the job dies and does not
#      come back. Its output dir is left partial and must be cleared before
#      resubmission, or the new run's early epochs mix with the dead run's later
#      ones inside a single arm.
#
#   2. PREEMPTED and REQUEUED. Submitted with --requeue: SLURM puts it straight
#      back in the queue, so it appears as an ordinary PENDING job and `sacct`
#      reports the CURRENT attempt (PENDING, Elapsed 00:00:00) rather than the
#      preemption -- section 1 below will NOT list it. Nothing looks wrong.
#      This case is worse than case 1, because AdaptiveEngine.run has no resume
#      logic: it iterates `for epoch in range(n_epochs)` and overwrites epoch
#      dirs in place starting from 0. A requeued arm therefore rewrites its early
#      epochs with whatever code is on the cluster NOW while its later epochs keep
#      the previous code's output. On 2026-08-04 that nearly stitched
#      `clf_xhigh_total` across the member-seed fix (4c7c561), in an arm that was
#      part of an already-recorded verdict.
#      Response: scancel, delete the output dir on Amarel AND the local mirror,
#      then relaunch clean.
#
# The lookback is a fixed window, NOT `date +%Y-%m-%d`. Anchoring on today means
# the history silently empties at midnight: a job preempted at 23:50 disappears
# from the report at 00:00 and reads as "no preemptions" -- the same false-clean
# signal this script exists to prevent. (That happened on 2026-08-04: the count
# went 9 -> 0 purely from the date rolling over.)
DAYS=${1:-3}
SINCE=$(date -d "$DAYS days ago" +%Y-%m-%d)

echo "# 1. PREEMPTED since $SINCE (lookback ${DAYS}d) -- clear the output dir before resubmitting"
timeout 120 ssh amarel.rutgers.edu \
  "sacct -u st1122 -S $SINCE -o JobID,JobName%30,State,Start --parsable2 \
   | grep -v '\.batch\|\.extern' | grep PREEMPTED" 2>/dev/null | sed 's/|/ /g'

echo
echo "# 2. RESTART-IN-PLACE RISK: pending jobs that ALREADY have epochs on disk."
echo "#    These are requeued preemptions. They are invisible to section 1 and"
echo "#    will overwrite their own early epochs on restart. Treat as corruption:"
echo "#    scancel -> rm -rf output dir on BOTH sides -> relaunch."
timeout 150 ssh amarel.rutgers.edu \
  'E=/scratch/st1122/adaptive_roa/experiments/ensemble_epistemic
   found=0
   while read -r id name; do
     d="$E/${name#en_}"
     n=$(ls -d "$d"/epoch_* 2>/dev/null | wc -l)
     if [ "$n" -gt 0 ]; then
       echo "  RISK: $name (job $id) already has $n epochs on disk"
       found=1
     fi
   done < <(squeue -u st1122 -h -t PD -o "%i %j")
   [ "$found" -eq 0 ] && echo "  none"' 2>/dev/null
