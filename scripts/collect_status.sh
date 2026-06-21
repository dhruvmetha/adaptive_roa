#!/usr/bin/env bash
# Overnight status collector for the classifier-vs-FM study.
# Parses slurm_logs/<jobname>_<jobid>.out for per-epoch ROA metrics, tagged by job name.
cd /common/home/dm1487/robotics_research/tripods/olympics-classifier
echo "================ STATUS $(date '+%H:%M:%S') ================"
echo "--- queue ---"
squeue -u "$USER" -o "%.8i %.12j %.8T %.10M %.14R" 2>/dev/null | tail -n +1
echo "--- per-job last metric (clf_* and fm_*) ---"
for name in clf_pend_a clf_pend_r clf_cp_a clf_cp_r clf_q2d_a clf_q2d_r clf_q3d_a clf_q3d_r \
            fm_pend_a fm_pend_r fm_cp_a fm_cp_r fm_q2d_a fm_q2d_r; do
  f=$(ls -t slurm_logs/${name}_*.out 2>/dev/null | head -1)
  if [ -z "$f" ]; then echo "  $name : (no log)"; continue; fi
  nev=$(grep -cE "λ±δ|F1=" "$f" 2>/dev/null)
  done=$(grep -c "^Done:" "$f" 2>/dev/null)
  err=$(grep -cE "Traceback \(most recent" "$f" 2>/dev/null)
  last=$(grep -E "F1=" "$f" 2>/dev/null | tail -1 | sed 's/^[[:space:]]*//' | cut -c1-90)
  echo "  $name : epochs_eval=$nev done=$done tracebacks=$err | $last"
done