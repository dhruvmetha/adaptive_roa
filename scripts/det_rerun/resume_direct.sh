#!/bin/bash
# Resume a cell from its last banked epoch on a direct-box card, from the snapshot of the
# commit it was launched with (never from 0). A cell banked on Amarel must have no live
# Amarel job: its final Amarel dir is pulled into the iLab output dir and the
# .amarel_mirror marker dropped, so sync_amarel.sh never overwrites the local run.
# Paths resolve on this side (data_dir from the snapshot's .env, output_dir from --run-dir).
#   resume_direct.sh <name> <host> <gpu_index>
set -euo pipefail
export SSH_AUTH_SOCK=
HERE=/common/home/st1122/Projects/adaptive_roa/slurm_logs/det_rerun
WT=/common/home/st1122/Projects/adaptive_roa/.claude/worktrees/det-paper
EXP=/common/users/shared/pracsys/adaptive_roa_experiments/det_rerun
AM=amarel.rutgers.edu; AEXP=/scratch/st1122/det_rerun_out
NAME=$1; HOST=$2; GPU=$3
LAST=$(awk -F'\t' -v n="$NAME" '$1==n || index($1, n"~")==1' "$HERE/launched.tsv" | tail -1)
[ -n "$LAST" ] || { echo "$NAME has no registry row" >&2; exit 1; }
COMMIT=$(cut -f7 <<< "$LAST")
ROW=$(awk -F'\t' -v n="$NAME" 'NR>1 && $1==n' "$HERE/manifest.tsv")
OUT=$(grep -o 'output_dir=[^ ]*' <<< "$(cut -f8 <<< "$ROW")" | cut -d= -f2)
if [ -f "$OUT/.amarel_mirror" ]; then
  LIVE=$(ssh -n -o BatchMode=yes $AM "squeue -u st1122 -h -n $NAME -o %i" 2>/dev/null)
  [ -z "$LIVE" ] || { echo "$NAME still has Amarel job $LIVE; cancel it first" >&2; exit 1; }
  rc=0; rsync -a --delete --exclude .amarel_mirror "$AM:${OUT/#$EXP/$AEXP}/" "$OUT/" </dev/null || rc=$?
  [ $rc = 0 ] || { echo "final pull of $NAME from Amarel failed (rsync $rc)" >&2; exit 1; }
  rm "$OUT/.amarel_mirror"
fi
ls "$OUT"/epoch_*/artifacts_v2.json >/dev/null 2>&1 || { echo "$NAME has no banked epoch; launch it instead" >&2; exit 1; }
CODE=$(bash "$WT/scripts/det_rerun/snapshot.sh" "$COMMIT")
TS=$(date +%Y%m%d_%H%M); LOG=$HERE/${TS}_${NAME}_resume.log; SESS=gpu_${NAME}
CORES=""; [ "$HOST" = "1spring-303-desktop-10" ] && CORES="taskset -c 0-9"
ENVV="PYTHONPATH=$CODE PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TMPDIR=/tmp/st1122_$NAME ADAPTIVE_ROA_COMMIT=$COMMIT"
CMD="cd $CODE && mkdir -p /tmp/st1122_$NAME && export $ENVV && CUDA_VISIBLE_DEVICES=$GPU $CORES /common/home/st1122/Projects/adaptive_roa/env/bin/python scripts/resume_adaptive.py --run-dir $OUT > $LOG 2>&1"
if [ "$(hostname -s)" = "$HOST" ]; then tmux new-session -d -s "$SESS" "$CMD"
else ssh -n -o BatchMode=yes "$HOST.cs.rutgers.edu" "tmux new-session -d -s '$SESS' \"$CMD\"" 2>/dev/null; fi
sleep 3
if [ "$(hostname -s)" = "$HOST" ]; then PID=$(tmux list-panes -t "$SESS" -F '#{pane_pid}')
else PID=$(ssh -n -o BatchMode=yes "$HOST.cs.rutgers.edu" "tmux list-panes -t '$SESS' -F '#{pane_pid}'" 2>/dev/null | grep -E '^[0-9]+$'); fi
[ -n "$PID" ] || { echo "tmux resume failed for $NAME on $HOST" >&2; exit 1; }
printf '%s\t%s\tgpu%s\t%s\t%s\t%s\t%s\n' "$HOST" "$PID" "$GPU" "$NAME" "$LOG" "$TS" "scripts/resume_adaptive.py --run-dir $OUT" >> /common/home/st1122/Projects/adaptive_roa/slurm_logs/direct_jobs.tsv
printf '%s\tdirect\t%s:%s\t%s\t%s\t%s\t%s\n' "$NAME" "$HOST" "$GPU" "$PID" "$TS" "$LOG" "$COMMIT" >> "$HERE/launched.tsv"
echo "$NAME resumed -> $HOST gpu$GPU pid $PID session $SESS commit $COMMIT"
