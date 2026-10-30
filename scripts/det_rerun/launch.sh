#!/bin/bash
# Launch one det-rerun manifest row onto a placement from a fresh gpu-free.sh scan.
#   launch.sh <name> ilab   <gres_type> [walltime]
#   launch.sh <name> direct <host> <gpu_index>
#   launch.sh <name> ilabcpu [walltime]            (Part-X: no GPU, no quota)
# Refuses a name already in launched.tsv or whose output dir has a banked epoch
# (resume those with scripts/resume_adaptive.py instead). Pins the worktree commit.
set -euo pipefail
export SSH_AUTH_SOCK=                 # westeros's ssh-agent hangs key auth
WT=/common/home/st1122/Projects/adaptive_roa/.claude/worktrees/det-paper
HERE=/common/home/st1122/Projects/adaptive_roa/slurm_logs/det_rerun
NAME=$1; WORLD=$2
ROW=$(awk -F'\t' -v n="$NAME" 'NR>1 && $1==n' "$HERE/manifest.tsv")
[ -n "$ROW" ] || { echo "no manifest row for $NAME" >&2; exit 1; }
MEM=$(cut -f6 <<< "$ROW"); OV=$(cut -f8 <<< "$ROW"); SCRIPT=$(cut -f9 <<< "$ROW")
OUTDIR=$(grep -o 'output_dir=[^ ]*' <<< "$OV" | cut -d= -f2)
if ls "$OUTDIR"/epoch_*/artifacts_v2.json >/dev/null 2>&1; then
  echo "$NAME already has banked epochs in $OUTDIR; resume it instead" >&2; exit 1
fi
if grep -qP "^$NAME\t" "$HERE/launched.tsv" 2>/dev/null; then
  echo "$NAME is already in launched.tsv; check it before launching again" >&2; exit 1
fi
cd "$WT"
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "worktree has uncommitted changes; commit before launching" >&2; exit 1
fi
COMMIT=$(git rev-parse --short HEAD)
TS=$(date +%Y%m%d_%H%M)
case "$WORLD" in
ilabcpu)
  WALL=${3:-7-00:00:00}
  JID=$(ssh -n -o BatchMode=yes ilab1.cs.rutgers.edu \
    "cd $WT && sbatch --parsable -J $NAME --mem=$MEM --time=$WALL --exclude=ilab3,ilab4 --export=ALL,ADAPTIVE_ROA_COMMIT=$COMMIT,DET_SCRIPT=$SCRIPT,OMP_NUM_THREADS=8 scripts/det_rerun/sbatch_det.sh $OV </dev/null" 2>/dev/null | grep -E '^[0-9]+$')
  [ -n "$JID" ] || { echo "sbatch failed for $NAME" >&2; exit 1; }
  LOG=$HERE/${NAME}_${JID}.out
  printf '%s\tilabcpu\tcpu\t%s\t%s\t%s\t%s\n' "$NAME" "$JID" "$TS" "$LOG" "$COMMIT" >> "$HERE/launched.tsv"
  echo "$NAME -> iLab CPU job $JID (mem $MEM, wall $WALL, commit $COMMIT)";;
ilab)
  GRES=$3; WALL=${4:-3-00:00:00}
  JID=$(ssh -n -o BatchMode=yes ilab1.cs.rutgers.edu \
    "cd $WT && ADAPTIVE_ROA_COMMIT=$COMMIT sbatch --parsable -J $NAME --gres=gpu:$GRES:1 --mem=$MEM --time=$WALL --exclude=ilab3,ilab4 --export=ALL,ADAPTIVE_ROA_COMMIT=$COMMIT,DET_SCRIPT=$SCRIPT scripts/det_rerun/sbatch_det.sh $OV </dev/null" 2>/dev/null | grep -E '^[0-9]+$')
  [ -n "$JID" ] || { echo "sbatch failed for $NAME" >&2; exit 1; }
  LOG=$HERE/${NAME}_${JID}.out
  printf '%s\tilab\t%s\t%s\t%s\t%s\t%s\n' "$NAME" "$GRES" "$JID" "$TS" "$LOG" "$COMMIT" >> "$HERE/launched.tsv"
  echo "$NAME -> iLab job $JID ($GRES, mem $MEM, wall $WALL, commit $COMMIT)";;
direct)
  HOST=$3; GPU=$4; LOG=$HERE/${TS}_${NAME}.log; SESS=gpu_${NAME}
  CORES=""; [ "$HOST" = "1spring-303-desktop-10" ] && CORES="taskset -c 0-9"
  ENVV="PYTHONPATH=$WT PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TMPDIR=/tmp/st1122_$NAME ADAPTIVE_ROA_COMMIT=$COMMIT"
  CMD="cd $WT && mkdir -p /tmp/st1122_$NAME && export $ENVV && CUDA_VISIBLE_DEVICES=$GPU $CORES /common/home/st1122/Projects/adaptive_roa/env/bin/python $SCRIPT $OV > $LOG 2>&1"
  if [ "$(hostname -s)" = "$HOST" ]; then tmux new-session -d -s "$SESS" "$CMD"
  else ssh -n -o BatchMode=yes "$HOST.cs.rutgers.edu" "tmux new-session -d -s '$SESS' \"$CMD\"" 2>/dev/null; fi
  sleep 3
  if [ "$(hostname -s)" = "$HOST" ]; then PID=$(tmux list-panes -t "$SESS" -F '#{pane_pid}')
  else PID=$(ssh -n -o BatchMode=yes "$HOST.cs.rutgers.edu" "tmux list-panes -t '$SESS' -F '#{pane_pid}'" 2>/dev/null | grep -E '^[0-9]+$'); fi
  [ -n "$PID" ] || { echo "tmux launch failed for $NAME on $HOST" >&2; exit 1; }
  printf '%s\t%s\tgpu%s\t%s\t%s\t%s\t%s\n' "$HOST" "$PID" "$GPU" "$NAME" "$LOG" "$TS" "$SCRIPT $OV" >> /common/home/st1122/Projects/adaptive_roa/slurm_logs/direct_jobs.tsv
  printf '%s\tdirect\t%s:%s\t%s\t%s\t%s\t%s\n' "$NAME" "$HOST" "$GPU" "$PID" "$TS" "$LOG" "$COMMIT" >> "$HERE/launched.tsv"
  echo "$NAME -> $HOST gpu$GPU pid $PID session $SESS commit $COMMIT";;
*) echo "world must be ilab or direct" >&2; exit 1;;
esac
