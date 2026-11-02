#!/bin/bash
# Resume a preempted or timed-out Amarel GPU cell from its last banked epoch, from the
# snapshot of the commit it was launched with (never from 0). Part-X (amarelcpu) resumes
# by rerunning its command on the same output dir, from partx_state.pkl.
# A copied cell (extend_cells.py) is pushed to Amarel and marked as a mirror on its first run.
#   resume_amarel.sh <name> [walltime]
set -euo pipefail
export SSH_AUTH_SOCK=
HERE=/common/home/st1122/Projects/adaptive_roa/slurm_logs/det_rerun
EXP=/common/users/shared/pracsys/adaptive_roa_experiments/det_rerun
AM=amarel.rutgers.edu; AEXP=/scratch/st1122/det_rerun_out
NAME=$1; WALL=${2:-3-00:00:00}
LAST=$(awk -F'\t' -v n="$NAME" '$1==n' "$HERE/launched.tsv" | tail -1)
WORLD=$(cut -f2 <<< "$LAST")
[ "$WORLD" = amarel ] || [ "$WORLD" = copied ] || { echo "$NAME was not launched on Amarel" >&2; exit 1; }
COMMIT=$(cut -f7 <<< "$LAST")
ROW=$(awk -F'\t' -v n="$NAME" 'NR>1 && $1==n' "$HERE/manifest.tsv")
MEM=$(cut -f6 <<< "$ROW")
ADIR=$(grep -o 'output_dir=[^ ]*' <<< "$(cut -f8 <<< "$ROW")" | cut -d= -f2); ADIR=${ADIR/#$EXP/$AEXP}
if [ "$WORLD" = copied ]; then
  # extend_cells.py copy: its first run. Push the copy, then mirror it back like any Amarel cell.
  OUT=$(grep -o 'output_dir=[^ ]*' <<< "$(cut -f8 <<< "$ROW")" | cut -d= -f2)
  ! ssh -n -o BatchMode=yes $AM "test -e $ADIR" || { echo "$ADIR already exists on Amarel" >&2; exit 1; }
  ssh -n -o BatchMode=yes $AM "mkdir -p $(dirname "$ADIR")"
  rsync -a --exclude .amarel_mirror "$OUT/" "$AM:$ADIR/" </dev/null
  touch "$OUT/.amarel_mirror"
fi
ACODE=$(ssh -n -o BatchMode=yes $AM "ls -d /home/st1122/det_rerun_code/${COMMIT}*" | head -1)
[ -n "$ACODE" ] || { echo "no Amarel snapshot for $COMMIT" >&2; exit 1; }
if [ "$(cut -f9 <<< "$ROW")" = scripts/run_partx_faithful.py ]; then
  # Part-X resumes from partx_state.pkl when its own command reruns on the same output dir.
  ssh -n -o BatchMode=yes $AM "test -f $ADIR/partx_state.pkl" \
    || { echo "$NAME has no partx_state.pkl on Amarel; launch it instead" >&2; exit 1; }
  AOV=$(cut -f8 <<< "$ROW"); AOV=${AOV//$EXP/$AEXP}
  JID=$(ssh -n -o BatchMode=yes $AM \
    "cd $ACODE && sbatch --parsable -J $NAME --mem=$MEM --time=$WALL --partition=main --cpus-per-task=8 --export=ALL,OMP_NUM_THREADS=8,ADAPTIVE_ROA_COMMIT=$COMMIT,DET_CODE=$ACODE,DET_SCRIPT=scripts/run_partx_faithful.py scripts/det_rerun/sbatch_det_amarel.sh $AOV </dev/null" 2>/dev/null | grep -E '^[0-9]+$')
  [ -n "$JID" ] || { echo "Amarel sbatch failed for $NAME" >&2; exit 1; }
  printf '%s\tamarel\t--partition=main\t%s\t%s\t%s\t%s\n' "$NAME" "$JID" "$(date +%Y%m%d_%H%M)" "$HERE/amarel_logs/${NAME}_${JID}.out" "$COMMIT" >> "$HERE/launched.tsv"
  echo "$NAME resumed -> Amarel CPU job $JID (commit $COMMIT)"; exit 0
fi
ssh -n -o BatchMode=yes $AM "ls $ADIR/epoch_*/artifacts_v2.json" >/dev/null 2>&1 \
  || { echo "$NAME has no banked epoch on Amarel; launch it instead" >&2; exit 1; }
JID=$(ssh -n -o BatchMode=yes $AM \
  "cd $ACODE && sbatch --parsable -J $NAME --mem=$MEM --time=$WALL --partition=gpu --gres=gpu:1 --cpus-per-task=4 --exclude=gpu029,gpu046 --export=ALL,OMP_NUM_THREADS=4,ADAPTIVE_ROA_COMMIT=$COMMIT,DET_CODE=$ACODE,DET_SCRIPT=scripts/resume_adaptive.py scripts/det_rerun/sbatch_det_amarel.sh --run-dir $ADIR </dev/null" 2>/dev/null | grep -E '^[0-9]+$')
[ -n "$JID" ] || { echo "Amarel sbatch failed for $NAME" >&2; exit 1; }
printf '%s\tamarel\t--partition=gpu\t%s\t%s\t%s\t%s\n' "$NAME" "$JID" "$(date +%Y%m%d_%H%M)" "$HERE/amarel_logs/${NAME}_${JID}.out" "$COMMIT" >> "$HERE/launched.tsv"
echo "$NAME resumed -> Amarel job $JID (commit $COMMIT)"
