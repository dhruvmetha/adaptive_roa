#!/bin/bash
# Resume a preempted or timed-out Amarel cell from its last banked epoch, from the
# snapshot of the commit it was launched with (never from 0).
#   resume_amarel.sh <name> [walltime]
set -euo pipefail
export SSH_AUTH_SOCK=
HERE=/common/home/st1122/Projects/adaptive_roa/slurm_logs/det_rerun
EXP=/common/users/shared/pracsys/adaptive_roa_experiments/det_rerun
AM=amarel.rutgers.edu; AEXP=/home/st1122/det_rerun_out
NAME=$1; WALL=${2:-3-00:00:00}
LAST=$(awk -F'\t' -v n="$NAME" '$1==n' "$HERE/launched.tsv" | tail -1)
[ "$(cut -f2 <<< "$LAST")" = amarel ] || { echo "$NAME was not launched on Amarel" >&2; exit 1; }
COMMIT=$(cut -f7 <<< "$LAST")
ROW=$(awk -F'\t' -v n="$NAME" 'NR>1 && $1==n' "$HERE/manifest.tsv")
MEM=$(cut -f6 <<< "$ROW")
ADIR=$(grep -o 'output_dir=[^ ]*' <<< "$(cut -f8 <<< "$ROW")" | cut -d= -f2); ADIR=${ADIR/#$EXP/$AEXP}
ACODE=$(ssh -n -o BatchMode=yes $AM "ls -d /home/st1122/det_rerun_code/${COMMIT}*" | head -1)
[ -n "$ACODE" ] || { echo "no Amarel snapshot for $COMMIT" >&2; exit 1; }
ssh -n -o BatchMode=yes $AM "ls $ADIR/epoch_*/artifacts_v2.json" >/dev/null 2>&1 \
  || { echo "$NAME has no banked epoch on Amarel; launch it instead" >&2; exit 1; }
JID=$(ssh -n -o BatchMode=yes $AM \
  "cd $ACODE && sbatch --parsable -J $NAME --mem=$MEM --time=$WALL --export=ALL,ADAPTIVE_ROA_COMMIT=$COMMIT,DET_CODE=$ACODE,DET_SCRIPT=scripts/resume_adaptive.py scripts/det_rerun/sbatch_det_amarel.sh --run-dir $ADIR </dev/null" 2>/dev/null | grep -E '^[0-9]+$')
[ -n "$JID" ] || { echo "Amarel sbatch failed for $NAME" >&2; exit 1; }
printf '%s\tamarel\tgpu\t%s\t%s\t%s\t%s\n' "$NAME" "$JID" "$(date +%Y%m%d_%H%M)" "$HERE/amarel_logs/${NAME}_${JID}.out" "$COMMIT" >> "$HERE/launched.tsv"
echo "$NAME resumed -> Amarel job $JID (commit $COMMIT)"
