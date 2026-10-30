#!/bin/bash
# Resume an iLab cell that timed out or was preempted from its last banked epoch,
# from the read-only snapshot of the commit it was launched with (never from 0).
#   resume_ilab.sh <name> [walltime] [gres_type]    (defaults: 06:00:00, the type it last ran on)
set -euo pipefail
export SSH_AUTH_SOCK=
HERE=/common/home/st1122/Projects/adaptive_roa/slurm_logs/det_rerun
WT=/common/home/st1122/Projects/adaptive_roa/.claude/worktrees/det-paper
NAME=$1; WALL=${2:-06:00:00}
LAST=$(awk -F'\t' -v n="$NAME" '$1==n' "$HERE/launched.tsv" | tail -1)
[ "$(cut -f2 <<< "$LAST")" = ilab ] || { echo "$NAME was not last launched on iLab" >&2; exit 1; }
GRES=${3:-$(cut -f3 <<< "$LAST")}; COMMIT=$(cut -f7 <<< "$LAST")
ROW=$(awk -F'\t' -v n="$NAME" 'NR>1 && $1==n' "$HERE/manifest.tsv")
MEM=$(cut -f6 <<< "$ROW")
OUT=$(grep -o 'output_dir=[^ ]*' <<< "$(cut -f8 <<< "$ROW")" | cut -d= -f2)
ls "$OUT"/epoch_*/artifacts_v2.json >/dev/null 2>&1 || { echo "$NAME has no banked epoch; launch it instead" >&2; exit 1; }
CODE=$(bash "$WT/scripts/det_rerun/snapshot.sh" "$COMMIT")
JID=$(ssh -n -o BatchMode=yes ilab1.cs.rutgers.edu \
  "cd $CODE && sbatch --parsable -J $NAME --gres=gpu:$GRES:1 --mem=$MEM --time=$WALL --exclude=ilab3,ilab4 --export=ALL,ADAPTIVE_ROA_COMMIT=$COMMIT,DET_CODE=$CODE,DET_SCRIPT=scripts/resume_adaptive.py scripts/det_rerun/sbatch_det.sh --run-dir $OUT </dev/null" 2>/dev/null | grep -E '^[0-9]+$')
[ -n "$JID" ] || { echo "iLab sbatch failed for $NAME" >&2; exit 1; }
printf '%s\tilab\t%s\t%s\t%s\t%s\t%s\n' "$NAME" "$GRES" "$JID" "$(date +%Y%m%d_%H%M)" "$HERE/${NAME}_${JID}.out" "$COMMIT" >> "$HERE/launched.tsv"
echo "$NAME resumed -> iLab job $JID ($GRES, wall $WALL, commit $COMMIT)"
