#!/bin/bash
# Resume an iLab cell that timed out or was preempted from its last banked epoch,
# from the read-only snapshot of the commit it was launched with (never from 0).
# A cell banked on Amarel (no live Amarel job) moves here too: its final Amarel dir is
# pulled into the iLab output dir and the .amarel_mirror marker dropped, as resume_direct.sh does.
#   resume_ilab.sh <name> [walltime] [gres_type]    (defaults: 06:00:00, the type it last ran on;
#                                                    gres_type is required for a cell from Amarel)
set -euo pipefail
export SSH_AUTH_SOCK=
HERE=/common/home/st1122/Projects/adaptive_roa/slurm_logs/det_rerun
WT=/common/home/st1122/Projects/adaptive_roa/.claude/worktrees/det-paper
NAME=$1; WALL=${2:-06:00:00}
EXP=/common/users/shared/pracsys/adaptive_roa_experiments/det_rerun
AM=amarel.rutgers.edu; AEXP=/scratch/st1122/det_rerun_out
LAST=$(awk -F'\t' -v n="$NAME" '$1==n || index($1, n"~")==1' "$HERE/launched.tsv" | tail -1)
ROW=$(awk -F'\t' -v n="$NAME" 'NR>1 && $1==n' "$HERE/manifest.tsv")
MEM=$(cut -f6 <<< "$ROW")
OUT=$(grep -o 'output_dir=[^ ]*' <<< "$(cut -f8 <<< "$ROW")" | cut -d= -f2)
if [ -f "$OUT/.amarel_mirror" ]; then
  [ -n "${3:-}" ] || { echo "$NAME comes from Amarel; pass the gres type" >&2; exit 1; }
  LIVE=$(ssh -n -o BatchMode=yes $AM "squeue -u st1122 -h -n $NAME -o %i" 2>/dev/null)
  [ -z "$LIVE" ] || { echo "$NAME still has Amarel job $LIVE; cancel it first" >&2; exit 1; }
  rc=0; rsync -a --delete --exclude .amarel_mirror "$AM:${OUT/#$EXP/$AEXP}/" "$OUT/" </dev/null || rc=$?
  [ $rc = 0 ] || { echo "final pull of $NAME from Amarel failed (rsync $rc)" >&2; exit 1; }
  rm "$OUT/.amarel_mirror"
else
  [ "$(cut -f2 <<< "$LAST")" = ilab ] || { echo "$NAME was not last launched on iLab" >&2; exit 1; }
fi
GRES=${3:-$(cut -f3 <<< "$LAST")}; COMMIT=$(cut -f7 <<< "$LAST")
ls "$OUT"/epoch_*/artifacts_v2.json >/dev/null 2>&1 || { echo "$NAME has no banked epoch; launch it instead" >&2; exit 1; }
CODE=$(bash "$WT/scripts/det_rerun/snapshot.sh" "$COMMIT")
JID=$(ssh -n -o BatchMode=yes ilab1.cs.rutgers.edu \
  "cd $CODE && sbatch --parsable -J $NAME --gres=gpu:$GRES:1 --mem=$MEM --time=$WALL --exclude=ilab3,ilab4 --export=ALL,ADAPTIVE_ROA_COMMIT=$COMMIT,DET_CODE=$CODE,DET_SCRIPT=scripts/resume_adaptive.py scripts/det_rerun/sbatch_det.sh --run-dir $OUT </dev/null" 2>/dev/null | grep -E '^[0-9]+$')
[ -n "$JID" ] || { echo "iLab sbatch failed for $NAME" >&2; exit 1; }
printf '%s\tilab\t%s\t%s\t%s\t%s\t%s\n' "$NAME" "$GRES" "$JID" "$(date +%Y%m%d_%H%M)" "$HERE/${NAME}_${JID}.out" "$COMMIT" >> "$HERE/launched.tsv"
echo "$NAME resumed -> iLab job $JID ($GRES, wall $WALL, commit $COMMIT)"
