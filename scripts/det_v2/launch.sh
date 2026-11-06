#!/bin/bash
# Launch one det_v2 manifest row.
#   launch.sh <name> ilab <gres_type> [walltime]     iLab GPU            (default wall 3-00:00:00)
#   launch.sh <name> direct <host> <gpu>             direct box card, in tmux session gpu_<name>
#   launch.sh <name> amarel [walltime]               Amarel gpu partition (default 3-00:00:00)
#   launch.sh <name> ilabcpu [walltime]              Part-X rows, iLab CPU (default 7-00:00:00);
#                                                    rerun it to continue from partx_state.pkl
# Runs from a read-only snapshot of the repo's HEAD (DET_V2_REF=<ref> picks another commit),
# with the run dir the commit's config sets (run_dir.py). Refuses a name whose run dir
# already has a banked epoch (resume it with resume_*.sh) and a name another launch
# already claimed in the shared ledger (DET_V2_RELAUNCH=1 claims over it, for a run that
# died with nothing banked). Settings come from user.env (common.sh).
set -euo pipefail
source "$(dirname "$0")/common.sh"
[ $# -ge 2 ] || die "usage: launch.sh <name> ilab <gres> [wall] | direct <host> <gpu> | amarel [wall] | ilabcpu [wall]"
NAME=$1; WORLD=$2
CONFIG=$(manifest_get "$NAME" config); SCRIPT=$(manifest_get "$NAME" script); MEM=$(manifest_get "$NAME" mem)
[ -n "$CONFIG" ] || die "no manifest row for $NAME in $MANIFEST"
PARTX=0; [ "$SCRIPT" = scripts/run_partx_faithful.py ] && PARTX=1
case "$WORLD" in
  ilab) GRES=${3:?launch.sh $NAME ilab <gres_type> [walltime]}; WALL=${4:-3-00:00:00}; HOSTGPU=$GRES ;;
  direct) HOST=${3:?launch.sh $NAME direct <host> <gpu>}; GPU=${4:?launch.sh $NAME direct <host> <gpu>}; HOSTGPU=$HOST:$GPU ;;
  amarel) WALL=${3:-3-00:00:00}; HOSTGPU=$AMAREL_PARTITION; need_amarel ;;
  ilabcpu) WALL=${3:-7-00:00:00}; HOSTGPU=cpu ;;
  *) die "where must be ilab, direct, amarel or ilabcpu" ;;
esac
if [ $PARTX = 1 ] && [ "$WORLD" != ilabcpu ]; then die "$NAME is a Part-X (CPU) row: launch it with ilabcpu"; fi
if [ $PARTX = 0 ] && [ "$WORLD" = ilabcpu ]; then die "$NAME needs a GPU: ilabcpu is for the Part-X rows"; fi

cd "$DET_V2_REPO"
if [ -z "${DET_V2_REF:-}" ] && [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  die "$DET_V2_REPO has uncommitted changes; commit before launching (or set DET_V2_REF)"
fi
CODE=$(bash "$DET_V2_SCRIPTS/snapshot.sh" "${DET_V2_REF:-HEAD}")
COMMIT=$(basename "$CODE")
RUN_DIR=$(run_dir_of "$NAME" "$CODE") || die "could not compose $CONFIG at $COMMIT"
under_root "$RUN_DIR" || die "$NAME: the config's output_dir $RUN_DIR is not under $DET_V2_ROOT"

LAST=$(ledger_last_placed "$NAME")
[ -z "$LAST" ] || refuse_if_live "$NAME" "$LAST"
if [ $PARTX = 1 ]; then
  # Part-X continues from partx_state.pkl when its command reruns on the same dir.
  # Rerunning claims over a finished placement, never over another account's in-flight claim.
  [ -z "$LAST" ] || export DET_V2_RERUN_PLACED=1
  [ ! -f "$RUN_DIR/partx_state.pkl" ] || echo "$NAME continues from $RUN_DIR/partx_state.pkl"
elif [ "$(banked "$RUN_DIR")" -gt 0 ]; then
  die "$NAME already has $(banked "$RUN_DIR") banked epoch(s) in $RUN_DIR; resume it instead"
fi
if [ "$WORLD" = amarel ]; then
  ARUN=$(amarel_dir "$RUN_DIR")
  if rsh "$AMAREL_HOST" "ls $ARUN/epoch_*/artifacts_v2.json" >/dev/null 2>&1; then
    die "$NAME already has banked epochs on Amarel ($ARUN); resume it with resume_amarel.sh"
  fi
  FULL=$(git rev-parse "$COMMIT")
  on_origin "$FULL" || die "$COMMIT is not on origin; push it first (Amarel fetches it from there)"
  # The data must be staged on Amarel's scratch: refuse a missing file instead of a crash in the job.
  if [ "${DET_V2_SKIP_DATA_CHECK:-0}" != 1 ]; then
    LIST=$(PYTHONPATH="$CODE" PYTHONNOUSERSITE=1 "$PYTHON" "$DET_V2_SCRIPTS/preflight.py" "$NAME" --list --code "$CODE") \
      || die "could not list the data files of $NAME"
    FILES=$(awk -F'\t' '$1=="f" || $1=="d" {print $2}' <<< "$LIST" | sed -e "s#^$DATA_DIR#$AMAREL_DATA_DIR#" -e "s#^$SHARED_DATA_BASE#$AMAREL_DATA_DIR#" | tr '\n' ' ')
    MISSING=$(rsh "$AMAREL_HOST" "for f in $FILES; do test -e \$f || echo \$f; done" 2>/dev/null || true)
    [ -z "$MISSING" ] || die "$NAME: not staged on Amarel (rsync them from $DATA_DIR first):
$MISSING"
  fi
fi

ledger_claim "$NAME" "$WORLD" "$HOSTGPU" "$COMMIT" "$RUN_DIR" || die "not launching $NAME"
abandon() { ledger_append "$NAME" "$NET_ID" "$WORLD" - "$HOSTGPU" "$(stamp)" - "$COMMIT" "$RUN_DIR" abandon; die "$1"; }
ARGS="$SCRIPT $CONFIG output_dir=$RUN_DIR"
# Nothing is banked here, so a marker from an earlier Amarel attempt is stale; left in place,
# a later resume would pull the empty scratch dir over this run.
[ "$WORLD" = amarel ] || rm -f "$RUN_DIR/.amarel_mirror"
case "$WORLD" in
ilab|ilabcpu)
  RES=cpu; [ "$WORLD" = ilab ] && RES=$GRES
  JID=$(ilab_submit "$NAME" "$CODE" "$COMMIT" "$MEM" "$WALL" "$RES" $ARGS) || true
  [ -n "$JID" ] || abandon "sbatch failed for $NAME"
  LOG=$LOG_DIR/${NAME}_${JID}.out
  ledger_append "$NAME" "$NET_ID" "$WORLD" "$JID" "$HOSTGPU" "$(stamp)" "$LOG" "$COMMIT" "$RUN_DIR" launch
  KIND="iLab job"; [ "$WORLD" = ilabcpu ] && KIND="iLab CPU job"
  echo "$NAME -> $KIND $JID ($HOSTGPU, mem $MEM, wall $WALL, commit $COMMIT, run dir $RUN_DIR)";;
direct)
  mkdir -p "$LOG_DIR"
  LOG=$LOG_DIR/$(date +%Y%m%d_%H%M)_${NAME}.log
  PID=$(start_direct "$NAME" "$HOST" "$GPU" "$CODE" "$COMMIT" "$LOG" $ARGS) || abandon "tmux launch failed for $NAME on $HOST"
  ledger_append "$NAME" "$NET_ID" direct "$PID" "$HOSTGPU" "$(stamp)" "$LOG" "$COMMIT" "$RUN_DIR" launch
  echo "$NAME -> $HOST gpu$GPU pid $PID session gpu_$NAME (commit $COMMIT, run dir $RUN_DIR)";;
amarel)
  ACODE=$(amarel_snapshot "$FULL") || true
  [ -n "$ACODE" ] || abandon "Amarel snapshot of $COMMIT failed"
  JID=$(amarel_submit "$NAME" "$ACODE" "$COMMIT" "$MEM" "$WALL" "$SCRIPT" "$CONFIG" "output_dir=$ARUN") || true
  [ -n "$JID" ] || abandon "Amarel sbatch failed for $NAME"
  LOG=$LOG_DIR/amarel_logs/${NAME}_${JID}.out           # the mirror sync_amarel.sh keeps
  ledger_append "$NAME" "$NET_ID" amarel "$JID" "$HOSTGPU" "$(stamp)" "$LOG" "$COMMIT" "$RUN_DIR" launch
  echo "$NAME -> Amarel job $JID (mem $MEM, wall $WALL, commit $COMMIT, code $ACODE, scratch $ARUN)";;
esac
