#!/bin/bash
# Resume a det_v2 run on Amarel from its last banked epoch, from the commit it was launched
# with (scripts/resume_adaptive.py --run-dir on the scratch twin; never from 0).
#   resume_amarel.sh <name> [walltime]               (default 3-00:00:00)
# A run that last ran on Amarel resumes from its scratch dir (only the account that ran it
# can reach that scratch). A run banked elsewhere, with no live job, is pushed to the
# scratch twin first and its shared run dir becomes a mirror (.amarel_mirror) that
# sync_amarel.sh keeps. Part-X rows rerun launch.sh instead.
set -euo pipefail
source "$(dirname "$0")/common.sh"
NAME=${1:?usage: resume_amarel.sh <name> [walltime]}; WALL=${2:-3-00:00:00}
need_amarel
resume_prepare "$NAME"
ARUN=$(amarel_dir "$RUN_DIR")
ledger_claim_resume "$NAME" "$LAST" amarel "$AMAREL_PARTITION" "$COMMIT" "$RUN_DIR" || die "not resuming $NAME"
fail() { ledger_append "$NAME" "$NET_ID" amarel - "$AMAREL_PARTITION" "$(stamp)" - "$COMMIT" "$RUN_DIR" abandon; die "$1"; }
if [ "$PREV" = amarel ]; then
  [ "$PREV_USER" = "$NET_ID" ] || fail "$NAME ran on $PREV_USER's Amarel scratch; that account must resume it"
  rsh "$AMAREL_HOST" "ls $ARUN/epoch_*/artifacts_v2.json" >/dev/null 2>&1 \
    || fail "$NAME has no banked epoch on Amarel ($ARUN); launch it instead"
else
  [ ! -f "$RUN_DIR/.amarel_mirror" ] || fail "$RUN_DIR is an Amarel mirror but $NAME last ran on $PREV; check it"
  [ "$(banked "$RUN_DIR")" -gt 0 ] || fail "$NAME has no banked epoch in $RUN_DIR; launch it instead"
  ! rsh "$AMAREL_HOST" "test -e $ARUN" || fail "$ARUN already exists on Amarel; move it aside first"
  rsh "$AMAREL_HOST" "mkdir -p $(dirname "$ARUN")" || fail "could not make $(dirname "$ARUN") on Amarel"
  rc=0; rsync -rlptD --exclude .amarel_mirror "$RUN_DIR/" "$AMAREL_HOST:$ARUN/" </dev/null || rc=$?
  [ $rc = 0 ] || fail "push of $NAME to Amarel failed (rsync $rc)"
  touch "$RUN_DIR/.amarel_mirror"
fi
FULL=$(full_commit "$COMMIT") || fail "no commit $COMMIT in $DET_V2_REPO"
on_origin "$FULL" || fail "$COMMIT is not on origin; push it first"
ACODE=$(amarel_snapshot "$FULL") || true
[ -n "$ACODE" ] || fail "Amarel snapshot of $COMMIT failed"
MEM=${DET_V2_MEM:-$(manifest_get "$NAME" mem)}
JID=$(amarel_submit "$NAME" "$ACODE" "$COMMIT" "$MEM" "$WALL" scripts/resume_adaptive.py --run-dir "$ARUN") || true
[ -n "$JID" ] || fail "Amarel sbatch failed for $NAME"
ledger_append "$NAME" "$NET_ID" amarel "$JID" "$AMAREL_PARTITION" "$(stamp)" "$LOG_DIR/amarel_logs/${NAME}_${JID}.out" "$COMMIT" "$RUN_DIR" resume
echo "$NAME resumed -> Amarel job $JID (wall $WALL, commit $COMMIT, scratch $ARUN)"
