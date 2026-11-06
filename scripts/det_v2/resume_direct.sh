#!/bin/bash
# Resume a det_v2 run on a direct-box card from its last banked epoch, from the snapshot of
# the commit it was launched with (scripts/resume_adaptive.py --run-dir; never from 0).
#   resume_direct.sh <name> <host> <gpu>
# A run banked on Amarel (no live job) gets its final pull into the shared run dir first,
# so sync_amarel.sh never overwrites the run that continues here.
set -euo pipefail
source "$(dirname "$0")/common.sh"
[ $# -eq 3 ] || die "usage: resume_direct.sh <name> <host> <gpu>"
NAME=$1; HOST=$2; GPU=$3
resume_prepare "$NAME"
ledger_claim_resume "$NAME" "$LAST" direct "$HOST:$GPU" "$COMMIT" "$RUN_DIR" || die "not resuming $NAME"
fail() { ledger_append "$NAME" "$NET_ID" direct - "$HOST:$GPU" "$(stamp)" - "$COMMIT" "$RUN_DIR" abandon; die "$1"; }
( leave_amarel "$NAME" ) || fail "could not take $NAME off Amarel"
[ "$(banked "$RUN_DIR")" -gt 0 ] || fail "$NAME has no banked epoch in $RUN_DIR; launch it instead"
CODE=$(bash "$DET_V2_SCRIPTS/snapshot.sh" "$COMMIT") || fail "no snapshot of $COMMIT"
mkdir -p "$LOG_DIR"
LOG=$LOG_DIR/$(date +%Y%m%d_%H%M)_${NAME}_resume.log
PID=$(start_direct "$NAME" "$HOST" "$GPU" "$CODE" "$COMMIT" "$LOG" scripts/resume_adaptive.py --run-dir "$RUN_DIR") \
  || fail "tmux resume failed for $NAME on $HOST"
ledger_append "$NAME" "$NET_ID" direct "$PID" "$HOST:$GPU" "$(stamp)" "$LOG" "$COMMIT" "$RUN_DIR" resume
echo "$NAME resumed -> $HOST gpu$GPU pid $PID session gpu_$NAME ($(banked "$RUN_DIR") banked, commit $COMMIT)"
