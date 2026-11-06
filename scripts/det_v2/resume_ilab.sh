#!/bin/bash
# Resume a det_v2 run on an iLab card from its last banked epoch, from the snapshot of the
# commit it was launched with (scripts/resume_adaptive.py --run-dir; never from 0).
#   resume_ilab.sh <name> [walltime] [gres_type]     (defaults: 3-00:00:00, the type it last
#                                                    ran on; pass the type for a run from
#                                                    another place)
# A run banked on Amarel (no live job) gets its final pull into the shared run dir first.
# DET_V2_EXCLUDE=<nodes> replaces ILAB_EXCLUDE (the association allows one node: keep every
# job on it). DET_V2_MEM=<mem> replaces the manifest request.
set -euo pipefail
source "$(dirname "$0")/common.sh"
NAME=${1:?usage: resume_ilab.sh <name> [walltime] [gres_type]}; WALL=${2:-3-00:00:00}; GRES=${3:-}
resume_prepare "$NAME"
if [ -z "$GRES" ]; then
  if [ "$PREV" = ilab ]; then GRES=$(lf "$LAST" $L_HOSTGPU); else die "$NAME last ran on $PREV; pass the gres type"; fi
fi
ledger_claim_resume "$NAME" "$LAST" ilab "$GRES" "$COMMIT" "$RUN_DIR" || die "not resuming $NAME"
fail() { ledger_append "$NAME" "$NET_ID" ilab - "$GRES" "$(stamp)" - "$COMMIT" "$RUN_DIR" abandon; die "$1"; }
( leave_amarel "$NAME" ) || fail "could not take $NAME off Amarel"
[ "$(banked "$RUN_DIR")" -gt 0 ] || fail "$NAME has no banked epoch in $RUN_DIR; launch it instead"
CODE=$(bash "$DET_V2_SCRIPTS/snapshot.sh" "$COMMIT") || fail "no snapshot of $COMMIT"
MEM=${DET_V2_MEM:-$(manifest_get "$NAME" mem)}
JID=$(ilab_submit "$NAME" "$CODE" "$COMMIT" "$MEM" "$WALL" "$GRES" scripts/resume_adaptive.py --run-dir "$RUN_DIR") || true
[ -n "$JID" ] || fail "iLab sbatch failed for $NAME"
ledger_append "$NAME" "$NET_ID" ilab "$JID" "$GRES" "$(stamp)" "$LOG_DIR/${NAME}_${JID}.out" "$COMMIT" "$RUN_DIR" resume
echo "$NAME resumed -> iLab job $JID ($GRES, wall $WALL, $(banked "$RUN_DIR") banked, commit $COMMIT)"
