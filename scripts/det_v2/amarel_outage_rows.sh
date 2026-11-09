#!/bin/bash
# Inside Amarel's maintenance reservation (AMAREL_OUTAGE_START <= now < AMAREL_OUTAGE_END, common.sh),
# append one 'outage' ledger row per Amarel job of this account that is its run's latest placement
# and that the last snapshots sync_amarel.sh took before Amarel went down show as
#   - RUNNING in squeue (logs/amarel_queue_<NET_ID>.txt), log column squeue:RUNNING, or
#   - ended TIMEOUT, PREEMPTED or NODE_FAIL in sacct (logs/amarel_sacct_<NET_ID>.txt), log column
#     sacct:<state>; a job cancelled while PENDING has a cancel row instead.
# No Amarel job runs inside the reservation, so is_live takes such a job as not live without asking
# Amarel. With the row, leave_amarel resumes the run from the iLab copy (epochs only Amarel holds
# are redone); resume_*.sh still refuses a run with nothing banked, so nothing relaunches from 0.
#   amarel_outage_rows.sh          (refuses outside the window; writes each job id once)
# DET_V2_NOW=<YYYY-MM-DDTHH:MM:SS> stands in for the clock (the tests).
set -euo pipefail
source "$(dirname "$0")/common.sh"
now=${DET_V2_NOW:-$(stamp)}
if [[ "$now" < "$AMAREL_OUTAGE_START" || ! "$now" < "$AMAREL_OUTAGE_END" ]]; then
  die "now ($now) is outside the Amarel outage window [$AMAREL_OUTAGE_START, $AMAREL_OUTAGE_END); no outage rows written"
fi
QSNAP=$LOG_DIR/amarel_queue_$NET_ID.txt; ASNAP=$LOG_DIR/amarel_sacct_$NET_ID.txt
[ -f "$QSNAP" ] || [ -f "$ASNAP" ] || die "no squeue or sacct snapshot in $LOG_DIR (sync_amarel.sh writes them)"
n=0
# outage_row <name> <job id or -> <seen> <snapshot>: one row for the name's latest Amarel placement of
# this account, when its job is the given one (- takes the placement's) and has no outage row yet.
outage_row() {
  local placed job
  placed=$(ledger_last_placed "$1")
  [ -n "$placed" ] && [ "$(lf "$placed" $L_WHERE)" = amarel ] && [ "$(lf "$placed" $L_USER)" = "$NET_ID" ] || return 0
  job=$(lf "$placed" $L_JOB)
  [ "$2" = - ] || [ "$2" = "$job" ] || return 0
  ! awk -F'\t' -v n="$1" -v j="$job" 'NR>1 && $1==n && $3=="amarel" && $4==j && ($10=="outage" || $10=="cancel") {f=1}
    END {exit !f}' "$LEDGER" || return 0
  ledger_append "$1" "$NET_ID" amarel "$job" "$(lf "$placed" $L_HOSTGPU)" "$now" "$3" \
    "$(lf "$placed" $L_COMMIT)" "$(lf "$placed" $L_RUNDIR)" outage
  echo "outage $1: Amarel job $job, $3 in the snapshot of $(date -r "$4" +%Y-%m-%dT%H:%M:%S)"
  n=$((n + 1))
}
if [ -f "$ASNAP" ]; then
  while IFS='|' read -r job name state; do
    case "$state" in TIMEOUT|PREEMPTED|NODE_FAIL) outage_row "$name" "$job" "sacct:$state" "$ASNAP" ;; esac
  done < "$ASNAP"
fi
if [ -f "$QSNAP" ]; then
  while read -r name state _; do
    [ "$state" != RUNNING ] || outage_row "$name" - squeue:RUNNING "$QSNAP"
  done < "$QSNAP"
fi
echo "$n outage row(s) written"
