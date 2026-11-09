#!/bin/bash
# Final pull before Amarel goes down (maintenance-202611 from 2026-11-10 08:00): for each run whose
# latest placement is an Amarel launch or resume by this account, once squeue no longer lists its
# job, pull the run dir into the shared root and write .amarel_final there:
#   <job> <time> pulled    the pull returned 0
#   <job> <time> absent    Amarel holds no run dir: the job never started (cancelled while PENDING)
# is_live and leave_amarel accept the marker for that same job instead of asking Amarel, so
# resume_*.sh and DET_V2_RELAUNCH launches work while Amarel is down, whichever account runs them.
# A later Amarel placement has a new job id, so an old marker never counts for it.
#   final_pull_amarel.sh [name ...]    (default: every such run without final_results.json)
# Exit 1 when any run got no marker (still listed, Amarel unreachable, pull failed, refused).
# Only the account that ran a cell can read its scratch dir, so each account pulls its own.
set -euo pipefail
source "$(dirname "$0")/common.sh"
need_amarel
# One connection for the whole pull, as in sync_amarel.sh.
SSH="ssh -o BatchMode=yes -o ControlMaster=auto -o ControlPath=${XDG_RUNTIME_DIR:-/tmp}/det-v2-final-%C -o ControlPersist=60"
RSYNC_PERMS=(-rlptDO --chmod=Dg+rwxs,Fg+rw)
names=("$@"); explicit=${#names[@]}
if [ "$explicit" -eq 0 ] && [ -f "$LEDGER" ]; then
  # The latest launch/resume row decides, so a cancel row after it (a PENDING job cancelled
  # before the outage) still counts.
  mapfile -t names < <(awk -F'\t' -v u="$NET_ID" 'NR>1 && ($10=="launch" || $10=="resume") {w[$1]=$3; usr[$1]=$2}
    END {for (n in w) if (w[n]=="amarel" && usr[n]==u) print n}' "$LEDGER" | sort)
fi
rc_all=0
for NAME in "${names[@]}"; do
  PLACED=$(ledger_last_placed "$NAME")
  [ -n "$PLACED" ] && [ "$(lf "$PLACED" $L_WHERE)" = amarel ] && [ "$(lf "$PLACED" $L_USER)" = "$NET_ID" ] \
    || { echo "skip $NAME: last placement is not an Amarel run of $NET_ID"; continue; }
  JOB=$(lf "$PLACED" $L_JOB); OUT=$(lf "$PLACED" $L_RUNDIR)
  under_root "$OUT" || { echo "FAILED $NAME: bad run dir '$OUT'"; rc_all=1; continue; }
  [ "$explicit" -gt 0 ] || [ ! -f "$OUT/final_results.json" ] || continue
  probe=$($SSH -n "$AMAREL_HOST" "squeue -h -j $JOB -o %T 2>/dev/null; echo DET_V2_OK" </dev/null 2>/dev/null) || true
  grep -q DET_V2_OK <<< "$probe" || { echo "FAILED $NAME: could not ask Amarel about job $JOB"; rc_all=1; continue; }
  state=$(grep -v DET_V2_OK <<< "$probe" | head -1 | tr -d ' ' || true)     # grep exits 1 on no other line
  [ -z "$state" ] || { echo "skip $NAME: squeue still lists job $JOB ($state)"; rc_all=1; continue; }
  ADIR=$(amarel_dir "$OUT")
  has=$($SSH -n "$AMAREL_HOST" "if [ -d $ADIR ]; then echo DIR; else echo NODIR; fi" </dev/null 2>/dev/null) || has=""
  case "$has" in
    NODIR) how=absent ;;
    DIR)
      if [ -d "$OUT" ] && [ ! -f "$OUT/.amarel_mirror" ] && [ -n "$(ls -A "$OUT")" ]; then
        echo "REFUSED $NAME: $OUT has files but no .amarel_mirror marker"; rc_all=1; continue
      fi
      mkdir -p "$OUT" && touch "$OUT/.amarel_mirror"
      rc=0; rsync "${RSYNC_PERMS[@]}" --delete --exclude .amarel_mirror --exclude .amarel_final -e "$SSH" \
        "$AMAREL_HOST:$ADIR/" "$OUT/" </dev/null 2>/dev/null || rc=$?
      [ $rc = 0 ] || { echo "FAILED $NAME (rsync $rc)"; rc_all=1; continue; }
      how=pulled ;;
    *) echo "FAILED $NAME: could not list $ADIR on Amarel"; rc_all=1; continue ;;
  esac
  mkdir -p "$OUT"
  printf '%s %s %s\n' "$JOB" "$(stamp)" "$how" > "$OUT/.amarel_final.tmp"
  chmod g+rw "$OUT/.amarel_final.tmp" && mv -f "$OUT/.amarel_final.tmp" "$OUT/.amarel_final"
  echo "final $NAME: job $JOB, $how, $(banked "$OUT") banked epoch(s)"
done
exit $rc_all
