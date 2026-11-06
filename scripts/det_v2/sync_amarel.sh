#!/bin/bash
# Mirror this account's Amarel det_v2 runs into their shared run dirs, and the Amarel logs
# into LOG_DIR/amarel_logs, so status.py and the analysis scripts see them.
#   sync_amarel.sh [name ...]          (default: every run whose latest ledger row is an
#                                       Amarel launch or resume by NET_ID)
# Only the account that ran a cell can read its scratch dir, so each account syncs its own.
# A mirrored dir is written only by this sync, so --delete is safe there (a resumed cell
# drops its crashed partial epoch); a dir without the .amarel_mirror marker that already
# holds files is refused, since it may hold a run from iLab or a direct box. Everything is
# copied, .hydra included (plot_prob_fields.py skips a run without .hydra/config.yaml).
# -rlptDO with --chmod: -p keeps the umask from narrowing the group bits, and no -g/-o, so
# the setgid root's group applies. -O skips directory mtimes: a run dir the other account
# owns refuses them with EPERM (rsync exit 23), which would block cross-account resumes.
set -euo pipefail
source "$(dirname "$0")/common.sh"
source "$DET_V2_SCRIPTS/retry.sh"
need_amarel
# One connection for the whole sync: a fresh handshake costs ~2.4 s and a sync makes many.
SSH="ssh -o BatchMode=yes -o ControlMaster=auto -o ControlPath=${XDG_RUNTIME_DIR:-/tmp}/det-v2-sync-%C -o ControlPersist=60"
RSYNC_PERMS=(-rlptDO --chmod=Dg+rwxs,Fg+rw)
names=("$@")
if [ ${#names[@]} -eq 0 ] && [ -f "$LEDGER" ]; then
  mapfile -t names < <(awk -F'\t' -v u="$NET_ID" 'NR>1 {w[$1]=$3; usr[$1]=$2; a[$1]=$10}
    END {for (n in w) if (w[n]=="amarel" && usr[n]==u && (a[n]=="launch" || a[n]=="resume")) print n}' "$LEDGER" | sort)
fi
mkdir -p "$LOG_DIR"
# Live jobs; the snapshot (name, state) also tells status.py which runs are PENDING.
QUEUE_SNAP=$LOG_DIR/amarel_queue_$NET_ID.txt
queue=$($SSH -n "$AMAREL_HOST" "squeue -u $AMAREL_USER -h -o '%j %T'") \
  && printf '%s\n' "$queue" > "$QUEUE_SNAP.tmp" && mv "$QUEUE_SNAP.tmp" "$QUEUE_SNAP" \
  || queue=""
active=$(awk '{print $1}' <<< "$queue")
for NAME in "${names[@]}"; do
  ROW=$(ledger_last "$NAME")
  [ "$(lf "$ROW" $L_WHERE)" = amarel ] && [ "$(lf "$ROW" $L_USER)" = "$NET_ID" ] \
    || { echo "skip $NAME: latest row is not an Amarel run of $NET_ID"; continue; }
  OUT=$(lf "$ROW" $L_RUNDIR)
  under_root "$OUT" || { echo "skip $NAME: bad run dir '$OUT'"; continue; }
  # Finished and fully mirrored, no live job: nothing left to pull.
  if [ -f "$OUT/final_results.json" ] && ! grep -qx "$NAME" <<< "$active"; then continue; fi
  ADIR=$(amarel_dir "$OUT")
  $SSH -n "$AMAREL_HOST" "test -d $ADIR" || { echo "waiting $NAME (not started on Amarel)"; continue; }
  if [ -d "$OUT" ] && [ ! -f "$OUT/.amarel_mirror" ] && [ -n "$(ls -A "$OUT")" ]; then
    echo "REFUSED $NAME: $OUT has files but no .amarel_mirror marker"; continue
  fi
  mkdir -p "$OUT" && touch "$OUT/.amarel_mirror"
  rc=0; rsync "${RSYNC_PERMS[@]}" --delete --exclude .amarel_mirror -e "$SSH" "$AMAREL_HOST:$ADIR/" "$OUT/" </dev/null 2>/dev/null || rc=$?
  # 24: a running job replaced a checkpoint mid-copy; the next sync picks it up.
  case $rc in 0|24) echo "synced $NAME";; *) echo "FAILED $NAME (rsync $rc)";; esac
done
mkdir -p "$LOG_DIR/amarel_logs"
# Slurm only appends to a log, so --append sends the new tail instead of re-reading the file.
retry 3 rsync "${RSYNC_PERMS[@]}" --append -e "$SSH" "$AMAREL_HOST:$AMAREL_LOG_DIR/" "$LOG_DIR/amarel_logs/" </dev/null
