#!/bin/bash
# Mirror every Amarel-launched det-rerun cell to its iLab output dir, and the Amarel
# logs to slurm_logs/det_rerun/amarel_logs, so status.py and the analysis scripts see
# them. A mirrored dir is written only by this sync, so --delete is safe there (a
# resumed cell drops its crashed partial epoch); a dir without the .amarel_mirror
# marker that already holds files is refused, since it may hold an iLab run.
#   sync_amarel.sh [name ...]          (default: every amarel row in launched.tsv)
set -euo pipefail
export SSH_AUTH_SOCK=
HERE=/common/home/st1122/Projects/adaptive_roa/slurm_logs/det_rerun
EXP=/common/users/shared/pracsys/adaptive_roa_experiments/det_rerun
AM=amarel.rutgers.edu; AEXP=/scratch/st1122/det_rerun_out
names=("$@")
# Rows renamed <name>~... (moved off Amarel, superseded) are history, not cells to sync.
[ ${#names[@]} -gt 0 ] || mapfile -t names < <(awk -F'\t' '$2=="amarel" && $1 !~ /~/ {print $1}' "$HERE/launched.tsv" | sort -u)
present=$(ssh -n -o BatchMode=yes $AM "cd $AEXP 2>/dev/null && ls -d */*" || true)   # cells started on Amarel
for NAME in "${names[@]}"; do
  last=$(awk -F'\t' -v n="$NAME" '$1==n{w=$2} END{print w}' "$HERE/launched.tsv")
  [ "$last" = amarel ] || { echo "skip $NAME: last launch is not on Amarel ($last)"; continue; }
  OUT=$(awk -F'\t' -v n="$NAME" 'NR>1 && $1==n {print $8}' "$HERE/manifest.tsv" | grep -o 'output_dir=[^ ]*' | cut -d= -f2 || true)
  [[ "$OUT" == "$EXP"/* && "$OUT" != *[[:space:]]* ]] || { echo "skip $NAME: bad output dir '$OUT'"; continue; }
  [ -n "$OUT" ] || { echo "skip $NAME: no manifest row"; continue; }
  grep -qx "${OUT#$EXP/}" <<< "$present" || { echo "waiting $NAME (not started on Amarel)"; continue; }
  if [ -d "$OUT" ] && [ ! -f "$OUT/.amarel_mirror" ] && [ -n "$(ls -A "$OUT")" ]; then
    echo "REFUSED $NAME: $OUT has files but no .amarel_mirror marker"; continue
  fi
  mkdir -p "$OUT" && touch "$OUT/.amarel_mirror"
  rsync -a --delete --exclude .amarel_mirror "$AM:${OUT/#$EXP/$AEXP}/" "$OUT/" </dev/null \
    && echo "synced $NAME" || echo "FAILED $NAME"
done
mkdir -p "$HERE/amarel_logs"
rsync -a "$AM:/home/st1122/det_rerun_logs/" "$HERE/amarel_logs/" </dev/null
