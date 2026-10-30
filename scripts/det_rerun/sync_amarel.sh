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
[ ${#names[@]} -gt 0 ] || mapfile -t names < <(awk -F'\t' '$2=="amarel"{print $1}' "$HERE/launched.tsv" | sort -u)
for NAME in "${names[@]}"; do
  last=$(awk -F'\t' -v n="$NAME" '$1==n{w=$2} END{print w}' "$HERE/launched.tsv")
  [ "$last" = amarel ] || { echo "skip $NAME: last launch is not on Amarel ($last)"; continue; }
  OUT=$(awk -F'\t' -v n="$NAME" 'NR>1 && $1==n' "$HERE/manifest.tsv" | grep -o 'output_dir=[^ ]*' | cut -d= -f2)
  [ -n "$OUT" ] || { echo "skip $NAME: no manifest row"; continue; }
  if [ -d "$OUT" ] && [ ! -f "$OUT/.amarel_mirror" ] && [ -n "$(ls -A "$OUT")" ]; then
    echo "REFUSED $NAME: $OUT has files but no .amarel_mirror marker"; continue
  fi
  mkdir -p "$OUT" && touch "$OUT/.amarel_mirror"
  rsync -a --delete --exclude .amarel_mirror "$AM:${OUT/#$EXP/$AEXP}/" "$OUT/" </dev/null \
    && echo "synced $NAME" || echo "FAILED $NAME (nothing on Amarel yet?)"
done
mkdir -p "$HERE/amarel_logs"
rsync -a "$AM:/home/st1122/det_rerun_logs/" "$HERE/amarel_logs/" </dev/null
