#!/bin/bash
# Freeze one commit of the det-paper worktree into a read-only code dir and print its path.
#   snapshot.sh [ref]          (default HEAD; a branch tip works without checking it out)
# Launches run from the snapshot, so later edits to the worktree never reach a live
# job's re-imports. The dir sits outside every git repo: git there cannot report the
# main checkout's HEAD as this code's commit, and launch.sh pins ADAPTIVE_ROA_COMMIT.
# On Amarel: DET_WT=/home/st1122/Projects/adaptive_roa_det DET_CODE_ROOT=/home/st1122/det_rerun_code
# (its git 1.8 has no `git -C`, hence the cd).
set -euo pipefail
WT=${DET_WT:-/common/home/st1122/Projects/adaptive_roa/.claude/worktrees/det-paper}
ROOT=${DET_CODE_ROOT:-/common/users/st1122/det_rerun_code}
COMMIT=$(cd "$WT" && git rev-parse --short "${1:-HEAD}^{commit}")
CODE=$ROOT/$COMMIT
if [ ! -d "$CODE" ]; then
  mkdir -p "$ROOT"
  TMP=$(mktemp -d "$ROOT/.tmp_${COMMIT}_XXXXXX")
  (cd "$WT" && git archive "$COMMIT") | tar -x -C "$TMP"
  cp "$WT/.env" "$TMP/.env"          # gitignored; env_config.py reads paths from it
  echo "$COMMIT" > "$TMP/COMMIT"
  chmod -R a-w "$TMP"
  mv -T "$TMP" "$CODE" 2>/dev/null || { chmod -R u+w "$TMP"; rm -rf "$TMP"; }   # lost a race: keep the winner
fi
[ "$(cat "$CODE/COMMIT")" = "$COMMIT" ] || { echo "snapshot $CODE does not hold $COMMIT" >&2; exit 1; }
echo "$CODE"
