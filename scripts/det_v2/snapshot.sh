#!/bin/bash
# Freeze one commit of the det_v2 repo into a read-only code dir and print its path.
#   snapshot.sh [ref]          (default HEAD; any commit the clone has or can fetch from origin)
# Every launch and resume runs from SNAPSHOT_DIR/<commit>, so later commits and worktree
# edits never reach a live job's re-imports (mp.spawn children reload modules from disk).
# The snapshot's .env is written from the loaded settings (NET_ID, DATA_DIR,
# SHARED_DATA_BASE, EXP_DIR, USER_BASE), so env_config.py inside the job resolves the same
# paths the launcher composed. The dir sits outside every git repo: git there cannot
# report some other checkout's HEAD as this code's commit, and ADAPTIVE_ROA_COMMIT is pinned.
#
# On Amarel launch.sh pipes this file from the commit itself (git show <commit>:... | bash -s)
# with DET_V2_LOADED=1 and the Amarel values set, so the Amarel clone needs no user.env.
# Its git 1.8 has no `git -C`, hence the cd.
set -euo pipefail
if [ "${DET_V2_LOADED:-}" != 1 ]; then
  source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
fi
: "${DET_V2_REPO:?}" "${SNAPSHOT_DIR:?}" "${NET_ID:?}" "${DATA_DIR:?}" "${EXP_DIR:?}"
umask 002
REF=${1:-HEAD}
cd "$DET_V2_REPO"
if ! git rev-parse -q --verify "$REF^{commit}" >/dev/null; then
  git fetch -q origin || true                     # a commit another account launched
fi
COMMIT=$(git rev-parse --short=10 "$REF^{commit}") || { echo "no commit $REF in $DET_V2_REPO (not pushed?)" >&2; exit 1; }
CODE=$SNAPSHOT_DIR/$COMMIT
if [ ! -d "$CODE" ]; then
  mkdir -p "$SNAPSHOT_DIR"
  TMP=$(mktemp -d "$SNAPSHOT_DIR/.tmp_${COMMIT}_XXXXXX")
  git archive "$COMMIT" | tar -x -C "$TMP"
  {
    echo "# written by scripts/det_v2/snapshot.sh for $COMMIT"
    echo "NET_ID=$NET_ID"
    echo "DATA_DIR=$DATA_DIR"
    echo "SHARED_DATA_BASE=${SHARED_DATA_BASE:-$DATA_DIR}"
    echo "EXP_DIR=$EXP_DIR"
    [ -z "${USER_BASE:-}" ] || echo "USER_BASE=$USER_BASE"
  } > "$TMP/.env"
  echo "$COMMIT" > "$TMP/COMMIT"
  chmod 755 "$TMP"                                # mktemp makes 700; other accounts read snapshots too
  chmod -R a-w "$TMP"
  mv -T "$TMP" "$CODE" 2>/dev/null || { chmod -R u+w "$TMP"; rm -rf "$TMP"; }   # lost a race: keep the winner
fi
[ "$(cat "$CODE/COMMIT")" = "$COMMIT" ] || { echo "snapshot $CODE does not hold $COMMIT" >&2; exit 1; }
echo "$CODE"
