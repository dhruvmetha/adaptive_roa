#!/bin/bash
# Stage a dataset subtree from iLab to Amarel.
#
# Amarel does not mount /common, so each run's data has to be pushed over first.
# Run this FROM an iLab box (iLab can always ssh to Amarel; the reverse is not
# guaranteed from compute nodes).
#
# Usage:
#   scripts/stage_dataset_amarel.sh deterministic/pendulum
#   scripts/stage_dataset_amarel.sh deterministic/pendulum/lqr noisy/pendulum/lqr/med
#   DRY_RUN=1 scripts/stage_dataset_amarel.sh deterministic/pendulum
#
# Paths are relative to the dataset root on both sides.

set -euo pipefail

SRC_ROOT=${SRC_ROOT:-/common/users/shared/pracsys/genMoPlan/data_trajectories}
DST_HOST=${DST_HOST:-amarel.rutgers.edu}
DST_ROOT=${DST_ROOT:-/scratch/st1122/genMoPlan-exp/data_trajectories}

if [ $# -eq 0 ]; then
    echo "usage: $0 <subpath> [subpath ...]" >&2
    echo "  e.g. $0 deterministic/pendulum noisy/pendulum/lqr/med" >&2
    exit 1
fi

RSYNC_OPTS=(-avhP --partial --info=progress2)
[ -n "${DRY_RUN:-}" ] && RSYNC_OPTS+=(--dry-run) && echo "*** DRY RUN ***"

for SUB in "$@"; do
    SRC="$SRC_ROOT/${SUB%/}"
    if [ ! -e "$SRC" ]; then
        echo "ERROR: source does not exist: $SRC" >&2
        exit 1
    fi

    # Create the parent on the far side, then sync the leaf into it, so that
    # "deterministic/pendulum" lands at "$DST_ROOT/deterministic/pendulum".
    PARENT=$(dirname "${SUB%/}")
    ssh "$DST_HOST" "mkdir -p '$DST_ROOT/$PARENT'"

    echo "===> $SUB  ($(du -sh "$SRC" 2>/dev/null | cut -f1))"
    rsync "${RSYNC_OPTS[@]}" -e ssh "$SRC" "$DST_HOST:$DST_ROOT/$PARENT/"
done

echo "Staged: $*"
echo "Verify: ssh $DST_HOST 'du -sh $DST_ROOT/*/'"
