#!/bin/bash
# det_v3 entry point: the det_v2 harness script of the same name with DET_CAMPAIGN=det_v3, so the
# shared root, ledger, logs and manifest are det_v3's (common.sh). Same arguments as the det_v2 one.
export DET_CAMPAIGN=det_v3
exec bash "$(cd "$(dirname "$0")" && pwd)/../det_v2/$(basename "$0")" "$@"
