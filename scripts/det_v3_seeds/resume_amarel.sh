#!/bin/bash
# det_v3_seeds entry point: the det_v2 harness script of the same name with DET_CAMPAIGN=det_v3_seeds, so the
# shared root, ledger, logs and manifest are the det_v3_seeds ones (common.sh). Same arguments as the det_v2 one.
export DET_CAMPAIGN=det_v3_seeds
exec bash "$(cd "$(dirname "$0")" && pwd)/../det_v2/$(basename "$0")" "$@"
