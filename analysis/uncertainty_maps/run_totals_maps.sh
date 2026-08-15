#!/bin/bash
# The seven cells whose member checkpoints did not survive, so they are
# total-only. high/xhigh for both predictors are built separately, after the
# per-member recompute lands, so they are never rendered from a partial fleet.
set -uo pipefail
export PYTHONNOUSERSITE=1
CODE=/common/users/shared/pracsys/adaptive_roa_experiments/_uncertainty_maps/code
export PYTHONPATH=$CODE
BASE=/common/users/shared/pracsys/adaptive_roa_experiments/_uncertainty_maps
PY=/common/home/st1122/Projects/adaptive_roa/env/bin/python
cd "$CODE"
for CELL in "fm det" "fm low" "fm med" "clf det" "clf low" "clf med" "clf xhigh"; do
  set -- $CELL
  echo "=== $1 $2 ==="
  "$PY" analysis/uncertainty_maps/make_maps.py --pred "$1" --level "$2" \
      --out "$BASE/figures" --png-epochs "${PNG_EPOCHS:-0,9,18}" 2>&1 \
      | grep -vE "Theseus|^  page" | tail -6
done
