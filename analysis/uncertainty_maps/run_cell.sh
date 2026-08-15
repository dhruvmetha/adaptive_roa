#!/bin/bash
# Build the per-epoch pages plus the seven epoch sweeps for ONE decomposable
# cell, once that cell's member recompute has finished. Takes pred and level so
# a cell can be released the moment it is ready instead of waiting on the whole
# fleet.
#
#   run_cell.sh fm high
set -uo pipefail
PRED=${1:?pred}
LEVEL=${2:?level}
export PYTHONNOUSERSITE=1
CODE=/common/users/shared/pracsys/adaptive_roa_experiments/_uncertainty_maps/code
export PYTHONPATH=$CODE
BASE=/common/users/shared/pracsys/adaptive_roa_experiments/_uncertainty_maps
PY=/common/home/st1122/Projects/adaptive_roa/env/bin/python
cd "$CODE"

echo "=== per-epoch pages: ${PRED}_${LEVEL} ==="
"$PY" analysis/uncertainty_maps/make_maps.py --pred "$PRED" --level "$LEVEL" \
    --out "$BASE/figures" --png-epochs "0,9,18" 2>&1 \
    | grep -vE "Theseus|^  page" | tail -6

for Q in p h_total h_aleatoric h_epistemic var_total var_aleatoric var_epistemic; do
  "$PY" analysis/uncertainty_maps/make_sweep.py --pred "$PRED" --level "$LEVEL" \
      --quantity "$Q" --every 2 --out "$BASE/figures" 2>&1 \
      | grep -vE "Theseus" | tail -1
done
