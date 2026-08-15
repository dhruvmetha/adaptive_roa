#!/bin/bash
# Build every predictor x level PDF. Cells whose member checkpoints survived get
# the full decomposition; the rest render the four split columns as an explicit
# "not recoverable" tile rather than quietly dropping them, so a reader can see
# where the campaign's own retention policy limits the analysis.
set -uo pipefail
export PYTHONNOUSERSITE=1
CODE=/common/users/shared/pracsys/adaptive_roa_experiments/_uncertainty_maps/code
export PYTHONPATH=$CODE
BASE=/common/users/shared/pracsys/adaptive_roa_experiments/_uncertainty_maps
PY=/common/home/st1122/Projects/adaptive_roa/env/bin/python
cd "$CODE"
for P in fm clf; do
  for L in det low med high xhigh; do
    echo "=== $P $L ==="
    "$PY" analysis/uncertainty_maps/make_maps.py \
        --pred "$P" --level "$L" --out "$BASE/figures" --png-epochs "${PNG_EPOCHS:-0,9,18}" \
        2>&1 | grep -vE "Theseus|^  page" | tail -8
  done
done
