#!/bin/bash
# Everything that needs per-member data: the three cells whose checkpoints
# survived. Run only after the member recompute fleet has drained, so no page is
# rendered from a half-finished arm.
set -uo pipefail
export PYTHONNOUSERSITE=1
CODE=/common/users/shared/pracsys/adaptive_roa_experiments/_uncertainty_maps/code
export PYTHONPATH=$CODE
BASE=/common/users/shared/pracsys/adaptive_roa_experiments/_uncertainty_maps
PY=/common/home/st1122/Projects/adaptive_roa/env/bin/python
cd "$CODE"

for CELL in "fm high" "fm xhigh" "clf high"; do
  set -- $CELL
  echo "=== per-epoch pages: $1 $2 ==="
  "$PY" analysis/uncertainty_maps/make_maps.py --pred "$1" --level "$2" \
      --out "$BASE/figures" --png-epochs "0,9,18" 2>&1 \
      | grep -vE "Theseus|^  page" | tail -5
  for Q in p h_total h_aleatoric h_epistemic var_total var_aleatoric var_epistemic; do
    "$PY" analysis/uncertainty_maps/make_sweep.py --pred "$1" --level "$2" \
        --quantity "$Q" --every 2 --out "$BASE/figures" 2>&1 \
        | grep -vE "Theseus" | tail -2
  done
done
