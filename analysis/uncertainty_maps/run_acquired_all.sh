#!/bin/bash
# Recover acquired states for every predictor x level cell. CPU-only and cheap:
# it just indexes the pool's start states by the recorded d2_indices.
set -uo pipefail
export PYTHONNOUSERSITE=1
CODE=/common/users/shared/pracsys/adaptive_roa_experiments/_uncertainty_maps/code
export PYTHONPATH=$CODE
OUT=/common/users/shared/pracsys/adaptive_roa_experiments/_uncertainty_maps/acquired
PY=/common/home/st1122/Projects/adaptive_roa/env/bin/python
cd "$CODE"
for L in det low med high xhigh; do
  for P in fm clf; do
    "$PY" analysis/uncertainty_maps/acquired_states.py --pred "$P" --level "$L" --out "$OUT" 2>&1 \
      | grep -E "^(fm|clf)_"
  done
done
