#!/bin/bash
# One command for the whole deliverable: pull Amarel artifacts, score every arm,
# redraw the figures. Safe to run repeatedly while the runs are still training.
set -u
REPO=/common/home/st1122/Projects/adaptive_roa
OUT=${1:-$REPO/docs/stoch_compare}
cd "$REPO"

bash scripts/sync_amarel_results.sh 2>&1 | tail -2 || echo "(amarel sync skipped)"
PYTHONPATH=scripts ./env/bin/python scripts/stoch_compare_report.py --all-epochs --out "$OUT"
PYTHONPATH=scripts ./env/bin/python scripts/plot_stoch_compare.py \
    --metrics "$OUT/metrics.json" --out "$OUT"
PYTHONPATH=scripts ./env/bin/python scripts/acquisition_diagnostics.py \
    --out "$OUT/acquisition.md"
PYTHONPATH=scripts ./env/bin/python scripts/seed_variance.py \
    --out "$OUT/seed_variance.md"
PYTHONPATH=scripts ./env/bin/python scripts/plot_metric_grid.py \
    --metrics "$OUT/metrics.json" --out "$OUT/metric_grid.png"
PYTHONPATH=scripts ./env/bin/python scripts/plot_fm_vs_clf.py \
    --metrics "$OUT/metrics.json" --out "$OUT"
PYTHONPATH=scripts ./env/bin/python scripts/export_matched_epoch_csv.py \
    --metrics "$OUT/metrics.json" --out "$OUT/matched_epoch.csv"
PYTHONPATH=scripts ./env/bin/python scripts/export_matched_epoch_csv.py --per-predictor \
    --metrics "$OUT/metrics.json" --out "$OUT/matched_epoch_per_predictor.csv"
echo "== deliverable refreshed at $(date +%H:%M) -> $OUT =="
