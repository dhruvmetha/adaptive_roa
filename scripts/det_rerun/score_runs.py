#!/usr/bin/env python
"""Score every det-rerun run dir from its prediction store into one CSV (tracker H01).

Usage: score_runs.py [--root EXP/det_rerun] [--out det_rerun_scores.csv]
Reads only stored predictions; loads no model. Partial runs are scored on the
epochs they have, and each row records its train_trajectories.

--thresholds fprcap refits the FPR-capped rule (adaptive_v2/eval/fpr_cap.py) on
each epoch's stored val split at every --alphas cap, with all stored val draws
unless --fit-k names a stored subsample, and writes det_rerun_scores_fprcap.csv:
one row per (run, epoch, alpha_s).
"""
import argparse
import csv
from pathlib import Path

from adaptive_roa.adaptive_v2.eval.fpr_cap import POSTHOC_ALPHAS
from adaptive_roa.adaptive_v2.eval.fpr_cap import score_run as fpr_cap_score_run
from adaptive_roa.adaptive_v2.eval.store_score import score_run

ROOT = "/common/users/shared/pracsys/adaptive_roa_experiments/det_rerun"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--out", default=None,
                    help=f"default {ROOT}/det_rerun_scores.csv, or det_rerun_scores_fprcap.csv for fprcap")
    ap.add_argument("--thresholds", choices=["run", "val", "both", "fprcap"], default="run",
                    help="the run's lambda*/delta*, a refit on the stored val split (D27b), both, "
                         "or the FPR-capped rule refit on the stored val split")
    ap.add_argument("--alphas", default=",".join(str(x) for x in POSTHOC_ALPHAS),
                    help="fprcap: comma-separated success caps alpha_s")
    ap.add_argument("--fit-k", type=int, default=None,
                    help="fprcap: fit on the stored n_*_k<K> val counts instead of all draws")
    a = ap.parse_args()
    if a.out is None:
        a.out = f"{ROOT}/det_rerun_scores_fprcap.csv" if a.thresholds == "fprcap" else f"{ROOT}/det_rerun_scores.csv"
    modes = ["run", "val"] if a.thresholds == "both" else [a.thresholds]
    alphas = [float(x) for x in a.alphas.split(",") if x.strip()]
    rows = []
    for run in sorted(Path(a.root).glob("*/dr_*")):
        if (run / "partx_state.pkl").exists():
            continue                      # Part-X: scored by its own readout
        try:
            if a.thresholds == "fprcap":
                rows += [{"system": run.parent.name, "thresholds": "fprcap", **r}
                         for r in fpr_cap_score_run(run, alphas, fit_k=a.fit_k)]
            else:
                rows += [{"system": run.parent.name, **r} for m in modes for r in score_run(run, m)]
        except Exception as e:            # keep scoring the rest; report the failure
            print(f"FAIL {run}: {type(e).__name__}: {e}")
    if not rows:
        print("no scored epochs yet")
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} rows from {len({r['run'] for r in rows})} runs -> {a.out}")


if __name__ == "__main__":
    main()
