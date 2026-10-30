#!/usr/bin/env python
"""Score every det-rerun run dir from its prediction store into one CSV (tracker H01).

Usage: score_runs.py [--root EXP/det_rerun] [--out det_rerun_scores.csv]
Reads only stored predictions; loads no model. Partial runs are scored on the
epochs they have, and each row records its train_trajectories.
"""
import argparse
import csv
from pathlib import Path

from adaptive_roa.adaptive_v2.eval.store_score import score_run

ROOT = "/common/users/shared/pracsys/adaptive_roa_experiments/det_rerun"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--out", default=f"{ROOT}/det_rerun_scores.csv")
    a = ap.parse_args()
    rows = []
    for run in sorted(Path(a.root).glob("*/dr_*")):
        if (run / "partx_state.pkl").exists():
            continue                      # Part-X: scored by its own readout
        try:
            rows += [{"system": run.parent.name, **r} for r in score_run(run)]
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
