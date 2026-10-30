#!/usr/bin/env python
"""Ablation A-W1: refit lambda*/delta* at w in {0.5, 0.75, 0.9, 0.95, 0.99} from the store.

Usage: w_refit.py [--root EXP/det_rerun] [--out det_rerun_w_refit.csv]
Rows with matches_run=False at the run's own w mean the refit does not reproduce
the run and the whole file should be distrusted.
"""
import argparse
import contextlib
import csv
import io
from pathlib import Path

from adaptive_roa.adaptive_v2.eval.store_threshold import refit_run

ROOT = "/common/users/shared/pracsys/adaptive_roa_experiments/det_rerun"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--out", default=f"{ROOT}/det_rerun_w_refit.csv")
    a = ap.parse_args()
    rows = []
    for run in sorted(Path(a.root).glob("*/dr_*")):
        if (run / "partx_state.pkl").exists():
            continue
        try:
            with contextlib.redirect_stdout(io.StringIO()):     # the optimizer prints its grid
                rows += [{"system": run.parent.name, **r} for r in refit_run(run)]
        except Exception as e:
            print(f"FAIL {run}: {type(e).__name__}: {e}")
    bad = [r for r in rows if r.get("matches_run") is False]
    for r in bad:
        print(f"MISMATCH {r['run']} epoch {r['epoch']}: refit at the run's w does not reproduce it")
    if rows:
        with open(a.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
    print(f"{len(rows)} rows, {len(bad)} mismatches -> {a.out}")


if __name__ == "__main__":
    main()
