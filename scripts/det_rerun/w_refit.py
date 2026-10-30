#!/usr/bin/env python
"""Ablation A-W1: refit lambda*/delta* at w in {0.5, 0.75, 0.9, 0.95, 0.99} from the store.

Usage: w_refit.py [--root EXP/det_rerun] [--out det_rerun_w_refit.csv]
Rows with matches_run=False at the run's own w mean the refit does not reproduce
the run and the whole file should be distrusted. One refit is about 30 s on a
quad3d epoch, so runs go to --jobs worker processes.
"""
import argparse
import contextlib
import csv
import io
from multiprocessing import Pool
from pathlib import Path

from adaptive_roa.adaptive_v2.eval.store_threshold import refit_run

ROOT = "/common/users/shared/pracsys/adaptive_roa_experiments/det_rerun"


def _one(run: Path):
    try:
        with contextlib.redirect_stdout(io.StringIO()):     # the optimizer prints its grid
            return [{"system": run.parent.name, **r} for r in refit_run(run)], None
    except Exception as e:
        return [], f"FAIL {run}: {type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--out", default=f"{ROOT}/det_rerun_w_refit.csv")
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    runs = [r for r in sorted(Path(a.root).glob("*/dr_*")) if not (r / "partx_state.pkl").exists()]
    with Pool(max(1, min(a.jobs, len(runs)))) as pool:
        results = pool.map(_one, runs, chunksize=1)
    rows = [row for got, _ in results for row in got]
    for _, err in results:
        if err:
            print(err)
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
