#!/usr/bin/env python
"""Conformal alpha sweep for every det-rerun run dir, from the store (tracker H02).

Usage: alpha_sweep.py [--root EXP/det_rerun] [--out det_rerun_alpha_sweep.csv]
"""
import argparse
import csv
from pathlib import Path

from adaptive_roa.adaptive_v2.eval.store_conformal import sweep_run

ROOT = "/common/users/shared/pracsys/adaptive_roa_experiments/det_rerun"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--out", default=f"{ROOT}/det_rerun_alpha_sweep.csv")
    a = ap.parse_args()
    rows = []
    for run in sorted(Path(a.root).glob("*/dr_*")):
        if (run / "partx_state.pkl").exists():
            continue
        try:
            rows += [{"system": run.parent.name, **r} for r in sweep_run(run)]
        except Exception as e:
            print(f"FAIL {run}: {type(e).__name__}: {e}")
    if not rows:
        print("no epochs with stored cal predictions yet")
        return
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} rows -> {a.out}")


if __name__ == "__main__":
    main()
