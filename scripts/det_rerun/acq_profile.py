#!/usr/bin/env python
"""Acquisition profile per round for every det-rerun run dir (tracker H09).

Usage: acq_profile.py [--root EXP/det_rerun] [--out det_rerun_acq_profile.csv]
"""
import argparse
import csv
from pathlib import Path

from adaptive_roa.adaptive_v2.eval.store_acq import acquisition_profile

ROOT = "/common/users/shared/pracsys/adaptive_roa_experiments/det_rerun"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--out", default=f"{ROOT}/det_rerun_acq_profile.csv")
    a = ap.parse_args()
    rows = []
    for run in sorted(Path(a.root).glob("*/dr_*")):
        if not (run / "dataset_builder_state.json").exists():
            continue
        try:
            rows += [{"system": run.parent.name, **r} for r in acquisition_profile(run)]
        except Exception as e:
            print(f"FAIL {run}: {type(e).__name__}: {e}")
    if rows:
        with open(a.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
    print(f"{len(rows)} rows -> {a.out}")


if __name__ == "__main__":
    main()
