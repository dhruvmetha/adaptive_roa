#!/usr/bin/env python
"""Eval-K check (tracker H04): metrics at K' in {5, 10, 20, 50} subsampled from the stored K.

Usage: k_check.py [--root EXP/det_rerun] [--out det_rerun_k_check.csv]
"""
import argparse
import csv
from pathlib import Path

from adaptive_roa.adaptive_v2.eval.store_k import k_check_run

ROOT = "/common/users/shared/pracsys/adaptive_roa_experiments/det_rerun"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--out", default=f"{ROOT}/det_rerun_k_check.csv")
    a = ap.parse_args()
    rows = []
    for run in sorted(Path(a.root).glob("*/dr_*")):
        if (run / "partx_state.pkl").exists() or not (run / "resolved_config.yaml").exists():
            continue
        try:
            rows += [{"system": run.parent.name, **r} for r in k_check_run(run)]
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
