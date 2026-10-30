#!/usr/bin/env python
"""Abstention and error diagnostics for every det-rerun run dir (tracker H07).

Usage: diagnostics.py [--root EXP/det_rerun] [--out det_rerun_diagnostics.csv]
"""
import argparse
import csv
from pathlib import Path

from adaptive_roa.adaptive_v2.eval.store_diag import diagnose_run

ROOT = "/common/users/shared/pracsys/adaptive_roa_experiments/det_rerun"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--out", default=f"{ROOT}/det_rerun_diagnostics.csv")
    a = ap.parse_args()
    rows = []
    for run in sorted(Path(a.root).glob("*/dr_*")):
        if (run / "partx_state.pkl").exists() or not (run / "resolved_config.yaml").exists():
            continue
        try:
            rows += [{"system": run.parent.name, **r} for r in diagnose_run(run)]
        except Exception as e:
            print(f"FAIL {run}: {type(e).__name__}: {e}")
    if rows:
        with open(a.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
    print(f"{len(rows)} rows -> {a.out}")
    for r in rows:
        print(r["run"], r["epoch"], r["train_trajectories"], "inv", round(r["invalid_share"], 3),
              "succ_in_inv", round(r["success_share_in_invalid"], 3), "vbw", round(r["valid_but_wrong_share_of_committed"], 4),
              "auc_pinv_bad", round(r["auroc_p_invalid_ranks_bad"], 3))


if __name__ == "__main__":
    main()
