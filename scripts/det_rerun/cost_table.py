#!/usr/bin/env python
"""Wall time per epoch and per run from run artifacts (tracker H05).

Epoch e's wall time is the gap between consecutive artifacts_v2.json writes
(epoch 0 from provenance.json's start). GPU model and host come from
provenance.json, so rows are only comparable within one GPU class.
Usage: cost_table.py [--root EXP/det_rerun] [--out det_rerun_cost.csv]
"""
import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

ROOT = "/common/users/shared/pracsys/adaptive_roa_experiments/det_rerun"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--out", default=f"{ROOT}/det_rerun_cost.csv")
    a = ap.parse_args()
    rows = []
    for run in sorted(Path(a.root).glob("*/dr_*")):
        prov = run / "provenance.json"
        arts = sorted(run.glob("epoch_*/artifacts_v2.json"))
        if not prov.exists() or not arts:
            continue
        p = json.loads(prov.read_text())
        t_prev = datetime.fromisoformat(p["started_utc"]).timestamp()
        for art in arts:
            t = art.stat().st_mtime
            a_ = json.loads(art.read_text())
            rows.append({"system": run.parent.name, "run": run.name, "epoch": a_["epoch"],
                         "train_trajectories": a_["train_trajectories"], "epoch_minutes": round((t - t_prev) / 60, 2),
                         "gpu": p.get("gpu"), "host": p.get("host")})
            t_prev = t
    if rows:
        with open(a.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
    print(f"{len(rows)} rows -> {a.out}")
    for r in rows:
        print(r["run"], r["epoch"], r["train_trajectories"], r["epoch_minutes"], r["gpu"])


if __name__ == "__main__":
    main()
