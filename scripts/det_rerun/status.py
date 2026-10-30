#!/usr/bin/env python
"""Per-cell status of the det-paper rerun, counted from disk (tracker rules).

A cell is DONE only when its final epoch's artifacts_v2.json exists, reports the
final budget, and the prediction-store check passes. A cell short of its epochs
whose log has not been written for STALE_MIN minutes shows as "stalled?": check
its process before trusting "running". Usage: status.py [--check]
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path

HERE = Path("/common/home/st1122/Projects/adaptive_roa/slurm_logs/det_rerun")
STALE_MIN = 90      # an FM eval writes its tqdm line every few seconds; training logs per epoch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="run the store check on finished cells")
    a = ap.parse_args()
    rows = list(csv.DictReader(open(HERE / "manifest.tsv"), delimiter="\t"))
    launched = {}
    if (HERE / "launched.tsv").exists():
        for line in open(HERE / "launched.tsv"):
            f = line.rstrip("\n").split("\t")
            launched[f[0]] = f
    counts = {}
    for r in rows:
        out = Path(re.search(r"output_dir=(\S+)", r["overrides"]).group(1))
        arts = sorted(out.glob("epoch_*/artifacts_v2.json"))
        m = re.search(r"(?:^| )n_epochs=(\d+)", r["overrides"])
        n_ep = int(m.group(1)) if m else 11
        state = "not launched" if r["name"] not in launched else "launched"
        detail = ""
        if arts:
            last = json.loads(arts[-1].read_text())
            detail = f"epochs {len(arts)}/{n_ep}, budget {last.get('train_trajectories')}"
            state = "running" if len(arts) < n_ep else "finished"
            log = Path(launched[r["name"]][5]) if r["name"] in launched else None
            if state == "running" and log and log.exists() and time.time() - log.stat().st_mtime > 60 * STALE_MIN:
                state = "stalled?"
                detail += f" | log silent {int((time.time() - log.stat().st_mtime) / 60)} min"
            if state == "finished" and a.check:
                from adaptive_roa.adaptive_v2.eval.store_check import check_run
                problems = check_run(out)
                state = "DONE" if not problems else "FAILED CHECK"
                if problems:
                    detail += " | " + problems[0]
        where = "" if r["name"] not in launched else " ".join(launched[r["name"]][1:4])
        counts[state] = counts.get(state, 0) + 1
        print(f"{r['name']:40s} {state:14s} {where:30s} {detail}")
    print(" ".join(f"{k}={v}" for k, v in sorted(counts.items())))


if __name__ == "__main__":
    main()
