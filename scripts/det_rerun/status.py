#!/usr/bin/env python
"""Per-cell status of the det-paper rerun, counted from disk (tracker rules).

A cell is DONE only when its final epoch's artifacts_v2.json exists, reports the
final budget, and the prediction-store check passes. A cell short of its epochs
whose logs (stdout and, for iLab, the sibling .err) and in-progress epoch dir
(trainer metrics, checkpoints) have been silent for longer than 1.5x its last
epoch, and at least STALE_MIN minutes, shows as "stalled?":
check its process before trusting "running". Some trainers print nothing for a
whole epoch, and epochs lengthen as the train set grows. An Amarel cell whose job is
PENDING in sync_amarel.sh's last squeue snapshot (a resume waiting for a slot) shows
as "queued" instead, since its silence is expected. Usage: status.py [--check]
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
QUEUE_MAX_AGE = 30 * 60   # an older squeue snapshot says nothing about the queue now


def amarel_pending(path: Path, now: float | None = None) -> set[str]:
    """Job names PENDING in sync_amarel.sh's squeue snapshot ("<name> <state>" lines), if fresh."""
    if not path.exists() or (now or time.time()) - path.stat().st_mtime > QUEUE_MAX_AGE:
        return set()
    return {f[0] for f in (line.split() for line in path.read_text().splitlines())
            if len(f) >= 2 and f[1] == "PENDING"}


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
    pending = amarel_pending(HERE / "amarel_queue.txt")
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
            logs = [q for q in (log, log and log.with_suffix(".err")) if q and q.exists()]
            logs += [q for q in (out / f"epoch_{len(arts):03d}").rglob("*") if q.is_file()]
            world = launched.get(r["name"], ["", ""])[1]
            if state == "running" and world == "amarel" and r["name"] in pending:
                state = "queued"
                detail += " | resume queued on Amarel"
            elif state == "running" and logs:
                silent = time.time() - max(q.stat().st_mtime for q in logs)
                last_epoch = (arts[-1].stat().st_mtime - arts[-2].stat().st_mtime) if len(arts) > 1 else 0.0
                if silent > max(60 * STALE_MIN, 1.5 * last_epoch):
                    state = "stalled?"
                    detail += f" | silent {int(silent / 60)} min, last epoch {int(last_epoch / 60)} min"
            if state == "finished" and a.check:
                from adaptive_roa.adaptive_v2.eval.store_check import check_run
                # Part-X writes no prediction store; its run ends with final_results.json.
                problems = (check_run(out) if r["arm"] != "partx_faithful" else
                            ([] if (out / "final_results.json").exists() else ["no final_results.json"]))
                state = "DONE" if not problems else "FAILED CHECK"
                if problems:
                    detail += " | " + problems[0]
        where = "" if r["name"] not in launched else " ".join(launched[r["name"]][1:4])
        counts[state] = counts.get(state, 0) + 1
        print(f"{r['name']:40s} {state:14s} {where:30s} {detail}")
    print(" ".join(f"{k}={v}" for k, v in sorted(counts.items())))


if __name__ == "__main__":
    main()
