#!/usr/bin/env python
"""Per-run status of the det_v2 campaign, counted from disk.

For each manifest row (rank order) it reads the shared ledger and the run dir:

  not launched   no ledger row (or the last launch failed to submit)
  claimed        a launch or resume is submitting right now, or died while submitting
  launched       placed, nothing banked yet
  running        placed, k of 11 epochs banked
  queued         an Amarel run whose job is PENDING in the last sync_amarel.sh snapshot
  stalled?       placed and short of 11 epochs, with logs and the open epoch dir silent for
                 longer than 1.5x the last epoch and at least STALE_MIN minutes: check it
  cancelled      moved off a queue (rebalance) and not placed again
  finished       epoch_000 .. epoch_010 all hold artifacts_v2.json

Epochs are counted from the run dir only (Amarel runs from the mirror sync_amarel.sh
keeps). The run dir is the ledger's; a run with no ledger row is looked up as
<DET_V2_ROOT>/*/<name>_s<seed>, or composed from its config with --compose.

    status.py [--tsv] [--compose] [--n-epochs 11]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import det_v2_lib as lib  # noqa: E402

STALE_MIN = 90
QUEUE_MAX_AGE = 30 * 60
TSV_COLUMNS = ["id", "name", "rank", "owner", "state", "banked", "n_epochs", "where", "hostgpu", "job",
               "user", "commit", "run_dir"]


def amarel_pending(log_dir: Path, now: float | None = None) -> set[str]:
    """Run names PENDING in any account's fresh squeue snapshot (amarel_queue_<user>.txt)."""
    now = now or time.time()
    out = set()
    for snap in log_dir.glob("amarel_queue_*.txt"):
        if now - snap.stat().st_mtime > QUEUE_MAX_AGE:
            continue
        out |= {f[0] for f in (l.split() for l in snap.read_text().splitlines()) if len(f) >= 2 and f[1] == "PENDING"}
    return out


def silent_for(run_dir: Path, k: int, log: str) -> tuple[float, float] | None:
    """(seconds since the run last wrote a log or epoch file, length of its last epoch)."""
    logs = [Path(log), Path(log).with_suffix(".err")] if log not in ("", "-") else []
    files = [q for q in logs if q.exists()]
    open_epoch = run_dir / f"epoch_{k:03d}"
    if open_epoch.exists():
        files += [q for q in open_epoch.rglob("*") if q.is_file()]
    if not files:
        return None
    arts = sorted(run_dir.glob("epoch_*/artifacts_v2.json"))
    last_epoch = arts[-1].stat().st_mtime - arts[-2].stat().st_mtime if len(arts) > 1 else 0.0
    return time.time() - max(q.stat().st_mtime for q in files), last_epoch


def row_status(env: dict, r: dict, last: dict | None, pending: set[str], n_epochs: int, compose: bool) -> dict:
    run_dir = None
    if last and last.get("run_dir", "-") != "-":
        run_dir = Path(last["run_dir"])
    if run_dir is None:
        run_dir = lib.find_run_dir(env, r["name"])
    if run_dir is None and compose:
        try:
            run_dir = Path(lib.compose_row(env, r).output_dir)
        except Exception:
            run_dir = None
    k = lib.epoch_count(run_dir, n_epochs) if run_dir else 0
    action = last["action"] if last else ""
    if run_dir and k == n_epochs:
        state = "finished"
    elif not last or action == "abandon":
        state = "not launched"
    elif action == "claim":
        state = "claimed"
    elif action == "cancel":
        state = "cancelled"
    else:
        state = "running" if k else "launched"
        if last["where"] == "amarel" and r["name"] in pending:
            state = "queued"
        elif run_dir and k:
            s = silent_for(run_dir, k, last.get("log", "-"))
            if s and s[0] > max(60 * STALE_MIN, 1.5 * s[1]):
                state = "stalled?"
    out = {c: r.get(c, "") for c in ("id", "name", "rank", "owner")}
    out.update(state=state, banked=k, n_epochs=n_epochs, run_dir=str(run_dir or "-"))
    for c in ("where", "hostgpu", "job", "user", "commit"):
        out[c] = last.get(c, "-") if last else "-"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", action="store_true", help="tab-separated output with a header")
    ap.add_argument("--compose", action="store_true", help="compose configs to find unlaunched run dirs")
    ap.add_argument("--n-epochs", type=int, default=lib.N_EPOCHS)
    a = ap.parse_args()
    env = lib.load_user_env()
    rows = sorted(lib.read_manifest(env["MANIFEST"]), key=lambda r: int(r["rank"]))
    latest = lib.latest_rows(env["LEDGER"])
    pending = amarel_pending(Path(env["LOG_DIR"]))
    out = [row_status(env, r, latest.get(r["name"]), pending, a.n_epochs, a.compose) for r in rows]
    if a.tsv:
        print("\t".join(TSV_COLUMNS))
        for s in out:
            print("\t".join(str(s[c]) for c in TSV_COLUMNS))
        return
    counts: dict[str, int] = {}
    for s in out:
        counts[s["state"]] = counts.get(s["state"], 0) + 1
        where = "" if s["where"] == "-" else f"{s['where']} {s['hostgpu']} job {s['job']} ({s['user']})"
        print(f"{s['rank']:>3} {s['name']:32s} {s['state']:13s} {s['banked']:>2}/{s['n_epochs']}  {where}")
    print(" ".join(f"{k}={v}" for k, v in sorted(counts.items())))


if __name__ == "__main__":
    main()
