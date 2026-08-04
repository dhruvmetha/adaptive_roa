#!/usr/bin/env python
"""Append-only experiment log for the ensemble-epistemic campaign.

Every launched run gets one JSON line. `code_hash` is a content hash over the
source files that determine a run's behaviour: the previous campaign rsynced
UNCOMMITTED working-tree changes to a second cluster, and there is now no way to
reconstruct which code any given run used. A git SHA would not have helped.

HASHED intentionally lists files that may not exist yet (e.g. the ensemble
flow-matching trainer, which lands in a later task). `code_hash` hashes the
literal bytes b"<missing>" for any path that doesn't exist, so the recorded
hash changes the moment such a file appears -- there is no need to keep this
list in sync with what has landed so far.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
HASHED = [
    "adaptive_roa/adaptive_v2/strategy/decomposition.py",
    "adaptive_roa/adaptive_v2/strategy/uncertainty_scores.py",
    "adaptive_roa/adaptive_v2/probability/ensemble_prob.py",
    "adaptive_roa/adaptive_v2/trainers/ensemble_flow_matching_trainer.py",
    "adaptive_roa/adaptive_v2/eval/full_roa.py",
]


def code_hash(paths: list[str] | None = None) -> str:
    h = hashlib.sha256()
    for rel in sorted(paths or HASHED):
        p = REPO / rel
        h.update(rel.encode())
        h.update(p.read_bytes() if p.exists() else b"<missing>")
    return h.hexdigest()[:12]


def _read(log: Path) -> list[dict]:
    if not log.exists():
        return []
    return [json.loads(l) for l in log.read_text().splitlines() if l.strip()]


def _write(log: Path, recs: list[dict]) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text("".join(json.dumps(r) + "\n" for r in recs))


def cmd_append(a) -> int:
    log = Path(a.log)
    recs = _read(log)
    recs.append({
        "run_id": a.run_id,
        "launched_at": datetime.now().isoformat(timespec="seconds"),
        "system": a.system, "level": a.level, "predictor": a.predictor, "arm": a.arm,
        "score_mode": a.score_mode, "seed": a.seed, "n_members": a.n_members,
        "k_acq": a.k_acq, "d2_ratio": a.d2_ratio, "cluster": a.cluster,
        "job_id": a.job_id, "output_dir": a.output_dir,
        "code_hash": code_hash(), "status": "launched", "notes": a.notes or "",
    })
    _write(log, recs)
    print(f"appended {a.run_id} ({recs[-1]['code_hash']})")
    return 0


def cmd_update(a) -> int:
    log = Path(a.log)
    recs = _read(log)
    hit = [r for r in recs if r["run_id"] == a.run_id]
    if not hit:
        print(f"no run with run_id {a.run_id!r} in {log}", file=sys.stderr)
        return 1
    for r in hit:
        r["status"] = a.status
        if a.notes:
            r["notes"] = (r.get("notes", "") + " | " + a.notes).strip(" |")
    _write(log, recs)
    print(f"{a.run_id} -> {a.status}")
    return 0


def cmd_report(a) -> int:
    recs = _read(Path(a.log))
    if not recs:
        print("(empty log)")
        return 0
    by: dict[str, list[dict]] = {}
    for r in recs:
        by.setdefault(r["status"], []).append(r)
    for status in sorted(by):
        print(f"\n== {status} ({len(by[status])}) ==")
        for r in sorted(by[status], key=lambda x: x["run_id"]):
            print(f"  {r['run_id']:<28} {r['predictor']:<4} {r['level']:<6} "
                  f"{r['arm']:<10} {r['cluster']:<8} job={r['job_id']:<10} "
                  f"code={r['code_hash']}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_a = sub.add_parser("append")
    ap_a.add_argument("--log", required=True)
    for f in ("run-id", "system", "level", "predictor", "arm", "cluster",
              "job-id", "output-dir"):
        ap_a.add_argument(f"--{f}", required=True)
    ap_a.add_argument("--score-mode", default="")
    ap_a.add_argument("--seed", type=int, default=42)
    ap_a.add_argument("--n-members", type=int, default=5)
    ap_a.add_argument("--k-acq", type=int, default=20)
    ap_a.add_argument("--d2-ratio", type=float, default=1.0)
    ap_a.add_argument("--notes", default="")
    ap_a.set_defaults(func=cmd_append)

    ap_u = sub.add_parser("update-status")
    ap_u.add_argument("--log", required=True)
    ap_u.add_argument("--run-id", required=True)
    ap_u.add_argument("--status", required=True)
    ap_u.add_argument("--notes", default="")
    ap_u.set_defaults(func=cmd_update)

    ap_r = sub.add_parser("report")
    ap_r.add_argument("--log", required=True)
    ap_r.set_defaults(func=cmd_report)

    a = ap.parse_args()
    return a.func(a)


if __name__ == "__main__":
    raise SystemExit(main())
