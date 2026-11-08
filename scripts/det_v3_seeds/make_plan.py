#!/usr/bin/env python
"""Build scripts/det_v3_seeds/run_plan.json from det_v3's plan (Sumanth, 2026-11-08).

The 60 det_v3 main rows, each at two more seeds (det_v3 itself ran seed 99512). Nothing else
changes: a row keeps its det_v3 setting, model, arm, owner and estimated hours. Names add the
repeat tag (m05_q3dlqr_fm_qa -> m05_q3dlqr_fm_qa_r2 at seed 33407, _r3 at seed 24475); ids put the
repeat number before the det_v3 number (M05 -> S205, S305), and base_id keeps the det_v3 id.
Ranks are seed-major: every seed-33407 row first, in det_v3 main order (longest first), then
every seed-24475 row, so a short campaign still finishes one whole extra seed.

    make_plan.py [--out scripts/det_v3_seeds/run_plan.json]
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SEEDS = ((2, 33407), (3, 24475))        # (repeat number, seed): the other two paper seeds (tracker D04);
                                         # repeat 1 is det_v3 at 99512. Their pool files already exist.
KEEP = ("setting", "model", "arm", "hours", "owner")


def plan(v3: dict) -> dict:
    base = sorted(v3["main"], key=lambda r: r["rank"])
    rows = []
    for rep, seed in SEEDS:
        for r in base:
            rows.append({"id": f"S{rep}{int(r['id'][1:]):02d}", "base_id": r["id"], "base_name": r["name"],
                         "seed": seed, "name": f"{r['name']}_r{rep}", **{k: r[k] for k in KEEP},
                         "kind": "main", "rank": len(rows) + 1})
    return {"main": rows, "abl": []}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "scripts/det_v3_seeds/run_plan.json")
    a = ap.parse_args()
    p = plan(json.loads((ROOT / "scripts/det_v3/run_plan.json").read_text()))
    a.out.write_text(json.dumps(p, indent=1) + "\n")
    rows = p["main"]
    tot = {(o, s): round(sum(r["hours"] for r in rows if r["owner"] == o and r["seed"] == s), 1)
           for o in ("Sumanth", "Dhruv") for _, s in SEEDS}
    print(f"{len(rows)} runs; est GPU-h by (owner, seed) {tot}")


if __name__ == "__main__":
    main()
