#!/usr/bin/env python
"""Build scripts/det_v3_seeds/run_plan.json from det_v3's plan (Sumanth, 2026-11-08).

Two blocks of det_v3 rows, each at two more seeds (det_v3 itself ran seed 99512):

  main   the 60 det_v3 main rows, ranks 1-120. Ids put the repeat number before the det_v3 number
         (M05 -> S205, S305).
  abl    the 30 det_v3 A7 rows, U_H instead of q_A (X34-X63), ranks 121-180, added ~19:10 so every
         row has both a q_A and a U_H run per seed. Ids are U + repeat + X number (X43 -> U243, U343).

Nothing else changes: a row keeps its det_v3 setting, model, arm or ablation, owner and estimated
hours, and base_id keeps the det_v3 id. Names add the repeat tag (m05_q3dlqr_fm_qa -> _r2 at seed
33407, _r3 at seed 24475). Within each block ranks are seed-major: every seed-33407 row first, in
det_v3 order (longest first), then every seed-24475 row, so a short campaign still finishes one
whole extra seed.

    make_plan.py [--out scripts/det_v3_seeds/run_plan.json]
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SEEDS = ((2, 33407), (3, 24475))        # (repeat number, seed): the other two paper seeds (tracker D04);
                                         # repeat 1 is det_v3 at 99512. Their pool files already exist.
KEEP = ("setting", "model", "arm", "hours", "owner")
KEEP_ABL = ("setting", "model", "ab", "change", "hours", "owner")
ABLATIONS = ("A7",)                      # det_v3 ablations repeated here, by "ab"


def repeats(base: list[dict], prefix: str, keep: tuple, kind: str, first_rank: int) -> list[dict]:
    rows = []
    for rep, seed in SEEDS:
        for r in base:
            rows.append({"id": f"{prefix}{rep}{int(r['id'][1:]):02d}", "base_id": r["id"], "base_name": r["name"],
                         "seed": seed, "name": f"{r['name']}_r{rep}", **{k: r[k] for k in keep},
                         "kind": kind, "rank": first_rank + len(rows)})
    return rows


def plan(v3: dict) -> dict:
    main = repeats(sorted(v3["main"], key=lambda r: r["rank"]), "S", KEEP, "main", 1)
    abl = sorted((r for r in v3["abl"] if r.get("ab") in ABLATIONS), key=lambda r: r["rank"])
    return {"main": main, "abl": repeats(abl, "U", KEEP_ABL, "abl", len(main) + 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "scripts/det_v3_seeds/run_plan.json")
    a = ap.parse_args()
    p = plan(json.loads((ROOT / "scripts/det_v3/run_plan.json").read_text()))
    a.out.write_text(json.dumps(p, indent=1) + "\n")
    for kind, rows in p.items():
        tot = {(o, s): round(sum(r["hours"] for r in rows if r["owner"] == o and r["seed"] == s), 1)
               for o in ("Sumanth", "Dhruv") for _, s in SEEDS}
        print(f"{kind}: {len(rows)} runs; est GPU-h by (owner, seed) {tot}")


if __name__ == "__main__":
    main()
