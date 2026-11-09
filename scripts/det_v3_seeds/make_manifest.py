#!/usr/bin/env python
"""Write scripts/det_v3_seeds/manifest.tsv from scripts/det_v3_seeds/run_plan.json (every block).

The det_v2 columns, then base_id and seed. script, mem and est_hours are copied from the det_v3
manifest row the run repeats, so a repeat asks for what its det_v3 run asked for. (Use this, not
scripts/det_v2/make_manifest.py, which writes neither base_id nor seed.)

    make_manifest.py [--out scripts/det_v3_seeds/manifest.tsv]
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COLUMNS = ["id", "name", "rank", "owner", "config", "script", "mem", "est_hours", "base_id", "seed"]


def rows(plan: dict, v3_manifest: list[dict]) -> list[dict]:
    base = {r["name"]: r for r in v3_manifest}
    out = []
    for run in sorted((r for rows in plan.values() for r in rows), key=lambda r: r["rank"]):
        b = base[run["base_name"]]
        if b["id"] != run["base_id"] or b["owner"] != run["owner"]:
            raise SystemExit(f"{run['name']}: plan and det_v3 manifest disagree on {run['base_name']}")
        out.append({"id": run["id"], "name": run["name"], "rank": run["rank"], "owner": run["owner"],
                    "config": f"+experiment=det_v3_seeds/runs/{run['name']}", "script": b["script"],
                    "mem": b["mem"], "est_hours": b["est_hours"], "base_id": run["base_id"], "seed": run["seed"]})
    for key in ("id", "name"):
        if len({r[key] for r in out}) != len(out):
            raise SystemExit(f"duplicate {key}s in the plan")
    if [r["rank"] for r in out] != list(range(1, len(out) + 1)):
        raise SystemExit(f"ranks are not 1..{len(out)}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", type=Path, default=ROOT / "scripts/det_v3_seeds/run_plan.json")
    ap.add_argument("--out", type=Path, default=ROOT / "scripts/det_v3_seeds/manifest.tsv")
    a = ap.parse_args()
    with open(ROOT / "scripts/det_v3/manifest.tsv") as f:
        v3 = list(csv.DictReader(f, delimiter="\t"))
    out = rows(json.loads(a.plan.read_text()), v3)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(out)
    print(f"{len(out)} rows -> {a.out}")


if __name__ == "__main__":
    main()
