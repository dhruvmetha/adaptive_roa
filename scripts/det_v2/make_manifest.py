#!/usr/bin/env python
"""Write scripts/<campaign>/manifest.tsv (default det_v2) from the campaign run plan.

One row per to-do run of scripts/det_v2/run_plan.json (keys "main" and "abl"): the
rows with a rank. Reused rows have no rank and are skipped. Columns:

    id  name  rank  owner  config  script  mem  est_hours

config is the run's own Hydra experiment (configs/adaptive_v2/experiment/det_v2/runs/<name>.yaml),
script is the runner, and mem is the det_rerun request for the same model on the
matching old system (det_rerun_mem.tsv, extracted from the det_rerun manifest).

    make_manifest.py                                   # rebuild manifest.tsv from the repo copies
    make_manifest.py --det-rerun-manifest <path>       # re-extract det_rerun_mem.tsv first
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
COLUMNS = ["id", "name", "rank", "owner", "config", "script", "mem", "est_hours"]

# det_v2 setting -> det_rerun system key whose cells ran the same dataset and controller.
SETTING_TO_OLD = {"q3d-lqr": "q3d30", "q3d-ppo": "q3drl40", "q2d-ppo": "q2d",
                  "cp-lqr": "cp", "cp-ppo": "cpppo", "pen-lqr": "pen"}
# run-plan model -> det_rerun arm (predictor) name.
MODEL_TO_ARM = {"FM": "fm_det", "MLP classifier": "det_mlp_clf", "BNN classifier": "det_bnn_clf",
                "MLP final-state": "det_mlp_reg", "BNN final-state": "det_bnn_reg",
                "Part-X": "partx_faithful"}
RUN_SCRIPT = "scripts/run_adaptive.py"
PARTX_SCRIPT = "scripts/run_partx_faithful.py"


def _gb(mem: str) -> int:
    return int(mem.rstrip("G"))


def extract_mem(det_rerun_manifest: Path) -> dict[tuple[str, str], str]:
    """(old system, arm) -> mem, the largest request over that pair's det_rerun cells."""
    out: dict[tuple[str, str], str] = {}
    with open(det_rerun_manifest) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            key = (r["system"], r["arm"])
            if key[0] in SETTING_TO_OLD.values() and key[1] in MODEL_TO_ARM.values():
                if key not in out or _gb(r["mem"]) > _gb(out[key]):
                    out[key] = r["mem"]
    return out


def write_mem_table(mem: dict[tuple[str, str], str], path: Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["system", "arm", "mem"])
        for (system, arm), m in sorted(mem.items()):
            w.writerow([system, arm, m])


def read_mem_table(path: Path) -> dict[tuple[str, str], str]:
    with open(path) as f:
        return {(r["system"], r["arm"]): r["mem"] for r in csv.DictReader(f, delimiter="\t")}


def is_partx(run: dict) -> bool:
    return run.get("arm") == "Part-X" or run["model"] == "Part-X"


def rows(plan: dict, mem: dict[tuple[str, str], str], campaign: str = "det_v2") -> list[dict]:
    out = []
    for run in plan["main"] + plan["abl"]:
        if run.get("rank") is None or run.get("reused"):
            continue
        old, arm = SETTING_TO_OLD[run["setting"]], MODEL_TO_ARM[run["model"]]
        if (old, arm) not in mem:
            raise SystemExit(f"{run['name']}: no det_rerun mem for {arm} on {old}")
        out.append({"id": run["id"], "name": run["name"], "rank": int(run["rank"]), "owner": run["owner"],
                    "config": f"+experiment={campaign}/runs/{run['name']}",
                    "script": PARTX_SCRIPT if is_partx(run) else RUN_SCRIPT,
                    "mem": mem[(old, arm)], "est_hours": run["hours"]})
    out.sort(key=lambda r: r["rank"])
    names = [r["name"] for r in out]
    ranks = [r["rank"] for r in out]
    if len(set(names)) != len(names):
        raise SystemExit("duplicate run names in the plan")
    if ranks != list(range(1, len(out) + 1)):
        raise SystemExit(f"ranks are not 1..{len(out)}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", default=os.environ.get("DET_CAMPAIGN", "det_v2"),
                    help="scripts/<campaign>/ holds the plan and gets the manifest (default det_v2)")
    ap.add_argument("--plan", default=None, help="default scripts/<campaign>/run_plan.json")
    ap.add_argument("--mem-table", default=str(HERE / "det_rerun_mem.tsv"))
    ap.add_argument("--det-rerun-manifest", default=None,
                    help="re-extract --mem-table from this det_rerun manifest.tsv first")
    ap.add_argument("--out", default=None, help="default scripts/<campaign>/manifest.tsv")
    a = ap.parse_args()
    a.plan = a.plan or str(HERE.parent / a.campaign / "run_plan.json")
    a.out = a.out or str(HERE.parent / a.campaign / "manifest.tsv")
    if a.det_rerun_manifest:
        write_mem_table(extract_mem(Path(a.det_rerun_manifest)), Path(a.mem_table))
    plan = json.loads(Path(a.plan).read_text())
    out = rows(plan, read_mem_table(Path(a.mem_table)), a.campaign)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(out)
    print(f"{len(out)} rows -> {a.out}")


if __name__ == "__main__":
    main()
