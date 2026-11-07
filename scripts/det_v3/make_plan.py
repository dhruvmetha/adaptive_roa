#!/usr/bin/env python
"""Build scripts/det_v3/run_plan.json from det_v2's plan (Sumanth, 2026-11-07).

Changes from det_v2: every run reruns (no reused rows); Part-X is recertified offline, not rerun;
A9 is dropped; A10 (batch selection) and A11 (initial uniform batch) are new. Ranks: all main runs
first, then all ablations; within each, longest first (estimated hours), then ID. Owners by setting:
Sumanth q3d-ppo, q2d-ppo; Dhruv q3d-lqr, cp-lqr, cp-ppo, pen-lqr (the det_v2 ablation split).

    make_plan.py HOURS_JSON        # HOURS_JSON: det_v2 median wall hours per "setting|model"
"""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
V2 = json.loads((ROOT / "scripts/det_v2/run_plan.json").read_text())
HOURS = json.loads(Path(sys.argv[1]).read_text())
SCALE = 1.25                      # every trajectory trains now (no 20% threshold-fit slice)
OWNER = {"q3d-ppo": "Sumanth", "q2d-ppo": "Sumanth", "q3d-lqr": "Dhruv", "cp-lqr": "Dhruv",
         "cp-ppo": "Dhruv", "pen-lqr": "Dhruv"}
SHORT = {"q3d-lqr": "q3dlqr", "q3d-ppo": "q3dppo", "q2d-ppo": "q2dppo", "cp-lqr": "cplqr",
         "cp-ppo": "cpppo", "pen-lqr": "penlqr"}
MTAG = {"MLP classifier": "mlpclf", "BNN classifier": "bnnclf", "MLP final-state": "mlpreg",
        "BNN final-state": "bnnreg", "FM": "fm"}
NEW = [("A10", "plain top-k (tie_tol = 0)", "a10topk"), ("A10", "proportional sampling", "a10prop"),
       ("A11", "initial batch x 0.5", "a11half"), ("A11", "initial batch x 2", "a11double")]

def est(r):
    return max(0.1, round(SCALE * HOURS[f"{r['setting']}|{r['model']}"], 1))

rows = []
for r in V2["main"]:
    if r["model"] == "Part-X":
        continue
    rows.append({k: r[k] for k in ("id", "setting", "model", "arm", "name")} | {"kind": "main"})
for r in V2["abl"]:
    if r.get("ab") == "A9":
        continue
    rows.append({k: r[k] for k in ("id", "ab", "setting", "model", "change", "name")} | {"kind": "abl"})
nxt = 1 + max(int(r["id"][1:]) for r in V2["abl"])
for ab, change, tag in NEW:
    for st in ("q3d-lqr", "q3d-ppo", "q2d-ppo", "cp-lqr", "cp-ppo", "pen-lqr"):
        for m in ("MLP classifier", "BNN classifier", "MLP final-state", "BNN final-state", "FM"):
            rows.append({"id": f"X{nxt}", "ab": ab, "setting": st, "model": m, "change": change,
                         "name": f"x{nxt}_{SHORT[st]}_{MTAG[m]}_qa_{tag}", "kind": "abl"})
            nxt += 1
for r in rows:
    r["hours"], r["owner"] = est(r), OWNER[r["setting"]]
    r["tier"] = 1 if r["kind"] == "main" else 2
order = sorted(rows, key=lambda r: (r["tier"], -r["hours"], int(r["id"][1:]) if r["id"][0] == "M" else 1000 + int(r["id"][1:])))
for i, r in enumerate(order, 1):
    r["rank"] = i
plan = {"main": [r for r in order if r["kind"] == "main"], "abl": [r for r in order if r["kind"] == "abl"]}
out = ROOT / "scripts/det_v3/run_plan.json"
out.write_text(json.dumps(plan, indent=1) + "\n")
tot = {o: round(sum(r["hours"] for r in order if r["owner"] == o), 1) for o in ("Sumanth", "Dhruv")}
print(f"{len(order)} runs ({len(plan['main'])} main, {len(plan['abl'])} ablation); est GPU-h {round(sum(r['hours'] for r in order),1)}; by owner {tot}")
