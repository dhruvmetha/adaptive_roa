#!/usr/bin/env python
"""Section 6 certificates (6a, 6c) for every banked epoch of every det_v2 run, from the prediction store.

Reads each epoch's predictions_val.npz (testing order), predictions_cal.npz (certificate) and
full_roa_per_point.npz (test report), plus tau* from artifacts_v2.json for 6a's F_0. Writes one
JSON row per (run, epoch, certificate). Unchanged epochs are reused from --cache. Read-only on run dirs.

    certify.py --out certificates.jsonl [--cache cert_cache.json] [--jobs 8] [--alpha 0.05 --beta 0.05 --delta 0.05]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from adaptive_roa.adaptive_v2.eval import certificates as C  # noqa: E402

EXP = Path(os.environ.get("DET_V2_EXP", "/common/users/shared/pracsys/adaptive_roa_experiments"))
PLAN = ROOT / "scripts" / "det_v2" / "run_plan.json"
# The tracker's Source column for the reused det-rerun rows (Part-X M11/M22/M33 are rerun in det_v2).
REUSED = {
    "M06": "det_rerun/q3d30/dr_q3d30_mlpclf_uniform_s99512", "M07": "det_rerun/q3d30/dr_q3d30_bnnclf_uniform_s99512",
    "M08": "det_rerun/q3d30/dr_q3d30_mlpreg_uniform_s99512", "M09": "det_rerun/q3d30/dr_q3d30_bnnreg_uniform_s99512",
    "M10": "det_rerun/q3d30/dr_q3d30_fm_uniform_s99512",
    "M17": "det_rerun/q3drl40/dr_q3drl40_mlpclf_uniform_s99512", "M18": "det_rerun/q3drl40/dr_q3drl40_bnnclf_uniform_s99512",
    "M19": "det_rerun/q3drl40/dr_q3drl40_mlpreg_uniform_s99512", "M20": "det_rerun/q3drl40/dr_q3drl40_bnnreg_uniform_s99512",
    "M21": "det_rerun/q3drl40/dr_q3drl40_fm_uniform_s99512/rescored_ternary",
    "M28": "det_rerun/q2d/dr_q2d_mlpclf_uniform_s99512", "M29": "det_rerun/q2d/dr_q2d_bnnclf_uniform_s99512",
    "M30": "det_rerun/q2d/dr_q2d_mlpreg_uniform_s99512", "M31": "det_rerun/q2d/dr_q2d_bnnreg_uniform_s99512",
    "M32": "det_rerun/q2d/dr_q2d_fm_uniform_s99512",
}
FILES = ("predictions_val.npz", "predictions_cal.npz", "full_roa_per_point.npz", "artifacts_v2.json")


def run_dirs() -> list[tuple[dict, Path]]:
    plan = json.load(open(PLAN))
    out = []
    for r in plan["main"] + plan["abl"]:
        if r["reused"]:
            if r["id"] not in REUSED:
                continue                                    # reused Part-X: no val/cal store; rerun in det_v2
            d = EXP / REUSED[r["id"]]
        else:
            hits = glob.glob(str(EXP / "det_v2" / "*" / f"{r['name']}_s99512"))
            if not hits:
                continue
            d = Path(hits[0])
        out.append((r, d))
    return out


def epoch_key(ep: Path) -> str | None:
    try:
        return ";".join(f"{f}:{os.path.getmtime(ep / f)}" for f in FILES)
    except OSError:
        return None


def one_epoch(args) -> list[dict] | None:
    ep, alpha, beta_, delta = args
    ep = Path(ep)
    try:
        val, cal = np.load(ep / "predictions_val.npz"), np.load(ep / "predictions_cal.npz")
        test = np.load(ep / "full_roa_per_point.npz", allow_pickle=True)
        ts = json.load(open(ep / "artifacts_v2.json")).get("threshold_state") or {}
    except (OSError, ValueError):
        return None                                         # missing or unreadable (another account's 0600 file)
    classifier = "n_success" not in val.files
    V, Cc, T = C.split_probs(val), C.split_probs(cal), C.split_probs(test)
    tau = (ts.get("tau_s_star"), ts.get("tau_f_star"))
    res = C.certify(V, Cc, T, classifier=classifier, tau=tau, alpha=alpha, beta_=beta_, delta=delta)
    rows = []
    a = res["6a"]
    rows.append({"cert": "6a", "passed": a["passed"], "steps": a["steps"], "n_order": a["n_order"],
                 "t_s": a.get("t_s"), "t_f": a.get("t_f"), "cal_n": a.get("cal_n"), "cal_fp": a.get("cal_fp"),
                 **{k: a[k] for k in a if k.startswith("test_")}})
    c = res["6c"]
    rows.append({"cert": "6c", "passed": c["S"]["passed"] and c["F"]["passed"],
                 "S_passed": c["S"]["passed"], "F_passed": c["F"]["passed"],
                 "t_s": c["S"].get("t_s"), "t_f": c["S"].get("t_f"), "u_s": c["F"].get("u_s"), "u_f": c["F"].get("u_f"),
                 "cal_n": c["S"].get("cal_n"), "cal_fp": c["S"].get("cal_fp"), "cal_F_succ": c["F"].get("cal_succ"),
                 **{k: c[k] for k in c if k.startswith("test_")}})
    for r in rows:
        r.update(classifier=classifier, alpha=alpha, beta=beta_, delta=delta)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--beta", type=float, default=0.05)
    ap.add_argument("--delta", type=float, default=0.05)
    a = ap.parse_args()
    tag = f"{a.alpha}/{a.beta}/{a.delta}"
    cache = json.load(open(a.cache)) if a.cache and os.path.exists(a.cache) else {}
    todo, plan_rows = [], []
    for r, d in run_dirs():
        for ep in sorted(d.glob("epoch_*")):
            key = epoch_key(ep)
            if key is None:
                continue
            ck = f"{ep}|{tag}"
            plan_rows.append((r, ep, ck))
            if cache.get(ck, {}).get("key") != key:
                todo.append((ck, key, (str(ep), a.alpha, a.beta, a.delta)))
    skipped = []
    with ProcessPoolExecutor(max_workers=a.jobs) as pool:
        for (ck, key, args), rows in zip(todo, pool.map(one_epoch, [t[2] for t in todo], chunksize=4)):
            if rows is not None:
                cache[ck] = {"key": key, "rows": rows}
            else:
                skipped.append(args[0])                  # missing store file, or another account's 0600 file
    n = 0
    with open(a.out + ".tmp", "w") as f:
        for r, ep, ck in plan_rows:
            hit = cache.get(ck)
            if not hit:
                continue
            art = json.load(open(ep / "artifacts_v2.json"))
            for row in hit["rows"]:
                f.write(json.dumps({"id": r["id"], "run": r["name"], "setting": r["setting"], "model": r["model"],
                                    "arm": r.get("arm") or ("U_H" if r.get("ab") == "A7" else "q_A"),
                                    "ablation": r.get("ab", "main"), "change": r.get("change", ""),
                                    "reused": bool(r["reused"]), "epoch": int(art["epoch"]),
                                    "train": int(art["train_trajectories"]), **row}) + "\n")
                n += 1
    os.replace(a.out + ".tmp", a.out)
    if a.cache:
        with open(a.cache + ".tmp", "w") as f:
            json.dump(cache, f)
        os.replace(a.cache + ".tmp", a.cache)
    print(f"{n} certificate rows from {len(plan_rows)} epochs ({len(todo)} computed), written to {a.out}")
    if skipped:
        print(f"skipped {len(skipped)} epochs that could not be read, e.g. {skipped[0]}")


if __name__ == "__main__":
    main()
