#!/usr/bin/env python
"""Per-epoch metrics of every det-rerun cell, as the JSON files the results dashboard reads.

Writes three files to --out:
  nominal_epochs.json  every lambda*/delta* cell (the main matrix, the 200k extensions, Part-X)
  fprcap_epochs.json   every FPR-capped tau-rule cell (Idea 6, *_fprcap1pct), with its thresholds
  snapshot.json        when the files were made

Metrics are the in-run ones plot_progress.py draws (a folded epoch reads its three-way
re-score). One row per banked epoch:
  setting, system, controller, budget, model, regime, cell, epoch, train, n_epochs,
  status (finished | running), kl, f1, abstain_pct, invalid_pct, uncertain_pct, fpr, fnr[, tau_s, tau_f, tau_inv]

    dashboard_data.py [--out DIR]
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import glob
import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_progress as pp  # noqa: E402

# setting -> (system, controller, final training budget)
SETTINGS = {
    "pen": ("Pendulum", "LQR", 600),
    "cp": ("Cartpole", "LQR", 1100),
    "cpppo": ("Cartpole", "Safe Explorer PPO", 3000),
    "q2d": ("Quadrotor 2D", "Safe Explorer PPO", 12000),
    "q2dc": ("Quadrotor 2D corridor", "Safe Explorer PPO", 15000),
    "q3d30": ("Quadrotor 3D", "LQR", 30000),
    "q3d60": ("Quadrotor 3D", "LQR", 60000),
    "q3d110": ("Quadrotor 3D", "LQR", 110000),
    "q3d200": ("Quadrotor 3D", "LQR", 200000),
    "q3drl40": ("Quadrotor 3D", "PPO", 40000),
    "q3drl100": ("Quadrotor 3D", "PPO", 100000),
    "q3drl200": ("Quadrotor 3D", "PPO", 199000),
}
MODEL = {"fm_det": "FM", "det_mlp_clf": "MLP classifier", "det_bnn_clf": "BNN classifier",
         "det_mlp_reg": "MLP final-state", "det_bnn_reg": "BNN final-state", "partx_faithful": "Part-X"}
REGIME = {"uniform": "uniform", "direct": "direct", "ranked": "ranked", "entropy": "entropy",
          "builtin": "Part-X built-in", "direct_fprcap1pct": "direct", "ranked_fprcap1pct": "ranked"}
METRIC_KEYS = ["kl", "f1", "abstain_pct", "invalid_pct", "uncertain_pct", "fpr", "fnr"]


def _num(v, digits=6):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return None
    return float(f"{float(v):.{digits}g}")


def cell_rows(r: dict) -> list[dict]:
    out_dir = re.search(r"output_dir=(\S+)", r["overrides"]).group(1)
    m = re.search(r"(?:^| )n_epochs=(\d+)", r["overrides"])
    n_ep = int(m.group(1)) if m else 11
    system, controller, budget = SETTINGS[r["system"]]
    rows, paths = [], sorted(glob.glob(f"{out_dir}/epoch_*/artifacts_v2.json"))
    finished = len(paths) >= n_ep
    for p in paths:
        p = Path(p)
        if r["arm"] in pp.ENDPOINT_ARMS and r["system"] in pp.NPZ_TREE_SETTINGS and pp.folded(p.parent):
            p = Path(out_dir) / "rescored_ternary" / p.parent.name / "artifacts_v2.json"
            if not p.exists():
                continue                  # scored two ways by mistake and not re-scored yet
        try:
            art = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            break                         # mid-write: stop at the last complete epoch
        row = {"setting": r["system"], "system": system, "controller": controller, "budget": budget,
               "model": MODEL[r["arm"]], "regime": REGIME[r["mode"]], "cell": r["name"],
               "epoch": int(p.parent.name[-3:]), "train": art.get("train_trajectories"),
               "n_epochs": n_ep, "status": "finished" if finished else "running"}
        row.update({k: _num(v) for k, v in zip(METRIC_KEYS, pp.metrics(art))})
        ts = art.get("threshold_state") or {}
        if ts.get("tau_s") is not None:
            row.update(tau_s=_num(ts["tau_s"]), tau_f=_num(ts["tau_f"]), tau_inv=_num(ts.get("tau_inv")))
        rows.append(row)
    return rows


# The FPR-capped quad3d families run only their 200k schedules; the 110k (LQR) and 100k
# (PPO) settings are those runs' epochs 0-10 (same schedule, user-approved 2026-11-02).
READ_OFF = {"q3d200": ("q3d110", 110000), "q3drl200": ("q3drl100", 100000)}


def read_off(rows: list[dict]) -> list[dict]:
    banked = {}
    for r in rows:
        banked[r["cell"]] = max(banked.get(r["cell"], 0), r["epoch"] + 1)
    return [dict(r, setting=READ_OFF[r["setting"]][0], budget=READ_OFF[r["setting"]][1], n_epochs=11,
                 status="finished" if banked[r["cell"]] >= 11 else "running")
            for r in rows if r["setting"] in READ_OFF and r["epoch"] <= 10]


def build(manifest: Path) -> tuple[list[dict], list[dict]]:
    nominal, fprcap = [], []
    for r in csv.DictReader(open(manifest), delimiter="\t"):
        if not r["name"].startswith("dr_") or r["arm"] not in MODEL or r["system"] not in SETTINGS \
                or r["seed"] != "99512" or r["mode"] not in REGIME:
            continue
        (fprcap if "fprcap" in r["mode"] else nominal).extend(cell_rows(r))
    return nominal, fprcap + read_off(fprcap)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=pp.HERE / "dashboard")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    nominal, fprcap = build(pp.HERE / "manifest.tsv")
    now = dt.datetime.now(dt.timezone.utc)
    snap = [{"generated_at": now.isoformat(timespec="seconds"),
             "generated_label": now.astimezone().strftime("%b %-d, %H:%M %Z"),
             "nominal_rows": len(nominal), "fprcap_rows": len(fprcap)}]
    for name, rows in (("nominal_epochs", nominal), ("fprcap_epochs", fprcap), ("snapshot", snap)):
        (a.out / f"{name}.json").write_text(json.dumps(rows, separators=(",", ":")))
    print(f"{a.out}: nominal {len(nominal)} rows, fprcap {len(fprcap)} rows, at {snap[0]['generated_label']}")


if __name__ == "__main__":
    main()
