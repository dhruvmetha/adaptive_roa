#!/usr/bin/env python
"""Write the det-paper rerun manifest (papers/det_paper/EXPERIMENT_TRACKER.md section 6).

One row per cell: name, system, arm, mode, schedule, seed, mem, priority, overrides.
Every cell has its own fixed output dir, so launch.sh can refuse a relaunch over
banked epochs. Usage:

    make_manifest.py --seed 99512 [--out slurm_logs/det_rerun/manifest.tsv]
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

EXP = os.environ.get("DET_RERUN_EXP", "/common/users/shared/pracsys/adaptive_roa_experiments/det_rerun")

# system key -> (hydra system, samples_per_epoch override or None, endpoint epochs, mem)
SYSTEMS = {
    "pen":    ("pendulum", None, [0, 5, 10], "24G"),
    "cp":     ("cartpole_pybullet", None, [0, 5, 10], "24G"),
    "q2d":    ("quadrotor2d", None, [0, 5, 10], "40G"),
    "q3d30":  ("quadrotor3d", 2000, [0, 5, 10], "64G"),
    "q3d60":  ("quadrotor3d", 5000, [0, 6, 10], "64G"),
    "q3d110": ("quadrotor3d", 10000, [0, 5, 10], "80G"),
}
# The shared uniform quad3d run (D24) also reports the epochs that land on the
# other schedules' budgets: 20k, 30k, 40k and 60k.
Q3D_UNIFORM_EPOCHS = [0, 1, 2, 3, 5, 10]

ARMS = ["fm_det", "det_mlp_clf", "det_bnn_clf", "det_mlp_reg", "det_bnn_reg"]
SHORT = {"fm_det": "fm", "det_mlp_clf": "mlpclf", "det_bnn_clf": "bnnclf",
         "det_mlp_reg": "mlpreg", "det_bnn_reg": "bnnreg"}
MODES = {"uniform": ("direct", ["acquisition.d2_ratio=0"]),
         "direct": ("direct", []), "ranked": ("ranked", []), "entropy": ("entropy", [])}
# Lower runs first: quad2d and quad3d separate the signals; longest schedules first.
PRIORITY = {"q3d110": 0, "q3d60": 1, "q3d30": 2, "q2d": 3, "cp": 4, "pen": 5}


def cells(seed: int):
    for sk, (system, step, epochs, mem) in SYSTEMS.items():
        for arm in ARMS:
            for mode, (acq, extra) in MODES.items():
                if mode == "uniform" and sk in ("q3d30", "q3d60"):
                    continue          # read from the q3d110 uniform run (D24)
                ep = Q3D_UNIFORM_EPOCHS if (mode == "uniform" and sk == "q3d110") else epochs
                name = f"dr_{sk}_{SHORT[arm]}_{mode}_s{seed}"
                ov = [f"system={system}", f"predictor={arm}", f"acquisition={acq}",
                      "+experiment=det_rerun", f"seed={seed}", *extra]
                if step is not None:
                    ov.append(f"samples_per_epoch={step}")
                # Single-quoted: tmux and remote ssh run the line through zsh, which
                # globs an unquoted [..] and aborts with "no matches found".
                ov.append("'prediction_store.endpoint_epochs=[" + ",".join(map(str, ep)) + "]'")
                ov.append(f"output_dir={EXP}/{sk}/{name}")
                yield {"name": name, "system": sk, "arm": arm, "mode": mode, "seed": seed,
                       "mem": mem, "priority": PRIORITY[sk], "overrides": " ".join(ov)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", default="/common/home/st1122/Projects/adaptive_roa/slurm_logs/det_rerun/manifest.tsv")
    a = ap.parse_args()
    rows = sorted(cells(a.seed), key=lambda r: (r["priority"], r["name"]))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    existing = set()
    if out.exists():
        with open(out) as f:
            existing = {r["name"] for r in csv.DictReader(f, delimiter="\t")}
    new = [r for r in rows if r["name"] not in existing]
    write_header = not out.exists()
    with open(out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        if write_header:
            w.writeheader()
        w.writerows(new)
    print(f"{len(new)} new rows ({len(rows)} cells for seed {a.seed}) -> {out}")


if __name__ == "__main__":
    main()
