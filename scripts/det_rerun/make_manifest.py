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
    # mem from measured MaxRSS (2026-10-30): quad2d FM peaked at 4.1G over a full
    # run, quad3d FM at 9.1G halfway through the 110k schedule. About 2.5-3x headroom.
    "pen":    ("pendulum", None, [0, 5, 10], "8G"),
    "cp":     ("cartpole_pybullet", None, [0, 5, 10], "8G"),
    "q2d":    ("quadrotor2d", None, [0, 5, 10], "12G"),
    "q3d30":  ("quadrotor3d", 2000, [0, 5, 10], "16G"),
    "q3d60":  ("quadrotor3d", 5000, [0, 6, 10], "20G"),
    "q3d110": ("quadrotor3d", 10000, [0, 5, 10], "24G"),
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
                       "mem": mem, "priority": PRIORITY[sk], "overrides": " ".join(ov),
                       "script": "scripts/run_adaptive.py"}
        # Part-X (P01, D09/D22c/C21/C23): faithful runner on the suffix state pool, CPU only.
        name = f"dr_{sk}_partx_builtin_s{seed}"
        ov = [f"system={system}", "+experiment=det_rerun", f"seed={seed}", "partx_faithful.state_pool=true"]
        if step is not None:
            ov.append(f"samples_per_epoch={step}")
        ov.append(f"output_dir={EXP}/{sk}/{name}")
        yield {"name": name, "system": sk, "arm": "partx_faithful", "mode": "builtin", "seed": seed,
               "mem": mem, "priority": PRIORITY[sk], "overrides": " ".join(ov),
               "script": "scripts/run_partx_faithful.py"}


ABLATION_SYSTEMS = ("cp", "q2d", "q3d30")   # D23: quad3d ablations on the 30k schedule


def ablation_cells(seed: int, signal: str):
    """Tracker section 8 on FM with the D08 signal; the main matrix supplies the starred values."""
    acq = MODES[signal][0]
    variants = [("d2_0.75", ["acquisition.d2_ratio=0.75"]), ("d2_1.0", ["acquisition.d2_ratio=1.0"]),
                ("K50", ["acquisition_mc_samples=50"]), ("K100", ["acquisition_mc_samples=100"]),
                ("startpool", ["candidate_mode=start"])]
    if signal in ("direct", "ranked"):   # entropy never reads lambda*/delta*, so A-W2 is moot
        variants += [(f"w{w}", [f"threshold.w={w}"]) for w in (0.5, 0.75, 0.95, 0.99)]
    for sk in ABLATION_SYSTEMS:
        system, step, epochs, mem = SYSTEMS[sk]
        for tag, extra in variants:
            name = f"dr_{sk}_fm_{signal}_{tag}_s{seed}"
            ov = [f"system={system}", "predictor=fm_det", f"acquisition={acq}", "+experiment=det_rerun",
                  f"seed={seed}", *extra]
            if step is not None:
                ov.append(f"samples_per_epoch={step}")
            ov.append("'prediction_store.endpoint_epochs=[10]'")    # final budget only for ablations
            ov.append(f"output_dir={EXP}/{sk}/{name}")
            yield {"name": name, "system": sk, "arm": "fm_det", "mode": f"{signal}_{tag}", "seed": seed,
                   "mem": mem, "priority": 10 + PRIORITY[sk], "overrides": " ".join(ov),
                   "script": "scripts/run_adaptive.py"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--ablations", choices=["direct", "ranked", "entropy"], default=None,
                    help="append the ablation cells for the D08 signal instead of the main matrix")
    ap.add_argument("--out", default="/common/home/st1122/Projects/adaptive_roa/slurm_logs/det_rerun/manifest.tsv")
    a = ap.parse_args()
    source = ablation_cells(a.seed, a.ablations) if a.ablations else cells(a.seed)
    rows = sorted(source, key=lambda r: (r["priority"], r["name"]))
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
