#!/usr/bin/env python
"""Probability-quality metrics at the final common adaptive epoch.

Reads the wide all-levels CSVs (one row per (level, arm, epoch)) and, for a
fixed set of paper arms, reports KL / sAUROC / skill_score / REL_debiased / RES
at the last epoch every present arm reached. The three non-adaptive FM seeds
are pooled into one row (mean +- 2 sd across seeds at that epoch); a separate
row carries the pooled control floor 2*sqrt(mean_e var_seeds(e)) over the
epochs >= 1 shared by all three seeds.

Outputs a long CSV, a markdown file with one table per (system, level), and a
compact stdout summary of FM-BALD vs the control.
"""
from __future__ import annotations

import importlib.util
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

# Arm aliases and budget caps live in the sibling figure script (scripts/paper
# is not a package), so every reader canonicalises arm keys the same way.
_SIB = Path(__file__).resolve().parent / "plot_levelsets_paper.py"
_spec = importlib.util.spec_from_file_location("plot_levelsets_paper", _SIB)
_ls = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ls)
apply_aliases = _ls.apply_aliases
FINAL_EPOCHS = _ls.FINAL_EPOCHS

ROOT = Path("/common/users/shared/pracsys/genMoPlan/docs/stochastic")
OUT_DIR = ROOT / "paper_draft" / "tables"

SOURCES = [
    ("pendulum_lqr", ROOT / "pendulum/lqr/gaussian_all_levels.csv", ["low", "med", "high"]),
    ("cartpole_ppo", ROOT / "cartpole/safe_explorer_ppo/gaussian_all_levels.csv",
     ["baseline", "low", "med", "high"]),
    ("quad2d_rl", ROOT / "quadrotor2d/rl/quad2d_corridor_sine_ambient_all_levels.csv",
     ["corridor_sine_ambient_baseline", "corridor_sine_ambient_smooth", "corridor_sine_ambient_loud"]),
    # quad3d = 800k-pool campaign (decision 2026-09-10); 100k is obsolete
    ("quad3d_ppo800k", ROOT / "quadrotor3d/ppo_800k/quad3d800k_corridor_sine_ambient_all_levels.csv",
     ["corridor_sine_ambient_f_0.00", "corridor_sine_ambient_f_0.12", "corridor_sine_ambient_f_0.12_a0.03",
      "corridor_sine_ambient_f_0.20", "corridor_sine_ambient_f_0.20_a0.04",
      "corridor_sine_ambient_f_0.40", "corridor_sine_ambient_f_0.40_a0.04"]),
]

# Single-seed control (user decision 2026-09-11): no mean, 2sd or pooled floor.
CONTROL_SEEDS = ["dir00_s42"]
CONTROL_LABEL = "Non-adaptive (FM)"
CONTROL_KEY = "dir00_s42"
FLOOR_LABEL = "FM floor (pooled 2sd)"

# (arm_key, label) in display order. The control row goes first.
# BALD arms are the top-N (greedy) runs (decision 2026-09-10); greedy_diverse is out of the paper
ARMS = [
    ("epi_bald_greedy", "FM-BALD (ours)"),
    ("epi_var_greedy", "FM epi-var"),
    ("clf_epi_bald_greedy", "CLF-BALD"),
    ("bnn_mfvi_a1_greedy", "BNN-BALD"),
    ("partx_fix", "Part-X (GP)"),
    ("clf_dir00", "Non-adaptive (CLF)"),
    ("bnn_a1_dir00", "Non-adaptive (BNN)"),
]
EXPECTED_KEYS = CONTROL_SEEDS + [k for k, _ in ARMS]
# Runs excluded at a given level (kept in sync with plot_levelsets_paper.EXCLUDE_ARMS).
EXCLUDE_ARMS = {   # quad2d loud epi-var finished 2026-09-11; its entry is removed
                # q2d_csbase has a 1-epoch bnn_a1_dir00 stub that would drag the common epoch to 0
                ("quad2d_rl", "corridor_sine_ambient_baseline"): {"bnn_a1_dir00"}}

# Table sets, written as SEPARATE markdown files. The main evaluation is the
# method against the baselines; the ablation is the method against its own
# variants and the non-adaptive control (with the floor row). Bold marks the
# best within the set's rows.
TABLE_SETS = {
    "main":     ["FM-BALD (ours)", "Part-X (GP)", "CLF-BALD"],
    "ablation": ["FM-BALD (ours)", "FM epi-var", "BNN-BALD", CONTROL_LABEL],
}

METRICS = ["KL", "sAUROC", "skill_score", "REL_debiased", "RES"]
LOWER_BETTER = {"KL": True, "sAUROC": False, "skill_score": False, "REL_debiased": True, "RES": False}

DETERMINISTIC = {("cartpole_ppo", "baseline"), ("quad2d_rl", "corridor_sine_ambient_baseline"),
                 ("quad3d_ppo800k", "corridor_sine_ambient_f_0.00")}
DET_NOTE = ("Note: deterministic level (one rollout per eval cell), so KL is a clipped "
            "log-loss and is not on the same scale as the noisy levels.")


def chmod_group(path: Path) -> None:
    try:
        os.chmod(path, 0o660)
    except OSError:
        pass


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for c in METRICS + ["train_trajectories", "epoch"]:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["epoch"] = df["epoch"].astype(int)
    return df


def control_floor(ctrl: pd.DataFrame) -> dict[str, float]:
    """2*sqrt(mean over shared epochs >= 1 of the sample variance across seeds)."""
    shared = None
    for s in CONTROL_SEEDS:
        eps = set(ctrl.loc[ctrl["arm"] == s, "epoch"])
        shared = eps if shared is None else shared & eps
    shared = sorted(e for e in (shared or set()) if e >= 1)
    out = {}
    for m in METRICS:
        variances = []
        for e in shared:
            vals = ctrl.loc[ctrl["epoch"] == e, m].dropna()
            if len(vals) == 3:
                variances.append(vals.var(ddof=1))
        out[m] = 2.0 * math.sqrt(float(np.mean(variances))) if variances else np.nan
    return out


def build_level(system: str, level: str, df: pd.DataFrame, missing: list[str],
                force_epoch: int | None = None):
    """`force_epoch` reads every arm at that epoch instead of the final common
    one (a fixed-budget snapshot, e.g. epoch 3 = 14,500 trajectories on quad3d)."""
    keys = [k for k in EXPECTED_KEYS if k not in EXCLUDE_ARMS.get((system, level), set())]
    sub = df[(df["level"] == level) & (df["arm"].isin(keys))].copy()
    present = sorted(set(sub["arm"]))
    for k in EXPECTED_KEYS:
        if k not in present:
            missing.append(f"{system}/{level}: {k}")
    if not present:
        return [], None
    final_epoch = int(sub.groupby("arm")["epoch"].max().min()) if force_epoch is None else int(force_epoch)
    at = sub[sub["epoch"] == final_epoch]

    rows = []
    ctrl = at[at["arm"].isin(CONTROL_SEEDS)]
    ctrl_all = sub[sub["arm"].isin(CONTROL_SEEDS)]
    if len(ctrl):
        row = {"system": system, "level": level, "arm_label": CONTROL_LABEL, "arm_key": CONTROL_KEY,
               "epoch": final_epoch, "train_trajectories": ctrl["train_trajectories"].mean(),
               "n_seeds": int(len(ctrl))}
        for m in METRICS:
            vals = ctrl[m].dropna()
            row[m] = vals.mean() if len(vals) else np.nan
            row[f"{m}_2sd"] = 2.0 * vals.std(ddof=1) if len(vals) >= 2 else np.nan
        rows.append(row)
        floor = control_floor(ctrl_all) if len(set(ctrl_all["arm"])) == 3 else {m: np.nan for m in METRICS}
        if len(CONTROL_SEEDS) < 3:
            floor = None   # single-seed control: no floor row at all
        frow = {"system": system, "level": level, "arm_label": FLOOR_LABEL, "arm_key": "floor",
                "epoch": final_epoch, "train_trajectories": np.nan, "n_seeds": np.nan}
        if floor is not None:
            frow.update({m: floor[m] for m in METRICS})
            rows.append(frow)
    for key, label in ARMS:
        r = at[at["arm"] == key]
        if not len(r):
            continue
        r = r.iloc[0]
        row = {"system": system, "level": level, "arm_label": label, "arm_key": key,
               "epoch": final_epoch, "train_trajectories": r["train_trajectories"], "n_seeds": 1}
        for m in METRICS:
            row[m] = r[m]
            row[f"{m}_2sd"] = np.nan
        rows.append(row)
    return rows, final_epoch


def fmt(v: float) -> str:
    return "" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.4f}"


def md_table(system: str, level: str, rows: list[dict], table_set: str) -> str:
    labels = TABLE_SETS[table_set]
    by = {r["arm_label"]: r for r in rows}
    body = [by[l] for l in labels if l in by]
    floor = by.get(FLOOR_LABEL) if CONTROL_LABEL in labels else None
    best = {}
    for m in METRICS:
        cands = [(r[m], r["arm_label"]) for r in body if not math.isnan(r[m])]
        if cands:
            pick = min(cands) if LOWER_BETTER[m] else max(cands)
            best[m] = pick[1]
    epoch = rows[0]["epoch"]
    lines = [f"### {system} / {level} (epoch {epoch}) — {table_set}", "",
             "| Arm | n_traj | " + " | ".join(METRICS) + " |",
             "|---|---:|" + "---:|" * len(METRICS)]
    for r in body:
        cells = []
        for m in METRICS:
            s = fmt(r[m])
            if r["arm_label"] == CONTROL_LABEL and not math.isnan(r.get(f"{m}_2sd", np.nan)):
                s = f"{s} ± {fmt(r[f'{m}_2sd'])}"
            if best.get(m) == r["arm_label"] and s:
                s = f"**{s}**"
            cells.append(s)
        nt = r["train_trajectories"]
        nt_s = "" if (nt is None or (isinstance(nt, float) and math.isnan(nt))) else f"{int(round(nt))}"
        lines.append(f"| {r['arm_label']} | {nt_s} | " + " | ".join(cells) + " |")
    if floor is not None:
        lines.append(f"| {FLOOR_LABEL} | | " + " | ".join(fmt(floor[m]) for m in METRICS) + " |")
    if (system, level) in DETERMINISTIC:
        lines += ["", DET_NOTE]
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epoch", type=int, default=None,
                    help="read every arm at this epoch instead of the final common epoch")
    ap.add_argument("--systems", nargs="*", default=None, help="subset of systems")
    ap.add_argument("--suffix", default="", help="output filename suffix, e.g. _15k")
    ap.add_argument("--paper", action="store_true",
                    help="read each system at the paper budget (FINAL_EPOCHS in "
                         "plot_levelsets_paper.py); writes the _paper tables")
    args = ap.parse_args()
    if args.epoch is not None and not args.suffix:
        ap.error("--epoch needs --suffix so the final-epoch tables are not overwritten")
    if args.paper:
        if args.epoch is not None:
            ap.error("--paper and --epoch are exclusive")
        args.suffix = args.suffix or "_paper"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    md_parts = {
        "main": ["# Main evaluation: probability-quality metrics at the final common adaptive epoch", "",
                 "FM-BALD (ours) against the baselines Part-X and CLF-BALD. Bold marks the best of "
                 "the three per column (KL, REL lower is better; sAUROC, skill_score, RES higher).", ""],
        "ablation": ["# Ablation: probability-quality metrics at the final common adaptive epoch", "",
                     "FM-BALD (ours) against its variants (FM epi-var, BNN-BALD) and the non-adaptive FM "
                     "control (uniform sampling, seed 42, same ensemble). Every arm is a single seed. "
                     "Bold marks the best of the four per column.", ""],
    }
    summary = []
    missing: list[str] = []
    for system, path, levels in SOURCES:
        if args.systems and system not in args.systems:
            continue
        df = apply_aliases(load(path), system)
        for level in levels:
            force = args.epoch
            if args.paper:
                spec = FINAL_EPOCHS.get(system, "common")
                force = None if spec == "common" else int(spec)
            rows, ep = build_level(system, level, df, missing, force)
            if not rows:
                summary.append(f"{system:12s} {level:32s} no rows")
                continue
            all_rows.extend(rows)
            for table_set in TABLE_SETS:
                md_parts[table_set].append(md_table(system, level, rows, table_set))
            by = {r["arm_label"]: r for r in rows}
            ctrl = by.get(CONTROL_LABEL)
            floor = by.get(FLOOR_LABEL)
            ours = by.get("FM-BALD (ours)")
            if ctrl is None or ours is None or math.isnan(ours["KL"]) or math.isnan(ctrl["KL"]):
                summary.append(f"{system:12s} {level:32s} ep={ep:2d} FM-BALD or control missing")
                continue
            diff = ours["KL"] - ctrl["KL"]
            fl = floor["KL"] if floor is not None else np.nan
            if not math.isnan(fl) and abs(diff) <= fl:
                verdict = "inside floor"
            else:
                verdict = "win" if diff < 0 else "loss"
            summary.append(
                f"{system:12s} {level:32s} ep={ep:2d} KL bald={ours['KL']:.4f} "
                f"ctrl={ctrl['KL']:.4f} diff={diff:+.4f} floor={fl:.4f} {verdict}")

    cols = ["system", "level", "arm_label", "arm_key", "epoch", "train_trajectories", "n_seeds"]
    cols += METRICS + [f"{m}_2sd" for m in METRICS]
    out = pd.DataFrame(all_rows)[cols]
    csv_path = OUT_DIR / f"final_epoch_prob_metrics{args.suffix}.csv"
    out.to_csv(csv_path, index=False, float_format="%.6g")
    chmod_group(csv_path)
    md_paths = []
    for table_set, parts in md_parts.items():
        if args.epoch is not None:
            parts[0] = parts[0].replace("final common adaptive epoch",
                                        f"epoch {args.epoch} (fixed-budget snapshot)")
        elif args.paper:
            parts[0] = parts[0].replace("final common adaptive epoch",
                                        "paper budget (cartpole 1,950, quad2d 8,000, quad3d 14,500 "
                                        "trajectories; pendulum final)")
        md_path = OUT_DIR / f"final_epoch_prob_metrics_{table_set}{args.suffix}.md"
        md_path.write_text("\n".join(parts))
        chmod_group(md_path)
        md_paths.append(md_path)
    stale = OUT_DIR / "final_epoch_prob_metrics.md"   # pre-split combined file
    if stale.exists():
        stale.unlink()

    print("\n".join(summary))
    if missing:
        print("\nmissing arms:")
        print("\n".join(f"  {m}" for m in missing))
    print(f"\nwrote {csv_path}")
    for p in md_paths:
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
