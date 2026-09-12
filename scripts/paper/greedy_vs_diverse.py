#!/usr/bin/env python
"""Greedy-vs-diverse batch-selection ablation: table + figure for the paper.

Each adaptive arm has a `_greedy` twin that takes the top-N pool points by
acquisition score instead of the paper's `greedy_diverse` rule (top 5N, then a
farthest-point subsample of N). For every (system, level, pair) this script
finds the deepest epoch both twins reached, takes greedy minus diverse on the
probability metrics (KL, sAUROC, skill score) and the level-set areas (F0.5,
TPR, precision), and compares the delta against a run-to-run floor: twice the
sample sd across the three non-adaptive FM seeds at that same epoch.

Every pair is a single seed, so the floor is the only noise estimate we have.

Inputs
  (a) wide probability-metric CSVs, one row per (level, arm, epoch)
  (b) level-set area CSVs from `levelsets_greedy/`, same keying

Outputs (under PAPER_DIR)
  tables/greedy_vs_diverse.csv   long table, one row per cell
  tables/greedy_vs_diverse.md    summary counts + one table per system
  figures/greedy_vs_diverse.{png,pdf}

Usage:
    python scripts/paper/greedy_vs_diverse.py
"""
from __future__ import annotations
import importlib.util
import math
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# =============================================================================
# STYLE  -- every design choice is here. Nothing below this block is a choice.
# =============================================================================

STOCH_DIR = Path("/common/users/shared/pracsys/genMoPlan/docs/stochastic")
PAPER_DIR = STOCH_DIR / "paper_draft"
AREAS_DIR = PAPER_DIR / "levelsets_greedy"
TABLE_DIR = PAPER_DIR / "tables"
FIG_DIR = PAPER_DIR / "figures"
OUT_STEM = "greedy_vs_diverse"
FILE_MODE = 0o660
FORMATS = ("png",)   # png only (2026-09-11)
DPI = 200

# Colours and markers come from the level-set figure script so the arms look
# the same across the paper.
STYLE_SRC = Path(__file__).resolve().parent / "plot_levelsets_paper.py"

# System order, wide input, and which levels to keep (None -> every level, in
# LEVEL_ORDER then file order).
SYSTEMS = {
    "pendulum_lqr": dict(
        wide=STOCH_DIR / "pendulum/lqr/gaussian_all_levels.csv",
        levels=["low", "med", "high"], short="pend"),
    "cartpole_ppo": dict(
        wide=STOCH_DIR / "cartpole/safe_explorer_ppo/gaussian_all_levels.csv",
        levels=["baseline", "low", "med", "high"], short="cp"),
    "quad2d_rl": dict(
        wide=STOCH_DIR / "quadrotor2d/rl/quad2d_corridor_sine_ambient_all_levels.csv",
        levels=["corridor_sine_ambient_smooth"], short="q2d"),
    "quad3d_ppo800k": dict(
        # quad3d is read from the 800k-pool campaign (user decision 2026-09-10);
        # the 100k file is quadrotor3d/ppo/quad3dppo_corridor_sine_ambient_all_levels.csv
        wide=STOCH_DIR / "quadrotor3d/ppo_800k/quad3d800k_corridor_sine_ambient_all_levels.csv",
        levels=["corridor_sine_ambient_f_0.00", "corridor_sine_ambient_f_0.12",
                "corridor_sine_ambient_f_0.20", "corridor_sine_ambient_f_0.40"],
        short="q3d"),
}
SYSTEM_TITLES = {
    "pendulum_lqr": "Pendulum (LQR)", "cartpole_ppo": "Cartpole (RL)",
    "quad2d_rl": "Quadrotor 2D (RL)", "quad3d_ppo800k": "Quadrotor 3D (RL)",
}
SYSTEM_STRIP = {   # short names for the strip above the panels
    "pendulum_lqr": "Pendulum", "cartpole_ppo": "Cartpole",
    "quad2d_rl": "Quad 2D", "quad3d_ppo800k": "Quad 3D",
}
LEVEL_TITLES = {
    "baseline": "no noise", "low": "low", "med": "med", "high": "high",
    "corridor_sine_ambient_smooth": "smooth",
}


def level_title(level: str) -> str:
    if level in LEVEL_TITLES:
        return LEVEL_TITLES[level]
    if "_f_" in level:
        return f"f = {level.split('_f_')[-1]}"
    return level


def level_short(level: str) -> str:
    if level == "baseline":
        return "base"
    if "_f_" in level:
        return "f" + level.split("_f_")[-1]
    return level_title(level)


# Pairs: (diverse arm, greedy twin, label). Order is legend order.
PAIRS = [
    ("epi_bald",     "epi_bald_greedy",     "FM-BALD"),
    ("epi_var",      "epi_var_greedy",      "FM epi-var"),
    ("clf_epi_bald", "clf_epi_bald_greedy", "CLF-BALD"),
    ("bnn_mfvi_a1",  "bnn_mfvi_a1_greedy",  "BNN-BALD"),
]
HEADLINE_PAIR = "FM-BALD"

# Non-adaptive FM seeds that give the run-to-run floor.
SEED_ARMS = ["dir00_s42", "dir00_s43", "dir00_s44"]
FLOOR_MULT = 2.0          # floor = FLOOR_MULT * sample sd over the seeds
MIN_SEEDS = 3             # fewer seeds present -> floor is NaN

# Metrics: column -> (source, higher_is_better, table header, figure label).
METRICS = {
    "KL":          ("wide",  False, "ΔKL",     "ΔKL"),
    "sAUROC":      ("wide",  True,  "ΔsAUROC", "ΔsAUROC"),
    "skill_score": ("wide",  True,  "Δskill",  "Δskill score"),
    "auc_f05":     ("areas", True,  "ΔF0.5",   "ΔF0.5 area"),
    "auc_tpr":     ("areas", True,  "ΔTPR",    "ΔTPR area"),
    "auc_prec":    ("areas", True,  "Δprec",   "Δprecision area"),
}
FIG_METRICS = ["KL", "sAUROC", "auc_f05"]
# y scale per figure metric. symlog keeps the floor bars readable when one
# cell's delta is an order of magnitude beyond the rest; linthresh is the
# half-width of the linear region (data units), linscale its size in decades.
Y_SCALE = {
    "KL":      dict(scale="symlog", linthresh=0.01, linscale=1.5),
    "sAUROC":  dict(scale="linear"),
    "auc_f05": dict(scale="symlog", linthresh=0.05, linscale=1.5),
}
TABLE_DECIMALS = 4
UP, DOWN = "▲", "▼"       # greedy better / diverse better
UNDECIDED = "undecided"   # verdict when a value or the floor is missing
                          # (not "n/a": pandas reads that back as NaN)

# Figure geometry and typography.
PANEL_W, PANEL_H = 3.3, 2.3
LEFT_MARGIN, RIGHT_MARGIN = 0.80, 0.12
TOP_BLOCK_H = 0.30        # system-name strip above the panels
XLABEL_BLOCK_H = 0.62     # rotated cell labels under the panels
LEGEND_BLOCK_H = 0.36
SUBPLOT_WSPACE = 0.30
FONT_FAMILY = "sans-serif"
FONT_SIZE = 8
TITLE_SIZE = 9
LABEL_SIZE = 8
TICK_SIZE = 7
LEGEND_SIZE = 7.5
X_TICK_ROTATION = 60
MARKER_SIZE = 4.6
MARKER_EDGE = 0.5
PAIR_OFFSETS = [-0.27, -0.09, 0.09, 0.27]   # x jitter per pair inside a cell
FLOOR_BAR_HALF_W = 0.42
FLOOR_BAR_COLOR = "0.72"
FLOOR_BAR_ALPHA = 0.55
FLOOR_LABEL = "± FM run-to-run floor (2 sd, 3 seeds)"
ZERO_LINE = dict(color="black", lw=0.7)
SYSTEM_SEP = dict(color="0.5", lw=0.6, ls=":")
GRID = dict(color="0.88", lw=0.5)
SPINES_OFF = ("top", "right")
Y_PAD_FRAC = 0.12         # linear axes: padding beyond the data/floor extent
SYMLOG_PAD_FACTOR = 2.0   # symlog axes: ylim = extent * factor (log region)
SYSTEM_STRIP_Y = 1.04     # axes-fraction height of the system names
SYSTEM_STRIP_SIZE = 7.5

# =============================================================================
# Plumbing
# =============================================================================


def _apply_rc() -> None:
    plt.rcParams.update({
        "font.family": FONT_FAMILY, "font.size": FONT_SIZE,
        "axes.titlesize": TITLE_SIZE, "axes.labelsize": LABEL_SIZE,
        "xtick.labelsize": TICK_SIZE, "ytick.labelsize": TICK_SIZE,
        "legend.fontsize": LEGEND_SIZE, "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def _load_arm_styles() -> dict:
    spec = importlib.util.spec_from_file_location("plot_levelsets_paper", STYLE_SRC)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.ARM_STYLES


def _read(path: Path) -> pd.DataFrame:
    """Read a metrics CSV; tolerate a file another job is appending to."""
    if not path.exists():
        print(f"  missing input: {path}")
        return pd.DataFrame(columns=["level", "arm", "epoch"])
    df = pd.read_csv(path, on_bad_lines="skip")
    df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
    df = df.dropna(subset=["epoch"]).copy()
    df["epoch"] = df["epoch"].astype(int)
    return df


def _value(df: pd.DataFrame, level: str, arm: str, epoch: int, col: str) -> float:
    if col not in df.columns:
        return float("nan")
    sub = df[(df["level"] == level) & (df["arm"] == arm) & (df["epoch"] == epoch)]
    if sub.empty:
        return float("nan")
    v = pd.to_numeric(sub[col], errors="coerce").iloc[-1]
    return float(v) if pd.notna(v) else float("nan")


def _floor(df: pd.DataFrame, level: str, epoch: int, col: str) -> float:
    vals = [_value(df, level, a, epoch, col) for a in SEED_ARMS]
    vals = [v for v in vals if not math.isnan(v)]
    if len(vals) < MIN_SEEDS:
        return float("nan")
    return FLOOR_MULT * float(np.std(vals, ddof=1))


def _verdict(delta: float, floor: float, higher_better: bool) -> str:
    if math.isnan(delta) or math.isnan(floor):
        return UNDECIDED
    favours_greedy = delta if higher_better else -delta
    if favours_greedy > floor:
        return "greedy"
    if favours_greedy < -floor:
        return "diverse"
    return "tie"


def _chmod(path: Path) -> None:
    try:
        os.chmod(path, FILE_MODE)
    except OSError:
        pass


# -----------------------------------------------------------------------------
# Compute
# -----------------------------------------------------------------------------

def compute() -> tuple[pd.DataFrame, list[str]]:
    rows, skipped = [], []
    for system, cfg in SYSTEMS.items():
        wide = _read(cfg["wide"])
        areas = _read(AREAS_DIR / f"{system}_levelset_areas_b50.csv")
        src = {"wide": wide, "areas": areas}
        for level in cfg["levels"]:
            for arm_d, arm_g, label in PAIRS:
                ep_d = set(wide[(wide["level"] == level) & (wide["arm"] == arm_d)]["epoch"])
                ep_g = set(wide[(wide["level"] == level) & (wide["arm"] == arm_g)]["epoch"])
                common = ep_d & ep_g
                if not common:
                    skipped.append(f"{system}/{level}/{label}: no common epoch "
                                   f"(diverse {sorted(ep_d)[-1] if ep_d else None}, "
                                   f"greedy {sorted(ep_g)[-1] if ep_g else None})")
                    continue
                ep = max(common)
                ntraj = _value(wide, level, arm_d, ep, "train_trajectories")
                if math.isnan(ntraj):
                    ntraj = _value(wide, level, arm_g, ep, "train_trajectories")
                row = dict(system=system, level=level, pair_label=label,
                           arm_diverse=arm_d, arm_greedy=arm_g, epoch=ep,
                           max_epoch_diverse=max(ep_d), max_epoch_greedy=max(ep_g),
                           train_trajectories=int(ntraj) if not math.isnan(ntraj) else np.nan)
                for m, (source, hib, _, _) in METRICS.items():
                    df = src[source]
                    vd = _value(df, level, arm_d, ep, m)
                    vg = _value(df, level, arm_g, ep, m)
                    delta = vg - vd
                    floor = _floor(df, level, ep, m)
                    row[f"{m}_diverse"] = vd
                    row[f"{m}_greedy"] = vg
                    row[f"{m}_delta"] = delta
                    row[f"{m}_2sd"] = floor
                    row[f"{m}_verdict"] = _verdict(delta, floor, hib)
                rows.append(row)
    return pd.DataFrame(rows), skipped


# -----------------------------------------------------------------------------
# Markdown
# -----------------------------------------------------------------------------

def _counts(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for m, (_, _, hdr, _) in METRICS.items():
        vc = df[f"{m}_verdict"].value_counts()
        out.append(dict(metric=hdr, diverse=int(vc.get("diverse", 0)),
                        greedy=int(vc.get("greedy", 0)), tie=int(vc.get("tie", 0)),
                        na=int(vc.get(UNDECIDED, 0))))
    return pd.DataFrame(out)


def _counts_table(c: pd.DataFrame, n_cells: int) -> str:
    show_na = int(c["na"].sum()) > 0
    hdr = "| metric | diverse better | greedy better | tie |" + (" undecided |" if show_na else "")
    sep = "|---|---:|---:|---:|" + ("---:|" if show_na else "")
    lines = [hdr, sep]
    for _, r in c.iterrows():
        line = f"| {r['metric']} | {r['diverse']} | {r['greedy']} | {r['tie']} |"
        if show_na:
            line += f" {r['na']} |"
        lines.append(line)
    return f"{n_cells} cells (system × level × pair).\n\n" + "\n".join(lines)


def _fmt_cell(delta: float, floor: float, verdict: str) -> str:
    if math.isnan(delta):
        return "n/a"
    d = f"{delta:+.{TABLE_DECIMALS}f}"
    f = "n/a" if math.isnan(floor) else f"{floor:.{TABLE_DECIMALS}f}"
    mark = {"greedy": f" {UP}", "diverse": f" {DOWN}"}.get(verdict, "")
    return f"{d} ({f}){mark}"


def write_md(df: pd.DataFrame, skipped: list[str], path: Path) -> None:
    lines = ["# Greedy vs greedy_diverse batch selection", ""]
    lines.append(
        "Each adaptive arm against its `_greedy` twin (plain top-N by acquisition score "
        "instead of top 5N followed by a farthest-point subsample of N). Every cell is "
        "greedy minus diverse at the deepest epoch both twins reached in the probability "
        "metrics file. The floor in parentheses is 2 × the sample sd across the three "
        "non-adaptive FM seeds (dir00_s42/s43/s44) at that same epoch. A delta beyond the "
        f"floor is marked {UP} when it favours greedy and {DOWN} when it favours diverse; "
        "no mark means the two are within the floor. KL: lower is better. All other "
        "metrics: higher is better. F0.5, TPR and precision are level-set areas over "
        "β = 0.50..0.95 from `levelsets_greedy/`.")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("### All pairs")
    lines.append("")
    lines.append(_counts_table(_counts(df), len(df)))
    lines.append("")
    head = df[df["pair_label"] == HEADLINE_PAIR]
    lines.append(f"### {HEADLINE_PAIR} pair only")
    lines.append("")
    lines.append(_counts_table(_counts(head), len(head)))
    lines.append("")

    cols = [(m, METRICS[m][2]) for m in METRICS]
    for system in SYSTEMS:
        sub = df[df["system"] == system]
        if sub.empty:
            continue
        lines.append(f"## {SYSTEM_TITLES.get(system, system)} (`{system}`)")
        lines.append("")
        lines.append("| level | pair | epoch | n_traj | " + " | ".join(h for _, h in cols) + " |")
        lines.append("|---|---|---:|---:|" + "|".join("---:" for _ in cols) + "|")
        for _, r in sub.iterrows():
            ep = int(r["epoch"])
            ep_txt = str(ep)
            if r["max_epoch_diverse"] != ep or r["max_epoch_greedy"] != ep:
                ep_txt += f" (max {int(r['max_epoch_diverse'])}/{int(r['max_epoch_greedy'])})"
            nt = "" if pd.isna(r["train_trajectories"]) else f"{int(r['train_trajectories']):,}"
            cells = [_fmt_cell(r[f"{m}_delta"], r[f"{m}_2sd"], r[f"{m}_verdict"]) for m, _ in cols]
            lines.append(f"| {level_title(r['level'])} | {r['pair_label']} | {ep_txt} | {nt} | "
                         + " | ".join(cells) + " |")
        lines.append("")
        shallow = sub[(sub["max_epoch_diverse"] != sub["epoch"]) | (sub["max_epoch_greedy"] != sub["epoch"])]
        if not shallow.empty:
            parts = []
            for _, r in shallow.iterrows():
                parts.append(f"{level_title(r['level'])} {r['pair_label']}: compared at epoch "
                             f"{int(r['epoch'])}, diverse reached {int(r['max_epoch_diverse'])}, "
                             f"greedy reached {int(r['max_epoch_greedy'])}")
            lines.append("Epoch column shows `epoch (max diverse/max greedy)` where the twins "
                         "are not equally deep; the shallower run sets the comparison epoch. "
                         + "; ".join(parts) + ".")
            lines.append("")

    lines.append("## Caveats")
    lines.append("")
    lines.append("- Every pair is a single seed per arm. The floor is the only noise estimate "
                 "and it comes from the non-adaptive FM control, so for the CLF and BNN pairs "
                 "it is a proxy, not their own run-to-run spread.")
    q3 = df[(df["system"] == "quad3d_ppo800k") & (df["max_epoch_greedy"] < df["max_epoch_diverse"])]
    if not q3.empty:
        parts = [f"{level_title(r['level'])} {r['pair_label']} {int(r['max_epoch_greedy'])} vs "
                 f"{int(r['max_epoch_diverse'])}" for _, r in q3.iterrows()]
        lines.append("- The quad3d greedy twins stopped earlier than their partners, so those "
                     "cells are compared at a shallower epoch than the rest of the paper "
                     "(greedy vs diverse max epoch): " + "; ".join(parts) + ".")
    na = df[[f"{m}_verdict" for m in METRICS]].eq(UNDECIDED).any(axis=1)
    if na.any():
        parts = [f"{r['system']}/{level_title(r['level'])} {r['pair_label']} (epoch {int(r['epoch'])})"
                 for _, r in df[na].iterrows()]
        lines.append("- Cells marked n/a (verdict `undecided` in the CSV) have no level-set-area "
                     "row for one twin at the comparison epoch: " + "; ".join(parts) + ".")
    for s in skipped:
        lines.append(f"- Skipped: {s}")
    lines.append("")
    path.write_text("\n".join(lines))
    _chmod(path)


# -----------------------------------------------------------------------------
# Figure
# -----------------------------------------------------------------------------

def _cells(df: pd.DataFrame) -> list[tuple[str, str]]:
    seen, cells = set(), []
    for system in SYSTEMS:
        for level in SYSTEMS[system]["levels"]:
            if (system, level) in seen:
                continue
            if not df[(df["system"] == system) & (df["level"] == level)].empty:
                cells.append((system, level))
                seen.add((system, level))
    return cells


def plot(df: pd.DataFrame, styles: dict, out_stem: Path) -> None:
    _apply_rc()
    cells = _cells(df)
    n = len(cells)
    x_of = {c: i for i, c in enumerate(cells)}
    labels = [f"{SYSTEMS[s]['short']} {level_short(l)}" for s, l in cells]

    n_pan = len(FIG_METRICS)
    fig_w = LEFT_MARGIN + n_pan * PANEL_W + (n_pan - 1) * SUBPLOT_WSPACE * PANEL_W + RIGHT_MARGIN
    fig_h = TOP_BLOCK_H + PANEL_H + XLABEL_BLOCK_H + LEGEND_BLOCK_H
    fig, axes = plt.subplots(1, n_pan, figsize=(fig_w, fig_h), squeeze=False)
    axes = axes[0]
    fig.subplots_adjust(left=LEFT_MARGIN / fig_w, right=1 - RIGHT_MARGIN / fig_w,
                        top=1 - TOP_BLOCK_H / fig_h,
                        bottom=(XLABEL_BLOCK_H + LEGEND_BLOCK_H) / fig_h,
                        wspace=SUBPLOT_WSPACE)

    # System group boundaries (x positions between consecutive systems).
    bounds, spans = [], []
    start = 0
    for i in range(1, n + 1):
        if i == n or cells[i][0] != cells[start][0]:
            spans.append((cells[start][0], start, i - 1))
            if i < n:
                bounds.append(i - 0.5)
            start = i

    for ax, m in zip(axes, FIG_METRICS):
        _, hib, _, fig_label = METRICS[m]
        ymax = 0.0
        for (system, level), xi in x_of.items():
            sub = df[(df["system"] == system) & (df["level"] == level)]
            if sub.empty:
                continue
            floor = float(sub[f"{m}_2sd"].iloc[0])
            if not math.isnan(floor):
                ax.add_patch(Rectangle((xi - FLOOR_BAR_HALF_W, -floor), 2 * FLOOR_BAR_HALF_W,
                                       2 * floor, color=FLOOR_BAR_COLOR, alpha=FLOOR_BAR_ALPHA,
                                       lw=0, zorder=1))
                ymax = max(ymax, floor)
            for k, (arm_d, _, label) in enumerate(PAIRS):
                r = sub[sub["pair_label"] == label]
                if r.empty:
                    continue
                d = float(r[f"{m}_delta"].iloc[0])
                if math.isnan(d):
                    continue
                _, _, color, _, marker = styles[arm_d]
                ax.plot([xi + PAIR_OFFSETS[k]], [d], marker=marker, ms=MARKER_SIZE, color=color,
                        mec="black", mew=MARKER_EDGE, ls="none", zorder=3)
                ymax = max(ymax, abs(d))
        ax.axhline(0, **ZERO_LINE, zorder=2)
        for b in bounds:
            ax.axvline(b, **SYSTEM_SEP, zorder=0)
        for system, lo, hi in spans:
            ax.text((lo + hi) / 2, SYSTEM_STRIP_Y, SYSTEM_STRIP.get(system, system),
                    transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                    fontsize=SYSTEM_STRIP_SIZE, color="0.25")
        ysc = Y_SCALE.get(m, dict(scale="linear"))
        scale_tag = ""
        if ysc["scale"] == "symlog":
            ax.set_yscale("symlog", linthresh=ysc["linthresh"], linscale=ysc["linscale"])
            scale_tag = f", symlog beyond ±{ysc['linthresh']:g}"
        if ymax <= 0:
            lim = 1.0
        elif ysc["scale"] == "symlog" and ymax > ysc["linthresh"]:
            lim = ymax * SYMLOG_PAD_FACTOR
        else:
            lim = ymax * (1 + Y_PAD_FRAC)
        ax.set_ylim(-lim, lim)
        ax.set_xlim(-0.6, n - 0.4)
        ax.set_xticks(range(n))
        ax.set_xticklabels(labels, rotation=X_TICK_ROTATION, ha="right", rotation_mode="anchor")
        direction = "higher favours greedy" if hib else "lower favours greedy"
        ax.set_ylabel(f"{fig_label}: greedy − diverse\n({direction}{scale_tag})")
        ax.grid(axis="y", **GRID)
        ax.set_axisbelow(True)
        for s in SPINES_OFF:
            ax.spines[s].set_visible(False)

    handles = [Line2D([], [], marker=styles[a][4], color=styles[a][2], mec="black",
                      mew=MARKER_EDGE, ms=MARKER_SIZE, ls="none", label=lbl)
               for a, _, lbl in PAIRS]
    handles.append(Patch(color=FLOOR_BAR_COLOR, alpha=FLOOR_BAR_ALPHA, label=FLOOR_LABEL))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False,
               bbox_to_anchor=(0.5, 0.0), handletextpad=0.5, columnspacing=1.4)

    for ext in FORMATS:
        p = out_stem.with_suffix(f".{ext}")
        fig.savefig(p, dpi=DPI)
        _chmod(p)
        print(f"  wrote {p}")
    plt.close(fig)


# -----------------------------------------------------------------------------

def main() -> None:
    styles = _load_arm_styles()
    for d in (TABLE_DIR, FIG_DIR):
        d.mkdir(parents=True, exist_ok=True)

    df, skipped = compute()
    if df.empty:
        print("no cells computed")
        return

    csv_path = TABLE_DIR / f"{OUT_STEM}.csv"
    df.to_csv(csv_path, index=False)
    _chmod(csv_path)
    print(f"  wrote {csv_path}")

    md_path = TABLE_DIR / f"{OUT_STEM}.md"
    write_md(df, skipped, md_path)
    print(f"  wrote {md_path}")

    plot(df, styles, FIG_DIR / OUT_STEM)

    print()
    print("All pairs:")
    print(_counts(df).to_string(index=False))
    print(f"\n{HEADLINE_PAIR} only:")
    print(_counts(df[df["pair_label"] == HEADLINE_PAIR]).to_string(index=False))
    for s in skipped:
        print("skipped:", s)


if __name__ == "__main__":
    main()
