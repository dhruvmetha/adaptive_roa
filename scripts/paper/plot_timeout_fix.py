#!/usr/bin/env python
"""Figures for the timeout-corrected campaign, in ITS OWN docs tree.

    docs/stochastic/timeout_fix/paper_draft/figures/   three metric rows, epoch
                                                       subtitle, every level
    docs/stochastic/timeout_fix/paper_final/figures/   F0.5 only, paper level
                                                       names, no run chrome

Both profiles come from plot_levelsets_paper.py, the campaign's single design
surface, with its output root redirected here; nothing re-chooses a colour, a
font or a panel size. Each profile also gets a KL figure per system.

    ./env/bin/python scripts/paper/plot_timeout_fix.py [--systems ...]
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

_spec = importlib.util.spec_from_file_location(
    "levelsets_paper", Path(__file__).with_name("plot_levelsets_paper.py"))
S = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S)

DOCS = Path("/common/users/shared/pracsys/genMoPlan/docs/stochastic")
ROOT = DOCS / "timeout_fix"
SET = "timeoutfix"                      # FIGURE_SETS entry: the campaign's 4 arms
ARMS = ["epi_bald_greedy", "dir00_s42", "bnn_mfvi_a1_greedy", "clf_epi_bald_greedy"]
UNIFORM_STYLE = ("FM uniform", "fm", "#000000", 1.4, "o")

# system -> its wide CSV in this campaign's scorer output
SYSTEMS = {
    "pendulum_lqr":  ROOT / "pendulum_lqr_all_levels.csv",
    "cartpole_ppo":  ROOT / "cartpole_safe_explorer_ppo_all_levels.csv",
    "quad2d_rl":     ROOT / "quad2d_corridor_sine_ambient_all_levels.csv",
    "quad3d_ppo800k": ROOT / "quad3d_ppo1500k_corridor_sine_ambient_all_levels.csv",
    # The 40k-budget rerun of the same four q3d cells (10,000 + 10x3,000 instead
    # of 5,000 + 10x1,000). Its CSV lives in the sibling timeout_fix_40k docs
    # dir -- deliberately NOT merged with the 15k one, since a paper script that
    # averaged two budgets under one arm name would produce a curve belonging to
    # neither. The figures land beside the 15k ones with a _40k suffix, which is
    # what the system key gives us for free (user, 2026-09-12).
    "quad3d_40k": DOCS / "timeout_fix_40k" / "quad3d_40k_corridor_sine_ambient_all_levels.csv",
}
# The final profile renames levels; this campaign carries levels the paper set
# does not (cartpole/pendulum high, q3d f_0.20_a0.035), so extend rather than
# inherit, or those columns would be filtered out of the final figures.
EXTRA_PAPER_LEVELS = {
    "pendulum_lqr": [("baseline", "deterministic"), ("low", "low"), ("med", "med"),
                     ("high", "high")],
    "cartpole_ppo": [("baseline", "deterministic"), ("low", "low"), ("med", "med"),
                     ("high", "high")],
    "quad2d_rl": [("corridor_sine_ambient_baseline", "deterministic"),
                  ("corridor_sine_ambient_smooth", "low"),
                  ("corridor_sine_ambient_loud", "high")],
    "quad3d_ppo800k": [("corridor_sine_ambient_f_0.00", "deterministic"),
                       ("corridor_sine_ambient_f_0.12_a0.03", "low"),
                       ("corridor_sine_ambient_f_0.20_a0.035", "medium"),
                       ("corridor_sine_ambient_f_0.40_a0.04", "high")],
    "quad3d_40k": [("corridor_sine_ambient_f_0.00", "deterministic"),
                   ("corridor_sine_ambient_f_0.12_a0.03", "low"),
                   ("corridor_sine_ambient_f_0.20_a0.035", "medium"),
                   ("corridor_sine_ambient_f_0.40_a0.04", "high")],
}


def levelsets_path(csv_path: Path) -> Path:
    return csv_path.with_name(csv_path.stem + "_levelsets" + csv_path.suffix)


def style_of(arm):
    return UNIFORM_STYLE if arm == "dir00_s42" else S.ARM_STYLES[arm]


def kl_figure(system: str, wide: Path, out_dir: Path) -> Path | None:
    """KL per arm at each arm's deepest epoch, grouped by noise level."""
    df = pd.read_csv(wide)
    levels = [l for l in S.LEVEL_ORDER if l in set(df["level"])] + \
             [l for l in dict.fromkeys(df["level"]) if l not in S.LEVEL_ORDER]
    if S.LEVEL_FILTER.get(system):
        levels = [l for l in levels if l in S.LEVEL_FILTER[system]]
    if not levels:
        return None
    fig, ax = plt.subplots(figsize=(S.PANEL_W * len(levels) + 1.2, S.PANEL_H + 1.1), dpi=S.DPI)
    xs, labels, drew = [], [], False
    for i, lv in enumerate(levels):
        for j, arm in enumerate(ARMS):
            sub = df[(df["level"] == lv) & (df["arm"] == arm) & df["KL"].notna()]
            if sub.empty:
                continue
            row = sub.loc[sub["epoch"].idxmax()]
            x = i * (len(ARMS) + 1) + j
            colour, marker = style_of(arm)[2], style_of(arm)[4]
            ax.vlines(x, 0, row["KL"], color=colour, lw=1.3, alpha=0.5)
            ax.plot([x], [row["KL"]], marker=marker, color=colour, markersize=S.MARKER_SIZE + 2)
            if S.SHOW_EPOCH_SUBTITLE:      # draft only: say which epoch each point is
                ax.annotate(f"ep{int(row['epoch'])}", (x, row["KL"]), textcoords="offset points",
                            xytext=(0, 7), ha="center", fontsize=S.TRUE_SET_FONT,
                            color=S.TRUE_SET_COLOR)
            xs.append(x); labels.append(style_of(arm)[0].replace(" (ours)", ""))
            drew = True
    if not drew:
        plt.close(fig); return None
    ax.set_xticks(xs); ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=S.TICK_SIZE)
    ax.set_yscale("log"); ax.set_ylabel("KL (lower is better)", fontsize=S.LABEL_SIZE)
    ax.grid(True, axis="y", **S.GRID)
    for sp in S.SPINES_OFF:
        ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=S.TICK_SIZE)
    for i, lv in enumerate(levels):
        ax.text(i * (len(ARMS) + 1) + (len(ARMS) - 1) / 2, ax.get_ylim()[1],
                S.level_title(lv, system), ha="center", va="bottom", fontsize=S.TITLE_SIZE)
    ax.legend(handles=[Line2D([], [], color=style_of(a)[2], marker=style_of(a)[4], ls="",
                              markersize=4, label=style_of(a)[0]) for a in ARMS],
              loc="upper left", ncol=2, frameon=False, fontsize=S.LEGEND_SIZE)
    fig.suptitle(f"{S.SYSTEM_TITLES.get(system, system)}: KL", fontsize=S.SUPTITLE_SIZE)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"kl_{system}.png"
    fig.savefig(p); plt.close(fig)
    return p


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--systems", nargs="*", default=list(SYSTEMS))
    ap.add_argument("--profiles", nargs="*", default=["draft", "final"])
    args = ap.parse_args()
    # This campaign's level names, so the final profile keeps every column.
    S.PAPER_LEVELS.update(EXTRA_PAPER_LEVELS)
    S.PAPER_LEVEL_NAMES.update({(s, k): n for s, lv in EXTRA_PAPER_LEVELS.items() for k, n in lv})
    S.FINAL_LEVEL_FILTER.update({s: [k for k, _ in lv] for s, lv in EXTRA_PAPER_LEVELS.items()})
    # FINAL_EPOCHS pins each system to the ORIGINAL campaign's budget epoch
    # (cartpole 11, quad2d 12, quad3d 3). These runs are 11 epochs, 0..10, on a
    # different schedule, so asking for epoch 11 plots nothing at all. Read every
    # system at its deepest common epoch instead.
    S.FINAL_EPOCHS.update({s: "common" for s in SYSTEMS})
    # Without this the 40k figures carry the bare dict key as their title.
    # The budget is in the title because it is the ONLY thing separating these
    # panels from the 15k ones they sit beside in the same directory.
    S.SYSTEM_TITLES.setdefault("quad3d_40k", "Quadrotor 3D (RL) — 40k budget")
    for profile in args.profiles:
        S.set_profile(profile)
        S.FIG_DIR = ROOT / f"paper_{profile}" / "figures"      # redirect the output root
        S._select_set(SET)
        for system in args.systems:
            wide = SYSTEMS[system]
            if not wide.exists():
                print(f"[skip] {profile}/{system}: not scored yet")
                continue
            # Only ask for levels this campaign has actually scored: the final
            # profile filters by name, and a named-but-empty level would render
            # an empty column.
            present = set(pd.read_csv(wide)["level"])
            if S.LEVEL_FILTER.get(system):
                S.LEVEL_FILTER[system] = [l for l in S.LEVEL_FILTER[system] if l in present]
            outs, skipped = S.render(system, levelsets_path(wide), "common", wide)
            for p in outs:
                print(f"[ok] {p}")
            for s in skipped:
                print(f"     skipped {s}")
            p = kl_figure(system, wide, S.FIG_DIR)
            print(f"[ok] {p}" if p else f"[skip] {profile}/{system}: no KL rows")


if __name__ == "__main__":
    main()
