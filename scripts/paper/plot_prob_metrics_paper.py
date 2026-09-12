#!/usr/bin/env python
"""Paper figures: probability-quality metrics across noise levels and methods.

Reads the long table `final_epoch_prob_metrics.csv` and renders, per (system,
figure set), one row of five panels (KL, sAUROC, skill score, REL, RES) with
the noise levels on the x axis and one line per arm. Styles, figure sets and
output locations are imported from the sibling `plot_levelsets_paper.py` so the
two figure families match.

Usage:
    python scripts/paper/plot_prob_metrics_paper.py --paper
    python scripts/paper/plot_prob_metrics_paper.py --paper --set ablation
    python scripts/paper/plot_prob_metrics_paper.py --paper --csv other.csv
"""
from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

# Shared style from the level-set figure script (scripts/paper is not a package).
_SIB = Path(__file__).resolve().parent / "plot_levelsets_paper.py"
_spec = importlib.util.spec_from_file_location("plot_levelsets_paper", _SIB)
_ls = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ls)

# BALD arms are the top-N (greedy) runs (decision 2026-09-10); greedy_diverse is out of the paper
# (the arm keys come from the sibling's ARM_STYLES / FIGURE_SETS; nothing is keyed by arm here)
ARM_STYLES = _ls.ARM_STYLES
FAMILY_LS = _ls.FAMILY_LS
FIGURE_SETS = _ls.FIGURE_SETS
SEED_LABEL = _ls.SEED_LABEL
SEED_MEAN_COLOR = _ls.SEED_MEAN_COLOR
SEED_MEAN_LW = _ls.SEED_MEAN_LW
SEED_BAND_COLOR = _ls.SEED_BAND_COLOR
SEED_BAND_ALPHA = _ls.SEED_BAND_ALPHA
MARKER_SIZE = _ls.MARKER_SIZE
LINE_ALPHA = _ls.LINE_ALPHA
level_title = _ls.level_title
SYSTEM_TITLES = _ls.SYSTEM_TITLES
LEVEL_ORDER = _ls.LEVEL_ORDER
PAPER_DIR = _ls.PAPER_DIR
FIG_DIR = _ls.FIG_DIR
FILE_MODE = _ls.FILE_MODE
FORMATS = _ls.FORMATS
DPI = _ls.DPI
FONT_FAMILY = getattr(_ls, "FONT_FAMILY", "sans-serif")
FONT_SIZE = getattr(_ls, "FONT_SIZE", 8)
TITLE_SIZE = getattr(_ls, "TITLE_SIZE", 9)
SUPTITLE_SIZE = getattr(_ls, "SUPTITLE_SIZE", 10)
SUPTITLE_SUB_SIZE = getattr(_ls, "SUPTITLE_SUB_SIZE", 7)
LABEL_SIZE = getattr(_ls, "LABEL_SIZE", 8)
TICK_SIZE = getattr(_ls, "TICK_SIZE", 7)
LEGEND_SIZE = getattr(_ls, "LEGEND_SIZE", 7.5)

# =============================================================================
# STYLE: everything specific to this figure family.
# =============================================================================
DEFAULT_CSV = PAPER_DIR / "tables" / "final_epoch_prob_metrics.csv"
# The final profile reads the per-system paper-budget table (final_epoch_table.py --paper).
PAPER_CSV = PAPER_DIR / "tables" / "final_epoch_prob_metrics_paper.csv"
OUT_SUFFIX = ""   # set from --suffix; appended to the output filenames

# Control row (aggregated non-adaptive FM, mean over seeds + `_2sd` columns).
CONTROL_KEY = "dir00_s42"   # single-seed control (2026-09-11)
# Rows to drop entirely: the pooled floor row.
SKIP_LABELS = {"FM floor (pooled 2sd)"}
SKIP_KEYS = {"floor"}

# Panels: (column, title, y scale). Order is the paper's order.
# REL_debiased can be slightly negative (debiasing subtracts a noise term), so a
# plain log axis is impossible: we use `symlog` with linthresh REL_LINTHRESH,
# which is linear in [-linthresh, linthresh] and log outside. Nothing is clipped.
REL_LINTHRESH = 1e-3
METRICS = [
    ("KL",           "KL divergence", "log"),
    ("sAUROC",       "sAUROC ↑",      "linear"),
    ("skill_score",  "skill score ↑", "linear"),
    ("REL_debiased", "REL ↓",         "symlog"),
    ("RES",          "RES ↑",         "linear"),
]
FINAL_METRICS = [("KL", "KL divergence", "log")]   # the final profile keeps KL only

# Deterministic levels get an asterisk in the tick label and one footnote.
DETERMINISTIC_LEVELS = {
    "cartpole_ppo": {"baseline"},
    "quad2d_rl": {"corridor_sine_ambient_baseline"},
    # quad3d = 800k-pool campaign (decision 2026-09-10); 100k is obsolete
    "quad3d_ppo800k": {"corridor_sine_ambient_f_0.00"},
}
FOOTNOTE = "* deterministic level (one rollout per eval cell): KL is a clipped log-loss"

SET_TITLES = {"main": "main evaluation", "ablation": "ablation"}

# Geometry (inches).
PANEL_W, PANEL_H = 1.75, 1.75
PANEL_W_PER_LEVEL = 0.42        # panel grows with the level count so 7 groups still fit
WRAP_ABOVE_LEVELS = 4            # more levels than this -> metric panels wrap into 2 rows
HSPACE_WRAPPED = 0.75            # vertical gap between the two rows (fraction of panel height)
LEFT_MARGIN, RIGHT_MARGIN = 0.6, 0.1
WSPACE = 0.5                     # fraction of panel width between panels
TITLE_BLOCK_H = 0.85             # suptitle + footnote above the panel titles (draft)
TITLE_BLOCK_H_FINAL = 0.32       # panel titles only (final profile)
SUPTITLE_Y_IN = 0.15             # suptitle baseline, inches from the top
FOOTNOTE_Y_IN = 0.42             # footnote top, inches from the top
XTICK_BLOCK_H = 0.5              # rotated tick labels under the panels
LEGEND_BLOCK_H = 0.3             # shared legend row under the tick labels
# Log axes: major ticks at 1, 2, 5 per decade labelled as plain decimals
# (0.05, 0.1, 0.2); minor tick labels off so narrow panels stay readable.
LOG_SUBS = (1.0, 2.0, 5.0)
XTICK_ROTATION = 30              # keeps "medium noise" / "f = 0.12, a = 0.03" apart
XPAD = 0.45                      # categorical x padding either side
GRID_KW = dict(color="0.85", lw=0.5, zorder=0)
CONTROL_Z = 1.5                  # (unused since the switch to bars; kept for the style block)
SINGLE_LEVEL_EBAR_LW, SINGLE_LEVEL_EBAR_CAP = 1.2, 3
# Grouped-bar rendering (decision 2026-09-10: levels are categories, no lines).
# Layout and ink follow the dataviz method: thin marks with a surface gap between
# them, hairline recessive grid, baseline only, text in ink tokens (never the
# series colour), one direct-labelled series, legend always present.
BAR_GROUP_WIDTH = 0.72           # total width of one level's group of bars
BAR_FILL = 0.78                  # bar width as a fraction of its slot (the rest is the gap)
# Validated categorical palette (dataviz reference instance, light surface;
# validator: main trio passes all-pairs, ablation trio passes adjacent).
# Overrides the shared ARM_STYLES colour for THIS figure family only.
PALETTE: dict = {}   # colours come from the shared ARM_STYLES (one palette for every figure)
SURFACE = "#fcfcfb"
INK_PRIMARY, INK_SECONDARY, INK_MUTED = "#0b0b0b", "#52514e", "#898781"
GRID_COLOR, GRID_LW = "#e1e0d9", 0.6
AXIS_COLOR, AXIS_LW = "#c3c2b7", 0.8
TITLE_PAD = 6
# Control as a reference mark, not a bar: dark hairline at the seed mean, light band ± 2sd.
CONTROL_AS_BAR = True            # the control is a peer arm (same model, uniform
                                 # acquisition): draw it as the first bar of each
                                 # group, not as a reference line (2026-09-11)
CONTROL_BAR_COLOR = "#52514e"    # neutral ink; the method hues stay reserved for the adaptive arms
CONTROL_LINE_COLOR, CONTROL_LINE_LW = "#0b0b0b", 1.1
CONTROL_BAND_COLOR = "#e1e0d9"
CONTROL_OVERHANG = 0.04          # how far the reference mark extends past the bar group
# Direct labels: one series only (the method), value on the cap, secondary ink.
DIRECT_LABELS = True         # draft only
SHOW_SUPTITLE = True         # draft only
SHOW_DET_MARK = True         # asterisk + footnote on deterministic levels; draft only
DIRECT_LABEL_ARM = "epi_bald_greedy"
DIRECT_LABEL_SIZE = 5.4
DIRECT_LABEL_FMT = {"KL": "{:.3f}", "sAUROC": "{:.3f}", "skill_score": "{:.2f}",
                    "REL_debiased": "{:.4f}", "RES": "{:.3f}"}
# Linear panels whose values sit near 1: start the y axis this far below the
# smallest bar rather than at 0, so differences are visible.
ZOOM_Y = {"sAUROC": 0.02, "skill_score": 0.05, "RES": 0.0}
BOUNDED_1 = {"sAUROC", "skill_score"}   # never draw axis headroom above 1
# =============================================================================


def _load(csv: Path) -> pd.DataFrame:
    df = pd.read_csv(csv)
    keep = ~df["arm_label"].isin(SKIP_LABELS) & ~df["arm_key"].isin(SKIP_KEYS)
    return df[keep].copy()


def _levels(df_sys: pd.DataFrame, system: str | None = None) -> list[str]:
    file_order = list(dict.fromkeys(df_sys["level"].tolist()))
    keep = _ls.LEVEL_FILTER.get(system) if system else None
    if keep:
        return [lv for lv in keep if lv in file_order]
    first = [lv for lv in LEVEL_ORDER if lv in file_order]
    return first + [lv for lv in file_order if lv not in first]


def _series(df_sys: pd.DataFrame, arm_key: str, levels: list[str], col: str):
    sub = df_sys[df_sys["arm_key"] == arm_key].set_index("level")
    return np.array([sub[col].get(lv, np.nan) if lv in sub.index else np.nan
                     for lv in levels], dtype=float)


def _draw_panel(ax, df_sys, system, spec, levels, x, ticklabels, col, title, scale, handles):
    """One grouped-bar panel: the adaptive arms of `spec` at every level, the
    control as a reference mark, one direct-labelled series. Shared by the
    per-system figure (one panel per metric) and the combined figure (one
    panel per system for a single metric)."""
    # Grouped bars for the adaptive arms (levels are categories, so no lines).
    # The non-adaptive control is not a series: it is drawn as a reference mark
    # across each level's group (a short dark line at the seed mean with a light
    # band for ± 2sd), so the eye reads every bar against "the baseline to beat".
    bar_arms = list(spec["arms"])
    if spec["band"] and CONTROL_AS_BAR:
        bar_arms.insert(_ls.CONTROL_POSITION, CONTROL_KEY)
    n_bars = len(bar_arms)
    slot = BAR_GROUP_WIDTH / n_bars
    width = slot * BAR_FILL
    offsets = (np.arange(n_bars) - (n_bars - 1) / 2) * slot
    ax.figure.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.grid(True, axis="y", color=GRID_COLOR, lw=GRID_LW, zorder=0)
    ax.set_axisbelow(True)
    if spec["band"] and not CONTROL_AS_BAR:
        mean = _series(df_sys, CONTROL_KEY, levels, col)
        sd2 = np.nan_to_num(_series(df_sys, CONTROL_KEY, levels, f"{col}_2sd"))
        half = BAR_GROUP_WIDTH / 2 + CONTROL_OVERHANG
        for xi, m, s in zip(x, mean, sd2):
            if np.isnan(m):
                continue
            ax.fill_between([xi - half, xi + half], m - s, m + s, color=CONTROL_BAND_COLOR,
                            lw=0, zorder=1)
            ax.hlines(m, xi - half, xi + half, color=CONTROL_LINE_COLOR, lw=CONTROL_LINE_LW,
                      zorder=4)
        handles.setdefault(SEED_LABEL, (Patch(facecolor=CONTROL_BAND_COLOR, lw=0),
                                        Line2D([], [], color=CONTROL_LINE_COLOR,
                                               lw=CONTROL_LINE_LW))
                           if np.nanmax(sd2) > 0 else Line2D([], [], color=CONTROL_LINE_COLOR,
                                                              lw=CONTROL_LINE_LW))
    for k, arm in enumerate(bar_arms):
        if arm == CONTROL_KEY:
            label, color = SEED_LABEL, CONTROL_BAR_COLOR
        else:
            label = ARM_STYLES[arm][0]
            color = PALETTE.get(arm, ARM_STYLES[arm][2])
        y = _series(df_sys, arm, levels, col)
        ok = ~np.isnan(y)
        if not ok.any():
            continue
        ax.bar(x[ok] + offsets[k], y[ok], width, color=color, lw=0, zorder=3)
        handles.setdefault(label, Patch(facecolor=color, lw=0))
        if arm == DIRECT_LABEL_ARM and DIRECT_LABELS:
            for xi, yi in zip(x[ok] + offsets[k], y[ok]):
                ax.annotate(DIRECT_LABEL_FMT.get(col, "{:.2f}").format(yi), (xi, yi),
                            xytext=(0, 2), textcoords="offset points", ha="center",
                            va="bottom", fontsize=DIRECT_LABEL_SIZE, color=INK_SECONDARY,
                            zorder=6)
    plain = FuncFormatter(lambda v, _: f"{v:g}")
    if scale == "log":
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=LOG_SUBS))
        ax.yaxis.set_major_formatter(plain)
        ax.yaxis.set_minor_formatter(NullFormatter())
    elif scale == "symlog":
        ax.set_yscale("symlog", linthresh=REL_LINTHRESH)
        ax.yaxis.set_major_formatter(plain)
        ax.yaxis.set_minor_formatter(NullFormatter())
    elif col in ZOOM_Y:
        # Bars from zero hide a 0.96 vs 0.99 difference; start the axis
        # just under the smallest value instead (stated in the y label).
        vals = np.concatenate([_series(df_sys, a, levels, col) for a in bar_arms]
                              + ([_series(df_sys, CONTROL_KEY, levels, col)] if spec["band"] else []))
        lo, hi = np.nanmin(vals), np.nanmax(vals)
        ax.set_ylim(max(0.0, lo - ZOOM_Y[col]), min(1.0, hi + ZOOM_Y[col]) if col in BOUNDED_1 else None)
    ax.set_title(title, fontsize=TITLE_SIZE, color=INK_PRIMARY, pad=TITLE_PAD)
    ax.set_xticks(x)
    ax.set_xticklabels(ticklabels, fontsize=TICK_SIZE, color=INK_MUTED,
                       rotation=XTICK_ROTATION, ha="right", rotation_mode="anchor")
    ax.set_xlim(-XPAD, len(levels) - 1 + XPAD)
    ax.tick_params(axis="y", labelsize=TICK_SIZE, colors=INK_MUTED, length=0)
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", which="minor", length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS_COLOR)
    ax.spines["bottom"].set_linewidth(AXIS_LW)


def render(df: pd.DataFrame, system: str, set_name: str) -> list[Path]:
    spec = FIGURE_SETS[set_name]
    df_sys = df[df["system"] == system]
    if df_sys.empty:
        print(f"[skip] {system}: no rows")
        return []
    levels = _levels(df_sys, system)
    x = np.arange(len(levels))
    det = DETERMINISTIC_LEVELS.get(system, set()) if SHOW_DET_MARK else set()
    ticklabels = [level_title(lv, system) + ("*" if lv in det else "") for lv in levels]
    has_det = any(lv in det for lv in levels)

    n = len(METRICS)
    panel_w = max(PANEL_W, PANEL_W_PER_LEVEL * len(levels))
    # Many levels: wrap the metric panels into two rows so the page stays readable.
    nrows = 2 if len(levels) > WRAP_ABOVE_LEVELS else 1
    ncols = int(np.ceil(n / nrows))
    fig_w = LEFT_MARGIN + ncols * panel_w + (ncols - 1) * WSPACE * panel_w + RIGHT_MARGIN
    title_h = TITLE_BLOCK_H if SHOW_SUPTITLE else TITLE_BLOCK_H_FINAL
    fig_h = title_h + nrows * (PANEL_H + XTICK_BLOCK_H) + LEGEND_BLOCK_H
    plt.rcParams.update({"font.family": FONT_FAMILY, "font.size": FONT_SIZE})
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)
    axes = axes.flatten()
    for ax in axes[n:]:
        ax.axis("off")
    fig.subplots_adjust(
        left=LEFT_MARGIN / fig_w, right=1 - RIGHT_MARGIN / fig_w,
        top=1 - title_h / fig_h, bottom=(LEGEND_BLOCK_H + XTICK_BLOCK_H) / fig_h,
        wspace=WSPACE, hspace=HSPACE_WRAPPED)

    handles: dict[str, object] = {}
    for ax, (col, title, scale) in zip(axes, METRICS):
        _draw_panel(ax, df_sys, system, spec, levels, x, ticklabels, col, title, scale, handles)

    order = [ARM_STYLES[a][0] for a in spec["arms"]]
    if spec["band"]:
        order.insert(_ls.CONTROL_POSITION, SEED_LABEL)
    keys = [k for k in order if k in handles]
    fig.legend([handles[k] for k in keys], keys, loc="lower center",
               ncol=len(keys), frameon=False, fontsize=LEGEND_SIZE, labelcolor=INK_SECONDARY,
               bbox_to_anchor=(0.5, 0.0), handlelength=1.6, handleheight=0.9, columnspacing=1.6)

    if SHOW_SUPTITLE:
      fig.suptitle(f"{SYSTEM_TITLES.get(system, system)} — {SET_TITLES[set_name]}",
                 fontsize=SUPTITLE_SIZE, y=1 - SUPTITLE_Y_IN / fig_h, color=INK_PRIMARY)
    if has_det:
        fig.text(0.5, 1 - FOOTNOTE_Y_IN / fig_h, FOOTNOTE, ha="center", va="top",
                 fontsize=SUPTITLE_SUB_SIZE, color=INK_MUTED)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    for fmt in FORMATS:
        p = FIG_DIR / f"probmetrics_{system}_{set_name}{OUT_SUFFIX}.{fmt}"
        fig.savefig(p, dpi=DPI)
        os.chmod(p, FILE_MODE)
        out.append(p)
    plt.close(fig)
    return out


def render_combined(df: pd.DataFrame, set_name: str, systems: list[str]) -> list[Path]:
    """One figure per set: a panel per SYSTEM for the first metric in METRICS
    (the paper's KL-only view), levels on x, the same bar encoding."""
    spec = FIGURE_SETS[set_name]
    col, mtitle, scale = METRICS[0]
    systems = [s for s in systems if not df[df["system"] == s].empty]
    n = len(systems)
    n_lv = max(len(_levels(df[df["system"] == s], s)) for s in systems)
    panel_w = max(PANEL_W, PANEL_W_PER_LEVEL * n_lv)
    fig_w = LEFT_MARGIN + n * panel_w + (n - 1) * WSPACE * panel_w + RIGHT_MARGIN
    title_h = TITLE_BLOCK_H if SHOW_SUPTITLE else TITLE_BLOCK_H_FINAL
    fig_h = title_h + PANEL_H + XTICK_BLOCK_H + LEGEND_BLOCK_H
    plt.rcParams.update({"font.family": FONT_FAMILY, "font.size": FONT_SIZE})
    fig, axes = plt.subplots(1, n, figsize=(fig_w, fig_h), squeeze=False)
    axes = axes.flatten()
    fig.subplots_adjust(left=LEFT_MARGIN / fig_w, right=1 - RIGHT_MARGIN / fig_w,
                        top=1 - title_h / fig_h,
                        bottom=(LEGEND_BLOCK_H + XTICK_BLOCK_H) / fig_h, wspace=WSPACE)
    handles: dict[str, object] = {}
    has_det = False
    for ax, system in zip(axes, systems):
        df_sys = df[df["system"] == system]
        pending: list[int] = []
        if _ls.USE_PAPER_NAMES:
            # Final profile: every system gets the same three slots, in the
            # grid's column order; a slot with no dataset yet is an empty
            # placeholder so the panels line up (pendulum deterministic).
            by_name = {n: k for k, n in _ls.PAPER_LEVELS.get(system, [])}
            present = set(df_sys["level"])
            levels, ticklabels = [], []
            for cname in _ls.GRID_COLUMNS:
                key = by_name.get(cname)
                if key is None or key not in present:
                    key = f"__{cname}__"
                    pending.append(len(levels))
                levels.append(key)
                ticklabels.append(cname)
        else:
            levels = _levels(df_sys, system)
            ticklabels = None
        x = np.arange(len(levels))
        det = DETERMINISTIC_LEVELS.get(system, set()) if SHOW_DET_MARK else set()
        has_det |= any(lv in det for lv in levels)
        if ticklabels is None:
            ticklabels = [level_title(lv, system) + ("*" if lv in det else "") for lv in levels]
        _draw_panel(ax, df_sys, system, spec, levels, x, ticklabels, col,
                    SYSTEM_TITLES.get(system, system), scale, handles)
        for xi in pending:
            ax.text(xi, 0.5, _ls.GRID_PLACEHOLDER_TEXT, transform=ax.get_xaxis_transform(),
                    ha="center", va="center", color=_ls.GRID_PLACEHOLDER_COLOR,
                    fontsize=TICK_SIZE)
    for ax in axes[1:]:
        ax.set_ylabel("")
    axes[0].set_ylabel(mtitle, fontsize=LABEL_SIZE, color=INK_SECONDARY)
    order = [ARM_STYLES[a][0] for a in spec["arms"]]
    if spec["band"]:
        order.insert(_ls.CONTROL_POSITION, SEED_LABEL)
    keys = [k for k in order if k in handles]
    fig.legend([handles[k] for k in keys], keys, loc="lower center",
               ncol=len(keys), frameon=False, fontsize=LEGEND_SIZE, labelcolor=INK_SECONDARY,
               bbox_to_anchor=(0.5, 0.0), handlelength=1.6, handleheight=0.9, columnspacing=1.6)
    if SHOW_SUPTITLE:
      fig.suptitle(f"{mtitle} — {SET_TITLES[set_name]}", fontsize=SUPTITLE_SIZE,
                 y=1 - SUPTITLE_Y_IN / fig_h, color=INK_PRIMARY)
    if has_det:
        fig.text(0.5, 1 - FOOTNOTE_Y_IN / fig_h, FOOTNOTE, ha="center", va="top",
                 fontsize=SUPTITLE_SUB_SIZE, color=INK_MUTED)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    for fmt in FORMATS:
        p = FIG_DIR / f"probmetrics_{col}_{set_name}{OUT_SUFFIX}.{fmt}"
        fig.savefig(p, dpi=DPI)
        os.chmod(p, FILE_MODE)
        out.append(p)
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--paper", action="store_true", help="render every system")
    ap.add_argument("--set", default="all", choices=["main", "ablation", "all"])
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--systems", nargs="*", default=None)
    ap.add_argument("--suffix", default="", help="output filename suffix, e.g. _15k")
    ap.add_argument("--profile", default="draft", choices=["draft", "final"],
                    help="final: KL only, the three named levels, figures under paper_final")
    ap.add_argument("--combined", action="store_true",
                    help="one figure per set with a panel per system (first metric only)")
    args = ap.parse_args()
    if not args.paper and not args.systems:
        ap.error("pass --paper or --systems ...")
    global OUT_SUFFIX, METRICS, FIG_DIR, DIRECT_LABELS, SHOW_SUPTITLE, SHOW_DET_MARK
    OUT_SUFFIX = args.suffix
    _ls.set_profile(args.profile)
    FIG_DIR = _ls.FIG_DIR
    if args.profile == "final":
        METRICS = list(FINAL_METRICS)
        DIRECT_LABELS = SHOW_SUPTITLE = SHOW_DET_MARK = False
        if args.csv == DEFAULT_CSV:
            args.csv = PAPER_CSV

    df = _load(args.csv)
    systems = args.systems or list(dict.fromkeys(df["system"].tolist()))
    sets = list(FIGURE_SETS) if args.set == "all" else [args.set]
    if args.combined:
        for s in sets:
            for p in render_combined(df, s, systems):
                print(p)
        return
    for system in systems:
        for s in sets:
            for p in render(df, system, s):
                print(p)


if __name__ == "__main__":
    main()
