#!/usr/bin/env python
"""Variant of the final level-set grid (levelsets_ablation.png): FM-BALD and
BNN-BALD only, no uniform-sampling line, no true level-set mass shading.

Two y-axis modes, each its own file:
  fit   each panel's y range is fitted to its two curves, clamped to [0, 1]
  full  every panel runs exactly from 0 to 1, no padding

Epochs, levels, arm styles and the level-set stores all come from the sibling
`plot_levelsets_paper.py` (final profile). The epoch per level is resolved
against the ablation set's arms, so every value matches levelsets_ablation.png.
Output goes to its own directory, never next to the paper figures.

Usage:
    python scripts/paper/plot_levelsets_bald_only.py            # both modes
    python scripts/paper/plot_levelsets_bald_only.py --y fit
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FormatStrFormatter, MaxNLocator  # noqa: E402

# Shared style and plumbing (scripts/paper is not a package).
_SIB = Path(__file__).resolve().parent / "plot_levelsets_paper.py"
_spec = importlib.util.spec_from_file_location("plot_levelsets_paper", _SIB)
_ls = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ls)
_ls.set_profile("final")

# =============================================================================
# STYLE
# =============================================================================
ARMS = ["epi_bald_greedy", "bnn_mfvi_a1_greedy"]   # legend order
EPOCH_SET = "ablation"          # epochs resolved exactly as in levelsets_ablation.png
OUT_DIR = _ls.STOCH_DIR / "paper_final" / "figures_bald_only"
DIR_MODE = 0o2770
FIT_PAD_FRAC = 0.08             # fit mode: padding beyond the data, as a fraction of its span
FIT_MIN_SPAN = 0.10             # fit mode: narrowest y range, so a 0.01 gap does not fill a panel
FIT_NBINS = 5
FIT_WSPACE = 0.24               # fit mode labels every panel's y axis, so columns sit further apart
FULL_TICKS = [0.0, 0.25, 0.5, 0.75, 1.0]

# =============================================================================
# Plumbing
# =============================================================================


def _fit_ylim(vals: list[float]) -> tuple[float, float]:
    v = [x for x in vals if not math.isnan(x)]
    if not v:
        return 0.0, 1.0
    lo, hi = min(v), max(v)
    pad = FIT_PAD_FRAC * (hi - lo)
    lo, hi = lo - pad, hi + pad
    if hi - lo < FIT_MIN_SPAN:
        mid = (lo + hi) / 2
        lo, hi = mid - FIT_MIN_SPAN / 2, mid + FIT_MIN_SPAN / 2
    if hi > 1.0:
        lo, hi = lo - (hi - 1.0), 1.0
    if lo < 0.0:
        lo, hi = 0.0, min(1.0, hi - lo)
    return max(lo, 0.0), min(hi, 1.0)


def render(mode: str) -> Path:
    _ls._select_set(EPOCH_SET)
    _ls._apply_rc()
    col, ylab = _ls.METRICS[0]
    frames = {}
    for system in _ls.GRID_SYSTEMS:
        path = _ls.LEVELSETS_DIR / f"{system}_levelsets_b50.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df = df[df["beta"] >= _ls.BETA_MIN - 1e-9].copy()
        frames[system] = _ls.apply_aliases(df, system)
    systems = [s for s in _ls.GRID_SYSTEMS if s in frames]
    n_rows, n_cols = len(systems), len(_ls.GRID_COLUMNS)
    wspace = FIT_WSPACE if mode == "fit" else _ls.GRID_WSPACE
    legend_h = _ls.LEGEND_ROW_H + 0.12
    fig_w = (_ls.GRID_ROW_LABEL_W + _ls.LEFT_MARGIN + n_cols * _ls.GRID_PANEL_W
             + (n_cols - 1) * (wspace - _ls.GRID_WSPACE) * _ls.GRID_PANEL_W + _ls.RIGHT_MARGIN)
    fig_h = n_rows * _ls.GRID_PANEL_H + 0.35 + _ls.XLABEL_BLOCK_H + legend_h
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), squeeze=False)
    handles: dict[str, object] = {}
    for i, system in enumerate(systems):
        df = frames[system]
        by_name = {v: k for k, v in dict(_ls.PAPER_LEVELS[system]).items()}
        for j, cname in enumerate(_ls.GRID_COLUMNS):
            ax = axes[i][j]
            ax.grid(True, **_ls.GRID)
            for sp in _ls.SPINES_OFF:
                ax.spines[sp].set_visible(False)
            level = by_name.get(cname)
            placeholder = level is None or level not in set(df["level"])
            plotted: list[float] = []
            if placeholder:
                betas = [round(0.5 + 0.05 * k, 2) for k in range(10)]
                ax.text(0.5, 0.5, _ls.GRID_PLACEHOLDER_TEXT, transform=ax.transAxes, ha="center",
                        va="center", color=_ls.GRID_PLACEHOLDER_COLOR, fontsize=_ls.FONT_SIZE)
            else:
                lvl = df[df["level"] == level]
                ep = _ls._pick_epoch(lvl, _ls.paper_epoch(system, "common"))
                at = lvl[lvl["epoch"].astype(int) == ep]
                betas = sorted(at["beta"].round(6).unique())
                print(f"{system:15s} {cname:13s} {level:36s} epoch {ep}")
                for arm in ARMS:
                    rows = at[at["arm"] == arm].sort_values("beta")
                    if rows.empty:
                        print(f"   missing {arm}")
                        continue
                    label, fam, color, lw, mk = _ls.ARM_STYLES[arm]
                    ys = _ls._series(rows, col, betas)
                    plotted += ys
                    ax.plot(betas, ys, ls=_ls.FAMILY_LS[fam], color=color, lw=lw, marker=mk,
                            ms=_ls.MARKER_SIZE, alpha=_ls.LINE_ALPHA, clip_on=False,
                            zorder=4 if arm == _ls.OURS else 3)
                    handles.setdefault(label, Line2D([], [], ls=_ls.FAMILY_LS[fam], color=color,
                                                     lw=lw, marker=mk, ms=_ls.MARKER_SIZE))
            ax.set_xlim(betas[0] - _ls.X_PAD, betas[-1] + _ls.X_PAD)
            ax.set_xticks(betas)
            ax.set_xticklabels([f"{b:.2f}" if k % _ls.X_TICK_LABEL_EVERY == 0 else ""
                                for k, b in enumerate(betas)])
            if mode == "fit" and plotted:
                ax.set_ylim(*_fit_ylim(plotted))
                ax.yaxis.set_major_locator(MaxNLocator(nbins=FIT_NBINS, steps=[1, 2, 5, 10],
                                                           min_n_ticks=3))
                ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
            else:
                ax.set_ylim(0.0, 1.0)
                ax.set_yticks(FULL_TICKS)
            if i == 0:
                ax.set_title(cname)
            if i == n_rows - 1:
                ax.set_xlabel(_ls.X_LABEL)
            else:
                ax.tick_params(labelbottom=False)
            if j == 0:
                ax.set_ylabel(ylab)
            elif mode == "full" or placeholder:
                ax.tick_params(labelleft=False)
        axes[i][0].annotate(_ls.SYSTEM_TITLES.get(system, system), xy=(0, 0.5),
                            xycoords="axes fraction",
                            xytext=(-_ls.GRID_ROW_LABEL_W * 72 - 8, 0), textcoords="offset points",
                            rotation=90, ha="center", va="center", fontsize=_ls.GRID_ROW_LABEL_SIZE)
    keys = [_ls.ARM_STYLES[a][0] for a in ARMS if _ls.ARM_STYLES[a][0] in handles]
    fig.legend([handles[k] for k in keys], keys, loc="lower center", ncol=len(keys), frameon=False,
               bbox_to_anchor=(0.5, 0.0), handlelength=2.6, columnspacing=1.2)
    left = (_ls.GRID_ROW_LABEL_W + _ls.LEFT_MARGIN) / fig_w
    fig.subplots_adjust(left=left, right=1 - _ls.RIGHT_MARGIN / fig_w, top=1 - 0.35 / fig_h,
                        bottom=(legend_h + _ls.XLABEL_BLOCK_H) / fig_h,
                        wspace=wspace, hspace=_ls.GRID_HSPACE)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    os.chmod(OUT_DIR, DIR_MODE)
    out = OUT_DIR / f"levelsets_bald_{mode}.png"
    fig.savefig(out, dpi=_ls.DPI)
    os.chmod(out, _ls.FILE_MODE)
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--y", default="both", choices=["both", "fit", "full"],
                    help="fit: per-panel y range clamped to [0, 1]; full: every panel 0 to 1")
    args = ap.parse_args()
    for mode in (["fit", "full"] if args.y == "both" else [args.y]):
        print(f"[ok] {render(mode)}")


if __name__ == "__main__":
    main()
