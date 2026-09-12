#!/usr/bin/env python
"""Fraction of the eval set inside the true level set, for every stochastic dataset.

For each dataset under the stochastic data tree that ships an evaluation grid
(`<system>/<family>/<controller>/<level>/eval_success_prob.npz`), plot the
fraction of eval cells whose Monte Carlo success probability is at least beta,
as beta sweeps the level grid. One panel per (system, family, controller), one
line per noise level. Slice sub-grids (`.../slices/...`) are skipped: they are
2-D projections, not the eval set.

Data only; no runs are read. Every design choice is in the STYLE block.
"""
from __future__ import annotations
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# =============================================================================
# STYLE
# =============================================================================
DATA = Path("/common/users/shared/pracsys/genMoPlan/data_trajectories/stochastic")
PAPER = Path("/common/users/shared/pracsys/genMoPlan/docs/stochastic/paper_draft")
FIG_DIR, TABLE_DIR = PAPER / "figures", PAPER / "tables"
OUT_STEM = "true_levelset_mass"
FILE_MODE = 0o660
FORMATS, DPI = ("png",), 200   # png only (2026-09-11)

BETAS = np.round(np.arange(0.05, 0.96, 0.05), 2)     # x grid for the curves
PAPER_BETAS = np.round(np.arange(0.50, 0.96, 0.05), 2)  # the paper's level grid, shaded
PAPER_SHADE = dict(color="0.85", alpha=0.5, lw=0)

# Panel per (system, family, controller); title override and level ordering.
PANEL_TITLES = {
    ("pendulum", "gaussian_signal", "lqr"): "Pendulum, LQR",
    ("cartpole", "gaussian_signal", "lqr"): "Cartpole, LQR",
    ("cartpole", "gaussian_signal", "safe_explorer_ppo"): "Cartpole, RL",
    ("quadrotor2D", "corridor_sine_ambient", "rl"): "Quadrotor 2D, RL",
    ("quadrotor3D", "corridor_sine_ambient", "lqr"): "Quadrotor 3D, LQR",
    ("quadrotor3D", "corridor_sine_ambient", "ppo"): "Quadrotor 3D, RL (100k pool)",
    ("quadrotor3D", "corridor_sine_ambient", "ppo_800k"): "Quadrotor 3D, RL (800k pool)",
}
LEVEL_ORDER = ["baseline", "low", "med", "high", "smooth", "sharp"]   # named levels first
# Sequential colours per level within a panel (light -> dark = low -> high noise).
CMAP = "viridis"
LINE_LW, MARKER, MARKER_SIZE = 1.6, "o", 2.8
Y_LOG = False
Y_LIM = (0.0, 1.0)
N_COLS = 4
PANEL_W, PANEL_H = 2.7, 2.2
FONT_SIZE, TITLE_SIZE, LEGEND_SIZE = 8, 9, 6.5
X_LABEL, Y_LABEL = "level β", "fraction of eval set with p ≥ β"
SUPTITLE = "True level-set mass of every stochastic evaluation grid"
# =============================================================================


def level_key(level: str):
    if level in LEVEL_ORDER:
        return (0, LEVEL_ORDER.index(level), level)
    try:
        return (1, float(level.split("_f_")[-1].split("_")[0].replace("f_", "")), level)
    except ValueError:
        return (2, 0.0, level)


def find_datasets():
    out = []
    for npz in sorted(DATA.rglob("eval_success_prob.npz")):
        rel = npz.relative_to(DATA).parts
        if "slices" in rel or len(rel) != 5:
            continue
        system, family, controller, level, _ = rel
        out.append((system, family, controller, level, npz))
    return out


def main() -> None:
    rows = []
    panels: dict[tuple, dict[str, np.ndarray]] = {}
    for system, family, controller, level, npz in find_datasets():
        with np.load(npz) as z:
            p = z["p_success"].astype(float)
            trials = float(np.mean(z["trials"])) if "trials" in z.files else float("nan")
        frac = np.array([(p >= b).mean() for b in BETAS])
        panels.setdefault((system, family, controller), {})[level] = frac
        for b, f in zip(BETAS, frac):
            rows.append(dict(system=system, family=family, controller=controller, level=level,
                             n_cells=len(p), trials_per_cell=trials, beta=b, frac_pos=f))
    df = pd.DataFrame(rows)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = TABLE_DIR / f"{OUT_STEM}.csv"
    df.to_csv(csv_path, index=False, float_format="%.6g")
    os.chmod(csv_path, FILE_MODE)

    keys = [k for k in PANEL_TITLES if k in panels] + [k for k in panels if k not in PANEL_TITLES]
    n = len(keys)
    ncols = min(N_COLS, n)
    nrows = int(np.ceil(n / ncols))
    plt.rcParams.update({"font.size": FONT_SIZE, "axes.titlesize": TITLE_SIZE,
                         "legend.fontsize": LEGEND_SIZE, "pdf.fonttype": 42})
    fig, axes = plt.subplots(nrows, ncols, figsize=(PANEL_W * ncols + 0.6, PANEL_H * nrows + 0.9),
                             squeeze=False, sharex=True, sharey=True)
    cmap = plt.get_cmap(CMAP)
    for ax, key in zip(axes.flat, keys):
        levels = sorted(panels[key], key=level_key)
        ax.axvspan(PAPER_BETAS[0] - 0.025, PAPER_BETAS[-1] + 0.025, **PAPER_SHADE, zorder=0)
        for i, lv in enumerate(levels):
            c = cmap(0.15 + 0.7 * i / max(1, len(levels) - 1))
            ax.plot(BETAS, panels[key][lv], color=c, lw=LINE_LW, marker=MARKER, ms=MARKER_SIZE,
                    label=lv.replace("corridor_sine_ambient_", ""))
        ax.set_title(PANEL_TITLES.get(key, " / ".join(key)))
        ax.grid(True, color="0.9", lw=0.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        if Y_LOG:
            ax.set_yscale("log")
        else:
            ax.set_ylim(*Y_LIM)
        ax.set_xticks(BETAS[1::2])
        ax.legend(frameon=False, loc="upper right", ncol=1)
    for ax in axes.flat[n:]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel(X_LABEL)
    for row in axes:
        row[0].set_ylabel(Y_LABEL)
    fig.suptitle(SUPTITLE)
    fig.text(0.5, 0.005, "shaded: the β range the paper reports (0.50 to 0.95)", ha="center",
             fontsize=FONT_SIZE - 1, color="0.35")
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for fmt in FORMATS:
        out = FIG_DIR / f"{OUT_STEM}.{fmt}"
        fig.savefig(out, dpi=DPI)
        os.chmod(out, FILE_MODE)
        print(f"[ok] {out}")
    print(f"[ok] {csv_path}")
    for key in keys:
        print(PANEL_TITLES.get(key, key), {lv: f"{panels[key][lv][BETAS.tolist().index(0.5)]:.3f}"
                                          for lv in sorted(panels[key], key=level_key)}, "(frac at β=0.5)")


if __name__ == "__main__":
    main()
