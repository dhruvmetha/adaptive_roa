#!/usr/bin/env python3
"""Plot pendulum ground-truth ROA from roa_labels.txt using TP/TN color scheme."""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# ── Data ─────────────────────────────────────────────────────────────────────
DATA_FILE = Path(
    "/common/users/shared/pracsys/genMoPlan/data_trajectories"
    "/deterministic/pendulum_lqr_50k/roa_labels.txt"
)
OUTPUT_DIR = Path("results/figures")

# ── Color scheme (matches _error_cmap in plot_csv_comparison.py) ─────────────
# TP = yellow (#FDE725), TN = purple (#440154)
TP_COLOR = "#FDE725"
TN_COLOR = "#440154"


def main():
    # Load data: theta, theta_dot, label
    data = np.loadtxt(DATA_FILE, delimiter=",")
    theta = data[:, 0]
    theta_dot = data[:, 1]
    labels = data[:, 2].astype(int)

    print(f"Loaded {len(data)} points")
    print(f"  Success (TP): {np.sum(labels == 1)} ({100 * np.mean(labels == 1):.1f}%)")
    print(f"  Failure (TN): {np.sum(labels == 0)} ({100 * np.mean(labels == 0):.1f}%)")

    # Build grid matching the actual data resolution (158 x 315, step=0.04)
    u_theta = np.sort(np.unique(theta))
    u_tdot = np.sort(np.unique(theta_dot))
    nx, ny = len(u_theta), len(u_tdot)
    print(f"  Grid: {nx} x {ny}")

    # Map each point to its grid cell
    theta_idx = np.searchsorted(u_theta, theta)
    tdot_idx = np.searchsorted(u_tdot, theta_dot)
    grid = np.full((nx, ny), 1, dtype=np.int8)  # default TN
    grid[theta_idx, tdot_idx] = np.where(labels == 1, 0, 1)  # 0=TP, 1=TN

    # Colormap: 0=TP(yellow), 1=TN(purple)
    cmap = mcolors.ListedColormap([TP_COLOR, TN_COLOR])
    norm = mcolors.BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)

    theta_range = (u_theta[0], u_theta[-1])
    theta_dot_range = (u_tdot[0], u_tdot[-1])
    extent = [*theta_range, *theta_dot_range]

    # Bare image, no axes
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(
        grid.T, origin="lower", aspect="auto",
        cmap=cmap, norm=norm, extent=extent, interpolation="nearest",
    )
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        out = OUTPUT_DIR / f"pendulum_gt_roa.{ext}"
        fig.savefig(str(out), dpi=300, bbox_inches="tight", pad_inches=0)
        print(f"Saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
