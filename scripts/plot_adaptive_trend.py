#!/usr/bin/env python3
"""4×4 trend plot: Adaptive vs Non-adaptive across all systems.
Row 1: F1 score; Row 2: Sep% (total); Row 3: Invalid%; Row 4: Uncertain%.
Publication-quality for IROS (double-column, ~7.16in wide).
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── Style ─────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
})

BASE = "/common/home/dm1487/robotics_research/tripods/olympics-classifier"

COLOR_ADAPT = "#1f77b4"   # blue
COLOR_NONAD = "#d62728"   # red

# ── Data ──────────────────────────────────────────────────────────────────

# Pendulum (2D)
pend_nonad = {
    "traj": [100, 150, 200, 250, 300, 350, 400, 450, 500],
    "f1":   [0.9341, 0.9833, 0.9830, 0.9932, 0.9824, 0.9801, 0.9826, 0.9896, 0.9932],
    "sep":  [51.62, 18.72, 11.57, 9.88, 6.34, 5.54, 7.70, 7.11, 8.01],
    "invalid":   [51.56, 18.17, 11.21, 9.65, 5.99, 5.08, 6.15, 5.85, 3.10],
    "uncertain": [0.07, 0.55, 0.37, 0.24, 0.35, 0.46, 1.55, 1.26, 4.91],
}
pend_adapt = {
    "traj": [100, 150, 200, 250, 300, 350, 400, 450, 500],
    "f1":   [0.9341, 0.9717, 0.9742, 0.9945, 0.9793, 0.9867, 0.9726, 0.9857, 0.9741],
    "sep":  [51.62, 20.55, 11.74, 9.54, 1.77, 3.92, 2.04, 3.07, 5.95],
    "invalid":   [51.56, 19.11, 9.29, 6.61, 1.14, 3.27, 1.25, 2.48, 5.44],
    "uncertain": [0.07, 1.44, 2.45, 2.93, 0.63, 0.65, 0.78, 0.58, 0.51],
}

# CartPole (4D)
cart_nonad = {
    "traj": [300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000],
    "f1":   [0.9568, 0.9820, 0.9784, 0.9732, 0.9773, 0.9857, 0.9874, 0.9801, 0.9843,
             0.9824, 0.9815, 0.9862, 0.9905, 0.9874, 0.9839],
    "sep":  [35.71, 20.70, 16.82, 15.02, 14.80, 17.09, 16.15, 13.65, 13.47,
             8.43, 13.13, 12.12, 9.61, 8.38, 7.59],
    "invalid":   [35.60, 18.96, 14.87, 10.80, 8.12, 13.10, 13.51, 11.36, 9.20,
                  6.87, 11.09, 6.12, 4.75, 3.38, 2.08],
    "uncertain": [0.11, 1.74, 1.94, 4.23, 6.68, 3.99, 2.64, 2.29, 4.27,
                  1.55, 2.03, 6.00, 4.86, 5.00, 5.51],
}
cart_adapt = {
    "traj": [300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000],
    "f1":   [0.9568, 0.9849, 0.9871, 0.9720, 0.9687, 0.9901, 0.9891, 0.9904, 0.9923,
             0.9901, 0.9902, 0.9879, 0.9871, 0.9887, 0.9896],
    "sep":  [35.71, 20.86, 13.11, 10.76, 8.97, 10.24, 6.92, 8.12, 4.87,
             3.48, 2.63, 1.61, 1.37, 1.26, 0.99],
    "invalid":   [35.60, 18.93, 10.81, 2.73, 0.73, 4.82, 2.08, 3.08, 2.09,
                  0.66, 0.44, 0.57, 0.20, 0.39, 0.23],
    "uncertain": [0.11, 1.93, 2.30, 8.03, 8.24, 5.42, 4.84, 5.05, 2.78,
                  2.83, 2.19, 1.04, 1.18, 0.87, 0.76],
}

# Planar Quadrotor (6D)
q2d_nonad = {
    "traj": [3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000],
    "f1":   [0.5873, 0.4782, 0.6285, 0.7506, 0.7188, 0.8231, 0.8105, 0.8694, 0.8849, 0.8547],
    "sep":  [24.45, 22.92, 15.86, 18.78, 13.19, 15.12, 14.95, 18.60, 19.05, 13.21],
    "invalid":   [20.38, 18.39, 11.67, 14.73, 8.76, 11.22, 10.78, 15.19, 15.45, 9.25],
    "uncertain": [4.08, 4.53, 4.18, 4.05, 4.42, 3.90, 4.18, 3.41, 3.60, 3.96],
}
q2d_adapt = {
    "traj": [3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000],
    "f1":   [0.5873, 0.5931, 0.7104, 0.8708, 0.8517, 0.9130, 0.9154, 0.9211, 0.9422, 0.9329],
    "sep":  [24.45, 15.62, 10.45, 11.10, 7.39, 8.11, 7.76, 7.10, 8.84, 6.00],
    "invalid":   [20.38, 10.44, 4.75, 6.84, 2.60, 4.59, 3.68, 3.34, 5.04, 3.01],
    "uncertain": [4.08, 5.18, 5.69, 4.26, 4.79, 3.52, 4.08, 3.77, 3.80, 2.99],
}

# 3D Quadrotor (13D)
q3d_nonad = {
    "traj": [10000, 11000, 12000, 13000, 14000, 15000, 16000, 17000, 18000, 19000, 20000, 21000, 22000, 23000, 24000],
    "f1":   [0.9391, 0.9309, 0.9363, 0.9376, 0.9499, 0.9382, 0.9533, 0.9368, 0.9475, 0.9546, 0.9525, 0.9529, 0.9511, 0.9511, 0.9470],
    "sep":  [42.89, 31.65, 30.64, 29.85, 33.37, 28.28, 34.05, 25.99, 32.38, 29.73, 29.15, 28.43, 27.43, 26.75, 25.31],
    "invalid":   [39.74, 25.12, 21.31, 20.99, 24.80, 19.63, 28.63, 18.31, 28.36, 21.54, 21.05, 20.64, 19.50, 18.62, 18.45],
    "uncertain": [3.14, 6.53, 9.33, 8.86, 8.58, 8.66, 5.43, 7.68, 4.01, 8.19, 8.11, 7.79, 7.93, 8.13, 6.86],
}
q3d_adapt = {
    "traj": [10000, 11000, 12000, 13000, 14000, 15000, 16000, 17000, 18000, 19000, 20000, 21000, 22000, 23000, 24000, 25000],
    "f1":   [0.9391, 0.9357, 0.9359, 0.9360, 0.9527, 0.9526, 0.9495, 0.9469, 0.9485, 0.9521, 0.9491, 0.9577, 0.9471, 0.9577, 0.9576, 0.9525],
    "sep":  [42.90, 30.94, 31.79, 27.36, 31.88, 30.94, 29.21, 27.93, 27.61, 27.42, 25.81, 29.41, 23.92, 27.32, 27.44, 25.22],
    "invalid":   [39.74, 21.67, 24.71, 18.68, 23.35, 23.35, 21.45, 20.90, 19.92, 19.16, 17.53, 23.41, 16.04, 21.97, 21.12, 20.16],
    "uncertain": [3.14, 9.26, 7.08, 8.68, 8.53, 7.59, 7.77, 7.03, 7.68, 8.26, 8.28, 5.99, 7.88, 5.35, 6.32, 5.06],
}

# ── System configs ────────────────────────────────────────────────────────
systems = [
    {"title": "Pendulum (2D)",     "adapt": pend_adapt, "nonad": pend_nonad},
    {"title": "CartPole (4D)",     "adapt": cart_adapt, "nonad": cart_nonad},
    {"title": "Planar Quad (6D)",  "adapt": q2d_adapt,  "nonad": q2d_nonad},
    {"title": "Spatial Quad (13D)",     "adapt": q3d_adapt,  "nonad": q3d_nonad},
]

# Row configs: (data_key, ylabel)
rows = [
    ("f1",        r"F1 $\uparrow$"),
    ("sep",       r"Unc% $\downarrow$"),
    ("invalid",   r"ME% $\downarrow$"),
    ("uncertain", r"OU% $\downarrow$"),
]

# ── Figure: 4 rows × 4 columns ───────────────────────────────────────────
fig, axes = plt.subplots(4, 4, figsize=(7.16, 5.0))
fig.subplots_adjust(wspace=0.12, hspace=0.15, top=0.95, bottom=0.14,
                    left=0.075, right=0.97)

def k_formatter(x, pos):
    return f"{int(x/1000)}k" if x >= 1000 else f"{int(x)}"

for col, sys_cfg in enumerate(systems):
    adapt_traj = np.array(sys_cfg["adapt"]["traj"], dtype=float)
    nonad_traj = np.array(sys_cfg["nonad"]["traj"], dtype=float)
    use_k = max(adapt_traj[-1], nonad_traj[-1]) > 1000

    for row_idx, (key, ylabel) in enumerate(rows):
        ax = axes[row_idx, col]
        adapt_vals = np.array(sys_cfg["adapt"][key])
        nonad_vals = np.array(sys_cfg["nonad"][key])

        ax.plot(adapt_traj, adapt_vals, "-o", color=COLOR_ADAPT, markersize=2.5,
                linewidth=1.2, label="Adaptive (Ours)", zorder=3)
        ax.plot(nonad_traj, nonad_vals, "-s", color=COLOR_NONAD, markersize=2.5,
                linewidth=1.2, label="Non-adaptive (Ours)", zorder=3)

        # Title only on top row
        if row_idx == 0:
            ax.set_title(sys_cfg["title"], fontweight="bold", pad=4)

        # Y-axis label only on leftmost column
        if col == 0:
            ax.set_ylabel(ylabel, fontweight="bold")

        # Y-limits
        if key == "f1":
            if col in (0, 1, 3):  # Pendulum, CartPole, 3D Quad — zoom
                f1_min = min(min(adapt_vals), min(nonad_vals))
                ax.set_ylim(max(0, f1_min - 0.02), 1.005)
            else:
                ax.set_ylim(0, 1.05)
        else:
            # For Sep/Invalid/Uncertain: auto with small padding
            all_vals = np.concatenate([adapt_vals, nonad_vals])
            ymax = max(all_vals) * 1.15
            ax.set_ylim(-0.5, max(ymax, 1.0))

        # k-format for large trajectory counts
        if use_k:
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(k_formatter))

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linewidth=0.3, alpha=0.5)

        # Strip redundant ticks: x-labels only on bottom row
        if row_idx < 3:
            ax.set_xticklabels([])
            ax.tick_params(axis="x", bottom=False)

        # Strip redundant ticks: y-labels only on leftmost column
        if col > 0:
            ax.set_yticklabels([])
            ax.tick_params(axis="y", left=False)

# ── Centered x-label ──────────────────────────────────────────────────────
fig.text(0.5, 0.07, "Trajectories", ha="center", fontsize=7, fontweight="bold")

# ── Shared legend ─────────────────────────────────────────────────────────
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=2,
           frameon=True, fancybox=False, edgecolor="0.8",
           fontsize=7, bbox_to_anchor=(0.52, 0.005),
           columnspacing=2.0, handlelength=2.0,
           prop={"weight": "bold"})

# ── Save ──────────────────────────────────────────────────────────────────
out_dir = f"{BASE}/results/figures"
os.makedirs(out_dir, exist_ok=True)
for ext in ("pdf", "png"):
    p = f"{out_dir}/adaptive_trend.{ext}"
    fig.savefig(p, format=ext, dpi=300, bbox_inches="tight", pad_inches=0.03)
    print(f"Saved {p}")
plt.close(fig)

# ══════════════════════════════════════════════════════════════════════════
# Second figure: Adaptive only — Sep%, Invalid%, Uncertain% on one panel
# 2 rows × 2 columns
# ══════════════════════════════════════════════════════════════════════════

COLOR_UNC = "#2ca02c"   # green  — Uncertain% (Unc%)
COLOR_ME  = "#d62728"   # red   — ME% (invalid)
COLOR_OU  = "#1f77b4"   # blue  — OU% (uncertain)

fig2, axes2 = plt.subplots(2, 2, figsize=(3.58, 2.8))
fig2.subplots_adjust(wspace=0.12, hspace=0.55, top=0.93, bottom=0.18,
                     left=0.13, right=0.97)

for idx, sys_cfg in enumerate(systems):
    row, col = divmod(idx, 2)
    ax = axes2[row, col]
    traj = np.array(sys_cfg["adapt"]["traj"])
    x = traj.astype(float)
    use_k = traj[-1] > 1000

    sep = np.array(sys_cfg["adapt"]["sep"])
    inv = np.array(sys_cfg["adapt"]["invalid"])
    unc = np.array(sys_cfg["adapt"]["uncertain"])

    ax.plot(x, sep, "-o", color=COLOR_UNC, markersize=2,
            linewidth=1.0, label="Unc%", zorder=3)
    ax.plot(x, inv, "--^", color=COLOR_ME, markersize=2,
            linewidth=1.0, label="ME%", zorder=3)
    ax.plot(x, unc, ":s", color=COLOR_OU, markersize=2,
            linewidth=1.0, label="OU%", zorder=3)

    ax.set_title(sys_cfg["title"], fontweight="bold", pad=3, fontsize=8)

    # Y-label only on left column
    if col == 0:
        ax.set_ylabel("Rate (%)", fontweight="bold")
    else:
        ax.set_yticklabels([])
        ax.tick_params(axis="y", left=False)

    ymax = max(sep) * 1.15
    ax.set_ylim(-0.5, max(ymax, 1.0))

    if use_k:
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(k_formatter))

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linewidth=0.3, alpha=0.5)

fig2.text(0.5, 0.07, "Trajectories", ha="center", fontsize=7, fontweight="bold")

handles2, labels2 = axes2[0, 0].get_legend_handles_labels()
axes2[0, 1].legend(handles2, labels2, loc="upper right",
                   frameon=True, fancybox=False, edgecolor="0.8",
                   fontsize=6, columnspacing=1.0, handlelength=1.5,
                   prop={"weight": "bold"})

for ext in ("pdf", "png"):
    p = f"{out_dir}/adaptive_only_breakdown.{ext}"
    fig2.savefig(p, format=ext, dpi=300, bbox_inches="tight", pad_inches=0.03)
    print(f"Saved {p}")
plt.close(fig2)
