#!/usr/bin/env python3
"""Compact 2×3 data efficiency figure: CartPole, Quad2D, Quad3D.
Row 1: F1 score. Row 2: Separatrix %.
Baselines shown as horizontal lines (best value)."""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": 8,
    "axes.labelsize": 10,
    "axes.titlesize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
})

C_ADAPT = "#1f77b4"
C_NONAD = "#d62728"
BSTY = {
    "DeepReach":      {"color": "#7b4f8a", "ls": ":",               },
    "Classification": {"color": "#2ca02c", "ls": (0, (4, 2)),       },
    "MORALS":         {"color": "#8c564b", "ls": (0, (1, 1)),       },
    "Lyapunov":       {"color": "#e377c2", "ls": (0, (3, 1, 1, 1)), },
}
BORDER = ["DeepReach", "Classification", "MORALS", "Lyapunov"]
BLABEL = {"DeepReach": "DeepReach", "Classification": "Classification",
          "MORALS": "MORALS", "Lyapunov": "Lyapunov NN"}

# ── Data ─────────────────────────────────────────────────────────────────

cart_nonad = {
    "traj": [300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000],
    "f1":   [0.9568, 0.9820, 0.9784, 0.9732, 0.9773, 0.9857, 0.9874, 0.9801, 0.9843,
             0.9824, 0.9815, 0.9862, 0.9905, 0.9874, 0.9839],
    "sep":  [35.71, 20.70, 16.82, 15.02, 14.80, 17.09, 16.15, 13.65, 13.47,
             8.43, 13.13, 12.12, 9.61, 8.38, 7.59],
}
cart_adapt = {
    "traj": [300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000],
    "f1":   [0.9568, 0.9849, 0.9871, 0.9720, 0.9687, 0.9901, 0.9891, 0.9904, 0.9923,
             0.9901, 0.9902, 0.9879, 0.9871, 0.9887, 0.9896],
    "sep":  [35.71, 20.86, 13.11, 10.76, 8.97, 10.24, 6.92, 8.12, 4.87,
             3.48, 2.63, 1.61, 1.37, 1.26, 0.99],
}
cart_baselines = {
    "Classification": {"f1": 0.9460, "sep": 0.15},
    "DeepReach":      {"f1": 0.901,  "sep": 16.667},
    "Lyapunov":       {"f1": 0.978,  "sep": 71.85},
    "MORALS":         {"f1": 0.699,  "sep": 17.9},
}

q2d_nonad = {
    "traj": [3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000],
    "f1":   [0.5873, 0.4782, 0.6285, 0.7506, 0.7188, 0.8231, 0.8105, 0.8694, 0.8849, 0.8547],
    "sep":  [24.45, 22.92, 15.86, 18.78, 13.19, 15.12, 14.95, 18.60, 19.05, 13.21],
}
q2d_adapt = {
    "traj": [3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000],
    "f1":   [0.5873, 0.5931, 0.7104, 0.8708, 0.8517, 0.9130, 0.9154, 0.9211, 0.9422, 0.9329],
    "sep":  [24.45, 15.62, 10.45, 11.10, 7.39, 8.11, 7.76, 7.10, 8.84, 6.00],
}
q2d_baselines = {
    "Classification": {"f1": 0.7767, "sep": 0.46},
    "DeepReach":      {"f1": 0.353,  "sep": 0},
    "MORALS":         {"f1": 0.292,  "sep": 32.4},
    "Lyapunov":       {"f1": 0.571,  "sep": 97.53},
}

q3d_nonad = {
    "traj": [10000, 11000, 12000, 13000, 14000, 15000, 16000, 17000, 18000, 19000, 20000, 21000, 22000, 23000, 24000],
    "f1":   [0.9391, 0.9309, 0.9363, 0.9376, 0.9499, 0.9382, 0.9533, 0.9368, 0.9475, 0.9546, 0.9525, 0.9529, 0.9511, 0.9511, 0.9470],
    "sep":  [42.89, 31.65, 30.64, 29.85, 33.37, 28.28, 34.05, 25.99, 32.38, 29.73, 29.15, 28.43, 27.43, 26.75, 25.31],
}
q3d_adapt = {
    "traj": [10000, 11000, 12000, 13000, 14000, 15000, 16000, 17000, 18000, 19000, 20000, 21000, 22000, 23000, 24000],
    "f1":   [0.9391, 0.9357, 0.9359, 0.9360, 0.9527, 0.9526, 0.9495, 0.9469, 0.9485, 0.9521, 0.9491, 0.9577, 0.9471, 0.9577, 0.9525],
    "sep":  [42.90, 30.94, 31.79, 27.36, 31.88, 30.94, 29.21, 27.93, 27.61, 27.42, 25.81, 29.41, 23.92, 27.32, 27.44],
}
q3d_baselines = {
    "DeepReach":      {"f1": 0.701,  "sep": 42.769},
    "Classification": {"f1": 0.7995, "sep": 0.72},
    "MORALS":         {"f1": 0.194,  "sep": 81.8},
    "Lyapunov":       {"f1": 0.921,  "sep": 32.14},
}

panels = [
    ("CartPole (4D)",    cart_nonad, cart_adapt, cart_baselines),
    ("Planar Quad (6D)", q2d_nonad,  q2d_adapt,  q2d_baselines),
    ("Spatial Quad (13D)",    q3d_nonad,  q3d_adapt,  q3d_baselines),
]

# Per-panel y-limits: (ymin, ymax)
f1_ylims = [(0.15, 1.02), (0.10, 0.96), (0.15, 0.97)]
sep_ylims = [(-2, 90),    (-2, 102),    (-2, 90)]

# ── Figure: 2×3 ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(5.0, 2.8))
fig.subplots_adjust(wspace=0.10, hspace=0.15, top=0.93, bottom=0.22,
                    left=0.08, right=0.97)

def k_fmt(x, _):
    return f"{int(x/1000)}k" if x >= 1000 else f"{int(x)}"

for col, (title, nonad, adapt, baselines) in enumerate(panels):
    for row, key in enumerate(("f1", "sep")):
        ax = axes[row, col]

        # Our methods as trend lines
        ax.plot(adapt["traj"], adapt[key], "-o", color=C_ADAPT,
                markersize=2.5, linewidth=1.3, markeredgewidth=0, zorder=5)
        ax.plot(nonad["traj"], nonad[key], "-s", color=C_NONAD,
                markersize=2.5, linewidth=1.3, markeredgewidth=0, zorder=5)

        # Baselines as horizontal lines
        for bname in BORDER:
            if bname not in baselines:
                continue
            val = baselines[bname][key]
            s = BSTY[bname]
            ax.axhline(y=val, linestyle=s["ls"], color=s["color"],
                       linewidth=1.1, zorder=2)

        # Titles & labels
        if row == 0:
            ax.set_title(title, fontweight="bold", pad=4)
            ax.set_ylim(*f1_ylims[col])
        else:
            ax.set_ylim(*sep_ylims[col])

        if col == 0:
            ax.set_ylabel("F1 Score" if row == 0 else "Unc. (%)",
                          fontweight="bold")
        else:
            ax.set_yticklabels([])
            ax.tick_params(axis="y", left=False)

        # Explicit x-limits with margin
        tmin_val = min(adapt["traj"])
        tmax_val = max(adapt["traj"])
        margin = (tmax_val - tmin_val) * 0.04
        ax.set_xlim(tmin_val - margin, tmax_val + margin)

        # Hide x-tick labels on top row; show xlabel on bottom row center col
        if row == 0:
            ax.tick_params(axis="x", labelbottom=False, bottom=False)
        elif col == 1:
            ax.set_xlabel("Trajectories", fontweight="bold")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linewidth=0.25, alpha=0.4, zorder=0)

        if tmax_val > 1000:
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(k_fmt))
            trange = tmax_val - tmin_val
            if trange > 10000:
                ax.xaxis.set_major_locator(mticker.MultipleLocator(5000))
            elif trange > 5000:
                ax.xaxis.set_major_locator(mticker.MultipleLocator(2000))

# ── Legend ────────────────────────────────────────────────────────────────
handles = [
    Line2D([], [], color=C_ADAPT, marker="o", markersize=3.5, markeredgewidth=0,
           linewidth=1.3, label="Adaptive (Ours)"),
    Line2D([], [], color=C_NONAD, marker="s", markersize=3.5, markeredgewidth=0,
           linewidth=1.3, label="Non-adaptive (Ours)"),
]
for bname in BORDER:
    s = BSTY[bname]
    handles.append(Line2D([], [], color=s["color"], linestyle=s["ls"],
                          linewidth=1.1, label=BLABEL[bname]))

fig.legend(handles=handles, loc="lower center", ncol=3, frameon=True,
           fancybox=False, edgecolor="0.8", fontsize=6.5,
           bbox_to_anchor=(0.53, -0.06), columnspacing=1.2,
           handlelength=2.0, handletextpad=0.4,
           prop={"weight": "bold"})

# ── Save ─────────────────────────────────────────────────────────────────
BASE = "/common/home/dm1487/robotics_research/tripods/olympics-classifier"
out_dir = f"{BASE}/results/figures"
os.makedirs(out_dir, exist_ok=True)
for ext in ("pdf", "png"):
    p = f"{out_dir}/data_efficiency_compact.{ext}"
    fig.savefig(p, format=ext, dpi=300, bbox_inches="tight", pad_inches=0.03)
    print(f"Saved {p}")
plt.close(fig)
