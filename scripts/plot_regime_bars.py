#!/usr/bin/env python3
"""3×4 grouped bar chart: methods compared at Low / Mid / High data regimes.
Each column is a system; each row is a data regime.
F1 bars are solid; Sep% bars have hatching.
Colors encode methods; x-ticks are short-form names.
Publication-quality for IROS (double-column, ~7.16in wide).
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.patches import Patch

# ── Style ─────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.weight": "normal",
    "mathtext.fontset": "cm",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.labelweight": "bold",
    "axes.titlesize": 9,
    "axes.titleweight": "bold",
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 7,
    "legend.fontsize": 6.5,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
})

BASE = "/common/home/dm1487/robotics_research/tripods/olympics-classifier"

# ── Method registry ───────────────────────────────────────────────────────
# Colors per method (consistent across all panels)
METHOD_COLORS = {
    "Ad": "#1f77b4",  # blue
    "NA": "#d62728",  # red
    "DR": "#7b4f8a",  # purple
    "Cl": "#2ca02c",  # green
    "MO": "#8c564b",  # brown
    "Ly": "#e377c2",  # pink
}

METHOD_LONG = {
    "Ad": "Ours (adaptive)",
    "NA": "Ours (non-adaptive)",
    "DR": "DeepReach",
    "Cl": "Classification",
    "MO": "MORALS",
    "Ly": "Lyapunov NN",
}

# ── Helper: find nearest value in a method's data ────────────────────────
def nearest_val(method_data, target_traj, metric):
    """Return the metric value at the trajectory count nearest to target_traj."""
    traj = np.array(method_data["traj"])
    idx = np.argmin(np.abs(traj - target_traj))
    return method_data[metric][idx], traj[idx]


# ── Data (same as plot_data_efficiency.py) ────────────────────────────────

# Pendulum (2D)
pend_nonad = {
    "traj": [50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
    "f1":   [0.9990, 0.9341, 0.9833, 0.9830, 0.9932, 0.9824, 0.9801, 0.9826, 0.9896, 0.9932],
    "sep":  [73.76, 51.62, 18.72, 11.57, 9.88, 6.34, 5.54, 7.70, 7.11, 8.01],
}
pend_adapt = {
    "traj": [100, 150, 200, 250, 300, 350, 400, 450, 500],
    "f1":   [0.9341, 0.9717, 0.9742, 0.9945, 0.9793, 0.9867, 0.9726, 0.9857, 0.9741],
    "sep":  [51.62, 20.55, 11.74, 9.54, 1.77, 3.92, 2.04, 3.07, 5.95],
}
pend_baselines = {
    "DR": {
        "traj": [50, 250, 500],
        "f1":   [0.937, 0.984, 0.982],
        "sep":  [0, 1.927, 3.631],
    },
    "Cl": {
        "traj": [50, 100, 250, 300, 500],
        "f1":   [0.9299, 0.9476, 0.9694, 0.9714, 0.9744],
        "sep":  [0.08, 0.19, 0.02, 0.34, 0.15],
    },
    "MO": {
        "traj": [250, 500],
        "f1":   [0.753, 0.905],
        "sep":  [38.138, 24.355],
    },
    "Ly": {
        "traj": [50, 250, 500],
        "f1":   [0.758, 0.718, 0.851],
        "sep":  [0, 0, 0],
    },
}

# CartPole (4D)
cart_nonad = {
    "traj": [300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000],
    "f1":   [0.9568, 0.9820, 0.9784, 0.9732, 0.9773, 0.9857, 0.9874, 0.9801, 0.9843,
             0.9824, 0.9815, 0.9862, 0.9905, 0.9874, 0.9839],
    "sep":  [35.71, 20.70, 16.82, 15.02, 14.80, 17.09, 16.15, 13.65, 13.47,
             8.43, 13.13, 12.12, 9.61, 8.38, 7.59],
}
cart_adapt = {
    "traj": [300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000],
    "f1":   [0.9562, 0.9849, 0.9871, 0.9720, 0.9687, 0.9901, 0.9891, 0.9904, 0.9923,
             0.9901, 0.9902, 0.9879, 0.9871, 0.9887, 0.9896],
    "sep":  [35.66, 20.86, 13.11, 10.76, 8.97, 10.24, 6.92, 8.12, 4.87,
             3.48, 2.63, 1.61, 1.37, 1.26, 0.99],
}
cart_baselines = {
    "DR": {
        "traj": [300, 650, 1000],
        "f1":   [0.879, 0.840, 0.901],
        "sep":  [8.696, 10.292, 16.667],
    },
    "Cl": {
        "traj": [300, 650, 1000],
        "f1":   [0.8882, 0.9291, 0.9460],
        "sep":  [0.07, 0.06, 0.15],
    },
    "Ly": {
        "traj": [300, 650, 1000],
        "f1":   [0.958, 0.966, 0.978],
        "sep":  [85.61, 79.67, 71.85],
    },
    "MO": {
        "traj": [300, 650, 1000],
        "f1":   [0.396, 0.221, 0.699],
        "sep":  [10.4, 62.9, 17.9],
    },
}

# Planar Quadrotor (6D)
q2d_nonad = {
    "traj": [3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000],
    "f1":   [0.5873, 0.4782, 0.6285, 0.7506, 0.7188, 0.8231, 0.8105, 0.8694, 0.8849, 0.8547],
    "sep":  [24.45, 22.92, 15.86, 18.78, 13.19, 15.12, 14.95, 18.60, 19.05, 13.21],
}
q2d_adapt = {
    "traj": [3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000],
    "f1":   [0.6608, 0.5931, 0.7104, 0.8708, 0.8517, 0.9130, 0.9154, 0.9211, 0.9422, 0.9329],
    "sep":  [32.05, 15.62, 10.45, 11.10, 7.39, 8.11, 7.76, 7.10, 8.84, 6.00],
}
q2d_baselines = {
    "DR": {
        "traj": [3000, 7500, 12000],
        "f1":   [0.374, 0.361, 0.353],
        "sep":  [0, 0, 0],
    },
    "Cl": {
        "traj": [3000, 7000, 12000],
        "f1":   [0.6088, 0.7313, 0.7767],
        "sep":  [0.32, 0.26, 0.46],
    },
    "MO": {
        "traj": [3000, 7500, 12000],
        "f1":   [0.150, 0.166, 0.292],
        "sep":  [0.3, 4.2, 32.4],
    },
    "Ly": {
        "traj": [3000, 7500, 12000],
        "f1":   [0.232, 0.758, 0.571],
        "sep":  [88.15, 98.07, 97.53],
    },
}

# 3D Quadrotor (13D)
q3d_nonad = {
    "traj": [10000, 11000, 12000, 13000, 14000, 15000, 16000, 17000, 18000, 19000, 20000, 21000, 22000, 23000, 24000],
    "f1":   [0.9391, 0.9309, 0.9363, 0.9376, 0.9499, 0.9382, 0.9533, 0.9368, 0.9475, 0.9546, 0.9525, 0.9529, 0.9511, 0.9511, 0.9470],
    "sep":  [42.89, 31.65, 30.64, 29.85, 33.37, 28.28, 34.05, 25.99, 32.38, 29.73, 29.15, 28.43, 27.43, 26.75, 25.31],
}
q3d_adapt = {
    "traj": [10000, 11000, 12000, 13000, 14000, 15000, 16000, 17000, 18000, 19000, 20000, 21000, 22000, 23000, 24000],
    "f1":   [0.9388, 0.9357, 0.9359, 0.9360, 0.9527, 0.9526, 0.9495, 0.9469, 0.9485, 0.9521, 0.9491, 0.9577, 0.9471, 0.9577, 0.9525],
    "sep":  [42.90, 30.94, 31.79, 27.36, 31.88, 30.94, 29.21, 27.93, 27.61, 27.42, 25.81, 29.41, 23.92, 27.32, 27.44],
}
q3d_baselines = {
    "DR": {
        "traj": [10000, 17500, 25000],
        "f1":   [0.860, 0.861, 0.701],
        "sep":  [68.568, 61.758, 42.769],
    },
    "Cl": {
        "traj": [10000, 17500, 24000],
        "f1":   [0.7622, 0.7667, 0.7995],
        "sep":  [0.83, 0.53, 0.72],
    },
    "MO": {
        "traj": [24000],
        "f1":   [0.194],
        "sep":  [81.8],
    },
    "Ly": {
        "traj": [10000, 17500, 24000],
        "f1":   [0.909, 0.918, 0.921],
        "sep":  [31.60, 31.81, 32.14],
    },
}

# ── Per-system config ─────────────────────────────────────────────────────
# (title, adapt, nonad, baselines_dict, [low_traj, mid_traj, high_traj], method_order)
ALL_METHODS = ["Ad", "NA", "DR", "Cl", "MO", "Ly"]

systems = [
    {
        "title": "Pendulum (2D)",
        "adapt": pend_adapt,
        "nonad": pend_nonad,
        "baselines": pend_baselines,
        "regimes": [100, 300, 500],            # low, mid, high
        "methods": ALL_METHODS,
    },
    {
        "title": "CartPole (4D)",
        "adapt": cart_adapt,
        "nonad": cart_nonad,
        "baselines": cart_baselines,
        "regimes": [300, 650, 1000],
        "methods": ALL_METHODS,
    },
    {
        "title": "Planar Quad (6D)",
        "adapt": q2d_adapt,
        "nonad": q2d_nonad,
        "baselines": q2d_baselines,
        "regimes": [3000, 7000, 12000],
        "methods": ALL_METHODS,
    },
    {
        "title": "3D Quad (13D)",
        "adapt": q3d_adapt,
        "nonad": q3d_nonad,
        "baselines": q3d_baselines,
        "regimes": [10000, 17000, 24000],
        "methods": ALL_METHODS,
    },
]

regime_labels = ["Low data\nregime", "Medium data\nregime", "High data\nregime"]

# ── Build data matrix ─────────────────────────────────────────────────────
def get_method_val(sys_cfg, method_key, target_traj, metric):
    """Get F1 or Sep for a method at a target trajectory count."""
    if method_key == "Ad":
        val, _ = nearest_val(sys_cfg["adapt"], target_traj, metric)
        return val
    elif method_key == "NA":
        val, _ = nearest_val(sys_cfg["nonad"], target_traj, metric)
        return val
    else:
        if method_key in sys_cfg["baselines"]:
            val, actual_traj = nearest_val(sys_cfg["baselines"][method_key], target_traj, metric)
            # Allow small snaps (≤10%), block large ones
            if abs(actual_traj - target_traj) / target_traj > 0.10:
                return None
            return val
    return None


# ── Figure: 3 rows × 4 columns ───────────────────────────────────────────
fig, axes = plt.subplots(3, 4, figsize=(7.16, 5.2))
fig.subplots_adjust(wspace=0.55, hspace=0.38, top=0.94, bottom=0.14,
                    left=0.12, right=0.98)

for col, sys_cfg in enumerate(systems):
    methods = sys_cfg["methods"]
    n_methods = len(methods)

    for row, (regime_label, target_traj) in enumerate(zip(regime_labels, sys_cfg["regimes"])):
        ax = axes[row, col]

        # Gather F1 and Sep values
        f1_vals = []
        sep_vals = []
        colors = []
        labels = []
        for m in methods:
            f1 = get_method_val(sys_cfg, m, target_traj, "f1")
            sep = get_method_val(sys_cfg, m, target_traj, "sep")
            f1_vals.append(f1)
            sep_vals.append(sep)
            colors.append(METHOD_COLORS[m])
            labels.append(m)

        x = np.arange(n_methods)
        bar_w = 0.35

        # Draw bars individually to skip missing data
        ax2 = ax.twinx()
        for i in range(n_methods):
            if f1_vals[i] is not None:
                ax.bar(x[i] - bar_w / 2, f1_vals[i], bar_w,
                       color=colors[i], alpha=0.85,
                       edgecolor="white", linewidth=0.4, zorder=3)
            if sep_vals[i] is not None:
                ax2.bar(x[i] + bar_w / 2, sep_vals[i], bar_w,
                        color=colors[i], alpha=0.40,
                        edgecolor=colors[i], linewidth=0.6,
                        hatch="//", zorder=3)

        # X-axis
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0, ha="center", fontweight="bold")

        # Format trajectory count for subtitle
        if target_traj >= 1000:
            traj_str = f"{target_traj / 1000:.0f}k"
        else:
            traj_str = str(target_traj)

        # Title: system name on top row only, traj count on every row
        if row == 0:
            ax.set_title(f"{sys_cfg['title']}\n({traj_str} traj)",
                         fontsize=8, fontweight="bold", pad=4)
        else:
            ax.set_title(f"({traj_str} traj)", fontsize=7.5, fontweight="bold", pad=3)

        # Row label on left-most column
        if col == 0:
            ax.set_ylabel(f"{regime_label}\n\nF1", fontsize=6.5, fontweight="bold", color="0.3")
        else:
            ax.set_ylabel("", fontsize=7)

        # F1 y-axis (left)
        ax.set_ylim(0, 1.12)
        ax.tick_params(axis="y", labelsize=6, colors="0.3")

        # Sep y-axis (right) — scale to data
        valid_sep = [s for s in sep_vals if s is not None]
        sep_max = max(valid_sep) * 1.3 if valid_sep and max(valid_sep) > 0 else 10
        ax2.set_ylim(0, sep_max)
        ax2.set_ylabel("Sep%" if col == 3 else "", fontsize=7, fontweight="bold", color="0.5")
        ax2.tick_params(axis="y", labelsize=5.5, colors="0.5")

        # Clean up
        ax.spines["top"].set_visible(False)
        ax2.spines["top"].set_visible(False)


# ── Legend ─────────────────────────────────────────────────────────────────
# Build legend with color, short name, long name
# Two rows: methods + bar type indicator
legend_handles = []
for short, long in METHOD_LONG.items():
    legend_handles.append(
        Patch(facecolor=METHOD_COLORS[short], alpha=0.85,
              edgecolor="white", linewidth=0.4,
              label=f"{short} = {long}")
    )

# Add bar-type indicators
legend_handles.append(
    Patch(facecolor="0.6", alpha=0.85, edgecolor="white", linewidth=0.4,
          label=r"Solid = F1 $\uparrow$")
)
legend_handles.append(
    Patch(facecolor="0.6", alpha=0.40, edgecolor="0.4", linewidth=0.6,
          hatch="//", label=r"Hatched = Sep% $\downarrow$")
)

fig.legend(handles=legend_handles, loc="lower center",
           ncol=4, frameon=True, fancybox=False, edgecolor="0.8",
           fontsize=6, bbox_to_anchor=(0.52, 0.0),
           columnspacing=1.0, handlelength=1.5, handletextpad=0.4,
           prop={"weight": "bold"})

# ── Save ──────────────────────────────────────────────────────────────────
out_dir = f"{BASE}/results/figures"
os.makedirs(out_dir, exist_ok=True)
for ext in ("pdf", "png"):
    p = f"{out_dir}/regime_comparison.{ext}"
    fig.savefig(p, format=ext, dpi=300, bbox_inches="tight", pad_inches=0.03)
    print(f"Saved {p}")
plt.close(fig)
