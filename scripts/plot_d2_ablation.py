"""
2×2 bar plots: F1 and Sep% vs uncertain sampling ratio (d2) for all 4 systems.
Publication-quality for IROS (double-column, ~7.16in wide).
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# ---------- LaTeX-compatible fonts (IROS-sized) ----------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 10,
    "axes.linewidth": 0.8,
    "grid.linewidth": 0.5,
    "grid.alpha": 0.35,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
})

# ---------- Data ----------
d2_labels = ["0\n(non-adapt.)", "0.5", "0.75", "1.0"]
n = len(d2_labels)

systems = [
    ("Pendulum (2D)", {
        "f1":  np.array([0.9932, 0.9902, 0.9849, 0.9741]),
        "sep": np.array([8.01,   1.57,   1.33,   5.95]),
    }),
    ("CartPole (4D)", {
        "f1":  np.array([0.9839, 0.9947, 0.9867, 0.9896]),
        "sep": np.array([7.59,   3.93,   1.68,   0.99]),
    }),
    ("Planar Quad (6D)", {
        "f1":  np.array([0.8547, 0.9261, 0.9300, 0.9329]),
        "sep": np.array([13.21,  9.62,   9.36,   6.00]),
    }),
    ("3D Quad (13D)", {
        "f1":  np.array([0.9470, 0.9502, 0.9486, 0.9525]),
        "sep": np.array([25.31,  25.85,  24.27,  25.22]),
    }),
]

COLOR_F1  = "#1f77b4"   # blue
COLOR_SEP = "#d62728"   # red

# Per-system F1 y-limits (ymin, ymax)
f1_ylims = [
    (0.92, 1.02),   # Pendulum — tight range
    (0.92, 1.02),   # CartPole — tight range
    (0.78, 1.02),   # Planar Quad — wider for 0.85
    (0.90, 1.02),   # 3D Quad — tight range
]

# ---------- Figure ----------
fig, all_axes = plt.subplots(2, 2, figsize=(7.16, 5.4))

x = np.arange(n)
bar_w = 0.32

for idx, (name, d) in enumerate(systems):
    row, col = divmod(idx, 2)
    ax = all_axes[row, col]
    ymin_f1, ymax_f1 = f1_ylims[idx]

    # F1 bars (left axis)
    bars_f1 = ax.bar(x - bar_w / 2, d["f1"], bar_w,
                     color=COLOR_F1, alpha=0.85,
                     edgecolor="white", linewidth=0.5, zorder=3)
    ax.set_ylabel(r"F1 $\uparrow$", color=COLOR_F1, fontweight="bold")
    ax.tick_params(axis="y", colors=COLOR_F1)
    ax.set_ylim(ymin_f1, ymax_f1)

    # Sep% bars (right axis)
    ax2 = ax.twinx()
    bars_sep = ax2.bar(x + bar_w / 2, d["sep"], bar_w,
                       color=COLOR_SEP, alpha=0.75,
                       edgecolor="white", linewidth=0.5, zorder=3)
    ax2.set_ylabel(r"Sep. Rate (%) $\downarrow$", color=COLOR_SEP, fontweight="bold")
    ax2.tick_params(axis="y", colors=COLOR_SEP)
    sep_max = max(d["sep"]) * 1.30
    ax2.set_ylim(0, sep_max)

    # Value labels on bars
    for bar in bars_f1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + (ymax_f1 - ymin_f1) * 0.02,
                f"{h:.2f}".lstrip("0"), ha="center", va="bottom", fontsize=7.5,
                color=COLOR_F1, fontweight="bold")
    for bar in bars_sep:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h + sep_max * 0.02,
                 f"{h:.1f}", ha="center", va="bottom", fontsize=7.5,
                 color=COLOR_SEP, fontweight="bold")

    # Shared x-axis
    ax.set_xticks(x)
    ax.set_xticklabels(d2_labels)
    if row == 1:
        ax.set_xlabel("Uncertain Sampling Ratio", fontweight="bold")
    ax.set_title(name, fontweight="bold", pad=6)

    # Light grid (left axis only)
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", linestyle="--", zorder=0)
    ax2.grid(False)

    # Highlight baseline bar with hatching
    bars_f1[0].set_hatch("//")
    bars_sep[0].set_hatch("//")

# Combined legend
legend_handles = [
    Patch(facecolor=COLOR_F1, alpha=0.85, edgecolor="white", linewidth=0.5),
    Patch(facecolor=COLOR_SEP, alpha=0.75, edgecolor="white", linewidth=0.5),
    Patch(facecolor="0.65", alpha=0.85, edgecolor="white", linewidth=0.5, hatch="//"),
]
legend_labels = [r"F1 $\uparrow$", r"Sep. Rate (%) $\downarrow$", "Non-adaptive baseline"]
fig.legend(legend_handles, legend_labels, loc="upper center", ncol=3,
           frameon=True, fancybox=False, edgecolor="0.7",
           bbox_to_anchor=(0.5, 1.01), fontsize=10)

fig.subplots_adjust(left=0.08, right=0.92, top=0.88, bottom=0.10,
                    wspace=0.50, hspace=0.40)

# ---------- Save ----------
BASE = "/common/home/dm1487/robotics_research/tripods/olympics-classifier"
out_dir = f"{BASE}/results/figures"
os.makedirs(out_dir, exist_ok=True)
out_stem = f"{out_dir}/d2_ablation"
fig.savefig(f"{out_stem}.pdf", bbox_inches="tight", pad_inches=0.03)
fig.savefig(f"{out_stem}.png", bbox_inches="tight", pad_inches=0.03, dpi=300)
print(f"Saved {out_stem}.pdf and {out_stem}.png")
