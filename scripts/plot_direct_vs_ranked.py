"""
Grouped bar chart: Direct vs Ranked acquisition strategy separatrix rates.
Publication-quality wrap figure for IROS.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------- LaTeX-compatible fonts ----------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
})

# ---------- Data ----------
systems = ["Pend.", "CartP.", "PQ-2D", "SQ-3D"]
direct  = np.array([6.00,  0.99,  6.00, 25.22])
ranked  = np.array([11.84, 1.64, 14.00, 24.85])

COLOR_DIRECT = "#2166ac"  # darker blue
COLOR_RANKED = "#e08214"  # darker orange

# ---------- Figure ----------
fig, ax = plt.subplots(figsize=(3.5, 2.2))

x = np.arange(len(systems))
bar_w = 0.35

bars_d = ax.bar(x - bar_w / 2, direct, bar_w,
                color=COLOR_DIRECT, edgecolor="white", linewidth=0.3,
                label="Direct (Ours)", zorder=3)
bars_r = ax.bar(x + bar_w / 2, ranked, bar_w,
                color=COLOR_RANKED, edgecolor="white", linewidth=0.3,
                label="Ranked", zorder=3)

# Value labels on top of bars — offset when bars are close in height
y_max = max(max(direct), max(ranked))
for i in range(len(systems)):
    hd, hr = direct[i], ranked[i]
    # If labels would overlap (heights within 15% of each other), stagger
    if abs(hd - hr) < y_max * 0.08:
        higher, lower = (hd, hr) if hd >= hr else (hr, hd)
        off_hi = y_max * 0.10
        off_lo = y_max * 0.02
        for bar, h in [(bars_d[i], hd), (bars_r[i], hr)]:
            off = off_hi if h == higher else off_lo
            ax.text(bar.get_x() + bar.get_width() / 2, h + off,
                    f"{h:.1f}", ha="center", va="bottom", fontsize=9,
                    fontweight="bold")
    else:
        for bar, h in [(bars_d[i], hd), (bars_r[i], hr)]:
            ax.text(bar.get_x() + bar.get_width() / 2, h + y_max * 0.02,
                    f"{h:.1f}", ha="center", va="bottom", fontsize=9,
                    fontweight="bold")

# Axes
ax.set_xticks(x)
ax.set_xticklabels(systems, fontweight="bold")
ax.set_ylabel("Unc. (%)", fontweight="bold")
ax.set_ylim(0, y_max * 1.22)
ax.set_xlim(-0.5, len(systems) - 0.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", linewidth=0.25, alpha=0.4, zorder=0)

# Legend — compact, top right
ax.legend(frameon=True, fancybox=False, edgecolor="0.8",
          loc="upper left", handlelength=1.2, handletextpad=0.4,
          borderpad=0.3, labelspacing=0.25,
          prop={"weight": "bold"})

fig.subplots_adjust(left=0.18, right=0.98, top=0.96, bottom=0.14)

# ---------- Save ----------
BASE = "/common/home/dm1487/robotics_research/tripods/olympics-classifier"
out_dir = f"{BASE}/results/figures/ablations"
os.makedirs(out_dir, exist_ok=True)
fig.savefig(f"{out_dir}/direct_vs_ranked.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig(f"{out_dir}/direct_vs_ranked.png", bbox_inches="tight", pad_inches=0.02, dpi=300)
print(f"Saved {out_dir}/direct_vs_ranked.pdf and {out_dir}/direct_vs_ranked.png")
