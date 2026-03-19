#!/usr/bin/env python3
"""Debug plot: Pendulum runs starting @50 traj only. Manifold versions marked with thicker lines + star markers."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Only runs starting at 50 trajectories ────────────────────────────────

runs = {
    "Non-adaptive (direct, d2=0.0)": {
        "traj": [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000],
        "f1":   [0.9990, 0.9341, 0.9833, 0.9830, 0.9932, 0.9824, 0.9801, 0.9826, 0.9896, 0.9932, 0.9882, 0.9869, 0.9922, 0.9877, 0.9629, 0.9884, 0.9921, 0.9880, 0.9947, 0.9941],
        "sep":  [73.76, 51.62, 18.72, 11.57, 9.88, 6.34, 5.54, 7.70, 7.11, 8.01, 3.53, 4.66, 57.64, 23.83, 33.66, 2.23, 48.16, 19.43, 32.06, 15.54],
        "color": "#d62728", "ls": "-", "marker": "o", "lw": 1.2, "ms": 4, "manifold": False,
    },
    "Adaptive (direct, d2=1.0)": {
        "traj": [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 792],
        "f1":   [0.9971, 0.9990, 0.9749, 0.9582, 0.9387, 0.9384, 0.9689, 0.9758, 0.9751, 0.9695, 0.9834, 0.9725, 0.9887, 0.9952, 0.9965, 0.9960],
        "sep":  [73.86, 66.79, 24.91, 8.95, 8.92, 13.47, 8.22, 10.40, 3.68, 10.69, 10.15, 10.69, 15.49, 18.48, 13.26, 14.87],
        "color": "#1f77b4", "ls": "-", "marker": "o", "lw": 1.2, "ms": 4, "manifold": False,
    },
    "Adaptive (direct, d2=0.75)": {
        "traj": [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 998],
        "f1":   [0.9992, 0.9961, 0.9863, 0.9886, 0.9913, 0.9717, 0.9767, 0.9797, 0.9953, 0.9849, 0.9898, 0.9977, 0.9957, 0.9902, 0.9974, 0.8994, 0.9952, 0.9952, 0.9946, 0.8254],
        "sep":  [75.40, 50.71, 23.21, 11.13, 5.36, 3.58, 1.61, 1.46, 2.14, 1.33, 2.02, 61.31, 34.70, 5.85, 3.88, 34.67, 1.65, 38.61, 1.75, 24.18],
        "color": "#9467bd", "ls": "-", "marker": "D", "lw": 1.2, "ms": 4, "manifold": False,
    },
    "Adaptive (direct, d2=0.5)": {
        "traj": [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000],
        "f1":   [0.9969, 0.9912, 0.9771, 0.9863, 0.9800, 0.9906, 0.9809, 0.9872, 0.9925, 0.9902, 0.9914, 0.9911, 0.9930, 0.9936, 0.9929, 0.9936, 0.9504, 0.9956, 0.9952, 0.9985],
        "sep":  [75.26, 44.80, 25.83, 11.23, 8.27, 12.20, 6.16, 3.41, 2.45, 1.57, 0.96, 1.41, 0.63, 1.90, 1.71, 2.69, 35.74, 19.74, 2.60, 8.61],
        "color": "#8c564b", "ls": "-", "marker": "v", "lw": 1.2, "ms": 4, "manifold": False,
    },
    "\u2605 Manifold (direct, d2=0.5, opt=loss)": {
        "traj": [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000],
        "f1":   [0.9998, 0.9902, 0.9705, 0.9749, 0.9897, 0.9904, 0.9907, 0.9870, 0.9914, 0.9935, 0.9936, 0.9944, 0.9937, 0.9915, 0.9958, 0.9940, 0.8505, 0.9863, 0.9979, 0.9974],
        "sep":  [73.72, 32.40, 11.73, 7.23, 10.14, 11.57, 4.46, 2.53, 2.24, 1.44, 2.75, 1.23, 2.37, 0.53, 0.38, 0.44, 49.40, 8.57, 16.17, 1.18],
        "color": "#17becf", "ls": "-", "marker": "*", "lw": 2.5, "ms": 8, "manifold": True,
    },
}

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
fig.suptitle("Pendulum — Runs @50 traj (debug)", fontsize=14, fontweight="bold")

# Plot non-manifold first, then manifold on top
for name, d in sorted(runs.items(), key=lambda x: x[1]["manifold"]):
    ax1.plot(d["traj"], d["f1"], linestyle=d["ls"], marker=d["marker"],
             color=d["color"], markersize=d["ms"], linewidth=d["lw"],
             label=name, alpha=0.9 if d["manifold"] else 0.7,
             zorder=10 if d["manifold"] else 3)
    ax2.plot(d["traj"], d["sep"], linestyle=d["ls"], marker=d["marker"],
             color=d["color"], markersize=d["ms"], linewidth=d["lw"],
             label=name, alpha=0.9 if d["manifold"] else 0.7,
             zorder=10 if d["manifold"] else 3)

ax1.set_ylabel("F1", fontsize=11)
ax1.set_ylim(0.8, 1.005)
ax1.grid(alpha=0.3)
ax1.legend(fontsize=8, ncol=2, loc="lower right")

ax2.set_ylabel("Sep%", fontsize=11)
ax2.set_xlabel("Trajectories", fontsize=11)
ax2.grid(alpha=0.3)
ax2.legend(fontsize=8, ncol=2, loc="upper right")

plt.tight_layout()
out = "/common/home/dm1487/robotics_research/tripods/olympics-classifier/results/figures/debug_pendulum_all.png"
plt.savefig(out, dpi=150)
print(f"Saved {out}")
