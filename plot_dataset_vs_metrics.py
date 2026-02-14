"""Plot dataset size vs F1 and dataset size vs separatrix % for quadrotor2d experiments."""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams.update({
    'font.size': 13,
    'axes.labelsize': 14,
    'axes.titlesize': 15,
    'legend.fontsize': 11,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.dpi': 150,
})

# === Data ===
# Directory 1: 20 adaptive iterations, starting with 2000 trajectories
# Full ROA eval at epochs 0, 5, 10
adapt_20_sizes = [2000, 4500, 7000]
adapt_20_f1 = [0.5207, 0.6761, 0.7535]
adapt_20_sep = [14.70, 11.98, 10.11]

# Directory 2: 1 adaptive iteration, 10000 trajectories
adapt_1_size = 10000
adapt_1_f1 = 0.7801
adapt_1_sep = 8.98

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# --- F1 Score plot ---
ax1.plot(adapt_20_sizes, adapt_20_f1, 'o-', color='#2176AE', linewidth=2.2,
         markersize=8, label='Adaptive (20 iter)', zorder=5)

# Iter 1 as a horizontal line spanning the plot
ax1.axhline(y=adapt_1_f1, color='#D7263D', linewidth=2, linestyle='--',
            label=f'Single training (10,000 trajs)', zorder=4)
ax1.plot(adapt_1_size, adapt_1_f1, 's', color='#D7263D', markersize=10, zorder=6)

ax1.set_xlabel('Dataset Size (# training trajectories)')
ax1.set_ylabel('F1 Score')
ax1.set_title('Dataset Size vs F1 Score')
ax1.legend(loc='lower right')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(1000, 11000)
ax1.set_ylim(0.45, 0.85)

# --- Separatrix % plot ---
ax2.plot(adapt_20_sizes, adapt_20_sep, 'o-', color='#2176AE', linewidth=2.2,
         markersize=8, label='Adaptive (20 iter)', zorder=5)

ax2.axhline(y=adapt_1_sep, color='#D7263D', linewidth=2, linestyle='--',
            label=f'Single training (10,000 trajs)', zorder=4)
ax2.plot(adapt_1_size, adapt_1_sep, 's', color='#D7263D', markersize=10, zorder=6)

ax2.set_xlabel('Dataset Size (# training trajectories)')
ax2.set_ylabel('Separatrix %')
ax2.set_title('Dataset Size vs Separatrix %')
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(1000, 11000)
ax2.set_ylim(7, 16)

plt.tight_layout()
plt.savefig('dataset_vs_metrics_quadrotor2d.pdf', bbox_inches='tight')
plt.savefig('dataset_vs_metrics_quadrotor2d.png', bbox_inches='tight')
print("Saved: dataset_vs_metrics_quadrotor2d.pdf and .png")
