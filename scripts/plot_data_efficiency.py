#!/usr/bin/env python3
"""Generate publication-quality data efficiency figure (2×4) for IEEE IROS paper."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.lines import Line2D

# ── Style setup ──────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 8,
    'axes.labelsize': 8,
    'axes.titlesize': 9,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 6.5,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.major.size': 2.5,
    'ytick.major.size': 2.5,
    'lines.linewidth': 1.5,
    'lines.markersize': 3,
})

# ── Colors ───────────────────────────────────────────────────────────────
C_ADAPT = '#1f77b4'   # blue
C_NONAD = '#d62728'   # red

BASELINE_STYLES = {
    'DeepReach':      {'color': '#7b4f8a', 'ls': ':',              'lw': 1.0},
    'Classification': {'color': '#2ca02c', 'ls': (0, (4, 2)),      'lw': 1.0},
    'MORALS':         {'color': '#8c564b', 'ls': (0, (1, 1)),      'lw': 1.0},
    'Lyapunov':       {'color': '#e377c2', 'ls': (0, (3, 1, 1, 1)),'lw': 1.0},
}

# ── Data ─────────────────────────────────────────────────────────────────

# Pendulum (2D)
pend_nonad = {
    'traj': [100, 150, 200, 250, 300, 350, 400, 450, 500],
    'f1':   [0.9445, 0.9869, 0.9783, 0.9860, 0.9842, 0.9691, 0.9849, 0.9841, 0.9856],
    'sep':  [28.45, 25.96, 15.75, 6.84, 10.18, 4.34, 6.40, 7.81, 4.28],
}
pend_adapt = {
    'traj': [100, 150, 200, 250, 300, 350, 400, 450, 500],
    'f1':   [0.9448, 0.9656, 0.9714, 0.9915, 0.9776, 0.9836, 0.9709, 0.9837, 0.9739],
    'sep':  [28.83, 19.25, 11.91, 9.24, 1.91, 3.86, 1.90, 2.82, 5.95],
}
# Baselines: (name, f1, sep%)
pend_baselines = [
    ('DeepReach',      0.975, 0.0),
    ('Classification', 0.974, 0.15),
    ('MORALS',         0.939, 7.30),
    ('Lyapunov',       0.906, 35.00),
]

# Cartpole (4D)
cart_nonad = {
    'traj': [300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000],
    'f1':   [0.9510, 0.9788, 0.9769, 0.9706, 0.9727, 0.9822, 0.9835, 0.9790, 0.9813,
             0.9811, 0.9793, 0.9836, 0.9900, 0.9870, 0.9823],
    'sep':  [33.96, 20.65, 16.95, 15.23, 15.14, 17.32, 16.30, 14.00, 13.72,
             8.61, 13.28, 12.40, 10.30, 8.87, 8.10],
}
cart_adapt = {
    'traj': [300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000],
    'f1':   [0.9519, 0.9821, 0.9858, 0.9657, 0.9599, 0.9884, 0.9871, 0.9890, 0.9914,
             0.9883, 0.9885, 0.9876, 0.9858, 0.9873, 0.9878],
    'sep':  [33.88, 20.79, 13.25, 11.07, 9.33, 10.93, 7.39, 8.89, 5.15,
             3.56, 2.77, 1.62, 1.44, 1.34, 0.99],
}
cart_baselines = [
    ('Lyapunov',       0.952, 26.80),
    ('Classification', 0.946, 0.15),
    ('DeepReach',      0.799, 0.0),
]

# Planar Quadrotor (6D)
q2d_nonad = {
    'traj': [3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000],
    'f1':   [0.5158, 0.6798, 0.7625, 0.8038, 0.7788, 0.8413, 0.7698, 0.8324, 0.8295, 0.8433],
    'sep':  [10.55, 9.27, 11.69, 10.27, 10.12, 8.76, 9.54, 8.59, 8.24, 8.02],
}
q2d_adapt = {
    'traj': [3000, 4000, 4969, 5709, 6166, 6514, 6777, 6981, 7147, 7294],
    'f1':   [0.5180, 0.7879, 0.8641, 0.8802, 0.8958, 0.8977, 0.9048, 0.9046, 0.9000, 0.9096],
    'sep':  [10.50, 8.14, 8.35, 7.51, 7.91, 7.09, 7.13, 6.55, 7.20, 6.54],
}
q2d_baselines = [
    ('Lyapunov',       0.804, 35.95),
    ('Classification', 0.777, 0.46),
]

# 3D Quadrotor (12D)
q3d_nonad = {
    'traj': [5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000],
    'f1':   [0.8441, 0.9182, 0.8988, 0.9104, 0.8868, 0.9147, 0.8776, 0.9112],
    'sep':  [35.65, 39.84, 31.53, 30.91, 32.63, 27.05, 29.63, 27.38],
}
q3d_adapt = {
    'traj': [5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000],
    'f1':   [0.8445, 0.9206, 0.9355, 0.9172, 0.9101, 0.9212, 0.9318, 0.9493],
    'sep':  [35.64, 38.01, 47.63, 34.91, 24.68, 35.15, 31.39, 37.40],
}
q3d_baselines = [
    ('Lyapunov',       0.935, 36.05),
    ('Classification', 0.763, 0.57),
    ('DeepReach',      0.618, 0.0),
]

panels = [
    ('Pendulum (2D)',    pend_nonad, pend_adapt, pend_baselines),
    ('Cartpole (4D)',    cart_nonad, cart_adapt, cart_baselines),
    ('Planar Quad (6D)', q2d_nonad,  q2d_adapt,  q2d_baselines),
    ('3D Quad (12D)',    q3d_nonad,  q3d_adapt,  q3d_baselines),
]

# Per-panel F1 y-limits: (ymin, ymax, major_tick_spacing)
f1_ylims = [
    (0.90, 1.005, 0.02),   # Pendulum
    (0.79, 1.005, 0.05),   # Cartpole
    (0.50, 0.92,  0.05),   # Quad2D
    (0.60, 0.96,  0.05),   # Quad3D
]

BASE = '/common/home/dm1487/robotics_research/tripods/olympics-classifier'

baseline_order = ['DeepReach', 'Classification', 'MORALS', 'Lyapunov']
baseline_labels = {
    'DeepReach': 'DeepReach',
    'Classification': 'Classification',
    'MORALS': 'MORALS',
    'Lyapunov': 'Lyapunov NN',
}


def get_xlims_and_format(ax, adapt, nonad):
    """Set x-axis limits and tick formatting."""
    xlims = (min(min(adapt['traj']), min(nonad['traj'])),
             max(max(adapt['traj']), max(nonad['traj'])))
    ax.set_xlim(xlims[0] - (xlims[1] - xlims[0]) * 0.03,
                xlims[1] + (xlims[1] - xlims[0]) * 0.03)
    traj_range = xlims[1] - xlims[0]
    if traj_range > 5000:
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f'{x/1000:.0f}k'))
        ax.xaxis.set_major_locator(mticker.MultipleLocator(2000 if traj_range < 8000 else 3000))
    elif traj_range > 500:
        ax.xaxis.set_major_locator(mticker.MultipleLocator(200))


# ── Figure: 2×4, top = F1, bottom = separatrix ──────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(7.16, 3.6))
fig.subplots_adjust(wspace=0.45, hspace=0.45, top=0.92, bottom=0.22,
                    left=0.065, right=0.97)

for idx, (title, nonad, adapt, baselines) in enumerate(panels):
    ax_f1 = axes[0, idx]
    ax_sep = axes[1, idx]
    ymin, ymax, ytick = f1_ylims[idx]

    # ── Top row: F1 ──
    ax_f1.plot(adapt['traj'], adapt['f1'], '-o', color=C_ADAPT,
               markersize=3, markeredgewidth=0, zorder=5)
    ax_f1.plot(nonad['traj'], nonad['f1'], '-s', color=C_NONAD,
               markersize=2.5, markeredgewidth=0, zorder=5)
    for bname, bf1, bsep in baselines:
        sty = BASELINE_STYLES[bname]
        ax_f1.axhline(y=bf1, linestyle=sty['ls'], color=sty['color'],
                       linewidth=sty['lw'], zorder=2)

    ax_f1.set_title(title, fontsize=9, pad=4)
    ax_f1.set_ylim(ymin, ymax)
    ax_f1.yaxis.set_major_locator(mticker.MultipleLocator(ytick))
    ax_f1.yaxis.set_minor_locator(mticker.MultipleLocator(ytick / 2))
    ax_f1.set_ylabel('F1 Score' if idx == 0 else '', fontsize=8)
    ax_f1.set_xticklabels([])
    ax_f1.spines['top'].set_visible(False)
    get_xlims_and_format(ax_f1, adapt, nonad)

    # ── Bottom row: Separatrix ──
    ax_sep.plot(adapt['traj'], adapt['sep'], '-o', color=C_ADAPT,
                markersize=3, markeredgewidth=0, zorder=5)
    ax_sep.plot(nonad['traj'], nonad['sep'], '-s', color=C_NONAD,
                markersize=2.5, markeredgewidth=0, zorder=5)
    for bname, bf1, bsep in baselines:
        sty = BASELINE_STYLES[bname]
        ax_sep.axhline(y=bsep, linestyle=sty['ls'], color=sty['color'],
                        linewidth=sty['lw'], zorder=2)

    ax_sep.set_ylim(0, 50)
    ax_sep.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
    ax_sep.yaxis.set_major_locator(mticker.MultipleLocator(10))
    ax_sep.set_ylabel('Separatrix (%)' if idx == 0 else '', fontsize=8)
    ax_sep.set_xlabel('Trajectories', fontsize=8)
    ax_sep.spines['top'].set_visible(False)
    get_xlims_and_format(ax_sep, adapt, nonad)

# ── Legend ────────────────────────────────────────────────────────────────
method_handles = [
    Line2D([0], [0], color=C_ADAPT, linestyle='-', marker='o', markersize=3,
           markeredgewidth=0, linewidth=1.5, label='Ours (adaptive)'),
    Line2D([0], [0], color=C_NONAD, linestyle='-', marker='s', markersize=2.5,
           markeredgewidth=0, linewidth=1.5, label='Ours (non-adaptive)'),
]
baseline_handles = []
for bname in baseline_order:
    sty = BASELINE_STYLES[bname]
    baseline_handles.append(
        Line2D([0], [0], color=sty['color'], linestyle=sty['ls'],
               linewidth=sty['lw'], label=baseline_labels[bname])
    )

all_handles = method_handles + baseline_handles
fig.legend(handles=all_handles, loc='lower center',
           ncol=6, frameon=False, fontsize=6.5,
           bbox_to_anchor=(0.5, 0.02),
           columnspacing=1.2, handlelength=2.2, handletextpad=0.5)

# ── Save ─────────────────────────────────────────────────────────────────
for ext in ('pdf', 'png'):
    p = f'{BASE}/data_efficiency.{ext}'
    fig.savefig(p, format=ext, dpi=300, bbox_inches='tight', pad_inches=0.03)
    print(f'Saved {p}')
plt.close(fig)
