#!/usr/bin/env python3
"""Generate publication-quality data efficiency figure (2×4) for IEEE IROS paper.
All methods (including baselines) shown as trend lines."""

import os
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

# ── Colors & styles ──────────────────────────────────────────────────────
C_ADAPT = '#1f77b4'   # blue
C_NONAD = '#d62728'   # red

BASELINE_STYLES = {
    'DeepReach':      {'color': '#7b4f8a', 'ls': ':',              'lw': 1.2, 'marker': '^', 'ms': 3},
    'Classification': {'color': '#2ca02c', 'ls': (0, (4, 2)),      'lw': 1.2, 'marker': 'D', 'ms': 2.5},
    'MORALS':         {'color': '#8c564b', 'ls': (0, (1, 1)),      'lw': 1.2, 'marker': 'v', 'ms': 3},
    'Lyapunov':       {'color': '#e377c2', 'ls': (0, (3, 1, 1, 1)),'lw': 1.2, 'marker': 'x', 'ms': 3.5},
}

# ── Data ─────────────────────────────────────────────────────────────────

# Pendulum (2D)
pend_nonad = {
    'traj': [50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
    'f1':   [0.9990, 0.9341, 0.9833, 0.9830, 0.9932, 0.9824, 0.9801, 0.9826, 0.9896, 0.9932],
    'sep':  [73.76, 51.62, 18.72, 11.57, 9.88, 6.34, 5.54, 7.70, 7.11, 8.01],
}
pend_adapt = {
    'traj': [100, 150, 200, 250, 300, 350, 400, 450, 500],
    'f1':   [0.9341, 0.9717, 0.9742, 0.9945, 0.9793, 0.9867, 0.9726, 0.9857, 0.9741],
    'sep':  [51.62, 20.55, 11.74, 9.54, 1.77, 3.92, 2.04, 3.07, 5.95],
}
pend_baselines = {
    'Classification': {
        'traj': [50, 100, 250, 300, 500],
        'f1':   [0.9299, 0.9476, 0.9694, 0.9714, 0.9744],
        'sep':  [0.08, 0.19, 0.02, 0.34, 0.15],
    },
    'DeepReach': {
        'traj': [50, 250, 500],
        'f1':   [0.937, 0.984, 0.982],
        'sep':  [0, 1.927, 3.631],  # Uncertain% used as proxy
    },
    'MORALS': {
        'traj': [250, 500],
        'f1':   [0.753, 0.905],
        'sep':  [38.138, 24.355],
    },
    'Lyapunov': {
        'traj': [50, 250, 500],
        'f1':   [0.758, 0.718, 0.851],
        'sep':  [0, 0, 0],  # Uncertain%
    },
}

# Cartpole (4D)
cart_nonad = {
    'traj': [300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000],
    'f1':   [0.9568, 0.9820, 0.9784, 0.9732, 0.9773, 0.9857, 0.9874, 0.9801, 0.9843,
             0.9824, 0.9815, 0.9862, 0.9905, 0.9874, 0.9839],
    'sep':  [35.71, 20.70, 16.82, 15.02, 14.80, 17.09, 16.15, 13.65, 13.47,
             8.43, 13.13, 12.12, 9.61, 8.38, 7.59],
}
cart_adapt = {
    'traj': [300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000],
    'f1':   [0.9568, 0.9849, 0.9871, 0.9720, 0.9687, 0.9901, 0.9891, 0.9904, 0.9923,
             0.9901, 0.9902, 0.9879, 0.9871, 0.9887, 0.9896],
    'sep':  [35.71, 20.86, 13.11, 10.76, 8.97, 10.24, 6.92, 8.12, 4.87,
             3.48, 2.63, 1.61, 1.37, 1.26, 0.99],
}
cart_baselines = {
    'Classification': {
        'traj': [300, 650, 1000],
        'f1':   [0.8882, 0.9291, 0.9460],
        'sep':  [0.07, 0.06, 0.15],
    },
    'DeepReach': {
        'traj': [300, 650, 1000],
        'f1':   [0.879, 0.840, 0.901],
        'sep':  [8.696, 10.292, 16.667],  # Uncertain%
    },
    'Lyapunov': {
        'traj': [300, 650, 1000],
        'f1':   [0.958, 0.966, 0.978],
        'sep':  [85.61, 79.67, 71.85],  # Uncertain%
    },
    'MORALS': {
        'traj': [300, 650, 1000],
        'f1':   [0.396, 0.221, 0.699],
        'sep':  [10.4, 62.9, 17.9],
    },
}

# Planar Quadrotor (6D)
q2d_nonad = {
    'traj': [3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000],
    'f1':   [0.5873, 0.4782, 0.6285, 0.7506, 0.7188, 0.8231, 0.8105, 0.8694, 0.8849, 0.8547],
    'sep':  [24.45, 22.92, 15.86, 18.78, 13.19, 15.12, 14.95, 18.60, 19.05, 13.21],
}
q2d_adapt = {
    'traj': [3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000],
    'f1':   [0.5873, 0.5931, 0.7104, 0.8708, 0.8517, 0.9130, 0.9154, 0.9211, 0.9422, 0.9329],
    'sep':  [24.45, 15.62, 10.45, 11.10, 7.39, 8.11, 7.76, 7.10, 8.84, 6.00],
}
q2d_baselines = {
    'Classification': {
        'traj': [3000, 7000, 12000],
        'f1':   [0.6088, 0.7313, 0.7767],
        'sep':  [0.32, 0.26, 0.46],
    },
    'DeepReach': {
        'traj': [3000, 7500, 12000],
        'f1':   [0.374, 0.361, 0.353],
        'sep':  [0, 0, 0],  # Uncertain%
    },
    'MORALS': {
        'traj': [3000, 7500, 12000],
        'f1':   [0.150, 0.166, 0.292],
        'sep':  [0.3, 4.2, 32.4],
    },
    'Lyapunov': {
        'traj': [3000, 7500, 12000],
        'f1':   [0.232, 0.758, 0.571],
        'sep':  [88.15, 98.07, 97.53],  # Uncertain%
    },
}

# 3D Quadrotor (13D) — using lambda-delta results
q3d_nonad = {
    'traj': [10000, 11000, 12000, 13000, 14000, 15000, 16000, 17000, 18000, 19000, 20000, 21000, 22000, 23000, 24000],
    'f1':   [0.9391, 0.9309, 0.9363, 0.9376, 0.9499, 0.9382, 0.9533, 0.9368, 0.9475, 0.9546, 0.9525, 0.9529, 0.9511, 0.9511, 0.9470],
    'sep':  [42.89, 31.65, 30.64, 29.85, 33.37, 28.28, 34.05, 25.99, 32.38, 29.73, 29.15, 28.43, 27.43, 26.75, 25.31],
}
q3d_adapt = {
    'traj': [10000, 11000, 12000, 13000, 14000, 15000, 16000, 17000, 18000, 19000, 20000, 21000, 22000, 23000, 24000],
    'f1':   [0.9391, 0.9357, 0.9359, 0.9360, 0.9527, 0.9526, 0.9495, 0.9469, 0.9485, 0.9521, 0.9491, 0.9577, 0.9471, 0.9577, 0.9525],
    'sep':  [42.90, 30.94, 31.79, 27.36, 31.88, 30.94, 29.21, 27.93, 27.61, 27.42, 25.81, 29.41, 23.92, 27.32, 27.44],
}
q3d_baselines = {
    'DeepReach': {
        'traj': [10000, 17500, 25000],
        'f1':   [0.860, 0.861, 0.701],
        'sep':  [68.568, 61.758, 42.769],  # Uncertain%
    },
    'Classification': {
        'traj': [10000, 17500, 24000],
        'f1':   [0.7622, 0.7667, 0.7995],
        'sep':  [0.83, 0.53, 0.72],
    },
    'MORALS': {
        'traj': [24000],
        'f1':   [0.194],
        'sep':  [81.8],
    },
    'Lyapunov': {
        'traj': [10000, 17500, 24000],
        'f1':   [0.909, 0.918, 0.921],
        'sep':  [31.60, 31.81, 32.14],  # Uncertain% (25k values at 24k)
    },
}

panels = [
    ('Pendulum (2D)',    pend_nonad, pend_adapt, pend_baselines),
    ('CartPole (4D)',    cart_nonad, cart_adapt, cart_baselines),
    ('Planar Quad (6D)', q2d_nonad,  q2d_adapt,  q2d_baselines),
    ('3D Quad (13D)',    q3d_nonad,  q3d_adapt,  q3d_baselines),
]

# Per-panel F1 y-limits: (ymin, ymax, major_tick_spacing)
f1_ylims = [
    (0.65, 1.005, 0.1),   # Pendulum — accommodate Lyapunov 0.718, MORALS 0.753
    (0.20, 1.005, 0.1),   # Cartpole — accommodate MORALS low
    (0.10, 0.96,  0.1),   # Quad2D — accommodate MORALS 0.150, DeepReach 0.35
    (0.15, 0.97,  0.1),   # Quad3D — accommodate MORALS 0.194, Classification 0.76
]

sep_ylims = [
    (-3, 80, 20),    # Pendulum
    (-3, 90, 20),    # Cartpole — Lyapunov uncertain% is ~86
    (-2, 102, 20),   # Quad2D — accommodate Lyapunov ~98%
    (-2, 100, 20),   # Quad3D — accommodate MORALS ~82%, DeepReach ~69%
]

# Zoomed F1 y-limits (tighter for Pendulum/CartPole, wider for Quads to fit baselines)
f1_ylims_zoomed = [
    (0.92, 1.005, 0.02),  # Pendulum — zoomed
    (0.87, 1.005, 0.02),  # Cartpole — zoomed
    (0.10, 0.96,  0.1),   # Quad2D — same as full (baselines need room)
    (0.15, 0.97,  0.1),   # Quad3D — same as full (baselines need room)
]

BASE = '/common/home/dm1487/robotics_research/tripods/olympics-classifier'

baseline_order = ['DeepReach', 'Classification', 'MORALS', 'Lyapunov']
baseline_labels = {
    'DeepReach': 'DeepReach',
    'Classification': 'Classification',
    'MORALS': 'MORALS',
    'Lyapunov': 'Lyapunov NN',
}


def get_xlims_and_format(ax, *datasets):
    """Set x-axis limits and tick formatting from all datasets."""
    all_traj = []
    for d in datasets:
        if isinstance(d, dict) and 'traj' in d:
            all_traj.extend(d['traj'])
    if not all_traj:
        return
    xmin, xmax = min(all_traj), max(all_traj)
    margin = (xmax - xmin) * 0.03
    ax.set_xlim(xmin - margin, xmax + margin)
    traj_range = xmax - xmin
    if traj_range > 5000:
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f'{x/1000:.0f}k'))
        ax.xaxis.set_major_locator(mticker.MultipleLocator(2000 if traj_range < 8000 else 3000 if traj_range < 12000 else 5000))
    elif traj_range > 500:
        ax.xaxis.set_major_locator(mticker.MultipleLocator(200))
    elif traj_range > 200:
        ax.xaxis.set_major_locator(mticker.MultipleLocator(150))


# ── Figure: 2×4, top = F1, bottom = separatrix ──────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(7.16, 2.8))
fig.subplots_adjust(wspace=0.12, hspace=0.15, top=0.93, bottom=0.18,
                    left=0.075, right=0.97)

for idx, (title, nonad, adapt, baselines) in enumerate(panels):
    ax_f1 = axes[0, idx]
    ax_sep = axes[1, idx]
    ymin_f1, ymax_f1, ytick_f1 = f1_ylims[idx]
    ymin_sep, ymax_sep, ytick_sep = sep_ylims[idx]

    # Collect all datasets for x-axis limits
    all_datasets = [nonad, adapt] + list(baselines.values())

    # ── Top row: F1 ──
    ax_f1.plot(adapt['traj'], adapt['f1'], '-o', color=C_ADAPT,
               markersize=3, markeredgewidth=0, zorder=5)
    ax_f1.plot(nonad['traj'], nonad['f1'], '-s', color=C_NONAD,
               markersize=2.5, markeredgewidth=0, zorder=5)
    for bname in baseline_order:
        if bname not in baselines:
            continue
        bdata = baselines[bname]
        sty = BASELINE_STYLES[bname]
        ax_f1.plot(bdata['traj'], bdata['f1'],
                   linestyle=sty['ls'], color=sty['color'], linewidth=sty['lw'],
                   marker=sty['marker'], markersize=sty['ms'], markeredgewidth=0.5,
                   zorder=3)

    ax_f1.set_title(title, fontsize=9, pad=4)
    ax_f1.set_ylim(ymin_f1, ymax_f1)
    ax_f1.yaxis.set_major_locator(mticker.MultipleLocator(ytick_f1))
    if idx == 0:
        ax_f1.set_ylabel('F1 Score', fontsize=8)
    else:
        ax_f1.set_ylabel('')
        ax_f1.set_yticklabels([])
        ax_f1.tick_params(left=False)
    ax_f1.spines['top'].set_visible(False)
    get_xlims_and_format(ax_f1, *all_datasets)
    # Strip x-ticks AFTER formatter is set
    ax_f1.set_xticklabels([])
    ax_f1.tick_params(bottom=False)

    # ── Bottom row: Separatrix ──
    ax_sep.plot(adapt['traj'], adapt['sep'], '-o', color=C_ADAPT,
                markersize=3, markeredgewidth=0, zorder=5)
    ax_sep.plot(nonad['traj'], nonad['sep'], '-s', color=C_NONAD,
                markersize=2.5, markeredgewidth=0, zorder=5)
    for bname in baseline_order:
        if bname not in baselines:
            continue
        bdata = baselines[bname]
        sty = BASELINE_STYLES[bname]
        sep_vals = np.array(bdata['sep'])
        ax_sep.plot(bdata['traj'], sep_vals,
                    linestyle=sty['ls'], color=sty['color'], linewidth=sty['lw'],
                    marker=sty['marker'], markersize=sty['ms'], markeredgewidth=0.5,
                    zorder=3, clip_on=True)

    ax_sep.set_ylim(ymin_sep, ymax_sep)
    ax_sep.yaxis.set_major_locator(mticker.MultipleLocator(ytick_sep))
    if idx == 0:
        ax_sep.set_ylabel('Separatrix (%)', fontsize=8)
    else:
        ax_sep.set_ylabel('')
        ax_sep.set_yticklabels([])
        ax_sep.tick_params(left=False)
    ax_sep.set_xlabel('')
    ax_sep.spines['top'].set_visible(False)
    get_xlims_and_format(ax_sep, *all_datasets)

# Centered "Trajectories" label across all columns
fig.text(0.5, 0.09, 'Trajectories', ha='center', fontsize=6.5)

# ── Legend ────────────────────────────────────────────────────────────────
method_handles = [
    Line2D([0], [0], color=C_ADAPT, linestyle='-', marker='o', markersize=3,
           markeredgewidth=0, linewidth=1.5, label='Adaptive (Ours)'),
    Line2D([0], [0], color=C_NONAD, linestyle='-', marker='s', markersize=2.5,
           markeredgewidth=0, linewidth=1.5, label='Non-adaptive (Ours)'),
]
baseline_handles = []
for bname in baseline_order:
    sty = BASELINE_STYLES[bname]
    baseline_handles.append(
        Line2D([0], [0], color=sty['color'], linestyle=sty['ls'],
               linewidth=sty['lw'], marker=sty['marker'], markersize=sty['ms'],
               markeredgewidth=0.5, label=baseline_labels[bname])
    )

all_handles = method_handles + baseline_handles
fig.legend(handles=all_handles, loc='lower center',
           ncol=6, frameon=False, fontsize=6.5,
           bbox_to_anchor=(0.5, 0.01),
           columnspacing=1.2, handlelength=2.2, handletextpad=0.5)

# ── Save (trend version) ─────────────────────────────────────────────────
for ext in ('pdf', 'png'):
    p = f'{BASE}/results/figures/data_efficiency.{ext}'
    fig.savefig(p, format=ext, dpi=300, bbox_inches='tight', pad_inches=0.03)
    print(f'Saved {p}')
plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════
# Version 2: Horizontal lines for baselines (using last data point value)
# ══════════════════════════════════════════════════════════════════════════

fig2, axes2 = plt.subplots(2, 4, figsize=(7.16, 2.8))
fig2.subplots_adjust(wspace=0.12, hspace=0.15, top=0.93, bottom=0.18,
                     left=0.075, right=0.97)

for idx, (title, nonad, adapt, baselines) in enumerate(panels):
    ax_f1 = axes2[0, idx]
    ax_sep = axes2[1, idx]
    ymin_f1, ymax_f1, ytick_f1 = f1_ylims[idx]
    ymin_sep, ymax_sep, ytick_sep = sep_ylims[idx]

    # ── Top row: F1 ──
    ax_f1.plot(adapt['traj'], adapt['f1'], '-o', color=C_ADAPT,
               markersize=3, markeredgewidth=0, zorder=5)
    ax_f1.plot(nonad['traj'], nonad['f1'], '-s', color=C_NONAD,
               markersize=2.5, markeredgewidth=0, zorder=5)
    for bname in baseline_order:
        if bname not in baselines:
            continue
        bdata = baselines[bname]
        sty = BASELINE_STYLES[bname]
        ax_f1.axhline(y=bdata['f1'][-1], linestyle=sty['ls'], color=sty['color'],
                       linewidth=sty['lw'], zorder=2)

    ax_f1.set_title(title, fontsize=9, pad=4)
    ax_f1.set_ylim(ymin_f1, ymax_f1)
    ax_f1.yaxis.set_major_locator(mticker.MultipleLocator(ytick_f1))
    if idx == 0:
        ax_f1.set_ylabel('F1 Score', fontsize=8)
    else:
        ax_f1.set_ylabel('')
        ax_f1.set_yticklabels([])
        ax_f1.tick_params(left=False)
    ax_f1.spines['top'].set_visible(False)
    get_xlims_and_format(ax_f1, nonad, adapt)
    ax_f1.set_xticklabels([])
    ax_f1.tick_params(bottom=False)

    # ── Bottom row: Separatrix ──
    ax_sep.plot(adapt['traj'], adapt['sep'], '-o', color=C_ADAPT,
                markersize=3, markeredgewidth=0, zorder=5)
    ax_sep.plot(nonad['traj'], nonad['sep'], '-s', color=C_NONAD,
                markersize=2.5, markeredgewidth=0, zorder=5)
    for bname in baseline_order:
        if bname not in baselines:
            continue
        bdata = baselines[bname]
        sty = BASELINE_STYLES[bname]
        ax_sep.axhline(y=bdata['sep'][-1], linestyle=sty['ls'], color=sty['color'],
                        linewidth=sty['lw'], zorder=2)

    ax_sep.set_ylim(ymin_sep, ymax_sep)
    ax_sep.yaxis.set_major_locator(mticker.MultipleLocator(ytick_sep))
    if idx == 0:
        ax_sep.set_ylabel('Separatrix (%)', fontsize=8)
    else:
        ax_sep.set_ylabel('')
        ax_sep.set_yticklabels([])
        ax_sep.tick_params(left=False)
    ax_sep.set_xlabel('')
    ax_sep.spines['top'].set_visible(False)
    get_xlims_and_format(ax_sep, nonad, adapt)

fig2.text(0.5, 0.09, 'Trajectories', ha='center', fontsize=6.5)

# ── Legend (hline version — no markers) ──────────────────────────────────
method_handles2 = [
    Line2D([0], [0], color=C_ADAPT, linestyle='-', marker='o', markersize=3,
           markeredgewidth=0, linewidth=1.5, label='Adaptive (Ours)'),
    Line2D([0], [0], color=C_NONAD, linestyle='-', marker='s', markersize=2.5,
           markeredgewidth=0, linewidth=1.5, label='Non-adaptive (Ours)'),
]
baseline_handles2 = []
for bname in baseline_order:
    sty = BASELINE_STYLES[bname]
    baseline_handles2.append(
        Line2D([0], [0], color=sty['color'], linestyle=sty['ls'],
               linewidth=sty['lw'], label=baseline_labels[bname])
    )

all_handles2 = method_handles2 + baseline_handles2
fig2.legend(handles=all_handles2, loc='lower center',
            ncol=6, frameon=False, fontsize=6.5,
            bbox_to_anchor=(0.5, 0.02),
            columnspacing=1.2, handlelength=2.2, handletextpad=0.5)

# ── Save (hline version) ────────────────────────────────────────────────
for ext in ('pdf', 'png'):
    p = f'{BASE}/results/figures/data_efficiency_hline.{ext}'
    fig2.savefig(p, format=ext, dpi=300, bbox_inches='tight', pad_inches=0.03)
    print(f'Saved {p}')
plt.close(fig2)


# ══════════════════════════════════════════════════════════════════════════
# Version 3: Zoomed F1 only — Pendulum & CartPole with hlines
# ══════════════════════════════════════════════════════════════════════════

zoomed_panels = [
    ('Pendulum (2D)', pend_nonad, pend_adapt, pend_baselines),
    ('CartPole (4D)', cart_nonad, cart_adapt, cart_baselines),
]
zoomed_f1_ylims = [
    (0.92, 1.002, 0.02),   # Pendulum — tight around 0.93-1.0
    (0.87, 1.002, 0.02),   # CartPole — tight around 0.88-1.0
]

fig3, axes3 = plt.subplots(1, 2, figsize=(7.0, 2.2))
fig3.subplots_adjust(wspace=0.35, top=0.88, bottom=0.22,
                     left=0.07, right=0.97)

for idx, (title, nonad, adapt, baselines) in enumerate(zoomed_panels):
    ax = axes3[idx]
    ymin, ymax, ytick = zoomed_f1_ylims[idx]

    ax.plot(adapt['traj'], adapt['f1'], '-o', color=C_ADAPT,
            markersize=4, markeredgewidth=0, linewidth=1.5, zorder=5)
    ax.plot(nonad['traj'], nonad['f1'], '-s', color=C_NONAD,
            markersize=3.5, markeredgewidth=0, linewidth=1.5, zorder=5)

    for bname in baseline_order:
        if bname not in baselines:
            continue
        bdata = baselines[bname]
        sty = BASELINE_STYLES[bname]
        val = bdata['f1'][-1]
        # Only draw if within visible range
        if val >= ymin:
            ax.axhline(y=val, linestyle=sty['ls'], color=sty['color'],
                        linewidth=sty['lw'], zorder=2)
            # Label on right edge
            ax.text(ax.get_xlim()[1] if ax.get_xlim()[1] > 0 else max(nonad['traj']),
                    val, f' {baseline_labels[bname]}',
                    fontsize=5.5, color=sty['color'], va='center', ha='left',
                    clip_on=False)

    ax.set_title(title, fontsize=9, pad=4)
    ax.set_ylim(ymin, ymax)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(ytick))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(ytick / 2))
    ax.set_ylabel('F1 Score' if idx == 0 else '', fontsize=8)
    ax.set_xlabel('Trajectories', fontsize=8)
    ax.spines['top'].set_visible(False)
    get_xlims_and_format(ax, nonad, adapt)

# Legend
method_handles3 = [
    Line2D([0], [0], color=C_ADAPT, linestyle='-', marker='o', markersize=3.5,
           markeredgewidth=0, linewidth=1.5, label='Adaptive (Ours)'),
    Line2D([0], [0], color=C_NONAD, linestyle='-', marker='s', markersize=3,
           markeredgewidth=0, linewidth=1.5, label='Non-adaptive (Ours)'),
]
fig3.legend(handles=method_handles3, loc='upper center',
            ncol=2, frameon=False, fontsize=7,
            bbox_to_anchor=(0.5, 1.0),
            columnspacing=2.0, handlelength=2.2, handletextpad=0.5)

# ── Save (zoomed hline version) ──────────────────────────────────────────
for ext in ('pdf', 'png'):
    p = f'{BASE}/results/figures/data_efficiency_zoomed_f1_hline.{ext}'
    fig3.savefig(p, format=ext, dpi=300, bbox_inches='tight', pad_inches=0.03)
    print(f'Saved {p}')
plt.close(fig3)


# ══════════════════════════════════════════════════════════════════════════
# Version 4: Zoomed F1 only — Pendulum & CartPole with trend lines
# ══════════════════════════════════════════════════════════════════════════

fig4, axes4 = plt.subplots(1, 2, figsize=(7.0, 2.2))
fig4.subplots_adjust(wspace=0.35, top=0.88, bottom=0.22,
                     left=0.07, right=0.97)

for idx, (title, nonad, adapt, baselines) in enumerate(zoomed_panels):
    ax = axes4[idx]
    ymin, ymax, ytick = zoomed_f1_ylims[idx]

    ax.plot(adapt['traj'], adapt['f1'], '-o', color=C_ADAPT,
            markersize=4, markeredgewidth=0, linewidth=1.5, zorder=5)
    ax.plot(nonad['traj'], nonad['f1'], '-s', color=C_NONAD,
            markersize=3.5, markeredgewidth=0, linewidth=1.5, zorder=5)

    for bname in baseline_order:
        if bname not in baselines:
            continue
        bdata = baselines[bname]
        sty = BASELINE_STYLES[bname]
        # Only plot points within visible y-range
        f1_arr = np.array(bdata['f1'])
        traj_arr = np.array(bdata['traj'])
        mask = f1_arr >= ymin
        if mask.any():
            ax.plot(traj_arr[mask], f1_arr[mask],
                    linestyle=sty['ls'], color=sty['color'], linewidth=sty['lw'],
                    marker=sty['marker'], markersize=sty['ms'], markeredgewidth=0.5,
                    zorder=3)

    ax.set_title(title, fontsize=9, pad=4)
    ax.set_ylim(ymin, ymax)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(ytick))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(ytick / 2))
    ax.set_ylabel('F1 Score' if idx == 0 else '', fontsize=8)
    ax.set_xlabel('Trajectories', fontsize=8)
    ax.spines['top'].set_visible(False)
    get_xlims_and_format(ax, nonad, adapt)

# Legend
method_handles4 = [
    Line2D([0], [0], color=C_ADAPT, linestyle='-', marker='o', markersize=3.5,
           markeredgewidth=0, linewidth=1.5, label='Adaptive (Ours)'),
    Line2D([0], [0], color=C_NONAD, linestyle='-', marker='s', markersize=3,
           markeredgewidth=0, linewidth=1.5, label='Non-adaptive (Ours)'),
]
baseline_handles4 = []
for bname in baseline_order:
    sty = BASELINE_STYLES[bname]
    baseline_handles4.append(
        Line2D([0], [0], color=sty['color'], linestyle=sty['ls'],
               linewidth=sty['lw'], marker=sty['marker'], markersize=sty['ms'],
               markeredgewidth=0.5, label=baseline_labels[bname])
    )
all_handles4 = method_handles4 + baseline_handles4
fig4.legend(handles=all_handles4, loc='upper center',
            ncol=6, frameon=False, fontsize=6.5,
            bbox_to_anchor=(0.5, 1.0),
            columnspacing=1.2, handlelength=2.2, handletextpad=0.5)

# ── Save (zoomed trend version — 1×2) ────────────────────────────────────
for ext in ('pdf', 'png'):
    p = f'{BASE}/results/figures/data_efficiency_zoomed_f1.{ext}'
    fig4.savefig(p, format=ext, dpi=300, bbox_inches='tight', pad_inches=0.03)
    print(f'Saved {p}')
plt.close(fig4)


# ══════════════════════════════════════════════════════════════════════════
# Helper: build full 2×4 figure with given F1 ylims and baseline mode
# ══════════════════════════════════════════════════════════════════════════

def build_full_figure(f1_lims, use_hline=False):
    """Build 2×4 figure. If use_hline, baselines are horizontal lines; else trend lines."""
    fig, axes = plt.subplots(2, 4, figsize=(7.16, 2.8))
    fig.subplots_adjust(wspace=0.12, hspace=0.15, top=0.93, bottom=0.18,
                        left=0.075, right=0.97)

    for idx, (title, nonad, adapt, baselines) in enumerate(panels):
        ax_f1 = axes[0, idx]
        ax_sep = axes[1, idx]
        ymin_f1, ymax_f1, ytick_f1 = f1_lims[idx]
        ymin_sep, ymax_sep, ytick_sep = sep_ylims[idx]

        all_datasets = [nonad, adapt] + list(baselines.values())

        # ── Top: F1 ──
        ax_f1.plot(adapt['traj'], adapt['f1'], '-o', color=C_ADAPT,
                   markersize=3, markeredgewidth=0, zorder=5)
        ax_f1.plot(nonad['traj'], nonad['f1'], '-s', color=C_NONAD,
                   markersize=2.5, markeredgewidth=0, zorder=5)
        for bname in baseline_order:
            if bname not in baselines:
                continue
            bdata = baselines[bname]
            sty = BASELINE_STYLES[bname]
            if use_hline:
                ax_f1.axhline(y=bdata['f1'][-1], linestyle=sty['ls'],
                              color=sty['color'], linewidth=sty['lw'], zorder=2)
            else:
                ax_f1.plot(bdata['traj'], bdata['f1'],
                           linestyle=sty['ls'], color=sty['color'], linewidth=sty['lw'],
                           marker=sty['marker'], markersize=sty['ms'],
                           markeredgewidth=0.5, zorder=3)

        ax_f1.set_title(title, fontsize=9, pad=4)
        ax_f1.set_ylim(ymin_f1, ymax_f1)
        ax_f1.yaxis.set_major_locator(mticker.MultipleLocator(ytick_f1))
        if idx == 0:
            ax_f1.set_ylabel('F1 Score', fontsize=8)
        else:
            ax_f1.set_ylabel('')
            ax_f1.set_yticklabels([])
            ax_f1.tick_params(left=False)
        ax_f1.spines['top'].set_visible(False)
        if use_hline:
            get_xlims_and_format(ax_f1, nonad, adapt)
        else:
            get_xlims_and_format(ax_f1, *all_datasets)
        ax_f1.set_xticklabels([])
        ax_f1.tick_params(bottom=False)

        # ── Bottom: Sep ──
        ax_sep.plot(adapt['traj'], adapt['sep'], '-o', color=C_ADAPT,
                    markersize=3, markeredgewidth=0, zorder=5)
        ax_sep.plot(nonad['traj'], nonad['sep'], '-s', color=C_NONAD,
                    markersize=2.5, markeredgewidth=0, zorder=5)
        for bname in baseline_order:
            if bname not in baselines:
                continue
            bdata = baselines[bname]
            sty = BASELINE_STYLES[bname]
            if use_hline:
                ax_sep.axhline(y=bdata['sep'][-1], linestyle=sty['ls'],
                               color=sty['color'], linewidth=sty['lw'], zorder=2)
            else:
                ax_sep.plot(bdata['traj'], bdata['sep'],
                            linestyle=sty['ls'], color=sty['color'], linewidth=sty['lw'],
                            marker=sty['marker'], markersize=sty['ms'],
                            markeredgewidth=0.5, zorder=3, clip_on=True)

        ax_sep.set_ylim(ymin_sep, ymax_sep)
        ax_sep.yaxis.set_major_locator(mticker.MultipleLocator(ytick_sep))
        if idx == 0:
            ax_sep.set_ylabel('Separatrix (%)', fontsize=8)
        else:
            ax_sep.set_ylabel('')
            ax_sep.set_yticklabels([])
            ax_sep.tick_params(left=False)
        ax_sep.set_xlabel('')
        ax_sep.spines['top'].set_visible(False)
        if use_hline:
            get_xlims_and_format(ax_sep, nonad, adapt)
        else:
            get_xlims_and_format(ax_sep, *all_datasets)

    fig.text(0.5, 0.09, 'Trajectories', ha='center', fontsize=6.5)

    # Legend
    mh = [
        Line2D([0], [0], color=C_ADAPT, linestyle='-', marker='o', markersize=3,
               markeredgewidth=0, linewidth=1.5, label='Adaptive (Ours)'),
        Line2D([0], [0], color=C_NONAD, linestyle='-', marker='s', markersize=2.5,
               markeredgewidth=0, linewidth=1.5, label='Non-adaptive (Ours)'),
    ]
    bh = []
    for bname in baseline_order:
        sty = BASELINE_STYLES[bname]
        if use_hline:
            bh.append(Line2D([0], [0], color=sty['color'], linestyle=sty['ls'],
                             linewidth=sty['lw'], label=baseline_labels[bname]))
        else:
            bh.append(Line2D([0], [0], color=sty['color'], linestyle=sty['ls'],
                             linewidth=sty['lw'], marker=sty['marker'],
                             markersize=sty['ms'], markeredgewidth=0.5,
                             label=baseline_labels[bname]))
    fig.legend(handles=mh + bh, loc='lower center',
               ncol=6, frameon=False, fontsize=6.5,
               bbox_to_anchor=(0.5, 0.01),
               columnspacing=1.2, handlelength=2.2, handletextpad=0.5)
    return fig


# ══════════════════════════════════════════════════════════════════════════
# Version 5: Full 2×4, zoomed F1, trend lines
# ══════════════════════════════════════════════════════════════════════════
fig5 = build_full_figure(f1_ylims_zoomed, use_hline=False)
for ext in ('pdf', 'png'):
    p = f'{BASE}/results/figures/data_efficiency_zoomed.{ext}'
    fig5.savefig(p, format=ext, dpi=300, bbox_inches='tight', pad_inches=0.03)
    print(f'Saved {p}')
plt.close(fig5)

# ══════════════════════════════════════════════════════════════════════════
# Version 6: Full 2×4, zoomed F1, hlines
# ══════════════════════════════════════════════════════════════════════════
fig6 = build_full_figure(f1_ylims_zoomed, use_hline=True)
for ext in ('pdf', 'png'):
    p = f'{BASE}/results/figures/data_efficiency_zoomed_hline.{ext}'
    fig6.savefig(p, format=ext, dpi=300, bbox_inches='tight', pad_inches=0.03)
    print(f'Saved {p}')
plt.close(fig6)
