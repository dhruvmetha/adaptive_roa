"""The banded 2x3 layout of the adaptive_fraction combined_v2 preview (rejected by Dhruv 2026-11-12).

One panel per setting. Recall (R_s, R_f) sits in the top band on the left axis, FDR/FOR in the
bottom band on the right axis, so the two can never cross. check_gap() fails loudly if new data
would make the bands touch.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from common import SETTINGS, TITLES, BLUE, ORANGE, REF, RECALL, ERROR, RC, save
from split import draw

REC_LIM, REC_SPINE, REC_TICKS = (-50, 102), (40, 100), [40, 70, 100]
ERR_LIM, ERR_SPINE, ERR_TICKS = (-.6, 25), (0, 12), [0, 5, 10]


def frac(y, lim):
    return (y - lim[0]) / (lim[1] - lim[0])


def check_gap(values):
    rec_floor = min(v['min'] for v in values if v['metric'] in ('rec', 'fail_rec') and v['n'])
    err_ceil = max(v['max'] for v in values if v['metric'] in ('fdr', 'for') and v['n'])
    gap = frac(rec_floor, REC_LIM) - frac(err_ceil, ERR_LIM)
    assert rec_floor >= REC_SPINE[0] and err_ceil <= ERR_SPINE[1] and gap > .1, (rec_floor, err_ceil, gap)
    print(f'bands: lowest recall edge {rec_floor:.2f} at {frac(rec_floor, REC_LIM):.3f} of panel, '
          f'highest error edge {err_ceil:.2f} at {frac(err_ceil, ERR_LIM):.3f}, gap {gap:.3f}')
    return dict(rec_floor=rec_floor, err_ceil=err_ceil, gap=gap)


def banded_grid(series, x, xticklabels, xlim, xlabel, stem, paper_stem=None, paper=False):
    """series(setting, metric) -> (mean, lo, hi, n) arrays over x. Markers joined by lines, min-max shaded."""
    x = np.asarray(x, dtype=float)
    with plt.rc_context(RC):
        fig, axs = plt.subplots(2, 3, figsize=(6.5, 3.9))
        fig.subplots_adjust(left=.08, right=.93, top=.86, bottom=.13, wspace=.32, hspace=.45)
        for si, (ax, s, title) in enumerate(zip(axs.flat, SETTINGS, TITLES)):
            ax.set_title(title, pad=5)
            right = ax.twinx()
            for side in ('top', 'left', 'bottom'):  # the twin draws its own full-height left spine
                right.spines[side].set_visible(False)
            for a, pairs, ls, marker in [(ax, RECALL, '-', 'o'), (right, ERROR, '--', 's')]:
                for metric, color in pairs:
                    mean, lo, hi, n = series(s, metric)
                    draw(a, x, mean, lo, hi, n, color, ls=ls, marker=marker)
            right.axhline(5, color=REF, lw=.65, ls=':', zorder=0)
            for a, lim, spine, ticks, side in [(ax, REC_LIM, REC_SPINE, REC_TICKS, 'left'),
                                               (right, ERR_LIM, ERR_SPINE, ERR_TICKS, 'right')]:
                a.set_ylim(*lim); a.set_yticks(ticks)
                a.spines[side].set_visible(True); a.spines[side].set_bounds(*spine)
            for t in REC_TICKS:
                ax.axhline(t, color='#000000', alpha=.2 * .45 / .65, lw=.45, zorder=0)
            ax.set_xlim(*xlim)
            ax.set_xticks(x, xticklabels)
            # Center each axis label on its own band, not on the panel.
            if si % 3 == 0:
                ax.set_ylabel('Recall (%) ↑'); ax.yaxis.set_label_coords(-.21, .67)
            if si % 3 == 2:
                right.set_ylabel('Error (%) ↓'); right.yaxis.set_label_coords(1.18, frac(6, ERR_LIM))
        fig.supxlabel(xlabel, fontsize=7.5, y=.03)
        rec_ls, err_ls = '-', '--'
        handles = [Line2D([], [], color=BLUE, ls=rec_ls, marker='o', ms=3.2, label=r'$R_{\mathrm{s}}$'),
                   Line2D([], [], color=ORANGE, ls=rec_ls, marker='o', ms=3.2, label=r'$R_{\mathrm{f}}$'),
                   Line2D([], [], color=BLUE, ls=err_ls, marker='s', ms=3.2, label='FDR'),
                   Line2D([], [], color=ORANGE, ls=err_ls, marker='s', ms=3.2, label='FOR'),
                   Line2D([], [], color=REF, lw=.65, ls=':', label='5% error')]
        fig.legend(handles=handles, loc='upper center', ncol=len(handles), frameon=False,
                   bbox_to_anchor=(.54, 1.0), columnspacing=1.5, handlelength=2.2)
        return save(fig, stem, paper_stem=paper_stem, paper=paper)
