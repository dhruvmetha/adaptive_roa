"""FM adaptive-fraction line plots over rho, all six settings, empirical soft rule: previews only.

The paper's Fig 4 is the certified version (plot_adaptive_fraction_certified.py); Dhruv chose it on
2026-11-12 and adaptive_fraction.pdf left the paper. Nothing here writes to the paper any more.

  split_rowaxes_dotted  was the paper's Fig 4 until the certified version replaced it. Recall panel over an error panel
               per setting, y axes zoomed and shared within each row (LQR, RL), 1:1 panel heights.
               Recall solid, FDR/FOR dotted, solid gray 5% line (not in the legend), four-entry
               legend in the method section's terms. Fails if a marker or band would leave its panel.
  split_rowaxes  preview: the same without the dotted errors and four-entry legend
  split        preview: recall panel over an error panel on fixed 0-100 / 0-ERR_MAX axes
  combined     rejected preview (v1): all four lines overlaid on twin axes spanning the whole panel
  combined_v2  rejected preview: recall and errors in separate vertical bands (banded.py)

FM, U_H, rho in {0, .5, .75, 1} (rho=0 is uniform), soft four-threshold rule fit on validation
at w=0.2, final-budget test. Line = three-seed mean, band = min to max over seeds (common.spread;
never SD). FDR/FOR use only seeds with a nonempty predicted class. A hollow marker means fewer
than 3 such seeds.

"""
import hashlib, json, math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from common import (VALUES, SETTINGS, TITLES, BLUE, ORANGE, REF, RECALL, ERROR, RC, SOFT_FOUR as SRC,
                    load_final, cell, parse_args, save)
from split import split_figure, draw, assert_markers_inside, assert_bands_inside, REF_SOLID
from banded import check_gap, banded_grid

ARGS = parse_args(__doc__)
COMBOS = [('uniform', 0.), ('U_H', .5), ('U_H', .75), ('U_H', 1.)]
RHO = np.array([rho for _, rho in COMBOS])
RHO_LABELS = ['0', '.5', '.75', '1']
XLABEL = r'Adaptive fraction $\rho$  ($\rho=0$ is uniform)'
# Per-row y axes for the rowaxes splits: (ylim, ticks). Row 0 is LQR, row 1 is RL.
ROW_AXES = {0: dict(rec=((48, 102), [50, 75, 100]), err=((-.18, 6), [0, 5])),
            1: dict(rec=((38, 102), [40, 70, 100]), err=((-.36, 12), [0, 5, 10]))}

final = load_final()
values = [cell(final, s, a, rho, m) for s in SETTINGS for a, rho in COMBOS for m, _ in RECALL + ERROR]
lookup = {(v['setting'], v['rho'], v['metric']): v for v in values}
ERR_MAX = math.ceil((max(v['max'] for v in values if v['metric'] in ('fdr', 'for') and v['n']) + .1) / 2) * 2


def series(setting, metric):
    """(mean, min, max, n) over rho for one setting and metric; NaN where no seed contributes."""
    cc = [lookup[(setting, rho, metric)] for rho in RHO]
    f = lambda k: np.array([np.nan if c[k] is None else c[k] for c in cc], dtype=float)
    return f('mean'), f('min'), f('max'), np.array([c['n'] for c in cc])


def combined_v1():
    """Rejected preview: twin axes spanning the whole panel, all four lines overlaid."""
    fig, axs = plt.subplots(2, 3, figsize=(6.5, 3.9))
    fig.subplots_adjust(left=.08, right=.93, top=.86, bottom=.13, wspace=.42, hspace=.45)
    for si, (ax, s, title) in enumerate(zip(axs.flat, SETTINGS, TITLES)):
        ax.set_title(title, pad=5); ax.grid(axis='y', alpha=.2, lw=.45); ax.set_axisbelow(True)
        right = ax.twinx(); right.spines['right'].set_visible(True); right.spines['top'].set_visible(False)
        for a, pairs, ls, marker in [(ax, RECALL, '-', 'o'), (right, ERROR, '--', 's')]:
            for metric, color in pairs:
                draw(a, RHO, *series(s, metric), color, ls=ls, marker=marker)
        right.axhline(5, color=REF, lw=.65, ls=':', zorder=0)
        ax.set_ylim(-3, 103); ax.set_yticks([0, 50, 100])
        right.set_ylim(-.03 * ERR_MAX, ERR_MAX); right.set_yticks([0, 5, 10] if ERR_MAX >= 10 else [0, 5])
        for a, pairs in [(ax, RECALL), (right, ERROR)]:
            got = [series(s, m) for m, _ in pairs]
            assert_markers_inside(fig, a, np.concatenate([g[0] for g in got]))
            assert_bands_inside(a, [(g[1], g[2]) for g in got])
        ax.set_xlim(-.05, 1.05); ax.set_xticks(RHO, RHO_LABELS)
        if si % 3 == 0: ax.set_ylabel('Recall (%) ↑')
        if si % 3 == 2: right.set_ylabel('Error (%) ↓')
    fig.supxlabel(XLABEL, fontsize=7.5, y=.03)
    fig.legend(handles=[Line2D([], [], color=BLUE, marker='o', ms=3.2, label=r'$R_{\mathrm{s}}$ (left)'),
                        Line2D([], [], color=ORANGE, marker='o', ms=3.2, label=r'$R_{\mathrm{f}}$ (left)'),
                        Line2D([], [], color=BLUE, ls='--', marker='s', ms=3.2, label='FDR (right)'),
                        Line2D([], [], color=ORANGE, ls='--', marker='s', ms=3.2, label='FOR (right)'),
                        Line2D([], [], color=REF, lw=.65, ls=':', label='5% (right)')],
               loc='upper center', ncol=5, frameon=False, bbox_to_anchor=(.54, 1.0), columnspacing=1.5,
               handlelength=2.2)
    return save(fig, 'adaptive_fraction_lines_combined')


sizes = {}
common_args = (series, RHO, RHO_LABELS, (-.05, 1.05), XLABEL)
with plt.rc_context(RC):
    sizes['adaptive_fraction_lines_split'] = split_figure(*common_args, 'adaptive_fraction_lines_split',
                                                          err_max=ERR_MAX)
    sizes['adaptive_fraction_lines_split_rowaxes'] = split_figure(
        *common_args, 'adaptive_fraction_lines_split_rowaxes', height_ratios=(1, 1), row_axes=ROW_AXES)
    # Was the paper's Fig 4 (empirical); now a preview next to the certified one.
    sizes['adaptive_fraction_lines_split_rowaxes_dotted'] = split_figure(
        *common_args, 'adaptive_fraction_lines_split_rowaxes_dotted', height_ratios=(1, 1), row_axes=ROW_AXES,
        err_ls=':', ref=REF_SOLID, legend_four=True)
    sizes['adaptive_fraction_lines_combined'] = combined_v1()
    bands = check_gap(values)
    sizes['adaptive_fraction_lines_combined_v2'] = banded_grid(*common_args, 'adaptive_fraction_lines_combined_v2')

(VALUES / 'adaptive_fraction_lines_values.json').write_text(json.dumps(dict(
    source=str(SRC), source_sha256=hashlib.sha256(SRC.read_bytes()).hexdigest(),
    protocol='FM, U_H, soft four-threshold rule fit on validation at w=0.2, final-budget test; '
             'mean with min-max over 3 seeds; FDR/FOR over seeds with nonempty S/F only',
    error_axis_max_percent=ERR_MAX, row_axes=ROW_AXES, bands=bands, page_sizes=sizes, values=values), indent=2) + '\n')
print('error axis 0-%d%%, %d cells, min n=%d' % (ERR_MAX, len(values), min(v['n'] for v in values)))
