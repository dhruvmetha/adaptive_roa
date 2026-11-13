"""ARCHIVED 2026-11-12: bar version of adaptive_fraction (4 settings, recall row over FDR/FOR row).
Replaced in the paper by combined_v2 from ../plot_adaptive_fraction_lines.py (commit a52afc4).
No longer writes to the paper repo; rerunning it writes only to plots/archive and values/archive.


Same protocol as deterministic-verification/results-discussion-assets/plot_adaptive_comparisons.py:
FM, U_H, rho in {0, .5, .75, 1} (rho=0 is uniform), soft four-threshold rule fitted on
validation at w=0.2, final-budget test rates, three-seed mean +/- sample SD. FDR and FOR
average only over seeds whose predicted Success or Fail set is nonempty (the scorer stores
None there), the same rule as the tables.

Writes the paper figure into deterministic-verification/figures/results/ and a PDF/PNG copy
plus the plotted values into this folder.
"""
import hashlib, json, math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parents[2]           # tripods/dv_l4dc_plot
DV = HERE.parent / 'deterministic-verification'      # the paper repo
SRC = DV / 'results-discussion-assets/soft_four_wsweep.jsonl'
PLOTS, VALUES = HERE / 'plots/archive', HERE / 'values/archive'
PAPER = PLOTS / 'adaptive_fraction_bars.pdf'  # was the paper figure; now local only

SETTINGS = ['pen-lqr', 'cp-ppo', 'q2d-ppo', 'q3d-ppo']
TITLES = ['Pendulum (2D)', 'Cartpole (4D)', 'Planar Quad (6D)', 'Spatial Quad (13D)']
COMBOS = [('uniform', 0.), ('U_H', .5), ('U_H', .75), ('U_H', 1.)]
BLUE, ORANGE = '#0072B2', '#D55E00'  # same pairing as adaptive_learning.pdf
ROWS = [('recall', [('rec', BLUE), ('fail_rec', ORANGE)]),
        ('error', [('fdr', BLUE), ('for', ORANGE)])]

final = [r for l in SRC.read_text().splitlines() if (r := json.loads(l))['w'] == .2 and r['model'] == 'FM']
assert len(final) == 108
for r in final:
    t = r['test']
    assert (t['fdr'] is None) == (t['S'] == 0) and (t['for'] is None) == (t['F'] == 0)


def cell(setting, arm, rho, metric):
    rr = [r for r in final if (r['setting'], r['acq'], r['d2_ratio']) == (setting, arm, rho)]
    assert len(rr) == 3 and len({r['seed'] for r in rr}) == 3
    v = [100 * r['test'][metric] for r in rr if r['test'][metric] is not None]
    return dict(setting=setting, arm=arm, rho=rho, metric=metric, n=len(v),
                mean=float(np.mean(v)) if v else None,
                sd=float(np.std(v, ddof=1)) if len(v) > 1 else None)


values = [cell(s, a, rho, m) for s in SETTINGS for a, rho in COMBOS for _, pairs in ROWS for m, _ in pairs]
lookup = {(v['setting'], v['rho'], v['metric']): v for v in values}
err_top = max(v['mean'] + (v['sd'] or 0) for v in values if v['metric'] in ('fdr', 'for') and v['mean'] is not None)
err_ceiling = math.ceil((err_top + .1) / 2) * 2

rc = {'font.family': 'DejaVu Sans', 'font.size': 7.5, 'axes.titlesize': 7.5, 'axes.labelsize': 7,
      'xtick.labelsize': 7, 'ytick.labelsize': 6.5, 'legend.fontsize': 7, 'axes.linewidth': .65,
      'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False}
with plt.rc_context(rc):
    fig = plt.figure(figsize=(3.17, 2.95))
    outer = fig.add_gridspec(2, 2, left=.19, right=.99, top=.86, bottom=.11, hspace=.42, wspace=.2)
    xx = np.arange(len(COMBOS)); width = .36
    for si, (s, title) in enumerate(zip(SETTINGS, TITLES)):
        inner = outer[si // 2, si % 2].subgridspec(2, 1, height_ratios=[1.45, 1], hspace=.12)
        recall = fig.add_subplot(inner[0]); error = fig.add_subplot(inner[1], sharex=recall)
        recall.set_title(title, pad=3)
        for ax, (row, pairs) in zip((recall, error), ROWS):
            ax.grid(axis='y', alpha=.18, lw=.5); ax.set_axisbelow(True)
            ax.axhline(0, color='#555555', lw=.5, zorder=2)
            for mi, (metric, color) in enumerate(pairs):
                cc = [lookup[(s, rho, metric)] for _, rho in COMBOS]
                ax.bar(xx + (mi - .5) * width, [c['mean'] or 0 for c in cc], width, color=color,
                       edgecolor='white', linewidth=.25, zorder=3,
                       yerr=[c['sd'] or 0 for c in cc],
                       error_kw={'elinewidth': .6, 'capsize': 1.2, 'capthick': .6, 'ecolor': '#333333'})
        recall.set_ylim(0, 104); recall.set_yticks([0, 50, 100]); recall.tick_params(labelbottom=False)
        error.set_ylim(0, err_ceiling); error.set_yticks([0, 5, 10] if err_ceiling >= 10 else [0, 5])
        error.axhline(5, color='#666666', lw=.65, ls='--', zorder=2.5)
        error.set_xticks(xx, ['0', '.5', '.75', '1'])
        if si < 2: error.tick_params(labelbottom=False)
        if si % 2 == 0:
            recall.set_ylabel('Recall', labelpad=2); error.set_ylabel('Error', labelpad=2)
        else:
            recall.tick_params(labelleft=False); error.tick_params(labelleft=False)
    fig.supxlabel(r'Adaptive fraction $\rho$', fontsize=7.5, y=.005)
    fig.supylabel('Test (%)', fontsize=7.5, x=.005)
    handles = [Patch(color=BLUE, label=r'$R_{\mathrm{s}}$ ↑ / FDR ↓'), Patch(color=ORANGE, label=r'$R_{\mathrm{f}}$ ↑ / FOR ↓'),
               Line2D([], [], color='#666666', lw=.65, ls='--', label='5%')]
    fig.legend(handles=handles, loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(.57, 1.0),
               handlelength=1.2, columnspacing=1.0)
    fig.savefig(PAPER, bbox_inches='tight', pad_inches=.04)
    fig.savefig(PLOTS / 'adaptive_fraction_bars.png', dpi=220, bbox_inches='tight', pad_inches=.04)
    plt.close(fig)

(VALUES / 'adaptive_fraction_bars_values.json').write_text(json.dumps(dict(
    source=str(SRC), source_sha256=hashlib.sha256(SRC.read_bytes()).hexdigest(),
    protocol='FM, U_H, soft four-threshold rule fit on validation at w=0.2, final-budget test, '
             'mean and sample SD over 3 seeds; FDR/FOR over seeds with nonempty S/F only',
    error_axis_max_percent=err_ceiling, values=values), indent=2) + '\n')
print('wrote', PAPER, 'error axis 0-%d%%' % err_ceiling)
for v in values:
    print(f"{v['setting']:8s} rho={v['rho']:<4} {v['metric']:8s} {v['mean']:6.2f} ± {v['sd']:5.2f} (n={v['n']})")
