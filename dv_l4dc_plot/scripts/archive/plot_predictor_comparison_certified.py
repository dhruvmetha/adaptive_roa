"""ARCHIVED 2026-11-12: certified predictor comparison dot plot (was paper figure predictor_comparison_uniform_certified).
Dhruv picked the table instead: paper commit 7a21aa8 put tables/identical_data_certified_body.tex in Results as Table 4
(../make_identical_data_table.py) and deleted this PDF from the paper. It has no --paper option and writes only to
plots/archive and values/archive, so a rerun cannot put it back.


The question: with identical training trajectories and the same certification targets, which predictor correctly
identifies more successes and failures. Only the certified evaluation is shown: two-sided A-1d at
alpha_s = alpha_f = delta = 0.05 (method 'cal' of predictor_comparison_data.build(), c6c_* of matrix_rows.jsonl).
build() runs every check of the full comparison (identical training trajectories and evaluation sets, recomputed
counts, Table 1 30/30, Fig 3 24/24) before anything is drawn. Same 90 uniform runs, same aggregate().

Four panels: R_s, R_f, FDR, FOR, in percent. x = the six settings in SETTINGS order; within a setting, five dots in
table order (MLP-Label, BNN-Label, TERA-MLP, TERA-BNN, TERA-FM). Marker = three-seed mean, whisker = min to max
(common.spread). Recall averages all three seeds, the zero recall of all-Abstain runs included. FDR/FOR average only
seeds with a nonempty declaration set: a hollow marker with a seed-count numeral means 1 or 2 such seeds (no whisker
for 1), and a gray en dash near the top of the panel means none (undefined, never 0). The thin solid gray line on the
error panels is the 5% target. Nothing is clipped; the script fails if a marker, whisker or label would be.

--layout 2x2 (default, the layout that was in the paper) or 4x1, which writes
plots/archive/predictor_comparison_uniform_certified_4x1.{pdf,png}.
Writes plots/archive/predictor_comparison_uniform_certified.{pdf,png} and
values/archive/predictor_comparison_uniform_certified_values.json.
"""
import argparse, hashlib, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))     # scripts/
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory
from common import VALUES, SETTINGS, REF, RC, ASSETS, save
import certified
from split import REF_SOLID
from predictor_comparison_data import build, MODELS, DETV3
from plot_predictor_comparison import COLORS

PAPER_LAYOUT = '2x2'
ORDER = ['MLP classifier', 'BNN classifier', 'MLP final-state', 'BNN final-state', 'FM']     # table order, FM last
NAME = {m: plain for m, _, plain in MODELS}
# 4x1 names each setting in its tick label. 2x2 is too narrow for that ("Spatial Quad" is 42 pt at 6.5 pt, a slot
# 33 pt), so it breaks the quad names over two lines and puts LQR / PPO once under each group.
TICKS = {'4x1': ['Pendulum\nLQR', 'Cartpole\nLQR', 'Spatial Quad\nLQR', 'Cartpole\nPPO', 'Planar Quad\nPPO',
                 'Spatial Quad\nPPO'],
         '2x2': ['Pendulum', 'Cartpole', 'Spatial\nQuad', 'Cartpole', 'Planar\nQuad', 'Spatial\nQuad']}
GROUPS = [(1, 'LQR'), (4, 'PPO')]
# The 2x2 right column shares the left column's y axis and has no label, so the symbol goes in each title.
PANELS = [('rec', 'Recall (%)', r'Success recall $R_{\mathrm{s}}$'),
          ('fail_rec', 'Recall (%)', r'Fail recall $R_{\mathrm{f}}$'),
          ('fdr', 'Error rate (%)', 'FDR: error among predicted successes'),
          ('for', 'Error rate (%)', 'FOR: error among predicted failures')]
RECALL_YLIM, ERROR_YLIM = (-5.0, 105.0), (-0.3, 6.6)
EMPTY_Y = 0.94                     # en dash for an empty class, in axes fraction
MS, MEW = 3.4, .45
LAYOUTS = {'2x2': dict(figsize=(5.97, 4.0), rows=2, cols=2, tick_fs=6.5, spread=.3, xlim=(-.48, 5.48),
                       grid=dict(left=.07, right=.997, top=.89, bottom=.15, wspace=.045, hspace=.3)),
           '4x1': dict(figsize=(6.0, 5.6), xlim=(-.55, 5.45), rows=4, cols=1, tick_fs=7, spread=.3,
                       grid=dict(left=.075, right=.995, top=.925, bottom=.075, hspace=.32))}
MIN_TICK_FS, MIN_TICK_GAP_PT = 6.5, 2.0


def offsets(spread):
    return dict(zip(ORDER, np.linspace(-spread, spread, len(ORDER))))


def panel(ax, agg, metric, off):
    """Draw one metric; returns the plotted record per (setting, model) and every y the panel must contain."""
    recall = metric in ('rec', 'fail_rec')
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    out, ys = [], []
    for si, s in enumerate(SETTINGS):
        for model in ORDER:
            a = agg[s, model, 'cal'][metric]
            x, c = si + off[model], COLORS[model]
            rec = dict(setting=s, model=NAME[model], metric=metric, mean=a['mean'], min=a['min'], max=a['max'], n=a['n'])
            if recall:
                assert a['n'] == 3, (s, model, metric, a)
            if a['n'] == 0:
                assert not recall and a['mean'] is None
                ax.text(x, EMPTY_Y, '–', transform=trans, ha='center', va='center', fontsize=9, color='#666666',
                        clip_on=False)
                out.append(rec | dict(drawn='en dash (no seed with a nonempty declaration set)'))
                continue
            hollow = a['n'] < 3
            if a['n'] > 1:
                ax.vlines(x, a['min'], a['max'], colors=c, lw=.65, alpha=.85, zorder=2, clip_on=False)
            ax.plot(x, a['mean'], ls='none', marker='o', ms=MS, zorder=4, clip_on=False,
                    mfc='white' if hollow else c, mec=c if hollow else '#333333', mew=.8 if hollow else MEW)
            if hollow:
                ax.annotate(str(a['n']), (x, a['max']), xytext=(0, 1.6), textcoords='offset points', ha='center',
                            va='bottom', fontsize=5, color='#444444', annotation_clip=False)
            ys += [a['min'], a['max']]
            out.append(rec | dict(drawn='hollow marker + seed count' if hollow else 'filled marker',
                                  whisker=a['n'] > 1))
    return out, ys


def check_ticks(fig, ax):
    """Fail if bottom tick labels are smaller than MIN_TICK_FS or closer than MIN_TICK_GAP_PT to a neighbour."""
    r = fig.canvas.get_renderer()
    labs = [t for t in ax.get_xticklabels() if t.get_text()]
    assert all(t.get_fontsize() >= MIN_TICK_FS for t in labs)
    boxes = [t.get_window_extent(r) for t in labs]
    gaps = [(b.x0 - a.x1) * 72 / fig.dpi for a, b in zip(boxes, boxes[1:])]
    return min(gaps)


def draw(agg, layout):
    L = LAYOUTS[layout]
    off = offsets(L['spread'])
    plotted, ranges = [], {}
    with plt.rc_context(RC | {'xtick.labelsize': L['tick_fs'], 'ytick.labelsize': 6.5, 'axes.labelsize': 7,
                              'axes.titlesize': 7}):
        fig = plt.figure(figsize=L['figsize'])
        gs = fig.add_gridspec(L['rows'], L['cols'], **L['grid'])
        axes = {}
        for i, (metric, ylabel, title) in enumerate(PANELS):
            r, c = divmod(i, L['cols'])
            ax = fig.add_subplot(gs[r, c])
            axes[metric] = ax
            recall = metric in ('rec', 'fail_rec')
            ax.grid(axis='y', alpha=.2, lw=.45); ax.set_axisbelow(True)
            ax.axvline(2.5, color='#bbbbbb', lw=.5, zorder=0)              # LQR | PPO
            if not recall:
                ax.axhline(5, zorder=1, **REF_SOLID)
                ax.text(L['xlim'][0] + .06, 5, '5% target', ha='left', va='bottom', fontsize=5.5, color=REF)
            rows, ys = panel(ax, agg, metric, off)
            plotted += rows
            ylim = RECALL_YLIM if recall else ERROR_YLIM
            ax.set_ylim(*ylim); ax.set_xlim(*L['xlim'])
            assert ylim[0] < min(ys) and max(ys) < ylim[1], (metric, ylim, min(ys), max(ys))
            assert all(ylim[0] < y < ylim[1] for y in (0, 100) if recall), metric
            ax.set_yticks([0, 25, 50, 75, 100] if recall else [0, 2, 4, 6])
            ax.set_ylabel(ylabel, labelpad=1.5)
            ax.set_title(title, pad=3)
            bottom = r == L['rows'] - 1
            ax.set_xticks(range(len(SETTINGS)), TICKS[layout] if bottom else [''] * len(SETTINGS))
            if bottom and layout == '2x2':
                for gx, g in GROUPS:
                    ax.annotate(g, (gx, 0), xycoords=('data', 'axes fraction'), xytext=(0, -20.5),
                                textcoords='offset points', ha='center', va='top', fontsize=6.5, color='#555555')
            ax.tick_params(axis='x', length=0, pad=2)
            if layout == '2x2' and c == 1:
                ax.set_ylabel('')
                ax.tick_params(axis='y', labelleft=False)
            ranges[metric] = dict(x=ax.get_xlim(), y=ylim)
        if layout == '2x2':
            axes['fail_rec'].sharey(axes['rec']); axes['for'].sharey(axes['fdr'])
        handles = [Line2D([], [], ls='none', marker='o', ms=MS + .4, mfc=COLORS[m], mec='#333333', mew=MEW,
                          label=NAME[m]) for m in ORDER]
        fig.legend(handles=handles, loc='upper center', ncol=5, frameon=False, bbox_to_anchor=(.535, 1.0),
                   columnspacing=1.6, handletextpad=.25, fontsize=7)
        fig.canvas.draw()
        gap = min(check_ticks(fig, ax) for ax in axes.values() if any(t.get_text() for t in ax.get_xticklabels()))
        assert gap >= MIN_TICK_GAP_PT, f'{layout}: setting labels {gap:.1f} pt apart'
        stem = 'predictor_comparison_uniform_certified' + ('' if layout == PAPER_LAYOUT else f'_{layout}')
        size = save(fig, f'archive/{stem}')                         # plots/archive; never the paper
    assert 5.9 <= size['width_in'] <= 6.05, size
    return plotted, ranges, size, gap, stem


def main():
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument('--layout', choices=sorted(LAYOUTS), default=PAPER_LAYOUT)
    args = p.parse_args()
    rows, ident, agg, (n1, n2) = build()
    cal = [r for r in rows if r['method'] == 'cal']
    assert len(cal) == 90
    plotted, ranges, size, gap, stem = draw(agg, args.layout)
    assert len(plotted) == 4 * len(SETTINGS) * len(ORDER)
    if stem != 'predictor_comparison_uniform_certified':
        print(f'preview only ({args.layout}); values not written'); return
    (VALUES / 'archive/predictor_comparison_uniform_certified_values.json').write_text(json.dumps(dict(
        protocol='uniform runs, final round, two-sided A-1d at alpha_s = alpha_f = delta = 0.05 only; per-seed rates '
                 'recomputed from per-state test predictions; mean with min-max over 3 seeds; recall over all 3 seeds '
                 '(all-Abstain runs count as 0); FDR/FOR over seeds with a nonempty declaration set',
        encodings=dict(filled='3 contributing seeds', hollow_with_numeral='1 or 2 contributing seeds (numeral = count; '
                       'no whisker for 1)', en_dash='no seed with a nonempty declaration set (undefined)',
                       gray_line='5% target, error panels only'),
        sources={'soft_four_wsweep.jsonl': hashlib.sha256((ASSETS / 'soft_four_wsweep.jsonl').read_bytes()).hexdigest(),
                 str(certified.SRC): certified.SRC_SHA256},
        code=dict(certificates=str(DETV3 / 'adaptive_roa/adaptive_v2/eval/certificates.py')),
        checks=dict(training_and_eval_identity_groups=len(ident), table1_fm_uniform_cells=n1,
                    fig3_uniform_endpoint_cells=n2, calibrated_seed_runs=len(cal),
                    all_abstain_calibrated_seed_runs=sum(r['all_abstain'] for r in cal)),
        layout=args.layout, page_size=size, axis_ranges=ranges, min_setting_label_gap_pt=round(gap, 2),
        plotted=plotted), indent=1, default=str) + '\n')
    print(f'checks: identity groups {len(ident)}, Table 1 {n1}/30, Fig 3 {n2}/24; label gap {gap:.1f} pt; {size}')


if __name__ == '__main__':
    main()
