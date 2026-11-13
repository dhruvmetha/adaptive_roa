"""Compact main-text version of Figure S3 (predictor_comparison_uniform): paper figure predictor_comparison_compact.

Sumanth, 2026-11-13: the appendix figure (six settings, five predictors) holds an important result but is slow to
read. This version keeps the three main-text predictors (Unif-TERA, Unif-BNN-Label, Unif-TERA-BNN, as Table 4) and
two settings that carry the point:
  spatial quadrotor LQR  similar validation-fitted failure recall, very different certified recall;
  cartpole RL            the counterexample, where the binary-outcome predictor stays competitive or better.
Columns are the settings; the Success row plots R_s against FDR, the Fail row R_f against FOR, in percent. Open
markers = thresholds fitted on D_val at w = 0.2, filled = two-sided certification at alpha_s = alpha_f = delta = 0.05,
a line joins each predictor's two points. Marker = three-seed mean, bars = seed min to max on both axes. Error on the
x axis keeps a higher-recall point from reading as an improvement at equal error. A numeral beside a marker gives the
seeds with a defined error rate when fewer than three; the gray line is the 5% error limit. Both are explained in
the caption, not the legend.

Data and every check come from predictor_comparison_data.build(), the same aggregates as Figure S3; this script
checks each plotted mean and bar end against values/predictor_comparison_uniform_values.json, which it never writes.
Writes plots/predictor_comparison_compact.{pdf,png}; --paper copies the PDF into the paper clone.
"""
import argparse, json, shutil
import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, MultipleLocator
from common import (VALUES, PLOTS, PAPER_FIGS, FS, LINE, LS, LW, REC_S, REC_F, ERR_S, ERR_F, marker_kw, page_size,
                    paper_legend_kw, style_rc)
from predictor_comparison_data import build

STEM = 'predictor_comparison_compact'
WIDTH = 2.40          # in: the paper wraps it at 0.4\textwidth (Sumanth, 916d36a), so it prints at scale 1.00
SETTINGS = [('q3d-lqr', 'Spatial quadrotor LQR'), ('cp-ppo', 'Cartpole RL')]   # Figure 7's titles; the appendix's are too wide
MODELS = [('FM', 'Unif-TERA'), ('BNN final-state', 'Unif-TERA-BNN'), ('BNN classifier', 'Unif-BNN-Label')]   # legend order (Sumanth)
SIDES = [('fdr', 'rec', ERR_S + r' (\%)', REC_S + r' (\%)', LINE['success']),
         ('for', 'fail_rec', ERR_F + r' (\%)', REC_F + r' (\%)', LINE['fail'])]
MS, MEW, HALO, BAR, LINK = 4.6, .6, .5, .45, .7
BAR_ALPHA = .4                                   # bars recede behind the markers
# Second column has no y-axis label (Sumanth, 2026-11-13: the first column's label names the row), so the column
# gap only holds tick labels and the subplots take the freed width.
# Text just under the 9 pt \footnotesize captions (Sumanth, 2026-11-13), at the figure's existing 2.40 x 2.76 in.
F = dict(tick=8, legend=8.5, label=8.5, title=8.5, annotation=8)
RC = {'font.size': F['label'], 'axes.titlesize': F['title'], 'axes.labelsize': F['label'],
      'xtick.labelsize': F['tick'], 'ytick.labelsize': F['tick'], 'legend.fontsize': F['legend']}
HEIGHT = 2.76
LEGEND_ROW = .155                                # in per legend row at 8.5 pt
LEFT, BOTTOM, COLGAP, ROWGAP, RIGHT = .405, .31, .235, .345, .02    # in
TOP = .035 + 2 * LEGEND_ROW + .135               # two legend rows, then the panel titles
AX_W, AX_H = (WIDTH - LEFT - COLGAP - RIGHT) / 2, (HEIGHT - BOTTOM - ROWGAP - TOP) / 2   # subplot size (in)


def panel(ax, agg, setting, xm, ym, color):
    """One subplot; returns the drawn points and every plotted observation (means and bar ends)."""
    drawn, xs, ys = [], [], []
    for model, name in MODELS:
        pts = {}
        for method in ('val', 'cal'):
            a = agg[setting, model, method]
            x, y = a[xm], a[ym]
            assert x['n'] > 0 and y['n'] == 3, (setting, model, method, xm, x['n'], y['n'])
            bars = ax.errorbar(x['mean'], y['mean'], xerr=[[x['mean'] - x['min']], [x['max'] - x['mean']]],
                               yerr=[[y['mean'] - y['min']], [y['max'] - y['mean']]], fmt='none', ecolor=color,
                               elinewidth=BAR, capsize=0, zorder=2, alpha=BAR_ALPHA)
            kw = marker_kw(name, color, size=MS, edge=MEW)
            kw['markerfacecolor'] = color if method == 'cal' else 'white'
            mk, = ax.plot([x['mean']], [y['mean']], ls='none', zorder=5 if method == 'cal' else 4, **kw,
                          path_effects=[patheffects.withStroke(linewidth=MEW + 2 * HALO, foreground='white')])
            if x['n'] < 3:
                ax.annotate(str(x['n']), (x['mean'], y['mean']), xytext=(MS / 2 + 1.2, MS / 2 - .5),
                            textcoords='offset points', fontsize=F['annotation'], ha='left', va='bottom', zorder=6)
            drawn.append(dict(setting=setting, model=model, method=method, xm=xm, ym=ym, marker=mk, bars=bars))
            pts[method] = (x['mean'], y['mean'])
            xs += [x['min'], x['max']]; ys += [y['min'], y['max']]
        (x0, y0), (x1, y1) = pts['val'], pts['cal']
        ax.plot([x0, x1], [y0, y1], color=color, lw=LINK, alpha=.5, zorder=1, solid_capstyle='butt')
    return drawn, xs, ys


def pad(lo, hi, frac=.08):
    d = (hi - lo) * frac or 1.0
    return lo - d, hi + d


def draw(agg):
    H = HEIGHT
    fig = plt.figure(figsize=(WIDTH, H))
    drawn, axes = [], {}
    for c, (s, title) in enumerate(SETTINGS):
        for r, (xm, ym, xl, yl, color) in enumerate(SIDES):
            x0 = LEFT + c * (AX_W + COLGAP)
            y0 = BOTTOM + (1 - r) * (AX_H + ROWGAP)
            ax = fig.add_axes([x0 / WIDTH, y0 / H, AX_W / WIDTH, AX_H / H])
            ax.axvline(5, color=LINE['ref'], lw=LW['ref'], ls=LS['ref'], zorder=0)
            d, xs, ys = panel(ax, agg, s, xm, ym, color)
            drawn += d
            ax.set_xlim(*pad(min(xs + [0]), max(xs + [5.0])))
            ax.set_ylim(*pad(min(ys), max(ys)))
            assert ax.get_xlim()[0] <= min(xs) and max(xs) <= ax.get_xlim()[1]
            ax.set_xlabel(xl, labelpad=1)
            if c == 0:
                ax.set_ylabel(yl, labelpad=1.5)
            # Error axes: labelled ticks at multiples of 5 so the 5% limit line carries its own label (Sumanth),
            # unlabelled minor ticks every 1%.
            ax.xaxis.set_major_locator(MultipleLocator(5)); ax.xaxis.set_minor_locator(MultipleLocator(1))
            ax.tick_params(axis='x', which='minor', length=1.5, width=.5)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10], min_n_ticks=3))
            if r == 0:
                ax.set_title(title, pad=3)
            axes[s, r] = ax
    shapes = [Line2D([], [], ls='none', label=name,
                     **(marker_kw(name, 'black', size=MS, edge=MEW) | {'markerfacecolor': 'white'}))
              for _, name in MODELS]
    ev = [Line2D([], [], ls='none', marker='o', markersize=MS, markeredgewidth=MEW, markeredgecolor='black',
                 markerfacecolor='white'),
          Line2D([], [], ls='none', marker='o', markersize=MS, markeredgewidth=MEW, markeredgecolor='black',
                 markerfacecolor='black')]
    labels = ['fitted', 'certified']               # the caption gives w and alpha
    top = 1 - .035 / H
    tight = paper_legend_kw(F['legend']) | dict(columnspacing=.8, handletextpad=.35)
    # Three names do not fit one row at 8.5 pt. Two rows, in the order Sumanth set: Unif-TERA, Unif-TERA-BNN /
    # Unif-BNN-Label beside the fitted / certified key (a third row wasted space).
    row = LEGEND_ROW / H
    legs = [fig.legend(handles=shapes[:2], ncol=2, bbox_to_anchor=(.5, top), bbox_transform=fig.transFigure, **tight),
            fig.legend(handles=[shapes[2]] + ev, labels=[shapes[2].get_label()] + labels, ncol=3,
                       bbox_to_anchor=(.5, top - row), bbox_transform=fig.transFigure, **tight)]
    fit_check(fig, axes, legs)
    return fig, drawn


def fit_check(fig, axes, legs):
    """Legends inside the page; titles and legends overlap nothing."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    page = fig.bbox
    boxes = [l.get_window_extent(r) for l in legs] + [ax.title.get_window_extent(r) for ax in axes.values() if ax.get_title()]
    for b in boxes:
        assert page.x0 <= b.x0 and b.x1 <= page.x1 and b.y1 <= page.y1, ('outside the page', b, page)
    for i, a in enumerate(boxes):
        for b in boxes[i + 1:]:
            assert not a.overlaps(b), ('overlap', a, b)


def check(drawn):
    """Every plotted mean and bar end equals the stored Figure S3 aggregate."""
    stored = {(a['setting'], a['model'], a['method']): a
              for a in json.loads((VALUES / 'predictor_comparison_uniform_values.json').read_text())['aggregates']}
    for d in drawn:
        s = stored[d['setting'], d['model'], d['method']]
        x, y = s[d['xm']], s[d['ym']]
        assert d['marker'].get_xdata()[0] == x['mean'] and d['marker'].get_ydata()[0] == y['mean'], d
        segs = [seg for lc in d['bars'].lines[2] for seg in lc.get_segments()]
        ends = sorted({round(v, 9) for sg in segs for v in (*sg[:, 0], *sg[:, 1])})
        for v in (x['min'], x['max'], y['min'], y['max']):
            assert any(abs(v - e) < 1e-9 for e in ends), (d['setting'], d['model'], d['method'], v)
    return len(drawn)


def main(paper):
    rows, ident, agg, _ = build()
    with plt.rc_context(style_rc(**RC)):
        fig, drawn = draw(agg)
        n = check(drawn)
        PLOTS.mkdir(exist_ok=True)
        pdf, png = PLOTS / f'{STEM}.pdf', PLOTS / f'{STEM}.png'
        fig.savefig(pdf, metadata={'CreationDate': None})
        fig.savefig(png, dpi=300)
        plt.close(fig)
    w, h = page_size(pdf)
    assert abs(w - WIDTH) <= .01, (w, WIDTH)
    print(f'{STEM}: {w} x {h} in, {n} points checked against values/predictor_comparison_uniform_values.json')
    if paper:
        shutil.copyfile(pdf, PAPER_FIGS / f'{STEM}.pdf')
        print(f'{STEM} -> paper figures/results/{STEM}.pdf')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument('--paper', action='store_true', help='copy the PDF into the paper clone figures/results/')
    main(p.parse_args().paper)
