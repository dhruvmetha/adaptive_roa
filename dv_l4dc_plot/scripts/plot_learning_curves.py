"""Learning curves, uniform vs adaptive FM: paper figure adaptive_learning_a1d (certified, Fig 3).

Six systems in two rows (LQR, PPO) of three. Each system has a recall panel (Success and Failure
recall, solid) over a shorter error panel (FDR and FOR, split.DASHDOT), about 0.63 of its height. Blue
is the Success pair (Success recall, FDR), orange the Failure pair. The acquisition scheme is marker and
shade: adaptive (U_H, rho = 0.75) in full color with circles, uniform 30% toward white with squares.
One legend row names the six encodings: Success recall, Failure recall, FDR, FOR, TERA, Unif-TERA (from
method-names.tex).
A thin solid gray 5% line sits on every error panel, labelled once in the top-left one. Every x axis is
the training budget in trajectories, labelled as counts (Dhruv, 2026-11-12): under 1000 as integers (100, 500),
1000 and up with k (1k, 1.5k, 12k), with tick labels only under the error panels and one shared x label
"Training trajectories". Data x is stored in thousands internally; only the labels change. y ranges are shared within each row (LQR, PPO) and derived from the
data by row_axes(), so only the leftmost column carries y tick labels and the y labels.

Line = three-seed mean, band = min to max over seeds (common.spread; never SD), drawn only where at
least 2 seeds contribute. FDR/FOR use only seeds with a nonempty predicted class: a hollow marker means
fewer than 3 such seeds, a gap means none. Fails if a marker (edge stroke included) or band leaves its
panel, if the 5% line would leave an error axis, if the shared x label's text box overlaps the tick
labels above it, or if the saved PDF is not 6.0 in wide (5.95-6.05) and 3.95-4.2 in tall.

  a1d        THE PAPER FIGURE: A-1d two-sided calibration at delta=0.05, test rates (epochs.jsonl c6c_*)
  empirical  preview only since 2026-11-12, when Dhruv kept only the certified curves: soft rule fit on
             validation at w=0.2 (soft_learning_w02_complete.json). No paper output.

Since 2026-11-13 the paper figure is the common.py paper-style draw at print width, plots/adaptive_learning_a1d.pdf
(split.py restyle block); it fails unless its numbers equal the stored values JSON. A plain run first rebuilds values/
and the old-style draws (the a1d one now in plots/legacy/), then the paper figure. --restyle redraws only the paper
figure; --paper copies it into the paper clone.
"""
import hashlib, json, math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgb
from matplotlib.legend_handler import HandlerTuple
from common import ASSETS, SCORED, VALUES, SETTINGS, BLUE, ORANGE, RECALL, ERROR, RC, method_names, save, spread
from make_tables import macro_name
from split import draw, assert_markers_inside, assert_bands_inside, REF_SOLID, DASHDOT
from split import restyle_args, split_figure_restyled
from common import install_paper

PAPER_STEM = {'a1d': 'adaptive_learning_a1d', 'empirical': None}
# Local titles (common.TITLES carries dimensions and stays as other figures use it).
TITLE = {'pen-lqr': 'Pendulum LQR', 'cp-lqr': 'Cartpole LQR', 'q3d-lqr': 'Spatial quadrotor LQR',
         'cp-ppo': 'Cartpole PPO', 'q2d-ppo': 'Planar quadrotor PPO', 'q3d-ppo': 'Spatial quadrotor PPO'}
# x tick positions in thousands of training trajectories (the plotted x), inside each system's budget range.
XTICKS = {'pen-lqr': [.1, .2, .3, .4], 'cp-lqr': [.5, 1, 1.5, 2], 'q3d-lqr': [10, 20, 30],
          'cp-ppo': [.5, 1, 1.5, 2], 'q2d-ppo': [2, 4, 6, 8, 10, 12], 'q3d-ppo': [10, 20, 30, 40]}
# Uniform is its color mixed 30% toward white. At 45% (until 2026-11-12) it washed out over the seed
# bands; at 30% it reads over them and still sits apart from the full color. Marker shape tells too.
UNIFORM_TINT = .30
# (arm, marker, tint, size in pt); uniform drawn first. The square is 2.7pt so it carries about the
# ink of a 3pt circle instead of 27% more.
ARMS = [('uniform', 's', UNIFORM_TINT, 2.7), ('U_H', 'o', 0., 3.0)]
# Arm names follow the paper's method-names.tex (Dhruv, 2026-11-12, replacing Adaptive / Uniform): the U_H arm is the
# 'method' role (\\name, TERA) and the uniform arm the 'uniform data' role (\\nameUnif, Unif-TERA), via make_tables.macro_name.
_NAMES = method_names()
ARM_LABEL = {'U_H': _NAMES[macro_name('method', 'FM')], 'uniform': _NAMES[macro_name('uniform data', 'FM')]}
MEW = .6                    # marker edge, pt; hollow markers keep a ~2.1-2.4pt hole
BAND_ALPHA = .09
FONT = RC | {'axes.titlesize': 8.5, 'axes.titleweight': 'normal', 'axes.labelsize': 8, 'legend.fontsize': 8,
             'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
             'xtick.major.width': .65, 'ytick.major.width': .65, 'xtick.major.pad': 1.8, 'ytick.major.pad': 1.8}
# Layout in inches at the printed size. save() crops to the drawn content plus .04 in a side, so the
# canvas is the content: the PDF comes out 6.0 in wide.
FIG_W = 5.92
LEFT, RIGHT, COL_GAP = .385, .07, .15        # left holds the y label and tick labels
LEGEND_H, TITLE_H = .19, .16                 # legend row; title band above each recall panel
REC_H, ERR_H, PAIR_GAP = .88, .56, .04       # recall panel, error panel, gap between them
TICK_H, ROW_GAP, XLABEL_H = .15, .05, .2     # x tick labels; extra gap between rows; shared x label
# The x label's text box must clear the tick-label boxes above it (0.31 pt here). Boxes hold empty
# descent, so the ink gap is 4.7 pt; at .155 in the boxes overlapped by 2.9 pt and ink cleared 1.4 pt.
XLABEL_CLEAR = 0.


def tint(color, f):
    """Mix a color with white by fraction f (0 = unchanged)."""
    return tuple(c + (1 - c) * f for c in to_rgb(color))


def count_label(n):
    """A trajectory count as the paper prints it everywhere: under 1000 as an integer (100, 500), 1000 and up
    with k (1k, 1.5k, 12k, 40k). Never "0.5k"."""
    n = int(round(n))
    return str(n) if n < 1000 else f'{n / 1000:g}k'


def tick_labels(ticks):
    """Tick positions are in thousands of trajectories; labels are counts (count_label)."""
    return [count_label(t * 1000) for t in ticks]


def row_axes(points):
    """Per row (0 = LQR, 1 = PPO): recall and error (ylim, ticks) covering every mean and band in that row,
    padded so markers stay inside, with the error top above 5 so the 5% line shows. Two floors moved on
    2026-11-12 so a marker at 0%, edge stroke included ((3 + 0.6) / 2 = 1.8 pt), stays inside: error
    -0.05 x top (was -0.03) on the 0.56 in error panel, recall -4 (was -3) on the 0.88 in recall panel."""
    out = {}
    for row, settings in enumerate([SETTINGS[:3], SETTINGS[3:]]):
        pts = [p for p in points if p['setting'] in settings and p['n']]
        rec = [p for p in pts if p['metric'] in ('rec', 'fail_rec')]
        err = [p for p in pts if p['metric'] in ('fdr', 'for')]
        rec_lo = min(min(p['mean'], p['min'] if p['n'] > 1 else p['mean']) for p in rec)
        rec_ax = ((-4, 103), [0, 50, 100]) if rec_lo < 37 else ((38, 102), [40, 70, 100]) if rec_lo < 47 \
            else ((48, 102), [50, 75, 100])
        err_hi = max(max(p['mean'], p['max'] if p['n'] > 1 else p['mean']) for p in err)
        top = max(6, math.ceil((err_hi + .3) / 2) * 2)
        out[row] = dict(rec=rec_ax, err=((-top / 20, top), [t for t in (0, 5, 10, 15, 20) if t <= top]))
    return out


def compute(protocol):
    """The plotted numbers of one protocol: (stem, sources, rows, summaries, data, row axes). Both the paper path
    (generate) and the --restyle path (generate_restyled) draw from this, so they draw the same data."""
    softpath = ASSETS / 'soft_learning_w02_complete.json'
    soft = json.loads(softpath.read_text()); assert len(soft) == 396
    meta = {r['name']: r for r in soft}
    if protocol == 'empirical':
        src = [softpath]; raw = soft; stem = 'adaptive_learning'
        rows = [dict(name=r['name'], setting=r['setting'], seed=r['seed'], epoch=r['epoch'],
                     train=r['train'], arm=r['acq'], **r['test']) for r in raw]
    else:
        ep = SCORED / 'epochs.jsonl'
        src = [softpath, ep]; raw = [r for l in ep.read_text().splitlines() if (r := json.loads(l))['name'] in meta]
        assert len(raw) == 396 and all(r['c6c_src'] == 'A-1d at delta' for r in raw)
        stem = 'adaptive_learning_a1d'
        rows = [{k: r[k] for k in ['name', 'setting', 'seed', 'epoch', 'train', 'arm']} |
                {k: r[v] for k, v in [('rec', 'c6c_rec'), ('fail_rec', 'c6c_fail_rec'), ('fdr', 'c6c_fp'),
                                      ('for', 'c6c_ferr'), ('S', 'c6c_S'), ('F', 'c6c_F')]} for r in raw]
    assert {(r['name'], r['seed'], r['epoch'], r['train']) for r in rows} == \
           {(r['name'], r['seed'], r['epoch'], r['train']) for r in soft}
    for r in rows:
        assert (r['fdr'] is None) == (r['S'] == 0) and (r['for'] is None) == (r['F'] == 0)
    summaries = []

    def summarize(setting, arm, metric):
        rr = [r for r in rows if (r['setting'], r['arm']) == (setting, arm)]; assert len(rr) == 33
        x = []; mean = []; lo = []; hi = []; n = []
        for epoch in range(11):
            er = [r for r in rr if r['epoch'] == epoch]
            assert len(er) == 3 and len({r['seed'] for r in er}) == 3 and len({r['train'] for r in er}) == 1
            v = [100 * r[metric] for r in er if r[metric] is not None]
            m, a, b = spread(v)
            x.append(er[0]['train']); n.append(len(v))
            mean.append(m if v else float('nan'))
            lo.append(a if len(v) > 1 else float('nan')); hi.append(b if len(v) > 1 else float('nan'))
            summaries.append(dict(setting=setting, arm=arm, metric=metric, epoch=epoch, train=x[-1], n=n[-1],
                                  mean=m, min=a, max=b))
        return tuple(np.array(v, dtype=float) for v in (x, mean, lo, hi, n))

    data = {(s, arm, m): summarize(s, arm, m) for s in SETTINGS for arm, *_ in ARMS for m, _ in RECALL + ERROR}
    axes_by_row = row_axes(summaries)
    return stem, src, rows, summaries, data, axes_by_row


def generate(protocol, paper):
    stem, src, rows, summaries, data, axes_by_row = compute(protocol)

    row_h =TITLE_H + REC_H + PAIR_GAP + ERR_H + TICK_H
    fig_h = LEGEND_H + 2 * row_h + ROW_GAP + XLABEL_H
    panel_w = (FIG_W - LEFT - RIGHT - 2 * COL_GAP) / 3

    def rect(x0, y_top, h):
        """Axes rect in figure fractions from inches (x0 from the left, y_top from the top)."""
        return [x0 / FIG_W, (fig_h - y_top - h) / fig_h, panel_w / FIG_W, h / fig_h]

    with plt.rc_context(FONT):
        fig = plt.figure(figsize=(FIG_W, fig_h))
        left_axes, bottom_axes = [], []
        for si, setting in enumerate(SETTINGS):
            r, c = divmod(si, 3)
            x0 = LEFT + c * (panel_w + COL_GAP)
            y_top = LEGEND_H + TITLE_H + r * (row_h + ROW_GAP)
            rec = fig.add_axes(rect(x0, y_top, REC_H))
            err = fig.add_axes(rect(x0, y_top + REC_H + PAIR_GAP, ERR_H), sharex=rec)
            rec.set_title(TITLE[setting], pad=3.5)
            ra = axes_by_row[r]
            for ax, pairs, ls in [(rec, RECALL, '-'), (err, ERROR, DASHDOT)]:
                ax.grid(axis='y', alpha=.2, lw=.45); ax.set_axisbelow(True)
                means, bands = {arm: [] for arm, *_ in ARMS}, []
                for metric, color in pairs:
                    for arm, marker, f, ms in ARMS:
                        x, mean, lo, hi, n = data[setting, arm, metric]
                        draw(ax, x / 1000, mean, lo, hi, n, tint(color, f), ls=ls, marker=marker, ms=ms, mew=MEW,
                             band_alpha=BAND_ALPHA)
                        means[arm].append(mean); bands.append((lo, hi))
                key = 'rec' if ax is rec else 'err'
                ax.set_ylim(*ra[key][0]); ax.set_yticks(ra[key][1])
                for arm, _, _, ms in ARMS:  # ms + mew: the edge stroke reaches mew / 2 past the marker
                    assert_markers_inside(fig, ax, np.concatenate(means[arm]), ms=ms + MEW)
                assert_bands_inside(ax, bands)
                if c:  # same row, same y range as the leftmost panel: no repeated tick labels
                    ax.tick_params(labelleft=False)
            ticks = XTICKS[setting]
            assert x[0] / 1000 <= ticks[0] and ticks[-1] <= x[-1] / 1000, (setting, ticks)
            err.set_xticks(ticks, tick_labels(ticks))
            rec.tick_params(labelbottom=False)
            assert ra['err'][0][1] > 5  # the 5% line must stay visible
            err.axhline(5, zorder=0, **REF_SOLID)
            if si == 0:  # the one direct label of the 5% line, under it at the right end
                err.annotate('5% target', xy=(1, 5), xycoords=('axes fraction', 'data'), xytext=(-1.5, -1.5),
                             textcoords='offset points', ha='right', va='top', fontsize=7, color=REF_SOLID['color'])
            if c == 0:
                rec.set_ylabel('Recall (%)'); err.set_ylabel('Error (%)'); left_axes += [rec, err]
            if r == 1:
                bottom_axes.append(err)
        fig.align_ylabels(left_axes)
        x_mid = (LEFT + (FIG_W - RIGHT)) / 2 / FIG_W
        xlabel = fig.text(x_mid, 0, 'Training trajectories', ha='center', va='bottom',
                          fontsize=FONT['axes.labelsize'])
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        tick_y0 = min(t.get_window_extent(renderer).y0 for ax in bottom_axes for t in ax.get_xticklabels())
        clear = (tick_y0 - xlabel.get_window_extent(renderer).y1) * 72 / fig.dpi
        assert clear >= XLABEL_CLEAR, f'x label {clear:.2f} pt under the tick labels'
        handles = [Line2D([], [], color=c, ls=ls, lw=1.15)
                   for c, ls in [(BLUE, '-'), (ORANGE, '-'), (BLUE, DASHDOT), (ORANGE, DASHDOT)]]
        # A scheme is its marker in both of its shades, side by side, as the curves draw it.
        handles += [tuple(Line2D([], [], color=tint(c, f), ls='none', marker=m, ms=ms, mew=MEW) for c in (BLUE, ORANGE))
                    for _, m, f, ms in ARMS[::-1]]
        labels = ['Success recall', 'Failure recall', 'FDR', 'FOR'] + [ARM_LABEL[a] for a, *_ in ARMS[::-1]]
        fig.legend(handles, labels, handler_map={tuple: HandlerTuple(ndivide=None, pad=.6)}, loc='upper center',
                   bbox_to_anchor=(x_mid, 1), ncol=6, frameon=False, borderpad=0, borderaxespad=0,
                   handlelength=2.2, handletextpad=.5, columnspacing=1.3)
        size = save(fig, stem, paper_stem=PAPER_STEM[protocol], paper=paper)
    assert 5.95 <= size['width_in'] <= 6.05 and 3.95 <= size['height_in'] <= 4.2, size
    print(f'  x label box {clear:.2f} pt under the tick-label boxes')
    VALUES.mkdir(exist_ok=True)
    (VALUES / f'{stem}_values.json').write_text(json.dumps(dict(
        protocol=protocol, source_hashes={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in src},
        row_axes=axes_by_row, page_size=size, points=summaries), indent=2) + '\n')
    partial = sum(1 for p in summaries if 0 < p['n'] < 3); empty = sum(1 for p in summaries if p['n'] == 0)
    print(f'  {protocol}: {len(rows)} checkpoints, row axes {axes_by_row}, cells with 1-2 seeds {partial}, '
          f'with none {empty}')


# --restyle (dm1487, 2026-11-13): Figure 6 in the common.py paper style at its 4.20 in print width. The panel shapes
# are the paper figure's: recall height / panel width and error height / recall height as above.
RESTYLE_ASPECT = REC_H / ((FIG_W - LEFT - RIGHT - 2 * COL_GAP) / 3)
RESTYLE_ERR_RATIO = ERR_H / REC_H


def generate_restyled():
    """Figure 6 (a1d) in the paper style at PRINT_WIDTH['adaptive_learning_a1d'] into plots/ (the paper figure).
    Colour = the metric's side (Success recall and FDR gold, Failure recall and FOR purple), recall solid, error
    dash-dot; the arms differ only by marker (TERA filled circle, Unif-TERA open square, common.METHOD_MARKER), and
    a faded marker (split.partial_kw) means 1-2 seeds contribute. Draws from compute('a1d') and fails unless its points and
    source hashes equal the stored values/adaptive_learning_a1d_values.json. Writes nothing to values/ or the paper."""
    stem, src, rows, summaries, data, axes_by_row = compute('a1d')
    stored = json.loads((VALUES / f'{stem}_values.json').read_text())
    assert json.loads(json.dumps(summaries)) == stored['points'], 'restyle points differ from values/'
    assert {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in src} == stored['source_hashes']

    def series(arm):
        def f(setting, metric):
            x, mean, lo, hi, n = data[setting, arm, metric]
            return x / 1000, mean, lo, hi, n
        return f

    xticks = {}
    for s in SETTINGS:
        x = data[s, 'U_H', 'rec'][0] / 1000
        assert all((data[s, a, 'rec'][0] == data[s, 'U_H', 'rec'][0]).all() for a, *_ in ARMS)
        assert x[0] <= XTICKS[s][0] and XTICKS[s][-1] <= x[-1], (s, XTICKS[s])
        xticks[s] = (XTICKS[s], tick_labels(XTICKS[s]))
    row_base = {}
    for row, settings in enumerate([SETTINGS[:3], SETTINGS[3:]]):
        pts = [p for p in summaries if p['setting'] in settings and p['n'] and p['metric'] in ('rec', 'fail_rec')]
        lo = min(min(p['mean'], p['min'] if p['n'] > 1 else p['mean']) for p in pts)
        hi = max(max(p['mean'], p['max'] if p['n'] > 1 else p['mean']) for p in pts)
        rec_ticks, err_ticks = axes_by_row[row]['rec'][1], axes_by_row[row]['err'][1]
        row_base[row] = dict(rec=(min(rec_ticks[0], lo), max(rec_ticks[-1], hi), rec_ticks),
                             err=(0, axes_by_row[row]['err'][0][1], err_ticks))
    out = split_figure_restyled(
        stem, arms=[(series(arm), ARM_LABEL[arm]) for arm, *_ in ARMS], xticks=xticks,
        xlabel='Training trajectories', titles=[TITLE[s] for s in SETTINGS], row_base=row_base,
        metric_labels=['Success recall', 'Failure recall', 'FDR', 'FOR'],
        legend_methods=[ARM_LABEL[a] for a, *_ in ARMS[::-1]], ylabels=('Recall (%)', 'Error (%)'),
        rec_aspect=RESTYLE_ASPECT, err_ratio=RESTYLE_ERR_RATIO, band_alpha=BAND_ALPHA, ref_label='5% target')
    g = out['geometry']
    print(f"  panels {g['panel_w']:.3f} in wide, recall {g['rec_h']:.3f} in, error {g['err_h']:.3f} in; "
          f"legend {out['legend_ncol']} columns; layout settled in {g['iterations']} passes")
    print(f"  outer whitespace (in): {', '.join(f'{k} {v:.3f}' for k, v in out['edges'].items())}")
    print(f"  y limits: {out['ylims']}")
    print(f"  {len(summaries)} points equal values/{stem}_values.json; cells with 1-2 seeds "
          f"{sum(1 for p in summaries if 0 < p['n'] < 3)}, with none {sum(1 for p in summaries if p['n'] == 0)}")
    return out


if __name__ == '__main__':
    args = restyle_args(__doc__)
    if not args.restyle:
        for protocol in ['empirical', 'a1d']:
            generate(protocol, False)       # values/ and the old-style draws; a1d's goes to plots/legacy/
    generate_restyled()
    if args.paper:
        install_paper('adaptive_learning_a1d')
