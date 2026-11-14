"""Predictor comparison on identical uniform training data: paper figure predictor_comparison_uniform.

Data and every check come from predictor_comparison_data.py (identical training trajectories and evaluation sets,
recomputed counts, Table 1 and Fig 3 cross-checks); this script only draws and writes the outputs.

Per system a Success subplot (x = FDR, y = R_s) over a Fail subplot (x = FOR, y = R_f), both in percent. One color
per predictor; circles = thresholds fitted on D_val at w=0.2, squares = two-sided A-1d calibration at
alpha_s = alpha_f = delta = 0.05; a faint line joins a predictor's two points. Marker = three-seed mean, bars = min
to max over seeds on both axes (common.spread). Recall and abstention use all three seeds; FDR/FOR only seeds with a
nonempty declaration set, and a hollow marker means fewer than three such seeds. A run with no declaration of a kind
has an undefined error rate, never zero: it cannot be placed on that subplot and is listed under the system. Axes
cover every observation. The thin solid gray vertical line at 5% is the certification target. Wording follows the
paper's rule (WRITING_GUIDE, paper commit 811e036): certified for the procedure and its outputs, calibration only for
the independent data.

Writes plots/predictor_comparison_uniform.{pdf,png}, values/predictor_comparison_uniform_per_seed.csv (every seed,
both methods, with counts, abstention, thresholds and the calibration upper bounds kept apart from test rates),
values/predictor_comparison_uniform_values.json (aggregates, checks, provenance). --paper copies the PDF into the
paper clone as figures/results/predictor_comparison_uniform.pdf.

Since 2026-11-13 the paper figure is the same data in the paper style of common.py (colour = Success or Fail
subplot, marker shape = predictor, open = validation-fitted, filled = certified, a numeral = seeds with a defined error
rate when fewer than three), plots/predictor_comparison_uniform.{pdf,png}; it checks that the plotted means and
min-max bars equal the stored values/predictor_comparison_uniform_values.json. A plain run first rebuilds values/ and
the old-style draw (now in plots/legacy/), then the paper figure. --restyle redraws only the paper figure; --paper
copies it into the paper clone.
"""
import argparse, csv, hashlib, io, json, math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import patheffects
from matplotlib.legend_handler import HandlerBase
from matplotlib.path import Path
from matplotlib.text import Text
from matplotlib.ticker import MaxNLocator
from matplotlib.transforms import Bbox
from common import VALUES, SETTINGS, TITLES, REF, RC, ASSETS, save, install_paper, REC_S, REC_F, ERR_S, ERR_F, tt
from common import (HERE, PLOTS, FS, LINE, LS, LW, METHOD_MARKER, PRINT_WIDTH, marker_kw, page_size, paper_legend_kw,
                    style_rc)
import certified
from split import REF_SOLID
from predictor_comparison_data import build, MODELS, METRICS, DETV3

COLORS = {'FM': '#0072B2', 'MLP final-state': '#E69F00', 'BNN final-state': '#009E73',
          'MLP classifier': '#D55E00', 'BNN classifier': '#CC79A7'}
LEGEND_NAME = {m: tex for m, tex, _ in MODELS}
PLAIN_NAME = {m: plain for m, _, plain in MODELS}
MARK = {'val': 'o', 'cal': 's'}
MS = 3.6


def panel(ax, agg, setting, xm, ym):
    """One scatter subplot; returns (all x observations, all y observations) for the range check."""
    xs, ys = [], []
    for model, _, _ in MODELS:
        pts = {}
        for method in ('val', 'cal'):
            a = agg[setting, model, method]
            if a[xm]['n'] == 0:
                continue
            x, y = a[xm], a[ym]
            pts[method] = (x['mean'], y['mean'])
            hollow = x['n'] < 3
            c = COLORS[model]
            ax.errorbar(x['mean'], y['mean'], xerr=[[x['mean'] - x['min']], [x['max'] - x['mean']]],
                        yerr=[[y['mean'] - y['min']], [y['max'] - y['mean']]], fmt='none', ecolor=c, elinewidth=.6,
                        capsize=0, alpha=.75, zorder=2)
            ax.plot(x['mean'], y['mean'], ls='none', marker=MARK[method], ms=MS, mfc='white' if hollow else c,
                    mec='#333333' if not hollow else c, mew=.45 if not hollow else .8, zorder=4)
            xs += [x['min'], x['max']]; ys += [y['min'], y['max']]
        if len(pts) == 2:
            (x0, y0), (x1, y1) = pts['val'], pts['cal']
            ax.plot([x0, x1], [y0, y1], color=COLORS[model], lw=.5, alpha=.35, zorder=1)
    return xs, ys


def limits(xs, ys):
    xmax = max(6.0, max(xs) * 1.06 + .2)
    ylo = max(-2.0, min(ys) - 0.06 * (100 - min(ys)) - 1)
    return (-0.03 * xmax, xmax), (ylo, 101.5)


def note_parts(agg, setting):
    """Seeds the scatter cannot show for one system, as [(heading, items)] per method: all-Abstain runs, and runs
    with no Success (no FDR) or no Fail (no FOR) declarations beyond those."""
    parts = []
    for method, tag in (('val', 'validation-fitted'), ('cal', 'certified')):
        items = []
        for model, _, plain in MODELS:
            a = agg[setting, model, method]
            k = a['all_abstain_seeds']
            if k:
                items.append(f"{plain} {k}/3")
            for m, word in (('fdr', 'no Success'), ('for', 'no Fail')):
                extra = 3 - a[m]['n'] - k
                if extra:
                    items.append(f"{plain} {word} {extra}/3")
        if items:
            parts.append((f'{tag}, all-Abstain seeds:', items))
    return parts


def abstain_note(agg, setting, width=44):
    """Seeds the scatter cannot show for one system (note_parts), wrapped to fit under the system."""
    parts = note_parts(agg, setting)
    # Break only between items, so a predictor name never separates from its seed count (the Unif- names of
    # method-names.tex are long enough that word wrapping split them).
    lines = []
    for head, items in parts:
        line = head
        for i, item in enumerate(items):
            piece = item + (',' if i < len(items) - 1 else '')
            if len(line) + 1 + len(piece) > width and line.strip():
                lines.append(line); line = '  ' + piece
            else:
                line += ' ' + piece
        lines.append(line)
    return tt('\n'.join(lines))          # wrapped on the plain names, so the markup is not counted


def draw(agg, paper):
    with plt.rc_context(RC | {'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'axes.labelsize': 7}):
        fig = plt.figure(figsize=(6.5, 6.1))
        outer = fig.add_gridspec(2, 3, left=.075, right=.99, top=.885, bottom=.085, wspace=.27, hspace=.5)
        ranges = {}
        for si, (s, title) in enumerate(zip(SETTINGS, TITLES)):
            inner = outer[si // 3, si % 3].subgridspec(2, 1, hspace=.42)
            for row, (xm, ym, xl, yl) in enumerate([('fdr', 'rec', ERR_S + ' (%)', REC_S + ' (%)'),
                                                    ('for', 'fail_rec', ERR_F + ' (%)', REC_F + ' (%)')]):
                ax = fig.add_subplot(inner[row])
                ax.grid(alpha=.2, lw=.45); ax.set_axisbelow(True)
                ax.axvline(5, zorder=0, **REF_SOLID)
                xs, ys = panel(ax, agg, s, xm, ym)
                xl_, yl_ = limits(xs, ys)
                ax.set_xlim(*xl_); ax.set_ylim(*yl_)
                assert xl_[0] <= min(xs) and max(xs) <= xl_[1] and yl_[0] <= min(ys) and max(ys) <= yl_[1]
                ranges[f'{s}/{xm}'] = dict(x=xl_, y=yl_)
                ax.set_xlabel(xl, labelpad=1); ax.set_ylabel(yl, labelpad=1)
                if row == 0:
                    ax.set_title(title, pad=4)
            note = abstain_note(agg, s)
            if note:
                ax.text(-0.12, -0.42, note, transform=ax.transAxes, fontsize=5.6, color='#444444', va='top', ha='left',
                        linespacing=1.15)
        handles = [Line2D([], [], ls='none', marker='o', ms=MS + .4, mfc=COLORS[m], mec='#333333', mew=.45,
                          label=tt(PLAIN_NAME[m])) for m, _, _ in MODELS]
        fig.legend(handles=handles, loc='upper center', ncol=5, frameon=False, bbox_to_anchor=(.53, 1.0),
                   columnspacing=1.4, handletextpad=.3, fontsize=7)
        handles = [Line2D([], [], ls='none', marker='o', ms=MS + .4, mfc='#777777', mec='#333333', mew=.45,
                          label='validation-fitted ($w=0.2$)'),
                    Line2D([], [], ls='none', marker='s', ms=MS + .4, mfc='#777777', mec='#333333', mew=.45,
                           label=r'certified ($\alpha=\delta=0.05$)'),
                    Line2D([], [], **REF_SOLID, label='5% error limit')]
        fig.legend(handles=handles, loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(.53, .972),
                   columnspacing=1.4, handletextpad=.3, fontsize=7)
        size = save(fig, 'predictor_comparison_uniform', paper_stem='predictor_comparison_uniform', paper=paper)
    return size, ranges


# ==== --restyle: the paper style of common.py (dm1487, 2026-11-13) ================================================
# Colour names the subplot's side: Success subplots in LINE['success'], Fail subplots in LINE['fail']. Marker shape
# names the predictor (METHOD_MARKER). Fill names the evaluation: open = validation-fitted, filled = certified.
# METHOD_MARKER draws every Unif- predictor open, and every predictor here is a Unif- one, so fill is free to carry
# the evaluation (r_draw asserts both). A point whose error rate rests on fewer than three seeds carries that seed
# count as a numeral beside its marker. Bars, connectors, notes and the 5% line keep their meaning. The page is laid
# out in inches from measured text, so it prints at PRINT_WIDTH with no scaling, and every check below is a hard one.
R_STEM = 'predictor_comparison_uniform'
R_DIR = PLOTS                            # the paper figure itself since 2026-11-13
R_MS, R_MEW = 4.0, .6                 # marker size and edge (pt)
R_BAR, R_LINK = .5, .5                # min-max bar and val-to-certified connector widths (pt)
R_HALO = .5                           # white ring around every marker and numeral (pt): same-colour filled
                                      # markers that overlap stay separable
R_NUM_GAP = .7                        # marker edge to seed-count numeral (pt)
R_SIDES = [('fdr', 'rec', ERR_S + r' (\%)', REC_S + r' (\%)', LINE['success']),
           ('for', 'fail_rec', ERR_F + r' (\%)', REC_F + r' (\%)', LINE['fail'])]
R_AX_H = .92                          # subplot height (in); the width follows from the page width
R_GAP = dict(col=.13, inner=.07, note=.035, row=.09, legend=.05, legends=.03)   # in
R_INK = (.0115, .0185)                # inked outer margin the layout converges to (in); the hard check is 0.01-0.02
R_PNG_DPI = 300


class _SeedCount:
    """Legend key for the seed-count numeral."""


class _SeedCountHandler(HandlerBase):
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        return [Text(width / 2 - xdescent, height / 2 - ydescent, '2', ha='center', va='center',
                     fontsize=FS['annotation'], color='black', transform=trans)]


def r_panel(ax, agg, setting, xm, ym, color):
    """One restyled subplot. Returns (x observations, y observations, drawn points)."""
    xs, ys, drawn = [], [], []
    for model, _, plain in MODELS:
        pts = {}
        for method in ('val', 'cal'):
            a = agg[setting, model, method]
            x, y = a[xm], a[ym]
            if x['n'] == 0:
                continue
            assert y['n'] == 3, (setting, model, method, ym, y['n'])
            bars = ax.errorbar(x['mean'], y['mean'], xerr=[[x['mean'] - x['min']], [x['max'] - x['mean']]],
                               yerr=[[y['mean'] - y['min']], [y['max'] - y['mean']]], fmt='none', ecolor=color,
                               elinewidth=R_BAR, capsize=0, zorder=2)
            kw = marker_kw(plain, color, size=R_MS, edge=R_MEW)
            kw['markerfacecolor'] = color if method == 'cal' else 'white'
            mk, = ax.plot([x['mean']], [y['mean']], ls='none', zorder=4 if method == 'val' else 5, **kw,
                          path_effects=[patheffects.withStroke(linewidth=R_MEW + 2 * R_HALO, foreground='white')])
            drawn.append(dict(setting=setting, model=model, method=method, xm=xm, ym=ym, n=x['n'], ax=ax,
                              marker=mk, bars=bars))
            pts[method] = (x['mean'], y['mean'])
            xs += [x['min'], x['max']]; ys += [y['min'], y['max']]
        if len(pts) == 2:
            (x0, y0), (x1, y1) = pts['val'], pts['cal']
            ax.plot([x0, x1], [y0, y1], color=color, lw=R_LINK, alpha=.35, zorder=1, solid_capstyle='butt')
    return xs, ys, drawn


def r_limits(xs, ys, mx, my, w_pt, h_pt, partial_y=()):
    """limits() widened until a marker at any mean, edge and halo included, clears the panel edge by 0.5 pt, and
    a seed-count numeral fits above and below every point in partial_y. Only ever widens, so the range still
    covers every observation."""
    (x0, x1), (y0, y1) = limits(xs, ys)
    r = R_MS / 2 + R_MEW / 2 + R_HALO + .5
    rn = R_MS / 2 + R_MEW / 2 + R_NUM_GAP + .75 * FS['annotation'] + .5      # digit height ~ 0.7 em
    for _ in range(4):
        px, py, pn = r / w_pt * (x1 - x0), r / h_pt * (y1 - y0), rn / h_pt * (y1 - y0)
        x0, x1 = min(x0, min(mx) - px), max(x1, max(mx) + px)
        y0, y1 = min(y0, min(my) - py, *(y - pn for y in partial_y)), max(y1, max(my) + py, *(y + pn for y in partial_y))
    return (x0, x1), (y0, y1)


def r_wrap(fig, renderer, parts, max_px):
    """Note lines no wider than max_px, broken only between items so a name never leaves its seed count."""
    probe = fig.text(0, 0, '', fontsize=FS['annotation'])
    def width(s):
        probe.set_text(s)
        return probe.get_window_extent(renderer).width
    lines = []
    for head, items in parts:
        line = head
        for i, item in enumerate(items):
            piece = item + (',' if i < len(items) - 1 else '')
            if width(tt(line + ' ' + piece)) > max_px:
                lines.append(line); line = piece          # usetex crops leading space, so no indent
            else:
                line += ' ' + piece
        lines.append(line)
    fits = all(width(tt(l)) <= max_px for l in lines)
    probe.remove()
    assert fits, 'a single note item is wider than its column'
    return tt('\n'.join(lines))


def r_measure(fig, ax, renderer):
    """Decoration around one subplot in inches: tick labels, axis labels and title beyond each edge."""
    tb, bb, d = ax.get_tightbbox(renderer), ax.bbox, fig.dpi
    return dict(l=(bb.x0 - tb.x0) / d, r=(tb.x1 - bb.x1) / d, t=(tb.y1 - bb.y1) / d, b=(bb.y0 - tb.y0) / d)


def r_layout(fig, axes, notes, legends, margins):
    """Place subplots, notes and legends from measured decorations; returns (page height, subplot width) in in."""
    renderer = fig.canvas.get_renderer()
    m = {k: r_measure(fig, ax, renderer) for k, ax in axes.items()}
    ml, mr, mt, mb = margins
    W = PRINT_WIDTH[R_STEM]
    cols = [[(si, row) for si in (c, c + 3) for row in (0, 1)] for c in range(3)]
    DL = [max(m[k]['l'] for k in ks) for ks in cols]
    DR = [max(m[k]['r'] for k in ks) for ks in cols]
    ax_w = (W - ml - mr - sum(DL) - sum(DR) - 2 * R_GAP['col']) / 3
    x0, x = [], ml
    for c in range(3):
        x0.append(x + DL[c]); x += DL[c] + ax_w + DR[c] + R_GAP['col']
    leg_h = [lg.get_window_extent(renderer).height / fig.dpi for lg in legends]
    y = mt + sum(leg_h) + R_GAP['legends'] * (len(legends) - 1) + R_GAP['legend']
    top, note_top = {}, {}
    for br in (0, 1):
        sis = [br * 3 + c for c in range(3)]
        y += max(m[si, 0]['t'] for si in sis)
        top.update({(si, 0): y for si in sis})
        y += R_AX_H + max(m[si, 0]['b'] for si in sis) + R_GAP['inner'] + max(m[si, 1]['t'] for si in sis)
        top.update({(si, 1): y for si in sis})
        y += R_AX_H + max(m[si, 1]['b'] for si in sis)
        nh = [notes[si].get_window_extent(renderer).height / fig.dpi for si in sis if si in notes]
        if nh:
            y += R_GAP['note']
            note_top.update({si: y for si in sis})
            y += max(nh)
        if br == 0:
            y += R_GAP['row']
    H = y + mb
    fig.set_size_inches(W, H)
    for (si, row), ax in axes.items():
        ax.set_position([x0[si % 3] / W, (H - top[si, row] - R_AX_H) / H, ax_w / W, R_AX_H / H])
    for si, t in notes.items():
        t.set_position(((x0[si % 3] - DL[si % 3]) / W, (H - note_top[si]) / H))
    yy = H - mt
    for lg, h in zip(legends, leg_h):
        lg.set_bbox_to_anchor((.5, yy / H), transform=fig.transFigure)
        yy -= h + R_GAP['legends']
    return H, ax_w, [DL[c] + ax_w + DR[c] for c in range(3)]


def r_ink_margins(png_bytes_or_path, dpi=R_PNG_DPI):
    """Unpainted outer margin (left, right, top, bottom) of a raster, in inches."""
    from PIL import Image
    src = io.BytesIO(png_bytes_or_path) if isinstance(png_bytes_or_path, bytes) else png_bytes_or_path
    a = np.asarray(Image.open(src).convert('L'))
    ink = np.argwhere(a < 250)
    (r0, c0), (r1, c1) = ink.min(0), ink.max(0)
    h, w = a.shape
    return c0 / dpi, (w - 1 - c1) / dpi, r0 / dpi, (h - 1 - r1) / dpi


def r_numerals(fig, renderer, drawn):
    """Seed-count numerals beside markers whose error rate rests on fewer than three seeds. Each takes the first
    spot around its marker that clears every marker, every other numeral and the panel edge (hard), preferring
    spots off the bars, connectors and the 5% line (soft)."""
    k = fig.dpi / 72
    r_px = (R_MS / 2 + R_MEW / 2) * k
    def marker_box(d):
        cx, cy = d['ax'].transData.transform((d['marker'].get_xdata()[0], d['marker'].get_ydata()[0]))
        return Bbox([[cx - r_px, cy - r_px], [cx + r_px, cy + r_px]])
    g = R_MS / 2 + R_MEW / 2 + R_NUM_GAP
    spots = [(g, 0, 'left', 'center'), (-g, 0, 'right', 'center'), (0, g, 'center', 'bottom'),
             (0, -g, 'center', 'top'), (.8 * g, .8 * g, 'left', 'bottom'), (.8 * g, -.8 * g, 'left', 'top'),
             (-.8 * g, .8 * g, 'right', 'bottom'), (-.8 * g, -.8 * g, 'right', 'top')]
    out = []
    for ax in dict.fromkeys(d['ax'] for d in drawn):
        mine = [d for d in drawn if d['ax'] is ax]
        boxes = [marker_box(d) for d in mine]
        soft = [ln.get_path().transformed(ln.get_transform()) for ln in ax.lines if ln.get_marker() in ('None', '', None)
                and ln.get_linestyle() not in ('None', '')]
        for d in mine:
            for lc in d['bars'].lines[2]:
                soft += [Path(ax.transData.transform(s)) for s in lc.get_segments()]
        inner = ax.bbox.padded(-.5)
        placed = []
        for d in mine:
            if d['n'] >= 3:
                continue
            t = ax.annotate(str(d['n']), (d['marker'].get_xdata()[0], d['marker'].get_ydata()[0]), xytext=(0, 0),
                            textcoords='offset points', fontsize=FS['annotation'], color='black', zorder=6,
                            annotation_clip=False,
                            path_effects=[patheffects.withStroke(linewidth=2 * R_HALO, foreground='white')])
            best = None
            for i, (dx, dy, ha, va) in enumerate(spots):
                t.set_position((dx, dy)); t.set_ha(ha); t.set_va(va)
                bb = t.get_window_extent(renderer)
                if not (inner.x0 <= bb.x0 and bb.x1 <= inner.x1 and inner.y0 <= bb.y0 and bb.y1 <= inner.y1):
                    continue
                if any(bb.overlaps(b) for b in boxes) or any(bb.overlaps(p) for p in placed):
                    continue
                score = (sum(p.intersects_bbox(bb, filled=False) for p in soft), i)
                best = min(best, (score, i)) if best else (score, i)
            assert best, f"no room for the seed-count numeral of {d['setting']} {d['model']} {d['method']} {d['xm']}"
            dx, dy, ha, va = spots[best[1]]
            t.set_position((dx, dy)); t.set_ha(ha); t.set_va(va)
            placed.append(t.get_window_extent(renderer))
            out.append(dict(text=t, point=d))
    return out


def r_texts(fig, axes, numerals, notes, legends):
    """Every visible piece of text with its owner: (label, owner, Text)."""
    out = []
    for key, ax in axes.items():
        for name, t in (('title', ax.title), ('xlabel', ax.xaxis.label), ('ylabel', ax.yaxis.label)):
            if t.get_visible() and t.get_text().strip():
                out.append((f'{key} {name}', ('ax', key), t))
        for axis, lim in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
            locs = axis.get_majorticklocs()
            for loc, tick in zip(locs, axis.get_major_ticks(len(locs))):
                if lim[0] <= loc <= lim[1] and tick.label1.get_visible() and tick.label1.get_text().strip():
                    out.append((f'{key} tick {tick.label1.get_text()}', ('ax', key), tick.label1))
    out += [(f"numeral {n['point']['setting']} {n['point']['model']} {n['point']['method']}",
             ('ax', next(k for k, a in axes.items() if a is n['point']['ax'])), n['text']) for n in numerals]
    out += [(f'note {si}', ('note', si), t) for si, t in notes.items()]
    for i, lg in enumerate(legends):
        out += [(f'legend {i} "{t.get_text()}"', ('legend', i), t) for t in lg.get_texts()]
    return out


def r_checks(fig, axes, drawn, numerals, notes, legends):
    """Hard checks on the final layout: no text overlaps other text, a subplot or a legend; no text or legend
    leaves the page; no marker is cut by its subplot's edge; every numeral sits inside its subplot."""
    renderer = fig.canvas.get_renderer()
    page = fig.bbox
    texts = [(lab, own, t.get_window_extent(renderer)) for lab, own, t in r_texts(fig, axes, numerals, notes, legends)]
    boxes = texts + [(f'legend {i} frame', ('legend', i), lg.get_window_extent(renderer)) for i, lg in enumerate(legends)]
    shrink = .25 * fig.dpi / 72                         # boxes may touch, not overlap by more than 0.25 pt
    def hit(a, b):
        return a.x0 + shrink < b.x1 and b.x0 + shrink < a.x1 and a.y0 + shrink < b.y1 and b.y0 + shrink < a.y1
    bad = []
    for i, (la, oa, ba) in enumerate(boxes):
        if not (page.x0 <= ba.x0 and ba.x1 <= page.x1 and page.y0 <= ba.y0 and ba.y1 <= page.y1):
            bad.append(f'off the page: {la}')
        for lb, ob, bb in boxes[i + 1:]:
            if oa[0] == ob[0] == 'legend' and oa == ob:
                continue
            if hit(ba, bb):
                bad.append(f'overlap: {la} / {lb}')
        for key, ax in axes.items():
            own = oa == ('ax', key)
            if la.startswith('numeral') and own:
                if not (ax.bbox.x0 <= ba.x0 and ba.x1 <= ax.bbox.x1 and ax.bbox.y0 <= ba.y0 and ba.y1 <= ax.bbox.y1):
                    bad.append(f'numeral leaves its subplot: {la}')
            elif not own and hit(ba, ax.bbox):
                bad.append(f'overlap: {la} / subplot {key}')
    r_px = (R_MS / 2 + R_MEW / 2) * fig.dpi / 72
    for d in drawn:
        ax = d['ax']
        cx, cy = ax.transData.transform((d['marker'].get_xdata()[0], d['marker'].get_ydata()[0]))
        if not (ax.bbox.x0 <= cx - r_px and cx + r_px <= ax.bbox.x1 and ax.bbox.y0 <= cy - r_px and cy + r_px <= ax.bbox.y1):
            bad.append(f"clipped marker: {d['setting']} {d['model']} {d['method']} {d['xm']}")
        mbox = Bbox([[cx - r_px, cy - r_px], [cx + r_px, cy + r_px]])
        for n in numerals:
            if n['point']['ax'] is ax and n['text'].get_window_extent(renderer).overlaps(mbox):
                bad.append(f"numeral on a marker: {n['point']['model']} {n['point']['method']} / {d['model']} {d['method']}")
    assert not bad, 'layout checks failed:\n  ' + '\n  '.join(bad)
    return len(texts)


def r_draw(agg):
    """Figure S3 in the paper style; returns (figure, layout record, drawn points)."""
    names = [plain for _, _, plain in MODELS]
    assert all(n in METHOD_MARKER and not METHOD_MARKER[n]['filled'] for n in names), \
        'fill carries the evaluation only while every predictor here is an open (Unif-) one in METHOD_MARKER'
    assert len({METHOD_MARKER[n]['marker'] for n in names}) == len(names), 'two predictors share a marker shape'
    fig = plt.figure(figsize=(PRINT_WIDTH[R_STEM], 6.2))
    axes, drawn, obs, notes = {}, [], {}, {}
    for si, (s, title) in enumerate(zip(SETTINGS, TITLES)):
        for row, (xm, ym, xl, yl, color) in enumerate(R_SIDES):
            ax = fig.add_axes([0, 0, .2, .1])
            ax.axvline(5, color=LINE['ref'], lw=LW['ref'], ls=LS['ref'], zorder=0)
            xs, ys, dr = r_panel(ax, agg, s, xm, ym, color)
            obs[si, row] = (xs, ys, [d['marker'].get_xdata()[0] for d in dr], [d['marker'].get_ydata()[0] for d in dr],
                            [d['marker'].get_ydata()[0] for d in dr if d['n'] < 3])
            drawn += dr
            ax.set_xlabel(xl, labelpad=1); ax.set_ylabel(yl, labelpad=1.5)
            for axis in (ax.xaxis, ax.yaxis):
                axis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10], min_n_ticks=3))
            if row == 0:
                ax.set_title(title, pad=3)
            axes[si, row] = ax
        parts = note_parts(agg, s)
        if parts:
            notes[si] = fig.text(0, 0, '', fontsize=FS['annotation'], color='black', ha='left', va='top',
                                 linespacing=1.15)
    shape = [Line2D([], [], ls='none', label=tt(n), **(marker_kw(n, 'black', size=R_MS, edge=R_MEW) | {'markerfacecolor': 'white'}))
             for n in names]
    legends = [fig.legend(handles=shape, ncol=len(shape), bbox_to_anchor=(.5, 1), bbox_transform=fig.transFigure,
                          **paper_legend_kw(FS['legend']))]
    ev = [Line2D([], [], ls='none', marker='o', markersize=R_MS, markeredgewidth=R_MEW, markeredgecolor='black',
                 markerfacecolor='white', label=r'validation-fitted ($w=0.2$)'),
          Line2D([], [], ls='none', marker='o', markersize=R_MS, markeredgewidth=R_MEW, markeredgecolor='black',
                 markerfacecolor='black', label=r'certified ($\alpha=\delta=0.05$)'),
          _SeedCount(),
          Line2D([], [], color=LINE['ref'], lw=LW['ref'], ls=LS['ref'], label=r'5\% error limit')]
    labels = [h.get_label() for h in ev[:2]] + ['seeds with a defined error rate, if fewer than 3', ev[3].get_label()]
    legends.append(fig.legend(handles=ev, labels=labels, ncol=2, bbox_to_anchor=(.5, .9), bbox_transform=fig.transFigure,
                              handler_map={_SeedCount: _SeedCountHandler()}, **paper_legend_kw(FS['legend'])))
    margins = [R_INK[0] / 2 + R_INK[1] / 2] * 4
    ax_w, col_w = 1.55, [1.9] * 3
    for it in range(8):
        renderer = fig.canvas.get_renderer()
        for (si, row), ax in axes.items():
            xs, ys, mx, my, py = obs[si, row]
            (x0, x1), (y0, y1) = r_limits(xs, ys, mx, my, ax_w * 72, R_AX_H * 72, py)
            ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
            assert x0 <= min(xs) and max(xs) <= x1 and y0 <= min(ys) and max(ys) <= y1
        for si, t in notes.items():
            t.set_text(r_wrap(fig, renderer, note_parts(agg, SETTINGS[si]), col_w[si % 3] * fig.dpi))
        H, ax_w, col_w = r_layout(fig, axes, notes, legends, margins)
        if it < 2:
            continue                                    # let limits, ticks and wrapping settle first
        buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=R_PNG_DPI)
        ink = r_ink_margins(buf.getvalue())
        if all(R_INK[0] <= v <= R_INK[1] for v in ink):
            break
        margins = [mg + (R_INK[0] / 2 + R_INK[1] / 2) - v for mg, v in zip(margins, ink)]
    else:
        raise AssertionError(f'outer margins did not settle: {ink}')
    fig.canvas.draw()          # the 300 dpi PNG pass leaves axis labels at stale pixel positions; redraw at fig.dpi
    numerals = r_numerals(fig, fig.canvas.get_renderer(), drawn)
    fig.canvas.draw()
    n_text = r_checks(fig, axes, drawn, numerals, notes, legends)
    ranges = {f'{SETTINGS[si]}/{R_SIDES[row][0]}': dict(x=list(ax.get_xlim()), y=list(ax.get_ylim()))
              for (si, row), ax in axes.items()}
    lay = dict(page_in=(PRINT_WIDTH[R_STEM], H), subplot_in=(ax_w, R_AX_H), column_in=col_w, margins_set=margins,
               ink_margins_png=ink, texts_checked=n_text, numerals=len(numerals), ranges=ranges,
               layout_iterations=it + 1)
    return fig, lay, drawn


def r_check_data(agg, drawn, stored_path):
    """The plotted means, mins and maxes, read back from the drawn artists, equal the aggregates stored in
    values/predictor_comparison_uniform_values.json; the stored file is read, never written."""
    stored = {(a['setting'], a['model'], a['method']): a for a in json.loads(stored_path.read_text())['aggregates']}
    assert set(stored) == set(agg), 'stored and rebuilt aggregates cover different points'
    for k, a in agg.items():
        for m in METRICS:
            assert a[m] == stored[k][m], (k, m, a[m], stored[k][m])
        assert a['all_abstain_seeds'] == stored[k]['all_abstain_seeds'], k
    worst, seen = 0.0, set()
    for d in drawn:
        s = stored[d['setting'], d['model'], d['method']]
        x, y = s[d['xm']], s[d['ym']]
        assert d['marker'].get_xdata()[0] == x['mean'] and d['marker'].get_ydata()[0] == y['mean'], d
        segs = [seg for lc in d['bars'].lines[2] for seg in lc.get_segments()]
        h = next(sg for sg in segs if sg[0][1] == sg[1][1] == y['mean'] and min(sg[:, 0]) <= x['mean'] <= max(sg[:, 0]))
        v = next(sg for sg in segs if sg[0][0] == sg[1][0] == x['mean'] and min(sg[:, 1]) <= y['mean'] <= max(sg[:, 1])
                 and sg is not h)
        dev = [abs(min(h[:, 0]) - x['min']), abs(max(h[:, 0]) - x['max']), abs(min(v[:, 1]) - y['min']),
               abs(max(v[:, 1]) - y['max'])]
        worst = max(worst, *dev)
        assert max(dev) < 1e-9, (d['setting'], d['model'], d['method'], dev)
        seen.add((d['setting'], d['model'], d['method'], d['xm']))
    want = {(s, mo, me, xm) for (s, mo, me), a in stored.items() for xm in ('fdr', 'for') if a[xm]['n'] > 0}
    assert seen == want, ('plotted points differ from the stored ones with a defined rate', want ^ seen)
    return dict(points=len(drawn), stored_aggregates=len(stored), max_abs_bar_end_deviation=worst,
                means='exactly equal', unplotted_undefined=sum(1 for a in stored.values() for xm in ('fdr', 'for')
                                                               if a[xm]['n'] == 0))


def main_restyle():
    """The paper figure, plots/predictor_comparison_uniform.{pdf,png}; writes nothing under values/."""
    vals = [VALUES / 'predictor_comparison_uniform_values.json', VALUES / 'predictor_comparison_uniform_per_seed.csv']
    before = {p.name: hashlib.md5(p.read_bytes()).hexdigest() for p in vals}
    rows, ident, agg, (n1, n2) = build()
    with plt.rc_context(style_rc()):
        fig, lay, drawn = r_draw(agg)
        data = r_check_data(agg, drawn, vals[0])
        R_DIR.mkdir(exist_ok=True)
        pdf, png = R_DIR / f'{R_STEM}.pdf', R_DIR / f'{R_STEM}.png'
        fig.savefig(pdf, metadata={'CreationDate': None})
        fig.savefig(png, dpi=R_PNG_DPI)
        plt.close(fig)
    w, h = page_size(pdf)
    assert abs(w - PRINT_WIDTH[R_STEM]) <= .01, (w, PRINT_WIDTH[R_STEM])
    ink = r_ink_margins(str(png))
    assert all(.01 <= v <= .02 for v in ink), f'outer whitespace outside 0.01-0.02 in: {ink}'
    after = {p.name: hashlib.md5(p.read_bytes()).hexdigest() for p in vals}
    assert before == after, 'a values/ file changed during --restyle'
    print(f'{R_STEM} (restyle): {w} x {h} in -> {pdf.relative_to(HERE)}, {png.relative_to(HERE)} ({R_PNG_DPI} dpi)')
    print(f"  subplot {lay['subplot_in'][0]:.3f} x {lay['subplot_in'][1]:.3f} in; columns "
          f"{', '.join(f'{c:.3f}' for c in lay['column_in'])} in; outer ink margins (l, r, t, b) "
          f"{', '.join(f'{v:.4f}' for v in ink)} in; {lay['layout_iterations']} layout passes")
    print(f"  checks passed: {lay['texts_checked']} texts (overlap, page, subplot), {len(drawn)} markers inside, "
          f"{lay['numerals']} seed-count numerals; identity groups {len(ident)}, Table 1 {n1}/30, Fig 3 {n2}/24")
    print(f'  data: {data}')
    print(f'  values/ unchanged: {after}')
    return lay, data


def main(paper):
    rows, ident, agg, (n1, n2) = build()
    cols = ['setting', 'model', 'seed', 'method', 'run', 'n_train', 'n_test', 'P', 'N', 'TP', 'FP', 'FN', 'TN', 'S', 'F',
            'rec', 'fdr', 'fail_rec', 'for', 'abstain', 'all_abstain', 'thresholds', 'cert_pass',
            'cal_upper_bound_fdr', 'cal_upper_bound_for', 'config_seed', 'K', 'bnn_weight_samples', 'commit', 'checkpoints',
            'train_ids_from', 'train_ids_checked_against_own_list', 'train_ids_sha', 'train_idx_sha', 'pool_file',
            'pool_file_read', 'trajectories_file',
            'val_states_sha', 'cal_states_sha', 'test_states_sha', 'dir']
    VALUES.mkdir(exist_ok=True)
    with open(VALUES / 'predictor_comparison_uniform_per_seed.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in sorted(rows, key=lambda r: (SETTINGS.index(r['setting']), [m for m, _, _ in MODELS].index(r['model']),
                                             r['method'] != 'val', r['seed'])):
            w.writerow({k: ('' if r.get(k) is None else r.get(k)) for k in cols})
    size, ranges = draw(agg, paper)
    (VALUES / 'predictor_comparison_uniform_values.json').write_text(json.dumps(dict(
        protocol='uniform runs, final round; val = soft four-threshold rule fit on D_val at w=0.2; cal = A-1d at '
                 'alpha=delta=0.05; per-seed rates recomputed from per-state test predictions; mean with min-max '
                 'over 3 seeds; FDR/FOR over seeds with a nonempty declaration set',
        sources={'soft_four_wsweep.jsonl': hashlib.sha256((ASSETS / 'soft_four_wsweep.jsonl').read_bytes()).hexdigest(),
                 str(certified.SRC): certified.SRC_SHA256},
        code=dict(certificates=str(DETV3 / 'adaptive_roa/adaptive_v2/eval/certificates.py')),
        checks=dict(training_and_eval_identity=ident, table1_fm_uniform_cells=n1, fig3_uniform_endpoint_cells=n2),
        page_size=size, axis_ranges=ranges,
        aggregates=[dict(setting=s, model=m, method=meth, **{k: v for k, v in a.items()})
                    for (s, m, meth), a in sorted(agg.items())]), indent=1, default=str) + '\n')
    print(f'checks: identity groups {len(ident)}, Table 1 {n1}/30, Fig 3 {n2}/24; figure {size}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument('--paper', action='store_true',
                   help='also copy plots/predictor_comparison_uniform.pdf (the paper-style figure) into the paper clone')
    p.add_argument('--restyle', action='store_true',
                   help='redraw only the paper-style figure into plots/ (no values/ rebuild, no old-style draw)')
    a = p.parse_args()
    if not a.restyle:
        main(False)                         # values/ and the old-style draw, which goes to plots/legacy/
    main_restyle()
    if a.paper:
        install_paper(R_STEM)
