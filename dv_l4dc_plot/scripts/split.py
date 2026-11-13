"""The split layout of paper Fig 4: per setting, a recall panel (R_s, R_f) over an error panel (FDR, FOR).

Data-agnostic. A figure passes series(setting, metric) -> (mean, lo, hi, n) arrays over x, so the
band can be min-max across seeds (every paper figure since 2026-11-12) or mean +/- SD (old previews).
With row_axes, each row (LQR, RL) gets fixed y axes and the figure fails if a marker or band
would leave its panel or the 5% line would fall off an error axis.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from common import SETTINGS, TITLES, BLUE, ORANGE, REF, RECALL, ERROR, save, REC_S, REC_F, ERR_S, ERR_F

# Paper Fig S2 legend: one entry per line, in the method section's terms (04_proposed_method.tex
# defines recall Rec_l and prediction error Err_l of outcome l; common.REC_S ... are their empirical test rates).
LEGEND_FOUR = [REC_S + ' (Success recall)', REC_F + ' (Failure recall)',
               ERR_S + ' (Success prediction error)', ERR_F + ' (Failure prediction error)']
# Error-line style of the paper's line figures (Dhruv, 2026-11-12): dash-dot, tightened from matplotlib's
# default '-.' (6.4, 1.6, 1, 1.6) so every segment between Fig 2's 11 checkpoints shows a full dash-dot.
DASHDOT = (0, (3.2, 1.3, 0.9, 1.3))
# Marker size and edge of the paper's line figures (Figs 2 and 3), shared so the two match. Previews keep
# draw()'s 3.2pt default.
PAPER_MS, PAPER_MEW = 1.8, .5     # edge thinned from .75 so 1.8pt hollow markers keep a visible hole
REF_SOLID = dict(color=REF, lw=.7, ls='-')     # 5% line, solid so it can't pass for a dotted error line
REF_DASHED = dict(color=REF, lw=.65, ls='--')


def band_arrays(mean, lo, hi, n, clip=False, min_band_n=0):
    lo, hi = np.array(lo, dtype=float), np.array(hi, dtype=float)
    if clip:  # a rate lives in [0, 100]
        lo, hi = np.clip(lo, 0, 100), np.clip(hi, 0, 100)
    if min_band_n:  # no band where fewer than min_band_n seeds contribute
        lo[n < min_band_n] = np.nan; hi[n < min_band_n] = np.nan
    return lo, hi


def draw(ax, x, mean, lo, hi, n, color, ls='-', marker='o', clip=False, min_band_n=0, ms=3.2, band_alpha=.12,
         mew=.75):
    """Mean line with markers over a shaded band; hollow markers where 0 < n < 3; a gap where n = 0."""
    lo, hi = band_arrays(mean, lo, hi, n, clip, min_band_n)
    ax.fill_between(x, lo, hi, color=color, alpha=band_alpha, lw=0)
    ax.plot(x, mean, color=color, ls=ls, lw=1.15, marker=marker, ms=ms, mew=mew, mfc=color, mec=color, zorder=3)
    partial = (n > 0) & (n < 3)
    if partial.any():
        ax.plot(x[partial], mean[partial], ls='none', marker=marker, ms=ms, mfc='white', mec=color, mew=mew, zorder=4)


def assert_markers_inside(fig, ax, ys, ms=3.2):
    """Fail if a marker of size ms (points) at any y in ys would be cut by the panel edge."""
    ys = np.array([y for y in ys if y is not None], dtype=float)
    ys = ys[np.isfinite(ys)]
    bbox, r = ax.get_window_extent(), ms / 2 * fig.dpi / 72
    yd = ax.transData.transform(np.c_[np.zeros_like(ys), ys])[:, 1]
    assert (yd - r >= bbox.y0).all() and (yd + r <= bbox.y1).all(), (ax.get_ylim(), ys.min(), ys.max())


def assert_bands_inside(ax, bands):
    """Fail if any band (lo, hi) runs past the panel's y limits."""
    y0, y1 = ax.get_ylim()
    for lo, hi in bands:
        if np.isfinite(lo).any():
            assert np.nanmin(lo) >= y0 and np.nanmax(hi) <= y1, (ax.get_ylim(), np.nanmin(lo), np.nanmax(hi))


def split_figure(series, x, xticklabels, xlim, xlabel, stem, height_ratios=(1.5, 1), clip=False, row_axes=None,
                 err_ls='-', ref=None, legend_four=False, err_max=12, min_band_n=0, paper_stem=None, paper=False,
                 ms=3.2, mew=.75):
    """Recall panel over error panel per setting. Without row_axes: recall -3..103 and error
    -0.03*err_max..err_max on every panel. legend_four lists each line (LEGEND_FOUR) and leaves the
    5% line out of the legend; otherwise the legend pairs R_s/FDR and R_f/FOR and shows the 5% line."""
    ref = ref or REF_DASHED
    x = np.asarray(x, dtype=float)
    fig = plt.figure(figsize=(6.5, 4.85))
    outer = fig.add_gridspec(2, 3, left=.08, right=.99, top=.88, bottom=.1, wspace=.24, hspace=.4)
    for si, (s, title) in enumerate(zip(SETTINGS, TITLES)):
        inner = outer[si // 3, si % 3].subgridspec(2, 1, height_ratios=list(height_ratios), hspace=.1)
        rec = fig.add_subplot(inner[0]); err = fig.add_subplot(inner[1], sharex=rec)
        rec.set_title(title, pad=5)
        drawn = {}
        for ax, pairs, ls in [(rec, RECALL, '-'), (err, ERROR, err_ls)]:
            ax.grid(axis='y', alpha=.2, lw=.45); ax.set_axisbelow(True)
            for metric, color in pairs:
                mean, lo, hi, n = series(s, metric)
                draw(ax, x, mean, lo, hi, n, color, ls=ls, clip=clip, min_band_n=min_band_n, ms=ms, mew=mew)
                drawn[metric] = (mean, band_arrays(mean, lo, hi, n, clip, min_band_n))
        if row_axes is None:
            rec.set_ylim(-3, 103); rec.set_yticks([0, 50, 100])
        else:
            rec.set_ylim(*row_axes[si // 3]['rec'][0]); rec.set_yticks(row_axes[si // 3]['rec'][1])
        rec.tick_params(labelbottom=False)
        if row_axes is None:
            err.set_ylim(-.03 * err_max, err_max); err.set_yticks([0, 5, 10] if err_max >= 10 else [0, 5])
        else:
            err.set_ylim(*row_axes[si // 3]['err'][0]); err.set_yticks(row_axes[si // 3]['err'][1])
        err.axhline(5, zorder=0, **ref)
        err.set_xlim(*xlim)
        err.set_xticks(x, xticklabels)
        if row_axes is not None:
            assert row_axes[si // 3]['err'][0][1] > 5  # the 5% line must stay visible
            for ax, pairs in [(rec, RECALL), (err, ERROR)]:
                assert_markers_inside(fig, ax, np.concatenate([drawn[m][0] for m, _ in pairs]), ms=ms)
                assert_bands_inside(ax, [drawn[m][1] for m, _ in pairs])
        if si % 3 == 0:
            rec.set_ylabel('Recall (%) ↑'); err.set_ylabel('Error (%) ↓')
    fig.supxlabel(xlabel, fontsize=7.5, y=.035)
    if legend_four:
        styles = [(BLUE, '-'), (ORANGE, '-'), (BLUE, err_ls), (ORANGE, err_ls)]
        fig.legend(handles=[Line2D([], [], color=c, ls=ls, lw=1.15, marker='o', ms=ms, mew=mew, label=t)
                            for (c, ls), t in zip(styles, LEGEND_FOUR)],
                   loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(.54, 1.02),
                   columnspacing=2.0, handlelength=2.6, labelspacing=.35)
    else:
        fig.legend(handles=[Line2D([], [], color=BLUE, marker='o', ms=3.2, label=r'$R_{\mathrm{s}}$ / FDR'),
                            Line2D([], [], color=ORANGE, marker='o', ms=3.2, label=r'$R_{\mathrm{f}}$ / FOR'),
                            Line2D([], [], color=REF, lw=.65, ls='--', label='5% error')],
                   loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(.54, 1.0), columnspacing=1.5,
                   handlelength=2.2)
    return save(fig, stem, paper_stem=paper_stem, paper=paper)


# ==== Restyle (dm1487, 2026-11-13): the line and bar figures in the common.py paper style, at print width =========
# Only the --restyle paths of plot_learning_curves.py, plot_adaptive_fraction_certified.py and
# plot_acquisition_strategy_bars.py call this block. Nothing above it changed, so every default path keeps its bytes.
# Every style value (colours, fonts, sizes, line styles, markers, legend) comes from common.py; this block lays the
# figures out from measured text and checks them. It writes plots/<stem>.{pdf,png}, the paper figures,: no values/, tables/ or paper files.
import argparse
import math
import re
import matplotlib.colors
from matplotlib.legend_handler import HandlerBase, HandlerTuple
from matplotlib.patches import Patch, Rectangle
from matplotlib.transforms import ScaledTranslation
from common import PLOTS, PAPER, PRINT_WIDTH, FS, LINE, LS, LW, INK, marker_kw, paper_legend_kw, style_rc, PARTIAL_TINT, ours

RESTYLE_DIR = PLOTS                     # the paper figures themselves since 2026-11-13 (was plots/restyle/)
PT = 1 / 72                                                    # inches per point
# Colour names the metric's side and line style its kind (common.LINE, common.LS).
SIDE = {'rec': LINE['success'], 'fdr': LINE['success'], 'fail_rec': LINE['fail'], 'for': LINE['fail'],
        'abstain': LINE['abstain']}
KIND = {'rec': LS['recall'], 'fail_rec': LS['recall'], 'fdr': LS['error'], 'for': LS['error'],
        'abstain': LS['recall']}
MARKER_PT, EDGE_PT = 3.0, .6                                   # common.marker_kw defaults
REACH_PT = (MARKER_PT + EDGE_PT) / 2                           # marker centre to the outside of its edge stroke
# A method key's line (MethodKey) when line style names the method: a dash, dot, dash cycle at LW['curve'] stays
# readable between two markers at its ends.
METHOD_KEY_PT = 14.0
# The few fixed gaps, in points. Everything else is measured from the drawn text.
SEP = 2.0        # least space between two neighbouring pieces of text
LEG_GAP = 3.0    # legend to the first row's titles
ROW_GAP = 5.0    # first row's x tick labels to the second row's titles
XL_GAP = 1.5     # second row's x tick labels to the shared x label
PAIR_GAP = 3.0   # recall panel to error panel, at least
COL_GAP = 6.0    # between columns, at least (an interior y axis puts 2.5 pt tick marks into it)
MARGIN = .012    # outer whitespace, in; check_restyled accepts .005-.025
_UNICODE_TEX = {'↑': r'$\uparrow$', '↓': r'$\downarrow$'}


def restyle_args(doc):
    """--paper copies plots/<stem>.pdf, the paper-style figure, into the paper clone (common.install_paper).
    --restyle redraws only that figure: no values/ rebuild, no old-style draw."""
    p = argparse.ArgumentParser(description=doc.strip().splitlines()[0])
    p.add_argument('--paper', action='store_true',
                   help='also copy plots/<stem>.pdf (the paper-style figure) into deterministic-verification-pw')
    p.add_argument('--restyle', action='store_true',
                   help='redraw only the paper-style figure into plots/ (no values/ rebuild, no old-style draw)')
    return p.parse_args()


def tex(s):
    """A label made safe for usetex. An unescaped % starts a TeX comment and silently drops the rest of the label
    ("5% target" prints as "5"), and pdflatex has no glyph for the unicode arrows."""
    s = re.sub(r'(?<!\\)%', r'\\%', s)
    for k, v in _UNICODE_TEX.items():
        s = s.replace(k, v)
    return s


# A point from 1-2 seeds keeps its method's marker, faded: the side colour mixed this far toward white. Hollow can't
# mean 'partial' any more, because Unif-TERA's own square is open. Half-filled markers were tried and read as open
# at print size; a centre dot vanished. PARTIAL_TINT lives in common.py.


def partial_kw(method, color, ms=MARKER_PT, mew=EDGE_PT):
    """Marker for a point from 1-2 seeds: the method's marker (shape and fill unchanged) in the side colour faded by
    PARTIAL_TINT toward white, so shape still names the method and hue the side."""
    faded = tuple(c + (1 - c) * PARTIAL_TINT for c in matplotlib.colors.to_rgb(color))
    return marker_kw(method, faded, ms, mew)


def draw_restyled(ax, x, mean, lo, hi, n, metric, methods, band_alpha, ls=None, ms=MARKER_PT, mew=EDGE_PT):
    """One metric of one arm. Band = min-max over seeds where at least 2 contribute; line = mean, with a gap where no
    seed contributes; a marker per point: the method's marker (common.METHOD_MARKER) where 3 seeds contribute, faded
    (partial_kw) where 1-2 do. methods is one method name, or one per x (Figure S2: Unif-TERA at rho = 0).
    ls: the line style (None: the metric's kind, KIND); ms, mew: marker size and edge width in points.
    Returns the band (lo, hi) as drawn."""
    color = SIDE[metric]
    x, mean, n = np.asarray(x, dtype=float), np.asarray(mean, dtype=float), np.asarray(n)
    lo, hi = band_arrays(mean, lo, hi, n, min_band_n=2)
    if band_alpha:                                          # 0 or None: no seed band (Figure 6)
        ax.fill_between(x, lo, hi, color=color, alpha=band_alpha, lw=0)
    ax.plot(x, mean, color=color, ls=KIND[metric] if ls is None else ls, lw=LW['curve'], zorder=3)
    methods = np.array(np.broadcast_to(np.asarray(methods, dtype=object), x.shape))
    for method in dict.fromkeys(methods.tolist()):
        for sel, kw in [(n >= 3, marker_kw(method, color, ms, mew)), ((n > 0) & (n < 3), partial_kw(method, color, ms, mew))]:
            sel = sel & (methods == method)
            if sel.any():
                ax.plot(x[sel], mean[sel], ls='none', zorder=4, **kw)
    return lo, hi


def padded_ylim(lo, hi, panel_in, pad_pt=.6, reach_pt=REACH_PT):
    """y limits that keep a marker centred at lo or at hi, edge stroke included (reach_pt from its centre), pad_pt
    inside a panel panel_in tall. The padding is rounded up to two significant digits so the limits print cleanly."""
    k = (reach_pt + pad_pt) / (panel_in / PT)
    d = (hi - lo) * k / (1 - 2 * k)
    step = 10 ** (math.floor(math.log10(d)) - 1)
    d = math.ceil(d / step - 1e-9) * step
    return round(lo - d, 10), round(hi + d, 10)


def drawn_ticklabels(axis):
    """The tick labels the renderer draws: visible, non-empty, inside the view limits."""
    lo, hi = sorted(axis.get_view_interval())
    locs = axis.get_majorticklocs()
    out = []
    for tick, v in zip(axis.get_major_ticks(len(locs)), locs):
        if lo - 1e-9 <= v <= hi + 1e-9:
            out += [t for t in (tick.label1, tick.label2) if t.get_visible() and t.get_text()]
    return out


def dashdot_handlelength():
    """Legend handle length, in font sizes, that shows one whole dash-dot cycle and the next dash at LW['curve'].
    paper_legend_kw's 0.8 is shorter than one cycle, so FDR and FOR would not read as dash-dot."""
    dash = LS['error'][1]
    return (sum(dash) + dash[0]) * LW['curve'] / FS['legend']


def metric_handle(metric, ls=None):
    return Line2D([], [], color=SIDE[metric], ls=KIND[metric] if ls is None else ls, lw=LW['curve'])


# Legend keys that set their own width (Figure 7, Sumanth 2026-11-13). The legend gives every handle the same
# handlelength, but a method key there holds two markers, a dash-dot cycle between them and a colour square, while a
# colour-only metric key needs a short line. HandlerKey widens or narrows the key's handle box before drawing it; the
# legend lays its row out from the box widths when it draws, so the label moves with the key.
KEY_GAP_PT = 2.0                        # between the parts of one key: two squares, or the line and its square


def fill_square(x, y, side, color, alpha):
    """A legend colour square: the fill colour at alpha inside a .5 pt border in the full colour, the line the panel
    draws along the top of the same fill."""
    return Rectangle((x, y), side, side, facecolor=matplotlib.colors.to_rgba(color, alpha), edgecolor=color, lw=.5)


def centred_dashes(ls, length_pt, lw):
    """A dash tuple (offset, seq) shifted so the middle of seq[2], the dot of a dash-dot, falls at the middle of a key
    line length_pt long drawn at lw: the key then reads dash, dot, dash between its end markers. A named style such
    as '-' passes through. matplotlib scales the offset and the pattern by lw (lines.scale_dashes)."""
    if not isinstance(ls, tuple):
        return ls
    seq = ls[1]
    return ((seq[0] + seq[1] + seq[2] / 2 - length_pt / 2 / lw) % sum(seq), seq)


class LegendKey:
    """A legend handle that HandlerKey draws at width(h) points, h being the handle box height."""


class SwatchKey(LegendKey):
    """Fill colours as squares of the handle height side by side (the abstention key of Figure 7)."""
    def __init__(self, colors, alpha):
        self.colors, self.alpha = colors, alpha

    def width(self, h):
        return len(self.colors) * (h + KEY_GAP_PT) - KEY_GAP_PT

    def artists(self, x0, y0, h):
        return [fill_square(x0 + i * (h + KEY_GAP_PT), y0, h, c, self.alpha) for i, c in enumerate(self.colors)]


class MethodKey(LegendKey):
    """A method's key: its marker in both side colours (success left, failure right) at the two ends of an INK line
    line_pt long in the method's line style (no line when ls is None), then its fill colour as a square the size of
    SwatchKey's (none when fill is None)."""
    def __init__(self, method, ms, mew, line_pt, ls=None, fill=None, alpha=None):
        self.method, self.ms, self.mew, self.line_pt, self.ls, self.fill, self.alpha = method, ms, mew, line_pt, ls, fill, alpha

    def width(self, h):
        return self.line_pt + (KEY_GAP_PT + h if self.fill else 0)

    def artists(self, x0, y0, h):
        yc, r, L = y0 + h / 2, (self.ms + self.mew) / 2, self.line_pt
        out = []
        if self.ls is not None:
            out.append(Line2D([x0, x0 + L], [yc, yc], color=INK, lw=LW['curve'],
                              ls=centred_dashes(self.ls, L, LW['curve'])))
        out += [Line2D([x], [yc], ls='none', **marker_kw(self.method, SIDE[m], self.ms, self.mew))
                for x, m in [(x0 + r, 'rec'), (x0 + L - r, 'fail_rec')]]
        if self.fill:
            out.append(fill_square(x0 + L + KEY_GAP_PT, y0, h, self.fill, self.alpha))
        return out


class HandlerKey(HandlerBase):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        handlebox.width = orig_handle.width(handlebox.height)
        arts = orig_handle.artists(-handlebox.xdescent, -handlebox.ydescent, handlebox.height)
        for a in arts:
            handlebox.add_artist(a)
        return arts[0]


def method_handle(method, ms=MARKER_PT, mew=EDGE_PT):
    """A method's marker in both side colours, as the curves draw it."""
    return tuple(Line2D([], [], ls='none', **marker_kw(method, SIDE[m], ms, mew)) for m in ('rec', 'fail_rec'))


def top_legend(fig, handles, labels, width, ncols, handlelength=None, columnspacing=None):
    """The shared paper legend (common.paper_legend_kw at FS['legend']), with the first column count in ncols whose
    legend fits between the margins. tight_split_layout puts it at the top, centred on the figure. columnspacing
    overrides the shared column gap, in font sizes, for a figure whose one-row legend misses the width by a little."""
    kw = paper_legend_kw(FS['legend'])
    if handlelength:
        kw['handlelength'] = handlelength
    if columnspacing is not None:
        kw['columnspacing'] = columnspacing
    for ncol in ncols:
        leg = fig.legend(handles, labels, ncol=ncol, handler_map={tuple: HandlerTuple(ndivide=None, pad=.3),
                                                                 LegendKey: HandlerKey()}, **kw)
        fig.canvas.draw()
        if leg.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi <= width - 2 * MARGIN:
            return leg, ncol
        leg.remove()
    raise AssertionError(f'legend wider than {width} in at every column count in {ncols}')


def tight_split_layout(fig, pairs, legend, xlabel, *, width, rec_aspect, err_ratio, set_ylims):
    """Place a 2 x 3 grid of (recall, error) panel pairs (pairs in setting order) on a figure width in wide, as tight
    as the drawn text allows, and size the figure to fit. Every gap is measured, not guessed: draw, read the
    tick-label, title, legend and label extents, place in inches, and repeat until the panel width settles (the
    overhang of a tick label at a panel end depends on the panel width). Panel heights follow the width: recall
    height = rec_aspect x panel width, error height = err_ratio x recall height. set_ylims(rec_h, err_h) sets the y
    limits for those heights in inches. Titles are the recall axes' titles; only the left column carries y tick
    labels and y labels. The legend sits at the top, centred on the figure; xlabel (a figure text, va='bottom') is
    centred under the panel block. Returns the geometry in inches."""
    W, dpi = width, fig.dpi
    pw = (W - 1) / 3
    dec = dict(left=.35, right=0., gaps=[COL_GAP * PT] * 2, top=[.15, .15], pair=PAIR_GAP * PT, below=[.12, .12],
               leg=.15, xl=.12)
    for it in range(20):
        rec_h = rec_aspect * pw
        err_h = err_ratio * rec_h
        H = (2 * MARGIN + dec['leg'] + (LEG_GAP + ROW_GAP + XL_GAP) * PT + dec['xl']
             + sum(dec['top']) + sum(dec['below']) + 2 * (rec_h + dec['pair'] + err_h))
        fig.set_size_inches(W, H)
        xs = [MARGIN + dec['left'] + c * pw + sum(dec['gaps'][:c]) for c in range(3)]
        y = H - MARGIN - dec['leg'] - LEG_GAP * PT
        for r in range(2):
            y -= dec['top'][r]
            for c in range(3):
                rec, err = pairs[3 * r + c]
                rec.set_position([xs[c] / W, (y - rec_h) / H, pw / W, rec_h / H])
                err.set_position([xs[c] / W, (y - rec_h - dec['pair'] - err_h) / H, pw / W, err_h / H])
            y -= rec_h + dec['pair'] + err_h + dec['below'][r] + (ROW_GAP if r == 0 else XL_GAP) * PT
        set_ylims(rec_h, err_h)
        legend.set_bbox_to_anchor((.5, (H - MARGIN) / H), transform=fig.transFigure)
        block = (xs[0], xs[2] + pw)
        xlabel.set_position((sum(block) / 2 / W, MARGIN / H))

        fig.canvas.draw()
        rd = fig.canvas.get_renderer()
        box = lambda a: np.array(a.get_window_extent(rd).extents) / dpi
        pos = lambda ax: np.array(ax.get_position().extents) * [W, H, W, H]
        x_over, title_over = {}, {}       # (left, right) overhang past the panel, per pair
        top, below, pair_need, left = [0., 0.], [0., 0.], 0., 0.
        for si, (rec, err) in enumerate(pairs):
            r, c = divmod(si, 3)
            pr, pe = pos(rec), pos(err)
            xt = [box(t) for t in drawn_ticklabels(err.xaxis)]
            x_over[si] = (max([pe[0] - b[0] for b in xt], default=0.), max([b[2] - pe[2] for b in xt], default=0.))
            tb = box(rec.title)
            title_over[si] = (pr[0] - tb[0], tb[2] - pr[2])
            yr = [box(t) for t in drawn_ticklabels(rec.yaxis)]
            ye = [box(t) for t in drawn_ticklabels(err.yaxis)]
            top[r] = max(top[r], tb[3] - pr[3], *[b[3] - pr[3] for b in yr])
            below[r] = max([below[r]] + [pe[1] - b[1] for b in xt + ye])
            pair_need = max(pair_need, max([pr[1] - b[1] for b in yr], default=0.)
                            + max([b[3] - pe[3] for b in ye], default=0.) + SEP * PT)
            if c == 0:
                left = max(left, *[p[0] - np.array(a.yaxis.get_tightbbox(rd).extents)[0] / dpi
                                   for a, p in ((rec, pr), (err, pe))], x_over[si][0], title_over[si][0])
        gaps = []
        for c in range(2):
            need = [x_over[3 * r + c][1] + x_over[3 * r + c + 1][0] for r in range(2)]
            need += [title_over[3 * r + c][1] + title_over[3 * r + c + 1][0] for r in range(2)]
            gaps.append(max(COL_GAP * PT, max(need) + SEP * PT))
        right = max(0., *[x_over[si][1] for si in (2, 5)], *[title_over[si][1] for si in (2, 5)])
        new = dict(left=left, right=right, gaps=gaps, top=top, pair=max(PAIR_GAP * PT, pair_need), below=below,
                   leg=box(legend)[3] - box(legend)[1], xl=box(xlabel)[3] - box(xlabel)[1])
        new_pw = (W - 2 * MARGIN - left - right - sum(gaps)) / 3
        flat = lambda d: np.hstack([d['left'], d['right'], d['gaps'], d['top'], d['pair'], d['below'], d['leg'], d['xl']])
        settled = abs(new_pw - pw) < 1e-6 and np.allclose(flat(new), flat(dec), atol=1e-6)
        dec, pw = new, new_pw
        if settled:
            break
    else:
        raise AssertionError('layout did not settle')
    return dict(width=W, height=H, panel_w=pw, rec_h=rec_h, err_h=err_h, iterations=it + 1,
                **{k: v for k, v in dec.items()})


def align_left_ylabels(fig, axes):
    """Put the y labels of axes (the left column) on one vertical line, at the distance from the left spine that the
    widest tick labels among them need. fig.align_ylabels only acts on subplot axes, so on these add_axes panels it
    does nothing and each label sits at its own distance (Sumanth, 2026-11-13). Call once the ticks are final, then
    lay the figure out again. Returns that distance in inches."""
    fig.canvas.draw()
    d = max(ax.get_window_extent().x0 - ax.yaxis.label.get_position()[0] for ax in axes) / fig.dpi
    for ax in axes:
        ax.yaxis.set_label_coords(0, .5, transform=ax.transAxes + ScaledTranslation(-d, 0, fig.dpi_scale_trans))
    return d


def assert_markers_inside_xy(fig, ax, xs, ys, reach_pt=REACH_PT):
    """Fail if a marker at any finite (x, y), edge stroke included, would cross the panel edge in x or y."""
    xy = np.c_[np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)]
    xy = xy[np.isfinite(xy).all(1)]
    if not len(xy):
        return
    bb, r = ax.get_window_extent(), reach_pt * fig.dpi / 72
    d = ax.transData.transform(xy)
    assert ((d[:, 0] - r >= bb.x0) & (d[:, 0] + r <= bb.x1) & (d[:, 1] - r >= bb.y0) & (d[:, 1] + r <= bb.y1)).all(), \
        ('marker outside its panel', ax.get_xlim(), ax.get_ylim())


def check_restyled(fig, axes, legend, inside=()):
    """Hard checks on the final layout. Fails on overlapping text (tick labels, titles, axis labels, the legend, the
    x label, in-panel annotations), text off the page, text over a panel (except the in-panel annotations), an
    in-panel annotation outside its panel or over a marker, an unescaped % (usetex drops the rest of the label), a legend not centred on
    the figure, and outer whitespace outside .005-.025 in. Returns the outer whitespace per edge in inches."""
    fig.canvas.draw()
    rd = fig.canvas.get_renderer()
    texts = []
    for i, ax in enumerate(axes):
        texts += [(f'axes {i} tick {t.get_text()!r}', t) for t in drawn_ticklabels(ax.xaxis) + drawn_ticklabels(ax.yaxis)]
        texts += [(f'axes {i} {t.get_text()!r}', t) for t in (ax.title, ax.yaxis.label, ax.xaxis.label) if t.get_text()]
    texts += [(f'figure text {t.get_text()!r}', t) for t in fig.texts]
    texts += [(f'annotation {t.get_text()!r}', t) for t in inside]
    texts += [(f'legend {t.get_text()!r}', t) for t in legend.get_texts()]
    for name, t in texts:
        assert not re.search(r'(?<!\\)%', t.get_text()), ('unescaped % (usetex drops the rest of the label)', name)
    ext = [(n, t.get_window_extent(rd)) for n, t in texts if not n.startswith('legend ')]
    ext.append(('legend', legend.get_window_extent(rd)))
    page = fig.bbox
    for i, (na, a) in enumerate(ext):
        assert a.x0 >= page.x0 and a.x1 <= page.x1 and a.y0 >= page.y0 and a.y1 <= page.y1, ('text off the page', na)
        for nb, b in ext[i + 1:]:
            assert not a.overlaps(b), ('text overlap', na, nb)
    panels = [ax.get_window_extent(rd) for ax in axes]
    inside_names = {f'annotation {t.get_text()!r}' for t in inside}
    for na, a in ext:
        if na not in inside_names:
            assert not any(a.overlaps(p) for p in panels), ('text over a panel', na)
    for t in inside:                           # an in-panel annotation stays in its panel, clear of every marker
        bb, ax = t.get_window_extent(rd), t.axes
        pb = ax.get_window_extent(rd)
        assert pb.x0 <= bb.x0 and bb.x1 <= pb.x1 and pb.y0 <= bb.y0 and bb.y1 <= pb.y1, ('annotation leaves its panel',
                                                                                          t.get_text())
        for line in ax.lines:
            if line.get_marker() in (None, 'None', '', ' '):
                continue
            d = ax.transData.transform(np.asarray(line.get_xydata(), dtype=float))
            d = d[np.isfinite(d).all(1)]
            rr = (line.get_markersize() + line.get_markeredgewidth()) / 2 * fig.dpi / 72
            hit = (d[:, 0] + rr >= bb.x0) & (d[:, 0] - rr <= bb.x1) & (d[:, 1] + rr >= bb.y0) & (d[:, 1] - rr <= bb.y1)
            assert not hit.any(), ('annotation over a marker', t.get_text())
    ink = [b for _, b in ext] + panels
    edges = dict(left=min(b.x0 for b in ink) / fig.dpi, right=(page.x1 - max(b.x1 for b in ink)) / fig.dpi,
                 bottom=min(b.y0 for b in ink) / fig.dpi, top=(page.y1 - max(b.y1 for b in ink)) / fig.dpi)
    assert all(.005 <= v <= .025 for v in edges.values()), ('outer whitespace', edges)
    lb = legend.get_window_extent(rd)
    assert abs((lb.x0 + lb.x1) / 2 - page.x1 / 2) / fig.dpi < .005, 'legend not centred on the figure'
    return edges


def pdf_size(pdf):
    """Exact page size of a PDF in inches (common.page_size rounds to 2 dp)."""
    m = re.search(rb'/MediaBox\s*\[\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)', pdf.read_bytes())
    x0, y0, x1, y1 = map(float, m.groups())
    return (x1 - x0) / 72, (y1 - y0) / 72


def save_restyled(fig, stem):
    """Write plots/<stem>.{pdf,png} (PNG at 300 dpi) at the figure's own size, no cropping, so the page is
    exactly the print width. Fails unless the PDF is PRINT_WIDTH[stem] wide to within 0.01 in."""
    RESTYLE_DIR.mkdir(exist_ok=True)
    pdf, png = RESTYLE_DIR / f'{stem}.pdf', RESTYLE_DIR / f'{stem}.png'
    for p in (pdf, png):    # never anywhere else, least of all a paper checkout
        assert p.resolve().parent == RESTYLE_DIR.resolve() and 'deterministic-verification' not in str(p.resolve())
    fig.savefig(pdf, metadata={'CreationDate': None})
    fig.savefig(png, dpi=300)
    plt.close(fig)
    w, h = pdf_size(pdf)
    assert abs(w - PRINT_WIDTH[stem]) <= .01, (stem, w, PRINT_WIDTH[stem])
    print(f'restyle/{stem}: {w:.3f} x {h:.3f} in (print width {PRINT_WIDTH[stem]:.2f} in)')
    return dict(width_in=w, height_in=h, pdf=str(pdf), png=str(png))


def split_figure_restyled(stem, arms, xticks, xlabel, titles, row_base, metric_labels, legend_methods, ylabels,
                          rec_aspect, err_ratio, band_alpha, xlim=None, ref_label=None, rec_extra=(), rec_fill=None,
                          decorate=None, legend_columnspacing=None, method_ls=None, markers=(MARKER_PT, EDGE_PT)):
    """The split layout (recall panel over error panel per setting, settings in 2 rows x 3) in the paper style at
    PRINT_WIDTH[stem], laid out by tight_split_layout. Colour = the metric's side, line style = its kind (recall
    solid, error dash-dot), arm = marker shape and fill only; with method_ls, line style names the arm instead.

    arms: [(series, methods)], drawn in order (draw the reference arm first so the method sits on top);
      series(setting, metric) -> (x, mean, lo, hi, n); methods as in draw_restyled.
    xticks[setting] -> (positions, labels). row_base[row] -> dict(rec=(lo, hi, ticks), err=(lo, hi, ticks)): the
      y range the data must show, padded per panel height so a marker at either end stays inside.
    metric_labels: legend texts of rec, fail_rec, fdr, for. legend_methods: methods in the legend, in order.
    ref_label: text of the one direct label of the 5% line, in the first error panel (None: no label).
    rec_extra: [(metric, legend text)] drawn as curves in the upper panel under the recall curves.
    rec_fill: dict(metric, label, colors={method: colour}, alpha): that metric shaded from 0 up in the upper panel,
      one colour per arm (Figure 7: abstention); the legend shows the colours as bordered squares beside the label
      and each method (SwatchKey, MethodKey).
    method_ls: {method: line style} (Figure 7): line style names the method in both panels instead of the metric's
      kind; each arm's methods must then be one name. The metric keys become solid colour lines and each method key
      shows its line between its two markers (MethodKey).
    markers: (size, edge width) of every marker in points, panels and legend.
    legend_columnspacing: the legend's column gap in font sizes (None: the shared 1.3).
    decorate(panels): called with {setting: (upper, lower)} after the curves are drawn; returns in-panel texts,
      which check_restyled keeps inside their panel and clear of markers.
    Only the left column has y tick labels; the 5% line sits on every error panel. Fails if a marker or band leaves
    its panel, the 5% line leaves an error axis, or check_restyled fails. Returns the layout, checks and size."""
    W = PRINT_WIDTH[stem]
    with plt.rc_context(style_rc()):
        fig = plt.figure(figsize=(W, 4))
        pairs, drawn, ref_text = [], {}, None
        ms, mew = markers
        reach = (ms + mew) / 2
        assert method_ls is None or all(isinstance(methods, str) for _, methods in arms)
        for si, (s, title) in enumerate(zip(SETTINGS, titles)):
            rec = fig.add_axes([.1, .5, .2, .2]); err = fig.add_axes([.1, .2, .2, .2], sharex=rec)
            pairs.append((rec, err))
            rec.set_title(tex(title), pad=3)
            if rec_fill:
                for series, methods in arms:
                    x, mean, *_ = series(s, rec_fill['metric'])
                    col = rec_fill['colors'][methods]
                    rec.fill_between(x, 0, mean, color=col, alpha=rec_fill['alpha'], lw=0, zorder=1)
                    rec.plot(x, mean, color=col, lw=.5, zorder=1.5)
            for ax, metrics in [(rec, tuple(m for m, _ in rec_extra) + ('rec', 'fail_rec')), (err, ('fdr', 'for'))]:
                ax.grid(axis='y', alpha=.2, lw=.45); ax.set_axisbelow(True)
                for series, methods in arms:
                    for m in metrics:
                        x, mean, lo, hi, n = series(s, m)
                        band = draw_restyled(ax, x, mean, lo, hi, n, m, methods, band_alpha, ms=ms, mew=mew,
                                             ls=method_ls[methods] if method_ls else None)
                        drawn.setdefault(ax, []).append((np.asarray(x, dtype=float), np.asarray(mean, dtype=float), band))
            err.axhline(5, color=LINE['ref'], lw=LW['ref'], ls=LS['ref'], zorder=0)
            if xlim is not None:
                err.set_xlim(*xlim)
            err.set_xticks(*xticks[s])
            rec.tick_params(labelbottom=False)
            if si % 3:
                rec.tick_params(labelleft=False); err.tick_params(labelleft=False)
            else:
                rec.set_ylabel(tex(ylabels[0])); err.set_ylabel(tex(ylabels[1]))
            if si == 0 and ref_label:
                ref_text = err.annotate(tex(ref_label), xy=(1, 5), xycoords=('axes fraction', 'data'), xytext=(-1.5, -1.5),
                                        textcoords='offset points', ha='right', va='top', fontsize=FS['annotation'],
                                        color=LINE['ref'])
        notes = list(decorate(dict(zip(SETTINGS, pairs)))) if decorate else []
        metrics = ['rec', 'fail_rec', 'fdr', 'for'] + [m for m, _ in rec_extra]
        labels_ = [tex(t) for t in list(metric_labels) + [t for _, t in rec_extra]]
        m_handles, f_handles = [method_handle(m, ms, mew) for m in legend_methods], []
        if rec_fill or method_ls:               # keys of their own width: the fill colours, both beside the label and
            alpha = rec_fill['alpha'] + .25 if rec_fill else None      # each beside its method; the method's line
            fill = lambda m: rec_fill['colors'][m] if rec_fill else None
            if rec_fill:
                f_handles = [SwatchKey([fill(m) for m in legend_methods], alpha)]
                labels_ = labels_ + [tex(rec_fill['label'])]
            ls = lambda m: method_ls[m] if method_ls else None
            m_handles = [MethodKey(m, ms, mew, line_pt=METHOD_KEY_PT if method_ls else 2 * ms + 2 * mew + 2,
                                   ls=ls(m), fill=fill(m), alpha=alpha) for m in legend_methods]
        legend, ncol = top_legend(fig, [metric_handle(m, ls='-' if method_ls else None) for m in metrics] + f_handles
                                  + m_handles, labels_ + [ours(m) for m in legend_methods], W,
                                  ncols=(len(metrics) + len(f_handles) + len(m_handles), 4, 3, 2),
                                  handlelength=None if method_ls else dashdot_handlelength(),
                                  columnspacing=legend_columnspacing)
        xl = fig.text(.5, 0, tex(xlabel), ha='center', va='bottom', fontsize=FS['label'])
        limits = {}

        def set_ylims(rec_h, err_h):
            for si, (rec, err) in enumerate(pairs):
                base = row_base[si // 3]
                for ax, key, h in [(rec, 'rec', rec_h), (err, 'err', err_h)]:
                    lo, hi, ticks = base[key]
                    ax.set_ylim(*padded_ylim(lo, hi, h, reach_pt=reach)); ax.set_yticks(ticks)
                    limits[si // 3, key] = ax.get_ylim()

        layout = lambda: tight_split_layout(fig, pairs, legend, xl, width=W, rec_aspect=rec_aspect,
                                            err_ratio=err_ratio, set_ylims=set_ylims)
        layout()
        align_left_ylabels(fig, [a for p in pairs[::3] for a in p])
        geo = layout()
        fig.canvas.draw()
        assert len({round(a.yaxis.label.get_window_extent().x1, 3) for p in pairs[::3] for a in p}) == 1
        for rec, err in pairs:
            assert err.get_ylim()[0] < 5 < err.get_ylim()[1]        # the 5% line on every error axis
            assert any(abs(l.get_ydata()[0] - 5) < 1e-12 for l in err.lines if l.get_linestyle() == '-'
                       and l.get_color() == LINE['ref'])
            for ax in (rec, err):
                for x, mean, band in drawn[ax]:
                    assert_markers_inside_xy(fig, ax, x, mean, reach_pt=reach)
                    assert_bands_inside(ax, [band])
        edges = check_restyled(fig, [a for p in pairs for a in p], legend,
                               inside=([ref_text] if ref_text else []) + notes)
        size = save_restyled(fig, stem)
    return dict(geometry=geo, edges=edges, size=size, legend_ncol=ncol,
                ylims={f'row{r} {k}': v for (r, k), v in limits.items()})
