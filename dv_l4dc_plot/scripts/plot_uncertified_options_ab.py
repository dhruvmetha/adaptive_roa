r"""Table 2 as plots, options A and B: previews for choosing a figure, not paper figures.

Table 2 (tab:uncertified) is the empirical comparison at the validation-fitted w = 0.2 thresholds. Nothing in it is
certified, so nothing here says certified or target. Data, ordering and the read-back assert come from
uncertified_options_data.py; style from the paper block of common.py; the layout-from-measured-text approach and the
hard checks follow the --restyle path of plot_predictor_comparison.py (Figure S3), copied here so that script stays
as it is.

A  optA_scatter: 2 x 3, one panel per setting. Each method has a gold point (x = FDR, y = R_s) and a purple point
   (x = FOR, y = R_f), with min-max whiskers over seeds in x and y. Log x shared by every panel, linear y 0-100.
B  optB_rows: 4 x 3, the S3 arrangement. Per controller block (LQR, then RL) a success row (R_s over FDR, gold)
   over a failure row (R_f over FOR, purple); columns are the block's three settings. Log x shared by every panel;
   the two R_f rows zoom their y axis, and the note under the figure says so.

Both: colour = side, marker shape = method (common.marker_kw), whiskers = min and max over seeds, a faded marker =
an error rate from 2 of 3 seeds, n = 0 cells not drawn and named in a note, thin gray 5% reference line.
Writes plots/options_uncertified/<name>.{pdf,png} (PNG 300 dpi) and plots/options_uncertified/cvd/<name>_cvd.png,
nothing else.
"""
import hashlib, io, math, re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator
from PIL import Image
from common import PLOTS, SETTINGS, TITLES, FS, LINE, LS, LW, INK, PARTIAL_TINT, marker_kw, paper_legend_kw, style_rc
from uncertified_options_data import load, METHODS, missing, partial, check_drawn
from cvd_grid import grid as cvd_grid

OUT = PLOTS / 'options_uncertified'
CVD = OUT / 'cvd'
W = 6.00                                   # page width (in)
MS, MEW, HALO, BAR = 4.0, .6, .5, .5       # marker size, marker edge, white ring, whisker width (pt)
REACH = MS / 2 + MEW / 2 + HALO            # marker centre to the outside of its white ring (pt)
CLEAR = .5                                 # least gap from a marker ring or whisker end to the panel edge (pt)
PNG_DPI = 300
INK_TARGET = (.0115, .0185)                # outer whitespace the layout converges to (in)
INK_OK = (.005, .03)                       # the shared hard bounds (in)
SEP = 2 / 72                               # least gap between neighbouring pieces of text (in)
LEG_BETWEEN = .03                          # between the two legend rows (in)
PLAIN = [p for _, p in METHODS]
COLOR = {'success': LINE['success'], 'fail': LINE['fail']}
METRIC = {'success': ('fdr', 'rec'), 'fail': ('for', 'fail_rec')}          # (x error, y recall) per side
WORD = {'fdr': 'Success', 'for': 'Fail'}                                    # the empty declaration set behind n = 0
XTICKS = [0.1, 1, 10, 100]
XTICKLABELS = ['0.1', '1', '10', '100']


def faded(color):
    """The side colour mixed PARTIAL_TINT toward white (split.partial_kw, Figures 6 and S2)."""
    return tuple(c + (1 - c) * PARTIAL_TINT for c in mcolors.to_rgb(color))


def draw_side(ax, D, s, side):
    """One side's points in one panel, methods in METHODS order (TERA last, so on top). Whiskers are plain lines whose
    ends are the cell's min and max as stored, so the read-back is exact. Returns the point records."""
    err, rec = METRIC[side]
    color, out = COLOR[side], []
    for m in PLAIN:
        e, r = D[s, m, err], D[s, m, rec]
        if e['n'] == 0:
            continue
        assert r['n'] == 3, (s, m, rec, r['n'])
        hbar, = ax.plot([e['min'], e['max']], [r['mean'], r['mean']], color=color, lw=BAR, solid_capstyle='butt',
                        zorder=2)
        vbar, = ax.plot([e['mean'], e['mean']], [r['min'], r['max']], color=color, lw=BAR, solid_capstyle='butt',
                        zorder=2)
        kw = marker_kw(m, faded(color) if e['n'] < 3 else color, MS, MEW)
        mk, = ax.plot([e['mean']], [r['mean']], ls='none', zorder=4, **kw,
                      path_effects=[patheffects.withStroke(linewidth=MEW + 2 * HALO, foreground='white')])
        out.append(dict(setting=s, method=m, side=side, err=err, rec=rec, n=e['n'], ax=ax, marker=mk, hbar=hbar,
                        vbar=vbar))
    return out


def x_extent(D):
    """(whisker min, whisker max, mean min, mean max) over every drawn error cell: one log x range for all panels."""
    e = [v for k, v in D.items() if k[2] in ('fdr', 'for') and v['n'] > 0]
    return min(v['min'] for v in e), max(v['max'] for v in e), min(v['mean'] for v in e), max(v['mean'] for v in e)


def padded(lo_w, hi_w, lo_m, hi_m, span_pt, log, base=None):
    """Axis limits with every whisker end CLEAR pt inside and every marker ring CLEAR pt inside a panel span_pt long.
    lo_w/hi_w: whisker ends; lo_m/hi_m: marker centres; base: a range the axis shows at least, exactly where the
    whiskers and markers need no more (so a zoomed axis starts where the note says it does)."""
    f = math.log10 if log else (lambda v: v)
    ba, bb = (f(base[0]), f(base[1])) if base else (math.inf, -math.inf)
    a, b = min(f(lo_w), ba), max(f(hi_w), bb)
    for _ in range(50):
        u = (b - a) / span_pt                                  # data units per pt
        na = min(ba, f(lo_w) - CLEAR * u, f(lo_m) - (REACH + CLEAR) * u)
        nb = max(bb, f(hi_w) + CLEAR * u, f(hi_m) + (REACH + CLEAR) * u)
        if abs(na - a) < 1e-12 and abs(nb - b) < 1e-12:
            break
        a, b = na, nb
    else:
        raise AssertionError('axis padding did not settle')
    return (10 ** a, 10 ** b) if log else (a, b)


def setup_x(ax):
    ax.set_xscale('log')
    ax.xaxis.set_major_locator(FixedLocator(XTICKS))
    ax.xaxis.set_major_formatter(FixedFormatter(XTICKLABELS))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.axvline(5, color=LINE['ref'], lw=LW['ref'], ls=LS['ref'], zorder=0)


def missing_note(D):
    """'MORALS FDR undefined in Spatial Quad (12D), LQR and RL: no Success declarations in any seed', from missing()."""
    groups = {}
    for s, m, metric in missing(D):
        groups.setdefault((m, metric), []).append(TITLES[SETTINGS.index(s)])
    lines = []
    for (m, metric), titles in groups.items():
        systems = {t.rsplit(', ', 1)[0] for t in titles}
        where = (f"{systems.pop()}, {' and '.join(t.rsplit(', ', 1)[1] for t in titles)}" if len(systems) == 1
                 else '; '.join(titles))
        lines.append(f'{m} {metric.upper()} undefined in {where}: no {WORD[metric]} declarations in any seed')
    return lines


def legends(fig, D):
    """Two rows at the top: the six method shapes in INK, then the colour key, the faded key and the 5% line."""
    shapes = [Line2D([], [], ls='none', label=m, **marker_kw(m, INK, MS, MEW)) for m in PLAIN]
    kw = paper_legend_kw(FS['legend'])
    l1 = fig.legend(handles=shapes, ncol=len(shapes), bbox_to_anchor=(.5, 1), bbox_transform=fig.transFigure, **kw)
    part = partial(D)
    assert {D[k]['n'] for k in part} == {2} and len({(k[1], k[2]) for k in part}) == 1, part
    pm, pmetric = part[0][1], part[0][2]
    pside = 'success' if pmetric == 'fdr' else 'fail'
    keys = [Patch(facecolor=LINE['success'], edgecolor='none', label=r'success side: $R_{\mathrm{s}}$ vs FDR'),
            Patch(facecolor=LINE['fail'], edgecolor='none', label=r'failure side: $R_{\mathrm{f}}$ vs FOR'),
            Line2D([], [], ls='none', label='faded: 2 of 3 seeds', **marker_kw(pm, faded(COLOR[pside]), MS, MEW)),
            Line2D([], [], color=LINE['ref'], lw=LW['ref'], ls=LS['ref'], label=r'5\% reference')]
    for ncol in (4, 2):
        l2 = fig.legend(handles=keys, ncol=ncol, bbox_to_anchor=(.5, .9), bbox_transform=fig.transFigure, **kw)
        fig.canvas.draw()
        if l2.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi <= W - 2 * INK_TARGET[1]:
            break
        l2.remove()
    else:
        raise AssertionError('key legend too wide')
    return [l1, l2]


def measure(fig, ax, rd):
    """Decoration beyond each edge of one panel (in): tick labels, axis labels, title."""
    tb, bb, d = ax.get_tightbbox(rd), ax.bbox, fig.dpi
    return dict(l=max(0., (bb.x0 - tb.x0) / d), r=max(0., (tb.x1 - bb.x1) / d),
                t=max(0., (tb.y1 - bb.y1) / d), b=max(0., (bb.y0 - tb.y0) / d))


def place(fig, axes, shape, row_h, row_gaps, legs, bottoms, bottom_gaps, margins, col_gap, leg_gap):
    """Lay the panels, legends and bottom texts out in inches from the measured text. Returns (H, panel width, x0)."""
    nr, nc = shape
    fig.canvas.draw()
    rd = fig.canvas.get_renderer()
    m = {k: measure(fig, ax, rd) for k, ax in axes.items()}
    ml, mr, mt, mb = margins
    DL = [max(m[r, c]['l'] for r in range(nr)) for c in range(nc)]
    DR = [max(m[r, c]['r'] for r in range(nr)) for c in range(nc)]
    ax_w = (W - ml - mr - sum(DL) - sum(DR) - (nc - 1) * col_gap) / nc
    x0, x = [], ml
    for c in range(nc):
        x0.append(x + DL[c]); x += DL[c] + ax_w + DR[c] + col_gap
    leg_h = [lg.get_window_extent(rd).height / fig.dpi for lg in legs]
    y, leg_top = mt, []
    for h in leg_h:
        leg_top.append(y); y += h + LEG_BETWEEN
    y += leg_gap - LEG_BETWEEN
    top = []
    for r in range(nr):
        y += max(m[r, c]['t'] for c in range(nc))
        top.append(y)
        y += row_h[r] + max(m[r, c]['b'] for c in range(nc))
        if r < nr - 1:
            y += row_gaps[r]
    bot_top = []
    for t, g in zip(bottoms, bottom_gaps):
        y += g; bot_top.append(y); y += t.get_window_extent(rd).height / fig.dpi
    H = y + mb
    fig.set_size_inches(W, H)
    for (r, c), ax in axes.items():
        ax.set_position([x0[c] / W, (H - top[r] - row_h[r]) / H, ax_w / W, row_h[r] / H])
    for lg, yt in zip(legs, leg_top):
        lg.set_bbox_to_anchor((.5, (H - yt) / H), transform=fig.transFigure)
    centre = (x0[0] + x0[-1] + ax_w) / 2
    for t, yt in zip(bottoms, bot_top):
        t.set_position((centre / W, (H - yt) / H))
    return H, ax_w, x0


def ink_margins(png):
    """Unpainted outer margin (left, right, top, bottom) of a raster at PNG_DPI, in inches."""
    a = np.asarray(Image.open(io.BytesIO(png) if isinstance(png, bytes) else png).convert('L'))
    ink = np.argwhere(a < 250)
    (r0, c0), (r1, c1) = ink.min(0), ink.max(0)
    h, w = a.shape
    return c0 / PNG_DPI, (w - 1 - c1) / PNG_DPI, r0 / PNG_DPI, (h - 1 - r1) / PNG_DPI


def drawn_ticklabels(axis):
    """The tick labels the renderer draws: visible, non-empty, inside the view limits."""
    lo, hi = sorted(axis.get_view_interval())
    locs = axis.get_majorticklocs()
    return [t for tick, v in zip(axis.get_major_ticks(len(locs)), locs) if lo <= v <= hi
            for t in (tick.label1, tick.label2) if t.get_visible() and t.get_text().strip()]


def checks(fig, axes, pts, legs, bottoms):
    """Hard checks on the final layout: no unescaped %; no text overlaps other text, a panel it does not belong to, or
    a legend; nothing leaves the page; every marker ring and whisker end sits inside its panel; the 5% line is inside
    every panel; both legends are centred on the page. Returns the number of text boxes checked."""
    fig.canvas.draw()
    rd = fig.canvas.get_renderer()
    texts = []
    for key, ax in axes.items():
        for name, t in (('title', ax.title), ('xlabel', ax.xaxis.label), ('ylabel', ax.yaxis.label)):
            if t.get_visible() and t.get_text().strip():
                texts.append((f'{key} {name}', ('ax', key), t))
        for t in drawn_ticklabels(ax.xaxis) + drawn_ticklabels(ax.yaxis):
            texts.append((f'{key} tick {t.get_text()}', ('ax', key), t))
    texts += [(f'bottom {i}', ('bottom', i), t) for i, t in enumerate(bottoms)]
    for i, lg in enumerate(legs):
        texts += [(f'legend {i} "{t.get_text()}"', ('legend', i), t) for t in lg.get_texts()]
    bad = [f'unescaped %: {lab}' for lab, _, t in texts if re.search(r'(?<!\\)%', t.get_text())]
    boxes = [(lab, own, t.get_window_extent(rd)) for lab, own, t in texts]
    boxes += [(f'legend {i} frame', ('legend', i), lg.get_window_extent(rd)) for i, lg in enumerate(legs)]
    page = fig.bbox
    shrink = .25 * fig.dpi / 72                      # boxes may touch, not overlap by more than 0.25 pt
    def hit(a, b):
        return a.x0 + shrink < b.x1 and b.x0 + shrink < a.x1 and a.y0 + shrink < b.y1 and b.y0 + shrink < a.y1
    for i, (la, oa, ba) in enumerate(boxes):
        if not (page.x0 <= ba.x0 and ba.x1 <= page.x1 and page.y0 <= ba.y0 and ba.y1 <= page.y1):
            bad.append(f'off the page: {la}')
        for lb, ob, bb in boxes[i + 1:]:
            if oa == ob and oa[0] == 'legend':
                continue
            if hit(ba, bb):
                bad.append(f'overlap: {la} / {lb}')
        for key, ax in axes.items():
            if oa != ('ax', key) and hit(ba, ax.bbox):
                bad.append(f'over a panel: {la} / {key}')
    k = fig.dpi / 72
    for p in pts:
        ax, bb = p['ax'], p['ax'].bbox
        cx, cy = ax.transData.transform((p['marker'].get_xdata()[0], p['marker'].get_ydata()[0]))
        r = REACH * k
        if not (bb.x0 <= cx - r and cx + r <= bb.x1 and bb.y0 <= cy - r and cy + r <= bb.y1):
            bad.append(f"marker leaves its panel: {p['setting']} {p['method']} {p['side']}")
        for line in (p['hbar'], p['vbar']):
            for ex, ey in ax.transData.transform(np.c_[line.get_xdata(), line.get_ydata()]):
                if not (bb.x0 <= ex <= bb.x1 and bb.y0 <= ey <= bb.y1):
                    bad.append(f"whisker leaves its panel: {p['setting']} {p['method']} {p['side']}")
    for key, ax in axes.items():
        lo, hi = ax.get_xlim()
        if not lo < 5 < hi:
            bad.append(f'5% line outside panel {key}')
    for i, lg in enumerate(legs):
        lb = lg.get_window_extent(rd)
        if abs((lb.x0 + lb.x1) / 2 - page.x1 / 2) / fig.dpi > .005:
            bad.append(f'legend {i} not centred on the page')
    assert not bad, 'layout checks failed:\n  ' + '\n  '.join(bad)
    return len(boxes)


def check_data(D, pts):
    """Read the plotted means and whisker ends back from the artists. Means go through check_drawn: the error cells
    in full; the recall cells minus those whose paired error rate is undefined (that point cannot be placed), and the
    left-out recall cells must be exactly the partners of missing(). Whisker ends must equal min and max exactly."""
    errs, recs = {}, {}
    for p in pts:
        x, y = p['marker'].get_xdata()[0], p['marker'].get_ydata()[0]
        errs[p['setting'], p['method'], p['err']] = x
        recs[p['setting'], p['method'], p['rec']] = y
        e, r = D[p['setting'], p['method'], p['err']], D[p['setting'], p['method'], p['rec']]
        hx, hy = list(p['hbar'].get_xdata()), list(p['hbar'].get_ydata())
        vx, vy = list(p['vbar'].get_xdata()), list(p['vbar'].get_ydata())
        assert hx == [e['min'], e['max']] and hy == [r['mean'], r['mean']], ('x whisker', p['setting'], p['method'])
        assert vy == [r['min'], r['max']] and vx == [e['mean'], e['mean']], ('y whisker', p['setting'], p['method'])
    n_err = check_drawn(D, errs)
    partner = {'fdr': 'rec', 'for': 'fail_rec'}
    left_out = {(s, m, partner[e]) for s, m, e in missing(D)}
    sub = {k: v for k, v in D.items() if k not in left_out}
    n_rec = check_drawn(sub, recs)
    assert not left_out & set(recs)
    return dict(error_means=n_err, recall_means=n_rec, whisker_pairs=len(pts),
                not_placeable=sorted(f'{s} {m} {x}' for s, m, x in missing(D)))


def finish(fig, D, name, axes, shape, row_h, row_gaps, legs, bottoms, bottom_gaps, set_limits, pts, col_gap,
           leg_gap):
    """Iterate limits and layout until the outer whitespace settles, run the checks, save, and make the CVD grid."""
    margins = [sum(INK_TARGET) / 2] * 4
    ax_w = 1.7
    for it in range(12):
        set_limits(ax_w)
        H, ax_w, _ = place(fig, axes, shape, row_h, row_gaps, legs, bottoms, bottom_gaps, margins, col_gap, leg_gap)
        if it < 2:
            continue                                    # let limits and ticks settle first
        set_limits(ax_w)
        buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=PNG_DPI)
        ink = ink_margins(buf.getvalue())
        if all(INK_TARGET[0] <= v <= INK_TARGET[1] for v in ink):
            break
        margins = [mg + sum(INK_TARGET) / 2 - v for mg, v in zip(margins, ink)]
    else:
        raise AssertionError(f'outer margins did not settle: {ink}')
    n_text = checks(fig, axes, pts, legs, bottoms)
    data = check_data(D, pts)
    OUT.mkdir(exist_ok=True); CVD.mkdir(exist_ok=True)
    pdf, png = OUT / f'{name}.pdf', OUT / f'{name}.png'
    for p in (pdf, png):
        assert p.resolve().parent == OUT.resolve() and 'deterministic-verification' not in str(p.resolve())
    fig.savefig(pdf, metadata={'CreationDate': None})
    fig.savefig(png, dpi=PNG_DPI)
    plt.close(fig)
    w, h = pdf_size(pdf)
    assert abs(w - W) <= .01, (name, w)
    ink = ink_margins(str(png))
    assert all(INK_OK[0] <= v <= INK_OK[1] for v in ink), (name, 'outer whitespace', ink)
    img = Image.open(png)
    assert abs(img.info['dpi'][0] - PNG_DPI) < 1, img.info
    cvd = CVD / f'{name}_cvd.png'
    cvd_grid(img, name).save(cvd)
    md5 = {p.name: hashlib.md5(p.read_bytes()).hexdigest() for p in (pdf, png)}
    print(f'{name}: {w:.3f} x {h:.3f} in; panel {ax_w:.3f} in wide; layout passes {it + 1}')
    print(f"  ink margins (l, r, t, b) {', '.join(f'{v:.4f}' for v in ink)} in")
    print(f'  checks passed: {n_text} text boxes (overlap, page, panels), {len(pts)} markers and '
          f'{2 * len(pts)} whiskers inside, legends centred')
    print(f'  data: {data}')
    print(f'  md5 {md5}; cvd {cvd.relative_to(PLOTS.parent)}')
    return dict(w=w, h=h, md5=md5)


def pdf_size(pdf):
    m = re.search(rb'/MediaBox\s*\[\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)', pdf.read_bytes())
    x0, y0, x1, y1 = map(float, m.groups())
    return (x1 - x0) / 72, (y1 - y0) / 72


def option_a(D):
    """2 x 3 scatter, both sides in every panel."""
    fig = plt.figure(figsize=(W, 4))
    axes, pts = {}, []
    for si, (s, title) in enumerate(zip(SETTINGS, TITLES)):
        r, c = divmod(si, 3)
        ax = fig.add_axes([0, 0, .2, .2])
        setup_x(ax)
        for side in ('fail', 'success'):               # gold on top: the failure side crowds the top left
            pts += draw_side(ax, D, s, side)
        ax.set_title(title, pad=3)
        ax.set_yticks([0, 25, 50, 75, 100])
        if c:
            ax.tick_params(labelleft=False)
        else:
            ax.set_ylabel(r'$R_{\mathrm{s}}$, $R_{\mathrm{f}}$ (\%)', labelpad=1.5)
        axes[r, c] = ax
    legs = legends(fig, D)
    bottoms = [fig.text(.5, 0, r'FDR, FOR (\%)', ha='center', va='top', fontsize=FS['label']),
               fig.text(.5, 0, '\n'.join(missing_note(D)), ha='center', va='top', fontsize=FS['annotation'],
                        multialignment='center')]
    xw0, xw1, xm0, xm1 = x_extent(D)
    yw0 = min(D[p['setting'], p['method'], p['rec']]['min'] for p in pts)
    yw1 = max(D[p['setting'], p['method'], p['rec']]['max'] for p in pts)
    ym = [p['marker'].get_ydata()[0] for p in pts]
    row_h = [1.38, 1.38]

    def set_limits(ax_w):
        xl = padded(xw0, xw1, xm0, xm1, ax_w * 72, log=True)
        yl = padded(yw0, yw1, min(ym), max(ym), row_h[0] * 72, log=False, base=(0, 100))
        for ax in axes.values():
            ax.set_xlim(*xl); ax.set_ylim(*yl)

    return finish(fig, D, 'optA_scatter', axes, (2, 3), row_h, [.09], legs, bottoms, [.02, .03], set_limits, pts,
                  col_gap=.09, leg_gap=.05)


def option_b(D):
    """4 x 3, S3 arrangement: per block a success row over a failure row."""
    fig = plt.figure(figsize=(W, 6))
    axes, pts, rows = {}, [], []
    for blk in range(2):
        for side in ('success', 'fail'):
            rows.append((blk, side))
    for r, (blk, side) in enumerate(rows):
        err, rec = METRIC[side]
        for c in range(3):
            si = 3 * blk + c
            s = SETTINGS[si]
            ax = fig.add_axes([0, 0, .2, .2])
            setup_x(ax)
            pts += draw_side(ax, D, s, side)
            if side == 'success':
                ax.set_title(TITLES[si], pad=3)
            ax.set_xlabel(rf'{err.upper()} (\%)', labelpad=1)
            if c:
                ax.tick_params(labelleft=False)
            else:
                ax.set_ylabel(r'$R_{\mathrm{s}}$ (\%)' if side == 'success' else r'$R_{\mathrm{f}}$ (\%)',
                              labelpad=1.5)
            axes[r, c] = ax
    # y base per row: R_s rows 0-100; R_f rows from the 5-point step below their lowest whisker end, up to 100
    base, whisk, yticks, zoom = {}, {}, {}, []
    for r, (blk, side) in enumerate(rows):
        mine = [p for p in pts if p['ax'] in [axes[r, c] for c in range(3)]]
        lo_w = min(D[p['setting'], p['method'], p['rec']]['min'] for p in mine)
        hi_w = max(D[p['setting'], p['method'], p['rec']]['max'] for p in mine)
        whisk[r] = (lo_w, hi_w)
        if side == 'success':
            base[r] = (0, 100); yticks[r] = [0, 50, 100]
        else:
            lo = 5 * math.floor(lo_w / 5)
            base[r] = (lo, 100)
            step = 25 if 100 - lo > 45 else 20
            yticks[r] = [t for t in range(0, 101, step) if t >= lo]
            zoom.append(f"{lo}\\% ({['LQR', 'RL'][blk]})")
        for c in range(3):
            axes[r, c].set_yticks(yticks[r])
    legs = legends(fig, D)
    note = missing_note(D) + [rf"$R_{{\mathrm{{f}}}}$ rows zoomed: y axis starts at {' and '.join(zoom)}"]
    bottoms = [fig.text(.5, 0, '\n'.join(note), ha='center', va='top', fontsize=FS['annotation'],
                        multialignment='center')]
    xw0, xw1, xm0, xm1 = x_extent(D)
    row_h = [.86] * 4

    def set_limits(ax_w):
        xl = padded(xw0, xw1, xm0, xm1, ax_w * 72, log=True)
        for r in range(4):
            ym = [p['marker'].get_ydata()[0] for p in pts if p['ax'] in [axes[r, c] for c in range(3)]]
            yl = padded(*whisk[r], min(ym), max(ym), row_h[r] * 72, log=False, base=base[r])
            if rows[r][1] == 'fail':
                assert yl[0] == base[r][0], ('zoomed axis does not start where the note says', r, yl)
            for c in range(3):
                axes[r, c].set_xlim(*xl); axes[r, c].set_ylim(*yl)

    return finish(fig, D, 'optB_rows', axes, (4, 3), row_h, [.05, .1, .05], legs, bottoms, [.04], set_limits, pts,
                  col_gap=.09, leg_gap=.05)


def main():
    D = load()
    with plt.rc_context(style_rc()):
        a = option_a(D)
        b = option_b(D)
    return a, b


if __name__ == '__main__':
    main()
