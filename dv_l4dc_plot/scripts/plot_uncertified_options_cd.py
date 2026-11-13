r"""Table 2 as figures, options C and D: previews for choosing, never for the paper (dm1487, 2026-11-13).

Table 2 is the empirical comparison at the validation-fitted w = 0.2 thresholds (tab:uncertified), not a certified
result. All numbers come from uncertified_options_data.load(): six methods x six settings x five metrics, mean with
min-max over seeds, each cell's LaTeX equal to the paper's table body.

  C1 optC1_bars_numbers   one panel per setting (LQR row over RL row), methods as rows. R_s (gold) and R_f (purple)
                          bars on 0-100 with min-max whiskers; FDR printed past the end of the R_s bar, FOR past R_f.
  C2 optC2_bars_erroraxis the same bars, with FDR and FOR as markers (method marker shape, side colour, min-max
                          whiskers) on a small log error axis beside each panel, and a 5% reference line.
  D  optD_heatmap         the table as a shaded grid, LQR block over RL block, every cell printing the table's 2 dp
                          value. Gold = success side, purple = fail side, gray = abstention; darker = higher. Recalls
                          on a linear 0-100% scale; FDR, FOR and Abst. on one log10 scale from 0.1% (white) to 100%.

Style from common.py (paper style block), layout in inches from measured text as in the --restyle templates. Hard
checks: text overlap, text off the page, text over a panel, every bar/marker/whisker inside its panel, printed numbers
clear of bars, whiskers and each other, outer whitespace 0.005-0.03 in, PDF width 6.00 in. Read-back: every plotted
mean from the artists through check_drawn, whisker ends equal to min/max, every printed number equal to the table's
cell. Writes plots/options_uncertified/ (and its cvd/) only.
"""
import hashlib, io, math, re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.text import Text
from matplotlib.transforms import Bbox
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
from matplotlib.legend_handler import HandlerBase, HandlerTuple
from matplotlib.patheffects import withStroke
from PIL import Image
from common import (PLOTS, SETTINGS, TITLES, FS, FILL, LINE, LS, LW, INK, PARTIAL_TINT, METHOD_MARKER, marker_kw,
                    paper_legend_kw, style_rc)
from split import tex, drawn_ticklabels
from uncertified_options_data import load, METHODS, METRICS, missing, partial, check_drawn
from cvd_grid import grid

OUT = PLOTS / 'options_uncertified'
CVD_DIR = OUT / 'cvd'
W = 6.00                     # print width (in)
PT = 1 / 72                  # inches per point
DPI = 300                    # layout, checks and PNG at one resolution: usetex text extents shift with dpi
M = .012                     # outer margin the layout sets (in); the ink check accepts .005-.03
SEP = 2.0                    # least gap between neighbouring texts (pt)
LEG_GAP, TITLE_PAD, ROW_GAP, XL_GAP, NOTE_GAP, COL_GAP = 4.0, 2.0, 5.0, 1.5, 2.5, 8.0      # pt
NAMES = [m for _, m in METHODS]
COLOR = {'rec': LINE['success'], 'fdr': LINE['success'], 'fail_rec': LINE['fail'], 'for': LINE['fail'],
         'abstain': LINE['abstain']}
ERR_OF = {'rec': 'fdr', 'fail_rec': 'for'}
RECALLS = ('rec', 'fail_rec')
# The table's own column heads (tables/matrix_uncertified_body.tex).
HEAD = {'rec': r'$R_{\mathrm{s}}\uparrow$', 'fdr': r'FDR$\downarrow$', 'fail_rec': r'$R_{\mathrm{f}}\uparrow$',
        'for': r'FOR$\downarrow$', 'abstain': r'Abst.$\downarrow$'}
RS_RF = r'$R_{\mathrm{s}}$, $R_{\mathrm{f}}$'
MISSING = [('q3d-lqr', 'MORALS', 'fdr'), ('q3d-ppo', 'MORALS', 'fdr')]
PARTIAL = [('cp-ppo', 'MORALS', 'fdr')]
NOTE = 'MORALS FDR{} undefined in Spatial Quad LQR and RL: no Success declarations in any seed'

# Option C row geometry, in points. Data y is in points from the panel top (1 data unit = 1 pt), so these are exact.
BAR_T = 5.0                  # bar thickness
BAR_G = 1.6                  # R_s bar to R_f bar of one method (bar centres 6.6 pt apart > 6.3 pt number height)
METHOD_G = 4.4               # between methods
GROUP_G = 2.6                # extra gap between the prior methods and the two TERA rows (the table's midrule)
PAD_PT = 2.0                 # panel edge to the first and last bar
WHISK_LW, CAP_PT = .65, 1.4  # whisker width and cap half-length (pt): Figure S1's
HALO = [withStroke(linewidth=WHISK_LW + 1.1, foreground='white')]
NUM_GAP = 2.0                # whisker end to its printed number (pt)
SUP_PT = 5.0                 # superscript size of a 7 pt number (TeX scriptsize)
MS, MEW, MHALO = 3.6, .6, .5 # C2 error marker size, edge, white ring (pt)
ERR_W = .62                  # C2 error axis width (in)
# Option D cell geometry.
CELL_H = 9.5                 # pt
SET_GAP = .22                # gap between settings, in columns
BLOCK_GAP = 5.0              # pt between the LQR block's last row and the RL block's title
NUM_RE = re.compile(r'(\d+\.\d\d)(\^\{\(\d\)\})?')


def plain(cell_tex):
    """The table's cell LaTeX as printed here: digits in text mode (Times bold, like every tick label), the seed
    superscript through \text. '$97.47$' -> '97.47', '$19.25^{(2)}$' -> '19.25$^{\text{(2)}}$', '--' -> '--'."""
    if cell_tex == '--':
        return '--'
    m = NUM_RE.fullmatch(cell_tex.strip('$'))
    assert m and cell_tex == f'${m.group(0)}$', cell_tex
    return m.group(1) + (f'$^{{\\text{{{m.group(2)[2:-1]}}}}}$' if m.group(2) else '')


def unplain(printed):
    """Inverse of plain(): the cell LaTeX a printed string stands for."""
    if printed == '--':
        return '--'
    m = re.fullmatch(r'(\d+\.\d\d)(\$\^\{\\text\{(\(\d\))\}\}\$)?', printed)
    assert m, printed
    return f'${m.group(1)}' + (f'^{{{m.group(3)}}}' if m.group(2) else '') + '$'


def check_number(printed, cell):
    """A printed number equals its cell: the table's LaTeX exactly, and the mean's 2 dp rounding."""
    assert unplain(printed) == cell['tex'], (printed, cell['tex'])
    if cell['n'] == 0:
        assert printed == '--'
    else:
        assert f"{cell['mean']:.2f}" == NUM_RE.fullmatch(cell['tex'].strip('$')).group(1), (cell, printed)


def faded(color):
    return tuple(c + (1 - c) * PARTIAL_TINT for c in mcolors.to_rgb(color))


# ---------------------------------------------------------------------------------------------------- legend keys
class WhiskerKey:
    """Legend key: a min-max whisker as the bars draw it."""


class WhiskerHandler(HandlerBase):
    def create_artists(self, legend, orig, xd, yd, width, height, fontsize, trans):
        y, x0, x1 = height / 2 - yd, -xd + .1 * width, -xd + .9 * width
        return [Line2D([x0, x1], [y, y], color=INK, lw=WHISK_LW, transform=trans),
                Line2D([x0, x1], [y, y], ls='none', marker='|', markersize=2 * CAP_PT, markeredgewidth=WHISK_LW,
                       color=INK, transform=trans)]


class TextKey:
    def __init__(self, s):
        self.s = s


class TextKeyHandler(HandlerBase):
    def create_artists(self, legend, orig, xd, yd, width, height, fontsize, trans):
        # the seed superscript as a key glyph: '(2)' at superscript size, set high in the handle box. A lone TeX
        # superscript reports a box lower than its ink and pokes out of the legend's top edge.
        return [Text(width - xd, height - yd, orig.s, ha='right', va='top', fontsize=SUP_PT,
                     color='black', transform=trans)]


class RampKey:
    """Legend key: colour ramps stacked top to bottom, white end on the left."""
    def __init__(self, cmaps):
        self.cmaps = cmaps


class RampHandler(HandlerBase):
    N = 32

    def create_artists(self, legend, orig, xd, yd, width, height, fontsize, trans):
        out, k = [], len(orig.cmaps)
        gap = .25 * fontsize / 7
        hh = (height - (k - 1) * gap) / k
        for r, cmap in enumerate(orig.cmaps):
            y0 = -yd + height - (r + 1) * hh - r * gap
            for i in range(self.N):     # overlap each slice a little so no seam shows between them
                out.append(Rectangle((-xd + i * width / self.N, y0), width / self.N * (1.6 if i < self.N - 1 else 1),
                                     hh, facecolor=cmap((i + .5) / self.N), edgecolor='none', lw=0, transform=trans))
            out.append(Rectangle((-xd, y0), width, hh, fill=False, edgecolor=LINE['ref'], lw=.3, transform=trans))
        return out


HANDLERS = {WhiskerKey: WhiskerHandler(), TextKey: TextKeyHandler(), RampKey: RampHandler(),
            tuple: HandlerTuple(ndivide=None, pad=.3)}


def top_legend(fig, handles, labels, ncols, **over):
    kw = paper_legend_kw(FS['legend'])
    kw.update(over)
    for ncol in ncols:
        leg = fig.legend(handles, labels, ncol=ncol, handler_map=HANDLERS, **kw)
        fig.canvas.draw()
        if leg.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi <= W - 2 * M:
            return leg, ncol
        leg.remove()
    raise AssertionError(f'legend wider than the page at every column count in {ncols}')


# ---------------------------------------------------------------------------------------------------- hard checks
def mark_boxes(ax, line):
    """Display boxes of a Line2D's ink: the stroke between points, and each marker with its edge."""
    k = ax.figure.dpi / 72
    xy = ax.transData.transform(np.c_[line.get_xdata(), line.get_ydata()].astype(float))
    out = []
    if line.get_linestyle() not in ('None', '', ' ', 'none') and len(xy) > 1:
        lw = line.get_linewidth() * k / 2
        out.append(Bbox([[xy[:, 0].min(), xy[:, 1].min() - lw], [xy[:, 0].max(), xy[:, 1].max() + lw]]))
    mk = line.get_marker()
    if mk not in (None, 'None', '', ' ', 'none'):
        ms, mew = line.get_markersize() * k, line.get_markeredgewidth() * k
        rx, ry = ((mew / 2, (ms + mew) / 2) if mk == '|' else ((ms + mew) / 2, (ms + mew) / 2))
        out += [Bbox([[x - rx, y - ry], [x + rx, y + ry]]) for x, y in xy]
    return out


def hard_checks(fig, axes, legend, inside=(), marks=(), cells=()):
    """inside: [(Text, ax)] texts drawn inside a panel; marks: [(ax, name, Bbox)] the ink of every bar, whisker and
    marker; cells: [(Text, Bbox)] a heatmap number and its cell. Fails on overlapping text, text or legend off the
    page, text over a panel (other than its own in-panel text), an in-panel text outside its panel or over a mark,
    a mark outside its panel, a cell number outside its cell, an unescaped %, or a legend not centred."""
    fig.canvas.draw()
    rd = fig.canvas.get_renderer()
    texts = []
    for i, ax in enumerate(axes):
        texts += [(f'ax{i} tick {t.get_text()!r}', t, None) for t in drawn_ticklabels(ax.xaxis) + drawn_ticklabels(ax.yaxis)]
        texts += [(f'ax{i} {t.get_text()!r}', t, None) for t in (ax.title, ax.xaxis.label, ax.yaxis.label)
                  if t.get_visible() and t.get_text()]
    texts += [(f'figure text {t.get_text()!r}', t, None) for t in fig.texts]
    texts += [(f'in-panel {t.get_text()!r}', t, ax) for t, ax in inside]
    texts += [(f'cell {t.get_text()!r}', t, t.axes) for t, _ in cells]
    for name, t, _ in texts + [(f'legend {t.get_text()!r}', t, None) for t in legend.get_texts()]:
        assert not re.search(r'(?<!\\)%', t.get_text()), ('unescaped %', name)
    ext = [(n, t.get_window_extent(rd), own) for n, t, own in texts]
    ext.append(('legend', legend.get_window_extent(rd), None))
    page = fig.bbox
    tol = .01
    for i, (na, a, _) in enumerate(ext):
        assert a.x0 >= page.x0 - tol and a.x1 <= page.x1 + tol and a.y0 >= page.y0 - tol and a.y1 <= page.y1 + tol, \
            ('off the page', na)
        for nb, b, _ in ext[i + 1:]:
            assert not a.overlaps(b), ('text overlap', na, nb)
    panels = [(ax, ax.get_window_extent(rd)) for ax in axes]
    for na, a, own in ext:
        for ax, p in panels:
            if ax is not own:
                assert not a.overlaps(p), ('text over a panel', na)
    for t, ax in inside:
        bb, pb = t.get_window_extent(rd), ax.get_window_extent(rd)
        assert pb.x0 <= bb.x0 and bb.x1 <= pb.x1 and pb.y0 <= bb.y0 and bb.y1 <= pb.y1, ('leaves its panel', t.get_text())
        for mx, name, mb in marks:
            if mx is ax:
                assert not bb.overlaps(mb), ('printed number on a mark', t.get_text(), name)
    for ax, name, mb in marks:
        pb = ax.get_window_extent(rd)
        assert pb.x0 - tol <= mb.x0 and mb.x1 <= pb.x1 + tol and pb.y0 - tol <= mb.y0 and mb.y1 <= pb.y1 + tol, \
            ('mark outside its panel', name)
    pad = .5 * fig.dpi / 72
    for t, cb in cells:
        bb = t.get_window_extent(rd)
        assert cb.x0 + pad <= bb.x0 and bb.x1 <= cb.x1 - pad and cb.y0 + pad <= bb.y0 and bb.y1 <= cb.y1 - pad, \
            ('number leaves its cell', t.get_text())
    lb = legend.get_window_extent(rd)
    assert abs((lb.x0 + lb.x1) / 2 - page.x1 / 2) / fig.dpi < .005, 'legend not centred'
    return len(ext)


def ink_margins(png):
    """Unpainted outer margin (left, right, top, bottom) of a raster, in inches."""
    img = Image.open(png)
    dpi = img.info['dpi'][0]
    a = np.asarray(img.convert('L'))
    ink = np.argwhere(a < 250)
    (r0, c0), (r1, c1) = ink.min(0), ink.max(0)
    h, w = a.shape
    return c0 / dpi, (w - 1 - c1) / dpi, r0 / dpi, (h - 1 - r1) / dpi


def pdf_size(pdf):
    m = re.search(rb'/MediaBox\s*\[\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)', pdf.read_bytes())
    x0, y0, x1, y1 = map(float, m.groups())
    return (x1 - x0) / 72, (y1 - y0) / 72


def save_option(fig, name):
    """plots/options_uncertified/<name>.{pdf,png} (PNG 300 dpi) at the figure's own size, and the CVD grid."""
    OUT.mkdir(parents=True, exist_ok=True)
    CVD_DIR.mkdir(exist_ok=True)
    pdf, png, cvd = OUT / f'{name}.pdf', OUT / f'{name}.png', CVD_DIR / f'{name}_cvd.png'
    for p in (pdf, png, cvd):
        assert OUT.resolve() in p.resolve().parents and 'deterministic-verification' not in str(p.resolve())
    fig.savefig(pdf, metadata={'CreationDate': None})
    fig.savefig(png, dpi=300)
    plt.close(fig)
    w, h = pdf_size(pdf)
    assert abs(w - W) <= .01, (name, w)
    ink = ink_margins(png)
    assert all(.005 <= v <= .03 for v in ink), ('outer whitespace', name, ink)
    img = Image.open(png)
    assert abs(img.info['dpi'][0] - 300) < 1
    grid(img, name).save(cvd)
    md5 = hashlib.md5(pdf.read_bytes()).hexdigest()
    print(f'{name}: {w:.3f} x {h:.3f} in, pdf md5 {md5}, png md5 {hashlib.md5(png.read_bytes()).hexdigest()}')
    print(f"  outer ink margins (l, r, t, b) {', '.join(f'{v:.4f}' for v in ink)} in")
    return dict(width=w, height=h, md5=md5, ink=ink)


# ---------------------------------------------------------------------------------------------------- option C
def c_rows():
    """Bar centres {(method index, recall metric): y}, method label positions and the panel height, in points."""
    y, centres, labels = PAD_PT, {}, []
    for i in range(len(NAMES)):
        if i == 4:
            y += GROUP_G
        centres[i, 'rec'] = y + BAR_T / 2
        centres[i, 'fail_rec'] = y + 1.5 * BAR_T + BAR_G
        labels.append(y + BAR_T + BAR_G / 2)
        y += 2 * BAR_T + BAR_G + (METHOD_G if i < len(NAMES) - 1 else 0)
    return centres, labels, y + PAD_PT


def c_layout(fig, panels, legend, xlabel, note, err, set_xlims):
    """Place the 2 x 3 panels (settings in order, LQR row first) on a W-wide page as tight as the measured text
    allows and size the page to fit. panels[i] = dict(axes=[rec] or [rec, err], title=figure Text (va bottom)).
    Repeats draw-measure-place until every decoration settles. set_xlims(rec_w_in) sets the recall x limits."""
    _, _, ph_pt = c_rows()
    ph = ph_pt * PT
    dec = dict(left=.6, right=.05, gaps=[.12, .12], inner=.08, leg=.2, top=[.12, .12], below=[.12, .12],
               xl=.12 if xlabel else 0., note=.12)
    for it in range(40):
        pw = (W - 2 * M - dec['left'] - dec['right'] - sum(dec['gaps'])) / 3
        rec_w = pw - (ERR_W + dec['inner'] if err else 0)
        H = (2 * M + dec['leg'] + LEG_GAP * PT + sum(dec['top']) + 2 * TITLE_PAD * PT + 2 * ph + sum(dec['below'])
             + ROW_GAP * PT + (XL_GAP * PT + dec['xl'] if xlabel else 0) + NOTE_GAP * PT + dec['note'])
        fig.set_size_inches(W, H)
        xs = [M + dec['left'] + c * pw + sum(dec['gaps'][:c]) for c in range(3)]
        y = H - M - dec['leg'] - LEG_GAP * PT
        for r in range(2):
            y -= dec['top'][r]
            for c in range(3):
                p = panels[3 * r + c]
                p['title'].set_position(((xs[c] + pw / 2) / W, y / H))
                p['axes'][0].set_position([xs[c] / W, (y - TITLE_PAD * PT - ph) / H, rec_w / W, ph / H])
                if err:
                    p['axes'][1].set_position([(xs[c] + rec_w + dec['inner']) / W, (y - TITLE_PAD * PT - ph) / H,
                                               ERR_W / W, ph / H])
            y -= TITLE_PAD * PT + ph + dec['below'][r] + (ROW_GAP * PT if r == 0 else 0)
        block = (xs[0], xs[2] + pw)
        if xlabel:
            y -= XL_GAP * PT + dec['xl']
            xlabel.set_position((sum(block) / 2 / W, y / H))
        note.set_position((sum(block) / 2 / W, M / H))
        legend.set_bbox_to_anchor((.5, (H - M) / H), transform=fig.transFigure)
        set_xlims(rec_w)

        fig.canvas.draw()
        rd = fig.canvas.get_renderer()
        box = lambda a: np.array(a.get_window_extent(rd).extents) / fig.dpi
        pos = lambda ax: np.array(ax.get_position().extents) * [W, H, W, H]
        # get_tightbbox trims an x label wider than its axes, so take the label's own box too
        tight = lambda ax: np.array(Bbox.union([ax.get_tightbbox(rd)] + ([ax.xaxis.label.get_window_extent(rd)]
                                               if ax.xaxis.label.get_text() else [])).extents) / fig.dpi
        over = {}                                                  # (left, right) overhang past each axes
        for p in panels:
            for ax in p['axes']:
                a, t = pos(ax), tight(ax)
                over[ax] = (a[0] - t[0], t[2] - a[2])
        top = [max(box(panels[3 * r + c]['title'])[3] - box(panels[3 * r + c]['title'])[1] for c in range(3))
               for r in range(2)]
        below = [max(pos(ax)[1] - tight(ax)[1] for c in range(3) for ax in panels[3 * r + c]['axes']) for r in range(2)]
        t_over = lambda p: (pos(p['axes'][0])[0] - box(p['title'])[0], box(p['title'])[2] - pos(p['axes'][-1])[2])
        left = max(max(over[panels[3 * r]['axes'][0]][0], t_over(panels[3 * r])[0]) for r in range(2))
        right = max(0., *[max(over[panels[3 * r + 2]['axes'][-1]][1], t_over(panels[3 * r + 2])[1]) for r in range(2)])
        gaps = []
        for c in range(2):
            need = [over[panels[3 * r + c]['axes'][-1]][1] + over[panels[3 * r + c + 1]['axes'][0]][0] for r in range(2)]
            need += [t_over(panels[3 * r + c])[1] + t_over(panels[3 * r + c + 1])[0] for r in range(2)]
            gaps.append(max(COL_GAP * PT, max(need) + SEP * PT))
        inner = (max(over[p['axes'][0]][1] + over[p['axes'][1]][0] for p in panels) + SEP * PT) if err else 0.
        new = dict(left=left, right=right, gaps=gaps, inner=inner, leg=box(legend)[3] - box(legend)[1], top=top,
                   below=below, xl=(box(xlabel)[3] - box(xlabel)[1]) if xlabel else 0., note=box(note)[3] - box(note)[1])
        flat = lambda d: np.hstack([d[k] for k in sorted(d)])
        settled = np.allclose(flat(new), flat(dec), atol=1e-7)
        dec = new
        if settled:
            break
    else:
        raise AssertionError('option C layout did not settle')
    return dict(height=H, panel_w=pw, rec_w=rec_w, panel_h=ph, iterations=it + 1)


def draw_c(D, variant):
    """variant 'numbers' (C1) or 'erroraxis' (C2). Returns the figure and what the checks need."""
    err = variant == 'erroraxis'
    centres, label_y, ph_pt = c_rows()
    fig = plt.figure(figsize=(W, 4), dpi=DPI)
    panels, reg = [], dict(bar={}, whisker={}, number={}, marker={}, ewhisker={})
    inside, marks = [], []
    rd = fig.canvas.get_renderer()
    for si, (s, title) in enumerate(zip(SETTINGS, TITLES)):
        rec = fig.add_axes([.1, .1, .2, .2])
        axes = [rec]
        rec.set_ylim(ph_pt, 0)
        rec.grid(axis='x', alpha=.2, lw=.45)
        rec.set_axisbelow(True)
        rec.spines['left'].set_position(('data', 0))
        rec.spines['bottom'].set_bounds(0, 100)
        rec.xaxis.set_major_locator(FixedLocator([0, 50, 100]))
        rec.set_yticks(label_y, NAMES if si % 3 == 0 else [''] * len(NAMES))
        rec.tick_params(axis='y', length=0, pad=2.5)
        for i, m in enumerate(NAMES):
            for metric in RECALLS:
                c, y = D[s, m, metric], centres[i, metric]
                bar = rec.barh(y, c['mean'], height=BAR_T, left=0, color=COLOR[metric], edgecolor='white',
                               linewidth=.25, zorder=3)[0]
                wl, = rec.plot([c['min'], c['max']], [y, y], color=INK, lw=WHISK_LW, solid_capstyle='butt', zorder=4,
                               path_effects=HALO)
                cp, = rec.plot([c['min'], c['max']], [y, y], ls='none', marker='|', markersize=2 * CAP_PT,
                               markeredgewidth=WHISK_LW, color=INK, zorder=4, path_effects=HALO)
                reg['bar'][s, m, metric] = (rec, bar, y)
                reg['whisker'][s, m, metric] = (wl, cp)
                if not err:                 # C1: the matching error printed just past the bar's whisker
                    e = D[s, m, ERR_OF[metric]]
                    t = rec.annotate(plain(e['tex']), (c['max'], y), xytext=(NUM_GAP, 0), textcoords='offset points',
                                     ha='left', va='center', fontsize=FS['annotation'], color=COLOR[metric], zorder=5,
                                     annotation_clip=False)
                    reg['number'][s, m, ERR_OF[metric]] = (t, rec, c['max'], y)
                    inside.append((t, rec))
        if err:
            ex = fig.add_axes([.5, .1, .1, .2])
            axes.append(ex)
            ex.set_xscale('log')
            ex.set_ylim(ph_pt, 0)
            ex.set_yticks([])
            ex.grid(axis='x', alpha=.2, lw=.45)
            ex.set_axisbelow(True)
            ex.xaxis.set_major_locator(FixedLocator([.1, 1, 10, 100]))
            ex.xaxis.set_major_formatter(FixedFormatter(['0.1', '1', '10', '100']))
            ex.xaxis.set_minor_locator(NullLocator())
            ex.axvline(5, color=LINE['ref'], lw=LW['ref'], ls=LS['ref'], zorder=1)
            for i, m in enumerate(NAMES):
                for metric in RECALLS:
                    em = ERR_OF[metric]
                    e, y = D[s, m, em], centres[i, metric]
                    if e['n'] == 0:
                        continue
                    color = COLOR[em]
                    kw = marker_kw(m, color if e['n'] == 3 else faded(color), MS, MEW)
                    wl, = ex.plot([e['min'], e['max']], [y, y], color=color, lw=.6, solid_capstyle='butt', zorder=2)
                    mk, = ex.plot([e['mean']], [y], ls='none', zorder=4,
                                  path_effects=[withStroke(linewidth=MEW + 2 * MHALO, foreground='white')], **kw)
                    reg['marker'][s, m, em] = (ex, mk, y, e['n'])
                    reg['ewhisker'][s, m, em] = wl
        title = fig.text(.5, .5, tex(title), ha='center', va='bottom', fontsize=FS['title'])
        panels.append(dict(axes=axes, title=title))
        if si >= 3:
            if err:
                rec.set_xlabel(tex(RS_RF + ' (%)'), labelpad=1.5)
                axes[1].set_xlabel(tex('FDR, FOR (%)'), labelpad=1.5)
    # error axis limits: every whisker end inside, every marker (edge and ring) clear of the edge by 0.5 pt
    if err:
        vals = [D[k] for k in D if k[2] in ('fdr', 'for') and D[k]['n'] > 0]
        wlo, whi = math.log10(min(v['min'] for v in vals)), math.log10(max(v['max'] for v in vals))
        mlo, mhi = math.log10(min(v['mean'] for v in vals)), math.log10(max(v['mean'] for v in vals))
        r, rw = MS / 2 + MEW / 2 + MHALO + .5, .6 / 2 + .5
        lo, hi = wlo, whi
        for _ in range(50):
            d = (hi - lo) / (ERR_W / PT)
            lo, hi = min(wlo - rw * d, mlo - r * d), max(whi + rw * d, mhi + r * d)
        err_lim = (10 ** lo, 10 ** hi)
        for p in panels:
            p['axes'][1].set_xlim(*err_lim)
    # handles and texts outside the panels
    if err:
        handles = [Patch(facecolor=COLOR['rec'], edgecolor='none', lw=0), Patch(facecolor=COLOR['fail_rec'], edgecolor='none', lw=0), WhiskerKey(),
                   Line2D([], [], ls='none', **marker_kw('MORALS', faded(COLOR['fdr']), MS, MEW)),
                   Line2D([], [], color=LINE['ref'], lw=LW['ref'], ls=LS['ref'])]
        labels = [r'$R_{\mathrm{s}}$ bar, FDR marker', r'$R_{\mathrm{f}}$ bar, FOR marker', 'min to max over seeds',
                  'FDR from 2 seeds', '5% reference']
        xlabel = None
        note = fig.text(.5, 0, tex(NOTE.format('')), ha='center', va='bottom', fontsize=FS['annotation'])
    else:
        handles = [Patch(facecolor=COLOR['rec'], edgecolor='none', lw=0), Patch(facecolor=COLOR['fail_rec'], edgecolor='none', lw=0), WhiskerKey(), TextKey('(2)')]
        labels = [r'$R_{\mathrm{s}}$ bar, FDR (%) printed at its end', r'$R_{\mathrm{f}}$ bar, FOR (%) printed at its end',
                  'min to max over 3 seeds', 'mean of 2 seeds']
        xlabel = fig.text(.5, 0, tex(RS_RF + ' (%)'), ha='center', va='bottom', fontsize=FS['label'])
        note = fig.text(.5, 0, tex(NOTE.format(' (--)')), ha='center', va='bottom', fontsize=FS['annotation'])
    legend, ncol = top_legend(fig, handles, [tex(l) for l in labels], (len(handles), 3, 2))

    rec_axes = [p['axes'][0] for p in panels]
    widths = {}
    if not err:
        for t, ax, anchor, y in reg['number'].values():
            widths[t] = t.get_window_extent(rd).width / fig.dpi * 72

    def set_xlims(rec_w):
        """One recall scale for every panel. C1: room right of 100 for the widest number that sits past a bar
        near 100. Left: the whisker cap at 0 stays inside the panel."""
        w_pt = rec_w / PT
        left_pt = WHISK_LW / 2 + .3
        if err:
            right_pt = WHISK_LW / 2 + .3
            xmax = 100 * (w_pt - left_pt) / (w_pt - left_pt - right_pt)
        else:
            # anchor * ppu + NUM_GAP + width + 0.5 <= w_pt - left_pt, with ppu = (w_pt - left_pt) / xmax
            xmax = max(anchor * (w_pt - left_pt) / (w_pt - left_pt - NUM_GAP - widths[t] - .5)
                       for t, _, anchor, _ in reg['number'].values())
            xmax = max(xmax, 100 * (w_pt - left_pt) / (w_pt - left_pt - WHISK_LW / 2 - .3))
        xmin = -left_pt / (w_pt - left_pt) * xmax
        for ax in rec_axes:
            ax.set_xlim(xmin, xmax)

    geo = c_layout(fig, panels, legend, xlabel, note, err, set_xlims)
    fig.canvas.draw()
    for (s, m, metric), (ax, bar, y) in reg['bar'].items():
        marks.append((ax, f'bar {s} {m} {metric}', bar.get_window_extent()))
        for line in reg['whisker'][s, m, metric]:
            marks += [(ax, f'whisker {s} {m} {metric}', b) for b in mark_boxes(ax, line)]
    for (s, m, em), (ax, mk, y, n) in reg['marker'].items():
        marks += [(ax, f'marker {s} {m} {em}', b) for b in mark_boxes(ax, mk)]
        marks += [(ax, f'error whisker {s} {m} {em}', b) for b in mark_boxes(ax, reg['ewhisker'][s, m, em])]
    all_axes = [a for p in panels for a in p['axes']]
    n_text = hard_checks(fig, all_axes, legend, inside=inside, marks=marks)
    if err:
        for p in panels:
            ex = p['axes'][1]
            assert ex.get_xlim()[0] < 5 < ex.get_xlim()[1]
            assert sum(1 for l in ex.lines if list(l.get_xdata()) == [5, 5] and l.get_color() == LINE['ref']) == 1
    return fig, dict(reg=reg, geo=geo, n_text=n_text, n_marks=len(marks), legend_ncol=ncol,
                     err_lim=err_lim if err else None, xlim=rec_axes[0].get_xlim())


def readback_c(D, out, variant):
    """Plotted means from the artists through check_drawn; whisker ends equal min/max; printed numbers equal the
    table's cells; each number or marker sits on its bar's row."""
    reg, drawn = out['reg'], {}
    for (s, m, metric), (ax, bar, y) in reg['bar'].items():
        c = D[s, m, metric]
        assert bar.get_x() == 0 and abs(bar.get_y() + bar.get_height() / 2 - y) < 1e-12
        assert mcolors.same_color(bar.get_facecolor(), COLOR[metric])
        drawn[s, m, metric] = bar.get_width()
        for line in reg['whisker'][s, m, metric]:
            assert list(line.get_xdata()) == [c['min'], c['max']] and list(line.get_ydata()) == [y, y], (s, m, metric)
    whiskers = 2 * len(reg['bar'])
    if variant == 'numbers':
        assert check_drawn(D, drawn) == 72
        for k in [(s, m, em) for s in SETTINGS for m in NAMES for em in ('fdr', 'for')]:
            t, ax, anchor, y = reg['number'][k]
            check_number(t.get_text(), D[k])
            rk = (k[0], k[1], 'rec' if k[2] == 'fdr' else 'fail_rec')
            assert t.xy == (D[rk]['max'], reg['bar'][rk][2]), k
            assert mcolors.same_color(t.get_color(), COLOR[k[2]])
        assert [k for k in reg['number'] if reg['number'][k][0].get_text() == '--'] == MISSING
        assert [k for k in reg['number'] if '(2)' in reg['number'][k][0].get_text()] == PARTIAL
        printed = len(reg['number'])
    else:
        for k, (ax, mk, y, n) in reg['marker'].items():
            assert len(mk.get_xdata()) == 1
            drawn[k] = mk.get_xdata()[0]
            rk = (k[0], k[1], 'rec' if k[2] == 'fdr' else 'fail_rec')
            assert mk.get_ydata()[0] == reg['bar'][rk][2]
            assert list(reg['ewhisker'][k].get_xdata()) == [D[k]['min'], D[k]['max']], k
            want = COLOR[k[2]] if n == 3 else faded(COLOR[k[2]])
            assert mcolors.same_color(mk.get_markeredgecolor(), want), k
            assert mk.get_marker() == METHOD_MARKER[k[1]]['marker'], k
        assert check_drawn(D, drawn) == 72 + 70
        assert sorted(k for k, v in reg['marker'].items() if v[3] < 3) == PARTIAL
        whiskers += len(reg['ewhisker'])
        printed = 0
    return dict(means=len(drawn), whiskers=whiskers, numbers=printed)


# ---------------------------------------------------------------------------------------------------- option D
def d_maps():
    """Colour map and norm per metric. Gold ramp white -> FILL success -> LINE success; purple white -> fail;
    gray white -> FILL abstain -> LINE abstain. Recalls linear 0-100; FDR, FOR, Abst. log10 from 0.1 to 100 (values
    at or below 0.1, Abst. 0.00 included, take the white end)."""
    gold = mcolors.LinearSegmentedColormap.from_list('success', ['white', FILL['success'], LINE['success']])
    purple = mcolors.LinearSegmentedColormap.from_list('fail', ['white', FILL['fail']])
    gray = mcolors.LinearSegmentedColormap.from_list('abstain', ['white', FILL['abstain'], LINE['abstain']])
    for c in (gold, purple, gray):
        c.set_bad('white')
    lin = mcolors.Normalize(0, 100)
    log = mcolors.FuncNorm((lambda x: np.log10(np.maximum(x, .1)), lambda y: 10 ** y), vmin=.1, vmax=100)
    cmaps = dict(rec=gold, fdr=gold, fail_rec=purple, abstain=gray)
    cmaps['for'] = purple
    norms = dict(rec=lin, fail_rec=lin, fdr=log, abstain=log)
    norms['for'] = log
    return cmaps, norms


def rel_lum(rgb):
    c = np.asarray(mcolors.to_rgb(rgb))
    lin = np.where(c <= .04045, c / 12.92, ((c + .055) / 1.055) ** 2.4)
    return float(lin @ [.2126, .7152, .0722])


def contrast(a, b):
    la, lb = sorted([rel_lum(a), rel_lum(b)], reverse=True)
    return (la + .05) / (lb + .05)


def d_rows():
    """Row edges in points (a gap row between the prior methods and the TERA rows) and the method -> row index."""
    edges, idx, y = [0.], {}, 0.
    for i in range(len(NAMES)):
        if i == 4:
            y += GROUP_G
            edges.append(y)
        idx[i] = len(edges) - 1
        y += CELL_H
        edges.append(y)
    return np.array(edges), idx


def draw_d(D):
    cmaps, norms = d_maps()
    edges, idx = d_rows()
    gap_rows = sorted(set(range(len(edges) - 1)) - set(idx.values()))
    blocks = [SETTINGS[:3], SETTINGS[3:]]
    assert all(TITLES[SETTINGS.index(s)].endswith(' LQR') for s in blocks[0])
    assert all(TITLES[SETTINGS.index(s)].endswith(' RL') for s in blocks[1])
    ncols = 3 * len(METRICS) + 2 * SET_GAP
    fig = plt.figure(figsize=(W, 3), dpi=DPI)
    axes, titles, meshes, texts, cells = [], [], {}, {}, []
    for b, sets in enumerate(blocks):
        ax = fig.add_axes([.1, .1, .8, .3])
        axes.append(ax)
        centres = []
        for k, s in enumerate(sets):
            for j, metric in enumerate(METRICS):
                x0 = k * (len(METRICS) + SET_GAP) + j
                centres.append(x0 + .5)
                vals = np.ma.masked_all(len(edges) - 1)
                for i, m in enumerate(NAMES):
                    if D[s, m, metric]['n'] > 0:
                        vals[idx[i]] = D[s, m, metric]['mean']
                mesh = ax.pcolormesh([x0, x0 + 1], edges, vals.reshape(-1, 1), cmap=cmaps[metric], norm=norms[metric],
                                     edgecolors='white', linewidth=.6, antialiased=True)
                meshes[s, metric] = mesh
                rgba = mesh.to_rgba(vals)
                for i, m in enumerate(NAMES):
                    cell = D[s, m, metric]
                    face = rgba[idx[i]][:3] if cell['n'] > 0 else (1., 1., 1.)
                    ink = 'white' if contrast(face, 'white') > contrast(face, 'black') else 'black'
                    t = ax.text(x0 + .5, (edges[idx[i]] + edges[idx[i] + 1]) / 2, plain(cell['tex']), ha='center',
                                va='center', fontsize=FS['annotation'], color=ink, zorder=3)
                    texts[s, m, metric] = (t, face)
        ax.set_xlim(0, ncols)
        ax.set_ylim(edges[-1], 0)
        ax.set_yticks([(edges[idx[i]] + edges[idx[i] + 1]) / 2 for i in range(len(NAMES))], NAMES)
        ax.xaxis.tick_top()
        ax.set_xticks(centres, [tex(HEAD[m]) for _ in sets for m in METRICS])
        ax.tick_params(length=0, pad=2.0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        for k, s in enumerate(sets):
            titles.append((fig.text(.5, .5, tex(TITLES[SETTINGS.index(s)]), ha='center', va='bottom',
                                    fontsize=FS['title']), ax, k * (len(METRICS) + SET_GAP) + len(METRICS) / 2))
    gold, purple, gray = cmaps['rec'], cmaps['fail_rec'], cmaps['abstain']
    handles = [RampKey([gold, purple]), RampKey([gold, purple, gray]), TextKey('(2)')]
    labels = [RS_RF + ': 0 to 100% (linear), darker = higher', 'FDR, FOR, Abst.: 0.1 to 100% (log), darker = higher',
              'mean of 2 seeds']
    legend, ncol = top_legend(fig, handles, [tex(l) for l in labels], (3, 2, 1), handlelength=2.6, handleheight=1.35)
    note = fig.text(.5, 0, tex(NOTE.format(' (--)')), ha='center', va='bottom', fontsize=FS['annotation'])

    ph = edges[-1] * PT
    dec = dict(left=.6, right=0., leg=.25, title=[.11, .11], head=[.1, .1], note=.12)
    for it in range(40):
        cw = (W - 2 * M - dec['left'] - dec['right']) / ncols
        H = (2 * M + dec['leg'] + LEG_GAP * PT + sum(dec['title']) + 2 * TITLE_PAD * PT + sum(dec['head']) + 2 * ph
             + BLOCK_GAP * PT + NOTE_GAP * PT + dec['note'])
        fig.set_size_inches(W, H)
        y = H - M - dec['leg'] - LEG_GAP * PT
        x0 = M + dec['left']
        for b, ax in enumerate(axes):
            y -= dec['title'][b]
            for t, tax, xc in titles:
                if tax is ax:
                    t.set_position(((x0 + xc * cw) / W, y / H))
            y -= TITLE_PAD * PT + dec['head'][b]
            ax.set_position([x0 / W, (y - ph) / H, ncols * cw / W, ph / H])
            y -= ph + BLOCK_GAP * PT
        note.set_position(((x0 + ncols * cw / 2) / W, M / H))
        legend.set_bbox_to_anchor((.5, (H - M) / H), transform=fig.transFigure)
        fig.canvas.draw()
        rd = fig.canvas.get_renderer()
        box = lambda a: np.array(a.get_window_extent(rd).extents) / fig.dpi
        pos = lambda a: np.array(a.get_position().extents) * [W, H, W, H]
        tight = lambda a: np.array(a.get_tightbbox(rd).extents) / fig.dpi
        new = dict(left=max(pos(ax)[0] - tight(ax)[0] for ax in axes),
                   right=max(0., *[tight(ax)[2] - pos(ax)[2] for ax in axes],
                             *[box(t)[2] - (W - M - dec['right']) + dec['right'] for t, _, _ in titles]),
                   leg=box(legend)[3] - box(legend)[1],
                   title=[max(box(t)[3] - box(t)[1] for t, tax, _ in titles if tax is ax) for ax in axes],
                   head=[tight(ax)[3] - pos(ax)[3] for ax in axes], note=box(note)[3] - box(note)[1])
        flat = lambda d: np.hstack([d[k] for k in sorted(d)])
        settled = np.allclose(flat(new), flat(dec), atol=1e-7)
        dec = new
        if settled:
            break
    else:
        raise AssertionError('option D layout did not settle')
    fig.canvas.draw()
    for (s, m, metric), (t, face) in texts.items():
        ax = t.axes
        x0d = t.get_position()[0] - .5
        i = NAMES.index(m)
        cb = Bbox(ax.transData.transform([[x0d, edges[idx[i] + 1]], [x0d + 1, edges[idx[i]]]]))
        cells.append((t, cb))
    n_text = hard_checks(fig, axes, legend, cells=cells)
    return fig, dict(meshes=meshes, texts=texts, idx=idx, gap_rows=gap_rows, n_text=n_text, legend_ncol=ncol,
                     height=H, cell_w=cw, cell_h=CELL_H, iterations=it + 1, norms=norms)


def readback_d(D, out):
    """Means from the mesh arrays through check_drawn; n = 0 cells masked and white; every cell's printed text equals
    the table's cell; text colour is whichever of black and white contrasts more with the drawn cell colour, and every
    cell clears 4.5:1."""
    drawn, worst = {}, (99., None)
    for (s, metric), mesh in out['meshes'].items():
        arr = np.ma.ravel(mesh.get_array())          # pcolormesh keeps the (rows, 1) shape
        for g in out['gap_rows']:
            assert np.ma.getmaskarray(arr)[g]
        rgba = mesh.to_rgba(arr)
        for i, m in enumerate(NAMES):
            r = out['idx'][i]
            cell = D[s, m, metric]
            if np.ma.getmaskarray(arr)[r]:
                assert cell['n'] == 0 and mcolors.same_color(rgba[r], 'white'), (s, m, metric)
            else:
                drawn[s, m, metric] = float(arr[r])
            t, face = out['texts'][s, m, metric]
            assert np.allclose(face, rgba[r][:3]), (s, m, metric)
            check_number(t.get_text(), cell)
            other = 'black' if t.get_color() == 'white' else 'white'
            cr = contrast(face, t.get_color())
            assert cr >= contrast(face, other) and cr >= 4.5, (s, m, metric, cr)
            worst = min(worst, (cr, (s, m, metric)))
    assert check_drawn(D, drawn) == 178
    assert sorted(k for k, (t, _) in out['texts'].items() if t.get_text() == '--') == MISSING
    assert sorted(k for k, (t, _) in out['texts'].items() if '(2)' in t.get_text()) == PARTIAL
    return dict(means=len(drawn), numbers=len(out['texts']), worst_contrast=worst)


def main():
    D = load()
    assert missing(D) == MISSING and partial(D) == PARTIAL, (missing(D), partial(D))
    results = {}
    with plt.rc_context(style_rc()):
        for name, variant in [('optC1_bars_numbers', 'numbers'), ('optC2_bars_erroraxis', 'erroraxis')]:
            fig, out = draw_c(D, variant)
            data = readback_c(D, out, variant)
            size = save_option(fig, name)
            print(f"  layout: panel {out['geo']['panel_w']:.3f} x {out['geo']['panel_h']:.3f} in (recall axis "
                  f"{out['geo']['rec_w']:.3f} in), legend {out['legend_ncol']} columns, {out['geo']['iterations']} "
                  f"passes; recall x limits {out['xlim'][0]:.3f} to {out['xlim'][1]:.3f}"
                  + (f"; error axis {out['err_lim'][0]:.4f} to {out['err_lim'][1]:.2f} (log)" if out['err_lim'] else ''))
            print(f"  checks: {out['n_text']} texts, {out['n_marks']} marks inside their panels; read-back {data}")
            results[name] = size
        fig, out = draw_d(D)
        data = readback_d(D, out)
        size = save_option(fig, 'optD_heatmap')
        print(f"  layout: cell {out['cell_w']:.3f} in x {out['cell_h']:.1f} pt, legend {out['legend_ncol']} columns, "
              f"{out['iterations']} passes")
        print(f"  checks: {out['n_text']} texts; read-back {data}")
        results['optD_heatmap'] = size
    return results


if __name__ == '__main__':
    main()
