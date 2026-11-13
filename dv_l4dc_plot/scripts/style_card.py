"""Style card for the paper figures: plots/style/style_card.{pdf,png,txt}.

Everything on the card is read from common.py at run time (fills, line colours and styles, method markers, font
sizes, the signal colormap), so the card cannot disagree with the figures. The TikZ colours are parsed from the
paper clone's figures/*.tex (read only) and each one is matched against the scheme.

The .txt lists CIELAB dE76 between every colour pair that meets in some figure, under normal vision, grayscale and
the three Machado (2009) full-severity simulations (cvd_grid.py), and flags pairs under 10 (collapse) or 20 (close).
"""
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib.patheffects import withStroke
from common import (PLOTS, PAPER, FILL, LINE, LS, LW, FS, METHOD_MARKER, SIGNAL_CMAP, ABSTAIN_KIND, INK,
                    PARTIAL_TINT, PRINT_WIDTH, marker_kw, style_rc)
from cvd_grid import simulate

OUT = PLOTS / 'style'
W = 6.0
TIKZ = ['uncertainty_sources/uncertainty_sources.tex', 'pipeline/pipeline.tex',
        'abstention_triangle/abstention_triangle.tex']
BAND_ALPHA = {'Figure 6': .09, 'Figure S2': .12}       # set in plot_learning_curves.py / plot_adaptive_fraction_certified.py


def esc(s):
    return s.replace('#', r'\#').replace('_', r'\_').replace('%', r'\%')


def hexrgb(h):
    return np.array(matplotlib.colors.to_rgb(h))


def lab(rgb):
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    xyz = lin @ np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]).T
    xyz = xyz / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > (6 / 29) ** 3, np.cbrt(xyz), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])


def tint(h, a):
    return a * hexrgb(h) + (1 - a) * np.ones(3)


def tikz_colours():
    """{file: [(name, model, value)]} from the paper clone's TikZ sources."""
    out = {}
    for rel in TIKZ:
        s = (PAPER / 'figures' / rel).read_text()
        out[rel] = re.findall(r'\\definecolor\{([^}]*)\}\{([^}]*)\}\{([^}]*)\}', s)
    return out


SCHEME = {**{f'FILL.{k}': v for k, v in FILL.items() if k != 'boundary'}, **{f'LINE.{k}': v for k, v in LINE.items()},
          **{f'ABSTAIN_KIND.{k}': v for k, v in ABSTAIN_KIND.items()}, 'INK': INK}


def scheme_name(h):
    names = [k for k, v in SCHEME.items() if v.lower() == ('#' + h).lower()]
    return ' = '.join(names) if names else None


def pairs():
    """(figure, a, b, rgb_a, rgb_b): colour pairs that sit next to each other somewhere."""
    P = []
    fills = ['success', 'fail', 'fp', 'fn', 'abstain']
    for i, a in enumerate(fills):
        for b in fills[i + 1:]:
            P.append(('Fig 1, TikZ', f'FILL.{a}', f'FILL.{b}', hexrgb(FILL[a]), hexrgb(FILL[b])))
    f5 = {'FILL.success': FILL['success'], 'FILL.fail': FILL['fail'], 'FILL.fp': FILL['fp'], 'FILL.fn': FILL['fn'],
          **{f'ABSTAIN_KIND.{k}': v for k, v in ABSTAIN_KIND.items() if k != 'weak'}}
    k5 = list(f5)
    for i, a in enumerate(k5):
        for b in k5[i + 1:]:
            if not {a, b} <= {'FILL.success', 'FILL.fail', 'FILL.fp', 'FILL.fn'}:      # Fig 1 pairs listed above
                P.append(('Fig 5', a, b, hexrgb(f5[a]), hexrgb(f5[b])))
    for k in ('success', 'fail', 'abstain', 'ref'):
        P.append(('lines on white', f'LINE.{k}', 'white', hexrgb(LINE[k]), np.ones(3)))
    P.append(('Figs 6, S1-S3', 'LINE.success', 'LINE.fail', hexrgb(LINE['success']), hexrgb(LINE['fail'])))
    for fig, a in BAND_ALPHA.items():
        P.append((f'{fig} bands (alpha {a})', 'success band', 'fail band', tint(LINE['success'], a), tint(LINE['fail'], a)))
        P.append((f'{fig} bands (alpha {a})', 'success band', 'white', tint(LINE['success'], a), np.ones(3)))
    P.append(('Figs 6, S2 partial seeds', 'LINE.success faded', 'LINE.success', tint(LINE['success'], 1 - PARTIAL_TINT),
              hexrgb(LINE['success'])))
    return P


def pair_table():
    views = ['normal', 'grayscale', 'protanopia', 'deuteranopia', 'tritanopia']
    rows = []
    for fig, a, b, ca, cb in pairs():
        d = {}
        for v in views:
            sa, sb = (ca, cb) if v == 'normal' else (simulate(ca, v), simulate(cb, v))
            d[v] = float(np.linalg.norm(lab(sa) - lab(sb)))
        rows.append((fig, a, b, d))
    return views, rows


def draw():
    plt.rcParams.update(style_rc())
    H = 6.55
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis('off')
    head = dict(fontsize=FS['title'], va='baseline', ha='left')
    small = dict(fontsize=FS['tick'], va='top', ha='left')

    y = .18
    ax.text(.05, y, 'Paper figure style, read from scripts/common.py', **head)
    y += .3
    ax.text(.05, y, 'Region fills (maps, TikZ)', **head)
    fills = [('success', 'Success / TP'), ('fail', 'Fail / TN'), ('fp', 'FP'), ('fn', 'FN'), ('abstain', 'Abstention')]
    for i, (k, lbl) in enumerate(fills):
        x = .05 + i * 1.18
        ax.add_patch(Rectangle((x, y + .07), .32, .22, fc=FILL[k], ec='none'))
        ax.text(x + .38, y + .07, esc(lbl), **small)
        ax.text(x + .38, y + .2, esc(FILL[k]), **small)
    y += .62
    ax.text(.05, y, 'Figure 5 only: abstention kinds (FP and FN cells use the fills above)', **head)
    kinds = [('invalid', 'invalid'), ('weak', 'weak'), ('outcome_uncertain', 'outcome-uncertain')]
    for i, (k, lbl) in enumerate(kinds):
        x = .05 + i * 1.18
        ax.add_patch(Rectangle((x, y + .07), .32, .22, fc=ABSTAIN_KIND[k], ec='none'))
        ax.text(x + .38, y + .07, esc(lbl), **small)
        ax.text(x + .38, y + .2, esc(ABSTAIN_KIND[k]), **small)
    x = .05 + 3 * 1.18
    ax.add_patch(Rectangle((x, y + .07), .45, .22, fc=FILL['fail'], ec='none'))
    ax.add_line(Line2D([x + .02, x + .43], [y + .14, y + .14], color=FILL['boundary'], lw=LW['boundary'],
                       path_effects=[withStroke(linewidth=LW['boundary'] + .6, foreground='white')]))
    ax.add_line(Line2D([x + .02, x + .43], [y + .23, y + .23], color=FILL['boundary'], lw=.45, ls='--',
                       path_effects=[withStroke(linewidth=.45 + .6, foreground='white')]))
    ax.text(x + .52, y + .07, 'contours over', **small)
    ax.text(x + .52, y + .2, 'maps: white casing', **small)

    y += .62
    ax.text(.05, y, 'Lines: colour = side, style = metric kind', **head)
    lines = [('success', 'recall', r'$R_{\mathrm{s}}$'), ('success', 'error', 'FDR'),
             ('fail', 'recall', r'$R_{\mathrm{f}}$'), ('fail', 'error', 'FOR')]
    for i, (side, kind, lbl) in enumerate(lines):
        x = .05 + i * 1.18
        ax.add_line(Line2D([x, x + .45], [y + .18, y + .18], color=LINE[side], ls=LS[kind], lw=LW['curve']))
        ax.text(x + .52, y + .07, lbl + ' ' + kind, **small)
        ax.text(x + .52, y + .2, esc(LINE[side]), **small)
    x = .05 + 4 * 1.18
    ax.add_line(Line2D([x, x + .45], [y + .18, y + .18], color=LINE['ref'], ls=LS['ref'], lw=LW['ref']))
    ax.text(x + .52, y + .07, r'5\% target', **small)
    ax.text(x + .52, y + .2, esc(LINE['ref']), **small)
    y += .45
    for i, (fig_, a) in enumerate(BAND_ALPHA.items()):
        x = .05 + i * 2.36
        for j, side in enumerate(('success', 'fail')):
            ax.add_patch(Rectangle((x + j * .24, y + .05), .22, .2, fc=LINE[side], alpha=a, ec='none'))
        ax.text(x + .55, y + .05, f'min--max bands, alpha {a}', **small)
        ax.text(x + .55, y + .18, esc(fig_), **small)
    x = .05 + 2 * 2.2
    ax.plot([x + .08], [y + .15], **marker_kw('TERA', LINE['success'], 4.5), ls='none')
    faded = tuple(tint(LINE['success'], 1 - PARTIAL_TINT))
    ax.plot([x + .3], [y + .15], **marker_kw('TERA', faded, 4.5), ls='none')
    ax.text(x + .45, y + .05, '1--2 seeds: faded', **small)
    ax.text(x + .45, y + .18, f'tint {PARTIAL_TINT} to white', **small)

    y += .6
    ax.text(.05, y, 'Methods: marker shape and fill (never colour); shown on both sides', **head)
    names = list(METHOD_MARKER)
    per_row = 5
    for i, m in enumerate(names):
        r, c = divmod(i, per_row)
        x, yy = .05 + c * 1.18, y + .1 + r * .3
        ax.plot([x + .06], [yy + .08], **marker_kw(m, LINE['success'], 4.5), ls='none')
        ax.plot([x + .2], [yy + .08], **marker_kw(m, LINE['fail'], 4.5), ls='none')
        ax.text(x + .32, yy + .02, esc(m), **small)
    y += .1 + .3 * ((len(names) + per_row - 1) // per_row) + .25

    ax.text(.05, y, 'Type: usetex, Times bold, bold Computer Modern math', **head)
    samples = [('tick', r'tick 7 pt: 0.25  1.5k  $-2\pi$'), ('legend', 'legend 7 pt: Unif-TERA'),
               ('label', r'label 8 pt: $R_{\mathrm{s}}$ (\%)'), ('title', 'title 8 pt: Cartpole (4D), PPO')]
    for i, (k, s) in enumerate(samples):
        ax.text(.05 + i * 1.47, y + .1, s, fontsize=FS[k], va='top', ha='left')
    ax.text(.05, y + .32, 'Ink: black text and axes; whiskers and counts ' + esc(INK), **small)
    y += .65

    ax.text(.05, y, r'Signal map (Figure 5 $I(x)$, $D(x)$): ' + esc(SIGNAL_CMAP), **head)
    cax = fig.add_axes([.15 / W, 1 - (y + .32) / H, 2.1 / W, .14 / H])
    cax.imshow(np.linspace(0, 1, 256)[None, :], cmap=SIGNAL_CMAP, aspect='auto', extent=(0, 1, 0, 1))
    cax.set_yticks([]); cax.set_xticks([0, .5, 1])
    for s in ('top', 'right', 'left'):
        cax.spines[s].set_visible(False)
    ax.text(2.5, y + .12, r'Print widths (in), scale 1.00 in the paper:', **small)
    items = list(PRINT_WIDTH.items())
    for j in range(0, len(items), 2):
        ax.text(2.5, y + .26 + .13 * (j // 2), esc(', '.join(f'{k} {v:.2f}' for k, v in items[j:j + 2])), **small)
    y += .85

    ax.text(.05, y, 'TikZ colours in the paper clone (\\verb|\\definecolor|), matched to the scheme', **head)
    seen = {}
    for rel, defs in tikz_colours().items():
        for name, model, val in defs:
            seen.setdefault((name, val.upper()), []).append(rel.split('/')[0])
    for i, ((name, val), files) in enumerate(sorted(seen.items())):
        r, c = divmod(i, 2)
        x, yy = .05 + c * 2.95, y + .08 + r * .16
        ax.add_patch(Rectangle((x, yy + .01), .14, .1, fc='#' + val, ec='none'))
        match = scheme_name(val)
        note = match if match else ('outside the scheme' + (': unused' if name == 'cC' else ': structure ink'))
        ax.text(x + .2, yy, esc(f'{name} #{val}  {note}'), **small)
    y += .08 + .16 * ((len(seen) + 1) // 2) + .05
    assert y < H, y
    return fig, seen


def main():
    fig, seen = draw()
    OUT.mkdir(parents=True, exist_ok=True)
    pdf, png = OUT / 'style_card.pdf', OUT / 'style_card.png'
    fig.savefig(pdf, metadata={'CreationDate': None, 'Producer': None})
    fig.savefig(png, dpi=300)
    plt.close(fig)
    views, rows = pair_table()
    lines = ['Colour pairs that meet in some figure: CIELAB dE76 under each view (Machado 2009 full severity; CIE '
             'luminance grayscale).',
             'Flags: the worst colour-vision view (normal, protan, deutan, tritan) under 10 = CVD COLLAPSE, under 20 = '
             'CVD close; grayscale (black-and-white print) flagged separately, under 10 = GRAY collapse.', '',
             f"{'where':30s} {'pair':46s} " + ' '.join(f'{v[:6]:>7s}' for v in views) + '  flags']
    for fig_, a, b, d in rows:
        cvd = min(d[v] for v in views if v != 'grayscale')
        flags = ['CVD COLLAPSE' if cvd < 10 else ('CVD close' if cvd < 20 else '')]
        flags.append('GRAY collapse' if d['grayscale'] < 10 else '')
        lines.append(f'{fig_:30s} {a + " vs " + b:46s} ' + ' '.join(f'{d[v]:7.1f}' for v in views)
                     + '  ' + ', '.join(f for f in flags if f))
    lines += ['', 'TikZ colours outside the scheme:']
    lines += [f'  {n} #{v} in {", ".join(sorted(set(f)))}' for (n, v), f in sorted(seen.items()) if not scheme_name(v)]
    (OUT / 'style_card.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print(pdf, png)


if __name__ == '__main__':
    main()
