"""FM acquisition-score ablation at rho=0.75 as bars, six settings, in two versions.

  certified  THE PAPER FIGURE (Fig 5, Dhruv's pick 2026-11-12), acquisition_strategy_certified:
             two-sided A-1d at alpha = delta = 0.05 (c6c_* via certified.py, as Table 2). U_H must equal
             Table 2's FM adaptive row.
  empirical  preview only (acquisition_strategy_bars): soft four-threshold rule fit on validation at
             w=0.2, final-budget test (soft_four_wsweep.jsonl). It was the paper's acquisition_strategy.pdf
             until the certified version replaced it, and no longer writes to the paper.

Each setting has a recall row (R_s, R_f bars, 0-100) over an error row (FDR, FOR bars) with its own
axis and a thin solid gray 5% line that is not in the legend. x axis: U_H, q_A, U_tau. Bar = three-seed
mean, whisker = min to max over seeds (common.spread; never SD). FDR/FOR use only seeds with a nonempty
predicted class; a bar from fewer than three seeds carries an "n=k" label, and a cell with none shows
"n=0" and no bar. The legend pairs each color with its two metrics in the method section's terms.
Fails if a whisker leaves its panel.

With --paper only the certified version is copied into the paper clone's figures/results/.
Since 2026-11-13 the paper figure is the certified version in the common.py paper style at print width,
plots/acquisition_strategy_certified.pdf (split.py restyle block); it fails unless its numbers equal the stored values
JSON. A plain run first rebuilds values/ and the old-style draws (the certified one now in plots/legacy/), then the
paper figure. --restyle redraws only the paper figure; --paper copies it into the paper clone.
"""
import hashlib, json, math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.patheffects import withStroke
from common import (VALUES, SETTINGS, TITLES, BLUE, ORANGE, RECALL, ERROR, RC, SOFT_FOUR,
                    load_final, cell, save, PRINT_WIDTH, FS, LINE, LS, LW, style_rc, INK, install_paper)
import certified
from split import (assert_bands_inside, REF_SOLID, SIDE, restyle_args, tex, top_legend, tight_split_layout,
                   check_restyled, save_restyled)

COMBOS = [('U_H', .75), ('q_A', .75), ('U_tau', .75)]
XTICKS = [r'$U_H$', r'$q_A$', r'$U_\tau$']
# Legend in the method section's terms (04_proposed_method.tex), one entry per bar color.
LEGEND_PAIRS = [(BLUE, r'$R_{\mathrm{s}}$ (Success recall) / FDR (error of Success declarations)'),
                (ORANGE, r'$R_{\mathrm{f}}$ (Fail recall) / FOR (error of Fail declarations)')]


def bar_figure(values, stem, paper_stem, paper):
    lookup = {(v['setting'], v['arm'], v['metric']): v for v in values}
    err_top = max(v['max'] for v in values if v['metric'] in ('fdr', 'for') and v['n'])
    err_max = math.ceil((err_top + .1) / 2) * 2
    assert err_max > 5  # the 5% line must stay visible
    xx = np.arange(len(COMBOS)); width = .36
    with plt.rc_context(RC):
        fig = plt.figure(figsize=(6.5, 4.4))
        outer = fig.add_gridspec(2, 3, left=.08, right=.99, top=.86, bottom=.13, wspace=.24, hspace=.42)
        for si, (s, title) in enumerate(zip(SETTINGS, TITLES)):
            inner = outer[si // 3, si % 3].subgridspec(2, 1, height_ratios=[1.45, 1], hspace=.12)
            rec = fig.add_subplot(inner[0]); err = fig.add_subplot(inner[1], sharex=rec)
            rec.set_title(title, pad=5)
            whiskers = []
            for ax, pairs in [(rec, RECALL), (err, ERROR)]:
                ax.grid(axis='y', alpha=.2, lw=.45); ax.set_axisbelow(True)
                ax.axhline(0, color='#555555', lw=.5, zorder=2)
                for k, (metric, color) in enumerate(pairs):
                    cc = [lookup[(s, a, metric)] for a, _ in COMBOS]
                    nan = lambda key: np.array([np.nan if c[key] is None else c[key] for c in cc], dtype=float)
                    mean, lo, hi = nan('mean'), nan('min'), nan('max')
                    xk = xx + (k - .5) * width
                    ok = np.isfinite(mean)
                    ax.bar(xk[ok], mean[ok], width, color=color, edgecolor='white', linewidth=.25, zorder=3,
                           yerr=[(mean - lo)[ok], (hi - mean)[ok]],
                           error_kw={'elinewidth': .65, 'capsize': 1.4, 'capthick': .65, 'ecolor': '#333333'})
                    whiskers.append((ax, lo, hi))
                    for x, c, top in zip(xk, cc, np.where(ok, hi, 0)):
                        if c['n'] < 3:  # fewer than three seeds with a nonempty predicted class
                            ax.annotate(f"n={c['n']}", (x, top), xytext=(0, 1.5), textcoords='offset points',
                                        ha='center', va='bottom', fontsize=5.5, color='#333333', zorder=4)
            rec.set_ylim(0, 104); rec.set_yticks([0, 50, 100]); rec.tick_params(labelbottom=False)
            err.set_ylim(0, err_max); err.set_yticks([0, 5, 10] if err_max >= 10 else [0, 5])
            for ax in (rec, err):
                assert_bands_inside(ax, [(lo, hi) for a, lo, hi in whiskers if a is ax])
            err.axhline(5, zorder=2.5, **REF_SOLID)
            err.set_xticks(xx, XTICKS)
            if si % 3 == 0:
                rec.set_ylabel('Recall (%) ↑'); err.set_ylabel('Error (%) ↓')
        fig.supxlabel(r'Acquisition score ($\rho=0.75$)', fontsize=7.5, y=.025)
        fig.legend(handles=[Patch(color=c, label=t) for c, t in LEGEND_PAIRS], loc='upper center', ncol=1,
                   frameon=False, bbox_to_anchor=(.54, 1.01), handlelength=1.4, labelspacing=.35)
        size = save(fig, stem, paper_stem=paper_stem, paper=paper)
    return size, err_max


# --restyle (dm1487, 2026-11-13): Figure S1 (certified only) in the common.py paper style at its 6.00 in print width,
# into plots/ (the paper figure). Panel shapes are the paper figure's: recall 0.741 in and error 0.511 in tall, 1.70 in wide.
RESTYLE_ASPECT, RESTYLE_ERR_RATIO = .7411 / 1.6997, 1 / 1.45
WHISKER = INK           # common.py; bar_figure's whisker ink, kept; a white halo makes it show on the dark purple bars


def bar_figure_restyled(values):
    """bar_figure's layout and data in the paper style. Bar colour = the metric's side (R_s and FDR gold, R_f and
    FOR purple). The strategies are told apart by position and their x tick labels under every error panel, as
    before; colour never names a strategy. Whiskers min-max as before, with a white halo. Fails if a whisker leaves
    its panel, the 5% line leaves an error axis, or check_restyled fails."""
    stem = 'acquisition_strategy_certified'
    W = PRINT_WIDTH[stem]
    lookup = {(v['setting'], v['arm'], v['metric']): v for v in values}
    err_top = max(v['max'] for v in values if v['metric'] in ('fdr', 'for') and v['n'])
    err_max = math.ceil((err_top + .1) / 2) * 2
    assert err_max > 5  # the 5% line must stay visible
    xx = np.arange(len(COMBOS)); width = .36
    halo = [withStroke(linewidth=.65 + 1.1, foreground='white')]
    with plt.rc_context(style_rc()):
        fig = plt.figure(figsize=(W, 4))
        pairs, whiskers, counts = [], {}, []
        for si, (s, title) in enumerate(zip(SETTINGS, TITLES)):
            rec = fig.add_axes([.1, .5, .2, .2]); err = fig.add_axes([.1, .2, .2, .2], sharex=rec)
            pairs.append((rec, err))
            rec.set_title(tex(title), pad=3)
            for ax, pairs_ in [(rec, RECALL), (err, ERROR)]:
                ax.grid(axis='y', alpha=.2, lw=.45); ax.set_axisbelow(True)
                for k, (metric, _) in enumerate(pairs_):
                    cc = [lookup[(s, a, metric)] for a, _ in COMBOS]
                    nan = lambda key: np.array([np.nan if c[key] is None else c[key] for c in cc], dtype=float)
                    mean, lo, hi = nan('mean'), nan('min'), nan('max')
                    xk = xx + (k - .5) * width
                    ok = np.isfinite(mean)
                    bars = ax.bar(xk[ok], mean[ok], width, color=SIDE[metric], edgecolor='white', linewidth=.25,
                                  zorder=3, yerr=[(mean - lo)[ok], (hi - mean)[ok]],
                                  error_kw={'elinewidth': .65, 'capsize': 1.4, 'capthick': .65, 'ecolor': WHISKER})
                    for a in [*bars.errorbar.lines[1], *bars.errorbar.lines[2]]:
                        a.set_path_effects(halo)
                    whiskers.setdefault(ax, []).append((lo, hi))
                    for x, c, top in zip(xk, cc, np.where(ok, hi, 0)):
                        if c['n'] < 3:  # fewer than three seeds with a nonempty predicted class
                            counts.append(ax.annotate(f"n={c['n']}", (x, top), xytext=(0, 1.5), textcoords='offset points',
                                                      ha='center', va='bottom', fontsize=FS['annotation'],
                                                      color=WHISKER, zorder=4))
            rec.tick_params(labelbottom=False)
            err.axhline(5, color=LINE['ref'], lw=LW['ref'], ls=LS['ref'], zorder=2.5)
            err.set_xticks(xx, XTICKS)
            if si % 3:
                rec.tick_params(labelleft=False); err.tick_params(labelleft=False)
            else:
                rec.set_ylabel(tex('Recall (%) ↑')); err.set_ylabel(tex('Error (%) ↓'))
        fig.align_ylabels([a for p in pairs[::3] for a in p])
        legend, ncol = top_legend(fig, [Patch(color=SIDE[m]) for m in ('rec', 'fail_rec')],
                                  [tex(t) for _, t in LEGEND_PAIRS], W, ncols=(2, 1))
        xl = fig.text(.5, 0, tex(r'Acquisition score ($\rho=0.75$)'), ha='center', va='bottom', fontsize=FS['label'])

        def set_ylims(rec_h, err_h):
            for rec, err in pairs:
                rec.set_ylim(0, 104); rec.set_yticks([0, 50, 100])
                err.set_ylim(0, err_max); err.set_yticks([0, 5, 10] if err_max >= 10 else [0, 5])

        geo = tight_split_layout(fig, pairs, legend, xl, width=W, rec_aspect=RESTYLE_ASPECT,
                                 err_ratio=RESTYLE_ERR_RATIO, set_ylims=set_ylims)
        fig.canvas.draw()
        for rec, err in pairs:
            assert err.get_ylim()[0] < 5 < err.get_ylim()[1]
            for ax in (rec, err):
                assert_bands_inside(ax, whiskers[ax])
        edges = check_restyled(fig, [a for p in pairs for a in p], legend, inside=counts)
        size = save_restyled(fig, stem)
    print(f"  panels {geo['panel_w']:.3f} in wide, recall {geo['rec_h']:.3f} in, error {geo['err_h']:.3f} in; "
          f"legend {ncol} columns; layout settled in {geo['iterations']} passes")
    print(f"  outer whitespace (in): {', '.join(f'{k} {v:.3f}' for k, v in edges.items())}")
    return dict(geometry=geo, edges=edges, size=size, err_max=err_max)


def restyled():
    """--restyle: the certified figure only. Fails unless its cells, Table 2 cross-check and error axis equal the
    stored values/acquisition_strategy_certified_values.json. Writes nothing to values/ or the paper."""
    final = load_final()
    rows = certified.load_rows()
    cert = [certified.cell(final, rows, s, a, rho, m) for s in SETTINGS for a, rho in COMBOS for m, _ in RECALL + ERROR]
    cert_lookup = {(v['setting'], v['arm'], v['metric']): v for v in cert}
    checked = certified.check_table2(lambda s, m: cert_lookup[(s, 'U_H', m)]['mean'])
    print(f'Table 2 cross-check (certified U_H): {len(checked)}/24 cells match')
    stored = json.loads((VALUES / 'acquisition_strategy_certified_values.json').read_text())
    assert json.loads(json.dumps(cert)) == stored['values'], 'restyle cells differ from values/'
    assert json.loads(json.dumps(checked)) == stored['table2_check'] and stored['source_sha256'] == certified.SRC_SHA256
    out = bar_figure_restyled(cert)
    assert out['err_max'] == stored['error_axis_max_percent']
    print(f"  {len(cert)} cells equal values/acquisition_strategy_certified_values.json")
    return out


def main(paper):
    final = load_final()
    emp = [cell(final, s, a, rho, m) for s in SETTINGS for a, rho in COMBOS for m, _ in RECALL + ERROR]
    rows = certified.load_rows()
    cert = [certified.cell(final, rows, s, a, rho, m) for s in SETTINGS for a, rho in COMBOS for m, _ in RECALL + ERROR]
    cert_lookup = {(v['setting'], v['arm'], v['metric']): v for v in cert}
    checked = certified.check_table2(lambda s, m: cert_lookup[(s, 'U_H', m)]['mean'])
    print(f'Table 2 cross-check (certified U_H): {len(checked)}/24 cells match')
    VALUES.mkdir(exist_ok=True)
    for values, stem, paper_stem, src, sha, protocol, extra in [
            (emp, 'acquisition_strategy_bars', None, SOFT_FOUR,
             hashlib.sha256(SOFT_FOUR.read_bytes()).hexdigest(),
             'FM, rho=0.75, soft four-threshold rule fit on validation at w=0.2, final-budget test', {}),
            (cert, 'acquisition_strategy_certified', 'acquisition_strategy_certified', certified.SRC,
             certified.SRC_SHA256, 'FM, rho=0.75, two-sided A-1d at alpha=delta=0.05 (c6c_*), final budget',
             dict(table2_check=checked))]:
        size, err_max = bar_figure(values, stem, paper_stem, paper)
        partial = [v for v in values if v['n'] < 3]
        (VALUES / f'{stem}_values.json').write_text(json.dumps(dict(
            source=str(src), source_sha256=sha,
            protocol=protocol + '; mean with min-max over 3 seeds; FDR/FOR over seeds with nonempty S/F only',
            error_axis_max_percent=err_max, page_size=size, values=values, **extra), indent=2) + '\n')
        print(f'  {stem}: error axis 0-{err_max}%, {len(values)} cells, cells under 3 seeds: '
              + (', '.join(f"{v['setting']} {v['arm']} {v['metric']} n={v['n']}" for v in partial) or 'none'))


if __name__ == '__main__':
    args = restyle_args(__doc__)
    if not args.restyle:
        main(False)                         # values/ and the old-style draws; the certified one goes to plots/legacy/
    restyled()
    if args.paper:
        install_paper('acquisition_strategy_certified')
