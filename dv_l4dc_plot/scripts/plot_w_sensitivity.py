"""Sensitivity of the fitted operating point to w: candidate supplement figure w_sensitivity (Sumanth, 2026-11-13).

Redraws the 11-10 diagnostic w sweep (_index/analysis_code_snapshot/wsweep, two-threshold j_ver rule, precision /
specificity) on the paper's current rule: the soft four-threshold fit of J_w on D_val (Appendix A), read from
inputs/results-discussion-assets/soft_four_wsweep.jsonl, the same file Table 2 reads at w = 0.2. No refitting here.

Runs: the five adaptive predictors of the matrix (matrix_role 'method' = TERA, 'model comparison' = TERA-MLP, TERA-BNN,
MLP-Label, BNN-Label), six settings, three seeds, w in {0.1, ..., 0.9}. Rows: Rec_s, Err_s, Rec_f, Err_f, Abst on the
test states, in percent; columns: settings in table order. Averaging as the tables (Section 5.1.4): recall and
abstention over all three seeds; an error over the seeds with a nonempty predicted class, no point when none. Line =
mean; TERA alone carries a seed min-max band. Colour names the outcome (success blue, failure orange, abstention
gray), marker shape the predictor (common.METHOD_MARKER). The dotted line marks w = 0.2, the paper's setting.

Writes plots/w_sensitivity.{pdf,png} and values/w_sensitivity_values.json; --paper copies the PDF into the paper clone.
"""
import argparse, hashlib, json, shutil
import statistics as st
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from common import (ASSETS, VALUES, PLOTS, PAPER_FIGS, FS, LINE, LW, METHOD_MARKER, REC_S, REC_F, ERR_S, ERR_F,
                    marker_kw, page_size, paper_legend_kw, style_rc)

STEM = 'w_sensitivity'
SRC = ASSETS / 'soft_four_wsweep.jsonl'
SETTINGS = [('pen-lqr', 'Pendulum\nLQR'), ('cp-lqr', 'Cartpole\nLQR'), ('cp-ppo', 'Cartpole\nRL'),
            ('q2d-ppo', 'Planar quadrotor\nRL'), ('q3d-lqr', 'Spatial quadrotor\nLQR'), ('q3d-ppo', 'Spatial quadrotor\nRL')]
PREDICTORS = [('method', 'FM', 'TERA'), ('model comparison', 'BNN final-state', 'TERA-BNN'),
              ('model comparison', 'MLP final-state', 'TERA-MLP'), ('model comparison', 'BNN classifier', 'BNN-Label'),
              ('model comparison', 'MLP classifier', 'MLP-Label')]
ROWS = [('rec', REC_S + r' (\%)', LINE['success']), ('fdr', ERR_S + r' (\%)', LINE['success']),
        ('fail_rec', REC_F + r' (\%)', LINE['fail']), ('for', ERR_F + r' (\%)', LINE['fail']),
        ('abstain', r'$\widehat{\mathrm{Abst}}$ (\%)', LINE['abstain'])]
WS = [round(.1 * i, 1) for i in range(1, 10)]
W_PAPER = .2
WIDTH, HEIGHT = 6.0, 6.4
MS, MEW = 3.0, .5


def load():
    rows = [json.loads(l) for l in SRC.read_text().splitlines()]
    out = {}
    for r in rows:
        k = (r['setting'], r['matrix_role'], r['model'], r['w'])
        out.setdefault(k, []).append(r)
    return out


def agg(rs, key):
    """(mean, min, max, n) in percent; recall and abstention over all seeds, errors over defined seeds."""
    v = [100 * r['test'][key] for r in rs if r['test'][key] is not None]
    if not v:
        return None
    return st.mean(v), min(v), max(v), len(v)


def draw(data):
    fig, axes = plt.subplots(len(ROWS), len(SETTINGS), figsize=(WIDTH, HEIGHT), squeeze=False)
    fig.subplots_adjust(left=.072, right=.972, bottom=.047, top=.925, wspace=.42, hspace=.16)
    record = []
    for c, (s, title) in enumerate(SETTINGS):
        for r, (key, ylab, color) in enumerate(ROWS):
            ax = axes[r, c]
            ax.axvline(W_PAPER, color=LINE['ref'], lw=LW['ref'], ls=(0, (1, 1.5)), zorder=0)
            for role, model, name in PREDICTORS:
                pts = []
                for w in WS:
                    rs = data.get((s, role, model, w), [])
                    assert len(rs) == len({x['seed'] for x in rs}) == 3, (s, role, model, w, len(rs))
                    a = agg(rs, key)
                    record.append(dict(setting=s, predictor=name, w=w, metric=key,
                                       **(dict(zip(['mean', 'min', 'max', 'n'], a)) if a else dict(mean=None, n=0))))
                    if a:
                        pts.append((w, *a))
                if not pts:
                    continue
                xs, ms = [p[0] for p in pts], [p[1] for p in pts]
                if name == 'TERA':
                    ax.fill_between(xs, [p[2] for p in pts], [p[3] for p in pts], color=color, alpha=.18, lw=0, zorder=1)
                kw = marker_kw(name, color, size=MS, edge=MEW)
                ax.plot(xs, ms, color=color, lw=1.0 if name == 'TERA' else .6, alpha=1 if name == 'TERA' else .75,
                        ls='-' if not name.endswith('Label') else (0, (2.5, 1.2)), zorder=3 if name == 'TERA' else 2, **kw)
            ax.set_xlim(.05, .95)
            ax.set_xticks([.2, .5, .8])
            ax.yaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=2))
            if r < len(ROWS) - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel(r'$w$', labelpad=1)
            if c == 0:
                ax.set_ylabel(ylab, labelpad=1.5)
            if r == 0:
                ax.set_title(title, pad=3)
    handles = [Line2D([], [], color='black', lw=1.0 if n == 'TERA' else .6,
                      ls='-' if not n.endswith('Label') else (0, (2.5, 1.2)),
                      label=n, **(marker_kw(n, 'black', size=MS, edge=MEW))) for _, _, n in PREDICTORS]
    handles += [Patch(facecolor='black', alpha=.18, label='TERA seed min--max'),
                Line2D([], [], color=LINE['ref'], lw=LW['ref'], ls=(0, (1, 1.5)), label=r'$w=0.2$ (paper)')]
    fig.legend(handles=handles, ncol=7, bbox_to_anchor=(.5, .997), bbox_transform=fig.transFigure,
               **(paper_legend_kw(FS['legend']) | dict(columnspacing=1.0, handlelength=1.8)))
    return fig, record


def main(paper):
    data = load()
    with plt.rc_context(style_rc()):
        fig, record = draw(data)
        PLOTS.mkdir(exist_ok=True)
        pdf, png = PLOTS / f'{STEM}.pdf', PLOTS / f'{STEM}.png'
        fig.savefig(pdf, metadata={'CreationDate': None})
        fig.savefig(png, dpi=300)
        plt.close(fig)
    (VALUES / f'{STEM}_values.json').write_text(json.dumps(dict(
        generated_by='scripts/plot_w_sensitivity.py', source=str(SRC),
        source_sha256=hashlib.sha256(SRC.read_bytes()).hexdigest(), cells=record), indent=2) + '\n')
    w, h = page_size(pdf)
    print(f'{STEM}: {w} x {h} in, {len(record)} cells')
    if paper:
        shutil.copyfile(pdf, PAPER_FIGS / f'{STEM}.pdf')
        print(f'{STEM} -> paper figures/results/{STEM}.pdf')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument('--paper', action='store_true', help='copy the PDF into the paper clone figures/results/')
    main(p.parse_args().paper)
