"""Two uncertainty signals in pendulum state space: paper figure pendulum_uncertainty_signals.

Two paths:
  (default)   THE PAPER FIGURE since 2026-11-13 (paper commit 34eb36f): the paper style of common.py at print width
              (4.20 in), drawn on the FULL 158 x 315 grid (data/pendulum_full_grid/), abstention, FP and FN cells in C
              drawn as 1 pt squares (--rare-pt, default 1). Writes plots/pendulum_uncertainty_signals.{pdf,png,txt}
              and nothing else; other --rare-pt values (0 = true cell size) go to plots/variants/. --paper copies the
              paper version into the paper clone. See restyle_figure() for the layout and its checks.
  --baseline  the 3-column design of 2026-11-12 (md5 51f991cf), unchanged: test states as scatter squares, DejaVu
              fonts; the figure goes to plots/legacy/, and it rewrites values/pendulum_uncertainty_signals_*, which
              the work-in-progress plot_uncertainty_signals.py also writes (today's files are the WIP's).
  --restyle   accepted for old commands; the default already is the paper figure.

TERA-FM on Pendulum LQR, seed 99512, round 10 (the run behind figures/abstention_triangle). Rows: adaptive (U_H,
rho=0.75) and uniform acquisition, each with its OWN saved A-1d thresholds. Columns:
  A  invalid-terminal mass I(x) = p_inv(x)
  B  weighted disagreement D(x) = (1 - p_inv) H2(p_s / (p_s + p_f)) in bits, D = 0 with no valid endpoint
  C  certified decisions (the default path's title still reads "calibrated"; the restyle uses the paper's titles):
     Success, Fail, and the invalid / weak / outcome-uncertain abstentions (strict
     definitions; overlaps or uncategorized abstentions get their own color if a run has any), with incorrect
     committed predictions marked
A and B share one [0, 1] color scale. Every panel overlays the true success/failure boundary (from test labels) and
the contour between sparse and densely sampled locations (nearest training state farther than 0.02 in
(theta/2pi wrapped, omega/4pi)). All data and checks come from uncertainty_signals_data.py.

Also writes values/pendulum_uncertainty_signals_stats.csv (every arm, seed, round 2 and 10, all categories) and
values/pendulum_uncertainty_signals_values.json (stats, the comparison with sections/results_uncertainty.tex,
threshold sums, provenance). --paper copies the PDF into the paper clone as
figures/results/pendulum_uncertainty_signals.pdf. The figure and statistics describe aggregate regions; they do not
establish correctness at any individual state.
"""
import csv, json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from common import VALUES, RC, BLUE, ORANGE, parse_args, save
from uncertainty_signals_data import build, SEEDS, SPARSE as SPARSE_D

SEED, ROUND = 99512, 10
CATS = [('success', 'Success', BLUE), ('fail', 'Fail', ORANGE), ('inv', 'invalid abstention', '#4d4d4d'),
        ('weak', 'weak abstention', '#bdbdbd'), ('ou', 'outcome-uncertain abstention', '#CC79A7'),
        ('overlap', 'invalid and outcome-uncertain', '#000000'), ('other', 'other abstention', '#f2e6c9')]
# Manuscript claims (sections/results_uncertainty.tex, paper commit 5daf370), as fractions or p_inv means.
CLAIMS = [
    ('r10 adaptive: invalid abstentions in sparse data', .28, 'adaptive', 10, ('invalid', 'frac_in_sparse')),
    ('r10 uniform: invalid abstentions in sparse data', .72, 'uniform', 10, ('invalid', 'frac_in_sparse')),
    ('r10 adaptive: all test states in sparse data', .09, 'adaptive', 10, ('baseline_sparse',)),
    ('r10 uniform: all test states in sparse data', .09, 'uniform', 10, ('baseline_sparse',)),
    ('r10 uniform: near-boundary sparse states that are invalid abstentions', .40, 'uniform', 10, ('invalid_among_near_sparse',)),
    ('r10 uniform: near-boundary dense states that are invalid abstentions', .003, 'uniform', 10, ('invalid_among_near_dense',)),
    ('r10 adaptive: weak abstentions near the boundary', .47, 'adaptive', 10, ('weak', 'frac_near')),
    ('r10 uniform: weak abstentions near the boundary', .48, 'uniform', 10, ('weak', 'frac_near')),
    ('r10: all test states near the boundary', .03, 'adaptive', 10, ('baseline_near',)),
    ('r10 adaptive: outcome-uncertain abstentions near the boundary', .92, 'adaptive', 10, ('outcome_uncertain', 'frac_near')),
    ('r10 uniform: outcome-uncertain abstentions near the boundary', .70, 'uniform', 10, ('outcome_uncertain', 'frac_near')),
    ('r10 adaptive: mean p_inv of weak abstentions', .57, 'adaptive', 10, ('weak', 'mean_pinv')),
    ('r10 uniform: mean p_inv of weak abstentions', .82, 'uniform', 10, ('weak', 'mean_pinv')),
    ('r10 adaptive: mean p_inv of outcome-uncertain abstentions', .24, 'adaptive', 10, ('outcome_uncertain', 'mean_pinv')),
    ('r10 uniform: mean p_inv of outcome-uncertain abstentions', .51, 'uniform', 10, ('outcome_uncertain', 'mean_pinv')),
    ('r2 adaptive: invalid abstentions, share of test states', .050, 'adaptive', 2, ('invalid', 'frac_of_test')),
    ('r10 adaptive: invalid abstentions, share of test states', .004, 'adaptive', 10, ('invalid', 'frac_of_test')),
    ('r2 uniform: invalid abstentions, share of test states', .090, 'uniform', 2, ('invalid', 'frac_of_test')),
    ('r10 uniform: invalid abstentions, share of test states', .011, 'uniform', 10, ('invalid', 'frac_of_test')),
]
PRECISION = {c[0]: .001 for c in CLAIMS if c[0].startswith(('r2 ', 'r10 adaptive: invalid abstentions, share',
                                                             'r10 uniform: invalid abstentions, share'))}
PRECISION['r10 uniform: near-boundary dense states that are invalid abstentions'] = .001
VARIANT = {('weak', 'frac_near'): ('weak_residual', 'frac_near'), ('weak', 'mean_pinv'): ('weak_residual', 'mean_pinv'),
           ('invalid', 'frac_of_test'): ('invalid_excl_overlap', 'frac_of_test')}


def get(st, path):
    v = st
    for k in path:
        v = v[k]
    return v


def compare(stats):
    out = []
    for label, claim, arm, rnd, path in CLAIMS:
        rows = [s for s in stats if s['arm'] == arm and s['round'] == rnd and s['cert_pass']]
        vals = [get(s, path) for s in rows if get(s, path) is not None]
        mine = float(np.mean(vals)) if vals else None
        var = None
        if path in VARIANT:
            vv = [get(s, VARIANT[path]) for s in rows if get(s, VARIANT[path]) is not None]
            var = float(np.mean(vv)) if vv else None
        # Half a unit of the precision the manuscript states: "28%" -> 0.5 points, "5.0%" -> 0.05 points, "0.57" -> 0.005.
        tol = PRECISION.get(label, .01) / 2
        out.append(dict(claim=label, manuscript=claim, recomputed=mine, seeds=len(vals),
                        matches=mine is not None and abs(mine - claim) <= tol,
                        manuscript_variant=None if var is None else dict(definition=VARIANT[path][0], value=var,
                                                                          matches=abs(var - claim) <= tol)))
    return out


MS2 = 2.8          # scatter area (pt^2) per test state: the 42%-filled 158 x 315 lattice reads as continuous


def draw(arrays, stats, paper):
    """Each test state is drawn where it was evaluated (no interpolation into empty lattice cells); contours come
    from a triangulation of the test states."""
    import matplotlib.tri as mtri
    rows = [('adaptive', r'adaptive ($\rho=0.75$)'), ('uniform', 'uniform')]
    present = set()
    with plt.rc_context(RC | {'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5}):
        fig, axs = plt.subplots(2, 3, figsize=(6.5, 4.75), sharex=True, sharey=True)
        fig.subplots_adjust(left=.09, right=.985, top=.84, bottom=.15, wspace=.08, hspace=.16)
        for ri, (arm, name) in enumerate(rows):
            a = arrays[arm, SEED, ROUND]
            X = a['X']
            tri = mtri.Triangulation(X[:, 0], X[:, 1])
            th_ = next(s for s in stats if (s['arm'], s['seed'], s['round']) == (arm, SEED, ROUND))['thresholds']
            for ci, key in enumerate(['I', 'D']):
                o = np.argsort(a[key], kind='stable')          # high values drawn last, so they stay visible
                m = axs[ri, ci].scatter(X[o, 0], X[o, 1], c=a[key][o], s=MS2, marker='s', lw=0, cmap='Purples',
                                        vmin=0, vmax=1, rasterized=True)
            ov = a['inv'] & a['ou']
            masks = {'success': a['succ'], 'fail': a['fail'], 'inv': a['inv'] & ~ov, 'weak': a['weak'],
                     'ou': a['ou'] & ~ov, 'overlap': ov, 'other': a['other']}
            covered = np.zeros(len(a['y']), bool)
            for key, _, color in CATS:                         # decisions first, abstention kinds on top
                mk = masks[key]
                covered |= mk
                if mk.any():
                    present.add(key)
                    big = key not in ('success', 'fail')
                    axs[ri, 2].scatter(X[mk, 0], X[mk, 1], s=MS2 * (2.2 if big else 1), marker='s', lw=0, color=color,
                                       rasterized=True, zorder=3 if big else 2)
            assert covered.all()
            axs[ri, 2].plot(X[a['wrong'], 0], X[a['wrong'], 1], ls='none', marker='x', ms=1.6, mew=.45,
                            color='#d62728', zorder=5)
            for ax in axs[ri]:
                ax.tricontour(tri, a['y'].astype(float), levels=[.5], colors='k', linewidths=.6, zorder=6)
                ax.tricontour(tri, a['d_train'], levels=[SPARSE_D], colors='#e0a800', linewidths=.7,
                              linestyles='--', zorder=6)
            axs[ri, 0].set_ylabel(name + '\n' + r'$\dot{\theta}$', labelpad=1)
            axs[ri, 2].text(1.0, 1.02, rf"$t_s$={th_['t_s']}, $t_f$={th_['t_f']}, $u_s$={th_['u_s']}, $u_f$={th_['u_f']}",
                            transform=axs[ri, 2].transAxes, ha='right', va='bottom', fontsize=5.8, color='#444444')
        for ci, t in enumerate([r'A. invalid-terminal mass $I(x)=\hat{p}_{\mathrm{inv}}(x)$',
                                r'B. weighted disagreement $D(x)$', 'C. calibrated decisions']):
            axs[0, ci].set_title(t, pad=12 if ci == 2 else 5, fontsize=7.5)
        for ax in axs.flat:
            ax.set_xlim(-np.pi, np.pi); ax.set_ylim(-2 * np.pi, 2 * np.pi)
            ax.set_xticks([-np.pi, 0, np.pi], [r'$-\pi$', '0', r'$\pi$'])
            ax.set_yticks([-2 * np.pi, 0, 2 * np.pi], [r'$-2\pi$', '0', r'$2\pi$'])
        for ax in axs[1]:
            ax.set_xlabel(r'$\theta$', labelpad=1)
        cax = fig.add_axes([.12, .045, .5, .016])
        cb = fig.colorbar(m, cax=cax, orientation='horizontal', ticks=[0, .25, .5, .75, 1])
        cb.set_label(r'$I(x)$ and $D(x)$ (common scale)', fontsize=6.5, labelpad=1); cb.ax.tick_params(labelsize=6)
        handles = [Patch(color=c, label=l) for k, l, c in CATS if k in present]
        handles += [Line2D([], [], ls='none', marker='x', ms=3, mew=.6, color='#d62728', label='incorrect committed'),
                    Line2D([], [], color='k', lw=.6, label='true success/failure boundary'),
                    Line2D([], [], color='#e0a800', lw=.7, ls='--', label='sparse/dense boundary (0.02)')]
        fig.legend(handles=handles, loc='upper center', ncol=4, frameon=False, bbox_to_anchor=(.53, 1.0),
                   fontsize=6.2, columnspacing=1.0, handlelength=1.3, handletextpad=.4, labelspacing=.25)
        size = save(fig, 'pendulum_uncertainty_signals', paper_stem='pendulum_uncertainty_signals', paper=paper)
    return size, sorted(present)


def flatten(s):
    out = {k: v for k, v in s.items() if not isinstance(v, dict)}
    for k, v in s.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                out[f'{k}.{kk}'] = vv
    return out


def main(paper):
    data = build()
    stats, arrays = data['stats'], data['arrays']
    size, present = draw(arrays, stats, paper)
    flat = [flatten(s) for s in stats]
    cols = sorted({k for f in flat for k in f}, key=lambda k: (k.split('.')[0] not in ('arm', 'seed', 'round', 'run'), k))
    with open(VALUES / 'pendulum_uncertainty_signals_stats.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in sorted(flat, key=lambda r: (r['arm'], r['round'], SEEDS.index(r['seed']))):
            w.writerow({k: ('' if r.get(k) is None else r.get(k)) for k in cols})
    comp = compare(stats)
    (VALUES / 'pendulum_uncertainty_signals_values.json').write_text(json.dumps(dict(
        figure=dict(seed=SEED, round=ROUND, page_size=size, categories_drawn=present),
        manuscript_comparison=comp, threshold_sums=data['threshold_sums'],
        max_t_s_plus_u_f=max(x['t_s_plus_u_f'] for x in data['threshold_sums']),
        pool_sha256=data['pool_sha256'], provenance=data['provenance'], stats=stats), indent=1, default=str) + '\n')
    for c in comp:
        v = c['manuscript_variant']
        print(f"{'OK ' if c['matches'] else 'DIFF'} {c['claim']}: manuscript {c['manuscript']}, recomputed "
              f"{'n/a' if c['recomputed'] is None else round(c['recomputed'], 4)} ({c['seeds']} seeds)"
              + (f" | {v['definition']} {round(v['value'], 4)} {'matches' if v['matches'] else 'differs'}" if v else ''))


# ==== --restyle (dm1487, 2026-11-13): the paper style of common.py, at print width, on the full grid ================
# Content is parked: same panels, signals and annotations as the paper copy. Changed: fonts, sizes, legend, colours,
# and (dm1487's amendment) every map is drawn on the full 158 x 315 pendulum grid, one image cell per start state.
# Titles are the paper copy's ("A. invalid mass", "C. certified decisions").
import argparse, re
import uncertainty_signals_data as usd
from matplotlib.patheffects import withStroke
from common import (HERE, PLOTS, FILL, LW as STYLE_LW, FS, PRINT_WIDTH, SIGNAL_CMAP, style_rc, paper_legend_kw,
                    ABSTAIN_KIND, install_paper)

R_STEM = 'pendulum_uncertainty_signals'
GRID_DIR = HERE / 'data' / 'pendulum_full_grid'
GRID_FILE = {'adaptive': 'x231_penlqr_fm_uh_a6d075_r1_s99512_epoch010_grid.npz',
             'uniform': 'm65_penlqr_fm_unif_s99512_epoch010_grid.npz'}
GRID_SHAPE = (158, 315)                        # theta x omega, rows theta-major
EXPECT_TEST = {'adaptive': (7784, 12762, 156), 'uniform': (7896, 12557, 249)}     # Success, Fail, Abstain (stored)
# Rows carry the paper's method names; thresholds print in the method's notation (Sumanth, 2026-11-13):
# t_s, t_f are the success pair tau^s_s, tau^s_f and u_s, u_f the failure pair tau^f_s, tau^f_f of Eq. (3).
R_ROWS = [('adaptive', 'TERA'), ('uniform', 'Unif-TERA')]
R_TITLES = [r'A. invalid mass $I(x)=\hat{p}_{\mathrm{inv}}(x)$', r'B. weighted disagreement $D(x)$',
            'C. certified decisions']
R_TITLES_WRAPPED = [r'A. invalid mass' + '\n' + r'$I(x)=\hat{p}_{\mathrm{inv}}(x)$',
                    'B. weighted disagreement\n$D(x)$', 'C. certified decisions']
# Abstention kinds as a gray family (dm1487): weak is the common abstain gray. Overlap / other have no approved colour,
# so the restyle refuses to draw if a row has any (neither row does on the full grid).
ABSTAIN_GRAY = dict(inv=ABSTAIN_KIND['invalid'], weak=ABSTAIN_KIND['weak'], ou=ABSTAIN_KIND['outcome_uncertain'])
# Wrong declarations as in Figure 1 (paper_writer, 2026-11-13): a Success on a true failure is an FP cell, a Fail on
# a true success an FN cell, in Figure 1's fills; the red crosses are gone. Timeouts (label -1) have no truth and keep
# their decision's colour.
R_CATS = [('success', 'TP', FILL['success']), ('fail', 'TN', FILL['fail']),      # Figure 1's legend names
          ('fp', 'FP', FILL['fp']), ('fn', 'FN', FILL['fn']),
          ('inv', 'invalid abstention', ABSTAIN_GRAY['inv']), ('weak', 'weak abstention', ABSTAIN_GRAY['weak']),
          ('ou', 'outcome-uncertain abstention', ABSTAIN_GRAY['ou'])]
# Both contours wear a white casing (CASING_PT each side, dashed with the line), so they read on the Fail purple and
# on the dark parts of the signal maps; on white and yellow the casing barely shows.
CASING_PT = .3
# --rare-pt X (preview only): draw every abstention, FP and FN cell in C as an X pt square over the true boundary, so
# the classes that hug the boundary show at print (a cell is 0.51 x 0.21 pt). 0 = true size, the default output.
PAPER_RARE_PT = 1.0                                             # the paper's version (34eb36f)
RARE_PT = PAPER_RARE_PT
RARE_KEYS = ('inv', 'weak', 'ou', 'fp', 'fn')                   # draw order: FP and FN on top
SPARSE_LW = .45          # pt; '--' dashes scale with it to 1.67 / .72 pt, the dash the paper copy prints at 0.644 scale
# The paper copy's panel (6.5 x 4.75 in canvas, subplots_adjust below): its aspect is kept.
PW0 = 6.5 * (.985 - .09) / (3 + 2 * .08)
PH0 = 4.75 * (.84 - .15) / (2 + .16)
ASPECT = PH0 / PW0
PDF_DPI, PNG_DPI = 600, 300                    # PDF raster: >= 1 px per cell in both directions (checked)
MARGIN, PAD_PT, SEP_PT, LEG_GAP_PT, CB_GAP_PT, CB_H = .01, 1.5, 1.5, 3.0, 3.0, .05
PT = 1 / 72


def need(ok, msg):
    if not ok:
        raise AssertionError(msg)


def full_grid(arm, run, e, pool, ids, test):
    """One row's full-grid arrays, shape GRID_SHAPE (theta index, omega index). p_s, p_f, p_inv are tied with R.tie as
    in uncertainty_signals_data.analyse; I, D, the A-1d decisions (Success = S & ~F, Fail = F & ~S, Abstain
    otherwise) and the strict abstention kinds by its formulas; d_train against the run's own training states
    (Pool.states(ids[:n_train])) with its normalisation and periodic tree. ``test`` is analyse()'s arrays for the
    same run and round: every test cell must reproduce them exactly."""
    R, C = usd.R, usd.C
    z = np.load(GRID_DIR / GRID_FILE[arm], allow_pickle=False)
    prov = json.loads((GRID_DIR / GRID_FILE[arm].replace('.npz', '_provenance.json')).read_text())
    need(prov['scored_name'] == run['name'] and int(z['seed']) == 99512 and int(z['epoch']) == 10,
         f'{arm}: grid file is not {run["name"]} seed 99512 epoch 10')
    ni, nj = GRID_SHAPE
    need(tuple(z['grid_shape']) == GRID_SHAPE and len(z['theta_values']) == ni and len(z['omega_values']) == nj,
         f'{arm}: grid shape')
    ti, oi = z['theta_idx'].astype(int), z['omega_idx'].astype(int)
    cell = ti * nj + oi
    need(len(cell) == ni * nj and np.array_equal(np.sort(cell), np.arange(ni * nj)),
         f'{arm}: the grid is not every cell exactly once')
    need(np.array_equal(cell, np.arange(ni * nj)), f'{arm}: rows are not theta-major')
    X = z['start_states'].astype(float)
    ps, pf, pinv = R.tie(z['p_success']), R.tie(z['p_failure']), R.tie(z['p_invalid'])
    q = np.where(ps + pf > 0, ps / np.where(ps + pf > 0, ps + pf, 1), 0.5)
    I, D = pinv, np.where(ps + pf > 0, (1 - pinv) * usd.h2(q), 0.0)
    t_s, t_f, u_s, u_f = e['c6c_t_s'], e['c6c_t_f'], e['c6c_u_s'], e['c6c_u_f']
    S, F = C.in_S(ps, pf, t_s, t_f), C.in_F(ps, pf, u_s, u_f)
    succ, fail = S & ~F, F & ~S
    abst = ~(succ | fail)
    T = R.tie
    inv = (ps <= T(t_s)) & (pf <= T(u_f))
    weak = (ps > T(t_s)) & (ps <= T(u_s)) & (pf > T(u_f)) & (pf <= T(t_f))
    ou = (ps > T(u_s)) & (pf > T(t_f))
    need(not ((inv | weak | ou) & ~abst).any(), f'{arm}: an abstention kind outside the abstentions')
    other, overlap = abst & ~(inv | weak | ou), inv & ou
    need(not other.any() and not overlap.any() and not (inv & weak).any() and not (weak & ou).any(),
         f'{arm}: overlap or uncategorized abstentions present ({int(overlap.sum())}, {int(other.sum())}); '
         'no colour is approved for them')
    y = z['true_label'].astype(int)
    need(set(np.unique(y)) <= {-1, 0, 1}, f'{arm}: labels outside -1, 0, 1')
    known = y >= 0
    wrong = known & ((succ & (y == 0)) | (fail & (y == 1)))       # timeouts (-1) are never wrong
    art = json.load(open(run['dir'] / 'epoch_010/artifacts_v2.json'))
    n_train = art['train_trajectories']
    Xtr = pool.states(ids[:n_train])
    d_train, _ = usd.tree(Xtr).query(usd.norm(X))
    src = z['source'].astype(str)

    # Test cells reproduce analyse() exactly, and the stored decided-set sizes.
    ix = np.rint((test['X'][:, 0] - z['theta_values'][0]) / .04).astype(int) * nj + \
        np.rint((test['X'][:, 1] - z['omega_values'][0]) / .04).astype(int)
    need(len(np.unique(ix)) == len(ix) and (src[ix] == 'test').all() and (src == 'test').sum() == len(ix),
         f'{arm}: test states do not map one-to-one onto the grid\'s test cells')
    need(np.abs(X[ix] - test['X']).max() < 1e-5, f'{arm}: test coordinates differ from the grid')
    for k, g in dict(I=I, D=D, succ=succ, fail=fail, inv=inv, weak=weak, ou=ou, other=other, wrong=wrong).items():
        need(np.array_equal(g[ix], test[k]), f'{arm}: test cells differ from analyse() in {k}')
    need(np.array_equal(y[ix] == 1, test['y']) and known[ix].all(), f'{arm}: test labels differ')
    need(np.allclose(d_train[ix], test['d_train'], atol=1e-6, rtol=0), f'{arm}: test d_train differs')
    need(np.array_equal(d_train[ix] > usd.SPARSE, test['sparse']), f'{arm}: test sparse flags differ')
    tc = (int(succ[ix].sum()), int(fail[ix].sum()), int(abst[ix].sum()))
    need(tc == EXPECT_TEST[arm] and tc[:2] == (e['c6c_S'], e['c6c_F']),
         f'{arm}: test decisions {tc}, expected {EXPECT_TEST[arm]} (c6c_S {e["c6c_S"]}, c6c_F {e["c6c_F"]})')

    grid = lambda v: v.reshape(GRID_SHAPE)
    cat = np.full(ni * nj, -1)
    fp, fn = known & succ & (y == 0), known & fail & (y == 1)
    need(np.array_equal(fp | fn, wrong), f'{arm}: FP and FN do not make up the wrong declarations')
    for n, (key, _, _) in enumerate(R_CATS):
        m = dict(success=succ & ~fp, fail=fail & ~fn, fp=fp, fn=fn, inv=inv, weak=weak, ou=ou)[key]
        need(not (m & (cat >= 0)).any(), f'{arm}: {key} overlaps another category')
        cat[m] = n
    need((cat >= 0).all(), f'{arm}: a cell has no category')
    srcs = ['test', 'cal', 'val', 'fresh']
    count = lambda m: int(m.sum())
    info = dict(arm=arm, run=run['name'], thresholds=dict(t_s=t_s, t_f=t_f, u_s=u_s, u_f=u_f), n_train=n_train,
                n_train_states=len(Xtr), test_counts=tc,
                source_counts={s: count(src == s) for s in srcs},
                counts={s: dict(success=count(succ & (src == s)), fail=count(fail & (src == s)),
                                abstain=count(abst & (src == s)), invalid=count(inv & (src == s)),
                                weak=count(weak & (src == s)), outcome_uncertain=count(ou & (src == s)),
                                wrong=count(wrong & (src == s)), fp=count(fp & (src == s)), fn=count(fn & (src == s)),
                                timeouts=count((y < 0) & (src == s)))
                        for s in srcs},
                full=dict(success=count(succ), fail=count(fail), abstain=count(abst), invalid=count(inv),
                          weak=count(weak), outcome_uncertain=count(ou), overlap=count(overlap), other=count(other),
                          wrong=count(wrong), fp=count(fp), fn=count(fn), labelled=count(known),
                          timeouts=count(~known),
                          timeout_decisions=dict(success=count(succ & ~known), fail=count(fail & ~known),
                                                 abstain=count(abst & ~known)),
                          sparse=count(d_train > usd.SPARSE)))
    return dict(theta=np.asarray(z['theta_values'], float), omega=np.asarray(z['omega_values'], float),
                I=grid(I), D=grid(D), cat=grid(cat), y=grid(y), wrong=grid(wrong), d_train=grid(d_train),
                info=info)


def load_full_grid():
    pool, ep, runs = usd.Pool(), usd.epochs(), usd.runs()
    G = {}
    for arm, _ in R_ROWS:
        run = runs[arm, 99512]
        ids, _cfg = usd.verify_pool(pool, run)
        a = usd.analyse(run, 99512, 10, ep, pool, ids)
        need(a['arrays'] is not None, f'{arm}: round 10 is not certified')
        G[arm] = full_grid(arm, run, ep[run['name'], 99512, 10], pool, ids, a['arrays'])
    need(all(np.array_equal(G['adaptive'][k], G['uniform'][k]) for k in ('theta', 'omega')), 'rows on different grids')
    return G


def _extent(g):
    th, om = g['theta'], g['omega']
    dth, dom = np.diff(th), np.diff(om)
    need(np.allclose(dth, .04, atol=1e-9) and np.allclose(dom, .04, atol=1e-9), 'lattice spacing is not 0.04')
    return [th[0] - .02, th[-1] + .02, om[0] - .02, om[-1] + .02]


def _cat_rgb(cat):
    from matplotlib.colors import to_rgb
    rgb = np.zeros(cat.shape + (3,))
    for n, (_, _, col) in enumerate(R_CATS):
        rgb[cat == n] = to_rgb(col)
    return rgb


def restyle_figure(G):
    """Build Figure 5 in the paper style at its print width; returns (fig, layout). Every gap comes from a first draw
    that measures the text (titles, thresholds, row names, tick and axis labels, legend, colorbar), then panels and
    text are placed in inches. Fails on overlapping text, text off the page, text over a panel, text over another
    panel's ticks, clipped data (an image not showing all its cells, a cell lost to rasterisation, a contour off its
    panel), FP and FN cells that are not the wrong declarations, anything read back from the drawn artists that is
    not the computed grid, or outer whitespace outside 0-0.03 in."""
    W = PRINT_WIDTH[R_STEM]
    g0 = G['adaptive']
    ext = _extent(g0)
    th, om = g0['theta'], g0['omega']
    TH, OM = np.meshgrid(th, om, indexing='ij')
    ticks_x, ticks_y = [-np.pi, 0, np.pi], [-2 * np.pi, 0, 2 * np.pi]
    with plt.rc_context(style_rc()):
        H0 = 3.5
        fig = plt.figure(figsize=(W, H0))
        axs = [[fig.add_axes([.12 + .29 * c, .5 - .4 * r, .25, .3]) for c in range(3)] for r in range(2)]
        im_signal, rare = None, []
        for r, (arm, _) in enumerate(R_ROWS):
            g = G[arm]
            for c, key in enumerate(('I', 'D')):
                im = axs[r][c].imshow(g[key].T, origin='lower', extent=ext, interpolation='nearest', aspect='auto',
                                      cmap=SIGNAL_CMAP, vmin=0, vmax=1, zorder=1)
                im_signal = im_signal or im
            axs[r][2].imshow(_cat_rgb(g['cat']).transpose(1, 0, 2), origin='lower', extent=ext,
                             interpolation='nearest', aspect='auto', zorder=1)
            if RARE_PT:
                keys = [k_ for k_, _, _ in R_CATS]
                for key in RARE_KEYS:
                    m = g['cat'] == keys.index(key)
                    rare.append((arm, key, axs[r][2].scatter(TH[m], OM[m], s=RARE_PT ** 2, marker='s', linewidths=0,
                                                             c=[R_CATS[keys.index(key)][2]], zorder=5)))
            ylab = np.ma.masked_less(g['y'].astype(float), 0)          # timeouts (-1) masked out of the boundary
            for ax in axs[r]:
                cs = ax.contour(th, om, ylab.T, levels=[.5], colors=FILL['boundary'], linewidths=STYLE_LW['boundary'],
                                zorder=4 if RARE_PT else 6)
                cs.set_path_effects([withStroke(linewidth=STYLE_LW['boundary'] + 2 * CASING_PT, foreground='white')])
                cs = ax.contour(th, om, g['d_train'].T, levels=[usd.SPARSE], colors=FILL['boundary'],
                                linewidths=SPARSE_LW, linestyles='--', zorder=6)
                cs.set_path_effects([withStroke(linewidth=SPARSE_LW + 2 * CASING_PT, foreground='white')])
            for c, ax in enumerate(axs[r]):
                ax.set_xlim(ext[:2]); ax.set_ylim(ext[2:])
                ax.set_xticks(ticks_x); ax.set_yticks(ticks_y)
                ax.tick_params(labelbottom=(r == 1), labelleft=(c == 0))
                if r == 1:
                    ax.set_xticklabels([r'$-\pi$', '0', r'$\pi$'])
                    ax.set_xlabel(r'$\theta$', labelpad=1)
                if c == 0:
                    ax.set_yticklabels([r'$-2\pi$', '0', r'$2\pi$'])
                    ax.set_ylabel(r'$\dot{\theta}$', labelpad=1)
        # Titles: one Text per line on a common baseline grid (1.2 x the title size), so every line has its own box.
        titles = [[fig.text(0, 0, t, ha='center', va='baseline', fontsize=FS['title'])] for t in R_TITLES]
        # Thresholds above each C panel, right-aligned, on two lines (t_s, t_f, then u_s, u_f): one line at 7 pt is
        # wider than the panel and ran into column B ("D(x)  t_s=...").
        thr = []
        for arm, _ in R_ROWS:
            t = G[arm]['info']['thresholds']
            thr.append([fig.text(0, 0, s_, ha='right', va='baseline', fontsize=FS['annotation'])
                        for s_ in (rf"$\tau^{{\mathrm{{s}}}}_{{\mathrm{{s}}}}$={t['t_s']}, $\tau^{{\mathrm{{s}}}}_{{\mathrm{{f}}}}$={t['t_f']}",
                             rf"$\tau^{{\mathrm{{f}}}}_{{\mathrm{{s}}}}$={t['u_s']}, $\tau^{{\mathrm{{f}}}}_{{\mathrm{{f}}}}$={t['u_f']}")])
        rown = [fig.text(0, 0, name, rotation=90, ha='left', va='center', fontsize=FS['label']) for _, name in R_ROWS]
        cax = fig.add_axes([.12, .03, .5, .02])
        cb = fig.colorbar(im_signal, cax=cax, orientation='horizontal', ticks=[0, .25, .5, .75, 1])
        cb.set_label(r'$I(x)$ and $D(x)$ (common scale)', fontsize=FS['label'], labelpad=1)
        cax.tick_params(labelsize=FS['tick'])
        patch = {key: Patch(facecolor=col, edgecolor='none', label=lab) for key, lab, col in R_CATS}
        # Two legend rows (Sumanth, 2026-11-13: the four-row, three-column legend took too much height). Row A holds
        # the decisions and the two contours, row B the abstention kinds. Each row is its own centred legend, so
        # neither sets the other's column widths.
        row_a = [patch['success'], patch['fail'], patch['fp'], patch['fn'],
                 Line2D([], [], color=FILL['boundary'], lw=STYLE_LW['boundary'], label='success/failure boundary'),
                 Line2D([], [], color=FILL['boundary'], lw=SPARSE_LW, ls='--', label='sparse/dense boundary (0.02)')]
        row_b = [patch['inv'], patch['weak'], patch['ou']]
        ncol = len(row_a)
        legend = fig.legend(handles=row_a, ncol=len(row_a), **paper_legend_kw(FS['legend']))
        legend_b = fig.legend(handles=row_b, ncol=len(row_b), **paper_legend_kw(FS['legend']))
        LEG_ROW_GAP = 2.0 * PT

        # Pass 1: measure.
        fig.canvas.draw()
        rr = fig.canvas.get_renderer()
        inch = lambda bb: (bb.width / fig.dpi, bb.height / fig.dpi)
        need(max(inch(lg.get_window_extent(rr))[0] for lg in (legend, legend_b)) <= W - 2 * MARGIN,
             'a legend row is wider than the page')
        pos = lambda ax: ax.get_position()
        tb = lambda ax: ax.get_tightbbox(rr)
        left_dec = max(pos(ax).x0 * W - tb(ax).x0 / fig.dpi for ax in (axs[0][0], axs[1][0]))
        below = max(pos(ax).y0 * H0 - tb(ax).y0 / fig.dpi for ax in axs[1])
        below_top = max(pos(ax).y0 * H0 - tb(ax).y0 / fig.dpi for ax in axs[0])          # tick marks of row 1
        # what sticks out above each panel's top edge: the top y tick label (column A)
        above = [[max(0.0, tb(ax).y1 / fig.dpi - (pos(ax).y0 + pos(ax).height) * H0) for ax in row] for row in axs]
        # gap between columns: the right overhang of column c's labels plus the left overhang of column c + 1's
        gap = max(tb(axs[r][c]).x1 / fig.dpi - (pos(axs[r][c]).x0 + pos(axs[r][c]).width) * W
                  + pos(axs[r][c + 1]).x0 * W - tb(axs[r][c + 1]).x0 / fig.dpi for r in range(2) for c in range(2))
        gap += SEP_PT * PT
        right_over = max(0.0, max(tb(axs[r][2]).x1 / fig.dpi - (pos(axs[r][2]).x0 + pos(axs[r][2]).width) * W
                                  for r in range(2)))
        rn_w = max(inch(t.get_window_extent(rr))[0] for t in rown)
        rn_len = [inch(t.get_window_extent(rr))[1] for t in rown]
        thr_w = max(inch(t.get_window_extent(rr))[0] for lines in thr for t in lines)
        leg_a_h = inch(legend.get_window_extent(rr))[1]
        leg_h = leg_a_h + LEG_ROW_GAP + inch(legend_b.get_window_extent(rr))[1]
        cb_below = pos(cax).y0 * H0 - tb(cax).y0 / fig.dpi

        x_a = MARGIN + rn_w + PAD_PT * PT + left_dec
        pw = (W - MARGIN - right_over - x_a - 2 * gap) / 3
        ph = pw * ASPECT
        # One line per title if neighbours clear each other by SEP, else A and B on two lines at their break.
        tw = [inch(t[0].get_window_extent(rr))[0] for t in titles]
        wrapped = any(tw[c] / 2 + tw[c + 1] / 2 + SEP_PT * PT > pw + gap for c in range(2)) or \
            tw[0] / 2 > pw / 2 + left_dec + rn_w or tw[2] / 2 > pw / 2 + right_over
        if wrapped:
            for c, s_ in enumerate(R_TITLES_WRAPPED):
                lines = s_.split('\n')
                titles[c][0].set_text(lines[0])
                titles[c] += [fig.text(0, 0, l_, ha='center', va='baseline', fontsize=FS['title']) for l_ in lines[1:]]
            fig.canvas.draw()

        def asc_desc(t):
            bb = t.get_window_extent(rr)
            base = fig.transFigure.transform(t.get_position())[1]
            return (bb.y1 - base) / fig.dpi, (base - bb.y0) / fig.dpi

        SKIP, SKIP_A = 1.2 * FS['title'] * PT, 1.2 * FS['annotation'] * PT
        asc0 = max(asc_desc(t[0])[0] for t in titles)
        thr_asc = max(asc_desc(lines[0])[0] for lines in thr)
        thr_desc = max(asc_desc(lines[-1])[1] for lines in thr)
        # The tau symbols carry a superscript and a subscript, so two threshold lines need more than the plain 1.2 x
        # font-size skip: at least the first line's descent plus the second line's ascent, and half a point between.
        skip_thr = max([SKIP_A] + [asc_desc(lines[0])[1] + asc_desc(lines[1])[0] + .5 * PT for lines in thr])
        thr_h = thr_asc + skip_thr + thr_desc
        # header: from the top of the first title line down to the top of row 1's panels
        title_depth = [asc0 + (len(t) - 1) * SKIP + asc_desc(t[-1])[1] for t in titles]
        thr_pad = [max(PAD_PT * PT, above[r][2] + .5 * PT) for r in range(2)]
        header = max([title_depth[c] + SEP_PT * PT + above[0][c] for c in range(2)] +
                     [title_depth[2] + SEP_PT * PT + thr_h + thr_pad[0]])
        row_gap = max(below_top + SEP_PT * PT + thr_h + thr_pad[1], below_top + SEP_PT * PT + above[1][0])
        H = (MARGIN + leg_h + LEG_GAP_PT * PT + header + 2 * ph + row_gap + below + CB_GAP_PT * PT + CB_H + cb_below
             + MARGIN)
        fig.set_size_inches(W, H)

        # Place.
        y_head = H - MARGIN - leg_h - LEG_GAP_PT * PT
        tops = [y_head - header, y_head - header - ph - row_gap]
        xs = [x_a + c * (pw + gap) for c in range(3)]
        for r in range(2):
            for c in range(3):
                axs[r][c].set_position([xs[c] / W, (tops[r] - ph) / H, pw / W, ph / H])
            for k_, t in enumerate(thr[r]):                    # last line's descent sits thr_pad above the panel
                t.set_position(((xs[2] + pw) / W, (tops[r] + thr_pad[r] + thr_desc + (1 - k_) * skip_thr) / H))
            rown[r].set_position((MARGIN / W, (tops[r] - ph / 2) / H))
        for c, lines in enumerate(titles):
            for k_, t in enumerate(lines):
                w_ = inch(t.get_window_extent(rr))[0]
                xc = min(max(xs[c] + pw / 2, MARGIN + w_ / 2), W - MARGIN - w_ / 2)   # centred, kept on the page
                t.set_position((xc / W, (y_head - asc0 - k_ * SKIP) / H))
        cb_top = tops[1] - ph - below - CB_GAP_PT * PT
        cax.set_position([xs[0] / W, (cb_top - CB_H) / H, (xs[1] + pw - xs[0]) / W, CB_H / H])
        legend.set_bbox_to_anchor((0.5, (H - MARGIN) / H), transform=fig.transFigure)
        legend_b.set_bbox_to_anchor((0.5, (H - MARGIN - leg_a_h - LEG_ROW_GAP) / H), transform=fig.transFigure)

        # Checks.
        fig.canvas.draw()
        rr = fig.canvas.get_renderer()
        maps = [ax for row in axs for ax in row]

        def drawn(axis, lim):
            lo, hi = sorted(lim)
            return [t for t, v in zip(axis.get_ticklabels(), axis.get_majorticklocs())
                    if t.get_text() and t.get_visible() and lo - 1e-9 <= v <= hi + 1e-9]

        free = [(t.get_text(), t.get_window_extent(rr)) for t in fig.texts] + \
            [('legend', legend.get_window_extent(rr)), ('legend row B', legend_b.get_window_extent(rr))]
        owned = []
        for ax in maps + [cax]:
            owned += [(t.get_text(), t.get_window_extent(rr)) for t in drawn(ax.xaxis, ax.get_xlim()) +
                      drawn(ax.yaxis, ax.get_ylim())]
            owned += [(lab.get_text(), lab.get_window_extent(rr)) for lab in (ax.xaxis.label, ax.yaxis.label)
                      if lab.get_text()]
        boxes = free + owned
        need(sum(len(drawn(ax.xaxis, ax.get_xlim())) for ax in maps) == 9 and
             sum(len(drawn(ax.yaxis, ax.get_ylim())) for ax in maps) == 6, 'expected 9 x and 6 y tick labels')
        for i, (ta, a) in enumerate(boxes):
            for tbx, b in boxes[i + 1:]:
                need(not a.overlaps(b), ('text overlap', ta, tbx))
            need(a.x0 >= -0.5 and a.x1 <= fig.bbox.x1 + 0.5 and a.y0 >= -0.5 and a.y1 <= fig.bbox.y1 + 0.5,
                 ('text off the page', ta))
        panels = [ax.get_window_extent(rr) for ax in maps + [cax]]
        for ta, a in boxes:
            need(not any(a.overlaps(p) for p in panels), ('text over a panel', ta))
        for ta, a in free:                      # free text clear of every panel's tick marks and labels
            for ax in maps + [cax]:
                need(not a.overlaps(ax.get_tightbbox(rr)), ('text over a panel\'s decorations', ta))
        # Clipped data and read-back.
        need(PDF_DPI * pw >= GRID_SHAPE[0] and PDF_DPI * ph >= GRID_SHAPE[1], 'a cell is lost in the PDF raster')
        for r, (arm, _) in enumerate(R_ROWS):
            g = G[arm]
            for c, ax in enumerate(axs[r]):
                ims = ax.get_images()
                need(len(ims) == 1, f'{arm} {"ABC"[c]}: {len(ims)} images')
                e = ims[0].get_extent()
                need(np.allclose(e, ext) and np.allclose(list(ax.get_xlim()) + list(ax.get_ylim()), ext),
                     f'{arm} {"ABC"[c]}: image and axes limits differ (clipped cells)')
                arr = ims[0].get_array()
                need(not np.ma.getmaskarray(arr).any() and np.isfinite(np.ma.getdata(arr)).all(),
                     f'{arm} {"ABC"[c]}: missing cells')
                if c < 2:
                    need(np.array_equal(np.ma.getdata(arr), g['I' if c == 0 else 'D'].T) and
                         (ims[0].norm.vmin, ims[0].norm.vmax, ims[0].get_cmap().name) == (0, 1, SIGNAL_CMAP),
                         f'{arm} {"AB"[c]}: image is not the grid on the shared scale')
                else:
                    rgb = np.ma.getdata(arr)
                    need(np.array_equal(rgb, _cat_rgb(g['cat']).transpose(1, 0, 2)), f'{arm} C: decisions differ')
                    need(not (rgb == 1).all(-1).any(), f'{arm} C: white cells')
                for coll in ax.collections:
                    for p in coll.get_paths():
                        v = p.vertices
                        need(not len(v) or ((v[:, 0] >= ext[0]) & (v[:, 0] <= ext[1]) & (v[:, 1] >= ext[2]) &
                                            (v[:, 1] <= ext[3])).all(), f'{arm} {"ABC"[c]}: contour off the panel')
            if RARE_PT:
                keys = [k_ for k_, _, _ in R_CATS]
                for arm_, key, sc in rare:
                    if arm_ == arm:
                        m = g['cat'] == keys.index(key)
                        need(np.array_equal(sc.get_offsets(), np.c_[TH[m], OM[m]]), f'{arm}: {key} squares off cells')
            fpn = sum(int((g['cat'] == n).sum()) for n, (key, _, _) in enumerate(R_CATS) if key in ('fp', 'fn'))
            need(fpn == g['info']['full']['wrong'] == int(g['wrong'].sum()), f'{arm}: FP + FN cells != wrong')
        ink = [b for _, b in boxes] + [ax.get_tightbbox(rr) for ax in maps + [cax]]
        edges = dict(left=min(b.x0 for b in ink) / fig.dpi, right=W - max(b.x1 for b in ink) / fig.dpi,
                     bottom=min(b.y0 for b in ink) / fig.dpi, top=H - max(b.y1 for b in ink) / fig.dpi)
        need(all(-0.005 <= v <= 0.03 for v in edges.values()), ('outer whitespace', edges))
        for lg in (legend, legend_b):
            lb = lg.get_window_extent(rr)
            need(abs((lb.x0 + lb.x1) / 2 / fig.dpi - W / 2) < 0.005, 'legend not centred')
    layout = dict(width=W, height=H, panel_w=pw, panel_h=ph, col_gap=gap, row_gap=row_gap, legend_ncol=ncol,
                  titles_wrapped=wrapped, edges=edges, rare_pt=RARE_PT,
                  cell_pt=(72 * pw / GRID_SHAPE[0], 72 * ph / GRID_SHAPE[1]), thr_w=thr_w, rn_len=rn_len)
    return fig, layout


def _pdf_size(pdf):
    m = re.search(rb'/MediaBox\s*\[\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)', pdf.read_bytes())
    x0, y0, x1, y1 = map(float, m.groups())
    return (x1 - x0) / 72, (y1 - y0) / 72


def sidecar(G, L):
    out = [f'pendulum_uncertainty_signals, restyled (common.py paper style, usetex Times bold), full 158 x 315 grid.',
           f'Size {L["width"]:.3f} x {L["height"]:.3f} in; panels {L["panel_w"]:.3f} x {L["panel_h"]:.3f} in '
           f'(aspect of the paper copy, {ASPECT:.4f}); column gap {L["col_gap"]:.3f} in, row gap {L["row_gap"]:.3f} in; '
           f'cell {L["cell_pt"][0]:.2f} x {L["cell_pt"][1]:.2f} pt; outer whitespace (in) '
           + ', '.join(f'{k} {v:.3f}' for k, v in L['edges'].items()) + '.',
           f'Fonts (pt): {FS}. Legend at the top, {L["legend_ncol"]} columns; titles '
           f'{"on two lines (A, B)" if L["titles_wrapped"] else "on one line"}.',
           f'Colours: I(x), D(x) {SIGNAL_CMAP} on one [0, 1] scale; Success {FILL["success"]}, Fail {FILL["fail"]}, '
           f'invalid {ABSTAIN_GRAY["inv"]}, weak {ABSTAIN_GRAY["weak"]}, outcome-uncertain {ABSTAIN_GRAY["ou"]}; '
           f'wrong Success (FP) {FILL["fp"]}, wrong Fail (FN) {FILL["fn"]} (Figure 1 fills, one cell each); true '
           f'boundary black solid {STYLE_LW["boundary"]} pt; sparse/dense boundary black dashed {SPARSE_LW} pt; both '
           f'with a white casing {CASING_PT} pt each side.'
           + (f' Abstention, FP and FN cells in C drawn as {L["rare_pt"]} pt squares over the true boundary.'
              if L['rare_pt'] else ' All cells at true size.'),
           'Drawing: every grid cell is its own start state, one image cell per state with exact edges (0.04 rad '
           'lattice, cell edges at +-0.02 of each state; the axes run edge to edge, so no cell is clipped), nearest '
           f'interpolation, PDF raster {PDF_DPI} dpi. True boundary: the full-grid labels contoured at 0.5 on the '
           'lattice with the 68 timeouts (label -1) masked, so the line skips every lattice quad with a timeout '
           'corner. Sparse/dense: d_train > 0.02 for every grid state (uncertainty_signals_data.norm, periodic tree, '
           'the run\'s own training states), contoured at 0.02 on the lattice. FP / FN: wrong declarations where the true '
           'label is 0 or 1; timeouts keep their decision\'s colour.',
           'Decisions: S = in_S(ps, pf, t_s, t_f), F = in_F(ps, pf, u_s, u_f); Success = S & ~F, Fail = F & ~S, '
           'Abstain otherwise (A-1d scorer). p_s, p_f, p_inv tied with det_v2_rule.tie. Test cells reproduce '
           'uncertainty_signals_data.analyse exactly (I, D, decisions, kinds, wrong declarations, labels, sparse flags).', '']
    for arm, name in R_ROWS:
        i = G[arm]['info']
        f = i['full']
        out += [f'[{arm}] {i["run"]} seed 99512 round 10, thresholds {i["thresholds"]}, training trajectories '
                f'{i["n_train"]} ({i["n_train_states"]} states)',
                f'    source counts: ' + ', '.join(f'{s} {n}' for s, n in i['source_counts'].items()),
                f'    test cells: Success {i["test_counts"][0]}, Fail {i["test_counts"][1]}, Abstain '
                f'{i["test_counts"][2]} (stored c6c: {EXPECT_TEST[arm]})',
                f'    full grid ({sum(i["source_counts"].values())} states): Success {f["success"]}, Fail {f["fail"]}, '
                f'Abstain {f["abstain"]} (invalid {f["invalid"]}, weak {f["weak"]}, outcome-uncertain '
                f'{f["outcome_uncertain"]}, overlap {f["overlap"]}, other {f["other"]}); wrong declarations '
                f'{f["wrong"]} (FP {f["fp"]}, FN {f["fn"]}) of {f["labelled"]} labelled; timeouts {f["timeouts"]} (Success '
                f'{f["timeout_decisions"]["success"]}, Fail {f["timeout_decisions"]["fail"]}, Abstain '
                f'{f["timeout_decisions"]["abstain"]}); sparse {f["sparse"]}']
        for s, c in i['counts'].items():
            out.append(f'    {s:5s}: ' + ', '.join(f'{k} {v}' for k, v in c.items()))
    return '\n'.join(out) + '\n'


def main_restyle():
    G = load_full_grid()
    fig, L = restyle_figure(G)
    paper_version = RARE_PT == PAPER_RARE_PT
    out = PLOTS if paper_version else PLOTS / 'variants'
    out.mkdir(parents=True, exist_ok=True)
    stem = R_STEM if paper_version else R_STEM + (f'_rare{RARE_PT:g}pt' if RARE_PT else '_truesize')
    pdf, png, txt = out / f'{stem}.pdf', out / f'{stem}.png', out / f'{stem}.txt'
    with plt.rc_context(style_rc()):
        fig.savefig(pdf, dpi=PDF_DPI, metadata={'CreationDate': None})
        fig.savefig(png, dpi=PNG_DPI)
    plt.close(fig)
    w, h = _pdf_size(pdf)
    need(abs(w - PRINT_WIDTH[R_STEM]) <= .01, f'PDF width {w:.3f} in, print width {PRINT_WIDTH[R_STEM]}')
    txt.write_text(sidecar(G, L))
    print(f'{R_STEM} (restyle): {w:.3f} x {h:.3f} in, panels {L["panel_w"]:.3f} x {L["panel_h"]:.3f} in -> {pdf}')
    for arm, _ in R_ROWS:
        f = G[arm]['info']['full']
        print(f'  {arm}: full grid Success {f["success"]}, Fail {f["fail"]}, Abstain {f["abstain"]}, wrong {f["wrong"]}; '
              f'test {G[arm]["info"]["test_counts"]}')
    return pdf, png, txt, G, L


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument('--paper', action='store_true',
                   help='also copy plots/pendulum_uncertainty_signals.pdf (the paper version) into the paper clone')
    p.add_argument('--baseline', action='store_true',
                   help='draw the 2026-11-12 baseline instead (plots/legacy/; rewrites values/pendulum_uncertainty_*)')
    p.add_argument('--restyle', action='store_true', help='accepted for old commands; the default is the paper figure')
    p.add_argument('--rare-pt', type=float, default=PAPER_RARE_PT,
                   help=f'abstention / FP / FN cells in C as squares of this size in pt (paper: {PAPER_RARE_PT:g}; '
                        '0 = true cell size); other values go to plots/variants/')
    args = p.parse_args()
    if args.baseline:
        if args.restyle or args.rare_pt != PAPER_RARE_PT:
            p.error('--baseline draws the old design only')
        main(False)
    else:
        RARE_PT = args.rare_pt
        main_restyle()
        if args.paper:
            if RARE_PT != PAPER_RARE_PT:
                p.error(f'only the paper version (--rare-pt {PAPER_RARE_PT:g}) goes to the paper')
            install_paper(R_STEM)
