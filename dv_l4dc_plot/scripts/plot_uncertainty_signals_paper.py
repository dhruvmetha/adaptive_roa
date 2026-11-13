"""Two uncertainty signals in pendulum state space: paper figure pendulum_uncertainty_signals.

Two paths:
  (default)   THE PAPER FIGURE since 2026-11-13 afternoon: one row, TERA-FM (U_H, rho 0.75) seed 99512 at round 2,
              on the FULL 158 x 315 grid (data/pendulum_full_grid/, round-2 files from build_pendulum_full_grid.py).
              Columns: A abstention as a whole, B invalid mass I(x), C weighted disagreement D(x), every cell at true
              size, at the 3.00 in print width. Writes plots/pendulum_uncertainty_signals.{pdf,png,txt} and nothing
              else; --round N draws another round into plots/variants/. --paper copies the paper version into the
              paper clone. See restyle_figure() for the layout and its checks. (The two-row adaptive / uniform design
              at round 10 with a decisions column and 1 pt abstention squares is in git at e4fe9d2a.)
  --baseline  the 3-column design of 2026-11-12 (md5 51f991cf), unchanged: test states as scatter squares, DejaVu
              fonts; the figure goes to plots/legacy/, and it rewrites values/pendulum_uncertainty_signals_*, which
              the work-in-progress plot_uncertainty_signals.py also writes (today's files are the WIP's).
  --restyle   accepted for old commands; the default already is the paper figure.

--baseline, as of 2026-11-12: TERA-FM on Pendulum LQR, seed 99512, round 10. Rows: adaptive (U_H,
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


# ==== The paper figure (2026-11-13 afternoon, Sumanth via paper_writer): one TERA row at round 2 ======================
# Figure 5 shows one model, TERA-FM (U_H, rho = 0.75), seed 99512, at round 2 (the Appendix F comparison round, where
# abstention is common enough to show at true cell size). Columns: A abstention as a whole, B invalid mass I(x),
# C weighted disagreement D(x). Every map is drawn on the full 158 x 315 pendulum grid, one image cell per start
# state, nothing enlarged. Panels share axes, so only A carries tick labels and axis labels; B and C carry their own
# colour ramps (common.SIGNAL_RAMP), with colourbars under them in the band A uses for its x labels.
# The two-row design with a decisions column (adaptive and uniform, round 10, 1 pt abstention squares; paper commit
# 742c412) is in git history at e4fe9d2a.
import argparse, re
import uncertainty_signals_data as usd
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.patheffects import withStroke
from common import (HERE, PLOTS, FILL, LW as STYLE_LW, FS, PRINT_WIDTH, style_rc, paper_legend_kw, SIGNAL_RAMP,
                    ABSTAIN_ON_WHITE, install_paper)

R_STEM = 'pendulum_uncertainty_signals'
PAPER_ROUND = 2
R_ARM, R_RUN = 'adaptive', 'x231_penlqr_fm_uh_a6d075_r1_s99512'
GRID_DIR = HERE / 'data' / 'pendulum_full_grid'
GRID_SHAPE = (158, 315)                        # theta x omega, rows theta-major
# Test-set Success, Fail, Abstain under the run's certified thresholds (epochs.jsonl c6c_S, c6c_F, the rest)
EXPECT_TEST = {2: (7458, 12682, 562), 10: (7784, 12762, 156)}
R_TITLES = [r'A. abstention', r'B. invalid mass $I(x)$', r'C. weighted disagreement $D(x)$']
R_TITLES_WRAPPED = [r'A. abstention', 'B. invalid mass\n' + r'$I(x)$', 'C. weighted\n' + r'disagreement $D(x)$']
CMAP = {k: LinearSegmentedColormap.from_list(f'fig5_{k}', v) for k, v in SIGNAL_RAMP.items()}
# Both contours wear a white casing (CASING_PT each side), so they read on the dark ends of the ramps.
CASING_PT = .3
SPARSE_LW = .45          # pt; '--' dashes scale with it
# At round 2 the abstention and D(x) bands are about two cells (0.8 pt) wide per boundary branch, so a 0.6 pt boundary
# with its casing covers them. BOUNDARY_PANELS says which panels draw the true boundary; BOUNDARY_LW its width there.
BOUNDARY_PANELS, BOUNDARY_LW, BOUNDARY_CASED = (0, 1, 2), .35, False
# The sparse/dense contour (d_train = 0.02) traces a maze of tubes around round 2's 120 training trajectories and
# buries the bands, so it is off by default.
SHOW_SPARSE = False
# Panel aspect (height / width) of the paper copy, kept since 2026-11-12.
PW0 = 6.5 * (.985 - .09) / (3 + 2 * .08)
PH0 = 4.75 * (.84 - .15) / (2 + .16)
ASPECT = PH0 / PW0
PDF_DPI, PNG_DPI = 600, 300                    # PDF raster: >= 1 px per cell in both directions (checked)
MARGIN, SEP_PT, LEG_GAP_PT, TITLE_GAP_PT, CB_GAP_PT, CB_H, GAP_PT = .01, 1.5, 3.0, 2.0, 3.0, .045, 4.0
PT = 1 / 72
# matplotlib's usetex extents leave out the accent of \dot{\theta}: rotated, the dot sits 0.025 in (at 8 pt) left of
# the reported box, so the layout reserves this much beyond it. main_restyle checks the page's real ink margins.
ACCENT_PT = 2.0


def need(ok, msg):
    if not ok:
        raise AssertionError(msg)


def full_grid(rnd, run, e, pool, ids, test):
    """The round's full-grid arrays, shape GRID_SHAPE (theta index, omega index). p_s, p_f, p_inv are tied with R.tie
    as in uncertainty_signals_data.analyse; I, D, the A-1d decisions (Success = S & ~F, Fail = F & ~S, Abstain
    otherwise) and the strict abstention kinds by its formulas; d_train against the run's own training states
    (Pool.states(ids[:n_train])) with its normalisation and periodic tree. ``test`` is analyse()'s arrays for the
    same run and round: every test cell must reproduce them exactly."""
    R, C = usd.R, usd.C
    f = GRID_DIR / f'{R_RUN}_epoch{rnd:03d}_grid.npz'
    z = np.load(f, allow_pickle=False)
    prov = json.loads(f.with_name(f.stem + '_provenance.json').read_text())
    need(prov['scored_name'] == run['name'] and int(z['seed']) == 99512 and int(z['epoch']) == rnd,
         f'grid file is not {run["name"]} seed 99512 round {rnd}')
    ni, nj = GRID_SHAPE
    need(tuple(z['grid_shape']) == GRID_SHAPE and len(z['theta_values']) == ni and len(z['omega_values']) == nj,
         'grid shape')
    ti, oi = z['theta_idx'].astype(int), z['omega_idx'].astype(int)
    need(np.array_equal(ti * nj + oi, np.arange(ni * nj)), 'the grid is not every cell exactly once, theta-major')
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
    need(not ((inv | weak | ou) & ~abst).any(), 'an abstention kind outside the abstentions')
    other = abst & ~(inv | weak | ou)
    y = z['true_label'].astype(int)
    need(set(np.unique(y)) <= {-1, 0, 1}, 'labels outside -1, 0, 1')
    known = y >= 0
    fp, fn = known & succ & (y == 0), known & fail & (y == 1)        # timeouts (-1) are never wrong
    wrong = fp | fn
    art = json.load(open(run['dir'] / f'epoch_{rnd:03d}/artifacts_v2.json'))
    n_train = art['train_trajectories']
    Xtr = pool.states(ids[:n_train])
    d_train, _ = usd.tree(Xtr).query(usd.norm(X))
    src = z['source'].astype(str)

    # Test cells reproduce analyse() exactly, and the stored decided-set sizes.
    ix = np.rint((test['X'][:, 0] - z['theta_values'][0]) / .04).astype(int) * nj + \
        np.rint((test['X'][:, 1] - z['omega_values'][0]) / .04).astype(int)
    need(len(np.unique(ix)) == len(ix) and (src[ix] == 'test').all() and (src == 'test').sum() == len(ix),
         'test states do not map one-to-one onto the grid\'s test cells')
    need(np.abs(X[ix] - test['X']).max() < 1e-5, 'test coordinates differ from the grid')
    for k, g in dict(I=I, D=D, succ=succ, fail=fail, inv=inv, weak=weak, ou=ou, other=other, wrong=wrong).items():
        need(np.array_equal(g[ix], test[k]), f'test cells differ from analyse() in {k}')
    need(np.array_equal(y[ix] == 1, test['y']) and known[ix].all(), 'test labels differ')
    need(np.allclose(d_train[ix], test['d_train'], atol=1e-6, rtol=0), 'test d_train differs')
    need(np.array_equal(d_train[ix] > usd.SPARSE, test['sparse']), 'test sparse flags differ')
    tc = (int(succ[ix].sum()), int(fail[ix].sum()), int(abst[ix].sum()))
    need(tc == EXPECT_TEST[rnd] and tc[:2] == (e['c6c_S'], e['c6c_F']),
         f'test decisions {tc}, expected {EXPECT_TEST[rnd]} (c6c_S {e["c6c_S"]}, c6c_F {e["c6c_F"]})')
    need(abs(tc[2] / len(ix) - e['c6c_abstain']) < 1e-12, 'test abstention share differs from c6c_abstain')

    grid = lambda v: v.reshape(GRID_SHAPE)
    count = lambda m: int(m.sum())
    srcs = ['test', 'cal', 'val', 'fresh']
    info = dict(run=run['name'], round=rnd, thresholds=dict(t_s=t_s, t_f=t_f, u_s=u_s, u_f=u_f), n_train=n_train,
                n_train_states=len(Xtr), test_counts=tc, test_wrong=count(wrong[ix]),
                test_fp=count(fp[ix]), test_fn=count(fn[ix]),
                source_counts={s: count(src == s) for s in srcs},
                full=dict(success=count(succ), fail=count(fail), abstain=count(abst), invalid=count(inv),
                          weak=count(weak), outcome_uncertain=count(ou), other=count(other),
                          wrong=count(wrong), fp=count(fp), fn=count(fn), labelled=count(known),
                          timeouts=count(~known), sparse=count(d_train > usd.SPARSE),
                          abstain_isolated=None))
    A = grid(abst)
    nb = np.zeros_like(A)
    nb[1:] |= A[:-1]; nb[:-1] |= A[1:]; nb[:, 1:] |= A[:, :-1]; nb[:, :-1] |= A[:, 1:]
    info['full']['abstain_isolated'] = count(A & ~nb)
    return dict(theta=np.asarray(z['theta_values'], float), omega=np.asarray(z['omega_values'], float),
                I=grid(I), D=grid(D), abst=A, y=grid(y), d_train=grid(d_train), info=info)


def load_full_grid(rnd):
    pool, ep, runs = usd.Pool(), usd.epochs(), usd.runs()
    run = runs[R_ARM, 99512]
    ids, _cfg = usd.verify_pool(pool, run)
    a = usd.analyse(run, 99512, rnd, ep, pool, ids)
    need(a['arrays'] is not None, f'round {rnd} is not certified')
    return full_grid(rnd, run, ep[run['name'], 99512, rnd], pool, ids, a['arrays'])


def _extent(g):
    th, om = g['theta'], g['omega']
    need(np.allclose(np.diff(th), .04, atol=1e-9) and np.allclose(np.diff(om), .04, atol=1e-9),
         'lattice spacing is not 0.04')
    return [th[0] - .02, th[-1] + .02, om[0] - .02, om[-1] + .02]


def _abst_rgba(abst):
    rgba = np.zeros(abst.shape + (4,))
    rgba[abst] = to_rgb(ABSTAIN_ON_WHITE) + (1.0,)
    return rgba


def restyle_figure(g):
    """Build Figure 5 at its print width; returns (fig, layout). A first draw measures the text (legend, titles, tick
    and axis labels, colourbar ticks), then panels and text are placed in inches. Fails on overlapping text, text off
    the page, text over a panel, free text over a panel's decorations, clipped data (an image not showing all its
    cells, a cell lost to rasterisation, a contour off its panel), anything read back from the drawn artists that is
    not the computed grid, or outer whitespace outside 0-0.03 in."""
    W = PRINT_WIDTH[R_STEM]
    ext = _extent(g)
    th, om = g['theta'], g['omega']
    with plt.rc_context(style_rc()):
        H0 = 2.0
        fig = plt.figure(figsize=(W, H0))
        axs = [fig.add_axes([.12 + .29 * c, .3, .25, .4]) for c in range(3)]
        # A: white page, the boundary, then the abstaining cells on top of it (transparent elsewhere)
        axs[0].imshow(_abst_rgba(g['abst']).transpose(1, 0, 2), origin='lower', extent=ext, interpolation='nearest',
                      aspect='auto', zorder=7)
        ims = {key: axs[c].imshow(g[key].T, origin='lower', extent=ext, interpolation='nearest', aspect='auto',
                                  cmap=CMAP[key], vmin=0, vmax=1, zorder=1) for c, key in ((1, 'I'), (2, 'D'))}
        ylab = np.ma.masked_less(g['y'].astype(float), 0)          # timeouts (-1) masked out of the boundary
        for c, ax in enumerate(axs):
            if c in BOUNDARY_PANELS:
                cs = ax.contour(th, om, ylab.T, levels=[.5], colors=FILL['boundary'], linewidths=BOUNDARY_LW, zorder=6)
                if BOUNDARY_CASED:
                    cs.set_path_effects([withStroke(linewidth=BOUNDARY_LW + 2 * CASING_PT, foreground='white')])
            if SHOW_SPARSE:
                cs = ax.contour(th, om, g['d_train'].T, levels=[usd.SPARSE], colors=FILL['boundary'],
                                linewidths=SPARSE_LW, linestyles='--', zorder=6)
                cs.set_path_effects([withStroke(linewidth=SPARSE_LW + 2 * CASING_PT, foreground='white')])
        for c, ax in enumerate(axs):
            ax.set_xlim(ext[:2]); ax.set_ylim(ext[2:])
            ax.set_xticks([-np.pi, 0, np.pi]); ax.set_yticks([-2 * np.pi, 0, 2 * np.pi])
            ax.tick_params(labelbottom=(c == 0), labelleft=(c == 0))
        # End tick labels align inward (the first from its tick toward the panel centre, the last back from it), so
        # the two corner labels never meet and no label reaches past the panel into the column gap.
        for t, ha in zip(axs[0].set_xticklabels([r'$-\pi$', '0', r'$\pi$']), ('left', 'center', 'right')):
            t.set_ha(ha)
        # y tick labels upright along the axis: one text height wide instead of the width of "-2 pi"
        for t, va in zip(axs[0].set_yticklabels([r'$-2\pi$', '0', r'$2\pi$'], rotation=90),
                         ('bottom', 'center', 'top')):
            t.set_va(va)
        axs[0].set_xlabel(r'$\theta$', labelpad=1)
        axs[0].set_ylabel(r'$\dot{\theta}$', labelpad=1)
        titles = [[fig.text(0, 0, t, ha='center', va='baseline', fontsize=FS['title'])] for t in R_TITLES]
        caxs = []
        for key in ('I', 'D'):
            cax = fig.add_axes([.5, .05, .2, .02])
            cb = fig.colorbar(ims[key], cax=cax, orientation='horizontal', ticks=[0, .5, 1])
            for t, ha in zip(cb.ax.set_xticklabels(['0', '0.5', '1']), ('left', 'center', 'right')):
                t.set_ha(ha)                                         # end labels inward, as on the panels
            cb.outline.set_linewidth(.4)
            cax.tick_params(labelsize=FS['tick'], length=1.5, pad=1)
            caxs.append(cax)
        handles = [Patch(facecolor=ABSTAIN_ON_WHITE, edgecolor='none', label='abstention'),
                   Line2D([], [], color=FILL['boundary'], lw=BOUNDARY_LW, label='success/failure boundary')]
        if SHOW_SPARSE:
            handles.append(Line2D([], [], color=FILL['boundary'], lw=SPARSE_LW, ls='--', label='sparse/dense boundary'))
        for rows_ in ([handles], [handles[:2], handles[2:]] if len(handles) > 2 else [handles[:1], handles[1:]]):
            legends = [fig.legend(handles=h_, ncol=len(h_), **paper_legend_kw(FS['legend'])) for h_ in rows_]
            fig.canvas.draw()
            rr_ = fig.canvas.get_renderer()
            if max(lg.get_window_extent(rr_).width for lg in legends) / fig.dpi <= W - 2 * MARGIN:
                break
            for lg in legends:
                lg.remove()
        else:
            need(False, 'no legend layout fits the print width')
        LEG_ROW_GAP = 2.0 * PT

        # Pass 1: measure.
        fig.canvas.draw()
        rr = fig.canvas.get_renderer()
        inch = lambda bb: (bb.width / fig.dpi, bb.height / fig.dpi)
        pos = lambda ax: ax.get_position()
        tb = lambda ax: ax.get_tightbbox(rr)
        left_dec = pos(axs[0]).x0 * W - tb(axs[0]).x0 / fig.dpi
        below_a = pos(axs[0]).y0 * H0 - tb(axs[0]).y0 / fig.dpi                  # A's x tick labels and x label
        below_bc = max(pos(ax).y0 * H0 - tb(ax).y0 / fig.dpi for ax in axs[1:])  # tick marks of B and C
        cb_below = max(pos(c).y0 * H0 - tb(c).y0 / fig.dpi for c in caxs)       # colourbar tick labels
        above = max(max(0.0, tb(ax).y1 / fig.dpi - (pos(ax).y0 + pos(ax).height) * H0) for ax in axs)
        right_over = max(0.0, tb(axs[2]).x1 / fig.dpi - (pos(axs[2]).x0 + pos(axs[2]).width) * W)
        gap = GAP_PT * PT
        leg_hs = [inch(lg.get_window_extent(rr))[1] for lg in legends]
        leg_h = sum(leg_hs) + LEG_ROW_GAP * (len(legends) - 1)
        x_a = MARGIN + ACCENT_PT * PT + left_dec
        pw = (W - MARGIN - right_over - x_a - 2 * gap) / 3
        ph = pw * ASPECT
        tw = [inch(t[0].get_window_extent(rr))[0] for t in titles]
        # one line per title if every title stays within its panel plus half of each neighbouring gap
        wrapped = any(w_ > pw + gap - SEP_PT * PT for w_ in tw)
        if wrapped:
            for c, s_ in enumerate(R_TITLES_WRAPPED):
                lines = s_.split('\n')
                titles[c][0].set_text(lines[0])
                titles[c] += [fig.text(0, 0, l_, ha='center', va='baseline', fontsize=FS['title']) for l_ in lines[1:]]
            fig.canvas.draw()
            tw = [max(inch(t.get_window_extent(rr))[0] for t in lines) for lines in titles]
            need(all(w_ <= pw + gap - SEP_PT * PT for w_ in tw), ('a wrapped title is wider than its column', tw))

        def asc_desc(t):
            bb = t.get_window_extent(rr)
            base = fig.transFigure.transform(t.get_position())[1]
            return (bb.y1 - base) / fig.dpi, (base - bb.y0) / fig.dpi

        SKIP = 1.2 * FS['title'] * PT
        asc0 = max(asc_desc(t[0])[0] for t in titles)
        n_lines = max(len(t) for t in titles)
        desc = max(asc_desc(t[-1])[1] for t in titles)
        title_h = asc0 + (n_lines - 1) * SKIP + desc
        cb_band = CB_GAP_PT * PT + CB_H + cb_below
        bottom = max(below_a, below_bc + cb_band)
        H = MARGIN + leg_h + LEG_GAP_PT * PT + title_h + TITLE_GAP_PT * PT + above + ph + bottom + MARGIN
        fig.set_size_inches(W, H)

        # Place. Titles share a baseline grid; a one-line title sits on the last line of the wrapped ones.
        y_head = H - MARGIN - leg_h - LEG_GAP_PT * PT
        top = y_head - title_h - TITLE_GAP_PT * PT - above
        xs = [x_a + c * (pw + gap) for c in range(3)]
        for c, ax in enumerate(axs):
            ax.set_position([xs[c] / W, (top - ph) / H, pw / W, ph / H])
        for c, lines in enumerate(titles):
            first = n_lines - len(lines)
            for k_, t in enumerate(lines):
                t.set_position(((xs[c] + pw / 2) / W, (y_head - asc0 - (first + k_) * SKIP) / H))
        cb_top = top - ph - below_bc - CB_GAP_PT * PT
        for c, cax in zip((1, 2), caxs):
            cax.set_position([xs[c] / W, (cb_top - CB_H) / H, pw / W, CB_H / H])
        for k_, lg in enumerate(legends):
            lg.set_bbox_to_anchor((0.5, (H - MARGIN - sum(leg_hs[:k_]) - k_ * LEG_ROW_GAP) / H),
                                  transform=fig.transFigure)

        # Checks.
        fig.canvas.draw()
        rr = fig.canvas.get_renderer()

        def drawn(axis, lim):
            lo, hi = sorted(lim)
            return [t for t, v in zip(axis.get_ticklabels(), axis.get_majorticklocs())
                    if t.get_text() and t.get_visible() and lo - 1e-9 <= v <= hi + 1e-9]

        free = [(t.get_text(), t.get_window_extent(rr)) for t in fig.texts] + \
            [(f'legend row {k_}', lg.get_window_extent(rr)) for k_, lg in enumerate(legends)]
        owned = []
        for ax in axs + caxs:
            owned += [(t.get_text(), t.get_window_extent(rr)) for t in drawn(ax.xaxis, ax.get_xlim()) +
                      drawn(ax.yaxis, ax.get_ylim())]
            owned += [(lab.get_text(), lab.get_window_extent(rr)) for lab in (ax.xaxis.label, ax.yaxis.label)
                      if lab.get_text()]
        boxes = free + owned
        need(sum(len(drawn(ax.xaxis, ax.get_xlim())) for ax in axs) == 3 and
             sum(len(drawn(ax.yaxis, ax.get_ylim())) for ax in axs) == 3 and
             sum(len(drawn(c.xaxis, c.get_xlim())) for c in caxs) == 6, 'expected 3 x, 3 y and 6 colourbar labels')
        for i, (ta, a) in enumerate(boxes):
            for tbx, b in boxes[i + 1:]:
                need(not a.overlaps(b), ('text overlap', ta, tbx))
            need(a.x0 >= -0.5 and a.x1 <= fig.bbox.x1 + 0.5 and a.y0 >= -0.5 and a.y1 <= fig.bbox.y1 + 0.5,
                 ('text off the page', ta))
        panels = [ax.get_window_extent(rr) for ax in axs + caxs]
        for ta, a in boxes:
            need(not any(a.overlaps(p) for p in panels), ('text over a panel', ta))
        for ta, a in free:
            for ax in axs + caxs:
                need(not a.overlaps(ax.get_tightbbox(rr)), ('text over a panel\'s decorations', ta))
        need(PDF_DPI * pw >= GRID_SHAPE[0] and PDF_DPI * ph >= GRID_SHAPE[1], 'a cell is lost in the PDF raster')
        for c, ax in enumerate(axs):
            im = ax.get_images()
            need(len(im) == 1, f'{"ABC"[c]}: {len(im)} images')
            need(np.allclose(im[0].get_extent(), ext) and np.allclose(list(ax.get_xlim()) + list(ax.get_ylim()), ext),
                 f'{"ABC"[c]}: image and axes limits differ (clipped cells)')
            arr = im[0].get_array()
            need(not np.ma.getmaskarray(arr).any() and np.isfinite(np.ma.getdata(arr)).all(), f'{"ABC"[c]}: missing cells')
            if c == 0:
                need(np.array_equal(np.ma.getdata(arr), _abst_rgba(g['abst']).transpose(1, 0, 2)), 'A: abstentions differ')
            else:
                key = 'ID'[c - 1]
                need(np.array_equal(np.ma.getdata(arr), g[key].T) and (im[0].norm.vmin, im[0].norm.vmax) == (0, 1)
                     and im[0].get_cmap() is CMAP[key], f'{"BC"[c - 1]}: image is not {key} on [0, 1] in its ramp')
            for coll in ax.collections:
                for p in coll.get_paths():
                    v = p.vertices
                    need(not len(v) or ((v[:, 0] >= ext[0]) & (v[:, 0] <= ext[1]) & (v[:, 1] >= ext[2]) &
                                        (v[:, 1] <= ext[3])).all(), f'{"ABC"[c]}: contour off the panel')
        ink = [b for _, b in boxes] + [ax.get_tightbbox(rr) for ax in axs + caxs]
        edges = dict(left=min(b.x0 for b in ink) / fig.dpi - ACCENT_PT * PT, right=W - max(b.x1 for b in ink) / fig.dpi,
                     bottom=min(b.y0 for b in ink) / fig.dpi, top=H - max(b.y1 for b in ink) / fig.dpi)
        need(all(-0.005 <= v <= 0.03 for v in edges.values()), ('outer whitespace', edges))
        for lg in legends:
            lb = lg.get_window_extent(rr)
            need(abs((lb.x0 + lb.x1) / 2 / fig.dpi - W / 2) < 0.005, 'legend not centred')
    layout = dict(width=W, height=H, panel_w=pw, panel_h=ph, col_gap=gap, legend_rows=len(legends),
                  titles_wrapped=wrapped, edges=edges, cell_pt=(72 * pw / GRID_SHAPE[0], 72 * ph / GRID_SHAPE[1]))
    return fig, layout


def _pdf_size(pdf):
    m = re.search(rb'/MediaBox\s*\[\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)', pdf.read_bytes())
    x0, y0, x1, y1 = map(float, m.groups())
    return (x1 - x0) / 72, (y1 - y0) / 72


def sidecar(g, L):
    i, f = g['info'], g['info']['full']
    th = i['thresholds']
    out = [f'pendulum_uncertainty_signals: TERA-FM (U_H, rho 0.75), Pendulum LQR, {i["run"]}, seed 99512, round '
           f'{i["round"]} ({i["n_train"]} training trajectories, {i["n_train_states"]} states), full 158 x 315 grid.',
           f'Size {L["width"]:.3f} x {L["height"]:.3f} in; panels {L["panel_w"]:.3f} x {L["panel_h"]:.3f} in '
           f'(aspect {ASPECT:.4f}); column gap {L["col_gap"]:.3f} in; cell {L["cell_pt"][0]:.2f} x '
           f'{L["cell_pt"][1]:.2f} pt; outer whitespace (in) ' + ', '.join(f'{k} {v:.3f}' for k, v in L['edges'].items())
           + f'. Fonts (pt): {FS}. Legend in {L["legend_rows"]} row(s); titles '
           f'{"wrapped" if L["titles_wrapped"] else "on one line"}.',
           f'Columns: A abstention as a whole ({ABSTAIN_ON_WHITE} on white, true cell size), B invalid mass I(x) on '
           f'[0, 1] in {SIGNAL_RAMP["I"][-1]}-ended Reds, C weighted disagreement D(x) on [0, 1] in '
           f'{SIGNAL_RAMP["D"][-1]}-ended BrBG teal. True boundary black solid {BOUNDARY_LW} pt'
           f'{" with a white casing" if BOUNDARY_CASED else ", no casing"} in panels '
           f'{", ".join("ABC"[c] for c in BOUNDARY_PANELS)}; in A the abstaining cells are drawn over it. Sparse/dense '
           f'contour (d_train 0.02) {"drawn dashed" if SHOW_SPARSE else "not drawn"}.',
           f'Certified thresholds (A-1d at delta): t_s {th["t_s"]}, t_f {th["t_f"]}, u_s {th["u_s"]}, u_f {th["u_f"]} '
           '(tau^s_s, tau^s_f, tau^f_s, tau^f_f).',
           f'Test set ({sum(i["test_counts"])} states): Success {i["test_counts"][0]}, Fail {i["test_counts"][1]}, '
           f'Abstain {i["test_counts"][2]} ({100 * i["test_counts"][2] / sum(i["test_counts"]):.2f}%); wrong '
           f'predictions {i["test_wrong"]} (FP {i["test_fp"]}, FN {i["test_fn"]}).',
           f'Full grid ({sum(i["source_counts"].values())} states; sources {i["source_counts"]}): Success {f["success"]}, '
           f'Fail {f["fail"]}, Abstain {f["abstain"]} ({100 * f["abstain"] / sum(i["source_counts"].values()):.2f}%; '
           f'invalid {f["invalid"]}, weak {f["weak"]}, outcome-uncertain {f["outcome_uncertain"]}, other {f["other"]}; '
           f'{f["abstain_isolated"]} abstaining cells with no abstaining 4-neighbour); wrong predictions {f["wrong"]} '
           f'(FP {f["fp"]}, FN {f["fn"]}) of {f["labelled"]} labelled; timeouts {f["timeouts"]}; sparse {f["sparse"]}.',
           'Drawing: one image cell per grid state with exact edges (0.04 rad lattice, edges at +-0.02), nearest '
           f'interpolation, PDF raster {PDF_DPI} dpi. True boundary: full-grid labels contoured at 0.5 with the timeouts '
           'masked. d_train: distance to the nearest of the run\'s own training states (uncertainty_signals_data.norm, '
           'periodic tree); sparse = d_train > 0.02. Test cells reproduce uncertainty_signals_data.analyse exactly.',
           'Ink margins on the 300 dpi page (in): ' + ', '.join(f'{k} {v:.3f}' for k, v in L.get('ink_margins', {}).items())]
    return '\n'.join(out) + '\n'


def main_restyle(rnd):
    g = load_full_grid(rnd)
    fig, L = restyle_figure(g)
    paper_version = rnd == PAPER_ROUND
    out = PLOTS if paper_version else PLOTS / 'variants'
    out.mkdir(parents=True, exist_ok=True)
    stem = R_STEM if paper_version else f'{R_STEM}_round{rnd}'
    pdf, png, txt = out / f'{stem}.pdf', out / f'{stem}.png', out / f'{stem}.txt'
    with plt.rc_context(style_rc()):
        fig.savefig(pdf, dpi=PDF_DPI, metadata={'CreationDate': None})
        fig.savefig(png, dpi=PNG_DPI)
    plt.close(fig)
    w, h = _pdf_size(pdf)
    need(abs(w - PRINT_WIDTH[R_STEM]) <= .01, f'PDF width {w:.3f} in, print width {PRINT_WIDTH[R_STEM]}')
    # Ink margins on the rendered page: every side at least half a pixel (ink touching an edge may be cut off) and at
    # most 0.045 in (text boxes carry ascender / descender room without ink); the left side at most 0.03 in, since
    # without its accent the theta-dot label's ink starts 0.038 in in.
    from PIL import Image
    a = np.asarray(Image.open(png).convert('L')) < 200
    rows, cols = np.where(a.any(1))[0], np.where(a.any(0))[0]
    ink = dict(left=cols[0], right=a.shape[1] - 1 - cols[-1], top=rows[0], bottom=a.shape[0] - 1 - rows[-1])
    ink = {k: v / PNG_DPI for k, v in ink.items()}
    need(all(.5 / PNG_DPI <= v <= .045 for v in ink.values()) and ink['left'] <= .03, ('ink margins on the page', ink))
    L['ink_margins'] = ink
    txt.write_text(sidecar(g, L))
    f = g['info']['full']
    print(f'{stem}: {w:.3f} x {h:.3f} in, panels {L["panel_w"]:.3f} x {L["panel_h"]:.3f} in -> {pdf}')
    print(f'  round {rnd}: full grid Abstain {f["abstain"]}, wrong {f["wrong"]}; test {g["info"]["test_counts"]}, '
          f'wrong {g["info"]["test_wrong"]}')
    return pdf


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument('--paper', action='store_true',
                   help='also copy plots/pendulum_uncertainty_signals.pdf (the paper version) into the paper clone')
    p.add_argument('--baseline', action='store_true',
                   help='draw the 2026-11-12 baseline instead (plots/legacy/; rewrites values/pendulum_uncertainty_*)')
    p.add_argument('--restyle', action='store_true', help='accepted for old commands; the default is the paper figure')
    p.add_argument('--round', type=int, default=PAPER_ROUND,
                   help=f'training round to draw (paper: {PAPER_ROUND}); other rounds go to plots/variants/ and need '
                        'data/pendulum_full_grid/<run>_epoch<round>_grid.npz')
    args = p.parse_args()
    if args.baseline:
        if args.restyle or args.round != PAPER_ROUND:
            p.error('--baseline draws the old design only')
        main(False)
    else:
        main_restyle(args.round)
        if args.paper:
            if args.round != PAPER_ROUND:
                p.error(f'only round {PAPER_ROUND} goes to the paper')
            install_paper(R_STEM)
