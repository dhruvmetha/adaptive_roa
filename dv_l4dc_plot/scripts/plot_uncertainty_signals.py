"""Two uncertainty signals and the abstentions in pendulum state space: paper figure pendulum_uncertainty_signals
(Figure 5).

The question: where do the two uncertainty signals appear, and where does the model abstain?
TERA-FM on Pendulum LQR, seed 99512, round 10 (the run behind figures/abstention_triangle). Rows: adaptive (U_H,
rho=0.75) above uniform acquisition, each with its own saved A-1d thresholds. Columns:
  A  invalid-terminal mass I(x) = p_inv(x)
  B  weighted disagreement D(x) = (1 - p_inv) H2(p_s / (p_s + p_f)) in bits, D = 0 with no valid endpoint
  C  abstentions: one #333333 point (2.2 pt) at the centre of every abstaining state, on a white background
A and B share one fixed [0, 1] Purples scale and one colorbar ("Signal value"); neither is normalized on its own.
Every panel overlays the true success/failure boundary, 0.7 pt black: the labels contoured at 0.5 on a
triangulation of the drawn states. There is no Success / Fail fill, no incorrect-prediction mark and no hatching
(the 3-column version with a decision map is archive/plot_uncertainty_signals_with_decisions.py).

Drawn states: the run's test states (20,702) and its calibration states (1,000), 21,702 per row. Calibration
states go through the test states' path (calibration()): C.split_probs on their counts, p_inv tied with R.tie, I
and D by the same formulas, S and F from C.in_S / C.in_F with the row's thresholds. Abstain = ~(succ | fail) with
succ = S & ~F and fail = F & ~S, the certified output itself, for test and calibration alike.
A and B fill the 158 x 315 lattice of unique theta and omega values (spacing 0.04 rad; the axes run edge to edge,
+-pi and +-2pi). Every lattice cell shows its nearest evaluated state (test or calibration): nearest in (theta,
omega) rad, which is lattice-index distance, ties going to the smaller theta, then the smaller omega. 21,702 of the
49,770 cells (43.6%) are evaluated states themselves, and each shows its own value. Each map is one image with
exact cell edges and nearest-neighbour resampling, so there are no white cells, no seams and no interpolation
between states.

The figure is laid out at its printed width (6.0 in). draw() raises if the PDF leaves 5.95-6.05 x 3.55-3.85 in, if
an abstention count differs from the expected one (test 156 / 249, calibration 4 / 16) or from the statistics, or
if anything read back from the drawn artists is off: an A or B image off the shared [0, 1] Purples scale, with a
white or unassigned cell, or with a cell that does not show its nearest state's value (so every evaluated state's
own cell shows its own I and D); or C's points not exactly at the abstaining states' centres.
All data come from uncertainty_signals_data.py.

Also writes values/pendulum_uncertainty_signals_stats.csv (every arm, seed, round 2 and 10, all categories) and
values/pendulum_uncertainty_signals_values.json (stats, the comparison with sections/results_uncertainty.tex,
threshold sums, provenance). Both statistics are test states only; only the JSON's "figure" entry counts the
calibration states, per set. --paper copies the PDF into the paper clone as
figures/results/pendulum_uncertainty_signals.pdf. The figure and statistics describe aggregate regions; they do not
establish correctness at any individual state.
"""
import csv, json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from common import VALUES, RC, parse_args, save
import uncertainty_signals_data as usd
from uncertainty_signals_data import build, SEEDS

SEED, ROUND = 99512, 10
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


# Drawing at the printed size. Every length below is in inches on the 6.0 in page, every font size and line width in
# points at 100% print scale; save() trims the canvas to the tight bbox plus 0.04 in.
N_TEST, N_CAL = 20702, 1000
LATTICE = (158, 315)
EXPECT_ABSTAIN = {'adaptive': (156, 4), 'uniform': (249, 16)}       # abstaining states: (test, calibration)
ROWS = [('adaptive', 'Adaptive'), ('uniform', 'Uniform')]
COLS = [('I', 'A', 'Invalid-terminal mass'), ('D', 'B', 'Weighted disagreement'), (None, 'C', 'Abstentions')]
PAGE_W, PAGE_H = (5.95, 6.05), (3.55, 3.85)
CMAP, VMIN, VMAX = 'Purples', 0.0, 1.0                  # one fixed signal scale for A and B
ABST_PT = dict(marker='o', ms=2.2, mfc='#333333', mec='none', mew=0)  # C: one dark point per abstaining state
LW = .7            # true-outcome boundary line width
SEARCH = 8         # nearest-state search radius in cells; ownership() checks that every cell finds a state within it
FIG_W, FIG_H = 6.0, 3.70
RASTER_DPI = 400   # resolution of the cell images inside the PDF
LEFT, GAP, VGAP, BOTTOM = .58, .07, .07, .30            # column A's left edge, gaps between panels, lower row's bottom
PANEL_W = (FIG_W - .045 - LEFT - 2 * GAP) / 3
BAND = .26                                              # colorbar and legend band at the top
TITLE = .20                                             # column titles
PANEL_H = (FIG_H - .04 - BAND - TITLE - BOTTOM - VGAP) / 2


def check(ok, msg):
    """The figure's built-in checks: raise, never warn, so a wrong figure cannot be saved silently."""
    if not ok:
        raise RuntimeError(f'pendulum_uncertainty_signals check failed: {msg}')


def row_stats(stats, arm):
    return next(s for s in stats if (s['arm'], s['seed'], s['round']) == (arm, SEED, ROUND))


def calibration(data):
    """The drawn run's 1,000 calibration states per row (prediction_store_states_cal.npz and
    epoch_010/predictions_cal.npz, counts out of K = 100), through the test states' path: C.split_probs,
    p_inv = R.tie(n_invalid / K), I and D as in uncertainty_signals_data.analyse, and Success / Fail from C.in_S /
    C.in_F with the row's own thresholds. The data module's computations are reused, never changed."""
    run_of = usd.runs()
    out = {}
    for arm, _ in ROWS:
        run, st = run_of[arm, SEED], row_stats(data['stats'], arm)
        check(run['name'] == st['run'], f'{arm}: calibration run {run["name"]} is not the drawn run {st["run"]}')
        cs = np.load(run['dir'] / 'prediction_store_states_cal.npz')
        z = np.load(run['dir'] / f'epoch_{ROUND:03d}/predictions_cal.npz')
        X = cs['states'].astype(float)
        n = np.stack([z['n_success'], z['n_failure'], z['n_invalid']]).astype(float)
        K = float(z['num_mc_samples'])
        check(X.shape == (N_CAL, 2) and n.shape == (3, N_CAL), f'{arm}: calibration arrays are not {N_CAL} states')
        check((n.sum(0) == K).all(), f'{arm}: calibration counts do not sum to K = {K:g}')
        check(np.array_equal(z['true_labels'], cs['true_labels']), f'{arm}: calibration labels differ between files')
        ps, pf, y = usd.C.split_probs(z)
        check(np.allclose(np.round(n[2] / K, 9), np.round(1 - ps - pf, 9), atol=1e-6), f'{arm}: p_inv != 1 - p_s - p_f')
        pinv = usd.R.tie(n[2] / K)
        q = np.where(ps + pf > 0, ps / np.where(ps + pf > 0, ps + pf, 1), 0.5)
        th = st['thresholds']
        S, F = usd.C.in_S(ps, pf, th['t_s'], th['t_f']), usd.C.in_F(ps, pf, th['u_s'], th['u_f'])
        out[arm] = dict(X=X, y=y, I=pinv, D=np.where(ps + pf > 0, (1 - pinv) * usd.h2(q), 0.0), succ=S & ~F,
                        fail=F & ~S)
    return out


def lattice(X):
    """The lattice from the unique theta and omega values of the drawn states: every state's (i, j) index, the
    spacings, the first theta and omega, and the shape (158 x 315)."""
    th, om = np.unique(X[:, 0]), np.unique(X[:, 1])
    dth, dom = np.median(np.diff(th)), np.median(np.diff(om))
    check(np.allclose(np.diff(th), dth, atol=1e-5) and np.allclose(np.diff(om), dom, atol=1e-5),
          'drawn states are not on a regular lattice')
    check((len(th), len(om)) == LATTICE, f'lattice is {len(th)} x {len(om)}, expected {LATTICE}')
    ij = np.c_[np.rint((X[:, 0] - th[0]) / dth), np.rint((X[:, 1] - om[0]) / dom)].astype(int)
    check(len(np.unique(ij, axis=0)) == len(X), 'two drawn states share a lattice cell')
    return dict(ij=ij, d=(dth, dom), origin=(th[0], om[0]), shape=(len(th), len(om)))


def ownership(X):
    """The nearest drawn state (test or calibration) of every lattice cell, in (theta, omega) rad; both spacings are
    0.04, so this is lattice-index distance. Ties go to the smaller theta, then the smaller omega. Each drawn state
    owns its own cell. Returns the lattice, the owner grid and the number of cells with a tie."""
    L = lattice(X)
    (ni, nj), r = L['shape'], SEARCH
    idx = np.full((ni + 2 * r, nj + 2 * r), -1)                      # padded state-index grid
    idx[L['ij'][:, 0] + r, L['ij'][:, 1] + r] = np.arange(len(X))
    shifted = lambda di, dj: idx[r + di:r + di + ni, r + dj:r + dj + nj]
    offs = sorted((di * di + dj * dj, di, dj) for di in range(-r, r + 1) for dj in range(-r, r + 1)
                  if di * di + dj * dj <= r * r)
    owner, best, ties = np.full((ni, nj), -1), np.full((ni, nj), np.inf), np.zeros((ni, nj), bool)
    for d2, di, dj in offs:                                          # ascending d^2, then theta, then omega
        s = shifted(di, dj)
        hit = s >= 0
        new = hit & (owner < 0)
        owner[new], best[new] = s[new], d2
        ties |= hit & ~new & (best == d2)
    check((owner >= 0).all(), f'{int((owner < 0).sum())} lattice cells have no drawn state within {r} cells')
    check(np.array_equal(owner[L['ij'][:, 0], L['ij'][:, 1]], np.arange(len(X))), 'a drawn state does not own its cell')
    return L, owner, int(ties.sum())


def cell_image(ax, L, grid):
    """One image cell per lattice cell, exact edges, nearest-neighbour resampling: no interpolation, no seams."""
    (dth, dom), (t0, o0), (ni, nj) = L['d'], L['origin'], L['shape']
    ext = (t0 - dth / 2, t0 + (ni - .5) * dth, o0 - dom / 2, o0 + (nj - .5) * dom)
    return ax.imshow(grid.T, origin='lower', extent=ext, interpolation='nearest', aspect='auto', cmap=CMAP,
                     vmin=VMIN, vmax=VMAX, zorder=1)


def check_maps(axs, L, owner, u, arm):
    """Read the cell images back from A and B: one image per map covering the whole axes, on the one fixed
    [VMIN, VMAX] CMAP scale, every lattice cell with a value (0 white cells), every cell showing its owner's I or D,
    so every drawn state's own cell shows its own value."""
    (ni, nj), ij = L['shape'], L['ij']
    for ax, (key, letter, _) in zip(axs, COLS[:2]):
        ims = ax.get_images()
        check(len(ims) == 1, f'{arm} {letter}: {len(ims)} images, expected 1')
        im = ims[0]
        check((im.norm.vmin, im.norm.vmax, im.get_cmap().name) == (VMIN, VMAX, CMAP),
              f'{arm} {letter}: scale is {im.norm.vmin}-{im.norm.vmax} {im.get_cmap().name}, expected the shared one')
        x0, x1, y0, y1 = im.get_extent()
        check(x0 <= -np.pi and x1 >= np.pi and y0 <= -2 * np.pi and y1 >= 2 * np.pi,
              f'{arm} {letter}: the cell image does not cover the axes')
        arr = im.get_array()
        check(arr.shape == (nj, ni) and not np.ma.getmaskarray(arr).any() and np.isfinite(np.ma.getdata(arr)).all(),
              f'{arm} {letter}: the cell image has missing cells')
        white = int((im.to_rgba(arr, bytes=True)[..., :3] == 255).all(-1).sum())
        check(white == 0, f'{arm} {letter}: {white} white cells')
        check(np.array_equal(np.ma.getdata(arr), u[key][owner].T), f'{arm} {letter}: a cell does not show its owner')
        check(np.array_equal(np.ma.getdata(arr)[ij[:, 1], ij[:, 0]], u[key]),
              f'{arm} {letter}: a drawn state\'s own cell does not show its own {key}')
    return dict(cells=ni * nj, white=0)


def check_abstentions(axs, Xab, arm):
    """Read the abstention points back: one point set in C at exactly the abstaining states' centres (test then
    calibration), and none in A or B; C holds no image."""
    for ci, ax in enumerate(axs):
        lines = [l for l in ax.get_lines() if l.get_gid() == 'abstention']
        check(len(lines) == (ci == 2), f'{arm} panel {"ABC"[ci]}: {len(lines)} abstention point sets')
    check(not axs[2].get_images(), f'{arm} C: column C holds an image')
    xy = [l for l in axs[2].get_lines() if l.get_gid() == 'abstention'][0].get_xydata()
    check(np.array_equal(xy, Xab), f'{arm} C: the points are not exactly the abstaining states\' centres')
    return len(xy)


def axes_at(fig, x, y, w, h):
    return fig.add_axes([x / FIG_W, y / FIG_H, w / FIG_W, h / FIG_H])


def style_ticks(ax, bottom, left):
    """Ticks on every panel, labels only on the outer ones; the edge labels sit inside the panel's span so they never
    collide with the neighbouring panel's labels across the narrow gaps."""
    ax.set_xlim(-np.pi, np.pi); ax.set_ylim(-2 * np.pi, 2 * np.pi)
    ax.set_xticks([-np.pi, 0, np.pi]); ax.set_yticks([-2 * np.pi, 0, 2 * np.pi])
    ax.tick_params(length=2.5, width=.6, pad=1.5, labelsize=7.5, labelbottom=bottom, labelleft=left)
    if bottom:
        lab = ax.set_xticklabels([r'$-\pi$', '0', r'$\pi$'])
        lab[0].set_ha('left'); lab[-1].set_ha('right')
        ax.set_xlabel(r'$\theta$', fontsize=8, labelpad=0)
    if left:
        lab = ax.set_yticklabels([r'$-2\pi$', '0', r'$2\pi$'])
        lab[0].set_va('bottom'); lab[-1].set_va('top')
        ax.set_ylabel(r'$\dot{\theta}$', fontsize=8, rotation=0, ha='right', va='center', labelpad=2)


def set_counts(a):
    succ, fail = a['succ'], a['fail']
    abst = ~(succ | fail)
    check(((succ.astype(int) + fail + abst) == 1).all(), 'Success, Fail, Abstain do not partition the states')
    return dict(states=len(succ), success=int(succ.sum()), fail=int(fail.sum()), abstain=int(abst.sum()))


def draw(arrays, stats, paper, cal):
    """Two rows (adaptive, uniform) by three panels: the I and D maps, each lattice cell filled with its nearest
    drawn state's value, and the abstaining states on white."""
    import matplotlib.tri as mtri
    counts = {}
    with plt.rc_context(RC):
        # dpi sets the resolution of the cell images inside the PDF (the PNG keeps save()'s 220 dpi)
        fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=RASTER_DPI)
        axs = [[axes_at(fig, LEFT + ci * (PANEL_W + GAP), BOTTOM + (1 - ri) * (PANEL_H + VGAP), PANEL_W, PANEL_H)
                for ci in range(len(COLS))] for ri in range(len(ROWS))]
        for ri, (arm, name) in enumerate(ROWS):
            a, st, c = arrays[arm, SEED, ROUND], row_stats(stats, arm), cal[arm]
            ct, cc = set_counts(a), set_counts(c)
            # abstain = ~(succ | fail), the certified output: test from the data module, calibration from calibration()
            ab_t = ~(a['succ'] | a['fail'])
            check((ct['abstain'], cc['abstain']) == EXPECT_ABSTAIN[arm],
                  f"{arm}: abstaining states (test, cal) = {(ct['abstain'], cc['abstain'])}, expected {EXPECT_ABSTAIN[arm]}")
            check(ct['abstain'] == st['abstain']['count'], f"{arm}: test abstentions differ from the statistics")
            check(np.array_equal(ab_t, a['inv'] | a['weak'] | a['ou'] | a['other']),
                  f'{arm}: test abstentions differ from inv | weak | ou | other')
            u = {k: np.concatenate([a[k], c[k]]) for k in ('X', 'y', 'I', 'D', 'succ', 'fail')}
            X, abst = u['X'], ~(u['succ'] | u['fail'])
            check(len(X) == N_TEST + N_CAL, f'{arm}: {len(X)} drawn states, expected {N_TEST + N_CAL}')
            check(int(abst.sum()) == ct['abstain'] + cc['abstain'], f'{arm}: union abstentions != test + calibration')
            L, owner, n_ties = ownership(X)
            tri = mtri.Triangulation(X[:, 0], X[:, 1])
            for ci, (key, _, _) in enumerate(COLS):
                ax = axs[ri][ci]
                if key is not None:
                    check(u[key].min() >= VMIN and u[key].max() <= VMAX, f'{arm} {key} outside the [0, 1] scale')
                    m = cell_image(ax, L, u[key][owner])
                else:                                           # C: white, the boundary and the abstentions only
                    ax.set_facecolor('white')
                    ax.plot(X[abst, 0], X[abst, 1], ls='none', zorder=5, gid='abstention', **ABST_PT)
                ax.tricontour(tri, u['y'].astype(float), levels=[.5], colors='k', linewidths=LW, zorder=4)
                style_ticks(ax, bottom=ri == 1, left=ci == 0)
            mc = check_maps(axs[ri], L, owner, u, arm)
            n_ab = check_abstentions(axs[ri], X[abst], arm)
            counts[arm] = dict(test=ct, calibration=cc,
                               union=dict(states=len(X), lattice_cells=mc['cells'], own_cells=len(X),
                                          share_own=round(len(X) / mc['cells'], 4), white_cells=mc['white'],
                                          ownership_ties=n_ties, abstain_points=n_ab))
            print(f"{arm}: test {ct['states']} = success {ct['success']} + fail {ct['fail']} + abstain {ct['abstain']}; "
                  f"calibration {cc['states']} = success {cc['success']} + fail {cc['fail']} + abstain {cc['abstain']}")
            print(f"{arm}: A and B: {len(X)} drawn states fill all {mc['cells']} lattice cells ({len(X) / mc['cells']:.1%} "
                  f"are drawn states themselves; ownership ties {n_ties} cells); white cells {mc['white']}; every own "
                  f"cell shows its own I and D; both on {CMAP} [{VMIN:g}, {VMAX:g}]")
            print(f"{arm}: C: {n_ab} abstention points = test {ct['abstain']} + calibration {cc['abstain']}, at exactly "
                  f"those states' centres; none in A or B")
            y_mid = BOTTOM + (1 - ri) * (PANEL_H + VGAP) + PANEL_H / 2
            fig.text(.04 / FIG_W, y_mid / FIG_H, name, rotation=90, ha='left', va='center', fontsize=9)
        for ax, (_, letter, title) in zip(axs[0], COLS):
            ax.set_title(rf'$\mathbf{{{letter}}}$. {title}', loc='left', fontsize=9, pad=3)
        top = FIG_H - .04
        cax = axes_at(fig, LEFT + .2, top - .1, .75, .065)    # above A: the scale of A and B
        cb = fig.colorbar(m, cax=cax, orientation='horizontal', ticks=[0, .5, 1])
        check((cb.norm.vmin, cb.norm.vmax) == (VMIN, VMAX), 'the colorbar is not on the shared scale')
        cb.set_ticklabels(['0', '0.5', '1'])
        cax.annotate('Signal value', xy=(0, .5), xycoords='axes fraction', xytext=(-4, 0), textcoords='offset points',
                     ha='right', va='center', fontsize=8)
        cax.tick_params(labelsize=7.5, length=2, width=.5, pad=1.5)
        cb.outline.set_linewidth(.5)
        handles = [Line2D([], [], color='k', lw=LW, label='True outcome boundary'),
                   Line2D([], [], ls='none', label='Abstention', **ABST_PT)]
        fig.legend(handles=handles, loc='upper right', ncol=2, frameon=False,
                   bbox_to_anchor=((LEFT + 3 * PANEL_W + 2 * GAP) / FIG_W, top / FIG_H), fontsize=8, borderpad=0,
                   borderaxespad=0, columnspacing=1.5, handlelength=1.5, handleheight=.8, handletextpad=.5)
        # its own name since 2026-11-13: plots/pendulum_uncertainty_signals.pdf is the paper figure (paper-style)
        size = save(fig, 'pendulum_uncertainty_signals_wip', paper_stem=None, paper=paper)
    check(PAGE_W[0] <= size['width_in'] <= PAGE_W[1] and PAGE_H[0] <= size['height_in'] <= PAGE_H[1],
          f"PDF is {size['width_in']} x {size['height_in']} in, expected {PAGE_W} x {PAGE_H}")
    return size, counts


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
    size, counts = draw(arrays, stats, paper, calibration(data))
    flat = [flatten(s) for s in stats]
    cols = sorted({k for f in flat for k in f}, key=lambda k: (k.split('.')[0] not in ('arm', 'seed', 'round', 'run'), k))
    with open(VALUES / 'pendulum_uncertainty_signals_stats.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in sorted(flat, key=lambda r: (r['arm'], r['round'], SEEDS.index(r['seed']))):
            w.writerow({k: ('' if r.get(k) is None else r.get(k)) for k in cols})
    comp = compare(stats)
    (VALUES / 'pendulum_uncertainty_signals_values.json').write_text(json.dumps(dict(
        figure=dict(seed=SEED, round=ROUND, page_size=size, states_drawn='test + calibration',
                    panels=('rows adaptive, uniform; columns A invalid-terminal mass I, B weighted disagreement D, '
                            'C abstentions'),
                    fill='A and B: each lattice cell shows its nearest evaluated state',
                    abstentions='C: one point per abstaining state, ~(succ | fail), on white with the true boundary',
                    counts=counts),
        manuscript_comparison=comp, threshold_sums=data['threshold_sums'],
        max_t_s_plus_u_f=max(x['t_s_plus_u_f'] for x in data['threshold_sums']),
        pool_sha256=data['pool_sha256'], provenance=data['provenance'], stats=stats), indent=1, default=str) + '\n')
    for c in comp:
        v = c['manuscript_variant']
        print(f"{'OK ' if c['matches'] else 'DIFF'} {c['claim']}: manuscript {c['manuscript']}, recomputed "
              f"{'n/a' if c['recomputed'] is None else round(c['recomputed'], 4)} ({c['seeds']} seeds)"
              + (f" | {v['definition']} {round(v['value'], 4)} {'matches' if v['matches'] else 'differs'}" if v else ''))


if __name__ == '__main__':
    main(parse_args(__doc__).paper)
