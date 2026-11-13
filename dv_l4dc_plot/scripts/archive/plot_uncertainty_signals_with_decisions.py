"""ARCHIVED 2026-11-12: column C (certified decisions) cut by Dhruv; Figure 5 is the 2 x 2 A+B
(../plot_uncertainty_signals.py). This is the last 3-column version: A, B and C (Success / Fail / Abstain with
incorrect-prediction crosses), test + calibration states drawn as scatter markers, with the column-C variant
switches. It has no --paper option and writes only to plots/archive and values/archive, so a rerun cannot
reach the paper clone. It was archived without being run.

Two uncertainty signals in pendulum state space: formerly paper figure pendulum_uncertainty_signals.

TERA-FM on Pendulum LQR, seed 99512, round 10 (the run behind figures/abstention_triangle). Rows: adaptive (U_H,
rho=0.75) on top, uniform acquisition below, each with its OWN saved A-1d thresholds (not printed on the figure).
Every map shows the run's test states (20,702) and its calibration states (1,000) together: 21,702 states per row,
each one square marker where it was evaluated on the shared 158 x 315 lattice (no state shares a cell). Nothing is
interpolated, and white means no drawn state there. Calibration states go through the test states' decision path
(see calibration()). Columns:
  A  invalid-terminal mass I(x) = p_inv(x)
  B  weighted disagreement D(x) = (1 - p_inv) H2(p_s / (p_s + p_f)) in bits, D = 0 with no valid endpoint
  C  certified decisions: Success, Fail and Abstain, with incorrect Success/Fail predictions as dark-red crosses on
     top. Abstain is the certified output itself, ~(succ | fail) with succ = S & ~F and fail = F & ~S: a state in
     neither or both tentative sets. It is not built from the abstention kinds; draw() checks, on the test states,
     that it equals inv | weak | ou | other. Crosses on calibration states are errors on the data the thresholds
     were fitted to, not held-out errors.
     Defaults (switches below): Success BLUE, Fail ORANGE, Abstain #9b9b9b at the same marker size (true size);
     draw order Success and Fail, Abstain, the boundary line (0.7 pt black), the crosses.
A and B share one fixed [0, 1] Purples scale and one colorbar ("Signal value"); no panel is normalized on its own.
All six panels overlay the true success/failure boundary (the union's labels contoured at 0.5). Only A and B hatch
sparse training coverage, in #666666 '///' lines with no fill. A state is sparse when its nearest training state is
farther than 0.02 (d_train, in (theta/2pi wrapped, omega/4pi), against that row's own training states). The hatch is
built on the 158 x 315 lattice of unique theta and omega values. A lattice cell is painted when its centre lies inside
at least one drawn state's marker square (at this panel size, within 1 cell in theta and 2 in omega of a state). Each
painted cell belongs to its nearest drawn state in (theta, omega), ties going to the smaller theta, then the smaller
omega. The hatch is the union of the painted cells that sparse states own, merged into rectangles in one compound
path. So every sparse state's own cell is hatched, no dense state's own cell is, and no unpainted (white) cell is.
There is no contour and no contourf, and C has no hatching.
The figure is laid out at its printed width (6.0 in). draw() raises if the PDF leaves 5.95-6.05 x 3.55-3.85 in, if
a test-set count differs from the expected one, if test, calibration and union counts do not partition, if the
hatched cells read back from the drawn artists break any of the hatch rules above, or if a hatch pattern in the saved
PDF strokes another colour or width.
All data come from uncertainty_signals_data.py.

Writes plots/archive/pendulum_uncertainty_signals_with_decisions.{pdf,png},
values/archive/pendulum_uncertainty_signals_with_decisions_stats.csv and
values/archive/pendulum_uncertainty_signals_with_decisions_values.json (test-state statistics, as the main script
writes them). The figure and statistics describe aggregate regions; they do not establish correctness at any
individual state.
"""
import csv, json, sys
from pathlib import Path as FilePath
sys.path.insert(0, str(FilePath(__file__).resolve().parents[1]))     # scripts/
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, PathPatch
from matplotlib.path import Path
from matplotlib.colors import to_hex, to_rgb
from common import PLOTS, VALUES, RC, BLUE, ORANGE, save
import uncertainty_signals_data as usd
from uncertainty_signals_data import build, SEEDS, SPARSE as SPARSE_D

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
EXPECT = {'adaptive': dict(abstain=156, wrong=79, sparse=1978), 'uniform': dict(abstain=249, wrong=220, sparse=1579)}
ROWS = [('adaptive', 'Adaptive'), ('uniform', 'Uniform')]
PAGE_W, PAGE_H = (5.95, 6.05), (3.55, 3.85)
HATCH, HATCH_COLOR, HATCH_LW = '///', '#666666', .35  # sparse training coverage, columns A and B only: 18 lines/in
                                                       # (2.8 pt apart), so 0.1 in wide sparse streaks still show
WRONG, WRONG_MS, WRONG_MEW = '#a50f15', 2.6, .6     # dark red: differs from the orange Fail fill in lightness too
MS2 = 2.6          # scatter area (pt^2) per test state, about 2 lattice cells wide in theta and 5 in omega, drawn
                   # with snap=False so pixel snapping opens no gaps: white is left only where the test set has holes
# Column C switches, defaults = the round-2 figure. Variants for Dhruv to compare live in the scratchpad, not here.
ABSTAIN = '#9b9b9b'             # Abstain fill and its legend swatch
C_FILL_TINT = 0.0               # share of white mixed into C's Success/Fail fills and swatches (0: BLUE/ORANGE as is);
                                # precomputed RGB, never alpha, so the PDF has no transparency
ABSTAIN_AREA_SCALE = 1.0        # Abstain marker area relative to Success/Fail; 1.0 is true size. Enlarged gray covers
                                # the centres of committed states and puts red crosses on gray.
ABSTAIN_ABOVE_BOUNDARY = False  # draw Abstain above C's boundary line (still below the crosses)
C_BOUNDARY_LW, C_BOUNDARY_COLOR = .7, 'k'   # C's boundary line only; A and B always use LW and black
LW = .7            # true-outcome boundary line width
FIG_W, FIG_H = 6.0, 3.74
RASTER_DPI = 400   # resolution of the rasterized scatter layers inside the PDF
LEFT, GAP, VGAP, BOTTOM = .58, .07, .07, .30            # column A's left edge, gaps between panels, lower row's bottom
PANEL_W = (FIG_W - .045 - LEFT - 2 * GAP) / 3
BAND = .34                                              # legend and colorbar band at the top
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
    epoch_010/predictions_cal.npz, counts out of K = 100), through the test states' decision path: C.split_probs,
    p_inv = R.tie(n_invalid / K), I and D as in uncertainty_signals_data.analyse, S and F from C.in_S / C.in_F with
    the row's own thresholds, and d_train against the same training states with the same periodic tree. The data
    module's computations are reused, never changed."""
    pool, run_of = usd.Pool(), usd.runs()
    out = {}
    for arm, _ in ROWS:
        run, st, a = run_of[arm, SEED], row_stats(data['stats'], arm), data['arrays'][arm, SEED, ROUND]
        check(run['name'] == st['run'], f'{arm}: calibration run {run["name"]} is not the drawn run {st["run"]}')
        ids, _ = usd.verify_pool(pool, run)
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
        succ, fail = S & ~F, F & ~S
        train = usd.tree(pool.states(ids[:st['n_train']]))
        check(np.array_equal(train.query(usd.norm(a['X']))[0], a['d_train']),
              f'{arm}: the training tree does not reproduce the test states\' d_train')
        out[arm] = dict(X=X, y=y, I=pinv, D=np.where(ps + pf > 0, (1 - pinv) * usd.h2(q), 0.0), succ=succ, fail=fail,
                        wrong=(succ & ~y) | (fail & y), d_train=train.query(usd.norm(X))[0])
    return out


def set_counts(a):
    """Success, Fail, Abstain, incorrect and sparse counts of one set of states, with its partition checked."""
    succ, fail, wrong = a['succ'], a['fail'], a['wrong']
    abst = ~(succ | fail)
    check(((succ.astype(int) + fail + abst) == 1).all(), 'Success, Fail, Abstain do not partition the states')
    check(not (wrong & abst).any(), 'an incorrect prediction is not a Success or Fail prediction')
    return dict(states=len(succ), success=int(succ.sum()), fail=int(fail.sum()), abstain=int(abst.sum()),
                wrong=int(wrong.sum()), sparse=int((a['d_train'] > SPARSE_D).sum()))


def decision_masks(a, st, arm):
    """Success, Fail and the gray Abstain mask for one drawn row, checked against the counts this figure must show."""
    succ, fail, wrong = a['succ'], a['fail'], a['wrong']
    abst = ~(succ | fail)                  # the certified output: not exactly one of the tentative sets S and F
    n = len(succ)
    kinds = a['inv'] | a['weak'] | a['ou'] | a['other']
    check(n == N_TEST, f'{arm}: {n} test states, expected {N_TEST}')
    check(((succ.astype(int) + fail + abst) == 1).all(), f'{arm}: Success, Fail, Abstain do not partition the test set')
    check(int(abst.sum()) == EXPECT[arm]['abstain'], f"{arm}: {abst.sum()} abstentions, expected {EXPECT[arm]['abstain']}")
    check(int(abst.sum()) == st['abstain']['count'], f"{arm}: gray mask differs from the statistics' abstain count")
    check(np.array_equal(abst, kinds), f'{arm}: gray mask differs from inv | weak | ou | other')
    check(int(wrong.sum()) == EXPECT[arm]['wrong'], f"{arm}: {wrong.sum()} incorrect, expected {EXPECT[arm]['wrong']}")
    check(not (wrong & abst).any(), f'{arm}: an incorrect prediction is not a Success or Fail prediction')
    print(f'{arm}: {n} test states = success {succ.sum()} + fail {fail.sum()} + abstain {abst.sum()}; abstain == '
          f'inv|weak|ou|other ({kinds.sum()}); incorrect {wrong.sum()}, all within success|fail')
    return dict(success=succ, fail=fail, abstain=abst, wrong=wrong)


def lattice(X):
    """The lattice from the unique theta and omega values of the drawn states: every state's (i, j) index, the
    spacings, the first theta and omega, and the shape (158 x 315)."""
    th, om = np.unique(X[:, 0]), np.unique(X[:, 1])
    dth, dom = np.median(np.diff(th)), np.median(np.diff(om))
    check(np.allclose(np.diff(th), dth, atol=1e-5) and np.allclose(np.diff(om), dom, atol=1e-5),
          'test states are not on a regular lattice')
    ij = np.c_[np.rint((X[:, 0] - th[0]) / dth), np.rint((X[:, 1] - om[0]) / dom)].astype(int)
    check(len(np.unique(ij, axis=0)) == len(X), 'two drawn states share a lattice cell')
    check((len(th), len(om)) == (158, 315), f'lattice is {len(th)} x {len(om)}, expected 158 x 315')
    return dict(ij=ij, d=(dth, dom), origin=(th[0], om[0]), shape=(len(th), len(om)))


def ownership(X, sparse):
    """Which lattice cells the scatter paints, and the nearest drawn state of each painted cell.

    A cell is painted when its centre lies strictly inside at least one marker square (side sqrt(MS2) pt; at this
    panel size that is within 1 cell in theta and 2 in omega of a drawn state). Its owner is the nearest drawn state
    (test or calibration) in (theta, omega), both in rad; the two spacings are equal, so this is lattice-index distance. Ties go to the
    candidate with the smaller theta, then the smaller omega. Returns the lattice, the painted grid, the owner grid
    (-1 where unpainted), the number of tied cells and of tied cells whose candidates differ in sparseness."""
    L = lattice(X)
    (ni, nj), (dth, dom) = L['shape'], L['d']
    side = np.sqrt(MS2) / 72                                     # marker side, inches
    ri = int(np.ceil(side / PANEL_W * np.pi / dth)) - 1          # cells whose centre the marker covers, theta
    rj = int(np.ceil(side / PANEL_H * 2 * np.pi / dom)) - 1      # and omega
    idx = np.full((ni + 2 * ri + 2 * rj, nj + 4 * ri + 4 * rj), -1)  # padded state-index grid
    p0, q0 = ri + rj, 2 * (ri + rj)
    idx[L['ij'][:, 0] + p0, L['ij'][:, 1] + q0] = np.arange(len(X))
    shifted = lambda di, dj: idx[p0 + di:p0 + di + ni, q0 + dj:q0 + dj + nj]
    painted = np.zeros((ni, nj), bool)
    for di in range(-ri, ri + 1):
        for dj in range(-rj, rj + 1):
            painted |= shifted(di, dj) >= 0
    # every painted cell has a state within (ri, rj), so its nearest state lies within d^2 <= ri^2 + rj^2
    r2 = ri ** 2 + rj ** 2
    offs = sorted((di * di + dj * dj, di, dj) for di in range(-ri - rj, ri + rj + 1)
                  for dj in range(-ri - rj, ri + rj + 1) if di * di + dj * dj <= r2)
    owner, best = np.full((ni, nj), -1), np.full((ni, nj), np.inf)
    ties, mixed = np.zeros((ni, nj), bool), np.zeros((ni, nj), bool)
    for d2, di, dj in offs:                                      # ascending d^2, then theta, then omega
        s = shifted(di, dj)
        hit = (s >= 0) & painted
        new = hit & (owner < 0)
        owner[new], best[new] = s[new], d2
        tie = hit & ~new & (best == d2)
        ties |= tie
        mixed |= tie & (sparse[np.where(tie, s, 0)] != sparse[np.where(tie, owner, 0)])
    check((owner[painted] >= 0).all() and (owner[~painted] < 0).all(), 'a painted cell has no owner')
    return L, painted, owner, int(ties.sum()), int(mixed.sum())


def hatch_path(L, cells):
    """One compound path of rectangles covering exactly the given lattice cells: runs along omega in each theta
    column, then identical runs in neighbouring columns joined."""
    runs = []
    for i in np.flatnonzero(cells.any(axis=1)):
        js = np.flatnonzero(cells[i])
        cut = np.flatnonzero(np.diff(js) > 1)
        runs += [(js[a], js[b], i) for a, b in zip(np.r_[0, cut + 1], np.r_[cut, len(js) - 1])]
    rects = []
    for j0, j1, i in sorted(runs):                               # by run, then column
        if rects and rects[-1][:2] == [j0, j1] and rects[-1][3] == i - 1:
            rects[-1][3] = i
        else:
            rects.append([j0, j1, i, i])
    r = np.array(rects, float)
    (dth, dom), (t0, o0) = L['d'], L['origin']
    x0, x1 = t0 + (r[:, 2] - .5) * dth, t0 + (r[:, 3] + .5) * dth
    y0, y1 = o0 + (r[:, 0] - .5) * dom, o0 + (r[:, 1] + .5) * dom
    verts = np.stack([np.c_[x0, y0], np.c_[x1, y0], np.c_[x1, y1], np.c_[x0, y1], np.c_[x0, y0]], axis=1)
    codes = np.tile([Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY], len(r))
    return Path(verts.reshape(-1, 2), codes)


def check_hatching(axs, L, painted, owner, sparse, arm):
    """Read the hatched cells back from the drawn artists: in A and B every sparse state's own cell is hatched, no
    dense state's own cell is, no unpainted cell is, and exactly the sparse states own hatched cells; C holds no
    hatched artist."""
    (dth, dom), (t0, o0), (ni, nj), ij = L['d'], L['origin'], L['shape'], L['ij']
    for ci, ax in enumerate(axs):
        hatched = [p for p in ax.get_children() if getattr(p, 'get_hatch', lambda: None)()]
        if ci == 2:
            check(not hatched, f'{arm}: column C has hatching')
            continue
        check(len(hatched) == 1, f'{arm}: column {"AB"[ci]} has {len(hatched)} hatched artists, expected 1')
        r = hatched[0].get_path().vertices.reshape(-1, 5, 2)[:, :4]
        e = np.c_[(r[:, :, 0].min(1) - t0) / dth + .5, (r[:, :, 0].max(1) - t0) / dth - .5,
                  (r[:, :, 1].min(1) - o0) / dom + .5, (r[:, :, 1].max(1) - o0) / dom - .5]
        check(np.allclose(e, np.rint(e), atol=1e-6), f'{arm}: a hatched rectangle is not on lattice cell edges')
        e = np.rint(e).astype(int)
        H = np.zeros((ni, nj), int)
        for a, b, c, d in e:
            H[a:b + 1, c:d + 1] += 1
        check(H.max() <= 1, f'{arm}: hatched rectangles overlap')
        H = H.astype(bool)
        own_sparse = H[ij[sparse, 0], ij[sparse, 1]]
        own_dense = H[ij[~sparse, 0], ij[~sparse, 1]]
        unpainted = int((H & ~painted).sum())
        owners = np.unique(owner[H])
        check(own_sparse.all(), f'{arm}: {int((~own_sparse).sum())} sparse states\' own cells are not hatched')
        check(not own_dense.any(), f'{arm}: {int(own_dense.sum())} dense states\' own cells are hatched')
        check(unpainted == 0, f'{arm}: {unpainted} unpainted cells are hatched')
        check(np.array_equal(owners, np.flatnonzero(sparse)),
              f'{arm}: {len(owners)} states own hatched cells, expected exactly the {int(sparse.sum())} sparse states')
    return dict(cells=int(H.sum()), rects=len(e), own_sparse=int(own_sparse.sum()), own_dense=int(own_dense.sum()),
                unpainted=unpainted, owners=len(owners))


def check_pdf_hatch(pdf):
    """Every hatch pattern in the saved PDF strokes HATCH_COLOR at HATCH_LW (PathPatch's default edge would make
    it black)."""
    import re, zlib
    from matplotlib.colors import to_rgb
    b = pdf.read_bytes()
    found = []
    for m in re.finditer(rb'<<(.*?)>>\s*stream\r?\n', b, re.S):
        if b'/PatternType' in m.group(1):
            raw = b[m.end():b.index(b'endstream', m.end())]
            s = (zlib.decompress(raw) if b'FlateDecode' in m.group(1) else raw).decode()
            rgb = tuple(float(v) for v in re.search(r'([\d. ]+) RG', s).group(1).split())
            found.append((rgb, float(re.search(r'([\d.]+) w', s).group(1))))
    check(found and all(np.allclose(c, to_rgb(HATCH_COLOR), atol=1e-3) and abs(w - HATCH_LW) < 1e-9 for c, w in found),
          f'hatch patterns in the PDF stroke {found}, expected {to_rgb(HATCH_COLOR)} at {HATCH_LW} pt')
    return found


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


def draw(arrays, stats, cal):
    """Each test and calibration state is drawn where it was evaluated (no interpolation into empty lattice cells);
    the boundary comes from a triangulation of both sets."""
    import matplotlib.tri as mtri
    counts = {}
    tint = lambda c: to_hex(np.array(to_rgb(c)) + (1 - np.array(to_rgb(c))) * C_FILL_TINT)
    c_succ, c_fail = tint(BLUE), tint(ORANGE)
    with plt.rc_context(RC | {'hatch.color': HATCH_COLOR, 'hatch.linewidth': HATCH_LW}):
        # dpi sets the resolution of the rasterized scatter layers inside the PDF (the PNG keeps save()'s 220 dpi)
        fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=RASTER_DPI)
        axs = np.array([[axes_at(fig, LEFT + ci * (PANEL_W + GAP), BOTTOM + (1 - ri) * (PANEL_H + VGAP), PANEL_W, PANEL_H)
                         for ci in range(3)] for ri in range(2)])
        for ri, (arm, name) in enumerate(ROWS):
            a, st = arrays[arm, SEED, ROUND], row_stats(stats, arm)
            n_sp = int((a['d_train'] > SPARSE_D).sum())
            check(n_sp == EXPECT[arm]['sparse'], f"{arm}: {n_sp} sparse test states, expected {EXPECT[arm]['sparse']}")
            check(n_sp == round(st['baseline_sparse'] * N_TEST), f"{arm}: sparse count differs from baseline_sparse")
            decision_masks(a, st, arm)                          # test-only checks: counts unchanged by the union
            ct = set_counts(a)
            cc = set_counts(cal[arm])
            check(cc['states'] == N_CAL, f"{arm}: {cc['states']} calibration states, expected {N_CAL}")
            print(f"{arm}: {N_CAL} calibration states = success {cc['success']} + fail {cc['fail']} + abstain "
                  f"{cc['abstain']}; incorrect {cc['wrong']}, all within success|fail; sparse {cc['sparse']}")
            # draw test and calibration states together: every map, the hatch and the boundary use the union
            u = {k: np.concatenate([a[k], cal[arm][k]]) for k in ('X', 'y', 'I', 'D', 'succ', 'fail', 'wrong', 'd_train')}
            X, sparse = u['X'], u['d_train'] > SPARSE_D
            mk = dict(success=u['succ'], fail=u['fail'], abstain=~(u['succ'] | u['fail']), wrong=u['wrong'])
            cu = set_counts(u)
            check(cu['states'] == N_TEST + N_CAL and all(cu[k] == ct[k] + cc[k] for k in cu),
                  f'{arm}: the union does not partition into the test and calibration counts')
            print(f"{arm}: test + calibration = {cu['states']} states = success {cu['success']} + fail {cu['fail']} + "
                  f"abstain {cu['abstain']}; incorrect {cu['wrong']}; sparse {cu['sparse']}")
            tri = mtri.Triangulation(X[:, 0], X[:, 1])
            L, painted, owner, n_ties, n_mixed = ownership(X, sparse)
            cells = hatch_path(L, (owner >= 0) & sparse[np.maximum(owner, 0)])
            for ci, key in enumerate(['I', 'D']):
                v = u[key]
                check(v.min() >= 0 and v.max() <= 1, f'{arm} {key} outside the fixed [0, 1] scale')
                o = np.argsort(v, kind='stable')               # high values drawn last, so they stay visible
                m = axs[ri, ci].scatter(X[o, 0], X[o, 1], c=v[o], s=MS2, marker='s', lw=0, cmap='Purples',
                                        vmin=0, vmax=1, rasterized=True, zorder=1, snap=False)
                # transparent fill and lw=0: only hatch lines are drawn. The explicit edgecolor sets the hatch colour
                # (PathPatch's default black edge would override rcParams hatch.color); width from hatch.linewidth.
                axs[ri, ci].add_patch(PathPatch(cells, facecolor='none', edgecolor=HATCH_COLOR, lw=0, hatch=HATCH,
                                                zorder=2))
            counts[arm] = dict(test=ct, calibration=cc, union=cu)
            ax = axs[ri, 2]
            z_abst = 5 if ABSTAIN_ABOVE_BOUNDARY else 2              # boundary 4, crosses 6
            for key, color, z, scale in (('success', c_succ, 1, 1), ('fail', c_fail, 1, 1),
                                         ('abstain', ABSTAIN, z_abst, ABSTAIN_AREA_SCALE)):   # after Success and Fail
                ax.scatter(X[mk[key], 0], X[mk[key], 1], s=MS2 * scale, marker='s', lw=0, color=color,
                           rasterized=True, zorder=z, snap=False)
            w = mk['wrong']
            ax.plot(X[w, 0], X[w, 1], ls='none', marker='x', ms=WRONG_MS, mew=WRONG_MEW, color=WRONG, zorder=6)
            for ci, ax in enumerate(axs[ri]):
                ax.tricontour(tri, u['y'].astype(float), levels=[.5], colors='k' if ci < 2 else C_BOUNDARY_COLOR,
                              linewidths=LW if ci < 2 else C_BOUNDARY_LW, zorder=4)
            hc = check_hatching(axs[ri], L, painted, owner, sparse, arm)
            counts[arm]['union'] |= dict(sparse_owning_hatch=hc['owners'], hatched_cells=hc['cells'],
                                         painted_cells=int(painted.sum()), ownership_ties=n_ties,
                                         ties_sparse_vs_dense=n_mixed)
            print(f"{arm}: hatch = {hc['cells']} painted cells owned by sparse states ({hc['rects']} rectangles); "
                  f"sparse own cells hatched {hc['own_sparse']}/{cu['sparse']}, dense own cells hatched {hc['own_dense']}, "
                  f"unpainted hatched {hc['unpainted']}, states owning hatch {hc['owners']}; ownership ties {n_ties} "
                  f"cells ({n_mixed} between a sparse and a dense state); C unhatched")
            for ci, ax in enumerate(axs[ri]):
                style_ticks(ax, bottom=ri == 1, left=ci == 0)
            y_mid = BOTTOM + (1 - ri) * (PANEL_H + VGAP) + PANEL_H / 2
            fig.text(.04 / FIG_W, y_mid / FIG_H, name, rotation=90, ha='left', va='center', fontsize=9)
        for ax, (letter, title) in zip(axs[0], [('A', 'Invalid-terminal mass'), ('B', 'Weighted disagreement'),
                                                ('C', 'Certified decisions')]):
            ax.set_title(rf'$\mathbf{{{letter}}}$. {title}', loc='left', fontsize=9, pad=3)
        top = FIG_H - .04
        cax = axes_at(fig, .08, top - .19, .82, .065)          # top left, label above the bar
        cb = fig.colorbar(m, cax=cax, orientation='horizontal', ticks=[0, .5, 1])
        cb.set_ticklabels(['0', '0.5', '1'])
        cb.set_label('Signal value', fontsize=8, labelpad=2)
        cax.xaxis.set_label_position('top')
        cax.tick_params(labelsize=7.5, length=2, width=.5, pad=1.5)
        cb.outline.set_linewidth(.5)
        handles = [Patch(color=c_succ, label='Success'), Patch(color=c_fail, label='Fail'),
                   Patch(color=ABSTAIN, label='Abstain'),
                   Line2D([], [], ls='none', marker='x', ms=WRONG_MS, mew=WRONG_MEW, color=WRONG, label='Incorrect prediction'),
                   Line2D([], [], color='k', lw=LW, label='True outcome boundary'),
                   Patch(facecolor='none', edgecolor=HATCH_COLOR, lw=.4, hatch=HATCH,   # outlined so the swatch
                         label='Sparse training coverage (distance > 0.02)')]           # reads as an area
        fig.legend(handles=handles, loc='upper right', ncol=3, frameon=False,
                   bbox_to_anchor=((LEFT + 3 * PANEL_W + 2 * GAP) / FIG_W, top / FIG_H), fontsize=8, borderpad=0,
                   borderaxespad=0, columnspacing=1.2, handlelength=1.5, handleheight=.8, handletextpad=.5,
                   labelspacing=.35)
        size = save(fig, 'archive/pendulum_uncertainty_signals_with_decisions')    # plots/archive only, never -pw
    check(PAGE_W[0] <= size['width_in'] <= PAGE_W[1] and PAGE_H[0] <= size['height_in'] <= PAGE_H[1],
          f"PDF is {size['width_in']} x {size['height_in']} in, expected {PAGE_W} x {PAGE_H}")
    pat = check_pdf_hatch(PLOTS / 'archive/pendulum_uncertainty_signals_with_decisions.pdf')
    print(f'PDF hatch patterns: {pat} (stroke RGB, width pt)')
    return size, counts


def flatten(s):
    out = {k: v for k, v in s.items() if not isinstance(v, dict)}
    for k, v in s.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                out[f'{k}.{kk}'] = vv
    return out


def main():
    data = build()
    stats, arrays = data['stats'], data['arrays']
    size, counts = draw(arrays, stats, calibration(data))
    flat = [flatten(s) for s in stats]
    cols = sorted({k for f in flat for k in f}, key=lambda k: (k.split('.')[0] not in ('arm', 'seed', 'round', 'run'), k))
    with open(VALUES / 'archive/pendulum_uncertainty_signals_with_decisions_stats.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in sorted(flat, key=lambda r: (r['arm'], r['round'], SEEDS.index(r['seed']))):
            w.writerow({k: ('' if r.get(k) is None else r.get(k)) for k in cols})
    comp = compare(stats)
    (VALUES / 'archive/pendulum_uncertainty_signals_with_decisions_values.json').write_text(json.dumps(dict(
        figure=dict(seed=SEED, round=ROUND, page_size=size, states_drawn='test + calibration',
                    categories_drawn=['success', 'fail', 'abstain'], abstain_mask='~(succ | fail)',
                    marked='wrong (incorrect Success/Fail), test and calibration', counts=counts),
        manuscript_comparison=comp, threshold_sums=data['threshold_sums'],
        max_t_s_plus_u_f=max(x['t_s_plus_u_f'] for x in data['threshold_sums']),
        pool_sha256=data['pool_sha256'], provenance=data['provenance'], stats=stats), indent=1, default=str) + '\n')
    for c in comp:
        v = c['manuscript_variant']
        print(f"{'OK ' if c['matches'] else 'DIFF'} {c['claim']}: manuscript {c['manuscript']}, recomputed "
              f"{'n/a' if c['recomputed'] is None else round(c['recomputed'], 4)} ({c['seeds']} seeds)"
              + (f" | {v['definition']} {round(v['value'], 4)} {'matches' if v['matches'] else 'differs'}" if v else ''))


if __name__ == '__main__':
    main()
