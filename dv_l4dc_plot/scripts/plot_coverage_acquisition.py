"""Extended Figure 5 preview: where data are missing and where the next batch goes (TERA s99512, round 2, full grid).

Requested 2026-11-13 by Sumanth via paper_writer_2. Same checkpoint as Figure 5 (x231_penlqr_fm_uh_a6d075_r1_s99512,
round 2, 120 training trajectories), drawn on the full 158 x 315 grid with the true success/failure boundary in every
panel. Two layouts at the 6.0 in text width:
  v1  1 x 4: (a) distance to training data, (b) I(x), (c) D(x), (d) U(x) = I + D with the next batch's 35 initial states
  v2  1 x 5: (a) abstention (the current Figure 5 panel A), then the four panels of v1
Distance to training data is uncertainty_signals_data's d_train: coordinates (theta / 2pi wrapped, omega / 4pi), every
state of the first 120 training trajectories. I, D and U share one [0, 1] colour scale. On (d)/(e) the 27 states chosen
by U are filled circles and the 8 uniform states open triangles.

The acquired states come from the run's own files and are checked against them: epoch_002/artifacts_v2.json
(acquisition.d1_indices, d2_indices), epoch_002/acquisition_scores.npz, dataset_builder_state.json and
datasets/train_trajectories.txt. The tie rule of verdict_scores.select_topk (tie_tol > 0) and spread_farthest_point at
the runs' commit acae65bf is re-run on the stored scores and must give d2_indices in order.

  2x2 (--layout 2x2, W_2X2 x H_2X2 in, 8.5 pt text and 8 pt ticks, the main-text figure since 2026-11-13): (a) distance,
      (b) I on top; (c) D, (d) U with the markers below; theta-dot on the left column only, theta on the bottom row only;
      the two colour bars sit under the top row (teal under (a); under (b) the one Reds bar for (b)-(d), named I, D, U).

Writes plots/variants/pendulum_coverage_acquisition_{v1,v2}.{pdf,png} and pendulum_coverage_acquisition.txt (default),
or pendulum_coverage_acquisition_2x2.{pdf,png,txt} (--layout 2x2). Nothing goes to the paper; paper_writer_2 copies
the 2x2 and v2 (as _1x5; v1 as _1x4) into figures/results/ by hand.
"""
import json
from pathlib import Path
import numpy as np
import yaml
from matplotlib import patheffects
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.spatial import cKDTree
import plot_uncertainty_signals_paper as P
import uncertainty_signals_data as usd
from common import PLOTS, FILL, FS, style_rc, paper_legend_kw, SIGNAL_RAMP, ABSTAIN_ON_WHITE, plt

STEM = 'pendulum_coverage_acquisition'
OUT = PLOTS / 'variants'
W = 6.0                                       # text width, 433.6 pt
ROUND, SEED = 2, 99512
SPLIT = Path('/common/users/shared/pracsys/genMoPlan/data_trajectories/deterministic/pendulum/lqr_failset/'
             'splits_det_rerun')
COV_MAX = .06                                 # top of the distance scale; the few larger cells saturate
COV_TICKS = [0, .02, .04, .06]                # 0.02 is the paper's sparse cutoff
SIG_CMAP = LinearSegmentedColormap.from_list('cov_acq_signal', SIGNAL_RAMP['I'])
COV_CMAP = LinearSegmentedColormap.from_list('cov_acq_distance', SIGNAL_RAMP['D'])
MARK = dict(u=dict(marker='o', markersize=3.3, markerfacecolor='black', markeredgecolor='white', markeredgewidth=.45),
            unif=dict(marker='^', markersize=3.8, markerfacecolor='white', markeredgecolor='black',
                      markeredgewidth=.6))
TITLE = dict(abst=('abstention', None),
             cov=('distance to training data', 'distance to\ntraining data'),
             I=(r'invalid mass $I(x)$', 'invalid mass\n' + r'$I(x)$'),
             D=(r'weighted disagreement $D(x)$', 'weighted\n' + r'disagreement $D(x)$'),
             U=(r'acquisition score $U(x)$', 'acquisition\n' + r'score $U(x)$'))
VARIANTS = dict(v1=['cov', 'I', 'D', 'U'], v2=['abst', 'cov', 'I', 'D', 'U'])
PT = 1 / 72
MARGIN, SEP_PT, LEG_GAP_PT, TITLE_GAP_PT, CB_GAP_PT, CB_H, GAP_PT = .01, 1.5, 3.0, 2.0, 3.0, .045, 4.0
VBAR_GAP_PT = 6.0                             # v1: from the right end of the vertical bar's labels to panel (b)
need = P.need


# ---------------------------------------------------------------------------------------------------------- data
def acquired():
    """The 35 initial states the run acquired after round 2, checked against every file that records them."""
    run = usd.runs()['adaptive', SEED]
    R = run['dir']
    cfg = yaml.safe_load(open(R / 'resolved_config.yaml'))
    need(cfg['data_source']['shuffled_indices_file'] == str(SPLIT / f'pool_order_{SEED}.txt'), 'pool order file')
    need(cfg['data_source']['trajectories_dir'] == str(usd.POOL), 'pool file')
    art = json.load(open(R / f'epoch_{ROUND:03d}/artifacts_v2.json'))
    acq, dg = art['acquisition'], art['acquisition']['diagnostics']
    d1, d2 = list(acq['d1_indices']), list(acq['d2_indices'])
    need(art['train_trajectories'] == 120 and len(d1) == 8 and len(d2) == 27, 'round-2 batch sizes')
    need(not dg.get('fill_indices'), 'shortfall fill used')
    order = np.loadtxt(SPLIT / f'pool_order_{SEED}.txt', dtype=int)
    ti = np.array(json.load(open(R / 'dataset_builder_state.json'))['train_indices'])
    tt = np.loadtxt(R / 'datasets/train_trajectories.txt', dtype=int)
    need(np.array_equal(tt, order[ti]), 'train_trajectories.txt is not pool_order[train_indices]')
    need(ti[120:155].tolist() == d1 + d2, 'training rows 121-155 are not d1 + d2 in pool positions')
    need(np.array_equal(tt[120:155], order[d1 + d2]), 'trajectory ids of lines 121-155 differ')
    z = np.load(R / f'epoch_{ROUND:03d}/acquisition_scores.npz')
    ci, sel, u = z['candidate_indices'], z['selected'], z['u_h'].astype(float)
    need(np.array_equal(z['score'], z['u_h']) and dg['score'] == 'u_h' and dg['selection'] == 'topk', 'score is not U')
    need(set(ci[sel].tolist()) == set(d2) and not np.isin(d1, ci).any() and not np.isin(ti[:120], ci).any(),
         'candidate set')
    need(len(set(ti[:120].tolist()) | set(d1) | set(ci.tolist())) == len(order), 'candidates are not the unused pool')
    pool = usd.Pool()
    start = lambda pos: pool.S[pool.O[order[np.asarray(pos, int)]]]
    # Start state = first row of the trajectory in train.npz, looked up by pool position and by trajectory id.
    need(np.array_equal(start(d1 + d2), pool.S[pool.O[tt[120:155]]]), 'start states')

    # Tie rule (verdict_scores.select_topk, tie_tol > 0, at acae65bf): near-ties within tie_tol of the k-th score are
    # one band; candidates above the band are taken; the rest are filled from the band by greedy farthest-point
    # selection (spread_farthest_point) in the model's input embedding (sin theta, cos theta, omega / 2pi) of the
    # normalised state, anchored on every start state already rolled out (rounds 0-2 and this round's uniform 8).
    tol, k = float(dg['tie_tol']), 27
    ur = np.round(u, 12)
    o = np.lexsort((np.arange(len(u)), -ur))
    u_kth = ur[o[k - 1]]
    above = o[ur[o] > u_kth + tol]
    band = np.flatnonzero((ur >= u_kth - tol) & (ur <= u_kth + tol))
    need((len(band), u_kth, k - len(above)) == (dg['n_tie_group'], dg['u_kth'], dg['n_tie_taken']), 'tie band')
    emb = lambda x: np.c_[np.sin(x[:, 0]), np.cos(x[:, 0]), np.clip(x[:, 1] / (2 * np.pi), -1, 1)].astype(float)
    anchors = emb(start(sorted(set(ti[:120].tolist()) | set(d1))).astype(np.float32))
    cand = emb(start(ci[band]).astype(np.float32))
    d = cKDTree(anchors).query(cand)[0]
    taken = []
    for _ in range(k - len(above)):
        j = int(np.argmax(d))
        taken.append(j)
        d = np.minimum(d, np.linalg.norm(cand - cand[j], axis=1))
        d[j] = -np.inf
    picks = np.r_[above, band[taken]].astype(int)
    need([int(ci[i]) for i in picks] == d2, 'the tie rule does not reproduce d2_indices in order')
    nb, nsb, nfb = z['n_invalid'][band], z['n_success'][band], z['n_failure'][band]
    ns, nf, ni = z['n_success'][picks], z['n_failure'][picks], z['n_invalid'][picks]
    anchor_d = cKDTree(anchors).query(cand)[0]
    info = dict(run=run['name'], n_pool=len(order), n_candidates=len(ci), n_candidates_config=cfg['acquisition'].get(
        'n_candidates'), tie_tol=tol, u_kth=float(u_kth), n_above=len(above), n_band=len(band),
        n_band_u_eq_1=int((ur[band] == 1).sum()), band_all_invalid=int((nb == 100).sum()),
        band_even_split=int(((nsb == nfb) & (nsb > 0)).sum()), picks_all_invalid=int((ni == 100).sum()),
        picks_even_split=int(((ns == nf) & (ns > 0)).sum()), picks_u_eq_1=int((ur[picks] == 1).sum()),
        picks_u_min=float(u[picks].min()), band_anchor_dist=np.quantile(anchor_d, [0, .5, 1]).tolist(),
        picks_anchor_dist=np.quantile(anchor_d[taken], [0, .5, 1]).tolist(), score_mean=float(u.mean()),
        n_candidates_u_ge_0_5=int((u >= .5).sum()), candidate_u=u, candidate_states=start(ci).astype(float))
    return dict(unif=start(d1).astype(float), u=start(d2).astype(float), d1=d1, d2=d2, lines=(121, 155), info=info)


def cell_values(g, X, key):
    """Grid value at the lattice cell nearest each state (0.04 rad lattice; theta wrapped onto the grid's span)."""
    th, om = g['theta'], g['omega']
    i = np.rint((X[:, 0] - th[0]) / .04).astype(int)
    i = np.where(i >= len(th), i - int(round(2 * np.pi / .04)), i).clip(0, len(th) - 1)
    j = np.rint((X[:, 1] - om[0]) / .04).astype(int).clip(0, len(om) - 1)
    return g[key][i, j]


# -------------------------------------------------------------------------------------------------------- figure
def draw(g, acq, variant):
    keys = VARIANTS[variant]
    N = len(keys)
    ext = P._extent(g)
    th, om = g['theta'], g['omega']
    field = dict(cov=g['d_train'], I=g['I'], D=g['D'], U=g['I'] + g['D'])
    vbar = variant == 'v1'                    # v1: the distance bar stands upright right of panel (a)
    with plt.rc_context(style_rc()):
        H0 = 2.0
        fig = plt.figure(figsize=(W, H0))
        axs = [fig.add_axes([.05 + .19 * c, .3, .15, .4]) for c in range(N)]
        ims = {}
        ylab = np.ma.masked_less(g['y'].astype(float), 0)
        for ax, key in zip(axs, keys):
            if key == 'abst':
                ax.imshow(P._abst_rgba(g['abst']).transpose(1, 0, 2), origin='lower', extent=ext,
                          interpolation='nearest', aspect='auto', zorder=7)
            else:
                cov = key == 'cov'
                ims[key] = ax.imshow(field[key].T, origin='lower', extent=ext, interpolation='nearest', aspect='auto',
                                     cmap=COV_CMAP if cov else SIG_CMAP, vmin=0, vmax=COV_MAX if cov else 1, zorder=1)
            ax.contour(th, om, ylab.T, levels=[.5], colors=FILL['boundary'], linewidths=P.BOUNDARY_LW, zorder=6)
            ax.set_xlim(ext[:2]); ax.set_ylim(ext[2:])
            ax.set_xticks([-np.pi, 0, np.pi]); ax.set_yticks([-2 * np.pi, 0, 2 * np.pi])
        ax_u = axs[keys.index('U')]
        for kind in ('unif', 'u'):
            ax_u.plot(acq[kind][:, 0], acq[kind][:, 1], ls='none', zorder=9, clip_on=False, **MARK[kind])
        for c, ax in enumerate(axs):
            ax.tick_params(labelbottom=(c == 0), labelleft=(c == 0))
        for t, ha in zip(axs[0].set_xticklabels([r'$-\pi$', '0', r'$\pi$']), ('left', 'center', 'right')):
            t.set_ha(ha)
        for t, va in zip(axs[0].set_yticklabels([r'$-2\pi$', '0', r'$2\pi$'], rotation=90), ('bottom', 'center', 'top')):
            t.set_va(va)
        axs[0].set_xlabel(r'$\theta$', labelpad=1)
        axs[0].set_ylabel(r'$\dot{\theta}$', labelpad=1)
        names = [f'({chr(97 + c)}) ' for c in range(N)]
        titles = [[fig.text(0, 0, names[c] + TITLE[k][0], ha='center', va='baseline', fontsize=FS['title'])]
                  for c, k in enumerate(keys)]

        def colourbar(key, ticks, labels, orientation):
            cax = fig.add_axes([.5, .05, .2, .02])
            cb = fig.colorbar(ims[key], cax=cax, orientation=orientation, ticks=ticks)
            if orientation == 'horizontal':
                for t, ha in zip(cb.ax.set_xticklabels(labels), ['left'] + ['center'] * (len(ticks) - 2) + ['right']):
                    t.set_ha(ha)
            else:
                for t, va in zip(cb.ax.set_yticklabels(labels), ['bottom'] + ['center'] * (len(ticks) - 2) + ['top']):
                    t.set_va(va)
            cb.outline.set_linewidth(.4)
            cax.tick_params(labelsize=FS['tick'], length=1.5, pad=1)
            return cax

        cax_cov = colourbar('cov', COV_TICKS, ['0', '0.02', '0.04', '0.06'], 'vertical' if vbar else 'horizontal')
        cax_sig = colourbar('U', [0, .5, 1], ['0', '0.5', '1'], 'horizontal')
        caxs = [cax_cov, cax_sig]
        handles = ([Patch(facecolor=ABSTAIN_ON_WHITE, edgecolor='none', label='abstention')] if 'abst' in keys else []) + \
            [Line2D([], [], color=FILL['boundary'], lw=P.BOUNDARY_LW, label='success/failure boundary'),
             Line2D([], [], ls='none', label=r'next states chosen by $U$', **MARK['u']),
             Line2D([], [], ls='none', label='next states drawn uniformly', **MARK['unif'])]
        legend = fig.legend(handles=handles, ncol=len(handles), **paper_legend_kw(FS['legend']))

        # Pass 1: measure.
        fig.canvas.draw()
        rr = fig.canvas.get_renderer()
        inch = lambda bb: (bb.width / fig.dpi, bb.height / fig.dpi)
        pos = lambda a: a.get_position()
        tb = lambda a: a.get_tightbbox(rr)
        left_dec = pos(axs[0]).x0 * W - tb(axs[0]).x0 / fig.dpi
        below_a = pos(axs[0]).y0 * H0 - tb(axs[0]).y0 / fig.dpi
        below_rest = max(pos(a).y0 * H0 - tb(a).y0 / fig.dpi for a in axs[1:])
        hbars = [cax_sig] if vbar else caxs
        cb_below = max(pos(c).y0 * H0 - tb(c).y0 / fig.dpi for c in hbars)
        vbar_w = (tb(cax_cov).x1 / fig.dpi - (pos(cax_cov).x0 + pos(cax_cov).width) * W) if vbar else 0
        above = max(max(0.0, tb(a).y1 / fig.dpi - (pos(a).y0 + pos(a).height) * H0) for a in axs)
        gap = GAP_PT * PT
        wide_gap = (CB_GAP_PT * PT + CB_H + vbar_w + VBAR_GAP_PT * PT) if vbar else gap
        leg_h = inch(legend.get_window_extent(rr))[1]
        need(inch(legend.get_window_extent(rr))[0] <= W - 2 * MARGIN, 'legend wider than the page')
        x_a = MARGIN + P.ACCENT_PT * PT + left_dec
        mark_r = MARK['unif']['markersize'] / 2 * PT          # markers may reach past the right edge of the U panel
        right_pad = max(0.0, mark_r) if keys[-1] == 'U' else 0.0
        pw = (W - MARGIN - right_pad - x_a - wide_gap - (N - 2) * gap) / N
        ph = pw * P.ASPECT
        tw = [inch(t[0].get_window_extent(rr))[0] for t in titles]
        wrapped = any(w_ > pw + gap - SEP_PT * PT for w_ in tw)
        if wrapped:
            for c, key in enumerate(keys):
                s_ = TITLE[key][1] or TITLE[key][0]
                lines = (names[c] + s_).split('\n')
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
        bottom = max(below_a, below_rest + CB_GAP_PT * PT + CB_H + cb_below)
        H = MARGIN + leg_h + LEG_GAP_PT * PT + title_h + TITLE_GAP_PT * PT + above + ph + bottom + MARGIN
        fig.set_size_inches(W, H)

        # Place.
        y_head = H - MARGIN - leg_h - LEG_GAP_PT * PT
        top = y_head - title_h - TITLE_GAP_PT * PT - above
        xs = [x_a]
        for c in range(1, N):
            xs.append(xs[-1] + pw + (wide_gap if (vbar and c == 1) else gap))
        for c, ax in enumerate(axs):
            ax.set_position([xs[c] / W, (top - ph) / H, pw / W, ph / H])
        for c, lines in enumerate(titles):
            first = n_lines - len(lines)
            for k_, t in enumerate(lines):
                t.set_position(((xs[c] + pw / 2) / W, (y_head - asc0 - (first + k_) * SKIP) / H))
        cb_top = top - ph - below_rest - CB_GAP_PT * PT
        i_sig = [keys.index(k) for k in ('I', 'D', 'U')]
        need(i_sig == list(range(i_sig[0], i_sig[0] + 3)), 'I, D, U are not adjacent')
        x0s, x1s = xs[i_sig[0]], xs[i_sig[-1]] + pw
        cax_sig.set_position([x0s / W, (cb_top - CB_H) / H, (x1s - x0s) / W, CB_H / H])
        c_cov = keys.index('cov')
        if vbar:
            cax_cov.set_position([(xs[c_cov] + pw + CB_GAP_PT * PT) / W, (top - ph) / H, CB_H / W, ph / H])
        else:
            cax_cov.set_position([xs[c_cov] / W, (cb_top - CB_H) / H, pw / W, CB_H / H])
        legend.set_bbox_to_anchor((0.5, (H - MARGIN) / H), transform=fig.transFigure)

        # Checks: no overlapping text, nothing off the page, no text over a panel, images are the computed fields.
        fig.canvas.draw()
        rr = fig.canvas.get_renderer()

        def drawn(axis, lim):
            lo, hi = sorted(lim)
            return [t for t, v in zip(axis.get_ticklabels(), axis.get_majorticklocs())
                    if t.get_text() and t.get_visible() and lo - 1e-9 <= v <= hi + 1e-9]

        free = [(t.get_text(), t.get_window_extent(rr)) for t in fig.texts] + [('legend', legend.get_window_extent(rr))]
        owned = []
        for a in axs + caxs:
            owned += [(t.get_text(), t.get_window_extent(rr)) for t in drawn(a.xaxis, a.get_xlim()) +
                      drawn(a.yaxis, a.get_ylim())]
            owned += [(lab.get_text(), lab.get_window_extent(rr)) for lab in (a.xaxis.label, a.yaxis.label)
                      if lab.get_text()]
        boxes = free + owned
        for i, (ta, a) in enumerate(boxes):
            for tbx, b in boxes[i + 1:]:
                need(not a.overlaps(b), ('text overlap', ta, tbx))
            need(a.x0 >= -0.5 and a.x1 <= fig.bbox.x1 + 0.5 and a.y0 >= -0.5 and a.y1 <= fig.bbox.y1 + 0.5,
                 ('text off the page', ta))
        panels = [a.get_window_extent(rr) for a in axs + caxs]
        for ta, a in boxes:
            need(not any(a.overlaps(p) for p in panels), ('text over a panel', ta))
        need(P.PDF_DPI * pw >= P.GRID_SHAPE[0] and P.PDF_DPI * ph >= P.GRID_SHAPE[1], 'a cell is lost in the raster')
        for ax, key in zip(axs, keys):
            im = ax.get_images()
            need(len(im) == 1 and np.allclose(im[0].get_extent(), ext), f'{key}: image extent')
            arr = np.ma.getdata(im[0].get_array())
            want = P._abst_rgba(g['abst']).transpose(1, 0, 2) if key == 'abst' else field[key].T
            need(np.array_equal(arr, want), f'{key}: image is not the computed field')
    layout = dict(width=W, height=H, panel_w=pw, panel_h=ph, col_gap=gap, wide_gap=wide_gap, titles_wrapped=wrapped,
                  cell_pt=(72 * pw / P.GRID_SHAPE[0], 72 * ph / P.GRID_SHAPE[1]))
    return fig, layout


# ---------------------------------------------------------------------------------------------- 2 x 2 (half width)
# Sumanth via paper_writer_2, 2026-11-13: the main-text figure as a 2 x 2 wrap at 3.30 in (0.55 textwidth, as Figure 5).
# (a) distance, (b) I on top; (c) D, (d) U with the next batch below. theta-dot only on the left column, theta only on
# the bottom row, tick labels only there. The top row has no x labels, so its band holds the two colour bars: teal under
# (a), and under (b) the one Reds bar for (b)-(d), labelled I, D, U between its end ticks.
# Since 2026-11-13 evening (Sumanth via paper_writer_2): the page stays 3.300 x 3.524 in and the text sits just under
# the 9 pt \footnotesize caption: titles, axis labels, legend and the bar name 8.5 pt, tick labels 8 pt. The panels take
# whatever the larger text leaves, so their aspect follows from the page; each side's margin is tuned until the ink sits
# INK_LO-INK_HI in from the page edge.
# Then (d579c2c) the wrap became 0.45 textwidth: the page was 2.700 x 2.883 in, rendered natively (scale 1.00).
# Then Sumanth set the wrap to 0.35 textwidth (via paper_writer_2): 2.100 in wide, rendered natively; the height is ours
# and shorter is better (one wrap line per 13.6 pt), so the panels sit just above the 0.525 in that keeps every one of
# the grid's 315 omega cells in the 600 dpi PDF raster.
W_2X2, H_2X2 = 2.100, 2.265
# y tick labels: -2pi and 2pi only (the 0 tick stays unlabelled); at this panel height the rotated three-label set
# leaves under 3 pt between labels.
Y_TICKLABELS_2X2 = [r'$-2\pi$', '', r'$2\pi$']
# Titles: all four panels use one wording level, the longest at which both rows fit on one line: full (TITLE[k][0]),
# TITLE_SHORT, then TITLE_SYMBOL (the text beside the figure names I(x) and D(x)); else full on two lines.
TITLE_SHORT = dict(cov='distance to data', I=r'invalid mass $I(x)$', D=r'disagreement $D(x)$', U=r'acquisition $U(x)$')
TITLE_SYMBOL = dict(cov='distance to data', I=r'$I(x)$', D=r'$D(x)$', U=r'$U(x)$')
TITLE_SEP_PT = 6.0                    # least space between two titles of a row, so they never read as one line
# Legend: the first (labels, rows) that fits the width. A labels tuple of three puts its first item before the two marker
# entries as a row heading ("next states:"), so the short marker labels still say what the markers are.
LEG_LABELS = [(r'next states chosen by $U$', 'next states drawn uniformly'), (r'next states by $U$', 'next states, uniform'),
              ('next states:', r'by $U$', 'uniform')]
LEG_HEAD_PT = 4.0                     # from the row heading to the first marker
COV_TICKS_2X2 = [0, .03, .06]         # four labels (COV_TICKS) collide on a panel this narrow
FS_2X2 = dict(title=8.5, label=8.5, legend=8.5, bar=8.5, tick=8)
RC_2X2 = {'font.size': 8.5, 'axes.labelsize': 8.5, 'legend.fontsize': 8.5, 'xtick.labelsize': 8, 'ytick.labelsize': 8}
INK_LO, INK_HI, INK_TARGET = .010, .020, .013
ROW_GAP_PT = 3.0                      # from the colour-bar labels down to the second row's titles
KEYS_2X2 = [['cov', 'I'], ['D', 'U']]
# Three regions circled in (a), (b) and (c) (Sumanth, 2026-11-14; centres read off his sketch on (a), in (theta,
# theta-dot)). In his words: (1) high data density, relatively low-medium invalid mass and high disagreement;
# (2) high data density and medium-high invalid mass; (3) high invalid mass and low disagreement despite being at the
# boundary. The third centre is raised to -5.10 so its circle stays inside the panel. One size everywhere: a
# CIRCLE_PT diameter in print, black with a white casing so it reads on the teal and the Reds.
CIRCLES_2X2 = [(0.570, 2.195), (-0.643, 1.514), (-0.117, -5.10)]
CIRCLE_KEYS = ('cov', 'I', 'D')
CIRCLE_PT, CIRCLE_LW, CIRCLE_CASE = 7.0, .8, .5
# Panel (a) title as on Overleaf since 24d5f3f (Sumanth, 2026-11-13): 'distance to training states' in two lines at
# 7 pt, centred over the panel, the second baseline 2 pt below the row's baseline. The layout is still measured with
# the one-line symbol title, so the panels keep their size. The 24d5f3f PDF set the lines 7 pt apart, which puts the
# first line's box 0.4 pt into the legend heading's; 6.5 pt clears it and leaves about 0.6 pt of ink between lines.
COV_TITLE_LINES, COV_TITLE_PT, COV_TITLE_DROP_PT, COV_TITLE_SKIP_PT = ('(a) distance to', 'training states'), 7.0, 2.0, 6.5


def drawn_labels(cax):
    return [t for t, v in zip(cax.xaxis.get_ticklabels(), cax.xaxis.get_majorticklocs())
            if t.get_visible() and min(cax.get_xlim()) - 1e-9 <= v <= max(cax.get_xlim()) + 1e-9]


def draw_2x2(g, acq, edge):
    """edge: the layout margin (in) on each side, measured to the text and panel boxes."""
    ext = P._extent(g)
    th, om = g['theta'], g['omega']
    field = dict(cov=g['d_train'], I=g['I'], D=g['D'], U=g['I'] + g['D'])
    W_, H = W_2X2, H_2X2
    with plt.rc_context(style_rc(**RC_2X2)):
        H0 = 4.0
        fig = plt.figure(figsize=(W_, H0))
        axs = [[fig.add_axes([.12 + .45 * c, .55 - .45 * r, .38, .35]) for c in range(2)] for r in range(2)]
        flat = [(r, c, axs[r][c], KEYS_2X2[r][c]) for r in range(2) for c in range(2)]
        ims = {}
        ylab = np.ma.masked_less(g['y'].astype(float), 0)
        for r, c, ax, key in flat:
            cov = key == 'cov'
            ims[key] = ax.imshow(field[key].T, origin='lower', extent=ext, interpolation='nearest', aspect='auto',
                                 cmap=COV_CMAP if cov else SIG_CMAP, vmin=0, vmax=COV_MAX if cov else 1, zorder=1)
            ax.contour(th, om, ylab.T, levels=[.5], colors=FILL['boundary'], linewidths=P.BOUNDARY_LW, zorder=6)
            ax.set_xlim(ext[:2]); ax.set_ylim(ext[2:])
            ax.set_xticks([-np.pi, 0, np.pi]); ax.set_yticks([-2 * np.pi, 0, 2 * np.pi])
            ax.tick_params(labelbottom=(r == 1), labelleft=(c == 0))
            if r == 1:
                for t, ha in zip(ax.set_xticklabels([r'$-\pi$', '0', r'$\pi$']), ('left', 'center', 'right')):
                    t.set_ha(ha)
                ax.set_xlabel(r'$\theta$', labelpad=1)
            if c == 0:
                for t, va in zip(ax.set_yticklabels(Y_TICKLABELS_2X2, rotation=90), ('bottom', 'center', 'top')):
                    t.set_va(va)
                ax.set_ylabel(r'$\dot{\theta}$', labelpad=1)
        circles = [axs[r][c].plot(*np.array(CIRCLES_2X2).T, ls='none', marker='o', markersize=CIRCLE_PT,
                                  markerfacecolor='none', markeredgecolor='black', markeredgewidth=CIRCLE_LW, zorder=8,
                                  path_effects=[patheffects.withStroke(linewidth=CIRCLE_LW + 2 * CIRCLE_CASE,
                                                                       foreground='white')])[0]
                   for r, c, _, key in flat if key in CIRCLE_KEYS]
        ax_u = axs[1][1]
        marks = [ax_u.plot(acq[k][:, 0], acq[k][:, 1], ls='none', zorder=9, clip_on=False, **MARK[k])[0]
                 for k in ('unif', 'u')]
        names = {key: f'({chr(97 + 2 * r + c)}) ' for r, c, _, key in flat}
        titles = {key: [fig.text(0, 0, names[key] + TITLE[key][0], ha='center', va='baseline', fontsize=FS_2X2['title'])]
                  for _, _, _, key in flat}

        def hbar(key, ticks, labels):
            cax = fig.add_axes([.5, .05, .2, .02])
            cb = fig.colorbar(ims[key], cax=cax, orientation='horizontal', ticks=ticks)
            for t, ha in zip(cb.ax.set_xticklabels(labels), ['left'] + ['center'] * (len(ticks) - 2) + ['right']):
                t.set_ha(ha)
            cb.outline.set_linewidth(.4)
            cax.tick_params(labelsize=FS_2X2['tick'], length=1.5, pad=1)
            return cax

        cax_cov = hbar('cov', COV_TICKS_2X2, [f'{v:g}' for v in COV_TICKS_2X2])
        cax_sig = hbar('I', [0, 1], ['0', '1'])
        caxs = [cax_cov, cax_sig]
        def legend_handles(labs):
            lab_u, lab_unif = labs[-2:]
            return [Line2D([], [], color=FILL['boundary'], lw=P.BOUNDARY_LW, label='success/failure boundary'),
                    Line2D([], [], ls='none', label=lab_u, **MARK['u']),
                    Line2D([], [], ls='none', label=lab_unif, **MARK['unif'])]

        options = [(LEG_LABELS[0], lambda h: [h])] + [(labs, lambda h: [h[:1], h[1:]]) for labs in LEG_LABELS] + \
            [(LEG_LABELS[0], lambda h: [[x] for x in h])]
        head = None                                       # the marker row's heading text, when the labels carry one
        for labs, split in options:
            legends = [fig.legend(handles=h_, ncol=len(h_), **paper_legend_kw(FS_2X2['legend']))
                       for h_ in split(legend_handles(labs))]
            if len(labs) == 3:
                head = fig.text(0, 0, labs[0], ha='left', va='bottom', fontsize=FS_2X2['legend'])
            fig.canvas.draw()
            rr_ = fig.canvas.get_renderer()
            widths = [lg.get_window_extent(rr_).width / fig.dpi for lg in legends]
            if head is not None:
                head_w = head.get_window_extent(rr_).width / fig.dpi
                widths[-1] += head_w + LEG_HEAD_PT * PT
            if max(widths) <= W_ - edge['left'] - edge['right']:
                leg_labels = labs
                break
            for lg in legends:
                lg.remove()
            if head is not None:
                head.remove()
                head = None
        else:
            need(False, 'no legend layout fits the print width')
        LEG_ROW_GAP = 2.0 * PT

        # Pass 1: measure.
        fig.canvas.draw()
        rr = fig.canvas.get_renderer()
        inch = lambda bb: (bb.width / fig.dpi, bb.height / fig.dpi)
        pos = lambda a: a.get_position()
        tb = lambda a: a.get_tightbbox(rr)
        left_dec = max(pos(axs[r][0]).x0 * W_ - tb(axs[r][0]).x0 / fig.dpi for r in range(2))
        below = [max(pos(a).y0 * H0 - tb(a).y0 / fig.dpi for a in axs[r]) for r in range(2)]
        above = [max(max(0.0, tb(a).y1 / fig.dpi - (pos(a).y0 + pos(a).height) * H0) for a in axs[r]) for r in range(2)]
        right_over = max(max(0.0, tb(axs[r][1]).x1 / fig.dpi - (pos(axs[r][1]).x0 + pos(axs[r][1]).width) * W_)
                         for r in range(2))
        cb_below = max(pos(c_).y0 * H0 - tb(c_).y0 / fig.dpi for c_ in caxs)
        gap = GAP_PT * PT
        leg_hs = [inch(lg.get_window_extent(rr))[1] for lg in legends]
        leg_h = sum(leg_hs) + LEG_ROW_GAP * (len(legends) - 1)
        x_a = edge['left'] + P.ACCENT_PT * PT + left_dec
        pw = (W_ - edge['right'] - right_over - x_a - gap) / 2
        xs = [x_a, x_a + pw + gap]

        # Titles: a row stays on one line when its two titles fit side by side (each may run into the gap and the
        # outer margin, never past the page); full wording first, then TITLE_SHORT for one or both, else two lines.
        def fits(r, texts):
            for k, t in zip(KEYS_2X2[r], texts):
                titles[k][0].set_text(t)
            fig.canvas.draw()
            w_ = [inch(titles[k][0].get_window_extent(rr))[0] for k in KEYS_2X2[r]]
            centres = [xs[c] + pw / 2 for c in range(2)]
            return (centres[0] + w_[0] / 2 + TITLE_SEP_PT * PT <= centres[1] - w_[1] / 2 and
                    centres[0] - w_[0] / 2 >= edge['left'] and centres[1] + w_[1] / 2 <= W_ - edge['right'])

        levels = [('full', {k: TITLE[k][0] for k in TITLE_SHORT}), ('short', TITLE_SHORT), ('symbol', TITLE_SYMBOL)]
        mode = next((m for m, lv in levels
                     if all(fits(r, [names[k] + lv[k] for k in KEYS_2X2[r]]) for r in range(2))), 'two lines')
        if mode != 'two lines':
            for r in range(2):
                fits(r, [names[k] + dict(levels)[mode][k] for k in KEYS_2X2[r]])
        title_mode, wrapped = [mode] * 2, [mode == 'two lines'] * 2
        for r in range(2):
            if wrapped[r]:
                for key in KEYS_2X2[r]:
                    lines = (names[key] + (TITLE[key][1] or TITLE[key][0])).split('\n')
                    titles[key][0].set_text(lines[0])
                    titles[key] += [fig.text(0, 0, l_, ha='center', va='baseline', fontsize=FS_2X2['title'])
                                    for l_ in lines[1:]]
        fig.canvas.draw()

        def asc_desc(t):
            bb = t.get_window_extent(rr)
            base = fig.transFigure.transform(t.get_position())[1]
            return (bb.y1 - base) / fig.dpi, (base - bb.y0) / fig.dpi

        SKIP = 1.2 * FS_2X2['title'] * PT
        th_ = []
        for r in range(2):
            ts = [titles[k] for k in KEYS_2X2[r]]
            asc0 = max(asc_desc(t[0])[0] for t in ts)
            n_lines = max(len(t) for t in ts)
            desc = max(asc_desc(t[-1])[1] for t in ts)
            th_.append(dict(asc0=asc0, n=n_lines, h=asc0 + (n_lines - 1) * SKIP + desc))
        band = CB_GAP_PT * PT + CB_H + cb_below
        rest = (edge['top'] + leg_h + LEG_GAP_PT * PT + th_[0]['h'] + TITLE_GAP_PT * PT + above[0] + below[0] + band +
                ROW_GAP_PT * PT + th_[1]['h'] + TITLE_GAP_PT * PT + above[1] + below[1] + edge['bottom'])
        ph = (H - rest) / 2
        need(.45 <= ph / pw <= 1.0, ('panel aspect out of range', ph / pw))
        fig.set_size_inches(W_, H)

        # Place, top down.
        y = H - edge['top'] - leg_h - LEG_GAP_PT * PT
        tops = []
        for r in range(2):
            y_head = y
            top = y_head - th_[r]['h'] - TITLE_GAP_PT * PT - above[r]
            tops.append(top)
            for c in range(2):
                axs[r][c].set_position([xs[c] / W_, (top - ph) / H, pw / W_, ph / H])
                lines = titles[KEYS_2X2[r][c]]
                first = th_[r]['n'] - len(lines)
                for k_, t in enumerate(lines):
                    t.set_position(((xs[c] + pw / 2) / W_, (y_head - th_[r]['asc0'] - (first + k_) * SKIP) / H))
            y = top - ph - below[r]
            if r == 0:
                cb_top = y - CB_GAP_PT * PT
                for c, cax in enumerate(caxs):
                    cax.set_position([xs[c] / W_, (cb_top - CB_H) / H, pw / W_, CB_H / H])
                y -= band + ROW_GAP_PT * PT
        # Panel (a)'s two-line title replaces the one-line text the layout was measured with.
        t_a, base = titles['cov'][0], titles['I'][0].get_position()[1] * H
        t_a.set_text(COV_TITLE_LINES[0]); t_a.set_fontsize(COV_TITLE_PT)
        t_a.set_y((base - COV_TITLE_DROP_PT * PT + COV_TITLE_SKIP_PT * PT) / H)
        titles['cov'].append(fig.text(t_a.get_position()[0], (base - COV_TITLE_DROP_PT * PT) / H, COV_TITLE_LINES[1],
                                      ha='center', va='baseline', fontsize=COV_TITLE_PT))
        head_x = None
        for k_, lg in enumerate(legends):
            x_c = W_ / 2
            if head is not None and k_ == len(legends) - 1:       # heading + legend centred as one group
                lw = inch(lg.get_window_extent(rr))[0]
                head_x = (W_ - head_w - LEG_HEAD_PT * PT - lw) / 2
                x_c = head_x + head_w + LEG_HEAD_PT * PT + lw / 2
            lg.set_bbox_to_anchor((x_c / W_, (H - edge['top'] - sum(leg_hs[:k_]) - k_ * LEG_ROW_GAP) / H),
                                  transform=fig.transFigure)
        if head is not None:
            # On the baseline of the plain label 'uniform': both are plain text of one size, so their boxes keep the
            # same descent below the baseline and equal box bottoms mean equal baselines.
            fig.canvas.draw()
            ref = next(t for t in legends[-1].get_texts() if '$' not in t.get_text())
            head.set_position((head_x / W_, ref.get_window_extent(fig.canvas.get_renderer()).y0 / fig.dpi / H))

        # The Reds bar serves (b), (c) and (d): its name sits between its end labels, on their baseline (the digits
        # have no depth, so the bottom of a label's box is its baseline).
        fig.canvas.draw()
        rr = fig.canvas.get_renderer()
        ends = [t for t in drawn_labels(cax_sig) if t.get_text()]
        need(len(ends) == 2, 'Reds bar end labels')
        base = ends[0].get_window_extent(rr).y0
        need(abs(ends[1].get_window_extent(rr).y0 - base) < .5, 'end labels on different baselines')
        sig_name = fig.text((xs[1] + pw / 2) / W_, base / fig.dpi / H, r'$I$, $D$, $U$', ha='center', va='baseline',
                            fontsize=FS_2X2['bar'])

        # Checks.
        fig.canvas.draw()
        rr = fig.canvas.get_renderer()

        def drawn(axis, lim):
            lo, hi = sorted(lim)
            return [t for t, v in zip(axis.get_ticklabels(), axis.get_majorticklocs())
                    if t.get_text() and t.get_visible() and lo - 1e-9 <= v <= hi + 1e-9]

        all_ax = [a for row in axs for a in row]
        free = [(t.get_text(), t.get_window_extent(rr)) for t in fig.texts] + \
            [(f'legend row {k_}', lg.get_window_extent(rr)) for k_, lg in enumerate(legends)]
        owned = []
        for a in all_ax + caxs:
            owned += [(t.get_text(), t.get_window_extent(rr)) for t in drawn(a.xaxis, a.get_xlim()) +
                      drawn(a.yaxis, a.get_ylim())]
            owned += [(lab.get_text(), lab.get_window_extent(rr)) for lab in (a.xaxis.label, a.yaxis.label)
                      if lab.get_text()]
        boxes = free + owned
        need(sum(len(drawn(a.xaxis, a.get_xlim())) for a in all_ax) == 6 and
             sum(len(drawn(a.yaxis, a.get_ylim())) for a in all_ax) == 2 * sum(map(bool, Y_TICKLABELS_2X2)),
             'tick labels: 3 x per bottom panel, Y_TICKLABELS_2X2 per left panel')
        for i, (ta, a) in enumerate(boxes):
            for tbx, b in boxes[i + 1:]:
                need(not a.overlaps(b), ('text overlap', ta, tbx))
            slack = .02 * fig.dpi              # box padding beyond the ink; main_2x2 checks the ink itself
            need(a.x0 >= -slack and a.x1 <= fig.bbox.x1 + slack and a.y0 >= -slack and a.y1 <= fig.bbox.y1 + slack,
                 ('text off the page', ta))
        panels = [a.get_window_extent(rr) for a in all_ax + caxs]
        for ta, a in boxes:
            need(not any(a.overlaps(p) for p in panels), ('text over a panel', ta))
        for cl in circles:                                # each circle inside its panel, clear of text
            cb_ = cl.get_window_extent(rr)
            pb = cl.axes.get_window_extent(rr)
            need(pb.x0 <= cb_.x0 and cb_.x1 <= pb.x1 and pb.y0 <= cb_.y0 and cb_.y1 <= pb.y1, 'a circle leaves its panel')
            need(not any(cb_.overlaps(b) for _, b in boxes), 'a circle touches text')
        for m in marks:                                   # markers may leave their panel, never touch text
            mb = m.get_window_extent(rr)
            need(not any(mb.overlaps(b) for _, b in boxes), 'a marker touches text')
            need(mb.x1 <= fig.bbox.x1 - .5 and mb.y1 <= fig.bbox.y1 - .5, 'a marker leaves the page')
        need(P.PDF_DPI * pw >= P.GRID_SHAPE[0] and P.PDF_DPI * ph >= P.GRID_SHAPE[1], 'a cell is lost in the raster')
        for _, _, ax, key in flat:
            im = ax.get_images()
            need(len(im) == 1 and np.allclose(im[0].get_extent(), ext), f'{key}: image extent')
            need(np.array_equal(np.ma.getdata(im[0].get_array()), field[key].T), f'{key}: image is not the field')
        for k_, lg in enumerate(legends):
            lb = lg.get_window_extent(rr)
            x0 = lb.x0
            if head is not None and k_ == len(legends) - 1:
                hb = head.get_window_extent(rr)
                x0 = hb.x0
                need(hb.x1 < lb.x0 and abs(hb.y0 - ref.get_window_extent(rr).y0) < .05, 'legend heading misplaced')
            need(abs((x0 + lb.x1) / 2 / fig.dpi - W_ / 2) < 0.005, 'legend not centred')
    layout = dict(width=W_, height=H, panel_w=pw, panel_h=ph, edge=dict(edge), col_gap=gap, row_band=band + ROW_GAP_PT * PT,
                  titles_wrapped=wrapped, title_mode=title_mode, legend_rows=len(legends), legend_labels=leg_labels,
                  titles={k: ' / '.join(t.get_text() for t in v) for k, v in titles.items()},
                  cell_pt=(72 * pw / P.GRID_SHAPE[0], 72 * ph / P.GRID_SHAPE[1]))
    return fig, layout


def main_2x2():
    g = P.load_full_grid(ROUND)
    acq = acquired()
    OUT.mkdir(parents=True, exist_ok=True)
    pdf, png = OUT / f'{STEM}_2x2.pdf', OUT / f'{STEM}_2x2.png'
    edge = dict(left=.01, right=.01, top=.01, bottom=.01)
    for it in range(6):
        fig, L = draw_2x2(g, acq, edge)
        with plt.rc_context(style_rc(**RC_2X2)):
            fig.savefig(pdf, dpi=P.PDF_DPI, metadata={'CreationDate': None})
            fig.savefig(png, dpi=P.PNG_DPI)
        plt.close(fig)
        ink = ink_margins(png)
        if all(INK_LO <= x <= INK_HI for x in ink.values()):
            break
        edge = {k: edge[k] + INK_TARGET - ink[k] for k in edge}
    else:
        need(False, ('2x2 ink margins do not settle', ink))
    w, h = P._pdf_size(pdf)
    need(abs(w - W_2X2) <= .002 and abs(h - H_2X2) <= .002, f'2x2: PDF {w:.3f} x {h:.3f}')
    txt = (f'{STEM}_2x2: {w:.3f} x {h:.3f} in (page fixed by the paper); panels {L["panel_w"]:.3f} x {L["panel_h"]:.3f} '
           f'in (aspect {L["panel_h"] / L["panel_w"]:.4f}; Figure 5 and the 1x4 use {P.ASPECT:.4f}); column gap '
           f'{L["col_gap"]:.3f} in; row band (colour bars) {L["row_band"]:.3f} in; cell {L["cell_pt"][0]:.2f} x '
           f'{L["cell_pt"][1]:.2f} pt; legend in {L["legend_rows"]} row(s), marker labels {L["legend_labels"]}; title rows '
           f'{L["title_mode"]}: ' + '; '.join(L['titles'].values()) + '; '
           f'ink margins (in) ' + ', '.join(f'{k} {x:.3f}' for k, x in ink.items()) +
           f' after {it + 1} layout pass(es).\n'
           f'Text: titles, axis labels, legend and the bar name {FS_2X2["title"]} pt; tick labels (panels and bars) '
           f'{FS_2X2["tick"]} pt (caption: 9 pt footnotesize).\n'
           '(a) distance to training data, (b) I(x) top; (c) D(x), (d) U(x) with the 35 next initial states bottom. Same '
           'data, checkpoint, markers and boundary as the 1x4 / 1x5; the numbers are in pendulum_coverage_acquisition.txt.\n'
           'theta-dot label and y tick labels on the left column only; theta label and x tick labels on the bottom row '
           'only. Colour bars under the top row: teal under (a) for distance on [0, 0.06]; under (b) the one Reds bar on '
           '[0, 1] shared by (b), (c) and (d), named I, D, U between its end labels.\n')
    (OUT / f'{STEM}_2x2.txt').write_text(txt)
    print(txt, end='')
    print(f'-> {pdf}')


def ink_margins(png):
    from PIL import Image
    a = np.asarray(Image.open(png).convert('L')) < 200
    rows, cols = np.where(a.any(1))[0], np.where(a.any(0))[0]
    ink = dict(left=cols[0], right=a.shape[1] - 1 - cols[-1], top=rows[0], bottom=a.shape[0] - 1 - rows[-1])
    return {k: v / P.PNG_DPI for k, v in ink.items()}


def main():
    g = P.load_full_grid(ROUND)
    acq = acquired()
    OUT.mkdir(parents=True, exist_ok=True)
    U = g['I'] + g['D']
    g_u = dict(g, U=U)
    out, layouts = {}, {}
    for v in VARIANTS:
        fig, L = draw(g, acq, v)
        pdf, png = OUT / f'{STEM}_{v}.pdf', OUT / f'{STEM}_{v}.png'
        with plt.rc_context(style_rc()):
            fig.savefig(pdf, dpi=P.PDF_DPI, metadata={'CreationDate': None})
            fig.savefig(png, dpi=P.PNG_DPI)
        plt.close(fig)
        w, h = P._pdf_size(pdf)
        need(abs(w - W) <= .01, f'{v}: PDF width {w:.3f}')
        L['ink'] = ink_margins(png)
        need(all(.5 / P.PNG_DPI <= x <= .06 for x in L['ink'].values()), (v, 'ink margins', L['ink']))
        layouts[v], out[v] = L, (pdf, png)
        print(f'{v}: {w:.3f} x {h:.3f} in, panels {L["panel_w"]:.3f} x {L["panel_h"]:.3f} in -> {pdf}')

    # numbers behind the figure
    i, f = acq['info'], g['info']
    d = g['d_train']
    dtr_tree = usd.tree(usd.Pool().states([int(x) for x in (usd.runs()['adaptive', SEED]['dir'] /
                                                              'datasets/train_trajectories.txt').read_text().split()][:120]))
    acq_dtr = {k: dtr_tree.query(usd.norm(acq[k]))[0] for k in ('u', 'unif')}
    gU = {k: cell_values(g_u, acq[k], 'U') for k in ('u', 'unif')}
    gI = {k: cell_values(g_u, acq[k], 'I') for k in ('u', 'unif')}
    cu = cell_values(g_u, i['candidate_states'], 'U')
    q = lambda v: ', '.join(f'{x:.3f}' for x in np.quantile(v, [0, .5, 1]))
    lines = [
        f'{STEM}: extended Figure 5 previews, {i["run"]} (TERA-FM, U, rho 0.75), seed {SEED}, round {ROUND} '
        f'({f["n_train"]} training trajectories, {f["n_train_states"]} visited states), full 158 x 315 grid.',
        '',
        'Layouts (in): ' + '; '.join(f'{v} {L["width"]:.3f} x {L["height"]:.3f}, panels {L["panel_w"]:.3f} x '
                                     f'{L["panel_h"]:.3f}, titles {"wrapped" if L["titles_wrapped"] else "one line"}, '
                                     f'ink margins ' + ', '.join(f'{k} {x:.3f}' for k, x in L['ink'].items())
                                     for v, L in layouts.items()),
        'Panels: v1 = (a) distance to training data, (b) I(x), (c) D(x), (d) U(x) = I(x) + D(x) with the 35 next initial '
        'states; v2 = (a) abstention (as Figure 5 A) then the same four as (b)-(e). True success/failure boundary '
        f'(black, {P.BOUNDARY_LW} pt) in every panel. I, D, U share one [0, 1] scale (white to {SIGNAL_RAMP["I"][-1]}, '
        f'ColorBrewer Reds); distance on [0, {COV_MAX}] (white to {SIGNAL_RAMP["D"][-1]}, BrBG teal half); markers: '
        'chosen by U = black filled circle with a white rim, drawn uniformly = white open triangle with a black rim.',
        '',
        'DISTANCE TO TRAINING DATA (panel a in v1, b in v2)',
        f'  d = distance from each grid state to the nearest of the {f["n_train_states"]} states on the first 120 '
        'training trajectories (every state, last included), coordinates (theta / 2pi wrapped, omega / 4pi), periodic in '
        'theta (uncertainty_signals_data.norm / tree; on the test cells it equals analyse()["d_train"] exactly).',
        f'  grid quantiles 0/50/90/99/100%: ' + ', '.join(f'{x:.4f}' for x in np.quantile(d, [0, .5, .9, .99, 1])) +
        f'; d > 0.02 (sparse) on {(d > .02).mean() * 100:.1f}% of cells; d > {COV_MAX} (saturated at the top of the '
        f'scale) on {(d > COV_MAX).sum()} cells ({(d > COV_MAX).mean() * 100:.2f}%).',
        '',
        'ACQUIRED INITIAL STATES (next batch, trajectories 121-155)',
        '  Index space: acquisition.d1_indices and d2_indices are POOL POSITIONS, i.e. positions in the seed-99512 '
        'shuffled pool order (splits_det_rerun/pool_order_99512.txt); trajectory id = pool_order[position], the row of '
        'splits_det_rerun/train.npz. They are the same space as the VALUES of acquisition_scores.npz candidate_indices, '
        'not positions in that array.',
        f'  dataset_builder_state.json train_indices[120:155] == d1 + d2 (8 uniform first, then the 27 by U in pick '
        f'order), and datasets/train_trajectories.txt lines 121-128 are the 8 uniform trajectories, lines 129-155 the 27 '
        f'by U (ids = pool_order[d1 + d2]); checked. Lines 1-120 are the round-2 training set.',
        f'  Counts: 27 chosen by U (d2), 8 drawn uniformly from the unused pool (d1); no shortfall fill.',
        f'  Candidates: all {i["n_candidates"]} unused pool states (the pool has {i["n_pool"]}; 120 used + 8 uniform '
        f'removed first). The configured n_candidates ({i["n_candidates_config"]}) exceeds the pool, so every unused '
        f'state was scored. U from K = 100 samples each; mean U {i["score_mean"]:.3f}; {i["n_candidates_u_ge_0_5"]} '
        f'candidates with U >= 0.5.',
        '',
        'TIE RULE (verdict_scores.select_topk with tie_tol > 0 and spread_farthest_point, commit acae65bf; reproduced '
        'here from the stored scores, giving d2_indices exactly and in order)',
        f'  The 27th-highest U is {i["u_kth"]:.0f}. Every candidate within tie_tol = {i["tie_tol"]} of it counts as tied: '
        f'{i["n_band"]} candidates with U >= 0.999 ({i["n_band_u_eq_1"]} at exactly 1: {i["band_all_invalid"]} with all '
        f'100 samples invalid, {i["band_even_split"]} with valid samples split evenly between success and failure). '
        f'{i["n_above"]} candidates score above the band, so all 27 come from it.',
        '  Within the band the 27 are chosen by greedy farthest-point selection in the model\'s input coordinates '
        '(sin theta, cos theta, omega / 2pi): each pick is the tied candidate farthest from its nearest already-collected '
        'initial state (the 120 training trajectories and this round\'s 8 uniform states) or earlier pick; exact ties '
        'go to the lower candidate position.',
        f'  Picks: {i["picks_u_eq_1"]} at U = 1 exactly ({i["picks_all_invalid"]} all-invalid, {i["picks_even_split"]} '
        f'even split), the rest at U >= {i["picks_u_min"]:.4f}. Distance (input coordinates) to the nearest collected '
        f'start: band min/median/max {", ".join(f"{x:.3f}" for x in i["band_anchor_dist"])}; picks '
        f'{", ".join(f"{x:.3f}" for x in i["picks_anchor_dist"])}.',
        '  Caption-ready: "The 27 states chosen by U all score within 10^-3 of the maximum, U = 1; among the 159 such '
        'candidates, the selection takes the ones farthest from already collected initial states."',
        '',
        'MARKERS AGAINST THE DRAWN FIELDS',
        '  The U field in the figure is the grid evaluation (stored test/cal/val predictions, else one fresh K = 100 pass '
        'with the round-2 checkpoint), a separate Monte Carlo draw from the acquisition\'s own scoring; values below are '
        'the grid cell nearest each acquired state.',
        f'  grid U at the 27 by U: min/median/max {q(gU["u"])}; at the 8 uniform: {q(gU["unif"])}.',
        f'  grid I at the 27 by U: {q(gI["u"])}; at the 8 uniform: {q(gI["unif"])}.',
        f'  stored candidate U vs grid U at the nearest cell, all {i["n_candidates"]} candidates: Pearson r '
        f'{np.corrcoef(i["candidate_u"], cu)[0, 1]:.3f}, mean |diff| {np.abs(i["candidate_u"] - cu).mean():.3f}.',
        f'  distance to training data (d above) at the 27 by U: {q(acq_dtr["u"])}; at the 8 uniform: '
        f'{q(acq_dtr["unif"])}; sparse (> 0.02): {(acq_dtr["u"] > .02).sum()} of 27, {(acq_dtr["unif"] > .02).sum()} '
        'of 8.',
        '',
        f'Abstention panel (v2 only): as Figure 5 A; full grid abstains on {f["full"]["abstain"]} cells '
        f'({100 * f["full"]["abstain"] / P.GRID_SHAPE[0] / P.GRID_SHAPE[1]:.2f}%); test Abstain {g["info"]["test_counts"][2]} '
        f'of {sum(g["info"]["test_counts"])}.',
    ]
    (OUT / f'{STEM}.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument('--layout', choices=['rows', '2x2'], default='rows',
                    help='rows: the 1x4 (v1) and 1x5 (v2) previews and the numbers txt; 2x2: the half-width 2 x 2 only')
    a = ap.parse_args()
    main_2x2() if a.layout == '2x2' else main()
