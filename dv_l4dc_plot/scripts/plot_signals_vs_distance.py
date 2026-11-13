"""Appendix preview: raw I(x) against training coverage and D(x) against distance to the boundary, TERA round 2, 3 seeds.

Requested 2026-11-13 by Sumanth via paper_writer_2. Runs: the three TERA-FM seeds (x231 r1 s99512, r2 s33407,
r3 s24475; U, rho 0.75) at round 2, the checkpoint of Figure 5. States: the 20,702 test states with their stored
round-2 predictions (epoch_002/full_roa_per_point.npz). Everything per state comes from
uncertainty_signals_data.analyse, which also checks each run's pool copy and its stored decisions:
  left   I = p_inv against d_train, the distance to the nearest state on the run's own first 120 training trajectories
  right  D = (1 - p_inv) H2(p_s / (p_s + p_f)) against d_opp, the distance to the nearest test state with the opposite
         true label (an upper bound on the distance to the boundary)
Both distances in (theta / 2pi wrapped, omega / 4pi). Binned means per seed; the line is the mean of the three seeds'
bin means and the band their min-max (common.spread). Each point sits at the mean distance of its bin's states (pooled
over seeds). Dashed lines: the paper's sparse cutoff 0.02 (left) and near-boundary cutoff 0.01 (right).

Writes plots/variants/pendulum_signals_vs_distance.{pdf,png,txt}; nothing goes to the paper.
"""
import numpy as np
import plot_uncertainty_signals_paper as P
import uncertainty_signals_data as usd
from common import PLOTS, FS, INK, style_rc, spread, SIGNAL_RAMP, plt

STEM = 'pendulum_signals_vs_distance'
OUT = PLOTS / 'variants'
W, H = 6.0, 2.0
ROUND = 2
EDGES_L = np.r_[np.arange(0, .0501, .005), np.inf]                       # d_train; the last bin is open
EDGES_R = np.array([0, .004, .008, .012, .016, .02, .03, .04, .06, .08, .12, .16, .24, np.inf])   # d_opp
SHOW_NEAR_LINE = True
LINE_C, BAND_C = SIGNAL_RAMP['I'][-2], SIGNAL_RAMP['I'][2]
need = P.need


def load():
    pool, ep, runs = usd.Pool(), usd.epochs(), usd.runs()
    out = {}
    for seed in usd.SEEDS:
        run = runs['adaptive', seed]
        ids, _ = usd.verify_pool(pool, run)
        a = usd.analyse(run, seed, ROUND, ep, pool, ids)
        need(a['arrays'] is not None and a['stats']['n_train'] == 120 and a['stats']['n_test'] == 20702,
             f'{run["name"]} round {ROUND}')
        out[seed] = dict(run=run['name'], **a['arrays'])
    return out


def binned(A, xkey, ykey, edges, mask=None):
    """Per-seed bin means and counts; the plotted line (seed mean), band (seed min-max) and x (pooled mean distance)."""
    nb = len(edges) - 1
    per = {}
    for s, a in A.items():
        m = np.ones(len(a[xkey]), bool) if mask is None else mask(a)
        b = np.digitize(a[xkey][m], edges) - 1
        n = np.bincount(b, minlength=nb)[:nb]
        mu = np.array([a[ykey][m][b == k].mean() if n[k] else np.nan for k in range(nb)])
        sx = np.bincount(b, weights=a[xkey][m], minlength=nb)[:nb]
        per[s] = dict(n=n, mean=mu, sx=sx)
    n_all = sum(p['n'] for p in per.values())
    x = sum(p['sx'] for p in per.values()) / np.where(n_all > 0, n_all, 1)
    rows = [spread([per[s]['mean'][k] for s in per if per[s]['n'][k]]) for k in range(nb)]
    ok = np.array([all(per[s]['n'][k] > 0 for s in per) for k in range(nb)])
    return dict(x=x, mean=np.array([r[0] for r in rows], float), lo=np.array([r[1] for r in rows], float),
                hi=np.array([r[2] for r in rows], float), per=per, ok=ok, edges=edges)


def panel(ax, B, cut):
    ok = B['ok']
    ax.fill_between(B['x'][ok], B['lo'][ok], B['hi'][ok], color=BAND_C, lw=0, alpha=.8, zorder=1)
    ax.plot(B['x'][ok], B['mean'][ok], color=LINE_C, lw=1.1, marker='o', ms=2.6, mew=0, zorder=3)
    if cut is not None:
        ax.axvline(cut, color=INK, lw=.6, ls=(0, (3, 2)), zorder=2)


def main():
    A = load()
    BL = binned(A, 'd_train', 'I', EDGES_L)
    BR = binned(A, 'd_opp', 'D', EDGES_R)
    need(BL['ok'].all() and BR['ok'].all(), 'a bin is empty for some seed')
    # near / far split of the left panel (reported, not drawn)
    near = binned(A, 'd_train', 'I', EDGES_L, mask=lambda a: a['d_opp'] < usd.NEAR)
    far = binned(A, 'd_train', 'I', EDGES_L, mask=lambda a: a['d_opp'] >= usd.NEAR)
    with plt.rc_context(style_rc()):
        fig, axs = plt.subplots(1, 2, figsize=(W, H))
        panel(axs[0], BL, usd.SPARSE)
        panel(axs[1], BR, usd.NEAR if SHOW_NEAR_LINE else None)
        axs[0].set_xlim(0, None); axs[0].set_ylim(0, 1)
        axs[0].set_xticks([0, .02, .04, .06]); axs[0].set_xticklabels(['0', '0.02', '0.04', '0.06'])
        axs[0].set_yticks([0, .5, 1]); axs[0].set_yticklabels(['0', '0.5', '1'])
        axs[1].set_xscale('log')
        axs[1].set_xlim(.0015, .4)
        axs[1].set_xticks([.003, .01, .03, .1, .3]); axs[1].set_xticklabels(['0.003', '0.01', '0.03', '0.1', '0.3'])
        axs[1].minorticks_off()
        axs[1].set_ylim(0, None)
        axs[1].set_yticks([0, .1, .2]); axs[1].set_yticklabels(['0', '0.1', '0.2'])
        axs[0].set_xlabel('distance to nearest training state', labelpad=1)
        axs[1].set_xlabel('distance to nearest opposite-outcome test state', labelpad=1)
        axs[0].set_ylabel(r'mean $I(x)$', labelpad=2)
        axs[1].set_ylabel(r'mean $D(x)$', labelpad=2)
        axs[0].set_title(r'(a) invalid mass and training coverage', fontsize=FS['title'], pad=3)
        axs[1].set_title(r'(b) weighted disagreement and boundary distance', fontsize=FS['title'], pad=3)
        fig.tight_layout(pad=.15, w_pad=1.2)
        OUT.mkdir(parents=True, exist_ok=True)
        pdf, png = OUT / f'{STEM}.pdf', OUT / f'{STEM}.png'
        fig.savefig(pdf, metadata={'CreationDate': None})
        fig.savefig(png, dpi=P.PNG_DPI)
        plt.close(fig)
    w, h = P._pdf_size(pdf)
    need(abs(w - W) < .01, f'width {w}')

    def table(B, xname, yname, label):
        e = B['edges']
        out = [f'{label}: bins of {xname} (edges), x = pooled mean {xname} of the bin, {yname} mean over seeds '
               f"[seed min, max], n per seed ({', '.join(str(s) for s in B['per'])})"]
        for k in range(len(e) - 1):
            hi = 'inf' if np.isinf(e[k + 1]) else f'{e[k + 1]:.3f}'
            ns = ', '.join(str(int(B['per'][s]['n'][k])) for s in B['per'])
            per = ', '.join(f"{B['per'][s]['mean'][k]:.3f}" if B['per'][s]['n'][k] else '--' for s in B['per'])
            x, mu, lo, hi_ = B['x'][k], B['mean'][k], B['lo'][k], B['hi'][k]
            out.append(f'  [{e[k]:.3f}, {hi}) x {x:.4f}: {mu:.4f} [{lo:.4f}, {hi_:.4f}]  per seed {per}  n {ns}')
        return out

    seam = int((A[usd.SEEDS[0]]['d_opp'] < .001).sum())
    runs_txt = ', '.join(f"{s} ({a['run']})" for s, a in A.items())
    lines = [f'{STEM}: TERA-FM (U, rho 0.75) round 2, seeds {runs_txt}; '
             f'20,702 test states each, stored epoch_002 predictions. Size {w:.2f} x {h:.2f} in.',
             'Distances in (theta / 2pi wrapped, omega / 4pi) (uncertainty_signals_data.norm). Line = mean of the '
             'three per-seed bin means; band = their min-max. Dashed: sparse cutoff 0.02 (left), near-boundary 0.01 '
             '(right). The right panel uses a log distance axis. Its first bin holds the states one omega cell (0.0032) '
             f'from an opposite-outcome state plus {seam} on the theta = +-pi wrap seam at 0.0005 (the test lattice\'s '
             'first and last theta columns are 0.0032 rad apart). The last left bin is open, [0.05, inf).',
             'd_opp is the same for every seed (same test states and labels), so its n per bin repeats.', '']
    lines += table(BL, 'd_train', 'I', 'LEFT (a)') + ['']
    lines += table(BR, 'd_opp', 'D', 'RIGHT (b)') + ['']
    near_pct = np.mean([(a['d_opp'] < usd.NEAR).mean() for a in A.values()]) * 100
    lines += ['CAVEAT CHECK: does I rising toward the boundary show up as a bump in (a)?',
              f'  Near-boundary states (d_opp < {usd.NEAR}) are {near_pct:.1f}% '
              'of test states. Their I is higher at every coverage level, but removing them changes the curve by at '
              'most the amounts below, so they do not shape it.']
    lines += table(near, 'd_train', 'I', '  near boundary only') + table(far, 'd_train', 'I', '  far from boundary only')
    dmax = np.nanmax(np.abs(far['mean'] - BL['mean']))
    lines += [f'  max |far-only mean - all mean| over bins: {dmax:.4f}']
    (OUT / f'{STEM}.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print(f'{STEM}: {w:.2f} x {h:.2f} in -> {pdf}')


if __name__ == '__main__':
    main()
