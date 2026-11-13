"""ARCHIVED 2026-11-12: the half-width, four-setting Fig 5 with abstention bars.
Replaced in the paper by ../plot_acquisition_strategy_bars.py. Writes only to plots/archive and
values/archive, never to the paper.


Ported from the 'strategy' branch of
deterministic-verification/results-discussion-assets/plot_adaptive_comparisons.py (2026-11-12)
with the plotting code unchanged. Bars for U_H, q_A and U_tau show R_s, R_f and abstention, one
setting per physical system (PPO where available, LQR for the pendulum). Soft four-threshold rule
fit on validation at w=0.2, final-budget test rates, three-seed mean and sample SD.
"""
import hashlib, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/, for common
import numpy as np
import matplotlib.pyplot as plt
from common import ASSETS, VALUES, SETTINGS, TITLES, BLUE, ORANGE, GRAY, parse_args, save

SRC = ASSETS / 'soft_four_wsweep.jsonl'
BAR_SETTINGS = ['pen-lqr', 'cp-ppo', 'q2d-ppo', 'q3d-ppo']
COMBOS = [('U_H', .75), ('q_A', .75), ('U_tau', .75)]
METRICS = [('rec', r'$R_{\mathrm{s}}$ ↑', BLUE), ('fail_rec', r'$R_{\mathrm{f}}$ ↑', ORANGE), ('abstain', 'Abst. ↓', GRAY)]


def arm(r):
    return r['acq'].replace('τ', 'tau')


def main(paper):
    final = [r for l in SRC.read_text().splitlines() if (r := json.loads(l))['w'] == .2 and r['model'] == 'FM']
    assert len(final) == 108

    def group(s, a, rho, field):
        rr = [r for r in final if (r['setting'], arm(r), r['d2_ratio']) == (s, a, rho)]
        assert len(rr) == 3 and len({r['seed'] for r in rr}) == 3
        vv = np.array([r['test'][field] for r in rr]) * 100
        return vv.mean(), vv.std(ddof=1)

    records = []
    rc = {'font.family': 'DejaVu Sans', 'font.size': 7.5, 'axes.titlesize': 7.5, 'axes.labelsize': 7.5,
          'legend.fontsize': 7.5, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'pdf.fonttype': 42,
          'ps.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': .65}
    with plt.rc_context(rc):
        fig, axs = plt.subplots(2, 2, figsize=(3.17, 2.08), sharey=True)
        for ax, s in zip(axs.flat, BAR_SETTINGS):
            ax.set_ylim(-3, 103); ax.set_yticks([0, 50, 100]); ax.grid(axis='y', alpha=.18, lw=.5); ax.set_axisbelow(True)
            ax.set_ylim(-8, 108); ax.axhline(0, color='#555555', lw=.5)
            ax.set_title(TITLES[SETTINGS.index(s)].split(',')[0], pad=3)
            xx = np.arange(len(COMBOS)); width = .23
            for mi, (field, label, color) in enumerate(METRICS):
                v = [group(s, a, r, field) for a, r in COMBOS]
                ax.bar(xx + (mi - 1) * width, [a for a, b in v], width, color=color, label=label, edgecolor='white',
                       linewidth=.25, yerr=[b for a, b in v],
                       error_kw={'elinewidth': .65, 'capsize': 1.25, 'capthick': .65, 'ecolor': '#333333'}, zorder=3)
                for (a, rho), (mean, sd) in zip(COMBOS, v):
                    records.append(dict(setting=s, arm=a, rho=rho, metric=field, n=3, mean=float(mean), sd=float(sd)))
            ax.set_xticks(xx, [r'$U_H$', r'$q_A$', r'$U_\tau$'])
        for ax in axs[0]:
            ax.tick_params(labelbottom=False)
        fig.supxlabel(r'Acquisition score ($\rho=0.75$)', fontsize=7.5, y=.015)
        fig.supylabel('Test (%)', fontsize=7.5, x=.008)
        fig.legend(*axs.flat[0].get_legend_handles_labels(), loc='upper center', ncol=3, frameon=False,
                   bbox_to_anchor=(.55, 1.015), handlelength=1, columnspacing=.8)
        fig.subplots_adjust(left=.15, right=.99, top=.78, bottom=.18, hspace=.5, wspace=.19)
        size = save(fig, 'archive/acquisition_strategy_abstain', paper_stem=None, paper=False)
    VALUES.mkdir(exist_ok=True)
    (VALUES / 'archive/acquisition_strategy_abstain_values.json').write_text(json.dumps(dict(
        source=str(SRC), source_sha256=hashlib.sha256(SRC.read_bytes()).hexdigest(),
        protocol='FM, rho=0.75, soft four-threshold rule fit on validation at w=0.2, final-budget test, '
                 'mean and sample SD over 3 seeds', page_size=size, values=records), indent=2) + '\n')


if __name__ == '__main__':
    main(parse_args(__doc__).paper)
