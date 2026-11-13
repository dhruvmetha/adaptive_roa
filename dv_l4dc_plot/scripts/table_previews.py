r"""Layout previews of Table 1 (matrix_uncertified) for Dhruv to choose from (2026-11-12). Never copied
to the paper; paper_writer installs a chosen one by hand.

Tables print means only (Dhruv, 2026-11-12), so neither preview shows a range. Shared rules: one decimal,
no vertical rules (no \tablepairdivider or \tablesystemdivider), booktabs rules only, headers in the method
section's notation, text-mode cells (digits in the body font). The best displayed value per setting and
metric is judged at the displayed 1 dp, ties included, never Abst. A seed count below three follows the mean
as a superscript; -- means no seed contributes.

  clean    B: today's wide layout (two panels, 16 columns), best mean in bold.
  shaded   D: B without bold; the best cell gets \cellcolor{black!10}, the two FM rows \rowcolor{blue!4}.
           Needs \usepackage{colortbl}.

Retired 2026-11-12, files in tables/archive/: A (long_lqr, long_ppo) and C (clean_ranges), which printed
[min, max] ranges.
"""
from common import SETTINGS, spread

HEADS = {'pen-lqr': 'Pendulum', 'cp-lqr': 'Cartpole', 'cp-ppo': 'Cartpole', 'q2d-ppo': 'Planar quadrotor',
         'q3d-lqr': 'Spatial quadrotor', 'q3d-ppo': 'Spatial quadrotor'}
PANELS = [('LQR controllers', 'lqr', ['pen-lqr', 'cp-lqr', 'q3d-lqr']),
          ('RL controllers', 'ppo', ['cp-ppo', 'q2d-ppo', 'q3d-ppo'])]
METRICS = [('rec', r'$R_{\mathrm{s}}\uparrow$'), ('fdr', r'FDR$\downarrow$'),
           ('fail_rec', r'$R_{\mathrm{f}}\uparrow$'), ('for', r'FOR$\downarrow$'), ('abstain', r'Abst.$\downarrow$')]
FM_ROLES = ('uniform data', 'method')     # the two FM rows, tinted in D


def stats(rows, groups):
    """(setting, rowlabel, metric) -> dict(n, mean, best), best judged at the displayed 1 dp."""
    out = {}
    for setting in SETTINGS:
        for _, models in groups:
            for rowlabel, role, model in models:
                rs = [r for r in rows if (r['setting'], r['matrix_role'], r['model']) == (setting, role, model)]
                assert len(rs) == len({r['seed'] for r in rs}) == 3
                for key, _ in METRICS:
                    v = [100 * r['test'][key] for r in rs if r['test'][key] is not None]
                    mean, _, _ = spread(v)
                    out[setting, rowlabel, key] = dict(n=len(v), mean=mean, role=role, best=False)
    for setting in SETTINGS:
        for key, _ in METRICS:
            if key == 'abstain':
                continue
            shown = [float(f"{c['mean']:.1f}") for (s, _, k), c in out.items() if (s, k) == (setting, key) and c['n']]
            target = (max if key in ('rec', 'fail_rec') else min)(shown)
            for (s, _, k), c in out.items():
                if (s, k) == (setting, key) and c['n'] and float(f"{c['mean']:.1f}") == target:
                    c['best'] = True
    return out


def mean_text(c, bold=True):
    m = f"{c['mean']:.1f}"
    if bold and c['best']:
        m = r'\textbf{' + m + '}'
    if c['n'] < 3:
        m += r'\textsuperscript{(' + str(c['n']) + ')}'
    return m


def head(name):
    return ['% Generated table body (column headers and numbers only). Edit the generator, not this file.',
            '% The table environment, caption and label are in sections/06_experiments.tex.',
            f'% PREVIEW, not in the paper: Table 1 layout {name} (see dv_l4dc_plot/scripts/table_previews.py).']


def wide_body(st, groups, variant):
    """B/D: today's two-panel wide layout without vertical rules; settings separated by 12pt gaps."""
    columns = r'l@{\hspace{6pt}}' + r'@{\hspace{12pt}}'.join(['rrrrr'] * 3)
    t = head({'clean': 'B, wide, means only',
              'shaded': 'D, wide, best cell shaded, FM rows tinted (needs colortbl)'}[variant])
    for pi, (panel, _, settings) in enumerate(PANELS):
        if pi:
            t.append(r'\par\medskip')
        t += [r'\resizebox{\ifdim\width>\textwidth\textwidth\else\width\fi}{!}{%', r'\begin{tabular}{' + columns + '}',
              r'\toprule', r'\multicolumn{16}{c}{\textbf{' + panel + r'}} \\', r'\midrule',
              ' & ' + ' & '.join(r'\multicolumn{5}{c}{' + HEADS[s] + '}' for s in settings) + r' \\',
              r'\cmidrule(lr){2-6}\cmidrule(lr){7-11}\cmidrule(lr){12-16}',
              r'\textbf{Method} & ' + ' & '.join(label for _ in settings for _, label in METRICS) + r' \\', r'\midrule']
        for gi, (group, models) in enumerate(groups):
            if gi:
                t.append(r'\midrule')
            t.append(r'\multicolumn{16}{l}{\emph{' + group + r'}} \\')
            for rowlabel, role, _ in models:
                cells = []
                for s in settings:
                    for key, _ in METRICS:
                        c = st[s, rowlabel, key]
                        if not c['n']:
                            cells.append('--')
                        elif variant == 'clean':
                            cells.append(mean_text(c))
                        else:
                            cells.append((r'\cellcolor{black!10}' if c['best'] else '') + mean_text(c, bold=False))
                prefix = r'\rowcolor{blue!4}' if variant == 'shaded' and role in FM_ROLES else ''
                t.append(prefix + rowlabel + ' & ' + ' & '.join(cells) + r' \\')
        t += [r'\bottomrule', r'\end{tabular}}']
    return t


def previews(rows, groups):
    """name -> body lines for every Table 1 preview."""
    st = stats(rows, groups)
    out = {}
    for variant in ('clean', 'shaded'):
        out[f'matrix_uncertified_body_{variant}'] = wide_body(st, groups, variant)
    return out
