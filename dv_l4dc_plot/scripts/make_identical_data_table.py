r"""Certified identical-data table body: tables/identical_data_certified_body.tex, paper Table 4 (tab:identical_data,
sections/06_results.tex) since paper commit 7a21aa8. It replaced the dot plot predictor_comparison_uniform_certified,
now in scripts/archive.

Data: the 'cal' records of predictor_comparison_data.build() (uniform runs, final budget, two-sided A-1d at
alpha_s = alpha_f = delta = 0.05), so every check of the predictor comparison runs first. Three predictors:
\nameUnifBNNLabel{}, \nameUnifBNN{}, then a \midrule, then \nameUnif{}. Layout copied from matrix_certified_twosided_body.tex:
LQR and PPO panels, five columns per setting (R_s, FDR, R_f, FOR, Abst.).

\nameUnif{} row: absolute three-seed means. \nameUnifBNNLabel{} and \nameUnifBNN{} rows: R_s, R_f and Abst. are signed differences
from \nameUnif{} in percentage points (row mean minus \nameUnif{} mean, both unrounded, then rounded to 2 dp). All three
seeds count there, all-Abstain seeds included (zero recall, 100% abstention). FDR and FOR are absolute means over
seeds with a nonempty predicted class, never differences: superscript (n) when n < 3, -- when n = 0. No bolding:
the rows carry different units, so a best is undefined. Means only, as every table (assert_means_only).

Before writing, every cell is recomputed from the per-seed table of Figure S3,
values/predictor_comparison_uniform_per_seed.csv (cal rows): absolute cells must match mean, seed count and LaTeX;
difference cells must match the difference of the two per-seed means. --paper copies the body into the paper's
tables/ if it differs.
"""
import csv, hashlib, json, shutil
import statistics as st
from common import ASSETS, PAPER_TABLES, VALUES, method_names, parse_args
import certified
import make_tables as T
from predictor_comparison_data import build

NAME = 'identical_data_certified'
WRAPPED_IN = 'sections/06_results.tex'
NAMES = method_names()
ROLE = 'uniform data'   # predictor_comparison_data.load() keeps only this role and asserts matrix_role for every run
FM = (T.method_label(ROLE, 'FM'), 'FM')
ROWS = [(T.method_label(ROLE, m), m) for m in ('BNN classifier', 'BNN final-state')]      # differences from FM, above the rule
DIFF = ('rec', 'fail_rec', 'abstain')                                            # FDR, FOR stay absolute
PLAIN = {'FM': NAMES['nameUnif'], 'BNN final-state': NAMES['nameUnifBNN'], 'BNN classifier': NAMES['nameUnifBNNLabel']}
PER_SEED = VALUES / 'predictor_comparison_uniform_per_seed.csv'
# paper_writer's spot values for q3d-ppo (2026-11-12), checked after rendering.
SPOT = {('q3d-ppo', r'\nameUnifBNNLabel{}', 'rec'): '$-27.24$', ('q3d-ppo', r'\nameUnifBNNLabel{}', 'fail_rec'): '$-40.44$',
        ('q3d-ppo', r'\nameUnifBNNLabel{}', 'abstain'): '$+39.76$', ('q3d-ppo', r'\nameUnifBNN{}', 'rec'): '$-35.29$',
        ('q3d-ppo', r'\nameUnifBNN{}', 'fail_rec'): '$-63.33$', ('q3d-ppo', r'\nameUnif{}', 'rec'): '$43.18$',
        ('q3d-ppo', r'\nameUnif{}', 'fdr'): '$2.44$', ('q3d-ppo', r'\nameUnif{}', 'fail_rec'): '$71.69$',
        ('q3d-ppo', r'\nameUnif{}', 'for'): '$0.05$', ('q3d-ppo', r'\nameUnif{}', 'abstain'): '$29.80$'}


def absolute(a):
    """Mean in % to 2 dp, (n) superscript below three contributing seeds, -- for none."""
    if a['n'] == 0:
        return '--'
    return '$' + f"{a['mean']:.2f}" + (f"^{{({a['n']})}}" if a['n'] < 3 else '') + '$'


def signed(d):
    """A single signed difference, never a spread. A difference that rounds to zero prints as 0.00, unsigned."""
    s = f'{d:+.2f}'
    return '$0.00$' if s in ('+0.00', '-0.00') else f'${s}$'


def body(agg, cells):
    pair = r'rr!{\tablepairdivider}rr!{\tablepairdivider}r'
    columns = r'l@{\hspace{6pt}}' + r'!{\hspace{3pt}\tablesystemdivider\hspace{3pt}}'.join([pair] * 3)
    t = [T.HEADER, f'% The table environment, caption and label are in {WRAPPED_IN}.']
    for pi, (panel, indices) in enumerate(T.PANELS):
        settings = [T.SETTINGS[i] for i in indices]
        if pi:
            t.append(r'\par\medskip')
        t += [r'\resizebox{\ifdim\width>\textwidth\textwidth\else\width\fi}{!}{%', r'\begin{tabular}{' + columns + '}',
              r'\toprule', r'\multicolumn{16}{c}{\textbf{' + panel + r'}} \\', r'\midrule',
              ' & ' + ' & '.join(r'\multicolumn{5}{c}{' + T.HEADS[i] + '}' for i in indices) + r' \\',
              r'\cmidrule(lr){2-6}\cmidrule(lr){7-11}\cmidrule(lr){12-16}',
              r'\textbf{Method} & ' + ' & '.join(label for _ in settings for _, label in T.METRICS) + r' \\',
              r'\midrule']
        for row in ROWS + ['rule', FM]:
            if row == 'rule':
                t.append(r'\midrule'); continue
            label, model = row
            vv = []
            for s in settings:
                for key, _ in T.METRICS:
                    a, f = agg[s, model, 'cal'][key], agg[s, 'FM', 'cal'][key]
                    cell = dict(table=NAME, panel=panel, row=label, model=PLAIN[model], setting=s, metric=key,
                                n=a['n'], mean=a['mean'], min=a['min'], max=a['max'])
                    if model != 'FM' and key in DIFF:
                        assert a['n'] == f['n'] == 3, (s, model, key)
                        d = a['mean'] - f['mean']
                        tex = signed(d)
                        cell |= dict(kind=f"difference from {PLAIN['FM']} (pp)", fm_mean=f['mean'], difference=d)
                    else:
                        if key in DIFF:
                            assert a['n'] == 3, (s, model, key)
                        tex = absolute(a)
                        cell |= dict(kind='absolute')
                    vv.append(tex)
                    cells.append(cell | dict(tex=tex))
            t.append(label + ' & ' + ' & '.join(vv) + r' \\')
        t += [r'\bottomrule', r'\end{tabular}}']
    return t


def cross_check(cells):
    """Every cell against the per-seed table (cal rows): absolute = mean, n and LaTeX of the defined per-seed values;
    difference = mean of the row's per-seed values minus Unif-TERA's, all three seeds."""
    with open(PER_SEED, newline='') as fh:
        seeds = [r for r in csv.DictReader(fh) if r['method'] == 'cal']
    assert len(seeds) == 90
    model_of = {plain: m for m, plain in PLAIN.items()}
    def vals(setting, model, metric):
        rr = [r for r in seeds if (r['setting'], r['model']) == (setting, model)]
        assert len(rr) == 3
        return [float(r[metric]) for r in rr if r[metric] != '']
    checked = 0
    for c in cells:
        xs = vals(c['setting'], model_of[c['model']], c['metric'])
        if c['kind'] == 'absolute':
            assert len(xs) == c['n'], (c, xs)
            want = '--' if not xs else '$' + f'{st.mean(xs):.2f}' + (f'^{{({len(xs)})}}' if len(xs) < 3 else '') + '$'
            assert want == c['tex'] and (not xs or abs(st.mean(xs) - c['mean']) < 1e-9), (c, want)
        else:
            fm = vals(c['setting'], 'FM', c['metric'])
            assert len(xs) == len(fm) == 3, (c, xs, fm)
            d = st.mean(xs) - st.mean(fm)
            assert abs(d - c['difference']) < 1e-9 and signed(d) == c['tex'], (c, d)
        checked += 1
    return checked


def main(paper):
    rows, ident, agg, (n1, n2) = build()
    assert sum(r['method'] == 'cal' for r in rows) == 90
    cells = []
    lines = body(agg, cells)
    assert len(cells) == 3 * 6 * 5
    assert not any(r'\mathbf' in l for l in lines)
    T.assert_means_only(NAME, lines)
    n_checked = cross_check(cells)
    by = {(c['setting'], c['row'], c['metric']): c['tex'] for c in cells}
    for k, want in SPOT.items():
        assert by[k] == want, (k, by[k], want)
    out = T.TABLES / f'{NAME}_body.tex'
    out.write_text('\n'.join(lines) + '\n')
    where = ''
    if paper:
        target = PAPER_TABLES / out.name
        if target.exists() and target.read_bytes() == out.read_bytes():
            where = ' (paper copy already identical, not copied)'
        else:
            shutil.copyfile(out, target)
            where = f' -> paper tables/{out.name}'
    src = {str(certified.SRC): certified.SRC_SHA256,
           str(ASSETS / 'soft_four_wsweep.jsonl'): hashlib.sha256((ASSETS / 'soft_four_wsweep.jsonl').read_bytes()).hexdigest()}
    (VALUES / f'table_{NAME}_values.json').write_text(json.dumps(dict(
        generated_by='scripts/make_identical_data_table.py', source_hashes=src,
        protocol='cal records of predictor_comparison_data.build(): uniform runs, final budget, two-sided A-1d at '
                 f"alpha = delta = 0.05. BNN rows: R_s, R_f, Abst. = row mean minus {PLAIN['FM']} mean (unrounded, then 2 dp), "
                 'all 3 seeds; FDR, FOR absolute over seeds with a nonempty predicted class. No bold.',
        checks=dict(identity_groups=len(ident), table1_fm_uniform_cells=n1, fig3_uniform_endpoint_cells=n2,
                    cells_equal_to_per_seed_table=n_checked, spot_values=len(SPOT)),
        cells=cells), indent=2) + '\n')
    print(f'{out.name}: {len(cells)} cells, {n_checked} cross-checked, {len(SPOT)} spot values{where}')


if __name__ == '__main__':
    main(parse_args(__doc__).paper)
