r"""Certified identical-data table body: tables/identical_data_certified_body.tex, paper Table 4 (tab:identical_data,
sections/06_results.tex) since paper commit 7a21aa8. It replaced the dot plot predictor_comparison_uniform_certified,
now in scripts/archive.

Data: the 'cal' records of predictor_comparison_data.build() (uniform runs, final budget, two-sided A-1d at
alpha_s = alpha_f = delta = 0.05), so every check of the predictor comparison runs first. Three predictors:
\nameUnifBNNLabel{}, \nameUnifBNN{}, then a \midrule, then \nameUnif{}; LQR and RL panels, five columns per
setting (R_s, FDR, R_f, FOR, Abst.).

All three rows print absolute three-seed means (Sumanth, 2026-11-13, "Option 1": the earlier signed differences
from \nameUnif{} could not carry the shading). R_s, R_f and Abst. count all three seeds, all-Abstain seeds included
(zero recall, 100% abstention). FDR and FOR are means over seeds with a nonempty predicted class (no seed-count
superscript, Sumanth 2026-11-13), -- when n = 0. When n = 0, that outcome's R_s or R_f cell is -- too (Sumanth, 2026-11-13: "cases that
don't have results should have dashes and not 0s"). Shading as make_tables (T.jw_best over the three displayed
rows, per setting and outcome), no bold. Means only, as every table (assert_means_only). Layout as make_tables
(Option A, T.table_open / T.panel_head). The Results text quotes differences between rows; DIFF_SPOT checks them
against the unrounded means.

Before writing, every cell is recomputed from the per-seed table of Figure S3,
values/predictor_comparison_uniform_per_seed.csv (cal rows): each cell must match mean, seed count and LaTeX.
--paper copies the body into the paper's tables/ if it differs.
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
ROWS = [(T.method_label(ROLE, m), m) for m in ('BNN classifier', 'BNN final-state')]      # above the rule, FM below
ALL_SEEDS = ('rec', 'fail_rec', 'abstain')                                       # every seed counts; FDR, FOR do not
PLAIN = {'FM': NAMES['nameUnif'], 'BNN final-state': NAMES['nameUnifBNN'], 'BNN classifier': NAMES['nameUnifBNNLabel']}
PER_SEED = VALUES / 'predictor_comparison_uniform_per_seed.csv'
# paper_writer's spot values for q3d-ppo (2026-11-12), checked after rendering.
SPOT = {('q3d-ppo', r'\nameUnif{}', 'rec'): '$43.18$', ('q3d-ppo', r'\nameUnif{}', 'fdr'): '$2.44$',
        ('q3d-ppo', r'\nameUnif{}', 'fail_rec'): '$71.69$', ('q3d-ppo', r'\nameUnif{}', 'for'): '$0.05$',
        ('q3d-ppo', r'\nameUnif{}', 'abstain'): '$29.80$'}
# Row mean minus \nameUnif{} mean (unrounded, then 2 dp), the differences the table printed until 2026-11-13. The
# Results text quotes the first two ("exceed ... by 27.24 and 40.44 percentage points").
DIFF_SPOT = {('q3d-ppo', 'BNN classifier', 'rec'): '$-27.24$', ('q3d-ppo', 'BNN classifier', 'fail_rec'): '$-40.44$',
             ('q3d-ppo', 'BNN classifier', 'abstain'): '$+39.76$', ('q3d-ppo', 'BNN final-state', 'rec'): '$-35.29$',
             ('q3d-ppo', 'BNN final-state', 'fail_rec'): '$-63.33$'}


def absolute(a):
    """Mean in % to 2 dp, -- for none. No seed-count superscript."""
    if a['n'] == 0:
        return '--'
    return '$' + f"{a['mean']:.2f}" + '$'


def signed(d):
    """A single signed difference, never a spread. A difference that rounds to zero prints as 0.00, unsigned."""
    s = f'{d:+.2f}'
    return '$0.00$' if s in ('+0.00', '-0.00') else f'${s}$'


def shown(a):
    """The 2 dp mean the table prints, None for no contributing seed."""
    return float(f"{a['mean']:.2f}") if a['n'] else None


def body(agg, cells):
    label_of = {model: label for label, model in ROWS + [FM]}
    shaded = {(s, rec): T.jw_best({label_of[m]: (shown(agg[s, m, 'cal'][rec]), shown(agg[s, m, 'cal'][err]))
                                   for m in label_of})
              for s in T.SETTINGS for rec, err in T.OUTCOMES}
    t = [T.HEADER, f'% The table environment, caption and label are in {WRAPPED_IN}.'] + T.table_open()
    for pi, (panel, indices) in enumerate(T.PANELS):
        settings = [T.SETTINGS[i] for i in indices]
        t += T.panel_head(pi, panel, indices)
        for row in ROWS + ['rule', FM]:
            if row == 'rule':
                t.append(r'\midrule'); continue
            label, model = row
            vv = []
            for s in settings:
                for key, _ in T.METRICS:
                    a = agg[s, model, 'cal'][key]
                    if key in ALL_SEEDS:
                        assert a['n'] == 3, (s, model, key)
                    cell = dict(table=NAME, panel=panel, row=label, model=PLAIN[model], setting=s, metric=key,
                                n=a['n'], mean=a['mean'], min=a['min'], max=a['max'], kind='absolute')
                    tex = absolute(a)
                    if key in T.ERR_OF and agg[s, model, 'cal'][T.ERR_OF[key]]['n'] == 0:
                        tex = '--'   # no seed predicts this outcome: a dash, not a recall (Sumanth, 2026-11-13)
                        cell |= dict(empty_class=True)
                    winner = any(key in keys and label in shaded[s, keys[0]] for keys in T.OUTCOMES)
                    vv.append(T.SHADE + tex if winner else tex)
                    cells.append(cell | dict(tex=tex, shaded=winner))
            t.append(label + ' & ' + ' & '.join(vv) + r' \\')
    return t + T.table_close()


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
    err_of = dict(T.OUTCOMES)
    for c in cells:
        assert c['kind'] == 'absolute', c
        xs = vals(c['setting'], model_of[c['model']], c['metric'])
        assert len(xs) == c['n'], (c, xs)
        if c.get('empty_class'):
            assert c['tex'] == '--' and not vals(c['setting'], model_of[c['model']], err_of[c['metric']]), c
        else:
            want = '--' if not xs else '$' + f'{st.mean(xs):.2f}' + '$'
            assert want == c['tex'] and (not xs or abs(st.mean(xs) - c['mean']) < 1e-9), (c, want)
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
    for (s, model, key), want in DIFF_SPOT.items():
        assert signed(agg[s, model, 'cal'][key]['mean'] - agg[s, 'FM', 'cal'][key]['mean']) == want, (s, model, key, want)
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
                 'alpha = delta = 0.05. All rows absolute three-seed means; R_s, R_f, Abst. over all 3 seeds; FDR, FOR over seeds with a nonempty '
                 'predicted class. No bold; J_w shading as make_tables.',
        checks=dict(identity_groups=len(ident), table1_fm_uniform_cells=n1, fig3_uniform_endpoint_cells=n2,
                    cells_equal_to_per_seed_table=n_checked, spot_values=len(SPOT)),
        cells=cells), indent=2) + '\n')
    print(f'{out.name}: {len(cells)} cells, {n_checked} cross-checked, {len(SPOT)} spot values{where}')


if __name__ == '__main__':
    main(parse_args(__doc__).paper)
