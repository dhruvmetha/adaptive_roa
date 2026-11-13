"""Paper table bodies: matrix_uncertified (Table 1), matrix_certified (Table 2), predictor_ablation (Table S1).

Ported 2026-11-12 from deterministic-verification/results-discussion-assets/write_paper_tables.py with
every number and rounding rule unchanged. Writes only tables/<name>_body.tex: the tabular part (column
headers, numbers, shading marks, the \\resizebox wrapper). The table environment, caption and label live in
the paper's sections and this script never writes them. With --paper the bodies are copied, byte for byte,
into the paper's tables/.

Layout (Sumanth, 2026-11-13, "Option A"): one tabular holds both controller panels, so their columns align.
The controller name (LQR, RL) sits in the corner cell of each panel's system row, the metric header row
appears once under the first panel's system row, and a heavy rule separates the two panels.

Entries are test percentages averaged over three seeds within each setting. Tables print means only,
never a [min, max] range or an SD (Dhruv, 2026-11-12); assert_means_only() fails on any body that does.
Figures keep their min-max bands.
  matrix_uncertified, predictor_ablation  soft four-threshold rule fit on validation at w=0.2
  matrix_certified                        two-sided A-1d at delta=0.05 (matrix_rows c6c_*), plus one row per
                                          panel, "FM, Success only": the one-sided 6a certificate
                                          (c6a_rec, c6a_fp) of the FM method runs, with n/a for R_f, FOR, Abst.
                                          This row is outside the shading comparison.
Tables 1 and 2 have no group-label rows (2026-11-12, to save space); a \\midrule still separates method
groups. Table S1 keeps its label. The former Table 3 (verification_application) is folded into Table 2.
No bold (Sumanth, 2026-11-13: recall and error must be read together, not one column at a time). For each
setting and outcome, the (Rec, Err) cell pair of the displayed row with the lowest 0.8 Err + 0.2 (100 - Rec)
gets \\cellcolor{tablebest}: the outcome's term of the threshold-fitting objective J_w at w = 0.2. The score
is computed from the displayed 2 dp means, so a reader can recompute it from the table; ties are all shaded;
rows whose Err is -- (no seed predicted that outcome) take no part. Abst. is never shaded. The paper defines
tablebest (colortbl) in main.tex and supplementary.tex.
Conditional errors use seeds with a nonempty predicted class. No table prints that seed count (Sumanth, 2026-11-13:
the superscripts were removed from every table). -- means all three were empty. When all three are empty, the outcome's recall is -- as well,
not 0.00 (Sumanth, 2026-11-13: "cases that don't have results should have dashes and not 0s").
"""
import hashlib, json, re, shutil
import statistics as st
from common import ASSETS, PAPER_TABLES, HERE, VALUES, parse_args, spread, REC_S, REC_F, ERR_S, ERR_F
from table_previews import previews

TABLES = HERE / 'tables'
SETTINGS = ['pen-lqr', 'cp-lqr', 'cp-ppo', 'q2d-ppo', 'q3d-lqr', 'q3d-ppo']   # table order, not figure order
# Setting names as in common.TITLES (Sumanth, 2026-11-13): "System Controller" for a row per setting, the system
# alone under a controller panel, and no dimension (Table 1 of the paper lists it).
TITLES = ['Pendulum LQR', 'Cartpole LQR', 'Cartpole RL', 'Planar quadrotor RL',
          'Spatial quadrotor LQR', 'Spatial quadrotor RL']
HEADS = ['Pendulum', 'Cartpole', 'Cartpole', 'Planar quadrotor', 'Spatial quadrotor', 'Spatial quadrotor']
PANELS = [('LQR controllers', [0, 1, 4]), ('RL controllers', [2, 3, 5])]
# Column headers in the method section's notation (04_proposed_method.tex).
METRICS = [('rec', REC_S[:-1] + r'\uparrow$'), ('fdr', ERR_S[:-1] + r'\downarrow$'),
           ('fail_rec', REC_F[:-1] + r'\uparrow$'), ('for', ERR_F[:-1] + r'\downarrow$'), ('abstain', r'$\widehat{\mathrm{Abst}}\downarrow$')]
# Row labels use the complete shared manuscript macros in method-names.tex.
SUCCESS_ONLY = r'\name{}, Success only'   # the adaptive FM runs ('method' role)
# Rows available to the tables: (label, matrix_role, model).
# Baseline rows carry the method name only, no "score"/"levels" (Dhruv, 2026-11-12).
PRIOR_ROWS = [('DeepReach', 'prior work', 'DeepReach'), ('Lyapunov NN', 'prior work', 'Lyapunov NN'),
              ('MORALS', 'prior work', 'MORALS'), ('Part-X GP', 'prior work', 'Part-X')]
# Learned-predictor row labels come from the row's data, (matrix_role, model), never from old label text
# (Dhruv, 2026-11-12). The role fixes the acquisition arm: in runs_index.tsv 'method' and 'model comparison' are
# U_H at rho = 0.75 (adaptive) and 'uniform data' is uniform at rho = 0. check_roles() re-reads the index.
ARM_OF_ROLE = {'method': 'adaptive', 'model comparison': 'adaptive', 'uniform data': 'uniform'}
MACRO = {'FM': 'name', 'BNN final-state': 'nameBNN', 'MLP final-state': 'nameMLP',
         'MLP classifier': 'nameMLPLabel', 'BNN classifier': 'nameBNNLabel'}
RUNS_INDEX = '/common/users/shared/pracsys/adaptive_roa_experiments/det_v3_seeds/_index/runs_index.tsv'


def macro_name(role, model):
    """The method-names.tex macro for a learned predictor: adaptive arm -> \\name..., uniform arm -> \\nameUnif..."""
    base = MACRO[model]
    return base if ARM_OF_ROLE[role] == 'adaptive' else 'nameUnif' + base[len('name'):]


def method_label(role, model):
    return '\\' + macro_name(role, model) + '{}'


def check_roles():
    """Fail unless the index still says what ARM_OF_ROLE assumes, and every macro used is defined in the paper's
    method-names.tex."""
    import csv
    from common import method_names
    with open(RUNS_INDEX, newline='') as fh:
        rows = [r for r in csv.DictReader(fh, delimiter='\t') if r['role'] in ARM_OF_ROLE and r['model'] in MACRO]
    for r in rows:
        want = ('U_H', '0.75') if ARM_OF_ROLE[r['role']] == 'adaptive' else ('uniform', '0.0')
        assert (r['acquisition'], r['d2_ratio']) == want, (r['run'], r['role'], r['acquisition'], r['d2_ratio'])
    defined = method_names()
    for role in ARM_OF_ROLE:
        for model in MACRO:
            assert macro_name(role, model) in defined, macro_name(role, model)
    return len(rows)


FM_UNIFORM = (method_label('uniform data', 'FM'), 'uniform data', 'FM')
FM_ADAPTIVE = (method_label('method', 'FM'), 'method', 'FM')
TERA_MLP, TERA_BNN = [(method_label('model comparison', m), 'model comparison', m) for m in ('MLP final-state', 'BNN final-state')]
MLP_LABEL, BNN_LABEL = [(method_label('model comparison', m), 'model comparison', m) for m in ('MLP classifier', 'BNN classifier')]
FAMILY = ['MLP', 'BNN']
# TERA's own row prints as \nameOurs, "TERA (Ours)" in method-names.tex (Sumanth, 2026-11-13). Only the printed label
# changes: cells and every keyed check keep row=\name{}.
OURS_LABEL = r'\nameOurs{}'


def printed_label(rowlabel):
    return OURS_LABEL if rowlabel == FM_ADAPTIVE[0] else rowlabel


def ordered_groups(rows, fm_own_group=True):
    r"""Dhruv's standing row order for every table (2026-11-12): TERA rows always below the baselines.
    Groups, each separated by a \midrule: prior methods (given order); the -Label rows then the other TERA
    variants, both in FAMILY order so each TERA variant sits below its matching label row; then the
    TERA rows, uniform before adaptive. matrix_body appends the Success-only row last.
    fm_own_group=False keeps the TERA rows in the same group as the variants (Table 2, where the only rule
    sits directly above the Success-only row)."""
    family = lambda r: FAMILY.index(r[2].split()[0])
    prior = [r for r in rows if r[1] == 'prior work']
    labels = sorted((r for r in rows if r[2].endswith('classifier')), key=family)
    tera = sorted((r for r in rows if r[2] != 'FM' and r[1] != 'prior work' and not r[2].endswith('classifier')), key=family)
    fm = sorted((r for r in rows if r[2] == 'FM'), key=lambda r: r[1] != 'uniform data')
    assert len(prior) + len(labels) + len(tera) + len(fm) == len(rows)
    if fm_own_group:
        groups = [('Prior-method adaptations', prior), ('Other adaptive predictors', labels + tera), ('Our method', fm)]
    else:
        groups = [('Prior-method adaptations', prior), ('Adaptive predictor variants', labels + tera + fm)]
    return [(name, g) for name, g in groups if g]


MAIN_GROUPS = ordered_groups(PRIOR_ROWS + [BNN_LABEL, TERA_BNN, FM_UNIFORM, FM_ADAPTIVE])      # Table 1
CERT_GROUPS = ordered_groups([FM_ADAPTIVE, TERA_MLP, TERA_BNN, MLP_LABEL, BNN_LABEL], fm_own_group=False)  # Table 2
# Table S1 (predictor_ablation), Sumanth 2026-11-13: the MLP counterparts against TERA only, in ordered_groups'
# order (MLP-Label, TERA-MLP, TERA). The BNN variants are in the main paper's tables. Bold is among these rows.
ABLATION_GROUPS = ordered_groups([FM_ADAPTIVE, TERA_MLP, MLP_LABEL], fm_own_group=False)
HEADER = '% Generated table body (column headers and numbers only). Edit the generator, not this file.'
WRAPPED_IN = {'matrix_uncertified': 'sections/06_experiments.tex', 'matrix_certified': 'sections/06_experiments.tex',
              'predictor_ablation': 'sections/appendix_predictor_ablation.tex'}


def value(rs, key):
    """Mean in % to 2 dp. Tables print means only (Dhruv, 2026-11-12): never a range, an SD or a seed count."""
    vv = [100 * r[key] for r in rs if r[key] is not None]
    if not vv:
        return '--'
    mean, _, _ = spread(vv)
    return '$' + f'{mean:.2f}' + '$'


SPREAD_IN_TABLE = re.compile(r'\\pm|\[\s*-?[0-9]|\]\$|\^\{\\,|\\shortstack|[0-9]\$?\s*(--|–)\s*\$?[+-]?[0-9]')
# A single signed value (a difference such as $-27.24$ or $+7.29$, paper_writer 2026-11-12) is a mean, not a
# spread, and passes. The rule is checked on import against both kinds of cell.
for _ok in (r'$-27.24$ & $+7.29$ & $43.18^{(1)}$ & -- & $0.00$',):
    assert not SPREAD_IN_TABLE.search(_ok), _ok
for _bad in (r'$1.00 \pm 0.20$', r'$1.00\,[0.50, 1.50]$', r'$0.50$--$1.50$', r'$-0.5$ – $+1.5$',
             r'\shortstack{1.00\\0.20}', r'$1.00^{\,1.50}_{\,0.50}$'):
    assert SPREAD_IN_TABLE.search(_bad), _bad


def assert_means_only(name, lines):
    """Fail if a table body prints any spread: a [min, max] range, a +/- SD, a min--max range, or a
    stacked/scripted range. A single signed value is not a spread."""
    bad = [l for l in lines if not l.startswith('%') and SPREAD_IN_TABLE.search(l)]
    assert not bad, (name, bad[:2])


def header(name):
    return [HEADER, f'% The table environment, caption and label are in {WRAPPED_IN[name]}.']


OUTCOMES = [('rec', 'fdr'), ('fail_rec', 'for')]   # the (Rec, Err) column pair of each outcome
ERR_OF = dict(OUTCOMES)
SHADE = r'\cellcolor{tablebest}'


def jw_best(pairs):
    """pairs: row label -> (Rec, Err), the 2 dp percentages shown in the table, Err None when undefined. Returns the
    labels with the lowest 0.8 Err + 0.2 (100 - Rec), the outcome's J_w term at w = 0.2. Computed in exact integer
    hundredths (5 * 100 * term = 4 Err + 100 * 100 - Rec), so ties are exact and all shaded. Rows with Err None
    take no part."""
    score = {k: 4 * round(err * 100) + 10000 - round(rec * 100) for k, (rec, err) in pairs.items() if err is not None}
    return {k for k, v in score.items() if v == min(score.values())} if score else set()


def shaded_rows(groups, rows):
    r"""(setting, rec_key) -> the row labels to shade among the displayed rows (jw_best)."""
    def shown(rs, key):
        v = [100 * r['test'][key] for r in rs if r['test'][key] is not None]
        return float(f'{st.mean(v):.2f}') if v else None
    out = {}
    for setting in SETTINGS:
        for rec_key, err_key in OUTCOMES:
            pairs = {}
            for _, models in groups:
                for rowlabel, role, model in models:
                    rs = [r for r in rows if (r['setting'], r['matrix_role'], r['model']) == (setting, role, model)]
                    pairs[rowlabel] = (shown(rs, rec_key), shown(rs, err_key))
            out[setting, rec_key] = jw_best(pairs)
    return out


def table_open():
    """Both panels share one tabular (layout "Option A"), shrunk to the text width if wider."""
    pair = r'rr!{\tablepairdivider}rr!{\tablepairdivider}r'
    columns = r'l@{\hspace{6pt}}' + r'!{\hspace{3pt}\tablesystemdivider\hspace{3pt}}'.join([pair] * 3)
    return [r'\resizebox{\ifdim\width>\textwidth\textwidth\else\width\fi}{!}{%', r'\begin{tabular}{' + columns + '}',
            r'\toprule']


def panel_head(pi, panel, indices):
    """A panel's system row with the controller (LQR, RL) in the corner cell. The metric header row follows only
    the first panel's system row; a heavy rule separates the panels."""
    head = [r'\midrule[\heavyrulewidth]'] if pi else []
    head += [r'\textbf{' + panel.split()[0] + '} & ' + ' & '.join(r'\multicolumn{5}{c}{' + HEADS[i] + '}' for i in indices)
             + r' \\', r'\cmidrule(lr){2-6}\cmidrule(lr){7-11}\cmidrule(lr){12-16}']
    if not pi:
        head += [r'\textbf{Method} & ' + ' & '.join(label for _ in indices for _, label in METRICS) + r' \\', r'\midrule']
    return head


def table_close():
    return [r'\bottomrule', r'\end{tabular}}']


def matrix_body(name, groups, rows, cells, group_labels=True, extra_row=None, extra_row_rule=True):
    r"""group_labels=False drops the \emph group-label rows but keeps the \midrule between groups. extra_row,
    if given, maps a panel's settings to (label, cells); that row follows a \midrule at the panel's foot and
    takes no part in the shading comparison."""
    shaded = shaded_rows(groups, rows)
    t = header(name) + table_open()
    for pi, (panel, indices) in enumerate(PANELS):
        panel_settings = [SETTINGS[i] for i in indices]
        t += panel_head(pi, panel, indices)
        for gi, (group, models) in enumerate(groups):
            if gi:
                t.append(r'\midrule')
            if group_labels:
                t.append(r'\multicolumn{16}{l}{\emph{' + group + r'}} \\')
            for rowlabel, role, model in models:
                vv = []
                for setting in panel_settings:
                    rs = [r for r in rows if (r['setting'], r['matrix_role'], r['model']) == (setting, role, model)]
                    assert len(rs) == len({r['seed'] for r in rs}) == 3
                    for key, _ in METRICS:
                        v = value([r['test'] for r in rs], key)
                        if key in ERR_OF and all(r['test'][ERR_OF[key]] is None for r in rs):
                            v = '--'   # no seed predicts this outcome: a dash, not a 0.00 recall (Sumanth, 2026-11-13)
                        defined = [100 * r['test'][key] for r in rs if r['test'][key] is not None]
                        winner = any(key in keys and rowlabel in shaded[setting, keys[0]] for keys in OUTCOMES)
                        vv.append(SHADE + v if winner else v)
                        cells.append(dict(table=name, panel=panel, row=rowlabel, model=model, role=role, setting=setting,
                                          metric=key, n=len(defined),
                                          **dict(zip(['mean', 'min', 'max'],
                                                     [round(x, 6) if x is not None else None for x in spread(defined)])),
                                          tex=v, shaded=winner))
                t.append(printed_label(rowlabel) + ' & ' + ' & '.join(vv) + r' \\')
        if extra_row:
            label, extra = extra_row(panel_settings)
            t += ([r'\midrule'] if extra_row_rule else []) + [label + ' & ' + ' & '.join(extra) + r' \\']
    return t + table_close()


def success_only_row(own, cells):
    """Table 2's "FM, Success only" row: the one-sided 6a certificate of the FM method runs (the
    former Table 3). R_s = c6a_rec and FDR = c6a_fp as 2-dp means; R_f, FOR and Abst. are n/a because only
    Success is certified. Never shaded, and never part of the shading comparison."""
    def row(panel_settings):
        out = []
        for setting in panel_settings:
            rs = [r for r in own if r['setting'] == setting]
            assert len(rs) == 3 and all(r['c6a_pass'] for r in rs)
            for metric, key in [('rec', 'c6a_rec'), ('fdr', 'c6a_fp'), ('fail_rec', None), ('for', None), ('abstain', None)]:
                if key is None:
                    out.append('n/a')
                    continue
                v = value(rs, key)
                out.append(v)
                mean, lo, hi = spread([100 * r[key] for r in rs if r[key] is not None])
                cells.append(dict(table='matrix_certified', row=SUCCESS_ONLY, setting=setting,
                                  metric=metric, source=key, n=len(rs), mean=round(mean, 6), min=round(lo, 6),
                                  max=round(hi, 6), tex=v, shaded=False))
        return SUCCESS_ONLY, out
    return row


def main(paper):
    print(f'row labels from (role, model): {check_roles()} index rows agree with ARM_OF_ROLE')
    srcs = [ASSETS / 'soft_four_wsweep.jsonl', ASSETS / 'partx_soft_wsweep.jsonl', ASSETS / 'generalized_6c_values.json']
    soft = [json.loads(l) for f in srcs[:2] for l in f.read_text().splitlines() if json.loads(l)['w'] == .2]
    assert len(soft) == 324
    meta = json.loads(srcs[2].read_text())
    source = __import__('pathlib').Path(meta['source'])
    assert hashlib.sha256(source.read_bytes()).hexdigest() == meta['source_sha256'], 'matrix_rows.jsonl changed'
    srcs.append(source)
    assurance = [r for l in source.read_text().splitlines()
                 if (r := json.loads(l)).get('in_matrix') and r.get('matrix_role') in ('method', 'model comparison')]
    assert len(assurance) == 90
    own = [r for r in assurance if r['matrix_role'] == 'method']
    assert len(own) == 18
    # All predictors receive the same per-run A-1d procedure and split sizes.
    for setting in SETTINGS:
        rr = [r for r in assurance if r['setting'] == setting]
        assert len(rr) == 15 and len({r['n_cal'] for r in rr}) == len({r['n_val'] for r in rr}) == 1
    cert_rows = [dict(setting=r['setting'], seed=r['seed'], model=r['model'], matrix_role=r['matrix_role'],
                      test={k: r[src] for k, src in [('rec', 'c6c_rec'), ('fdr', 'c6c_fp'), ('fail_rec', 'c6c_fail_rec'),
                                                     ('for', 'c6c_ferr'), ('abstain', 'c6c_abstain')]})
                 for r in assurance]
    cells = {k: [] for k in WRAPPED_IN}
    bodies = {'matrix_uncertified': matrix_body('matrix_uncertified', MAIN_GROUPS, soft, cells['matrix_uncertified'],
                                                group_labels=False),
              'predictor_ablation': matrix_body('predictor_ablation', ABLATION_GROUPS, soft, cells['predictor_ablation'],
                                                group_labels=False),
              'matrix_certified': matrix_body('matrix_certified', CERT_GROUPS, cert_rows, cells['matrix_certified'],
                                              group_labels=False, extra_row_rule=True,
                                              extra_row=success_only_row(own, cells['matrix_certified']))}
    assert [len(cells[k]) for k in ['matrix_uncertified', 'predictor_ablation', 'matrix_certified']] == [240, 90, 162]
    TABLES.mkdir(exist_ok=True); VALUES.mkdir(exist_ok=True)
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in srcs}
    for name, lines in bodies.items():
        assert_means_only(name, lines)
        out = TABLES / f'{name}_body.tex'
        out.write_text('\n'.join(lines) + '\n')
        where = ''
        if paper:  # copy only bodies that differ, so unchanged tables stay untouched in the paper
            target = PAPER_TABLES / out.name
            if target.exists() and target.read_bytes() == out.read_bytes():
                where = ' (paper copy already identical, not copied)'
            else:
                shutil.copyfile(out, target)
                where = f' -> paper tables/{out.name}'
        (VALUES / f'table_{name}_values.json').write_text(json.dumps(dict(
            source_hashes=hashes, cells=cells[name]), indent=2) + '\n')
        print(f'{out.name}: {len(cells[name])} cells{where}')
    # Layout previews of Table 1 (table_previews.py). Written here only, never copied to the paper.
    for name, lines in previews(soft, MAIN_GROUPS).items():
        assert_means_only(name, lines)
        (TABLES / f'{name}.tex').write_text('\n'.join(lines) + '\n')
        print(f'{name}.tex: preview, not copied')


if __name__ == '__main__':
    main(parse_args(__doc__).paper)
