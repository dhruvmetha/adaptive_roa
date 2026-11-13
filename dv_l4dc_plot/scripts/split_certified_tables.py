"""Split the certified table into two-sided and success-only bodies.

Four bodies (Sumanth, 2026-11-13): matrix_certified_main, the main paper's two-sided table with BNN-Label,
TERA-BNN and TERA; matrix_certified_mlp, the supplement's table with MLP-Label, TERA-MLP and TERA; each is bold
among its own three rows. matrix_certified_twosided, all five adaptive predictors, is no longer in the paper and is
kept as the reference the two short tables are checked against. success_only is the conservative certificate of TERA.

Uses make_tables' existing hash-pinned source rows and rendering helpers. The
original matrix_certified_body.tex and its values JSON are read-only baselines.
This script writes previews in this pipeline only; it never writes a paper repo.
"""
import hashlib
import json
from pathlib import Path

import make_tables as tables
from common import ASSETS, VALUES


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


LEGACY_LABELS = {
    r'\name-FM, Success only': tables.SUCCESS_ONLY,
    r'\name-FM (adaptive)': r'\name{}',
    r'\name-MLP': r'\nameMLP{}',
    r'\name-BNN': r'\nameBNN{}',
    'MLP-Label': r'\nameMLPLabel{}',
    'BNN-Label': r'\nameBNNLabel{}',
}
LEGACY_SUCCESS_ONLY = r'\name-FM, Success only'


def key(cell):
    row = LEGACY_LABELS.get(cell['row'], cell['row'])
    return row, cell['setting'], cell['metric']


def normalize_row_label(line):
    for legacy, current in LEGACY_LABELS.items():
        if line.startswith(legacy + ' & '):
            return current + line[len(legacy):]
    return line


def main():
    baseline_tex = tables.TABLES / 'matrix_certified_body.tex'
    baseline_values = VALUES / 'table_matrix_certified_values.json'
    protected = {p: sha(p) for p in (baseline_tex, baseline_values)}
    original = json.loads(baseline_values.read_text())
    meta_path = ASSETS / 'generalized_6c_values.json'
    meta = json.loads(meta_path.read_text())
    source = Path(meta['source'])
    assert sha(source) == meta['source_sha256'], 'matrix_rows.jsonl changed'
    assert sha(source) == original['source_hashes'][str(source)]
    assurance = [r for line in source.read_text().splitlines()
                 if (r := json.loads(line)).get('in_matrix')
                 and r.get('matrix_role') in ('method', 'model comparison')]
    assert len(assurance) == 90
    own = [r for r in assurance if r['matrix_role'] == 'method']
    assert len(own) == 18
    for setting in tables.SETTINGS:
        rr = [r for r in assurance if r['setting'] == setting]
        assert len(rr) == 15
        assert len({r['n_cal'] for r in rr}) == len({r['n_val'] for r in rr}) == 1
    cert_rows = [dict(setting=r['setting'], seed=r['seed'], model=r['model'],
                      matrix_role=r['matrix_role'],
                      test={k: r[src] for k, src in [
                          ('rec', 'c6c_rec'), ('fdr', 'c6c_fp'),
                          ('fail_rec', 'c6c_fail_rec'), ('for', 'c6c_ferr'),
                          ('abstain', 'c6c_abstain')]})
                 for r in assurance]

    name = 'matrix_certified_twosided'
    tables.WRAPPED_IN[name] = 'no paper section (five-row reference, not in the paper)'
    main_cells = []
    main_lines = tables.matrix_body(name, tables.CERT_GROUPS, cert_rows,
                                    main_cells, group_labels=False)
    assert len(main_cells) == 150
    assert not any(tables.SUCCESS_ONLY in line for line in main_lines)

    # The main paper's table: the selected BNN variants and TERA, in ordered_groups' order (TERA last). Same
    # numbers as the five-row body; only the bold marks are taken among the three displayed rows.
    short_name = 'matrix_certified_main'
    tables.WRAPPED_IN[short_name] = 'sections/06_results.tex'
    short_groups = tables.ordered_groups([tables.FM_ADAPTIVE, tables.TERA_BNN, tables.BNN_LABEL], fm_own_group=False)
    assert [label for _, models in short_groups for label, _, _ in models] == \
        [tables.BNN_LABEL[0], tables.TERA_BNN[0], tables.FM_ADAPTIVE[0]]
    short_cells = []
    short_lines = tables.matrix_body(short_name, short_groups, cert_rows, short_cells, group_labels=False)
    assert len(short_cells) == 90
    full = {key(c): c for c in main_cells}
    for c in short_cells:
        for field in ('n', 'mean', 'min', 'max'):
            assert c[field] == full[key(c)][field], (key(c), field)

    # The supplement's table: the MLP counterparts and TERA, same checks as the main table.
    mlp_name = 'matrix_certified_mlp'
    tables.WRAPPED_IN[mlp_name] = 'sections/appendix_predictor_ablation.tex'
    mlp_groups = tables.ordered_groups([tables.FM_ADAPTIVE, tables.TERA_MLP, tables.MLP_LABEL], fm_own_group=False)
    assert [label for _, models in mlp_groups for label, _, _ in models] == \
        [tables.MLP_LABEL[0], tables.TERA_MLP[0], tables.FM_ADAPTIVE[0]]
    mlp_cells = []
    mlp_lines = tables.matrix_body(mlp_name, mlp_groups, cert_rows, mlp_cells, group_labels=False)
    assert len(mlp_cells) == 90
    for c in mlp_cells:
        for field in ('n', 'mean', 'min', 'max'):
            assert c[field] == full[key(c)][field], (key(c), field)

    success_cells = []
    success_row = tables.success_only_row(own, success_cells)
    success_lines = [tables.HEADER,
                     '% Generated by split_certified_tables.py; the caption and label live in the supplement.',
                     r'\begin{tabular}{lrr}', r'\toprule',
                     r'\textbf{Setting} & ' + tables.METRICS[0][1] + ' & ' + tables.METRICS[1][1] + r' \\',
                     r'\midrule']
    for title, setting in zip(tables.TITLES, tables.SETTINGS):
        _, values = success_row([setting])
        assert values[2:] == ['n/a'] * 3
        success_lines.append(title + ' & ' + ' & '.join(values[:2]) + r' \\')
    success_lines += [r'\bottomrule', r'\end{tabular}']
    assert len(success_cells) == 12

    baseline = {key(c): c for c in original['cells']}
    generated = {key(c): c for c in main_cells + success_cells}
    assert len(baseline) == len(generated) == 162
    assert baseline.keys() == generated.keys()
    for k, cell in generated.items():
        for field in ('n', 'mean', 'min', 'max', 'tex', 'best'):
            assert cell[field] == baseline[k][field], (k, field, cell[field], baseline[k][field])

    # A second check makes the two-sided body byte-equivalent to removing only
    # the success-only rows and their immediately preceding rules from the old body.
    existing = baseline_tex.read_text().splitlines()
    retained = []
    for line in existing:
        if line.startswith(tuple(label + ' & ' for label in (tables.SUCCESS_ONLY, LEGACY_SUCCESS_ONLY))):
            assert retained.pop() == r'\midrule'
        else:
            retained.append(normalize_row_label(line))
    assert retained[2:] == main_lines[2:]

    sources = {str(source): sha(source), str(meta_path): sha(meta_path),
               str(baseline_values): protected[baseline_values],
               str(baseline_tex): protected[baseline_tex]}
    for output_name, lines, cells in [
            (name, main_lines, main_cells), (short_name, short_lines, short_cells),
            (mlp_name, mlp_lines, mlp_cells),
            ('success_only', success_lines, success_cells)]:
        tables.assert_means_only(output_name, lines)
        output = tables.TABLES / f'{output_name}_body.tex'
        output.write_text('\n'.join(lines) + '\n')
        (VALUES / f'table_{output_name}_values.json').write_text(json.dumps(dict(
            generated_by='scripts/split_certified_tables.py',
            source_hashes=sources, verified_original_cells=len(cells), cells=cells), indent=2) + '\n')
        print(f'{output.name}: {len(cells)} cells, all identical to original certified table')
    assert all(sha(p) == digest for p, digest in protected.items())
    print('Original certified table and values JSON unchanged.')


if __name__ == '__main__':
    main()
