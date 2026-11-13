r"""Data for the Table 2 figure options (paper_writer / dm1487, 2026-11-13): the empirical comparison at the
validation-fitted w = 0.2 thresholds (tab:uncertified), six methods, six settings, previews only.

Source: values/table_matrix_uncertified_values.json, written by make_tables.py together with
tables/matrix_uncertified_body.tex. Every cell carries the three-seed mean, min, max and the number of contributing
seeds n (errors average only seeds with a nonempty predicted class). load() checks:
  - each (setting, method, metric) appears exactly once, 6 x 6 x 5 = 180 cells;
  - n is 3 for recalls and abstention; n = 0 means the predicted class is empty in every seed (the table's '--');
  - min <= mean <= max wherever n > 0, and mean, min, max are None where n = 0;
  - every cell's LaTeX equals the paper clone's tables/matrix_uncertified_body.tex for that row and column
    (bold stripped), so the figures and the printed table show the same numbers.
BNN-Label and TERA-BNN are left out (dm1487). Methods in dm1487's order; settings in the figures' order.
check_drawn() is the read-back assert every option calls on what it actually drew.
"""
import json, re
from common import VALUES, PAPER_TABLES, SETTINGS, TITLES, method_names
import make_tables as T

SOURCE = VALUES / 'table_matrix_uncertified_values.json'
BODY = PAPER_TABLES / 'matrix_uncertified_body.tex'
NAMES = method_names()
# (row label in the JSON and the table body, plain name for figures = a key of common.METHOD_MARKER)
METHODS = [('DeepReach', 'DeepReach'), ('Lyapunov NN', 'Lyapunov NN'), ('MORALS', 'MORALS'),
           ('Part-X GP', 'Part-X GP'), (r'\nameUnif{}', NAMES['nameUnif']), (r'\name{}', NAMES['name'])]
LEFT_OUT = (r'\nameBNNLabel{}', r'\nameBNN{}')
METRICS = ['rec', 'fdr', 'fail_rec', 'for', 'abstain']
SIDE = {'rec': 'success', 'fdr': 'success', 'fail_rec': 'fail', 'for': 'fail', 'abstain': 'abstain'}
LABEL = {'rec': r'$R_{\mathrm{s}}$', 'fdr': 'FDR', 'fail_rec': r'$R_{\mathrm{f}}$', 'for': 'FOR', 'abstain': 'Abst.'}
RECALL_OF = {'fdr': 'rec', 'for': 'fail_rec'}          # error -> its recall (same side)


def _body_cells():
    """{(row label, setting, metric): LaTeX} from the paper clone's table body, bold stripped."""
    out, panel = {}, -1
    for line in BODY.read_text().splitlines():
        if r'\multicolumn{16}{c}{\textbf{' in line:
            panel += 1
            continue
        label = line.split(' & ')[0].strip()
        if panel < 0 or label not in [m for m, _ in METHODS] + list(LEFT_OUT):
            continue
        vals = [v.strip().removesuffix(r'\\').strip() for v in line.split(' & ')[1:]]
        settings = [T.SETTINGS[i] for i in T.PANELS[panel][1]]
        assert len(vals) == 15, (label, len(vals))
        for k, v in enumerate(vals):
            v = re.sub(r'\\mathbf\{([^}]*)\}', r'\1', v)
            out[label, settings[k // 5], METRICS[k % 5]] = v
    return out


def load():
    """{(setting, plain method name, metric): dict(n, mean, min, max, tex)} for the 180 cells, checked."""
    cells = json.loads(SOURCE.read_text())['cells']
    plain = dict(METHODS)
    body = _body_cells()
    D = {}
    for c in cells:
        if c['row'] not in plain:
            assert c['row'] in LEFT_OUT, c['row']
            continue
        key = (c['setting'], plain[c['row']], c['metric'])
        assert key not in D, key
        n = c['n']
        if n == 0:
            assert c['mean'] is None and c['min'] is None and c['max'] is None and c['tex'] == '--', c
        else:
            assert c['min'] <= c['mean'] <= c['max'], c
        if c['metric'] in ('rec', 'fail_rec', 'abstain'):
            assert n == 3, c
        tex = re.sub(r'\\mathbf\{([^}]*)\}', r'\1', c['tex'])
        assert body[c['row'], c['setting'], c['metric']] == tex, (key, tex, body[c['row'], c['setting'], c['metric']])
        D[key] = dict(n=n, mean=c['mean'], min=c['min'], max=c['max'], tex=tex)
    assert len(D) == len(SETTINGS) * len(METHODS) * len(METRICS) == 180, len(D)
    return D


def missing(D):
    """Cells with no contributing seed: the predicted class is empty in all three seeds."""
    return sorted(k for k, v in D.items() if v['n'] == 0)


def partial(D):
    """Cells averaged over 1-2 seeds."""
    return sorted(k for k, v in D.items() if 0 < v['n'] < 3)


def check_drawn(D, drawn, expect_missing=True):
    """drawn: {(setting, method, metric): value as read back from the artists}. Every value must equal the JSON mean
    exactly, every cell with n > 0 must be drawn, and no n = 0 cell may be drawn (unless the option draws no such
    metric at all, e.g. a recall-only view: pass the subset it covers)."""
    for k, v in drawn.items():
        assert D[k]['n'] > 0, ('drew an empty cell', k)
        assert v == D[k]['mean'], ('drawn value differs from the JSON', k, v, D[k]['mean'])
    metrics = {k[2] for k in drawn}
    want = {k for k, v in D.items() if v['n'] > 0 and k[2] in metrics}
    assert set(drawn) == want, ('cells not drawn', sorted(want - set(drawn))[:5])
    return len(drawn)


if __name__ == '__main__':
    D = load()
    print(f'{len(D)} cells, all equal to {BODY.name} in the paper clone')
    print('missing (n = 0):', missing(D))
    print('partial (n < 3):', [(k, D[k]['n']) for k in partial(D)])
    print('titles:', dict(zip(SETTINGS, TITLES)))
