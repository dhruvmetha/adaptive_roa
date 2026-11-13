"""Two-sided A-1d certified test rates for the FM runs, shared by every certified figure.

Source: the shared _index/scored/matrix_rows.jsonl, pinned by sha256 in generalized_6c_values.json
(alpha_Succ = alpha_Fail = delta = 0.05). Metrics are read as make_tables.py reads them for
matrix_certified (Table 2): R_s = c6c_rec, R_f = c6c_fail_rec, FDR = c6c_fp, FOR = c6c_ferr.
An empty predicted class has recall 0.0 and FDR/FOR None, so recall averages all three seeds and
errors average only the seeds with a nonempty class.
"""
import hashlib, json, re
from pathlib import Path
from common import ASSETS, PAPER_TABLES, stats

FM_ROW = r'\name{}'   # Table 2's FM (adaptive) row label since the method-name macros (paper commit 811e036)
FIELDS = {'rec': 'c6c_rec', 'fail_rec': 'c6c_fail_rec', 'fdr': 'c6c_fp', 'for': 'c6c_ferr'}
META = json.loads((ASSETS / 'generalized_6c_values.json').read_text())
SRC = Path(META['source'])
SRC_SHA256 = META['source_sha256']


def load_rows():
    """(name, seed) -> matrix_rows row, for every FM run; fails if the pinned file changed."""
    assert hashlib.sha256(SRC.read_bytes()).hexdigest() == SRC_SHA256, 'matrix_rows.jsonl changed'
    rows = {}
    for line in SRC.read_text().splitlines():
        r = json.loads(line)
        if r.get('name') and r.get('model') == 'FM':
            assert (r['name'], r['seed']) not in rows, (r['name'], r['seed'])
            rows[r['name'], r['seed']] = r
    return rows


def cell(final, rows, setting, arm, rho, metric):
    """Mean and min-max in % over the three seeds of one FM arm, from the certified fields."""
    rr = [rows[r['name'], r['seed']] for r in final
          if (r['setting'], r['acq'].replace('τ', 'tau'), r['d2_ratio']) == (setting, arm, rho)]
    assert len(rr) == 3 and len({r['seed'] for r in rr}) == 3, (setting, arm, rho)
    key = FIELDS[metric]
    return stats(setting, arm, rho, metric, [100 * r[key] for r in rr if r[key] is not None])


def check_table2(mean_at):
    """Fail unless mean_at(setting, metric) equals, to 2 dp, the FM_ROW row of the paper's
    tables/matrix_certified_body.tex (U_H, rho = 0.75). Returns the 24 checked cells."""
    # One tabular holds both panels since the tables lost their \par\medskip split, so take the method's two rows in
    # order. The row prints as \nameOurs{} since 2026-11-13 (make_tables.OURS_LABEL); older bodies have \name{}.
    text = (PAPER_TABLES / 'matrix_certified_body.tex').read_text()
    body = [l for l in text.splitlines() if l.startswith((FM_ROW + ' &', r'\nameOurs{} &'))]
    assert len(body) == 2, len(body)
    order = {0: ['pen-lqr', 'cp-lqr', 'q3d-lqr'], 1: ['cp-ppo', 'q2d-ppo', 'q3d-ppo']}
    metrics = ['rec', 'fdr', 'fail_rec', 'for', 'abstain']
    checked = []
    for panel, line in enumerate(body):
        cells = [c.strip() for c in line.rstrip(' \\').split('&')[1:]]
        assert len(cells) == 15
        for i, tex in enumerate(cells):
            setting, metric = order[panel][i // 5], metrics[i % 5]
            if metric == 'abstain':
                continue
            m = re.search(r'([0-9]+\.[0-9]{2})', tex)
            mine = mean_at(setting, metric)
            assert m and mine is not None and f'{mine:.2f}' == m.group(1), (setting, metric, tex, mine)
            checked.append(dict(setting=setting, metric=metric, table=m.group(1), figure=round(mine, 6)))
    assert len(checked) == 24
    return checked
