"""Certified adaptive-fraction figure: paper figure adaptive_fraction_certified (candidate next to Fig 4).

Same 72 FM runs as Fig 4 (uniform at rho=0, U_H at rho .5/.75/1, six settings x three seeds), scored
with the two-sided A-1d certificate at alpha_Succ = alpha_Fail = delta = 0.05: the c6c_* fields of
the shared _index/scored/matrix_rows.jsonl, pinned by sha256 in generalized_6c_values.json. Metrics
are computed as make_tables.py does for matrix_certified (Table 2): R_s = c6c_rec, R_f = c6c_fail_rec,
FDR = c6c_fp, FOR = c6c_ferr, final budget.

Layout of Fig 4 (split.py): recall solid over FDR/FOR dash-dot, solid gray 5% line, four-entry legend,
1:1 panels, y axes shared within each row. Line = three-seed mean, band = min to max over seeds
(common.spread). Recall averages all three seeds (an empty predicted class has recall 0). FDR/FOR use
only seeds with a nonempty predicted class: hollow marker below 3 seeds, a gap at 0, band only where
at least 2 seeds contribute.

The script fails unless rho=0.75 reproduces the FM adaptive row of the paper's Table 2 cell by cell.
With --paper it is copied to figures/results/adaptive_fraction_certified.pdf.
Since 2026-11-13 the paper figure is the common.py paper-style draw at print width,
plots/adaptive_fraction_certified.pdf (split.py restyle block); it fails unless its numbers equal the stored values
JSON. A plain run first rebuilds values/ and the old-style draw (now in plots/legacy/), then the paper figure.
--restyle redraws only the paper figure; --paper copies it into the paper clone.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from common import VALUES, SETTINGS, TITLES, RECALL, ERROR, RC, load_final, method_names
from common import install_paper
from certified import SRC, SRC_SHA256, FIELDS, load_rows, cell, check_table2
import split
from split import split_figure, REF_SOLID, DASHDOT, LEGEND_FOUR, restyle_args, split_figure_restyled

ARGS = restyle_args(__doc__)
COMBOS = [('uniform', 0.), ('U_H', .5), ('U_H', .75), ('U_H', 1.)]
RHO = np.array([rho for _, rho in COMBOS])
NAMES = method_names()
# Per-row y axes from this data. RL recall reaches 0 (Planar Quad RL uniform: two seeds certify
# nothing), so that row spans 0-100. Both error rows top out below 5 (max 4.79), so 0-6 keeps the
# 5% line in view.
ROW_AXES = {0: dict(rec=((48, 102), [50, 75, 100]), err=((-.18, 6), [0, 5])),
            1: dict(rec=((-3, 103), [0, 50, 100]), err=((-.18, 6), [0, 5]))}

rows = load_rows()
final = load_final()   # names and seeds of the FM runs, with their acquisition score and rho
values = [cell(final, rows, s, a, rho, m) for s in SETTINGS for a, rho in COMBOS for m in FIELDS]
lookup = {(v['setting'], v['rho'], v['metric']): v for v in values}


def series(setting, metric):
    cc = [lookup[(setting, rho, metric)] for rho in RHO]
    f = lambda k: np.array([np.nan if c[k] is None else c[k] for c in cc], dtype=float)
    return f('mean'), f('min'), f('max'), np.array([c['n'] for c in cc])


checked = check_table2(lambda s, m: lookup[(s, .75, m)]['mean'])
# --restyle (dm1487, 2026-11-13): Figure S2 in the common.py paper style at its 6.00 in print width, into
# plots/ (the paper figure). Panel shapes are the paper figure's: recall and error panels 0.75 in tall, 1.70 in wide.
RESTYLE_ASPECT, RESTYLE_ERR_RATIO = .7506 / 1.6997, 1.


def restyled():
    """Same layout and data, paper style: colour = the metric's side, recall solid, error dash-dot, and the method by
    marker only (common.METHOD_MARKER): rho = 0 is Unif-TERA (open square), rho > 0 is TERA with U_H (filled circle);
    a faded marker means 1-2 seeds contribute. Fails unless the cells equal the stored
    values/adaptive_fraction_certified_values.json. Writes nothing to values/ or the paper."""
    stored = json.loads((VALUES / 'adaptive_fraction_certified_values.json').read_text())
    assert json.loads(json.dumps(values)) == stored['values'], 'restyle cells differ from values/'
    assert stored['source_sha256'] == SRC_SHA256 and json.loads(json.dumps(checked)) == stored['table2_check']
    row_base = {}
    for row, settings in enumerate([SETTINGS[:3], SETTINGS[3:]]):
        rec = [v for v in values if v['setting'] in settings and v['metric'] in ('rec', 'fail_rec') and v['n']]
        ticks = ROW_AXES[row]['rec'][1]
        row_base[row] = dict(rec=(min(ticks[0], *(v['min'] for v in rec)), max(ticks[-1], *(v['max'] for v in rec)), ticks),
                             err=(0, ROW_AXES[row]['err'][0][1], ROW_AXES[row]['err'][1]))
    out = split_figure_restyled(
        'adaptive_fraction_certified', arms=[(lambda s, m: (RHO, *series(s, m)),
                                              [NAMES['nameUnif'] if a == 'uniform' else NAMES['name'] for a, _ in COMBOS])],
        xticks={s: (RHO, ['0', '.5', '.75', '1']) for s in SETTINGS}, xlim=(-.05, 1.05),
        xlabel=rf"Adaptive fraction $\rho$  ($\rho=0$ is {NAMES['nameUnif']})", titles=TITLES, row_base=row_base,
        metric_labels=LEGEND_FOUR, legend_methods=[NAMES['name'], NAMES['nameUnif']],
        ylabels=('Recall (%) ↑', 'Error (%) ↓'), rec_aspect=RESTYLE_ASPECT, err_ratio=RESTYLE_ERR_RATIO, band_alpha=.12)
    g = out['geometry']
    print(f"  panels {g['panel_w']:.3f} in wide, recall {g['rec_h']:.3f} in, error {g['err_h']:.3f} in; "
          f"legend {out['legend_ncol']} columns; layout settled in {g['iterations']} passes")
    print(f"  outer whitespace (in): {', '.join(f'{k} {v:.3f}' for k, v in out['edges'].items())}")
    print(f"  y limits: {out['ylims']}")
    print(f"  {len(values)} cells equal values/adaptive_fraction_certified_values.json; Table 2 cross-check "
          f"{len(checked)}/24")
    return out


if ARGS.restyle:
    restyled()
    if ARGS.paper:
        install_paper('adaptive_fraction_certified')
    raise SystemExit
with plt.rc_context(RC):
    size = split_figure(series, RHO, ['0', '.5', '.75', '1'], (-.05, 1.05),
                        rf"Adaptive fraction $\rho$  ($\rho=0$ is {NAMES['nameUnif']})", 'adaptive_fraction_certified',
                        height_ratios=(1, 1), row_axes=ROW_AXES, err_ls=DASHDOT, ref=REF_SOLID, legend_four=True,
                        min_band_n=2, paper_stem='adaptive_fraction_certified', paper=False,
                        ms=split.PAPER_MS, mew=split.PAPER_MEW)
(VALUES / 'adaptive_fraction_certified_values.json').write_text(json.dumps(dict(
    source=str(SRC), source_sha256=SRC_SHA256,
    protocol='FM, uniform and U_H at rho .5/.75/1, A-1d two-sided certificate at alpha=delta=0.05 (c6c_*), '
             'final budget; mean with min-max over 3 seeds; FDR/FOR over seeds with nonempty S/F only; '
             'bands only where >= 2 seeds contribute',
    row_axes=ROW_AXES, page_size=size, table2_check=checked, values=values), indent=2) + '\n')
partial = [v for v in values if v['n'] < 3]
print(f'Table 2 cross-check: {len(checked)}/24 cells match. Partial cells:',
      ', '.join(f"{v['setting']} rho={v['rho']} {v['metric']} n={v['n']}" for v in partial) or 'none')
restyled()                                  # the paper figure, checked against the values/ just written
if ARGS.paper:
    install_paper('adaptive_fraction_certified')
