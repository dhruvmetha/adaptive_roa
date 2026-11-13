"""Data for the predictor comparison on identical uniform training data (paper figure predictor_comparison_uniform).

All five predictors (TERA-FM, TERA-MLP, TERA-BNN, MLP-Label, BNN-Label) trained on the same UNIFORM trajectories
within each setting and seed, final round (epoch_010), final budgets 400 / 2,000 / 2,000 / 12,000 / 30,000 /
40,000. Each final checkpoint is evaluated two ways on the setting's test set:
  val  thresholds fitted on D_val with the current objective at w=0.2 (soft_four_wsweep.jsonl, w=0.2)
  cal  the two-sided A-1d certificate at alpha_s = alpha_f = delta = 0.05 (c6c_* of matrix_rows.jsonl)

Nothing is taken on trust: every point is recomputed from the run's per-state test predictions
(epoch_010/full_roa_per_point.npz) and the stored thresholds with the paper's membership code
(certificates.in_S / in_F from the det_v3 worktree), and must reproduce the stored counts exactly.
Label models use p_s only (p_f = None). Per seed: R_s = TP/P, R_f = TN/N (abstained states count against
recall), FDR = FP/(TP+FP), FOR = FN/(FN+TN) (undefined, never zero, for an empty declaration set),
abstention = abstained/n. Calibration upper bounds (c6c_cal_bs/bf) are kept apart from test measurements.

Checks that fail loudly: identical training trajectories, D_val, D_cal and test sets across the five models
of each (setting, seed); budgets; recomputed counts; TERA-FM val points = Table 1's \\nameUnif{} row;
TERA-FM cal points = the uniform round-10 end points of adaptive_learning_a1d.
"""
import csv, hashlib, json, re, sys, collections
from pathlib import Path
import numpy as np
import yaml
from common import ASSETS, PAPER_TABLES, VALUES, HERE, REPO, SETTINGS, method_names, spread
import certified

DETV3 = REPO                                 # the adaptive_roa checkout this folder sits in
sys.path.insert(0, str(DETV3))
from adaptive_roa.adaptive_v2.eval import certificates as C          # noqa: E402

INDEX = Path('/common/users/shared/pracsys/adaptive_roa_experiments/det_v3_seeds/_index/runs_index.tsv')
BUDGET = {'pen-lqr': 400, 'cp-lqr': 2000, 'cp-ppo': 2000, 'q2d-ppo': 12000, 'q3d-lqr': 30000, 'q3d-ppo': 40000}
# Names follow the paper's method-names.tex (common.method_names()). Every run here has role 'uniform data'
# (load() asserts it), and the macro comes from (role, model) via make_tables.macro_name, never from label text.
# Keys remain the immutable runs_index model names.
from make_tables import macro_name                               # noqa: E402
ROLE = 'uniform data'
NAMES = method_names()
MODELS = [(m, '\\' + macro_name(ROLE, m), NAMES[macro_name(ROLE, m)])
          for m in ('FM', 'MLP final-state', 'BNN final-state', 'MLP classifier', 'BNN classifier')]
LABEL_MODELS = {'MLP classifier', 'BNN classifier'}
METRICS = ['rec', 'fdr', 'fail_rec', 'for', 'abstain']


GENMO = Path('/common/users/shared/pracsys/genMoPlan/data_trajectories')
FOREIGN = ['/scratch/dm1487/genMoPlan-exp/data_trajectories/', '/home/st1122/det_rerun_data/']


def resolve(path):
    """A config path, or its genMoPlan copy when the configured location is unreadable here (other hosts' /scratch,
    st1122's home). Returns (readable path or None, 'configured' / 'genMoPlan copy' / None)."""
    if Path(path).exists():
        return Path(path), 'configured'
    for pre in FOREIGN:
        if path.startswith(pre) and (GENMO / path[len(pre):]).exists():
            return GENMO / path[len(pre):], 'genMoPlan copy'
    return None, None


def sha(x):
    return hashlib.sha256(x if isinstance(x, bytes) else str(x).encode()).hexdigest()


def find(c, key):
    if isinstance(c, dict):
        for k, v in c.items():
            if k == key:
                return v
            x = find(v, key)
            if x is not None:
                return x
    return None


def arrays_sha(path, keys):
    z = np.load(path, allow_pickle=False)
    return sha(b''.join(np.ascontiguousarray(z[k]).tobytes() for k in keys))


def counts(ps, pf, y, thr):
    """(TP, FP, FN, TN) of the decided sets; thr = (t_s, t_f, u_s, u_f) or None for the all-Abstain predictor."""
    if thr is None:
        return 0, 0, 0, 0
    t_s, t_f, u_s, u_f = thr
    S, F = C.in_S(ps, pf, t_s, t_f), C.in_F(ps, pf, u_s, u_f)
    sd, fd = S & ~F, F & ~S
    return tuple(int(m.sum()) for m in (sd & y, sd & ~y, fd & y, fd & ~y))


def rates(tp, fp, fn, tn, n, P, N):
    return dict(rec=100 * tp / P, fail_rec=100 * tn / N, fdr=100 * fp / (tp + fp) if tp + fp else None,
                **{'for': 100 * fn / (fn + tn) if fn + tn else None}, abstain=100 * (n - tp - fp - fn - tn) / n)


def load():
    runs = [r for r in csv.DictReader(open(INDEX), delimiter='\t') if r['role'] == 'uniform data']
    assert len(runs) == 90, len(runs)
    soft = {(r['name'], r['seed']): r for l in open(ASSETS / 'soft_four_wsweep.jsonl') if (r := json.loads(l))['w'] == .2}
    assert hashlib.sha256(certified.SRC.read_bytes()).hexdigest() == certified.SRC_SHA256
    mrows = {(r['name'], r['seed']): r for l in open(certified.SRC) if (r := json.loads(l)).get('name')}
    pools, out = {}, []
    for r in runs:
        d, seed = Path(r['real_dir']), int(r['seed'])
        art = json.load(open(d / 'epoch_010/artifacts_v2.json'))
        n10 = art['train_trajectories']
        assert n10 == BUDGET[r['setting']], (r['run'], n10)
        cfg = yaml.safe_load(open(d / 'resolved_config.yaml'))
        pool = find(cfg, 'shuffled_indices_file')
        if pool not in pools:
            rp, how = resolve(pool)
            pools[pool] = (rp.read_text().split(), how, str(rp)) if rp else (None, None, None)
        idx = json.load(open(d / 'dataset_builder_state.json'))['train_indices'][:n10]
        ids = [pools[pool][0][i] for i in idx] if pools[pool][0] else None
        txt = d / 'datasets/train_trajectories.txt'
        txt_ids = txt.read_text().split()[:n10] if txt.exists() else None
        if ids is not None and txt_ids is not None:
            assert ids == txt_ids, (r['run'], 'pool order copy does not reproduce the run\'s own trajectory list')
        e = d / 'epoch_010'
        rec = dict(setting=r['setting'], model=r['model'], seed=seed, run=r['run'], dir=str(d), n_train=n10,
                   train_idx_sha=sha(' '.join(map(str, idx))), train_ids_sha=sha(' '.join(ids or txt_ids)) if (ids or txt_ids) else None,
                   train_ids_from=(f'pool order ({pools[pool][1]})' if ids else 'train_trajectories.txt' if txt_ids
                                   else 'indices only (pool file unreadable)'),
                   train_ids_checked_against_own_list=bool(ids and txt_ids),
                   pool_file=pool, pool_file_read=pools[pool][2], trajectories_file=find(cfg, 'trajectories_dir'),
                   val_states_sha=arrays_sha(d / 'prediction_store_states_val.npz', ['states', 'true_labels']),
                   cal_states_sha=arrays_sha(d / 'prediction_store_states_cal.npz', ['states', 'true_labels']),
                   test_states_sha=arrays_sha(e / 'full_roa_per_point.npz', ['start_states', 'true_labels']),
                   config_seed=cfg.get('seed'),
                   K=find(cfg, 'num_mc_samples'), bnn_weight_samples=find(cfg, 'n_marginal_samples') if 'BNN' in r['model'] else None,
                   commit=json.load(open(d / 'provenance.json')).get('commit'),
                   checkpoints=';'.join(sorted(p.name for p in (e / 'checkpoints').glob('*'))))
        ps, pf, y = C.split_probs(np.load(e / 'full_roa_per_point.npz', allow_pickle=False))
        if r['model'] in LABEL_MODELS:
            pf = None
        n, P, N = len(y), int(y.sum()), int((~y).sum())
        s, m = soft[(r['run'], seed)], mrows[(r['run'], seed)]
        assert s['model'] == m['model'] and s['matrix_role'] == m['matrix_role'] == 'uniform data'
        thr_val = s['val']['thresholds']
        c_val = counts(ps, pf, y, thr_val)
        assert list(c_val) == s['test']['counts'] and s['test']['n'] == n, (r['run'], c_val, s['test']['counts'])
        thr_cal = None if m['c6c_t_s'] is None else (m['c6c_t_s'], m['c6c_t_f'], m['c6c_u_s'], m['c6c_u_f'])
        c_cal = counts(ps, pf, y, thr_cal)
        rv, rc = rates(*c_val, n, P, N), rates(*c_cal, n, P, N)
        assert c_cal[0] + c_cal[1] == m['c6c_S'] and c_cal[2] + c_cal[3] == m['c6c_F'], (r['run'], c_cal, m['c6c_S'], m['c6c_F'])
        for k, mk in [('rec', 'c6c_rec'), ('fail_rec', 'c6c_fail_rec'), ('fdr', 'c6c_fp'), ('for', 'c6c_ferr'), ('abstain', 'c6c_abstain')]:
            st = m[mk]
            assert (st is None) == (rc[k] is None) and (st is None or abs(100 * st - rc[k]) < 1e-9), (r['run'], k, st, rc[k])
        for method, c, rr, thr in [('val', c_val, rv, thr_val), ('cal', c_cal, rc, thr_cal)]:
            out.append(dict(rec, method=method, n_test=n, P=P, N=N, TP=c[0], FP=c[1], FN=c[2], TN=c[3],
                            S=c[0] + c[1], F=c[2] + c[3], all_abstain=c[0] + c[1] + c[2] + c[3] == 0,
                            thresholds=None if thr is None else ' '.join(map(str, thr)), **rr,
                            cert_pass=m['c6c_pass'] if method == 'cal' else None,
                            cal_upper_bound_fdr=m['c6c_cal_bs'] if method == 'cal' else None,
                            cal_upper_bound_for=m['c6c_cal_bf'] if method == 'cal' else None))
    return out


def check_identity(rows):
    """Five models of each (setting, seed) share training trajectories; each setting shares D_val, D_cal, test."""
    report = []
    by = collections.defaultdict(list)
    for r in rows:
        if r['method'] == 'val':
            by[r['setting'], r['seed']].append(r)
    for (s, seed), v in sorted(by.items()):
        assert len(v) == 5
        assert len({x['train_idx_sha'] for x in v}) == 1, (s, seed, 'train indices differ')
        assert len({x['train_ids_sha'] for x in v if x['train_ids_sha']}) == 1, (s, seed, 'trajectory IDs differ')
        assert len({Path(x['pool_file']).name for x in v}) == 1, (s, seed, 'pool order files differ')
        assert all(x['train_ids_sha'] for x in v), (s, seed, 'some model lacks trajectory IDs')
        report.append(dict(setting=s, seed=seed, models=len(v), id_level=sum(1 for x in v if x['train_ids_sha']),
                           ids_checked_against_own_list=sum(1 for x in v if x['train_ids_checked_against_own_list']),
                           pool_files_read=sorted({x['pool_file_read'] for x in v})))
    for s in SETTINGS:
        v = [x for x in rows if x['setting'] == s and x['method'] == 'val']
        for k in ('val_states_sha', 'cal_states_sha', 'test_states_sha'):
            assert len({x[k] for x in v}) == 1, (s, k, 'differs across models or seeds')
    return report


def aggregate(rows):
    """(setting, model, method) -> metric -> dict(mean, min, max, n): recall/abstention over all seeds, errors over
    seeds with a nonempty declaration set."""
    agg = {}
    by = collections.defaultdict(list)
    for r in rows:
        by[r['setting'], r['model'], r['method']].append(r)
    for k, v in by.items():
        assert len(v) == 3
        agg[k] = {}
        for m in METRICS:
            xs = [x[m] for x in v if x[m] is not None]
            mean, lo, hi = spread(xs)
            agg[k][m] = dict(mean=mean, min=lo, max=hi, n=len(xs))
        agg[k]['all_abstain_seeds'] = sum(x['all_abstain'] for x in v)
    return agg


def check_table1(agg):
    """TERA-FM val points = Table 1's \\nameUnif{} row (2 dp, same seed-count superscripts)."""
    body = (PAPER_TABLES / 'matrix_uncertified_body.tex').read_text().split(r'\par\medskip')
    order = {0: ['pen-lqr', 'cp-lqr', 'q3d-lqr'], 1: ['cp-ppo', 'q2d-ppo', 'q3d-ppo']}
    checked, bad = 0, []
    for panel, text in enumerate(body):
        line = next(l for l in text.splitlines() if l.startswith(r'\nameUnif{} &'))
        cells = [c.strip() for c in line.rstrip(' \\').split('&')[1:]]
        for i, tex in enumerate(cells):
            s, m = order[panel][i // 5], METRICS[i % 5]
            a = agg[s, 'FM', 'val'][m]
            mine = '--' if a['n'] == 0 else f"{a['mean']:.2f}" + (f"^{{({a['n']})}}" if a['n'] < 3 else '')
            theirs = tex.replace('$', '').replace(r'\mathbf{', '').replace('}', '', 1 if r'\mathbf{' in tex else 0)
            if mine != theirs:
                bad.append((s, m, theirs, mine))
            checked += 1
    assert not bad, f'Table 1 cross-check failed: {bad}'
    return checked


def check_fig3(agg):
    """TERA-FM cal points = uniform round-10 end points of adaptive_learning_a1d (same seed counts)."""
    pts = json.load(open(VALUES / 'adaptive_learning_a1d_values.json'))['points']
    checked, bad = 0, []
    for s in SETTINGS:
        for m in ['rec', 'fdr', 'fail_rec', 'for']:
            p = next(p for p in pts if (p['setting'], p['arm'], p['metric'], p['epoch']) == (s, 'uniform', m, 10))
            a = agg[s, 'FM', 'cal'][m]
            same = p['n'] == a['n'] and ((p['mean'] is None and a['mean'] is None) or abs(p['mean'] - a['mean']) < 1e-9)
            if not same:
                bad.append((s, m, p['mean'], p['n'], a['mean'], a['n']))
            checked += 1
    assert not bad, f'Fig 3 cross-check failed: {bad}'
    return checked


def build():
    rows = load()
    ident = check_identity(rows)
    agg = aggregate(rows)
    n1, n2 = check_table1(agg), check_fig3(agg)
    return rows, ident, agg, (n1, n2)


if __name__ == '__main__':
    rows, ident, agg, (n1, n2) = build()
    print(f'{len(rows)} per-seed points; identity groups {len(ident)}; Table 1 check {n1}/30 cells; Fig 3 check {n2}/24 cells')
    for (s, m, meth), a in sorted(agg.items()):
        print(s, m, meth, ' '.join(f"{k}={a[k]['mean']:.2f}({a[k]['n']})" if a[k]['mean'] is not None else f'{k}=undef(0)'
                                   for k in METRICS), 'allAbst', a['all_abstain_seeds'])
