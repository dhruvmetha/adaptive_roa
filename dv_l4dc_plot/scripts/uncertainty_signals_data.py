"""Data for the pendulum uncertainty-signal figure (paper figure pendulum_uncertainty_signals) and its statistics.

Runs: TERA-FM on Pendulum LQR, adaptive (U_H, rho=0.75, x231_*) and uniform (m65_*), seeds 99512 / 33407 / 24475.
Per run and round, from the run's own files only:
  test predictions   epoch_<r>/full_roa_per_point.npz (20,702 test states on a 158 x 315 lattice)
  thresholds         that run's saved A-1d thresholds for that round (epochs.jsonl c6c_t_s, t_f, u_s, u_f), never
                     another run's; rounds where two-sided certification fails are excluded and reported
  training states    every state (last one included) of the run's first N_r training trajectories, taken from the
                     pendulum pool train.npz by trajectory ID (datasets/train_trajectories.txt). The shared pool copy
                     is verified against each run's own datasets/train_endpoint_dataset.txt before use.
Decisions use the paper's membership code (certificates.in_S / in_F, det_v3 worktree) and must reproduce the stored
decided-set sizes. Signals: I = p_inv, D = (1 - p_inv) H2(p_s / (p_s + p_f)) in bits (D = 0 with no valid endpoint).
Abstention kinds as in sections/results_uncertainty.tex: invalid (p_s <= t_s, p_f <= u_f), weak (t_s < p_s <= u_s,
u_f < p_f <= t_f), outcome-uncertain (p_s > u_s, p_f > t_f); no priority rule, overlaps and uncategorized
abstentions are counted. Coordinates (theta / 2pi wrapped, omega / 4pi): sparse = nearest training state farther than
0.02; near boundary = nearest opposite-label test state closer than 0.01.
"""
import csv, hashlib, json, sys
from pathlib import Path
import numpy as np
import yaml
from scipy.spatial import cKDTree
from common import REPO, SCORED

DETV3 = REPO                                 # the adaptive_roa checkout this folder sits in
sys.path.insert(0, str(DETV3))
from adaptive_roa.adaptive_v2.eval import certificates as C, det_v2_rule as R      # noqa: E402

INDEX = Path('/common/users/shared/pracsys/adaptive_roa_experiments/det_v3_seeds/_index/runs_index.tsv')
POOL = Path('/common/users/shared/pracsys/genMoPlan/data_trajectories/deterministic/pendulum/lqr_failset/splits_det_rerun/train.npz')
SEEDS = [99512, 33407, 24475]
ARMS = [('adaptive', 'method'), ('uniform', 'uniform data')]
SPARSE, NEAR = 0.02, 0.01
L = 10.0                                   # box for the non-periodic omega axis (data span ~1)


def runs():
    rows = [r for r in csv.DictReader(open(INDEX), delimiter='\t')
            if r['setting'] == 'pen-lqr' and r['model'] == 'FM' and r['role'] in ('method', 'uniform data')]
    out = {}
    for arm, role in ARMS:
        for seed in SEEDS:
            r = next(r for r in rows if r['role'] == role and int(r['seed']) == seed)
            out[arm, seed] = dict(name=r['run'], dir=Path(r['real_dir']))
    return out


def norm(x):
    """(theta / 2pi wrapped to [0, 1), omega / 4pi shifted into [0, L)) for a periodic-in-theta KD-tree."""
    a = (x[:, 0] / (2 * np.pi)) % 1.0
    b = x[:, 1] / (4 * np.pi) + L / 2
    return np.c_[a, b]


def tree(x):
    return cKDTree(norm(x), boxsize=[1.0, L])


def h2(q):
    q = np.clip(q, 0, 1)
    with np.errstate(divide='ignore', invalid='ignore'):
        h = -(q * np.log2(q) + (1 - q) * np.log2(1 - q))
    return np.nan_to_num(h)


class Pool:
    def __init__(self):
        z = np.load(POOL)
        self.S, self.O = z['states'], z['offsets']
        self.sha = hashlib.sha256(POOL.read_bytes()).hexdigest()

    def states(self, ids, drop_last=False):
        return np.concatenate([self.S[self.O[i]:self.O[i + 1] - (1 if drop_last else 0)] for i in ids])


def verify_pool(pool, run):
    """The shared pool must reproduce the run's own training rows exactly (they omit each trajectory's last state)."""
    ids = [int(x) for x in (run['dir'] / 'datasets/train_trajectories.txt').read_text().split()]
    ep = np.loadtxt(run['dir'] / 'datasets/train_endpoint_dataset.txt', dtype=np.float32)
    mine = pool.states(ids, drop_last=True)
    assert len(mine) == len(ep) and np.allclose(mine, ep[:, :2], atol=1e-5), run['name']
    cfg = yaml.safe_load(open(run['dir'] / 'resolved_config.yaml'))
    return ids, cfg


def epochs():
    rows = {}
    for l in open(SCORED / 'epochs.jsonl'):
        r = json.loads(l)
        if r['setting'] == 'pen-lqr':
            rows[r['name'], r['seed'], r['epoch']] = r
    return rows


def analyse(run, seed, rnd, ep, pool, ids):
    """All per-state arrays and the statistics for one run and round, or None when certification failed."""
    e = ep[run['name'], seed, rnd]
    art = json.load(open(run['dir'] / f'epoch_{rnd:03d}/artifacts_v2.json'))
    n_train = art['train_trajectories']
    assert e['train'] == n_train
    z = np.load(run['dir'] / f'epoch_{rnd:03d}/full_roa_per_point.npz', allow_pickle=False)
    X = z['start_states'].astype(float)
    ps, pf, y = C.split_probs(z)
    pinv_file = np.asarray(z['p_invalid'], float)
    assert np.allclose(np.round(pinv_file, 9), np.round(1 - z['p_success'] - z['p_failure'], 9), atol=1e-6)
    pinv = R.tie(pinv_file)
    q = np.where(ps + pf > 0, ps / np.where(ps + pf > 0, ps + pf, 1), 0.5)
    I, D = pinv, np.where(ps + pf > 0, (1 - pinv) * h2(q), 0.0)
    base = dict(arm=None, seed=seed, round=rnd, run=run['name'], n_train=n_train, cert_pass=bool(e['c6c_pass']))
    if not e['c6c_pass']:
        return dict(stats=base | dict(excluded='two-sided certification failed'), arrays=None)
    t_s, t_f, u_s, u_f = (e['c6c_t_s'], e['c6c_t_f'], e['c6c_u_s'], e['c6c_u_f'])
    S, F = C.in_S(ps, pf, t_s, t_f), C.in_F(ps, pf, u_s, u_f)
    succ, fail = S & ~F, F & ~S
    abst = ~(succ | fail)
    assert int(succ.sum()) == e['c6c_S'] and int(fail.sum()) == e['c6c_F'], (run['name'], rnd, succ.sum(), e['c6c_S'])
    T = lambda v: R.tie(v)
    inv = (ps <= T(t_s)) & (pf <= T(u_f))
    weak = (ps > T(t_s)) & (ps <= T(u_s)) & (pf > T(u_f)) & (pf <= T(t_f))
    ou = (ps > T(u_s)) & (pf > T(t_f))
    assert not (inv & ~abst).any() and not (weak & ~abst).any() and not (ou & ~abst).any()
    other = abst & ~(inv | weak | ou)
    # training coverage and boundary proximity
    Xtr = pool.states(ids[:n_train])
    d_train, _ = tree(Xtr).query(norm(X))
    d_opp = np.empty(len(X))
    for lab in (True, False):
        m = y == lab
        d_opp[m], _ = tree(X[~m]).query(norm(X[m]))
    sparse, near = d_train > SPARSE, d_opp < NEAR
    wrong = (succ & ~y) | (fail & y)
    n = len(y)
    frac = lambda m, w=None: (float(m[w].mean()) if w is not None and w.any() else None) if w is not None else float(m.mean())
    # Strict categories are primary. The two variants reproduce how the manuscript's numbers were computed (its weak
    # figures use every abstention that is neither invalid nor outcome-uncertain; its round-2 invalid shares leave
    # out states that are both invalid and outcome-uncertain). They are reported, never used to resolve overlaps.
    cats = dict(success=succ, fail=fail, abstain=abst, invalid=inv, weak=weak, outcome_uncertain=ou, other_abstain=other,
                weak_residual=abst & ~inv & ~ou, invalid_excl_overlap=inv & ~ou)
    st = base | dict(thresholds=dict(t_s=t_s, t_f=t_f, u_s=u_s, u_f=u_f), t_s_plus_u_f=round(t_s + u_f, 9),
                     ordering_ts_lt_us=t_s < u_s, ordering_uf_lt_tf=u_f < t_f, n_test=n,
                     n_train_states=len(Xtr), baseline_sparse=frac(sparse), baseline_near=frac(near),
                     baseline_near_and_sparse=frac(near & sparse),
                     overlap_invalid_weak=int((inv & weak).sum()), overlap_invalid_ou=int((inv & ou).sum()),
                     overlap_weak_ou=int((weak & ou).sum()), wrong_committed=int(wrong.sum()))
    for k, m in cats.items():
        st[k] = dict(count=int(m.sum()), frac_of_test=frac(m), frac_in_sparse=frac(sparse, m), frac_near=frac(near, m),
                     mean_pinv=float(pinv[m].mean()) if m.any() else None)
    st['invalid_among_near_sparse'] = frac(inv, near & sparse)
    st['invalid_among_near_dense'] = frac(inv, near & ~sparse)
    arrays = dict(X=X, y=y, I=I, D=D, succ=succ, fail=fail, inv=inv, weak=weak, ou=ou, other=other, wrong=wrong,
                  sparse=sparse, near=near, d_train=d_train, d_opp=d_opp)
    return dict(stats=st, arrays=arrays)


def build(rounds=(2, 10)):
    pool, ep, R_ = Pool(), epochs(), runs()
    stats, arrays, sums, prov = [], {}, [], {}
    for (arm, seed), run in sorted(R_.items()):
        ids, cfg = verify_pool(pool, run)
        prov[f'{arm}/{seed}'] = dict(run=run['name'], dir=str(run['dir']),
                                     commit=json.load(open(run['dir'] / 'provenance.json')).get('commit'),
                                     config_seed=cfg.get('seed'), K=cfg.get('num_mc_samples'),
                                     config_pool=cfg['data']['trajectories_dir'] if 'data' in cfg and 'trajectories_dir' in cfg['data'] else None,
                                     checkpoints=sorted(p.name for p in (run['dir'] / 'epoch_010/checkpoints').glob('*')))
        for rnd in range(11):
            e = ep[run['name'], seed, rnd]
            if e['c6c_pass']:
                sums.append(dict(arm=arm, seed=seed, round=rnd, t_s=e['c6c_t_s'], u_f=e['c6c_u_f'],
                                 t_s_plus_u_f=round(e['c6c_t_s'] + e['c6c_u_f'], 9)))
        for rnd in rounds:
            a = analyse(run, seed, rnd, ep, pool, ids)
            a['stats']['arm'] = arm
            stats.append(a['stats'])
            if a['arrays'] is not None:
                arrays[arm, seed, rnd] = a['arrays']
    return dict(stats=stats, arrays=arrays, threshold_sums=sums, provenance=prov, pool_sha256=pool.sha)


if __name__ == '__main__':
    out = build()
    for s in out['stats']:
        if not s['cert_pass']:
            print(s['arm'], s['seed'], s['round'], 'EXCLUDED', s['excluded']); continue
        th = s['thresholds']
        print(f"{s['arm']:8s} {s['seed']} r{s['round']:<2d} thr {th['t_s']},{th['t_f']},{th['u_s']},{th['u_f']} "
              f"order ts<us {s['ordering_ts_lt_us']} uf<tf {s['ordering_uf_lt_tf']} | inv {s['invalid']['frac_of_test']:.4f} "
              f"weak {s['weak']['frac_of_test']:.4f} ou {s['outcome_uncertain']['frac_of_test']:.4f} other {s['other_abstain']['count']} "
              f"| overlaps inv&ou {s['overlap_invalid_ou']} | base sparse {s['baseline_sparse']:.3f} near {s['baseline_near']:.3f}")
    print('max t_s+u_f over certified rounds:', max(x['t_s_plus_u_f'] for x in out['threshold_sums']))
