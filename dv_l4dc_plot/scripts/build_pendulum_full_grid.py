"""Full 158 x 315 pendulum grid for one round of the two Figure 5 runs (TERA s99512 and Unif-TERA s99512).

The round-10 grids in data/pendulum_full_grid/ were built by dm1487 (provenance JSON next to each npz). This script
builds the same file for another round, with the round-10 file as the lattice template:
  * start_states, theta/omega indices, true labels, sources and sequence ids are copied from the template;
  * p_success / p_failure / p_invalid are the run's STORED values where it stored one (source test, cal, val), else
    the fresh evaluation (the round-10 rule);
  * p_*_fresh is one fresh K = 100 evaluation of all 49,770 states with the round's checkpoint: 100 Euler steps,
    attractor radius 0.1, torch.manual_seed(99512) once before the first batch.
The fresh pass calls the flow matcher directly (predict_endpoint + system.classify_attractor), not
FullROAEvaluator; the provenance reports stored-vs-fresh agreement on the test rows so the MC noise is visible.

Run on a GPU with PYTHONPATH at a checkout of the runs' commit (acae65bf33), e.g.
  PYTHONPATH=<worktree at acae65bf33> python build_pendulum_full_grid.py --epoch 2
"""
import argparse
import hashlib
import json
import socket
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
GRID_DIR = HERE / 'data' / 'pendulum_full_grid'
DS = Path('/common/users/shared/pracsys/genMoPlan/data_trajectories/deterministic/pendulum/lqr_failset')
EXP = Path('/common/users/shared/pracsys/adaptive_roa_experiments')
RUNS = {'x231_penlqr_fm_uh_a6d075_r1_s99512': EXP / 'det_v3_seeds/pen-lqr/x231_penlqr_fm_uh_a6d075_r1_s99512',
        'm65_penlqr_fm_unif_s99512': EXP / 'det_v3/pen-lqr/m65_penlqr_fm_unif_s99512'}
K, SEED, STEPS, RADIUS, BATCH = 100, 99512, 100, 0.1, 500


def need(ok, msg):
    if not ok:
        raise AssertionError(msg)


def cells(X, tz):
    """Grid cell index of each state, from the template lattice (0.04 rad spacing)."""
    nj = len(tz['omega_values'])
    i = np.rint((X[:, 0] - tz['theta_values'][0]) / .04).astype(int)
    j = np.rint((X[:, 1] - tz['omega_values'][0]) / .04).astype(int)
    need(((i >= 0) & (i < len(tz['theta_values'])) & (j >= 0) & (j < nj)).all(), 'state off the lattice')
    return i * nj + j


def fresh(ckpt, X):
    import torch
    from adaptive_roa.flow_matching.pendulum.latent_conditional.flow_matcher import \
        PendulumLatentConditionalFlowMatcher as FM
    fm = FM.load_from_checkpoint(str(ckpt), device='cuda', dataset_dir=str(DS))
    fm.eval()
    torch.manual_seed(SEED)
    ks, kf = np.zeros(len(X), int), np.zeros(len(X), int)
    with torch.no_grad():
        for a in range(0, len(X), BATCH):
            x0 = torch.tensor(X[a:a + BATCH], dtype=torch.float32, device='cuda').repeat_interleave(K, 0)
            lab = fm.system.classify_attractor(fm.predict_endpoint(x0, num_steps=STEPS), RADIUS).view(-1, K).cpu().numpy()
            ks[a:a + BATCH], kf[a:a + BATCH] = (lab == 1).sum(1), (lab == -1).sum(1)
    return ks / K, kf / K, 1 - (ks + kf) / K, torch.cuda.get_device_name(0)


def agreement(a, b):
    d = np.abs(a - b)
    return dict(n=int(len(a)), mean_abs_diff=float(d.mean()), max_abs_diff=float(d.max()),
                pearson_r=float(np.corrcoef(a, b)[0, 1]))


def build(name, epoch):
    run = RUNS[name]
    tpl = GRID_DIR / f'{name}_epoch010_grid.npz'
    tz = np.load(tpl, allow_pickle=False)
    X, src, y = tz['start_states'], tz['source'].astype(str), tz['true_label'].astype(int)
    ed = run / f'epoch_{epoch:03d}'
    ck = sorted((ed / 'checkpoints').glob('best*.ckpt'))
    need(len(ck) == 1, f'{name} epoch {epoch}: {len(ck)} best checkpoints')
    t0 = time.time()
    ps_f, pf_f, pi_f, gpu = fresh(ck[0], X)
    secs = time.time() - t0
    ps, pf, pi = ps_f.copy(), pf_f.copy(), pi_f.copy()
    rep = {}
    # stored test rows
    zt = np.load(ed / 'full_roa_per_point.npz')
    ix = cells(zt['start_states'], tz)
    need(len(np.unique(ix)) == len(ix) and (src[ix] == 'test').all() and (src == 'test').sum() == len(ix),
         f'{name}: stored test rows do not map one-to-one onto the template test cells')
    need(np.abs(X[ix] - zt['start_states']).max() < 1e-5, f'{name}: test coordinates differ')
    need(np.array_equal(y[ix] == 1, zt['true_labels'] == 1), f'{name}: test labels differ')
    for k, (dst, f_) in dict(p_success=(ps, ps_f), p_failure=(pf, pf_f), p_invalid=(pi, pi_f)).items():
        dst[ix] = zt[k]
        rep[f'test_{k}_stored_vs_fresh'] = agreement(zt[k], f_[ix])
    # stored cal / val rows (counts / num_mc_samples, states from the run-level store)
    for s in ('cal', 'val'):
        zs = np.load(run / f'prediction_store_states_{s}.npz')
        zp = np.load(ed / f'predictions_{s}.npz')
        n = int(zp['num_mc_samples'])
        need(n == K, f'{name} {s}: num_mc_samples {n}')
        ix = cells(zs['states'], tz)
        need(len(np.unique(ix)) == len(ix) and (src[ix] == s).all() and (src == s).sum() == len(ix),
             f'{name}: stored {s} rows do not map onto the template {s} cells')
        need(np.array_equal(zs['true_labels'], zp['true_labels']), f'{name} {s}: store labels differ')
        need(np.array_equal(y[ix] == 1, zp['true_labels'] == 1), f'{name} {s}: labels differ from the grid')
        need((zp['n_success'] + zp['n_failure'] + zp['n_invalid'] == n).all(), f'{name} {s}: counts do not sum to K')
        for k, c, (dst, f_) in (('p_success', 'n_success', (ps, ps_f)), ('p_failure', 'n_failure', (pf, pf_f)),
                                ('p_invalid', 'n_invalid', (pi, pi_f))):
            dst[ix] = zp[c] / n
            rep[f'{s}_{k}_stored_vs_fresh'] = agreement(zp[c] / n, f_[ix])
    need(np.allclose(ps + pf + pi, 1), f'{name}: masses do not sum to 1')
    out = {k: tz[k] for k in tz.files if not k.startswith('p_') and k != 'epoch'}
    out.update(p_success=ps, p_failure=pf, p_invalid=pi, p_success_fresh=ps_f, p_failure_fresh=pf_f,
               p_invalid_fresh=pi_f, epoch=np.int32(epoch))
    dst = GRID_DIR / f'{name}_epoch{epoch:03d}_grid.npz'
    np.savez_compressed(dst, **out)
    prov = dict(what=f'Round-{epoch} predictions on every start of the Pendulum LQR 158x315 grid',
                builder='dv_l4dc_plot/scripts/build_pendulum_full_grid.py', run=name, run_dir=str(run),
                scored_name=name[:name.rindex('_s')], epoch=epoch, checkpoint=str(ck[0]),
                checkpoint_sha256=hashlib.sha256(ck[0].read_bytes()).hexdigest(),
                template=str(tpl), K=K, seed=f'torch.manual_seed({SEED}) once before the first batch',
                fresh_path=f'predict_endpoint(num_steps={STEPS}, Euler) + system.classify_attractor(r={RADIUS}), '
                           f'{BATCH} states per call',
                canonical='stored value where the run stored one (test, cal, val), else fresh',
                source_counts={s: int((src == s).sum()) for s in ('test', 'cal', 'val', 'fresh')},
                host=socket.gethostname(), gpu=gpu, fresh_seconds=round(secs, 1),
                written_utc=datetime.now(timezone.utc).isoformat(timespec='seconds'), reproduction=rep)
    dst.with_name(dst.stem + '_provenance.json').write_text(json.dumps(prov, indent=1))
    print(f'{name} epoch {epoch}: {dst.name} ({secs:.0f} s fresh); test p_s stored vs fresh mean abs diff '
          f'{rep["test_p_success_stored_vs_fresh"]["mean_abs_diff"]:.4f}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument('--epoch', type=int, required=True)
    a = p.parse_args()
    need(a.epoch != 10, 'round 10 is dm1487\'s grid; do not overwrite it')
    for name in RUNS:
        build(name, a.epoch)
