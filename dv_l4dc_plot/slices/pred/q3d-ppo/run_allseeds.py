'''All q3d-ppo slices x seeds at full resolution (slices/ALL_SEEDS.md).

    PYTHONPATH=$DETV3:$SLICES nice -n 5 $PY pred/q3d-ppo/run_allseeds.py --seed r2
    PYTHONPATH=$DETV3:$SLICES $PY pred/q3d-ppo/run_allseeds.py --index        # CSVs only, no GPU

--seed rK: every slice in descending in-box gt_clarity (finalists first), skipping any whose
pred/q3d-ppo/<plane>__<rK>.npz exists. For r1, an old-name median-seed file (<plane>.npz with its
png/pdf) is symlinked to the __r1 names instead of recomputed. Figures are written first and the
npz last, atomically (<name>.tmp.npz then os.replace).
--index: rebuild index_q3d-ppo_all.csv (one row per plane and seed) and
index_q3d-ppo_finalists.csv from the __rK files present.
'''
import argparse
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

SLICES = Path(__file__).resolve().parents[2]
from lib import metrics, sheet, style  # noqa: E402

SETTING = 'q3d-ppo'
DEVICE = 'cuda:4'
BATCH = 16384
SEED = 0
BASE = 'x70_q3dppo_fm_uh_a6d075'
SEEDS = ('r1', 'r2', 'r3')
FINALISTS = ['k100_tilt_p', 'k100_z_z_dot_at_x1.4', 'k100_x_dot_y_dot_at_x1.4']
PRED = SLICES / 'pred' / SETTING
FIGS = SLICES / 'figs' / SETTING
LOG = PRED / 'allseeds.log'


def log(msg):
    line = f'{time.strftime("%Y-%m-%d %H:%M:%S")} {msg}'
    with open(LOG, 'a') as fh:
        fh.write(line + '\n')
    print(line, flush=True)


def ordered_slices():
    sls = {f.stem: style.load_slice(f) for f in sorted((SLICES / 'data' / SETTING).glob('*.npz'))}
    clar = {p: metrics.slice_metrics(sl)[0]['gt_clarity'] for p, sl in sls.items()}
    single = {p for p, sl in sls.items() if len(np.unique(sl['labels'])) < 2}

    def key(p):
        c = clar[p]
        return (p not in FINALISTS, FINALISTS.index(p) if p in FINALISTS else 0, p in single,
                -c if not math.isnan(c) else math.inf, p)
    return [(p, sls[p]) for p in sorted(sls, key=key)], clar


def resolution(sl):
    return f'{sl["nx"]}x{sl["ny"]} GT; TERA {sl["nx"]}x{sl["ny"]} full resolution'


def run_seed(rk):
    from lib import infer
    name = f'{BASE}_{rk}'
    rdir, thr = infer.run_dir(name), infer.thresholds(name)
    order, clar = ordered_slices()
    log(f'{rk}: run={name} dir={rdir} thr={thr} pid={os.getpid()} order={[p for p, _ in order]}')
    for p, sl in order:
        stem = f'{p}__{rk}'
        dst = PRED / f'{stem}.npz'
        if dst.exists() or dst.is_symlink():
            log(f'{stem}: exists, skip')
            continue
        old = [PRED / f'{p}.npz', FIGS / f'{p}.png', FIGS / f'{p}.pdf']
        if rk == 'r1' and all(o.exists() for o in old):
            with np.load(old[0]) as z:
                assert str(z['run']) == name and int(z['stride']) == 1, p
            for o, new in zip(old[1:], (FIGS / f'{stem}.png', FIGS / f'{stem}.pdf')):
                if not (new.exists() or new.is_symlink()):
                    os.symlink(o.name, new)
            os.symlink(old[0].name, dst)                     # npz link last
            log(f'{stem}: linked to the median-seed files')
            continue
        t0 = time.time()
        c = infer.run_counts(rdir, np.ascontiguousarray(sl['model_states'], dtype=np.float32),
                             device=DEVICE, seed=SEED, batch_size=BATCH, out_dir=PRED / '_tmp')
        dec = infer.decide(c, thr).reshape(sl['ny'], sl['nx'])
        secs = time.time() - t0
        gt_m, tera_m = metrics.slice_metrics(sl, dec)
        style.render_pair(sl, dec, str(FIGS / stem), metrics=tera_m)       # figures first
        tmp = PRED / f'{stem}.tmp.npz'
        np.savez(tmp, n_success=c['n_success'], n_failure=c['n_failure'], n_invalid=c['n_invalid'],
                 K=np.int64(c['K']), decision=dec.astype(np.int8), stride=np.int64(1), seed=np.int64(SEED),
                 run=np.array(name), t_s=thr['t_s'], t_f=thr['t_f'], u_s=thr['u_s'], u_f=thr['u_f'],
                 seconds=secs, evaluator_seconds=c['seconds'], batch_size=np.int64(BATCH),
                 device=np.array(DEVICE), ckpt=np.array(str(c['ckpt'])))
        os.replace(tmp, dst)                                                # npz last
        log(f'{stem}: {secs:.0f}s rank {tera_m["rank_score"]:.3f} clarity {gt_m["gt_clarity"]:.3f} '
            f'R_s {tera_m["R_s"]:.3f} R_f {tera_m["R_f"]:.3f} abstain {tera_m["abstain_frac"]:.3f} '
            f'n_wrong {tera_m["n_wrong_s"]}+{tera_m["n_wrong_f"]}')
    log(f'{rk}: done')


def write_indexes():
    order, _ = ordered_slices()
    rows, missing = [], []
    for p, sl in order:
        for rk in SEEDS:
            stem = f'{p}__{rk}'
            pp, png, pdf = PRED / f'{stem}.npz', FIGS / f'{stem}.png', FIGS / f'{stem}.pdf'
            if not (pp.exists() and png.exists() and pdf.exists()):
                missing.append(stem)
                continue
            with np.load(pp) as z:
                dec = np.array(z['decision'])
                run = str(z['run'])
                assert run == f'{BASE}_{rk}' and int(z['stride']) == 1, stem
            gt_m, tera_m = metrics.slice_metrics(sl, dec)
            rows.append(sheet.index_row(sl, gt_m, tera_m, run=run,
                                        extra={'png': str(png), 'pdf': str(pdf), 'resolution': resolution(sl)}))
    a = sheet.write_index(rows, SLICES / f'index_{SETTING}_all.csv')
    f = sheet.write_index([r for r in rows if r['plane'] in FINALISTS], SLICES / f'index_{SETTING}_finalists.csv')
    log(f'index: {len(rows)} rows -> {a}; finalists -> {f}; missing {missing}')
    return rows, missing


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--seed', choices=SEEDS)
    g.add_argument('--index', action='store_true')
    a = ap.parse_args()
    try:
        if a.index:
            write_indexes()
        else:
            run_seed(a.seed)
    except Exception as e:
        import traceback
        log('FAILED: ' + repr(e) + '\n' + traceback.format_exc())
        sys.exit(1)
