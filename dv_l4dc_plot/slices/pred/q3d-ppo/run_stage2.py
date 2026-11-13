'''Stage 2 for q3d-ppo (slices/STAGE2.md): TERA-FM counts, decisions, paired figures, index, sheet.

All 12 q3d-ppo slices are 121x81, so one full-resolution pass (stride 1, <id> = plane).
Resumable: a slice whose pred/q3d-ppo/<id>.npz exists is not re-inferred, only re-rendered.

    PYTHONPATH=$DETV3:$SLICES nice -n 5 $PY pred/q3d-ppo/run_stage2.py
'''
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

SLICES = Path(__file__).resolve().parents[2]
from lib import infer, metrics, sheet, style  # noqa: E402

SETTING = 'q3d-ppo'
DEVICE = 'cuda:4'
BATCH = 16384
SEED = 0
PRED = SLICES / 'pred' / SETTING
FIGS = SLICES / 'figs' / SETTING
LOG = PRED / 'run.log'


def log(msg):
    line = f'{time.strftime("%Y-%m-%d %H:%M:%S")} {msg}'
    with open(LOG, 'a') as fh:
        fh.write(line + '\n')
    print(line, flush=True)


def main():
    PRED.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    name = infer.RUNS[SETTING]
    rdir = infer.run_dir(name)
    thr = infer.thresholds(name)
    log(f'start run={name} dir={rdir} thr={thr} device={DEVICE} batch={BATCH} seed={SEED} pid={os.getpid()}')
    files = sorted((SLICES / 'data' / SETTING).glob('*.npz'))
    t_all = time.time()
    rows = []
    logged_code = False
    for f in files:
        sl = style.load_slice(f)
        pid = sl['plane']
        ny, nx = sl['ny'], sl['nx']
        pp = PRED / f'{pid}.npz'
        if pp.exists():
            with np.load(pp) as z:
                dec = np.array(z['decision'])
            log(f'{pid}: pred exists, re-rendering only')
        else:
            X = np.ascontiguousarray(sl['model_states'], dtype=np.float32)
            t0 = time.time()
            c = infer.run_counts(rdir, X, device=DEVICE, seed=SEED, batch_size=BATCH,
                                 out_dir=PRED / '_tmp')
            if not logged_code:
                log(f'adaptive_roa from {infer.adaptive_roa_file()}; ckpt {c["ckpt"]}')
                logged_code = True
            dec = infer.decide(c, thr).reshape(ny, nx)
            secs = time.time() - t0
            tmp = PRED / f'{pid}.tmp.npz'
            np.savez(tmp, n_success=c['n_success'], n_failure=c['n_failure'], n_invalid=c['n_invalid'],
                     K=np.int64(c['K']), decision=dec.astype(np.int8), stride=np.int64(1), seed=np.int64(SEED),
                     run=np.array(name), t_s=thr['t_s'], t_f=thr['t_f'], u_s=thr['u_s'], u_f=thr['u_f'],
                     seconds=secs, evaluator_seconds=c['seconds'], batch_size=np.int64(BATCH),
                     device=np.array(DEVICE), ckpt=np.array(str(c['ckpt'])))
            os.replace(tmp, pp)
            log(f'{pid}: inferred {nx * ny} states in {secs:.0f}s '
                f'(invalid share {float(np.mean(c["n_invalid"])) / c["K"]:.4f})')
        gt_m, tera_m = metrics.slice_metrics(sl, dec)
        png, pdf = style.render_pair(sl, dec, str(FIGS / pid), metrics=tera_m)
        row = sheet.index_row(sl, gt_m, tera_m, run=name,
                              extra={'png': png, 'pdf': pdf,
                                     'resolution': f'{nx}x{ny} GT; TERA {nx}x{ny} full resolution'})
        rows.append(row)
        log(f'{pid}: rank {tera_m["rank_score"]:.3f} clarity {gt_m["gt_clarity"]:.3f} R_s {tera_m["R_s"]:.3f} '
            f'R_f {tera_m["R_f"]:.3f} abstain {tera_m["abstain_frac"]:.3f} '
            f'n_wrong {tera_m["n_wrong_s"]}+{tera_m["n_wrong_f"]}')
    idx = sheet.write_index(rows, SLICES / f'index_{SETTING}.csv')

    def key(r):
        v = float(r['rank_score']) if r['rank_score'] != '' else float('nan')
        return -v if not math.isnan(v) else math.inf
    srt = sorted(rows, key=key)
    sh = sheet.contact_sheet([r['png'] for r in srt], [sheet.pair_caption(r) for r in srt],
                             SLICES / 'sheets' / f'{SETTING}_pairs.png')
    log(f'done: {len(rows)} slices, index {idx}, sheet {sh}, wall {time.time() - t_all:.0f}s')


if __name__ == '__main__':
    try:
        main()
    except Exception as e:  # keep the failure in run.log
        import traceback
        log('FAILED: ' + repr(e) + '\n' + traceback.format_exc())
        sys.exit(1)
