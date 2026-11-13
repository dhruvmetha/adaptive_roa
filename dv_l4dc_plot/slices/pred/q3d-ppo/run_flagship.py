'''Flagship finalists for q3d-ppo (slices/FLAGSHIP.md): three planes x three seeds, full resolution.

r2 and r3 are inferred here (each model loaded once). r1 is the median seed that run_stage2.py
already covers: its pred/fig files are linked to the <id>__r1 names once they exist.
Writes pred/q3d-ppo/<id>__<rK>.npz, figs/q3d-ppo/<id>__<rK>.{png,pdf}, index_q3d-ppo_finalists.csv.

    PYTHONPATH=$DETV3:$SLICES nice -n 5 $PY pred/q3d-ppo/run_flagship.py
'''
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
BASE = 'x70_q3dppo_fm_uh_a6d075'
FINALISTS = ['k100_tilt_p', 'k100_z_z_dot_at_x1.4', 'k100_x_dot_y_dot_at_x1.4']
PRED = SLICES / 'pred' / SETTING
FIGS = SLICES / 'figs' / SETTING
LOG = PRED / 'flagship.log'


def log(msg):
    line = f'{time.strftime("%Y-%m-%d %H:%M:%S")} {msg}'
    with open(LOG, 'a') as fh:
        fh.write(line + '\n')
    print(line, flush=True)


def link(src_name, dst):
    if dst.is_symlink() or dst.exists():
        return
    os.symlink(src_name, dst)


def row_for(sl, dec, name, png, pdf):
    gt_m, tera_m = metrics.slice_metrics(sl, dec)
    row = sheet.index_row(sl, gt_m, tera_m, run=name,
                          extra={'png': png, 'pdf': pdf,
                                 'resolution': f'{sl["nx"]}x{sl["ny"]} GT; TERA {sl["nx"]}x{sl["ny"]} full resolution'})
    log(f'{sl["plane"]} {name[-2:]}: rank {tera_m["rank_score"]:.3f} clarity {gt_m["gt_clarity"]:.3f} '
        f'R_s {tera_m["R_s"]:.3f} R_f {tera_m["R_f"]:.3f} abstain {tera_m["abstain_frac"]:.3f} '
        f'n_wrong {tera_m["n_wrong_s"]}+{tera_m["n_wrong_f"]}')
    return row, tera_m


def main():
    slices = {p: style.load_slice(SLICES / 'data' / SETTING / f'{p}.npz') for p in FINALISTS}
    rows = []
    for rk in ('r2', 'r3'):
        name = f'{BASE}_{rk}'
        rdir, thr = infer.run_dir(name), infer.thresholds(name)
        log(f'{rk}: run={name} dir={rdir} thr={thr}')
        for p in FINALISTS:
            sl = slices[p]
            stem = f'{p}__{rk}'
            pp = PRED / f'{stem}.npz'
            if pp.exists():
                with np.load(pp) as z:
                    dec = np.array(z['decision'])
                log(f'{stem}: pred exists')
            else:
                t0 = time.time()
                c = infer.run_counts(rdir, np.ascontiguousarray(sl['model_states'], dtype=np.float32),
                                     device=DEVICE, seed=SEED, batch_size=BATCH, out_dir=PRED / '_tmp')
                dec = infer.decide(c, thr).reshape(sl['ny'], sl['nx'])
                secs = time.time() - t0
                tmp = PRED / f'{stem}.tmp.npz'
                np.savez(tmp, n_success=c['n_success'], n_failure=c['n_failure'], n_invalid=c['n_invalid'],
                         K=np.int64(c['K']), decision=dec.astype(np.int8), stride=np.int64(1),
                         seed=np.int64(SEED), run=np.array(name), t_s=thr['t_s'], t_f=thr['t_f'],
                         u_s=thr['u_s'], u_f=thr['u_f'], seconds=secs, evaluator_seconds=c['seconds'],
                         batch_size=np.int64(BATCH), device=np.array(DEVICE), ckpt=np.array(str(c['ckpt'])))
                os.replace(tmp, pp)
                log(f'{stem}: inferred {sl["nx"] * sl["ny"]} states in {secs:.0f}s, ckpt {c["ckpt"]}')
            gt_m, tera_m = metrics.slice_metrics(sl, dec)
            png, pdf = style.render_pair(sl, dec, str(FIGS / stem), metrics=tera_m)
            rows.append(row_for(sl, dec, name, png, pdf)[0])
    # r1: the stage 2 files, linked once present (run_stage2.py writes npz, then png/pdf)
    name = f'{BASE}_r1'
    for p in FINALISTS:
        src = [PRED / f'{p}.npz', FIGS / f'{p}.png', FIGS / f'{p}.pdf']
        t0 = time.time()
        while not all(s.exists() for s in src):
            if time.time() - t0 > 3600:
                raise TimeoutError(f'stage 2 output for {p} missing after 1 h')
            time.sleep(10)
        time.sleep(2)                                  # let the pdf write finish
        with np.load(src[0]) as z:
            dec = np.array(z['decision'])
            assert str(z['run']) == name and int(z['stride']) == 1
        link(f'{p}.npz', PRED / f'{p}__r1.npz')
        link(f'{p}.png', FIGS / f'{p}__r1.png')
        link(f'{p}.pdf', FIGS / f'{p}__r1.pdf')
        rows.append(row_for(slices[p], dec, name, str(FIGS / f'{p}__r1.png'), str(FIGS / f'{p}__r1.pdf'))[0])
    rows.sort(key=lambda r: (r['plane'], r['run']))
    out = sheet.write_index(rows, SLICES / f'index_{SETTING}_finalists.csv')
    log(f'done: {len(rows)} rows -> {out}')


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        import traceback
        log('FAILED: ' + repr(e) + '\n' + traceback.format_exc())
        sys.exit(1)
