#!/usr/bin/env python3
"""ALL_SEEDS.md for cp-lqr: TERA-FM on every slice x seed r1/r2/r3, full 400x400, no stride-2 screen.

Order: FLAGSHIP finalists (priority order, all seeds), then the other planes in descending in-box
gt_clarity, single-class planes last. Several copies can run at once (at most 2 per card); they
share the jobs through claim files pred/cp-lqr/.claim_<plane>__<rK> and skip any job whose pred npz
exists, so a rerun resumes.

    cd slices && PYTHONPATH=$DETV3:$SLICES:$SLICES/pred/cp-lqr nice -n 5 $PY pred/cp-lqr/all_seeds.py --device cuda:0

Per job: figs/cp-lqr/<plane>__<rK>.{png,pdf} first, then pred/cp-lqr/<plane>__<rK>.npz written
atomically (<name>.tmp.npz + os.replace). After each job: index_cp-lqr_finalists.csv,
index_cp-lqr_all.csv, and the STAGE2 median-seed index_cp-lqr.csv and sheets/cp-lqr_pairs.png.
"""
import argparse
import glob
import math
import os
import time
import traceback

import numpy as np

from lib import infer, metrics, sheet, style

SLICES = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SETTING = "cp-lqr"
BASE = "x29_cplqr_fm_uh_a6d075"
SEEDS = ["r1", "r2", "r3"]
RUN_SEED = {"r1": 99512, "r2": 33407, "r3": 24475}    # runs_index.tsv / real_dir suffix
MEDIAN = "r1"                                          # README seed rule
FINALISTS = ["viz_x_vs_xdot_old", "viz_theta_vs_thetadot", "viz_x_vs_xdot"]
BATCH, SEED = 16384, 0
PRED = os.path.join(SLICES, "pred", SETTING)
FIGS = os.path.join(SLICES, "figs", SETTING)
LOG = os.path.join(PRED, "run.log")
_SL = {}
DEVICE = None


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [all_seeds {DEVICE} pid {os.getpid()}] {msg}"
    with open(LOG, "a") as f:
        f.write(line + "\n")
    print(line, flush=True)


def pred_path(plane, rk):
    return os.path.join(PRED, f"{plane}__{rk}.npz")


def fig_stem(plane, rk):
    return os.path.join(FIGS, f"{plane}__{rk}")


def done(plane, rk):
    # a full-resolution median-seed file under the old STAGE2 name counts: link it, don't recompute
    if rk == MEDIAN:
        for d, ext in ((PRED, ".npz"), (FIGS, ".png"), (FIGS, ".pdf")):
            old, new = os.path.join(d, plane + ext), os.path.join(d, f"{plane}__{rk}{ext}")
            if os.path.exists(old) and not os.path.lexists(new):
                os.symlink(os.path.basename(old), new)
    return os.path.exists(pred_path(plane, rk))


def slice_of(plane):
    if plane not in _SL:
        _SL[plane] = style.load_slice(os.path.join(SLICES, "data", SETTING, f"{plane}.npz"))
    return _SL[plane]


def planes_in_order():
    """Finalists first, then the rest by descending in-box gt_clarity, single-class planes last."""
    allp = [os.path.basename(p)[:-4] for p in sorted(glob.glob(os.path.join(SLICES, "data", SETTING, "*.npz")))]
    rest = []
    for p in allp:
        if p in FINALISTS:
            continue
        sl = slice_of(p)
        gt_m, _ = metrics.slice_metrics(sl)
        single = len(np.unique(sl["labels"])) == 1
        c = gt_m["gt_clarity"]
        rest.append((single, -(c if math.isfinite(c) else -1.0), p))
    return [p for p in FINALISTS if p in allp] + [t[2] for t in sorted(rest)]


ADOPT = None   # pid of a placeholder that holds jobs back from older workers (--adopt)


def _adopt(path):
    """Take over a claim held by the ADOPT placeholder. The claim file never disappears (os.replace),
    so workers without --adopt keep seeing a live owner; .adopt_lock serialises the adopters."""
    lock = os.path.join(PRED, ".adopt_lock")
    for _ in range(600):
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            break
        except FileExistsError:
            try:
                os.kill(int(open(lock).read().strip() or 0), 0)
                time.sleep(0.1)
            except (ValueError, ProcessLookupError, FileNotFoundError):
                try:
                    os.remove(lock)
                except FileNotFoundError:
                    pass
    else:
        return None
    try:
        try:
            owner = int(open(path).read().strip() or 0)
        except (FileNotFoundError, ValueError):
            return None
        if owner != ADOPT:
            return None
        tmp = f"{path}.tmp{os.getpid()}"
        with open(tmp, "w") as f:
            f.write(str(os.getpid()))
        os.replace(tmp, path)
        return path
    finally:
        os.remove(lock)


def claim(plane, rk):
    path = os.path.join(PRED, f".claim_{plane}__{rk}")
    if ADOPT is not None:
        got = _adopt(path)
        if got is not None:
            return got
    for _ in range(2):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            return path
        except FileExistsError:
            try:
                pid = int(open(path).read().strip() or 0)
                os.kill(pid, 0)
                return None                      # a live process has it
            except (ValueError, ProcessLookupError, FileNotFoundError):
                try:
                    os.remove(path)              # stale claim from a dead process
                except FileNotFoundError:
                    pass
    return None


def run_job(plane, rk):
    name = f"{BASE}_{rk}"
    rd, thr = infer.run_dir(name), infer.thresholds(name)
    sl = slice_of(plane)
    X = sl["model_states"].astype(np.float32)
    log(f"{plane}__{rk}: inference on {len(X)} states, run {name}, thr {thr}")
    counts = infer.run_counts(rd, X, device=DEVICE, seed=SEED, out_dir=PRED, batch_size=BATCH,
                              labels01=sl["labels"].ravel())
    dec = infer.decide(counts, thr).reshape(sl["ny"], sl["nx"]).astype(np.int8)
    secs = float(counts["seconds"])
    gt_m, tera_m = metrics.slice_metrics(sl, dec)
    style.render_pair(sl, dec, fig_stem(plane, rk), metrics=tera_m)       # figures first
    final = pred_path(plane, rk)
    tmp = final[:-4] + ".tmp.npz"
    np.savez(tmp, n_success=counts["n_success"], n_failure=counts["n_failure"], n_invalid=counts["n_invalid"],
             K=counts["K"], decision=dec, stride=1, seed=SEED, run=name, run_seed=RUN_SEED[rk],
             t_s=thr["t_s"], t_f=thr["t_f"], u_s=thr["u_s"], u_f=thr["u_f"], seconds=secs,
             batch_size=BATCH, device=DEVICE, ckpt=str(counts.get("ckpt", "")))
    os.replace(tmp, final)                                                 # npz last, atomic
    log(f"{plane}__{rk}: {secs:.0f} s ({len(X) / max(secs, 1e-9):.0f}/s) rank {tera_m['rank_score']:.4f} "
        f"clarity {gt_m['gt_clarity']:.4f} R_s {tera_m['R_s']:.4f} R_f {tera_m['R_f']:.4f} "
        f"abstain {tera_m['abstain_frac']:.4f} n_wrong {tera_m['n_wrong_s']}+{tera_m['n_wrong_f']} | full slice: "
        f"R_s {tera_m['R_s_full']:.4f} R_f {tera_m['R_f_full']:.4f} abstain {tera_m['abstain_frac_full']:.4f}")


def _sub2(sl):
    """Stride-2 sub-slice (only to index the old STAGE2 stride-2 median-seed results)."""
    ny, nx = sl["ny"], sl["nx"]
    jj, ii = np.meshgrid(np.arange(0, ny, 2), np.arange(0, nx, 2), indexing="ij")
    out = dict(sl)
    out.update(xs=sl["xs"][::2], ys=sl["ys"][::2], labels=sl["labels"][::2, ::2].copy(),
               model_states=sl["model_states"][(jj * nx + ii).ravel()])
    out["ny"], out["nx"] = out["labels"].shape
    return out


def row_for(plane, rk, pred, stem, stride=1):
    sl = slice_of(plane) if stride == 1 else _sub2(slice_of(plane))
    with np.load(pred) as z:
        dec = np.array(z["decision"])
    gt_m, tera_m = metrics.slice_metrics(sl, dec)
    n = f"{sl['nx']}x{sl['ny']}"
    res = (f"{n} GT; TERA {n} full" if stride == 1 else
           f"{sl['nx'] * 2}x{sl['ny'] * 2} GT; TERA {n} stride 2 (metrics on the stride-2 grid)")
    row = sheet.index_row(sl, gt_m, tera_m, run=f"{BASE}_{rk}",
                          extra={"png": stem + ".png", "pdf": stem + ".pdf", "resolution": res})
    r = float(tera_m["rank_score"])
    return row, (r if math.isfinite(r) else -1.0)


def _atomic(write, final, suffix):
    tmp = os.path.join(PRED, f".tmp_{os.getpid()}{suffix}")
    write(tmp)
    os.replace(tmp, final)


def write_outputs(order):
    allrows = []
    for plane in order:
        for rk in SEEDS:
            if done(plane, rk):
                allrows.append((plane, row_for(plane, rk, pred_path(plane, rk), fig_stem(plane, rk))[0]))
    fin = [r for p, r in allrows if p in FINALISTS]
    if fin:
        _atomic(lambda p: sheet.write_index(fin, p), os.path.join(SLICES, f"index_{SETTING}_finalists.csv"), ".csv")
    if allrows:
        _atomic(lambda p: sheet.write_index([r for _, r in allrows], p),
                os.path.join(SLICES, f"index_{SETTING}_all.csv"), ".csv")
    # STAGE2 files: median seed, full resolution if present, else the old stride-2 result
    srows = []
    for plane in order:
        if done(plane, MEDIAN):
            row, rank = row_for(plane, MEDIAN, pred_path(plane, MEDIAN), fig_stem(plane, MEDIAN))
            srows.append((rank, row, "full"))
        elif os.path.exists(os.path.join(PRED, f"{plane}_s2.npz")):
            row, rank = row_for(plane, MEDIAN, os.path.join(PRED, f"{plane}_s2.npz"),
                                os.path.join(FIGS, f"{plane}_s2"), stride=2)
            srows.append((rank, row, "stride 2"))
    if srows:
        srows.sort(key=lambda t: -t[0])
        _atomic(lambda p: sheet.write_index([t[1] for t in srows], p), os.path.join(SLICES, f"index_{SETTING}.csv"), ".csv")
        caps = [sheet.pair_caption(t[1]) + f"   [{t[2]}]" for t in srows]
        _atomic(lambda p: sheet.contact_sheet([t[1]["png"] for t in srows], caps, p),
                os.path.join(SLICES, "sheets", f"{SETTING}_pairs.png"), ".png")
    log(f"outputs: {len(fin)} finalist rows, {len(allrows)} all rows, {len(srows)} STAGE2 rows")


def main():
    global DEVICE, ADOPT
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", required=True)
    ap.add_argument("--seeds", nargs="+", default=SEEDS, choices=SEEDS,
                    help="seeds this process may take (r3 waits for the infer.py Amarel path remap)")
    ap.add_argument("--adopt", type=int, default=None,
                    help="placeholder pid whose held claims this process may take over")
    args = ap.parse_args()
    DEVICE = args.device
    ADOPT = args.adopt
    t0 = time.time()
    order = planes_in_order()
    jobs = [(p, rk) for p in order for rk in SEEDS if rk in args.seeds]
    log(f"start: plane order {order}, seeds {args.seeds}, batch {BATCH}, seed {SEED}, "
        f"adaptive_roa {infer.adaptive_roa_file()}")
    failed = set()
    while True:
        left = [j for j in jobs if not done(*j) and j not in failed]
        if not left:
            break
        ran = False
        for job in left:
            c = claim(*job)
            if c is None:
                continue
            try:
                if not done(*job):
                    run_job(*job)
                    ran = True
            except Exception:
                failed.add(job)
                log(f"{job}: FAILED\n{traceback.format_exc()}")
            finally:
                os.remove(c)
            break
        if ran:
            try:
                write_outputs(order)
            except Exception:
                log(f"write_outputs FAILED\n{traceback.format_exc()}")
        else:
            break                                # the rest is claimed by live processes
    log(f"no unclaimed jobs left; exiting after {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
