#!/usr/bin/env python3
"""Stage 2 for cp-lqr (slices/STAGE2.md): TERA-FM counts, certified decisions and paired figures.

Pass 1: every 400x400 slice at stride 2 (id <plane>_s2). Pass 2: full resolution (id <plane>), in
descending pass-1 rank_score. Resumable: a slice whose pred npz exists is loaded, not recomputed.
After pass 1 and at the end it rewrites index_cp-lqr.csv and sheets/cp-lqr_pairs.png.

    cd slices && PYTHONPATH=$DETV3:$SLICES nice -n 5 $PY pred/cp-lqr/stage2.py
"""
import glob
import os
import sys
import time
import traceback

import numpy as np

SLICES = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SETTING = "cp-lqr"
DEVICE = "cuda:0"
BATCH = 16384
SEED = 0
PRED = os.path.join(SLICES, "pred", SETTING)
FIGS = os.path.join(SLICES, "figs", SETTING)
LOG = os.path.join(PRED, "run.log")

from lib import infer, metrics, sheet, style  # noqa: E402


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    with open(LOG, "a") as f:
        f.write(line + "\n")
    print(line, flush=True)


def sub_slice(sl, s):
    """Stride-s sub-slice: xs[::s], ys[::s], labels[::s, ::s] and the model_states rows of those cells."""
    if s == 1:
        return sl
    ny, nx = sl["ny"], sl["nx"]
    jj, ii = np.meshgrid(np.arange(0, ny, s), np.arange(0, nx, s), indexing="ij")
    rows = (jj * nx + ii).ravel()                       # C order of labels[ny, nx]
    out = dict(sl)
    out.update(xs=sl["xs"][::s], ys=sl["ys"][::s], labels=sl["labels"][::s, ::s].copy(),
               model_states=sl["model_states"][rows].copy())
    out["ny"], out["nx"] = out["labels"].shape
    assert out["model_states"].shape[0] == out["ny"] * out["nx"]
    # model_states rows still sit on the (xs, ys) grid of the sub-slice
    G = out["model_states"].reshape(out["ny"], out["nx"], -1)
    order = ["x", "theta", "x_dot", "theta_dot"]
    h, v = order.index(sl["dim_names"][0]), order.index(sl["dim_names"][1])
    assert np.allclose(G[:, :, h], out["xs"][None, :], atol=1e-6)
    assert np.allclose(G[:, :, v], out["ys"][:, None], atol=1e-6)
    return out


def run_slice(sl, sid, stride, name, thr, rd):
    """Counts (cached in pred/<sid>.npz), decisions, metrics, figure. Returns (row, rank, png)."""
    pred = os.path.join(PRED, f"{sid}.npz")
    if os.path.exists(pred):
        with np.load(pred) as z:
            counts = dict(n_success=z["n_success"], n_failure=z["n_failure"], n_invalid=z["n_invalid"],
                          K=int(z["K"]))
            dec = np.array(z["decision"])
            secs = float(z["seconds"])
        log(f"{sid}: pred exists, loaded")
    else:
        X = sl["model_states"].astype(np.float32)
        log(f"{sid}: inference on {len(X)} states (stride {stride})")
        counts = infer.run_counts(rd, X, device=DEVICE, seed=SEED, out_dir=PRED, batch_size=BATCH,
                                  labels01=sl["labels"].ravel())
        dec = infer.decide(counts, thr).reshape(sl["ny"], sl["nx"])
        secs = float(counts["seconds"])
        np.savez(pred, n_success=counts["n_success"], n_failure=counts["n_failure"],
                 n_invalid=counts["n_invalid"], K=counts["K"], decision=dec.astype(np.int8),
                 stride=stride, seed=SEED, run=name, t_s=thr["t_s"], t_f=thr["t_f"], u_s=thr["u_s"],
                 u_f=thr["u_f"], seconds=secs, batch_size=BATCH, device=DEVICE,
                 ckpt=str(counts.get("ckpt", "")))
        log(f"{sid}: saved counts in {secs:.0f} s ({len(X) / max(secs, 1e-9):.0f} states/s)")
    gt_m, tera_m = metrics.slice_metrics(sl, dec)
    png, pdf = style.render_pair(sl, dec, os.path.join(FIGS, sid), metrics=tera_m)
    full_ny, full_nx = sl["ny"] * stride, sl["nx"] * stride
    res = (f"{full_nx}x{full_ny} GT; TERA {sl['nx']}x{sl['ny']} stride {stride} (metrics on the stride-{stride} grid)"
           if stride > 1 else f"{full_nx}x{full_ny} GT; TERA {sl['nx']}x{sl['ny']} full")
    row = sheet.index_row(sl, gt_m, tera_m, run=name, extra={"png": png, "pdf": pdf, "resolution": res})
    log(f"{sid}: rank {tera_m['rank_score']:.4f} clarity {gt_m['gt_clarity']:.4f} R_s {tera_m['R_s']:.4f} "
        f"R_f {tera_m['R_f']:.4f} abstain {tera_m['abstain_frac']:.4f} "
        f"n_wrong {tera_m['n_wrong_s']}+{tera_m['n_wrong_f']} | full: R_s {tera_m['R_s_full']:.4f} "
        f"R_f {tera_m['R_f_full']:.4f} abstain {tera_m['abstain_frac_full']:.4f}")
    return row, float(tera_m["rank_score"]), png


def write_outputs(results):
    """results: plane -> {1: (row, rank, png), 2: ...}; use full res when present."""
    rows, pngs, caps = [], [], []
    for plane, r in results.items():
        stride = 1 if 1 in r else 2
        row, rank, png = r[stride]
        rows.append((rank, row, png, stride))
    rows.sort(key=lambda t: -t[0] if np.isfinite(t[0]) else np.inf)
    sheet.write_index([t[1] for t in rows], os.path.join(SLICES, f"index_{SETTING}.csv"))
    caps = [sheet.pair_caption(t[1]) + ("   [stride 2]" if t[3] == 2 else "   [full]") for t in rows]
    sheet.contact_sheet([t[2] for t in rows], caps, os.path.join(SLICES, "sheets", f"{SETTING}_pairs.png"))
    log(f"wrote index_{SETTING}.csv and sheets/{SETTING}_pairs.png ({len(rows)} rows)")


def main():
    t0 = time.time()
    name = infer.RUNS[SETTING]
    rd = infer.run_dir(name)
    thr = infer.thresholds(name)
    log(f"start: run {name} dir {rd} thr {thr} device {DEVICE} batch {BATCH} seed {SEED} "
        f"adaptive_roa {infer.adaptive_roa_file()}")
    paths = sorted(glob.glob(os.path.join(SLICES, "data", SETTING, "*.npz")))
    slices = {}
    for p in paths:
        sl = style.load_slice(p)
        assert sl["setting"] == SETTING
        slices[sl["plane"]] = sl
    results = {p: {} for p in slices}
    # pass 1: stride 2 for 400x400 slices, full for small ones
    for plane, sl in slices.items():
        try:
            if sl["nx"] >= 400 or sl["ny"] >= 400:
                results[plane][2] = run_slice(sub_slice(sl, 2), f"{plane}_s2", 2, name, thr, rd)
            else:
                results[plane][1] = run_slice(sl, plane, 1, name, thr, rd)
        except Exception:
            log(f"{plane}: pass 1 FAILED\n{traceback.format_exc()}")
    write_outputs({p: r for p, r in results.items() if r})
    log(f"pass 1 done in {time.time() - t0:.0f} s")
    # pass 2: full resolution, best pass-1 rank first
    todo = [p for p, r in results.items() if 2 in r and 1 not in r]
    todo.sort(key=lambda p: -results[p][2][1] if np.isfinite(results[p][2][1]) else np.inf)
    log(f"pass 2 order: {todo}")
    for plane in todo:
        try:
            results[plane][1] = run_slice(slices[plane], plane, 1, name, thr, rd)
            write_outputs({p: r for p, r in results.items() if r})
        except Exception:
            log(f"{plane}: pass 2 FAILED\n{traceback.format_exc()}")
    write_outputs({p: r for p, r in results.items() if r})
    log(f"all done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    sys.exit(main())
