#!/usr/bin/env python3
"""Stage 2 for q3d-lqr (STAGE2.md): TERA-FM counts, certified decisions, paired figures.

Pass 1: every slice at stride 2 (<plane>_s2), viz_psi_vs_r last. Then index + contact sheet.
Pass 2: full resolution (<plane>) in descending pass-1 rank_score, viz_psi_vs_r only after
all the others. Index + sheet rewritten after each pass-2 slice and at the end.
Resumable: a slice whose pred npz exists is not re-inferred (metrics and figure are redone).
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

SLICES = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SLICES))
from lib import infer, metrics, sheet, style  # noqa: E402

SETTING = "q3d-lqr"
RUN = infer.RUNS[SETTING]
DEVICE = "cuda:1"
SEED, BATCH = 0, 16384
DATA = SLICES / "data" / SETTING
PRED = SLICES / "pred" / SETTING
FIGS = SLICES / "figs" / SETTING
INDEX = SLICES / f"index_{SETTING}.csv"
SHEET = SLICES / "sheets" / f"{SETTING}_pairs.png"
PASS1 = ["viz_x_vs_xdot", "viz_y_vs_ydot", "viz_z_vs_zdot", "viz_phi_vs_p", "viz_theta_vs_q", "viz_psi_vs_r"]
LAST = "viz_psi_vs_r"  # single class

ROWS: dict = {}  # plane -> {stride: index row}


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def subslice(sl, s):
    if s == 1:
        return sl
    ny, nx = sl["ny"], sl["nx"]
    jj, ii = np.arange(0, ny, s), np.arange(0, nx, s)
    idx = (jj[:, None] * nx + ii[None, :]).ravel()
    sub = dict(sl)
    sub.update(xs=sl["xs"][ii], ys=sl["ys"][jj], labels=sl["labels"][np.ix_(jj, ii)].copy(),
               model_states=sl["model_states"][idx], nx=len(ii), ny=len(jj))
    assert np.array_equal(sub["labels"].ravel(), sl["labels"].ravel()[idx])
    return sub


def do_slice(plane, s, thr, rd):
    sid = plane if s == 1 else f"{plane}_s2"
    sl = style.load_slice(DATA / f"{plane}.npz")
    sub = subslice(sl, s)
    pred = PRED / f"{sid}.npz"
    if pred.exists():
        z = np.load(pred)
        dec, secs = z["decision"], float(z["seconds"])
        log(f"{sid}: pred exists, reusing ({sub['ny']}x{sub['nx']})")
    else:
        log(f"{sid}: inferring {sub['ny'] * sub['nx']} states")
        t0 = time.time()
        counts = infer.run_counts(rd, sub["model_states"].astype(np.float32), device=DEVICE, seed=SEED,
                                  batch_size=BATCH, out_dir=PRED / "_tmp")
        secs = time.time() - t0
        dec = infer.decide(counts, thr).reshape(sub["ny"], sub["nx"]).astype(np.int8)
        np.savez(pred, n_success=counts["n_success"], n_failure=counts["n_failure"],
                 n_invalid=counts["n_invalid"], K=np.int32(counts["K"]), decision=dec, stride=np.int32(s),
                 seed=np.int32(SEED), run=np.array(RUN), seconds=np.float64(secs),
                 ckpt=np.array(str(counts.get("ckpt", ""))), **{k: np.float64(v) for k, v in thr.items()})
        log(f"{sid}: inference {secs:.0f}s ({sub['ny'] * sub['nx'] / secs:.0f} states/s)")
    gt_m, tera_m = metrics.slice_metrics(sub, dec)
    png, pdf = style.render_pair(sub, dec, FIGS / sid, metrics=tera_m)
    res = f"{sl['nx']}x{sl['ny']} GT; TERA {sub['nx']}x{sub['ny']}" + ("" if s == 1 else f" stride {s}")
    row = sheet.index_row(sub, gt_m, tera_m, run=RUN, extra={"png": png, "pdf": pdf, "resolution": res})
    ROWS.setdefault(plane, {})[s] = row
    log(f"{sid}: rank {row['rank_score']} clarity {row['gt_clarity']} R_s {row['R_s']} R_f {row['R_f']} "
        f"abstain {row['abstain_frac']} n_wrong {int(row['n_wrong_s'] or 0) + int(row['n_wrong_f'] or 0)} "
        f"| full: R_s {row['R_s_full']} R_f {row['R_f_full']}")


def key(row):
    v = row.get("rank_score")
    try:
        v = float(v)
    except (TypeError, ValueError):
        return -math.inf
    return -math.inf if math.isnan(v) else v


def write_outputs():
    rows = [r[1] if 1 in r else r[2] for r in ROWS.values()]
    rows.sort(key=key, reverse=True)
    sheet.write_index(rows, INDEX)
    caps = [sheet.pair_caption(r) + ("" if "stride" not in r["resolution"] else "   [TERA stride 2]") for r in rows]
    SHEET.parent.mkdir(parents=True, exist_ok=True)
    sheet.contact_sheet([r["png"] for r in rows], caps, SHEET)
    log(f"wrote {INDEX.name} and {SHEET.name} ({len(rows)} rows)")


def main():
    PRED.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    thr = infer.thresholds(RUN)
    rd = infer.run_dir(RUN)
    log(f"run {RUN} dir {rd} thresholds {thr} device {DEVICE} batch {BATCH} seed {SEED}")
    t0 = time.time()
    for plane in PASS1:
        do_slice(plane, 2, thr, rd)
    write_outputs()
    log(f"PASS 1 DONE in {time.time() - t0:.0f}s")
    order = sorted([p for p in PASS1 if p != LAST], key=lambda p: key(ROWS[p][2]), reverse=True) + [LAST]
    log(f"pass 2 order: {order}")
    for plane in order:
        do_slice(plane, 1, thr, rd)
        write_outputs()
    log(f"PASS 2 DONE, total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    sys.exit(main())
