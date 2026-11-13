#!/usr/bin/env python3
"""FLAGSHIP.md finalists for q3d-lqr: viz_theta_vs_q and viz_phi_vs_p on seeds r1, r2, r3.

  flagship.py screen <plane>:<rK> ...   stride-2 counts, decisions, pair figures
  flagship.py full <plane>              full 400x400 for the seed with the best in-box
                                        stride-2 rank_score
  flagship.py index                     index_q3d-lqr_finalists.csv from every finalist pred

Naming (FLAGSHIP.md 5): pred/q3d-lqr/<id>__<rK>.npz, figs/q3d-lqr/<id>__<rK>.{png,pdf},
<id> = <plane>_s2 or <plane>. r2 is the median seed of STAGE2.md: its files are written
under the STAGE2 name <id> and symlinked to <id>__r2. Thresholds: infer.thresholds(<run>) of
the run whose checkpoint is loaded. A pred that exists is reused, not re-inferred.
"""
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

SLICES = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SLICES))
from lib import infer, metrics, sheet, style  # noqa: E402

SETTING = "q3d-lqr"
BASE = "x33_q3dlqr_fm_uh_a6d075"
MEDIAN = "r2"
SEEDS = ("r1", "r2", "r3")
FINALISTS = ("viz_theta_vs_q", "viz_phi_vs_p")
DEVICE, SEED, BATCH = "cuda:1", 0, 16384
DATA, PRED, FIGS = SLICES / "data" / SETTING, SLICES / "pred" / SETTING, SLICES / "figs" / SETTING
INDEX = SLICES / f"index_{SETTING}_finalists.csv"


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [{os.getpid()}] {msg}", flush=True)


def run_name(rk):
    return f"{BASE}_{rk}"


def sid(plane, stride):
    return plane if stride == 1 else f"{plane}_s{stride}"


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


def pred_path(plane, stride, rk):
    i = sid(plane, stride)
    return PRED / (f"{i}.npz" if rk == MEDIAN else f"{i}__{rk}.npz")


def fig_stem(plane, stride, rk):
    i = sid(plane, stride)
    return FIGS / (i if rk == MEDIAN else f"{i}__{rk}")


def link_median(plane, stride):
    """<id>__r2.{npz,png,pdf} -> <id>.{npz,png,pdf} (relative symlinks)."""
    i = sid(plane, stride)
    for d, ext in ((PRED, "npz"), (FIGS, "png"), (FIGS, "pdf")):
        src, dst = d / f"{i}.{ext}", d / f"{i}__{MEDIAN}.{ext}"
        if src.exists() and not dst.exists():
            dst.symlink_to(src.name)


def resolution(sl, sub, s):
    return f"{sl['nx']}x{sl['ny']} GT; TERA {sub['nx']}x{sub['ny']}" + ("" if s == 1 else f" stride {s}")


def evaluate(plane, stride, rk, infer_missing=True):
    """Counts (inferred if missing), decision, metrics, figure. Returns the index row or None."""
    name = run_name(rk)
    sl = style.load_slice(DATA / f"{plane}.npz")
    sub = subslice(sl, stride)
    pp = pred_path(plane, stride, rk)
    if pp.exists():
        z = np.load(pp)
        dec = z["decision"]
        assert str(z["run"]) == name and int(z["stride"]) == stride, (pp, str(z["run"]), int(z["stride"]))
    elif not infer_missing:
        return None
    else:
        thr = infer.thresholds(name)
        log(f"{pp.name}: inferring {sub['ny'] * sub['nx']} states with {name}, thresholds {thr}")
        t0 = time.time()
        counts = infer.run_counts(infer.run_dir(name), sub["model_states"].astype(np.float32), device=DEVICE,
                                  seed=SEED, batch_size=BATCH, out_dir=PRED / "_tmp")
        secs = time.time() - t0
        dec = infer.decide(counts, thr).reshape(sub["ny"], sub["nx"]).astype(np.int8)
        np.savez(pp, n_success=counts["n_success"], n_failure=counts["n_failure"], n_invalid=counts["n_invalid"],
                 K=np.int32(counts["K"]), decision=dec, stride=np.int32(stride), seed=np.int32(SEED),
                 run=np.array(name), seconds=np.float64(secs), ckpt=np.array(str(counts.get("ckpt", ""))),
                 **{k: np.float64(v) for k, v in thr.items()})
        log(f"{pp.name}: inference {secs:.0f}s ({sub['ny'] * sub['nx'] / secs:.0f} states/s)")
    gt_m, tera_m = metrics.slice_metrics(sub, dec)
    stem = fig_stem(plane, stride, rk)
    if not Path(f"{stem}.png").exists() or infer_missing:
        png, pdf = style.render_pair(sub, dec, stem, metrics=tera_m)
    else:
        png, pdf = f"{stem}.png", f"{stem}.pdf"
    if rk == MEDIAN:
        link_median(plane, stride)
    row = sheet.index_row(sub, gt_m, tera_m, run=name,
                          extra={"png": str(png), "pdf": str(pdf), "resolution": resolution(sl, sub, stride)})
    nw = int(row["n_wrong_s"] or 0) + int(row["n_wrong_f"] or 0)
    log(f"{sid(plane, stride)} {rk}: rank {row['rank_score']} R_s {row['R_s']} R_f {row['R_f']} "
        f"abstain {row['abstain_frac']} n_wrong {nw}")
    return row


def score(row):
    try:
        v = float(row["rank_score"])
    except (TypeError, ValueError):
        return -math.inf
    return -math.inf if math.isnan(v) else v


def main(argv):
    PRED.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    cmd = argv[0]
    if cmd == "screen":
        for job in argv[1:]:
            plane, rk = job.split(":")
            assert plane in FINALISTS and rk in SEEDS
            evaluate(plane, 2, rk)
    elif cmd == "full":
        plane = argv[1]
        rows = {rk: evaluate(plane, 2, rk, infer_missing=False) for rk in SEEDS}
        missing = [rk for rk, r in rows.items() if r is None]
        if missing:
            raise SystemExit(f"{plane}: stride-2 preds missing for {missing}")
        best = max(SEEDS, key=lambda rk: score(rows[rk]))
        log(f"{plane}: stride-2 in-box rank_score " + ", ".join(f"{rk} {rows[rk]['rank_score']}" for rk in SEEDS)
            + f" -> full resolution on {best}")
        evaluate(plane, 1, best)
    elif cmd == "index":
        rows = []
        for plane in FINALISTS:
            for rk in SEEDS:
                for stride in (2, 1):
                    r = evaluate(plane, stride, rk, infer_missing=False)
                    if r is not None:
                        rows.append(r)
        sheet.write_index(rows, INDEX)
        log(f"wrote {INDEX} ({len(rows)} rows)")
    else:
        raise SystemExit(f"unknown command {cmd}")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
