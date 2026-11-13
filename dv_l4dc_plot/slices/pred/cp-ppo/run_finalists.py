#!/usr/bin/env python3
"""cp-ppo, every slice on every seed's run at full resolution (slices/FLAGSHIP.md, slices/ALL_SEEDS.md).

    run_finalists.py --seeds r1           finalists first, then the other slices in descending in-box
                                          gt_clarity; writes figs/cp-ppo/<plane>__<rK>.{png,pdf}, then
                                          pred/cp-ppo/<plane>__<rK>.npz atomically and last
    run_finalists.py --index              link the median-seed r2 files (run_stage2.py, old names) to the
                                          __r2 names (png, pdf, then npz) and write
                                          index_cp-ppo_finalists.csv and index_cp-ppo_all.csv

r2 is the median seed: its full-resolution outputs come from run_stage2.py under the old names and
are symlinked, never recomputed. Each model loads once (infer.load_run caches per run dir).
Thresholds are that run's own c6c ones. Resumable: an existing pred npz is not re-inferred.
"""
import argparse
import os
import time
from pathlib import Path

import numpy as np

from lib import metrics, sheet, style

SETTING = "cp-ppo"
BASE = "x161_cpppo_fm_uh_a6d075"
FINALISTS = ["k100_x_theta_dot", "k100_x_x_dot", "k100_x_theta"]
DEVICE, SEED, BATCH = "cuda:2", 0, 16384
ROOT = Path(__file__).resolve().parents[2]
DATA, PRED, FIGS = ROOT / "data" / SETTING, ROOT / "pred" / SETTING, ROOT / "figs" / SETTING
TMP = PRED / "_tmp"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def plane_order():
    """Finalists in priority order, then the rest by descending in-box gt_clarity (single-class last)."""
    rest = []
    for p in sorted(DATA.glob("*.npz")):
        sl = style.load_slice(p)
        if sl["plane"] in FINALISTS:
            continue
        single = len(np.unique(sl["labels"])) == 1
        clar = metrics.slice_metrics(sl)[0]["gt_clarity"]
        rest.append((single, -np.nan_to_num(clar, nan=-np.inf), sl["plane"]))
    return FINALISTS + [p for _, _, p in sorted(rest)]


def figure_ok(stem):
    return Path(f"{stem}.png").exists() and Path(f"{stem}.pdf").exists()


def infer_seed(rk):
    from lib import infer
    name = f"{BASE}_{rk}"
    rdir, thr = infer.run_dir(name), infer.thresholds(name)
    log(f"{name}: dir {rdir} thresholds {thr}")
    for pid in plane_order():
        sl = style.load_slice(DATA / f"{pid}.npz")
        pf, stem = PRED / f"{pid}__{rk}.npz", FIGS / f"{pid}__{rk}"
        if pf.exists():
            with np.load(pf) as z:
                assert str(z["run"]) == name and int(z["stride"]) == 1
                dec = np.array(z["decision"])
            if not figure_ok(stem):      # an npz from before the figure-first rule
                style.render_pair(sl, dec, str(stem), metrics=metrics.slice_metrics(sl, dec)[1])
            log(f"{pid}__{rk}: pred exists, skipping inference")
            continue
        X = sl["model_states"].astype(np.float32)
        c = infer.run_counts(rdir, X, device=DEVICE, seed=SEED, out_dir=TMP, batch_size=BATCH,
                             labels01=sl["labels"].ravel())
        dec = infer.decide(c, thr).reshape(sl["ny"], sl["nx"])
        gt_m, tera_m = metrics.slice_metrics(sl, dec)
        style.render_pair(sl, dec, str(stem), metrics=tera_m)          # figure first
        tmp = PRED / f"{pid}__{rk}.tmp.npz"
        np.savez(tmp, n_success=c["n_success"], n_failure=c["n_failure"], n_invalid=c["n_invalid"], K=c["K"],
                 decision=dec.astype(np.int8), stride=1, seed=SEED, run=name, ckpt=str(c["ckpt"]),
                 batch_size=BATCH, device=DEVICE, seconds=c["seconds"], **thr)
        os.replace(tmp, pf)                                             # npz last, atomically
        log(f"{pid}__{rk}: {len(X)} states in {c['seconds']:.1f} s; rank {tera_m['rank_score']:.3f} "
            f"R_s {tera_m['R_s']:.3f} R_f {tera_m['R_f']:.3f} abstain {tera_m['abstain_frac']:.3f} "
            f"n_wrong {tera_m['n_wrong_s'] + tera_m['n_wrong_f']}")


def link_r2():
    """Median-seed full-resolution files keep their names; __r2 names are relative symlinks, npz last."""
    for pid in plane_order():
        src_npz = PRED / f"{pid}.npz"
        if not (src_npz.exists() and figure_ok(FIGS / pid)):
            continue
        with np.load(src_npz) as z:
            assert str(z["run"]) == f"{BASE}_r2" and int(z["stride"]) == 1
        for ext in ("png", "pdf"):
            dst = FIGS / f"{pid}__r2.{ext}"
            if not dst.exists():
                os.symlink(f"{pid}.{ext}", dst)
        dst = PRED / f"{pid}__r2.npz"
        if not dst.exists():
            os.symlink(f"{pid}.npz", dst)


def write_index():
    link_r2()
    rows = {}
    for pid in plane_order():
        sl = style.load_slice(DATA / f"{pid}.npz")
        for rk in ("r1", "r2", "r3"):
            pf, stem = PRED / f"{pid}__{rk}.npz", FIGS / f"{pid}__{rk}"
            if not pf.exists():
                log(f"{pid}__{rk}: no pred yet")
                continue
            with np.load(pf) as z:
                dec, run = np.array(z["decision"]), str(z["run"])
            assert run == f"{BASE}_{rk}", (pf, run)
            gt_m, tera_m = metrics.slice_metrics(sl, dec)
            res = f"{sl['nx']}x{sl['ny']} GT; TERA {sl['nx']}x{sl['ny']} full resolution"
            rows[(pid, rk)] = sheet.index_row(sl, gt_m, tera_m, run=run,
                                              extra={"png": f"{stem}.png", "pdf": f"{stem}.pdf", "resolution": res})
    fin = sheet.write_index([r for (p, _), r in rows.items() if p in FINALISTS],
                            ROOT / f"index_{SETTING}_finalists.csv")
    allp = sheet.write_index(list(rows.values()), ROOT / f"index_{SETTING}_all.csv")
    log(f"wrote {fin} and {allp} ({len(rows)} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="*", default=[])
    ap.add_argument("--index", action="store_true")
    a = ap.parse_args()
    assert "r2" not in a.seeds, "r2 is the median seed: link run_stage2.py's outputs, do not recompute"
    for rk in a.seeds:
        infer_seed(rk)
    if a.index:
        write_index()
    log("DONE")


if __name__ == "__main__":
    main()
