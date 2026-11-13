#!/usr/bin/env python3
"""Stage 2 for cp-ppo: TERA-FM counts and certified decisions on every slice, paired figures,
index_cp-ppo.csv and sheets/cp-ppo_pairs.png (slices/STAGE2.md).

All cp-ppo slices are 121x81, so there is one full-resolution pass. Resumable: a slice whose
pred/cp-ppo/<plane>.npz exists is not re-inferred (its figure is redrawn only if missing).
"""
import sys
import time
from pathlib import Path

import numpy as np

from lib import infer, metrics, sheet, style

SETTING = "cp-ppo"
DEVICE = "cuda:2"
SEED = 0
BATCH = 16384
ROOT = Path(__file__).resolve().parents[2]
DATA, PRED, FIGS = ROOT / "data" / SETTING, ROOT / "pred" / SETTING, ROOT / "figs" / SETTING
TMP = PRED / "_tmp"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    name = infer.RUNS[SETTING]
    rdir = infer.run_dir(name)
    thr = infer.thresholds(name)
    log(f"run {name} dir {rdir} thresholds {thr} device {DEVICE} batch {BATCH} seed {SEED}")
    log(f"adaptive_roa from {infer.adaptive_roa_file()}")
    rows, t_all = [], time.time()
    for path in sorted(DATA.glob("*.npz")):
        sl = style.load_slice(path)
        pid = sl["plane"]
        pf = PRED / f"{pid}.npz"
        if pf.exists():
            with np.load(pf) as z:
                dec = np.array(z["decision"])
                assert z["stride"] == 1 and str(z["run"]) == name
            log(f"{pid}: pred exists, skipping inference")
        else:
            X = sl["model_states"].astype(np.float32)
            counts = infer.run_counts(rdir, X, device=DEVICE, seed=SEED, out_dir=TMP, batch_size=BATCH,
                                      labels01=sl["labels"].ravel())
            dec = infer.decide(counts, thr).reshape(sl["ny"], sl["nx"])
            tmp = PRED / f"{pid}.tmp.npz"
            np.savez(tmp, n_success=counts["n_success"], n_failure=counts["n_failure"],
                     n_invalid=counts["n_invalid"], K=counts["K"], decision=dec.astype(np.int8),
                     stride=1, seed=SEED, run=name, ckpt=str(counts["ckpt"]), batch_size=BATCH, device=DEVICE,
                     seconds=counts["seconds"], **thr)
            tmp.replace(pf)
            log(f"{pid}: {len(X)} states in {counts['seconds']:.1f} s, ckpt {counts['ckpt']}")
        gt_m, tera_m = metrics.slice_metrics(sl, dec)
        stem = FIGS / pid
        if pf.exists() and Path(f"{stem}.png").exists() and Path(f"{stem}.pdf").exists():
            png, pdf = f"{stem}.png", f"{stem}.pdf"
        else:
            png, pdf = style.render_pair(sl, dec, str(stem), metrics=tera_m)
        res = f"{sl['nx']}x{sl['ny']} GT; TERA {sl['nx']}x{sl['ny']} full resolution"
        row = sheet.index_row(sl, gt_m, tera_m, run=name, extra={"png": png, "pdf": pdf, "resolution": res})
        rows.append(row)
        log(f"{pid}: rank {row['rank_score']:.3f} R_s {tera_m['R_s']:.3f} R_f {tera_m['R_f']:.3f} "
            f"abstain {tera_m['abstain_frac']:.3f} n_wrong {tera_m['n_wrong_s'] + tera_m['n_wrong_f']}")

    rows.sort(key=lambda r: -np.nan_to_num(float(r["rank_score"]), nan=-np.inf))
    idx = sheet.write_index(rows, ROOT / f"index_{SETTING}.csv")
    sh = sheet.contact_sheet([r["png"] for r in rows], [sheet.pair_caption(r) for r in rows],
                             ROOT / "sheets" / f"{SETTING}_pairs.png")
    log(f"wrote {idx} and {sh}; total {time.time() - t_all:.0f} s; DONE")


if __name__ == "__main__":
    sys.exit(main())
