#!/usr/bin/env python3
"""ALL_SEEDS.md for q3d-lqr: every slice x seeds r1, r2, r3 at full 400x400.

  all_seeds.py work <device>   claim and run jobs until none are left (several workers may run)
  all_seeds.py index           rewrite index_q3d-lqr_all.csv and index_q3d-lqr_finalists.csv
  all_seeds.py final           index, plus the STAGE2 median-seed index_q3d-lqr.csv and
                               sheets/q3d-lqr_pairs.png from the r2 rows

Job order: FLAGSHIP finalists (viz_theta_vs_q, viz_phi_vs_p) on r1, r2, r3; then the other
slices in descending in-box gt_clarity; viz_psi_vs_r (single class) last.
Outputs: figs/q3d-lqr/<plane>__<rK>.{png,pdf}, then pred/q3d-lqr/<plane>__<rK>.npz written
atomically (<name>.tmp.npz + os.replace) and last. Thresholds: infer.thresholds(<run>) of
the run whose checkpoint is loaded. A worker claims a job with an O_EXCL file
<plane>__<rK>.claim holding its pid; a claim whose pid is dead is taken over.
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
SEEDS = ("r1", "r2", "r3")
MEDIAN = "r2"
FINALISTS = ("viz_theta_vs_q", "viz_phi_vs_p")
SINGLE_LAST = ("viz_psi_vs_r",)
SEED, BATCH = 0, 16384
DATA, PRED, FIGS = SLICES / "data" / SETTING, SLICES / "pred" / SETTING, SLICES / "figs" / SETTING


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [{os.getpid()}] {msg}", flush=True)


def run_name(rk):
    return f"{BASE}_{rk}"


def planes_ordered():
    planes = sorted(p.stem for p in DATA.glob("*.npz"))
    rest = [p for p in planes if p not in FINALISTS and p not in SINGLE_LAST]
    clar = {}
    for p in rest:
        gt_m, _ = metrics.slice_metrics(style.load_slice(DATA / f"{p}.npz"))
        c = float(gt_m["gt_clarity"])
        clar[p] = -math.inf if math.isnan(c) else c
    rest.sort(key=lambda p: clar[p], reverse=True)
    return [p for p in FINALISTS if p in planes] + rest + [p for p in SINGLE_LAST if p in planes]


def jobs():
    return [(p, rk) for p in planes_ordered() for rk in SEEDS]


def pred_path(plane, rk):
    return PRED / f"{plane}__{rk}.npz"


def claim(plane, rk):
    c = PRED / f"{plane}__{rk}.claim"
    for _ in range(2):
        try:
            fd = os.open(c, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            return c
        except FileExistsError:
            try:
                pid = int(c.read_text().strip() or 0)
            except (OSError, ValueError):
                return None
            if pid and Path(f"/proc/{pid}").exists():
                return None
            log(f"stale claim {c.name} (pid {pid}), taking over")
            c.unlink(missing_ok=True)
    return None


def run_job(plane, rk, device):
    name = run_name(rk)
    sl = style.load_slice(DATA / f"{plane}.npz")
    thr = infer.thresholds(name)
    log(f"{plane}__{rk}: inferring {sl['ny'] * sl['nx']} states with {name} on {device}, thresholds {thr}")
    t0 = time.time()
    counts = infer.run_counts(infer.run_dir(name), sl["model_states"].astype(np.float32), device=device,
                              seed=SEED, batch_size=BATCH, out_dir=PRED / "_tmp")
    secs = time.time() - t0
    dec = infer.decide(counts, thr).reshape(sl["ny"], sl["nx"]).astype(np.int8)
    gt_m, tera_m = metrics.slice_metrics(sl, dec)
    style.render_pair(sl, dec, FIGS / f"{plane}__{rk}", metrics=tera_m)  # figures first
    out = pred_path(plane, rk)
    tmp = PRED / f"{plane}__{rk}.tmp.npz"
    np.savez(tmp, n_success=counts["n_success"], n_failure=counts["n_failure"], n_invalid=counts["n_invalid"],
             K=np.int32(counts["K"]), decision=dec, stride=np.int32(1), seed=np.int32(SEED), run=np.array(name),
             seconds=np.float64(secs), ckpt=np.array(str(counts.get("ckpt", ""))),
             **{k: np.float64(v) for k, v in thr.items()})
    os.replace(tmp, out)  # npz last
    nw = int(tera_m.get("n_wrong_s", 0) or 0) + int(tera_m.get("n_wrong_f", 0) or 0)
    log(f"{plane}__{rk}: done in {secs:.0f}s ({sl['ny'] * sl['nx'] / secs:.0f} states/s); in-box rank "
        f"{tera_m['rank_score']:.4f} R_s {tera_m['R_s']:.4f} R_f {tera_m['R_f']:.4f} "
        f"abstain {tera_m['abstain_frac']:.4f} n_wrong {nw}")


def row_for(plane, rk):
    p = pred_path(plane, rk)
    if not p.exists():
        return None
    z = np.load(p)
    name = run_name(rk)
    assert str(z["run"]) == name and int(z["stride"]) == 1, p
    sl = style.load_slice(DATA / f"{plane}.npz")
    dec = z["decision"]
    gt_m, tera_m = metrics.slice_metrics(sl, dec)
    stem = FIGS / f"{plane}__{rk}"
    return sheet.index_row(sl, gt_m, tera_m, run=name,
                           extra={"png": f"{stem}.png", "pdf": f"{stem}.pdf",
                                  "resolution": f"{sl['nx']}x{sl['ny']} GT; TERA {sl['nx']}x{sl['ny']}"})


def write_atomic(rows, path):
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    sheet.write_index(rows, tmp)
    os.replace(tmp, path)


def write_indexes():
    order = planes_ordered()
    rows = {(p, rk): row_for(p, rk) for p in order for rk in SEEDS}
    rows = {k: v for k, v in rows.items() if v is not None}
    write_atomic([rows[k] for k in rows], SLICES / f"index_{SETTING}_all.csv")
    write_atomic([rows[k] for k in rows if k[0] in FINALISTS], SLICES / f"index_{SETTING}_finalists.csv")
    log(f"indexes: {len(rows)} (plane, seed) rows")
    return rows


def rank_key(row):
    try:
        v = float(row["rank_score"])
    except (TypeError, ValueError):
        return -math.inf
    return -math.inf if math.isnan(v) else v


def main(argv):
    PRED.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    cmd = argv[0]
    if cmd == "work":
        device = argv[1]
        n = 0
        while True:
            todo = [(p, rk) for p, rk in jobs() if not pred_path(p, rk).exists()]
            got = None
            for p, rk in todo:
                c = claim(p, rk)
                if c is not None:
                    if pred_path(p, rk).exists():  # finished between listing and claiming
                        c.unlink(missing_ok=True)
                        continue
                    got = (p, rk, c)
                    break
            if got is None:
                log(f"no unclaimed job left; worker on {device} exits after {n} jobs")
                return 0
            p, rk, c = got
            try:
                run_job(p, rk, device)
                n += 1
            finally:
                c.unlink(missing_ok=True)
            write_indexes()
    elif cmd == "index":
        write_indexes()
    elif cmd == "final":
        rows = write_indexes()
        med = [r for (p, rk), r in rows.items() if rk == MEDIAN]
        med.sort(key=rank_key, reverse=True)
        write_atomic(med, SLICES / f"index_{SETTING}.csv")
        sheet.contact_sheet([r["png"] for r in med], [sheet.pair_caption(r) for r in med],
                            SLICES / "sheets" / f"{SETTING}_pairs.png")
        log(f"STAGE2 median-seed index_{SETTING}.csv and sheets/{SETTING}_pairs.png ({len(med)} rows)")
    else:
        raise SystemExit(f"unknown command {cmd}")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
