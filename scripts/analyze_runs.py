#!/usr/bin/env python
"""Parse slurm_logs/<job>_<jid>.out for per-epoch ROA metrics (classifier + FM).

Classifier lines:  "[Full ROA / classifier] λ±δ:  F1=.. acc=.. prec=.. recall=.. uncertain=..%"
                   "[Full ROA / classifier] conservative: F1=.. recall=.."
FM lines:          "[Full ROA] λ±δ results:  F1=.. acc=.. prec=.. recall=.. separatrix=..%"
                   "[Full ROA] conservative: F1=.. prec=.. recall=.."
Emits a per-job summary: #epochs, best & final (lambda_delta F1, conservative F1, recall).
"""
import glob
import os
import re
import sys

JOBS = ["clf_pend_a", "clf_pend_r", "clf_cp_a", "clf_cp_r", "clf_q2d_a", "clf_q2d_r",
        "clf_q3d_a", "clf_q3d_r", "fm_pend_a", "fm_pend_r", "fm_cp_a", "fm_cp_r",
        "fm_q2d_a", "fm_q2d_r"]
LOGDIR = "/common/home/dm1487/robotics_research/tripods/olympics-classifier/slurm_logs"

f = lambda m: float(m.group(1)) if m else None


def parse(path):
    """Return list of per-epoch dicts."""
    band, cons = [], []
    with open(path, errors="ignore") as fh:
        for line in fh:
            if "λ±δ" in line and "F1=" in line:
                band.append(dict(f1=f(re.search(r"F1=([\d.]+)", line)),
                                 acc=f(re.search(r"acc=([\d.]+)", line)),
                                 prec=f(re.search(r"prec=([\d.]+)", line)),
                                 rec=f(re.search(r"recall=([\d.]+)", line)),
                                 unc=f(re.search(r"(?:uncertain|separatrix)=([\d.]+)", line))))
            elif "conservative:" in line and "F1=" in line:
                cons.append(dict(f1=f(re.search(r"F1=([\d.]+)", line)),
                                 rec=f(re.search(r"recall=([\d.]+)", line))))
    return band, cons


def best(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return max(vals) if vals else None


print(f"{'job':<12} {'ep':>3} | {'band_F1 best/fin':>17} | {'cons_F1 best/fin':>17} | {'cons_rec best/fin':>17}")
print("-" * 78)
for name in JOBS:
    logs = sorted(glob.glob(os.path.join(LOGDIR, f"{name}_*.out")))
    if not logs:
        print(f"{name:<12}  -- no log --")
        continue
    band, cons = parse(logs[-1])
    n = max(len(band), len(cons))
    bf = best(band, "f1"); bff = band[-1]["f1"] if band else None
    cf = best(cons, "f1"); cff = cons[-1]["f1"] if cons else None
    cr = best(cons, "rec"); crf = cons[-1]["rec"] if cons else None
    fmt = lambda b, x: (f"{b:.3f}/{x:.3f}" if b is not None and x is not None else "  -  ")
    print(f"{name:<12} {n:>3} | {fmt(bf,bff):>17} | {fmt(cf,cff):>17} | {fmt(cr,crf):>17}")
