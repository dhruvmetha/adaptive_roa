#!/usr/bin/env python
"""Robust analyzer: read per-epoch results.json from tonight's run dirs.

Tags each run by .hydra/overrides.yaml (predictor, system, d2_ratio, sampling_mode),
so it works for classifier + FM, buffered or not. Reports best/final
band F1 (full_roa.lambda_delta.f1), conservative F1 (full_roa.conservative_lambda_delta.f1),
and abstain (lambda_delta.uncertain_pct or separatrix_pct).
"""
import glob
import json
import os
import re

EXP = "/common/users/shared/pracsys/adaptive_roa_experiments/dhruv"
ALT = "/common/users/shared/pracsys/adaptive_roa_experiments"


def read_overrides(run):
    ov = {}
    p = os.path.join(run, ".hydra", "overrides.yaml")
    if not os.path.exists(p):
        return ov
    for line in open(p):
        line = line.strip().lstrip("- ")
        if "=" in line:
            k, v = line.split("=", 1)
            ov[k.strip()] = v.strip()
    return ov


def get(d, *path):
    for k in path:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def scan(since_hhmm="2026-06-21"):
    runs = []
    for base in (EXP, ALT):
        runs += glob.glob(os.path.join(base, "adaptive_*", "outputs", "*", "*"))
    rows = {}
    for run in runs:
        # tonight's batch only: dir basename like 2026-06-21_0[1-9]-..
        bn = os.path.basename(run)
        if not re.match(r"2026-06-21_0[1-9]", bn):
            continue
        eps = sorted(glob.glob(os.path.join(run, "epoch_*")))
        if not eps:
            continue
        # only tonight's runs
        if os.path.getmtime(run) < 1750000000:  # ~2026; coarse guard, refined by dir name
            pass
        ov = read_overrides(run)
        pred = ov.get("predictor", "generative")
        system = ov.get("system", "?")
        d2 = ov.get("d2_ratio", "?")
        mode = "adaptive" if d2 not in ("0.0", "0", "?") else "random"
        band, cons, abst, ntraj = [], [], [], []
        for ed in eps:
            rp = os.path.join(ed, "results.json")
            if not os.path.exists(rp):
                continue
            try:
                d = json.load(open(rp))
            except Exception:
                continue
            fr = d.get("full_roa") or {}
            b = get(fr, "lambda_delta", "f1")
            c = get(fr, "conservative_lambda_delta", "f1")
            u = get(fr, "lambda_delta", "uncertain_pct") or get(fr, "lambda_delta", "separatrix_pct")
            if b is not None:
                band.append(b); cons.append(c if c is not None else 0); abst.append(u or 0)
                ntraj.append(d.get("train_trajectories"))
        if not band:
            continue
        key = (system, pred, mode)
        # keep the run with most epochs (the real one, not a stale dup)
        if key not in rows or len(band) > rows[key]["n"]:
            rows[key] = dict(n=len(band), band=band, cons=cons, abst=abst, ntraj=ntraj,
                             run=os.path.basename(run))
    return rows


rows = scan()
order = ["pendulum", "cartpole_pybullet", "quadrotor2d", "quadrotor3d"]
print(f"{'system':<18}{'pred':<11}{'mode':<9}{'ep':>3} | {'band best/fin':>14} | {'cons best/fin':>14} | {'abst%fin':>8}")
print("-" * 82)
for sys_ in order:
    for pred in ("classifier", "generative"):
        for mode in ("adaptive", "random"):
            r = rows.get((sys_, pred, mode))
            if not r:
                continue
            bb, bf = max(r["band"]), r["band"][-1]
            cb, cf = max(r["cons"]), r["cons"][-1]
            af = r["abst"][-1]
            print(f"{sys_:<18}{pred:<11}{mode:<9}{r['n']:>3} | "
                  f"{bb:.3f}/{bf:.3f} | {cb:.3f}/{cf:.3f} | {af:>7.1f}")
