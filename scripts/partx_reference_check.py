#!/usr/bin/env python
"""One-off reference check (spec §8.2): our Part-X loop in deterministic mode vs cpslab.

Runs in a scratch env, because cpslab needs numpy<2, treelib and pathos:
  PYTHONPATH=<cpslab clone>:<repo> <env>/bin/python scripts/partx_reference_check.py \
      --reps 10 --budget 5000 --jobs 16 --cpslab-commit <sha> --out docs/partx_faithful_reference_check.md
"""
from __future__ import annotations

import argparse
import pathlib
import tempfile
from concurrent.futures import ProcessPoolExecutor

import numpy as np


def himmelblau(x):
    return (x[0] ** 2 + x[1] - 11.0) ** 2 + (x[0] + x[1] ** 2 - 7.0) ** 2 - 40.0


def goldstein(x):
    a = 1 + (x[0] + x[1] + 1) ** 2 * (19 - 14 * x[0] + 3 * x[0] ** 2 - 14 * x[1] + 6 * x[0] * x[1] + 3 * x[1] ** 2)
    b = 30 + (2 * x[0] - 3 * x[1]) ** 2 * (18 - 32 * x[0] + 12 * x[0] ** 2 + 48 * x[1] - 36 * x[0] * x[1] + 27 * x[1] ** 2)
    return a * b - 50.0


FUNCS = {"himmelblau": (himmelblau, np.array([[-5.0, 5.0], [-5.0, 5.0]])),
         "goldstein": (goldstein, np.array([[-1.0, 1.0], [-1.0, 1.0]]))}
PAPER_T1 = {"himmelblau": 17.671, "goldstein": 0.304}     # P-X (dq = 0.5), R = 10, M = 100
PAPER_MC = {"himmelblau": 17.030, "goldstein": 0.302}
S = dict(n0=10, n_bo=10, n_c=100, R=10, M=100, B=2, delta_c=0.05, delta_v=0.001)   # paper §5.1
Q = [0.5, 0.05, 0.01]
GROUPS = {"+": ("+",), "-": ("-",), "remaining": ("r", "r+", "r-"), "u": ("u",)}


def area(sup):
    return float(np.prod(sup[:, 1] - sup[:, 0]))


def monte_carlo(fn, sup, n=1_000_000, seed=0):
    rng = np.random.default_rng(seed)
    U = sup[:, 0] + rng.random((n, 2)) * (sup[:, 1] - sup[:, 0])
    return float(np.mean(fn(U.T) < 0.0)) * area(sup)


def run_cpslab(args):
    name, rep, budget = args
    from partx.bayesianOptimization import InternalBO
    from partx.coreAlgorithm import PartXOptions, run_single_replication
    from partx.gprInterface import InternalGPR
    from partx.results import fv_using_gp
    from partx.utils import OracleCreator

    fn, sup = FUNCS[name]
    opts = PartXOptions("ref", sup, 2, budget, S["n0"], S["n_bo"], S["n_c"], S["delta_c"], S["R"], S["M"],
                        S["delta_v"], Q, S["B"], True, 12345, InternalGPR(), InternalBO())
    oracle = OracleCreator(None, 1, 1)
    with tempfile.TemporaryDirectory() as work:
        ftree = run_single_replication((rep, opts, fn, oracle, pathlib.Path(work)))["ftree"]
        fv = np.asarray(fv_using_gp(ftree, opts, oracle, Q, np.random.default_rng(12345 + rep))).sum(axis=0)
    return {"fv": (fv * area(sup)).tolist(), "types": [leaf.data.region_class for leaf in ftree.leaves()]}


def run_ours(args):
    name, rep, budget = args
    from adaptive_roa.partx_faithful.algorithm import PartX, PartXConfig
    from adaptive_roa.partx_faithful.backends import backend_factory
    from adaptive_roa.partx_faithful.sampling import ContinuousSampler
    from adaptive_roa.partx_faithful.volume import gp_quantile_falsification_volume

    fn, sup = FUNCS[name]
    cfg = PartXConfig(**S, ei_cand_cap=10_000, ei_f_star="observed", compat_ci_var=True,
                      compat_eq4_times_volume=True, seed=12345 + rep)
    px = PartX(cfg, sup[:, 0], sup[:, 1], ContinuousSampler(fn), backend_factory("gpr"), budget)
    px.run()
    leaves = px.leaves()
    models = {r.rid: px.fit(r) for r in leaves if r.rtype not in ("u", "i") and r.samples}
    fv = gp_quantile_falsification_volume(leaves, models, S["R"], S["M"], Q,
                                          np.random.default_rng(12345 + rep), px.root_volume)
    return {"fv": [fv[q] * px.root_volume for q in Q], "types": [r.rtype for r in leaves]}


def stats(results):
    fv = np.array([r["fv"][0] for r in results])
    counts = {g: np.array([sum(t in ts for t in r["types"]) for r in results], dtype=float)
              for g, ts in GROUPS.items()}
    return fv, counts


def within_2se(a, b):
    se = float(np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)))
    return abs(a.mean() - b.mean()) <= 2.0 * se + 1e-12, se


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--budget", type=int, default=5000)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--cpslab-commit", default="unknown")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    lines = ["# partx_faithful reference check", "",
             f"cpslab commit `{args.cpslab_commit}`; paper §5.1 settings {S}; T = {args.budget}; "
             f"{args.reps} replications per function per implementation.", "",
             "Volumes are raw areas, as in the paper's tables. Pass = within 2 SE (spec §8.2).", ""]
    overall = True
    for name, (fn, sup) in FUNCS.items():
        jobs = [(name, rep, args.budget) for rep in range(args.reps)]
        with ProcessPoolExecutor(args.jobs) as ex:
            ref = list(ex.map(run_cpslab, jobs))
            ours = list(ex.map(run_ours, jobs))
        fv_ref, c_ref = stats(ref)
        fv_our, c_our = stats(ours)
        ok_fv, se_fv = within_2se(fv_our, fv_ref)
        mc = monte_carlo(fn, sup)
        lines += [f"## {name}", "",
                  "| quantity | cpslab mean | ours mean | 2·SE | pass |", "|---|---:|---:|---:|---|",
                  f"| GP-quantile volume (q = 0.5) | {fv_ref.mean():.4f} | {fv_our.mean():.4f} | "
                  f"{2 * se_fv:.4f} | {'yes' if ok_fv else 'NO'} |"]
        ok_all = ok_fv
        for g in GROUPS:
            ok, se = within_2se(c_our[g], c_ref[g])
            ok_all &= ok
            lines.append(f"| leaves of type {g} | {c_ref[g].mean():.1f} | {c_our[g].mean():.1f} | "
                         f"{2 * se:.1f} | {'yes' if ok else 'NO'} |")
        rel = abs(fv_our.mean() - fv_ref.mean()) / mc
        lines += ["", f"Monte Carlo truth (1e6 points): {mc:.4f}. Paper Table 1 P-X (q = 0.5): "
                  f"{PAPER_T1[name]}, paper Monte Carlo: {PAPER_MC[name]}. "
                  f"|ours − cpslab| / truth = {rel:.2%}.", "",
                  f"**{name}: {'PASS' if ok_all else 'FAIL'}**", ""]
        overall &= ok_all
    lines.append(f"**Overall: {'PASS' if overall else 'FAIL'}**")
    pathlib.Path(args.out).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
