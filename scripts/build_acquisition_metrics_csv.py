#!/usr/bin/env python3
"""
One CSV comparing every acquisition arm on classification quality AND final-state error.

Rows are per-epoch (not just the final one) so arms can be compared at a MATCHED
trajectory budget -- arms run to different lengths, and comparing final epochs
alone compares different budgets.

Columns
-------
classification, from eval_metrics.lambda_delta:
  f1, tpr, fpr, tnr, fnr           TPR=recall, TNR=specificity, computed from tp/fp/tn/fn
  invalid_pct, uncertain_pct       fraction of eval points abstained on, by reason
  sep_pct                          invalid + uncertain (the "separatrix" fraction)

final-state error, from eval_metrics.mc_sample_errors:
  Each eval point has K MC samples, so the error is aggregated twice -- once over
  the K samples for a point, then once over the points. Both use mean/p50/p90,
  giving a 3x3 grid per class: <class>_<outer>_of_<inner>s.
  e.g. full_p90_of_medians = 90th pct over points of (median over MC samples).

  classes: full            all eval points
           certain_success points predicted success (confidently)
           certain_failure points predicted failure (confidently)

Two on-disk layouts are handled: the old baselines nest metrics under
evaluations/<eval>/epoch_XXX/, everything newer writes epoch_XXX/ inline.

Usage:  python scripts/build_acquisition_metrics_csv.py [-o out.csv]
"""
from __future__ import annotations

import argparse, csv, glob, json, os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "docs" / "plots"))
from plot_clf_vs_fm import (EXP, FM_EVAL, NEWCRIT_SYSTEMS, SCHED,   # noqa: E402
                            resolve_fm, resolve_fm_newcrit)

# system key -> experiment dir holding the NEW acquisition arms
NEW_ROOT = {"pendulum": "adaptive_pendulum_lqr", "cartpole": "adaptive_cartpole_pybullet",
            "quad2d": "adaptive_quadrotor2d", "quad3d": "adaptive_quadrotor3d"}
NEW_ARMS = ("entropy", "entropy_tiebreak", "mode_sep", "gated", "dispersion")
BASELINES = ("fm_ranked", "fm_direct", "fm_nonadapt")
ERR_CLASSES = ("full", "certain_success", "certain_failure")
OUTER = ("mean", "median", "p90")          # aggregation over eval points
INNER = ("means", "medians", "p90s")       # aggregation over the K MC samples
# tmux runs have no SLURM_JOB_ID so their dirs are all `exp_local`; the earlier
# smoke tests share that name and are separated by launch timestamp.
CAMPAIGN_CUTOFF = "2026-07-28_14"


def err_cols():
    return [f"{c}_{o}_of_{i}" for c in ERR_CLASSES for o in OUTER for i in INNER]


COLS = (["system", "arm", "selection_rule", "d2_ratio", "epoch", "train_traj", "n_total",
         "f1", "sep_pct", "tpr", "fpr", "tnr", "fnr", "invalid_pct", "uncertain_pct"]
        + err_cols())


def row_from_artifact(art: dict, system: str, arm: str, d2: str, traj, rule: str = "-"):
    em = art.get("eval_metrics") or {}
    ld = em.get("lambda_delta") or {}
    if not ld or ld.get("f1") is None:
        return None
    tp, fp, tn, fn = (ld.get(k) for k in ("tp", "fp", "tn", "fn"))
    div = lambda a, b: (a / b) if (a is not None and b) else None   # noqa: E731
    n_conf, n_unc, n_inv = ld.get("n_confident"), ld.get("n_uncertain"), ld.get("n_invalid")
    n_total = em.get("n_total") or (
        sum(x for x in (n_conf, n_unc, n_inv) if x is not None) or None)

    r = {
        "system": system, "arm": arm, "selection_rule": rule, "d2_ratio": d2,
        "epoch": art.get("epoch"), "train_traj": traj, "n_total": n_total,
        "f1": ld.get("f1"),
        "sep_pct": ld.get("separatrix_pct"),
        "tpr": div(tp, (tp + fn) if None not in (tp, fn) else None),
        "fpr": div(fp, (fp + tn) if None not in (fp, tn) else None),
        "tnr": div(tn, (tn + fp) if None not in (tn, fp) else None),
        "fnr": div(fn, (fn + tp) if None not in (fn, tp) else None),
        "invalid_pct": ld.get("invalid_pct"),
        "uncertain_pct": ld.get("uncertain_pct"),
    }
    mc = em.get("mc_sample_errors") or {}
    for c in ERR_CLASSES:
        blk = mc.get(c) or {}
        for o in OUTER:
            for i in INNER:
                r[f"{c}_{o}_of_{i}"] = blk.get(f"{o}_of_{i}")
    return r


def traj_map(run_dir: str, system: str) -> dict:
    """epoch -> train_trajectories, read from the run's own epoch dirs."""
    init_spe, out = SCHED.get(system), {}
    for e in sorted(glob.glob(f"{run_dir}/epoch_*")):
        ep = int(os.path.basename(e).split("_")[1])
        t = None
        try:
            t = json.load(open(f"{e}/results.json")).get("train_trajectories")
        except Exception:
            pass
        if (not t) and init_spe:
            t = init_spe[0] + ep * init_spe[1]
        out[ep] = t
    return out


def collect_new(system: str, rows: list):
    root = NEW_ROOT.get(system)
    if not root:
        return
    for run in sorted(glob.glob(f"{EXP}/{root}/outputs/*sampling_mode_*")):
        base = os.path.basename(run)
        if "d2_ratio_" not in base:
            continue
        arm = base.split("sampling_mode_")[1].split("_selection")[0]
        if arm not in NEW_ARMS:
            continue
        d2 = base.split("d2_ratio_")[1].split("_")[0]
        rule = base.split("_selection_")[1].rsplit("_exp_", 1)[0] if "_selection_" in base else "-"
        for ts_dir in sorted(glob.glob(f"{run}/*")):
            if not os.path.isdir(ts_dir):
                continue
            if "exp_local" in base and os.path.basename(ts_dir) < CAMPAIGN_CUTOFF:
                continue                      # earlier smoke test, not the campaign
            tmap = traj_map(ts_dir, system)
            for p in sorted(glob.glob(f"{ts_dir}/epoch_*/artifacts_v2.json")):
                try:
                    art = json.load(open(p))
                except Exception:
                    continue
                ep = art.get("epoch")
                r = row_from_artifact(art, system, arm, d2,
                                      art.get("train_trajectories") or tmap.get(ep), rule)
                if r:
                    rows.append(r)


def collect_baseline(system: str, rows: list):
    for arm in BASELINES:
        inline = system in NEWCRIT_SYSTEMS
        d = resolve_fm_newcrit(system, arm) if inline else resolve_fm(system, arm)
        if not d:
            continue
        tmap = traj_map(d, system)
        # old layout keeps the real metrics one level deeper, under evaluations/
        art_dirs = (sorted(glob.glob(f"{d}/epoch_*")) if inline
                    else sorted(glob.glob(f"{d}/evaluations/{FM_EVAL.get(system)}/epoch_*")))
        for e in art_dirs:
            ep = int(os.path.basename(e).split("_")[1])
            try:
                art = json.load(open(f"{e}/artifacts_v2.json"))
            except Exception:
                continue
            art.setdefault("epoch", ep)
            r = row_from_artifact(art, system, arm, "baseline", tmap.get(ep))
            if r:
                rows.append(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="+", default=["pendulum", "cartpole", "quad2d", "quad3d"])
    ap.add_argument("-o", "--out", default="docs/plots/acquisition_arms_metrics.csv")
    a = ap.parse_args()

    rows = []
    for s in a.systems:
        collect_new(s, rows)
        collect_baseline(s, rows)

    rows.sort(key=lambda r: (r["system"], r["arm"], str(r["selection_rule"]), str(r["d2_ratio"]),
                             r["train_traj"] if r["train_traj"] is not None else -1))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {a.out}  ({len(rows)} rows, {len(COLS)} columns)")

    seen = {}
    for r in rows:
        k = (r["system"], r["arm"], r["selection_rule"], r["d2_ratio"])
        seen[k] = seen.get(k, 0) + 1
    print(f"{len(seen)} run-arms:")
    for k in sorted(seen):
        print("  %-10s %-18s %-16s d2=%-9s %d epochs" % (k[0], k[1], k[2], k[3], seen[k]))


if __name__ == "__main__":
    main()
