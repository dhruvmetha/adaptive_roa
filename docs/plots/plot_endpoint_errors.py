#!/usr/bin/env python3
"""
Final-state (endpoint) prediction error by lambda_delta region, per FM arm.

Distinct from clf_vs_fm: that measures ROA *classification* quality (F1/FPR/FNR),
this measures how far the predicted final state lands from the true one, as mean
geodesic distance. A model can classify basins well while predicting endpoints
poorly, and vice versa.

Regions come from eval_metrics.endpoint_errors: full, certain, certain_success,
certain_failure, uncertain, invalid.

Rows are per-epoch (not just the final one) so arms can be compared at a MATCHED
trajectory budget -- the dispersion runs train far longer than the older
baselines on some systems, and comparing final epochs alone would be comparing
different budgets.

Two on-disk layouts, mirroring plot_clf_vs_fm:
  - old (pendulum/cartpole baselines): evaluations/<FM_EVAL>/epoch_XXX/artifacts_v2.json,
    with train_trajectories living in the run's own epoch_XXX/results.json
  - inline (newcrit + all dispersion): epoch_XXX/artifacts_v2.json

Arm resolution is imported from plot_clf_vs_fm so both CSVs describe the same runs.

Writes fm_endpoint_error_by_group.csv and fm_endpoint_error_by_group.png.
"""
import json, glob, os, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_clf_vs_fm import (OUT, SYSTEMS, ARMS, STYLE, SCHED, NEWCRIT_SYSTEMS, FM_EVAL,
                            resolve_fm, resolve_fm_newcrit, resolve_fm_dispersion)

GROUPS = ["full", "certain", "certain_success", "certain_failure", "uncertain", "invalid"]


def resolve(sys, arm):
    if arm.startswith("fm_disp"): return resolve_fm_dispersion(sys, arm)
    if sys in NEWCRIT_SYSTEMS:    return resolve_fm_newcrit(sys, arm)
    return resolve_fm(sys, arm)


def epoch_rows(sys, arm):
    """Per-epoch endpoint-error rows for one arm: [(traj, epoch, endpoint_errors)]."""
    d = resolve(sys, arm)
    if not d: return []
    inline = arm.startswith("fm_disp") or sys in NEWCRIT_SYSTEMS

    # train_trajectories always come from the run's own epoch dirs
    init_spe, traj_of = SCHED.get(sys), {}
    for e in sorted(glob.glob(f"{d}/epoch_*")):
        ep = int(os.path.basename(e).split("_")[1])
        t = None
        try: t = json.load(open(f"{e}/results.json")).get("train_trajectories")
        except Exception: pass
        if (not t) and init_spe: t = init_spe[0] + ep * init_spe[1]
        traj_of[ep] = t

    if inline:
        art_dirs = sorted(glob.glob(f"{d}/epoch_*"))
    else:
        ev = FM_EVAL.get(sys)
        if not ev or not os.path.isdir(f"{d}/evaluations/{ev}"): return []
        art_dirs = sorted(glob.glob(f"{d}/evaluations/{ev}/epoch_*"))

    out = []
    for e in art_dirs:
        ep = int(os.path.basename(e).split("_")[1])
        try: a = json.load(open(f"{e}/artifacts_v2.json"))
        except Exception: continue
        ee = (a.get("eval_metrics") or {}).get("endpoint_errors")
        if not isinstance(ee, dict) or "full" not in ee: continue
        t = traj_of.get(ep) or a.get("train_trajectories")
        if t is None: continue
        out.append((t, ep, ee))
    out.sort()
    return out


rows = []
for s in SYSTEMS:
    for a in [x for x in ARMS if x.startswith("fm")]:
        for traj, ep, ee in epoch_rows(s, a):
            r = {"system": s, "arm": a, "label": STYLE.get(a, {}).get("label", a),
                 "epoch": ep, "train_traj": traj}
            for g in GROUPS:
                v = ee.get(g) or {}
                r[f"{g}_mean"], r[f"{g}_median"], r[f"{g}_n"] = v.get("mean"), v.get("median"), v.get("n_points")
            rows.append(r)

csv_path = f"{OUT}/fm_endpoint_error_by_group.csv"
cols = ["system", "arm", "label", "epoch", "train_traj"] + \
       [f"{g}_{k}" for g in GROUPS for k in ("mean", "median", "n")]
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
    for r in rows: w.writerow(r)
print("wrote", csv_path, f"({len(rows)} rows)")

# --- matched-budget comparison -------------------------------------------
# For each system, use the largest trajectory budget every arm reaches, and take
# each arm's last epoch at or below it. Comparing final epochs directly would
# favour whichever arm simply trained longer.
print("\n" + "=" * 78)
print("MATCHED-BUDGET final-state error (mean geodesic; lower is better)")
print("=" * 78)
summary = {}
for s in SYSTEMS:
    per_arm = {}
    for a in sorted({r["arm"] for r in rows if r["system"] == s}):
        v = sorted([r for r in rows if r["system"] == s and r["arm"] == a], key=lambda r: r["train_traj"])
        if v: per_arm[a] = v
    if len(per_arm) < 2: continue
    cap = min(v[-1]["train_traj"] for v in per_arm.values())
    picked = []
    for a, v in per_arm.items():
        c = [r for r in v if r["train_traj"] <= cap]
        if c: picked.append(c[-1])
    picked.sort(key=lambda r: r["full_mean"] if r["full_mean"] is not None else float("inf"))
    summary[s] = (cap, picked)
    print(f"\n--- {s}   matched budget = {cap} trajectories ---")
    for r in picked:
        tag = "" if r["arm"].startswith("fm_disp") else "   <-- baseline"
        cs = r.get("certain_success_mean"); cf = r.get("certain_failure_mean")
        print("  %-22s traj=%-7s full=%8.4f  cert_succ=%s  cert_fail=%s%s" % (
            r["arm"], r["train_traj"], r["full_mean"],
            ("%7.4f" % cs) if cs is not None else "   n/a ",
            ("%8.4f" % cf) if cf is not None else "    n/a ", tag))

# --- plot: matched-budget bars per system --------------------------------
plot_sys = [s for s in SYSTEMS if s in summary]
PLOT_GROUPS = [("full", "all eval points"), ("certain_success", "certain & success"),
               ("certain_failure", "certain & failure")]
fig, axes = plt.subplots(len(PLOT_GROUPS), len(plot_sys),
                         figsize=(4.4 * len(plot_sys), 3.4 * len(PLOT_GROUPS)), squeeze=False)
for ci, s in enumerate(plot_sys):
    cap, picked = summary[s]
    for ri, (g, gname) in enumerate(PLOT_GROUPS):
        ax = axes[ri][ci]
        vals = [(r["arm"], r.get(f"{g}_mean")) for r in picked if r.get(f"{g}_mean") is not None]
        if vals:
            names, ys = zip(*vals)
            ax.bar(range(len(ys)), ys, color=[STYLE.get(n, {}).get("color", "#888") for n in names],
                   hatch=["" if n.startswith("fm_disp") else "//" for n in names], edgecolor="black", linewidth=0.4)
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels([n.replace("fm_", "").replace("disp_", "") for n in names],
                               rotation=60, ha="right", fontsize=7)
            ax.set_yscale("log")
        if ci == 0: ax.set_ylabel(f"{gname}\nmean geodesic err")
        if ri == 0: ax.set_title(f"{s}  (@{cap} traj)")
        ax.grid(True, axis="y", alpha=0.3)
fig.suptitle("FM final-state error at matched trajectory budget (hatched = baseline, log scale)", y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.98])
fig.savefig(f"{OUT}/fm_endpoint_error_by_group.png", dpi=130, bbox_inches="tight")
print("\nwrote fm_endpoint_error_by_group.png")
