#!/usr/bin/env python3
"""
Final-state error of every FM arm at its LAST training epoch, as bar charts.

The error at one query point is not a single number: the model draws K MC samples
per point, so there are two aggregation axes.

  inner  -- over the K MC samples AT one query point  -> mean / median / p90
  outer  -- over the query points OF the dataset      -> mean / median / p90

Each combination answers a different question ("typical spread of a typical
point" vs "worst-case spread of a typical point" vs "typical spread of a
worst-case point", ...), so this emits all 3x3 = 9 figures rather than picking
one. eval_metrics.mc_sample_errors already stores exactly this grid under
`{outer}_of_{inner}` keys, so nothing is recomputed here.

Each figure: 3 rows x 4 columns.
  rows    -- full dataset / predicted-success class / predicted-failure class
  columns -- pendulum, cartpole, quad2d, quad3d
  bars    -- FM arms, linear with a zero baseline (a bar's length must
             mean something; outliers are clipped and labelled)

Colour encodes the acquisition family and hatch encodes d2_ratio, so an arm is
identified by two channels rather than colour alone -- nine arms is past the
point where hue alone stays discriminable. The palette passes the categorical
checks of the dataviz validator (lightness band, chroma floor, adjacent-pair CVD
separation dE 15.2, normal-vision floor 18.7); its one residual sub-3:1 contrast
slot is relieved by the CSV table this script also writes.

NOTE ON BUDGETS: "last epoch" is each arm's own final epoch, which is NOT a
matched trajectory budget -- the dispersion arms train to 1900 trajectories on
pendulum where the baselines stop near 1000. Read these as "where each arm ended
up", not "which arm is better per unit data"; for the matched-budget view use
plot_endpoint_errors.py.

Writes fm_final_state_error_<outer>_of_<inner>.png (9 files) and
fm_final_state_error_last_epoch.csv.
"""
from __future__ import annotations

import csv, glob, json, os, sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from plot_clf_vs_fm import (OUT, SCHED, NEWCRIT_SYSTEMS, FM_EVAL,
                            resolve_fm, resolve_fm_newcrit, resolve_fm_dispersion)

SYSTEMS = ["pendulum", "cartpole", "quad2d", "quad3d"]
INNER = [("means", "mean over MC samples"),
         ("medians", "median over MC samples"),
         ("p90s", "p90 over MC samples")]
OUTER = [("mean", "mean over dataset"),
         ("median", "median over dataset"),
         ("p90", "p90 over dataset")]
REGIONS = [("full", "full dataset"),
           ("certain_success", "predicted success"),
           ("certain_failure", "predicted failure")]

# Colour = acquisition family, hatch = d2_ratio. Fixed order, never cycled.
STYLE = {
    "fm_nonadapt":        ("#D55E00", "",   "random (d2=0)"),
    "fm_ranked":          ("#0072B2", "//", "ranked (d2=1.0)"),
    "fm_direct":          ("#009E73", "//", "direct (d2=1.0)"),
    "fm_disp_greedy_d05": ("#7B3294", "",   "disp greedy (d2=0.5)"),
    "fm_disp_greedy_d10": ("#7B3294", "//", "disp greedy (d2=1.0)"),
    "fm_disp_gdiv_d05":   ("#B8860B", "",   "disp greedy-div (d2=0.5)"),
    "fm_disp_gdiv_d10":   ("#B8860B", "//", "disp greedy-div (d2=1.0)"),
    "fm_disp_prop_d05":   ("#CC79A7", "",   "disp proportional (d2=0.5)"),
    "fm_disp_prop_d10":   ("#CC79A7", "//", "disp proportional (d2=1.0)"),
}
ORDER = list(STYLE)
INK, MUTED = "#1a1a1a", "#6b6b6b"


def resolve(sys_key, arm):
    if arm.startswith("fm_disp"): return resolve_fm_dispersion(sys_key, arm)
    if sys_key in NEWCRIT_SYSTEMS: return resolve_fm_newcrit(sys_key, arm)
    return resolve_fm(sys_key, arm)


def last_epoch_mc_errors(sys_key, arm):
    """(epoch, train_traj, mc_sample_errors) at the arm's last epoch, or None.

    Handles both on-disk layouts: dispersion/newcrit write artifacts inline under
    epoch_XXX/, while the older pendulum/cartpole baselines write under
    evaluations/<FM_EVAL>/epoch_XXX/ with train_trajectories recorded in the run's
    own epoch dirs.
    """
    d = resolve(sys_key, arm)
    if not d:
        return None
    inline = arm.startswith("fm_disp") or sys_key in NEWCRIT_SYSTEMS

    init_spe, traj_of = SCHED.get(sys_key), {}
    for e in sorted(glob.glob(f"{d}/epoch_*")):
        ep = int(os.path.basename(e).split("_")[1])
        t = None
        try:
            t = json.load(open(f"{e}/results.json")).get("train_trajectories")
        except Exception:
            pass
        if (not t) and init_spe:
            t = init_spe[0] + ep * init_spe[1]
        traj_of[ep] = t

    if inline:
        art = sorted(glob.glob(f"{d}/epoch_*"))
    else:
        ev = FM_EVAL.get(sys_key)
        if not ev or not os.path.isdir(f"{d}/evaluations/{ev}"):
            return None
        art = sorted(glob.glob(f"{d}/evaluations/{ev}/epoch_*"))

    best = None
    for e in art:
        ep = int(os.path.basename(e).split("_")[1])
        try:
            a = json.load(open(f"{e}/artifacts_v2.json"))
        except Exception:
            continue
        mc = (a.get("eval_metrics") or {}).get("mc_sample_errors")
        if not isinstance(mc, dict) or "full" not in mc:
            continue
        if best is None or ep >= best[0]:
            best = (ep, traj_of.get(ep) or a.get("train_trajectories"), mc)
    return best


data = {}
for s in SYSTEMS:
    for a in ORDER:
        got = last_epoch_mc_errors(s, a)
        if got:
            data[(s, a)] = got

csv_path = f"{OUT}/fm_final_state_error_last_epoch.csv"
with open(csv_path, "w", newline="") as f:
    cols = ["system", "arm", "epoch", "train_traj", "region", "n_points"] + \
           [f"{o}_of_{i}" for i, _ in INNER for o, _ in OUTER]
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    for (s, a), (ep, traj, mc) in sorted(data.items()):
        for region, _ in REGIONS:
            r = mc.get(region) or {}
            if not r:
                continue
            row = dict(system=s, arm=a, epoch=ep, train_traj=traj,
                       region=region, n_points=r.get("n_points"))
            for i, _ in INNER:
                for o, _ in OUTER:
                    row[f"{o}_of_{i}"] = r.get(f"{o}_of_{i}")
            w.writerow(row)
print(f"wrote {csv_path}")

for inner, inner_lbl in INNER:
    for outer, outer_lbl in OUTER:
        key = f"{outer}_of_{inner}"
        fig, axes = plt.subplots(len(REGIONS), len(SYSTEMS),
                                 figsize=(4.6 * len(SYSTEMS), 3.5 * len(REGIONS)),
                                 squeeze=False)
        drawn = set()
        for ci, s in enumerate(SYSTEMS):
            for ri, (region, region_lbl) in enumerate(REGIONS):
                ax = axes[ri][ci]
                names, vals = [], []
                for a in ORDER:
                    got = data.get((s, a))
                    if not got:
                        continue
                    v = (got[2].get(region) or {}).get(key)
                    if v is None or not np.isfinite(v):
                        continue
                    names.append(a)
                    vals.append(v)
                if names:
                    # Linear with a zero baseline: a bar encodes magnitude by its
                    # length from zero, so a log axis (whose floor is arbitrary)
                    # would make length meaningless. Measured spread is 1.1-10x in
                    # 27 of 36 panels, where log would exaggerate near-identical
                    # bars; only pendulum's random arm is a true outlier (100-935x).
                    # Clip that one and print its value rather than let it flatten
                    # the eight arms the comparison is actually about.
                    med = float(np.median(vals))
                    inliers = [v for v in vals if v <= 5.0 * med] or vals
                    ymax = 1.18 * max(inliers)
                    for x, (a, v) in enumerate(zip(names, vals)):
                        c, hatch, _ = STYLE[a]
                        clipped = v > ymax
                        ax.bar(x, min(v, ymax), width=0.78, color=c, hatch=hatch,
                               edgecolor="white", linewidth=0.8)
                        if clipped:
                            ax.annotate(f"{v:,.3g} \u2191", (x, ymax),
                                        textcoords="offset points", xytext=(0, 2),
                                        ha="center", fontsize=6.5, color=INK)
                        drawn.add(a)
                    ax.set_ylim(0, ymax)
                    ax.set_xticks(range(len(names)))
                    ax.set_xticklabels([STYLE[a][2] for a in names],
                                       rotation=55, ha="right", fontsize=6.5, color=MUTED)
                else:
                    ax.set_xticks([])
                    ax.text(0.5, 0.5, "no data", ha="center", va="center",
                            transform=ax.transAxes, color=MUTED, fontsize=9)
                if ci == 0:
                    ax.set_ylabel(f"{region_lbl}\nerror", fontsize=9, color=INK)
                if ri == 0:
                    ax.set_title(s, fontsize=11, color=INK)
                ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)
                ax.set_axisbelow(True)
                for spine in ("top", "right"):
                    ax.spines[spine].set_visible(False)
                for spine in ("left", "bottom"):
                    ax.spines[spine].set_color(MUTED)
                    ax.spines[spine].set_linewidth(0.8)
                ax.tick_params(colors=MUTED, labelsize=7)

        handles = [Patch(facecolor=STYLE[a][0], hatch=STYLE[a][1], edgecolor="white",
                         label=STYLE[a][2]) for a in ORDER if a in drawn]
        fig.legend(handles=handles, loc="lower center", ncol=min(5, len(handles)),
                   frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.015))
        fig.suptitle(
            f"FM final-state error at last training epoch — {outer_lbl} of ({inner_lbl})",
            fontsize=13, color=INK, y=0.998)
        fig.tight_layout(rect=[0, 0.055, 1, 0.975])
        path = f"{OUT}/fm_final_state_error_{key}.png"
        fig.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {os.path.basename(path)}")

print(f"\n{len(data)} (system, arm) pairs across {len(SYSTEMS)} systems")
for s in SYSTEMS:
    arms = [a for (ss, a) in data if ss == s]
    if arms:
        print(f"  {s:9} {len(arms)} arms, last epochs: "
              f"{sorted({data[(s, a)][0] for a in arms})}")
