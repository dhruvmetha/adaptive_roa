#!/usr/bin/env python
"""One figure per noise family: rows = levels, columns = KL / debiased Brier / sAUROC.

Follows the convention already documented in /common/users/shared/pracsys/genMoPlan/docs/stochastic/{pendulum,
cartpole}/README.md and used by their `*_all_levels.png`:

    solid   = flow matching        dotted = classifier      dash-dot = Part-X GP
    colour  = acquisition arm (same colour for the same arm across every figure)
    shaded  = 3-seed non-adaptive range; the 2*SD floor is printed in the panel

Output goes NEXT TO the campaign README, not into figs/, and is named
<prefix><family>_all_levels.{png}. The quadrotor directory serves two systems, so
unlike pendulum/cartpole it needs the quad2d_/quad3d_ prefix to stay unambiguous --
`noisy_dynamics` exists for both.

The artifact AUC/Brier is never plotted: it scores against a 0.5-dichotomised truth.
Everything here comes from the stoch_prob_metrics projection over the continuous
ground-truth p_success grid.
"""
import argparse, csv, collections, datetime, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

PRED_STYLE = {"fm": "-", "clf": ":", "gp": "-."}      # README convention
PRED_LABEL = {"fm": "flow matching", "clf": "classifier", "gp": "Part-X GP"}
SEEDS = ["dir00_s42", "dir00_s43", "dir00_s44"]

# colour = acquisition arm, stable across every figure so two panels can be read together
ARM_C = {
    "dir00":        "#000000",
    "epi_var":      "#1f77b4",
    "epi_var_anch": "#17becf",
    "epi_bald":     "#d62728",
    "yield_a1":     "#2ca02c",
    "yield_mlp":    "#ff7f0e",
    "partx":        "#8c564b",
}
def base_arm(a):
    return "dir00" if a.startswith("dir00") else a[4:] if a.startswith("clf_") else a

METRICS = [("KL", "KL divergence", "down", True),
           ("brier_debiased", "debiased Brier", "down", True),
           ("sAUROC", "soft AUROC", "up", False)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--prefix", default="")
    ap.add_argument("--family-map", required=True,
                    help="JSON: {csv_level_value: [family, level_label]}")
    args = ap.parse_args()

    import json
    fmap = json.loads(args.family_map)
    rows = list(csv.DictReader(open(args.csv)))
    stamp = datetime.datetime.fromtimestamp(os.path.getmtime(args.csv)).strftime("%Y-%m-%d %H:%M")

    fams = collections.defaultdict(list)
    for lv, (fam, lab) in fmap.items():
        fams[fam].append((lv, lab))

    for fam, levels in fams.items():
        levels.sort(key=lambda t: t[1])
        nrow, ncol = len(levels), len(METRICS)
        fig, axes = plt.subplots(nrow, ncol, figsize=(6.2*ncol, 4.4*nrow), squeeze=False)

        for ri, (lv, lab) in enumerate(levels):
            sub = [r for r in rows if r["level"] == lv]
            series = collections.defaultdict(list)
            for r in sub:
                series[(r["predictor"], r["arm"])].append(r)
            for k in series:
                series[k].sort(key=lambda r: int(r["epoch"]))

            for ci, (metric, mlabel, direction, logy) in enumerate(METRICS):
                ax = axes[ri][ci]

                # 3-seed non-adaptive band + printed 2*SD floor
                have = [s for s in SEEDS if ("fm", s) in series]
                floor_txt = f"no floor: {len(have)}/3 seeds"
                if len(have) == 3:
                    per = collections.defaultdict(list)
                    for s in have:
                        for r in series[("fm", s)]:
                            e = int(r["epoch"])
                            if e == 0:
                                continue
                            try: per[e].append(float(r[metric]))
                            except Exception: pass
                    sh = [e for e in sorted(per) if len(per[e]) == 3]
                    if sh:
                        lo = np.array([min(per[e]) for e in sh])
                        hi = np.array([max(per[e]) for e in sh])
                        ax.fill_between(sh, lo, hi, color="0.72", alpha=.6, zorder=0)
                        sds = [np.std(per[e], ddof=1) for e in sh]
                        pooled = [v for e in sh for v in per[e]]
                        floor_txt = (f"2·SD floor  per-epoch mean {2*np.mean(sds):.4f}\n"
                                     f"            pooled       {2*np.std(pooled, ddof=1):.4f}")

                for (pred, arm), rs in sorted(series.items()):
                    if arm in SEEDS:
                        continue                      # seeds are the band
                    xs = [int(r["epoch"]) for r in rs]
                    ys = []
                    for r in rs:
                        try: ys.append(float(r[metric]))
                        except Exception: ys.append(np.nan)
                    ax.plot(xs, ys, PRED_STYLE.get(pred, "-"), lw=1.7,
                            color=ARM_C.get(base_arm(arm), "#7f7f7f"))

                ax.set_xlabel("acquisition epoch")
                ax.set_ylabel(mlabel)
                ax.set_title(f"{lab} — {mlabel} ({'lower' if direction=='down' else 'higher'} better)",
                             fontsize=10)
                if logy: ax.set_yscale("log")
                ax.grid(alpha=.3)
                ax.text(.98, .97, floor_txt, transform=ax.transAxes, ha="right", va="top",
                        fontsize=7.5, family="monospace",
                        bbox=dict(fc="white", ec="0.7", alpha=.85, pad=2.5))

        arms_seen = sorted({base_arm(r["arm"]) for r in rows
                            if r["level"] in dict(levels)} - {"dir00"})
        handles = [Line2D([], [], color="0.72", lw=8, label="3-seed non-adaptive range")]
        handles += [Line2D([], [], color=ARM_C.get(a, "#7f7f7f"), lw=2.2, label=a)
                    for a in arms_seen]
        handles += [Line2D([], [], color="0.2", lw=2.2, ls=st, label=PRED_LABEL[p])
                    for p, st in PRED_STYLE.items()]
        fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, fontsize=9,
                   bbox_to_anchor=(.5, -.01))
        fig.suptitle(f"{fam} — all levels     (scored {stamp})", fontsize=13)
        fig.tight_layout(rect=[0, .06, 1, .95])
        out = os.path.join(args.outdir, f"{args.prefix}{fam}_all_levels.png")
        fig.savefig(out, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {out}")


if __name__ == "__main__":
    main()
