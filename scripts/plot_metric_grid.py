#!/usr/bin/env python
"""One grid: every metric (rows) x every noise level (columns), all methods on the same axes.

Colour = acquisition arm, line style = predictor (solid flow matching, dashed
classifier). x is the acquisition epoch, so each panel shows how that metric
evolves as the adaptive process runs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

LEVELS = ["low", "med", "high", "xhigh"]
LEVEL_LABEL = {"low": "low noise", "med": "medium noise",
               "high": "high noise", "xhigh": "extreme noise"}
ARMS = ["dir00", "ent05", "ent10", "tb05", "tb10"]
ARM_LABEL = {"dir00": "non-adaptive d2=0", "ent05": "entropy d2=0.5",
             "ent10": "entropy d2=1.0", "tb05": "entropy+modesep d2=0.5",
             "tb10": "entropy+modesep d2=1.0"}
ARM_C = {"dir00": "#000000", "ent05": "#1f77b4", "ent10": "#d62728",
         "tb05": "#2ca02c", "tb10": "#ff7f0e"}
PRED_STYLE = {"fm": "-", "clf": "--"}

# direction: "down" = lower is better, "up" = higher is better,
# "match" = neither -- the metric is only meaningful against a reference.
METRICS = [
    ("brier_debiased", "debiased Brier", True, "down"),
    ("skill_score", "skill score", False, "up"),
    ("REL_debiased", "REL — miscalibration", False, "down"),
    ("RES", "RES — resolution", False, "up"),
    ("sAUROC", "soft AUROC", False, "up"),
    # Sharpness is NOT better-when-high or better-when-low: an overconfident model
    # and an underconfident one are both wrong. It is only interpretable against
    # the true field's own spread, SHARP*, which is drawn as the dotted line.
    ("SHARP", "sharpness", False, "match"),
    ("AURC", "AURC — selective risk", False, "down"),
    ("KL", "KL(true‖pred)", False, "down"),
    ("f1@0.5", "F1 @ 0.5", False, "up"),
]
DIR_TAG = {"down": "↓ lower is better",
           "up": "↑ higher is better",
           "match": "→ match dotted line"}

plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "legend.fontsize": 10})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", type=Path, default=Path("docs/stoch_compare/metrics.json"))
    ap.add_argument("--out", type=Path, default=Path("docs/stoch_compare/metric_grid.png"))
    ap.add_argument("--only", nargs="*", default=None,
                    help="restrict to these predictors (fm / clf)")
    args = ap.parse_args()

    rows = json.loads(args.metrics.read_text())
    preds = args.only or ["fm", "clf"]

    n_r, n_c = len(METRICS), len(LEVELS)
    fig, axes = plt.subplots(n_r, n_c, figsize=(4.6 * n_c, 2.9 * n_r), squeeze=False)

    for ri, (key, label, logy, direction) in enumerate(METRICS):
        # shared y per metric row so levels are visually comparable
        vals = [r[key] for r in rows if r.get(key) is not None
                and r["predictor"] in preds]
        for ci, lvl in enumerate(LEVELS):
            ax = axes[ri][ci]
            for pred in preds:
                for arm in ARMS:
                    d = {r["epoch"]: r[key] for r in rows
                         if r["predictor"] == pred and r["level"] == lvl
                         and r["arm"] == arm and r.get(key) is not None}
                    if not d:
                        continue
                    xs = sorted(d)
                    ax.plot(xs, [d[e] for e in xs], PRED_STYLE[pred],
                            color=ARM_C[arm], lw=1.7, marker="o", ms=2.6, alpha=0.9)
            if direction == "match":
                # the true field's own spread: the target sharpness, not a bound
                star = [r["SHARP_star"] for r in rows if r["level"] == lvl
                        and r.get("SHARP_star") is not None]
                if star:
                    ax.axhline(sum(star) / len(star), ls=":", color="0.35", lw=1.6)
            if logy and vals and min(v for v in vals if v > 0) > 0:
                ax.set_yscale("log")
            ax.grid(alpha=0.3)
            if ri == 0:
                ax.set_title(LEVEL_LABEL[lvl], fontsize=12)
            if ri == n_r - 1:
                ax.set_xlabel("acquisition epoch")
            if ci == 0:
                ax.set_ylabel(f"{label}\n{DIR_TAG[direction]}")
            # direction arrow on every panel, drawn just outside the axes so it
            # cannot be lost when a single panel is cropped for a slide
            if direction in ("down", "up"):
                ax.annotate("", xy=(1.035, 0.12 if direction == "down" else 0.88),
                            xytext=(1.035, 0.88 if direction == "down" else 0.12),
                            xycoords="axes fraction",
                            arrowprops=dict(arrowstyle="-|>", color="#2b8a3e",
                                            lw=2.0, alpha=0.85))
                ax.annotate("better", xy=(1.055, 0.5), xycoords="axes fraction",
                            rotation=90, va="center", ha="left",
                            fontsize=8.5, color="#2b8a3e")

    # Name the predictor in every entry. Encoding it only as solid-vs-dashed is
    # too easy to miss in a dense grid, and it invites the reading that the
    # classifier was never run adaptively -- it was, on every entropy arm.
    present = [(p, a) for p in preds for a in ARMS
               if any(r["predictor"] == p and r["arm"] == a for r in rows)]
    pretty = {"fm": "FM", "clf": "CLF"}
    handles = [Line2D([], [], color=ARM_C[a], lw=2.2, ls=PRED_STYLE[p],
                      label=f"{pretty[p]} · {ARM_LABEL[a]}") for p, a in present]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 1.012))
    fig.suptitle("Metrics vs acquisition epoch — all methods, all noise levels",
                 y=1.035, fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
