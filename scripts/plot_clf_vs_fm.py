#!/usr/bin/env python
"""Final comparison figure: classifier vs flow-matching across systems.
Reuses analyze_results.scan() for tonight's runs. Produces grouped bars of
best conservative F1 and best band F1, classifier vs FM, adaptive vs random.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_results import scan  # noqa

OUT = sys.argv[1] if len(sys.argv) > 1 else "/common/users/shared/pracsys/adaptive_roa_experiments/dhruv/clf_vs_fm_summary.png"
rows = scan()
systems = ["pendulum", "cartpole_pybullet", "quadrotor2d", "quadrotor3d"]
labels = ["pendulum", "cartpole", "quad2d", "quad3d"]


def bestcons(sys_, pred, mode):
    r = rows.get((sys_, pred, mode))
    return max(r["cons"]) if r else None


def bestband(sys_, pred, mode):
    r = rows.get((sys_, pred, mode))
    return max(r["band"]) if r else None


fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
x = np.arange(len(systems)); w = 0.2
for ax, metric, fn, title in [
    (axes[0], "cons", bestcons, "Conservative (full-coverage) F1 — best over epochs"),
    (axes[1], "band", bestband, "Band (non-abstained) F1 — best over epochs"),
]:
    series = [
        ("clf adaptive", "classifier", "adaptive", "#1b6ca8"),
        ("clf random", "classifier", "random", "#7fbfe0"),
        ("FM adaptive", "generative", "adaptive", "#c8541a"),
        ("FM random", "generative", "random", "#f0a878"),
    ]
    for i, (lab, pred, mode, c) in enumerate(series):
        vals = [fn(s, pred, mode) for s in systems]
        vals = [v if v is not None else 0 for v in vals]
        bars = ax.bar(x + (i - 1.5) * w, vals, w, label=lab, color=c)
        for b, v in zip(bars, vals):
            if v > 0:
                ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}",
                        ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05); ax.set_ylabel("F1"); ax.set_title(title)
    ax.grid(axis="y", alpha=0.3); ax.legend(fontsize=8, loc="lower left")
plt.suptitle("Classifier vs Flow-Matching for ROA (matched schedules) — missing bars = run incomplete/not-run")
plt.tight_layout()
plt.savefig(OUT, dpi=140)
print("SAVED", OUT)
