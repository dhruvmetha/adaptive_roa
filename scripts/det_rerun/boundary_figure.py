#!/usr/bin/env python
"""Decision maps at three budgets against rollout labels, shared axes (tracker H06).

Rows are runs, columns are the rollout labels followed by one panel per budget.
Every test state is colored by the run's decision (success, failure, abstain);
the black line is the rollout boundary, the 0.5 level of the labels over a
triangulation of the test states.

Usage: boundary_figure.py --runs RUN [RUN ...] --epochs 0 5 10 [--dims 0 1] --out fig.png
"""
import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from matplotlib.lines import Line2D

from adaptive_roa.adaptive_v2.eval.store_boundary import ABSTAIN, FAILURE, SUCCESS, decision_maps

INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
COLORS = {SUCCESS: "#2a78d6", FAILURE: "#eb6834", ABSTAIN: "#c9c8c2"}
NAMES = {SUCCESS: "success", FAILURE: "failure", ABSTAIN: "abstain"}


def rollout_boundary(ax, xy, y01):
    tri = mtri.Triangulation(xy[:, 0], xy[:, 1])
    # Drop long triangles: they bridge gaps in the state cloud (and the angle wrap).
    e = np.stack([np.hypot(*(xy[tri.triangles[:, i]] - xy[tri.triangles[:, (i + 1) % 3]]).T) for i in range(3)], 1)
    tri.set_mask(e.max(1) > 4 * np.median(e))
    ax.tricontour(tri, y01.astype(float), levels=[0.5], colors=INK, linewidths=1.0)


def panel(ax, xy, cls):
    order = np.argsort(cls == ABSTAIN)[::-1]     # abstain underneath, decisions on top
    ax.scatter(xy[order, 0], xy[order, 1], c=[COLORS[c] for c in cls[order]], s=1.2, linewidths=0, rasterized=True)
    ax.set_facecolor(SURFACE)
    for s in ax.spines.values():
        s.set_color(MUTED); s.set_linewidth(0.5)
    ax.tick_params(colors=MUTED, labelsize=7, width=0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", default=None, help="row names, one per run")
    ap.add_argument("--epochs", nargs="+", type=int, default=[0, 5, 10])
    ap.add_argument("--dims", nargs=2, type=int, default=[0, 1])
    ap.add_argument("--axis-names", nargs=2, default=[r"$\theta$", r"$\dot\theta$"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = [decision_maps(r, set(a.epochs), tuple(a.dims)) for r in a.runs]
    ncol = 1 + max(len(m) for m in rows)
    fig, axes = plt.subplots(len(rows), ncol, figsize=(2.3 * ncol, 2.3 * len(rows) + 0.5),
                             sharex=True, sharey=True, squeeze=False, facecolor="white")
    table = []
    for i, maps in enumerate(rows):
        name = a.labels[i] if a.labels else Path(a.runs[i]).name
        truth = maps[0]
        panel(axes[i, 0], truth["xy"], np.where(truth["label"] == 1, SUCCESS, FAILURE))
        rollout_boundary(axes[i, 0], truth["xy"], truth["label"])
        axes[i, 0].set_title("rollout labels", fontsize=8, color=INK)
        axes[i, 0].set_ylabel(f"{name}\n{a.axis_names[1]}", fontsize=8, color=INK)
        for j, m in enumerate(maps, start=1):
            panel(axes[i, j], m["xy"], m["decision"])
            rollout_boundary(axes[i, j], m["xy"], m["label"])
            axes[i, j].set_title(f"{m['train_trajectories']:,} trajectories\n"
                                 f"committed {m['committed_share']:.0%}, wrong {m['wrong_share_of_committed']:.1%}",
                                 fontsize=7.5, color=INK)
            table.append({k: m[k] for k in ("run", "epoch", "train_trajectories", "committed_share", "abstain_share",
                                             "wrong_share_of_committed", "success_recovery", "success_precision")})
        for j in range(len(maps) + 1, ncol):
            axes[i, j].set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel(a.axis_names[0], fontsize=8, color=INK)
    handles = [Line2D([], [], ls="", marker="o", ms=5, mfc=COLORS[c], mec="none", label=NAMES[c])
               for c in (SUCCESS, FAILURE, ABSTAIN)]
    handles.append(Line2D([], [], color=INK, lw=1.0, label="rollout boundary"))
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.35 / fig.get_figheight()))
    fig.savefig(a.out, dpi=300)
    with open(Path(a.out).with_suffix(".csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]), lineterminator="\n")
        w.writeheader(); w.writerows(table)
    print(f"wrote {a.out} and its table")


if __name__ == "__main__":
    main()
