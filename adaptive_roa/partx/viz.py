from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

_COLORS = {"+": "#2ca02c", "-": "#d62728", "r": "#7f7f7f", "min": "#bcbd22"}


def plot_region_tree(tree, out_path, gp=None, resolution=200):
    fig, ax = plt.subplots(figsize=(6, 5))
    root = tree.root
    for leaf in tree.leaves():
        w = leaf.high[0] - leaf.low[0]
        h = leaf.high[1] - leaf.low[1]
        ax.add_patch(mpatches.Rectangle(
            (leaf.low[0], leaf.low[1]), w, h,
            facecolor=_COLORS.get(leaf.region_class, "#999999"),
            edgecolor="black", alpha=0.5, linewidth=0.5))
    ax.set_xlim(root.low[0], root.high[0])
    ax.set_ylim(root.low[1], root.high[1])
    ax.set_xlabel("dim 0"); ax.set_ylabel("dim 1")
    ax.set_title("Part-X region tree (+ in-RoA, - out, r/min remaining)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
