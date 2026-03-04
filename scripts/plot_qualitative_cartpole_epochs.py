#!/usr/bin/env python3
"""
Qualitative 2D CartPole Heatmap: ROA estimation progression across epochs.

Pendulum-style multi-epoch progression figures for CartPole 2D slices.
Rows = methods, Columns = epochs. Each panel is a classification or
probability heatmap over a 2D slice of the 4D CartPole state space.

Supported slices:
  - theta_thetadot: (θ, θ̇) at x=0, ẋ=0  [comparable to pendulum]
  - x_xdot:         (x, ẋ) at θ=0, θ̇=0   [translational ROA]

Usage:
    # Plot from existing cache (no GPU needed if cache exists)
    python scripts/plot_qualitative_cartpole_epochs.py --slice theta_thetadot --plot_epochs 0 5 10 15

    # Both slices
    python scripts/plot_qualitative_cartpole_epochs.py --all_slices --plot_epochs 0 5 10 15

    # With fixed threshold
    python scripts/plot_qualitative_cartpole_epochs.py --slice theta_thetadot --fixed_lambda 0.5 --fixed_delta 0.05

    # Full recompute
    python scripts/plot_qualitative_cartpole_epochs.py --all_slices --device cuda:0 --recompute
"""

import argparse
import sys
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

# Import shared CartPole functions and constants
sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_qualitative_cartpole import (
    RUNS,
    SLICES,
    MC_BATCH_SIZE,
    OUTPUT_DIR,
    cache_file,
    compute_all_grids,
    load_cache_file,
    save_cache,
)

ALL_EPOCHS = list(range(20))  # 0..19
DEFAULT_PLOT_EPOCHS = [0, 4, 8, 12]


# ── Reclassification ─────────────────────────────────────────────────────────

def reclassify_with_fixed_threshold(results, fixed_lambda, fixed_delta):
    """Reclassify all grid points using a fixed (λ, δ) from cached p_success."""
    success_thresh = fixed_lambda + fixed_delta
    failure_thresh = fixed_lambda - fixed_delta
    for method_name in results:
        for epoch in results[method_name]:
            r = results[method_name][epoch]
            p_suc = r["p_success"]
            pred = np.full(len(p_suc), -1, dtype=np.int8)  # uncertain
            pred[p_suc > success_thresh] = 1                 # success
            pred[p_suc < failure_thresh] = 0                  # failure
            r["pred_labels"] = pred
            r["lambda_star"] = fixed_lambda
            r["delta_star"] = fixed_delta
    return results


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_figure(results, n0, n1, slice_name, epochs=None,
                fixed_lambda=None, fixed_delta=None):
    """Create rows=methods, cols=epochs classification heatmap figure for a slice."""
    sl = SLICES[slice_name]
    method_names = list(RUNS.keys())
    n_rows = len(method_names)
    if epochs is None:
        epochs = sorted(results[method_names[0]].keys())
    n_cols = len(epochs)

    # Discrete colormap: failure(red) → separatrix(yellow) → success(green)
    colors = ["#d73027", "#fee08b", "#1a9850"]
    cmap = mcolors.ListedColormap(colors)
    bounds = [-1.5, -0.5, 0.5, 1.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    extent = [*sl["sweep_ranges"][0], *sl["sweep_ranges"][1]]

    col_width = 1.8
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(col_width * n_cols, 2.0 * n_rows),
        sharex=True, sharey=True,
        constrained_layout=True,
    )
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    if n_cols == 1:
        axes = axes.reshape(-1, 1)

    for row, method_name in enumerate(method_names):
        for col, epoch in enumerate(epochs):
            ax = axes[row, col]
            r = results[method_name][epoch]
            pred = r["pred_labels"]
            n_traj = r["n_traj"]

            # Map labels to heatmap values: 1→+1 (success), 0→-1 (failure),
            # -1 (uncertain) and -2 (invalid) → 0 (separatrix)
            heatmap = np.zeros_like(pred, dtype=float)
            heatmap[pred == 1] = 1     # success
            heatmap[pred == 0] = -1    # failure

            heatmap_2d = heatmap.reshape(n0, n1).T

            ax.imshow(
                heatmap_2d,
                origin="lower",
                aspect="auto",
                cmap=cmap,
                norm=norm,
                extent=extent,
                interpolation="nearest",
            )

            # Column titles on top row only
            if row == 0:
                if fixed_lambda is not None:
                    ax.set_title(f"{n_traj} traj.", fontsize=9)
                else:
                    ax.set_title(f"{n_traj} trajectories", fontsize=9)

            # Row label on first column
            if col == 0:
                ax.set_ylabel(
                    f"{method_name}\n{sl['ylabel']}",
                    fontsize=8,
                )

            # Ticks from slice definition
            ax.set_xticks(sl["xticks"])
            ax.set_xticklabels(sl["xticklabels"], fontsize=7)
            ax.set_yticks(sl["yticks"])
            ax.set_yticklabels(sl["yticklabels"], fontsize=7)
            ax.tick_params(labelsize=7)

    # Single shared x-axis label
    fig.supxlabel(sl["xlabel"], fontsize=9)

    if fixed_lambda is not None:
        fig.suptitle(
            f"Fixed $\\lambda^*={fixed_lambda:.2f}$, $\\delta={fixed_delta:.2f}$"
            f"  (success: $p > {fixed_lambda + fixed_delta:.2f}$,"
            f" failure: $p < {fixed_lambda - fixed_delta:.2f}$)",
            fontsize=9, y=1.02,
        )

    # Legend below figure
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="#1a9850", edgecolor="none", label="Success"),
        Patch(facecolor="#d73027", edgecolor="none", label="Failure"),
        Patch(facecolor="#fee08b", edgecolor="none", label="Separatrix"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=3,
        fontsize=8,
        frameon=False,
        bbox_to_anchor=(0.5, -0.06),
    )

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    base_name = f"qualitative_cartpole_{slice_name}_epochs"
    if fixed_lambda is not None:
        base_name += f"_lambda{fixed_lambda:.2f}_delta{fixed_delta:.2f}"
    for ext in ["pdf", "png"]:
        out = str(OUTPUT_DIR / f"{base_name}.{ext}")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Saved {out}")
    plt.close(fig)


def plot_probabilities(results, n0, n1, slice_name, epochs=None):
    """Create rows=methods, cols=epochs probability heatmap figure for a slice."""
    sl = SLICES[slice_name]
    method_names = list(RUNS.keys())
    n_rows = len(method_names)
    if epochs is None:
        epochs = sorted(results[method_names[0]].keys())
    n_cols = len(epochs)

    # Diverging colormap: red (0) → white (0.5) → green (1)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "roa_prob", ["#d73027", "#fee08b", "#ffffbf", "#a6d96a", "#1a9850"]
    )

    extent = [*sl["sweep_ranges"][0], *sl["sweep_ranges"][1]]

    col_width = 1.8
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(col_width * n_cols + 0.6, 2.0 * n_rows),
        sharex=True, sharey=True,
        constrained_layout=True,
    )
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    if n_cols == 1:
        axes = axes.reshape(-1, 1)

    im = None
    for row, method_name in enumerate(method_names):
        for col, epoch in enumerate(epochs):
            ax = axes[row, col]
            r = results[method_name][epoch]
            p_suc = r["p_success"]
            n_traj = r["n_traj"]

            if hasattr(p_suc, "cpu"):
                p_suc = p_suc.cpu().numpy()
            p_2d = p_suc.reshape(n0, n1).T

            im = ax.imshow(
                p_2d,
                origin="lower",
                aspect="auto",
                cmap=cmap,
                vmin=0, vmax=1,
                extent=extent,
                interpolation="bilinear",
            )

            if row == 0:
                ax.set_title(f"{n_traj} trajectories", fontsize=9)

            if col == 0:
                ax.set_ylabel(
                    f"{method_name}\n{sl['ylabel']}",
                    fontsize=8,
                )

            # Ticks from slice definition
            ax.set_xticks(sl["xticks"])
            ax.set_xticklabels(sl["xticklabels"], fontsize=7)
            ax.set_yticks(sl["yticks"])
            ax.set_yticklabels(sl["yticklabels"], fontsize=7)
            ax.tick_params(labelsize=7)

    fig.supxlabel(sl["xlabel"], fontsize=9)

    # Colorbar
    cbar = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
    cbar.set_label(r"$p(\mathrm{success} \mid x)$", fontsize=9)
    cbar.ax.tick_params(labelsize=7)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    base_name = f"qualitative_cartpole_{slice_name}_epochs_probabilities"
    for ext in ["pdf", "png"]:
        out = str(OUTPUT_DIR / f"{base_name}.{ext}")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Saved {out}")
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Qualitative CartPole ROA epoch progression heatmaps"
    )
    parser.add_argument("--device", default="cuda:0", help="Device for inference")
    parser.add_argument("--batch_size", type=int, default=MC_BATCH_SIZE)
    parser.add_argument(
        "--recompute", action="store_true",
        help="Force recomputation even if cache exists",
    )
    parser.add_argument(
        "--slice", default="theta_thetadot",
        choices=list(SLICES.keys()),
        help="Which 2D slice (default: theta_thetadot)",
    )
    parser.add_argument(
        "--all_slices", action="store_true",
        help="Run both slices (theta_thetadot + x_xdot)",
    )
    parser.add_argument(
        "--epochs", type=int, nargs="+", default=None,
        help="Epochs to compute (default: all 0-19). "
             "E.g. --epochs 0 2 4 6 8 10 12 14 16 18",
    )
    parser.add_argument(
        "--plot_epochs", type=int, nargs="+", default=None,
        help="Epochs to plot (subset of computed). Uses cache if available. "
             "E.g. --plot_epochs 0 4 8 12",
    )
    parser.add_argument(
        "--fixed_lambda", type=float, default=None,
        help="Use a fixed λ* for all panels (reclassifies from cached p_success)",
    )
    parser.add_argument(
        "--fixed_delta", type=float, default=None,
        help="Use a fixed δ for all panels (default: 0.05)",
    )
    parser.add_argument(
        "--no_probabilities", action="store_true",
        help="Skip continuous p(success) probability heatmap",
    )
    args = parser.parse_args()

    slice_names = list(SLICES.keys()) if args.all_slices else [args.slice]
    compute_epochs = args.epochs or ALL_EPOCHS

    for slice_name in slice_names:
        print(f"\n{'#'*70}")
        print(f"# Slice: {slice_name}")
        print(f"{'#'*70}")

        cf = cache_file(slice_name)

        if cf.exists() and not args.recompute:
            print(f"Loading from cache: {cf}")
            results, gt_labels, axis0, axis1, n0, n1 = load_cache_file(slice_name)
            first_method = list(RUNS.keys())[0]
            missing = [e for e in compute_epochs if e not in results.get(first_method, {})]
            if missing:
                print(f"Cache missing epochs {missing}, computing...")
                new_results, gt_labels, axis0, axis1, n0, n1 = compute_all_grids(
                    args.device, slice_name, missing, args.batch_size)
                for m in new_results:
                    results.setdefault(m, {}).update(new_results[m])
                save_cache(results, gt_labels, axis0, axis1, n0, n1, slice_name)
        else:
            results, gt_labels, axis0, axis1, n0, n1 = compute_all_grids(
                args.device, slice_name, compute_epochs, args.batch_size)
            save_cache(results, gt_labels, axis0, axis1, n0, n1, slice_name)

        # Apply fixed threshold if requested
        fixed_lambda = args.fixed_lambda
        fixed_delta = args.fixed_delta if args.fixed_delta is not None else 0.05
        if fixed_lambda is not None:
            print(f"Reclassifying with fixed λ*={fixed_lambda}, δ={fixed_delta}")
            results = reclassify_with_fixed_threshold(results, fixed_lambda, fixed_delta)

        plot_epochs = args.plot_epochs or compute_epochs
        plot_figure(
            results, n0, n1, slice_name, plot_epochs,
            fixed_lambda=fixed_lambda,
            fixed_delta=fixed_delta if fixed_lambda is not None else None,
        )

        if not args.no_probabilities:
            plot_probabilities(results, n0, n1, slice_name, plot_epochs)


if __name__ == "__main__":
    main()
