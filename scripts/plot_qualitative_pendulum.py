#!/usr/bin/env python3
"""
Qualitative 2D Pendulum Heatmap: ROA estimation progression across epochs.

2×4 figure: Adaptive (top) vs Non-adaptive (bottom). Each panel is a
classification heatmap over (θ, θ̇) with success (green), failure (red),
and separatrix (yellow) regions.

Usage:
    python scripts/plot_qualitative_pendulum.py [--device cuda:0] [--recompute]
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch

from adaptive_roa.conformal.config import ConformalConfig
from adaptive_roa.conformal.probability_estimator import ProbabilityEstimator
from adaptive_roa.adaptive_v2.eval.full_roa import _predict_lambda_delta

# ── Paths ────────────────────────────────────────────────────────────────────
_BASE = Path(
    "/common/users/shared/pracsys/adaptive_roa_experiments/dhruv/"
    "adaptive_pendulum_dhruv/outputs"
)
RUNS = {
    "Adaptive (manifold, d2=0.5)": _BASE / (
        "training_index_0_d2_ratio_0.5_warm_start_False_threshold_mode_dynamic"
        "_adapt_iter_20_alpha_0.1_sampling_mode_direct"
    ) / "2026-02-26_15-22-40",
    "Adaptive (d2=0.5)": _BASE / (
        "training_index_0_d2_ratio_0.5_warm_start_False_threshold_mode_dynamic"
        "_adapt_iter_20_alpha_0.1_sampling_mode_direct"
    ) / "2026-02-25_12-11-15",
    "Adaptive (d2=0.75)": _BASE / (
        "training_index_0_d2_ratio_0.75_warm_start_False_threshold_mode_dynamic"
        "_adapt_iter_20_alpha_0.1_sampling_mode_direct"
    ) / "2026-02-25_12-11-22",
    "Adaptive (d2=1.0)": _BASE / (
        "training_index_0_d2_ratio_1.0_warm_start_False_threshold_mode_dynamic"
        "_adapt_iter_20_alpha_0.1_sampling_mode_direct"
    ) / "2026-02-20_01-56-07",
    "Non-adaptive": _BASE / (
        "training_index_0_d2_ratio_0.0_warm_start_False_threshold_mode_dynamic"
        "_adapt_iter_20_alpha_0.1_sampling_mode_ranked"
    ) / "2026-02-20_01-56-20",
}
ALL_EPOCHS = list(range(20))  # 0..19
DEFAULT_EPOCHS = [0, 2, 4, 6]

# ── Grid / MC parameters ────────────────────────────────────────────────────
GRID_RES = 200
THETA_RANGE = (-np.pi, np.pi)
THETA_DOT_RANGE = (-2 * np.pi, 2 * np.pi)
NUM_MC_SAMPLES = 20
MC_BATCH_SIZE = 2048
ATTRACTOR_RADIUS = 0.075
DECISION_RULE = "one_sided"

OUTPUT_DIR = Path("results/figures")
CACHE_FILE = OUTPUT_DIR / "qualitative_pendulum_manifold_cache.npz"


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_checkpoint(run_dir: Path, epoch: int, device: str):
    """Load flow matcher from best checkpoint in epoch directory."""
    import glob as _glob
    from adaptive_roa.flow_matching.pendulum.latent_conditional.flow_matcher import (
        PendulumLatentConditionalFlowMatcher,
    )

    epoch_dir = run_dir / f"epoch_{epoch:03d}"
    ckpt_dir = epoch_dir / "checkpoints"
    ckpts = _glob.glob(str(ckpt_dir / "best*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint in {ckpt_dir}")
    return PendulumLatentConditionalFlowMatcher.load_from_checkpoint(ckpts[0], device=device)


def load_thresholds(run_dir: Path, epoch: int):
    """Return (lambda_star, delta_star) from artifacts_v2.json."""
    arts = run_dir / f"epoch_{epoch:03d}" / "artifacts_v2.json"
    with open(arts) as f:
        d = json.load(f)
    ts = d["threshold_state"]
    return float(ts["lambda_star"]), float(ts["delta_star"])


def load_train_trajectories_count(run_dir: Path, epoch: int) -> int:
    """Return number of training trajectories at this epoch."""
    arts = run_dir / f"epoch_{epoch:03d}" / "artifacts_v2.json"
    with open(arts) as f:
        d = json.load(f)
    return int(d["train_trajectories"])


def make_grid():
    """Create dense evaluation grid. Returns (grid_tensor [N,2], theta, theta_dot)."""
    theta = np.linspace(*THETA_RANGE, GRID_RES)
    theta_dot = np.linspace(*THETA_DOT_RANGE, GRID_RES)
    TH, THD = np.meshgrid(theta, theta_dot, indexing="ij")
    grid = np.stack([TH.ravel(), THD.ravel()], axis=1)  # [N, 2]
    return grid, theta, theta_dot


def classify_grid(model, system, grid_np, lambda_star, delta_star, device, batch_size=MC_BATCH_SIZE):
    """Run MC estimation and classify grid points. Returns pred_labels [N]."""
    config = ConformalConfig(
        delta=delta_star,
        num_mc_samples=NUM_MC_SAMPLES,
        mc_batch_size=batch_size,
        attractor_radius=ATTRACTOR_RADIUS,
    )
    estimator = ProbabilityEstimator(model, system, config, device)
    grid_tensor = torch.from_numpy(grid_np).float().to(device)
    p_success, p_failure, p_invalid = estimator.estimate(grid_tensor, verbose=True)

    pred_labels, _ = _predict_lambda_delta(
        p_success, p_failure, p_invalid,
        lambda_star=lambda_star,
        delta=delta_star,
        decision_rule=DECISION_RULE,
        invalid_threshold=None,
    )
    return pred_labels, p_success, p_failure, p_invalid


# ── Compute or load cache ────────────────────────────────────────────────────

def compute_all_grids(device: str, epochs: list[int] = None, batch_size: int = MC_BATCH_SIZE):
    """Compute classification grids for all (method, epoch) combos."""
    if epochs is None:
        epochs = ALL_EPOCHS
    grid_np, theta, theta_dot = make_grid()
    results = {}

    for method_name, run_dir in RUNS.items():
        results[method_name] = {}
        for epoch in epochs:
            print(f"\n{'='*60}")
            print(f"{method_name} — Epoch {epoch}")
            print(f"{'='*60}")

            model = load_checkpoint(run_dir, epoch, device)
            lam, delta = load_thresholds(run_dir, epoch)
            print(f"  λ*={lam:.4f}, δ*={delta:.4f}")

            pred_labels, p_success, p_failure, p_invalid = classify_grid(
                model, model.system, grid_np, lam, delta, device, batch_size
            )
            n_traj = load_train_trajectories_count(run_dir, epoch)

            results[method_name][epoch] = {
                "pred_labels": pred_labels,
                "p_success": p_success,
                "lambda_star": lam,
                "delta_star": delta,
                "n_traj": n_traj,
            }

            # Free GPU memory
            del model
            torch.cuda.empty_cache()

    return results, theta, theta_dot


def save_cache(results, theta, theta_dot):
    """Save computed grids to .npz for fast re-plotting."""
    save_dict = {"theta": theta, "theta_dot": theta_dot}
    for method_name in results:
        for epoch in results[method_name]:
            prefix = f"{method_name}_epoch{epoch}"
            r = results[method_name][epoch]
            save_dict[f"{prefix}_pred"] = r["pred_labels"]
            save_dict[f"{prefix}_psuc"] = r["p_success"]
            save_dict[f"{prefix}_lam"] = np.array(r["lambda_star"])
            save_dict[f"{prefix}_delta"] = np.array(r["delta_star"])
            save_dict[f"{prefix}_ntraj"] = np.array(r["n_traj"])
    np.savez(str(CACHE_FILE), **save_dict)
    print(f"\nCache saved to {CACHE_FILE}")


def load_cache():
    """Load cached grids from .npz. Returns all epochs found in cache."""
    data = np.load(str(CACHE_FILE), allow_pickle=False)
    theta = data["theta"]
    theta_dot = data["theta_dot"]
    results = {}
    for method_name in RUNS:
        results[method_name] = {}
        for epoch in ALL_EPOCHS:
            prefix = f"{method_name}_epoch{epoch}"
            key = f"{prefix}_pred"
            if key not in data:
                continue
            results[method_name][epoch] = {
                "pred_labels": data[f"{prefix}_pred"],
                "p_success": data[f"{prefix}_psuc"],
                "lambda_star": float(data[f"{prefix}_lam"]),
                "delta_star": float(data[f"{prefix}_delta"]),
                "n_traj": int(data[f"{prefix}_ntraj"]),
            }
    return results, theta, theta_dot


# ── Plotting ─────────────────────────────────────────────────────────────────

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


def plot_figure(results, theta, theta_dot, epochs: list[int] = None,
                fixed_lambda: float = None, fixed_delta: float = None):
    """Create 2×N publication-quality heatmap figure."""
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

    col_width = 1.8
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(col_width * n_cols, 2.0 * n_rows),
        sharex=True, sharey=True,
        constrained_layout=True,
    )
    if n_cols == 1:
        axes = axes.reshape(n_rows, 1)

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

            heatmap_2d = heatmap.reshape(GRID_RES, GRID_RES).T

            ax.imshow(
                heatmap_2d,
                origin="lower",
                aspect="auto",
                cmap=cmap,
                norm=norm,
                extent=[*THETA_RANGE, *THETA_DOT_RANGE],
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
                    f"{method_name}\n" + r"$\dot{\theta}$ (rad/s)",
                    fontsize=8,
                )
            # π-based ticks
            ax.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
            ax.set_xticklabels(
                [r"$-\pi$", r"$-\frac{\pi}{2}$", "0", r"$\frac{\pi}{2}$", r"$\pi$"],
                fontsize=7,
            )
            ax.set_yticks([-2 * np.pi, -np.pi, 0, np.pi, 2 * np.pi])
            ax.set_yticklabels(
                [r"$-2\pi$", r"$-\pi$", "0", r"$\pi$", r"$2\pi$"],
                fontsize=7,
            )
            ax.tick_params(labelsize=7)

    # Single shared x-axis label
    fig.supxlabel(r"$\theta$ (rad)", fontsize=9)

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
    base_name = "qualitative_pendulum"
    if fixed_lambda is not None:
        base_name += f"_lambda{fixed_lambda:.2f}_delta{fixed_delta:.2f}"
    for ext in ["pdf", "png"]:
        out = str(OUTPUT_DIR / f"{base_name}.{ext}")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Saved {out}")
    plt.close(fig)


def plot_probabilities(results, theta, theta_dot, epochs: list[int] = None):
    """Create N×M probability heatmap figure showing continuous p(success)."""
    method_names = list(RUNS.keys())
    n_rows = len(method_names)
    if epochs is None:
        epochs = sorted(results[method_names[0]].keys())
    n_cols = len(epochs)

    # Diverging colormap: red (0) → white (0.5) → green (1)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "roa_prob", ["#d73027", "#fee08b", "#ffffbf", "#a6d96a", "#1a9850"]
    )

    col_width = 1.8
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(col_width * n_cols + 0.6, 2.0 * n_rows),
        sharex=True, sharey=True,
        constrained_layout=True,
    )
    if n_cols == 1:
        axes = axes.reshape(n_rows, 1)

    im = None
    for row, method_name in enumerate(method_names):
        for col, epoch in enumerate(epochs):
            ax = axes[row, col]
            r = results[method_name][epoch]
            p_suc = r["p_success"]
            n_traj = r["n_traj"]

            if hasattr(p_suc, "cpu"):
                p_suc = p_suc.cpu().numpy()
            p_2d = p_suc.reshape(GRID_RES, GRID_RES).T

            im = ax.imshow(
                p_2d,
                origin="lower",
                aspect="auto",
                cmap=cmap,
                vmin=0, vmax=1,
                extent=[*THETA_RANGE, *THETA_DOT_RANGE],
                interpolation="bilinear",
            )

            if row == 0:
                ax.set_title(f"{n_traj} trajectories", fontsize=9)

            if col == 0:
                ax.set_ylabel(
                    f"{method_name}\n" + r"$\dot{\theta}$ (rad/s)",
                    fontsize=8,
                )

            ax.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
            ax.set_xticklabels(
                [r"$-\pi$", r"$-\frac{\pi}{2}$", "0", r"$\frac{\pi}{2}$", r"$\pi$"],
                fontsize=7,
            )
            ax.set_yticks([-2 * np.pi, -np.pi, 0, np.pi, 2 * np.pi])
            ax.set_yticklabels(
                [r"$-2\pi$", r"$-\pi$", "0", r"$\pi$", r"$2\pi$"],
                fontsize=7,
            )
            ax.tick_params(labelsize=7)

    fig.supxlabel(r"$\theta$ (rad)", fontsize=9)

    # Colorbar
    cbar = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
    cbar.set_label(r"$p(\mathrm{success} \mid x)$", fontsize=9)
    cbar.ax.tick_params(labelsize=7)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ["pdf", "png"]:
        out = str(OUTPUT_DIR / f"qualitative_pendulum_probabilities.{ext}")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Saved {out}")
    plt.close(fig)


# ── Baseline plot ────────────────────────────────────────────────────────────

BASELINE_CSV = Path(
    "/common/home/dm1487/robotics_research/tripods/olympics-classifier/"
    "lyapunov_pend_roa_predictions.csv"
)


def plot_baseline():
    """Generate a standalone heatmap from the Lyapunov NN baseline CSV."""
    import pandas as pd

    df = pd.read_csv(BASELINE_CSV)
    thetas = np.sort(df["theta"].unique())
    theta_dots = np.sort(df["theta_dot"].unique())
    n_th, n_td = len(thetas), len(theta_dots)

    # Build 2D grid from scattered CSV rows
    heatmap = np.full((n_th, n_td), np.nan)
    th_idx = {v: i for i, v in enumerate(thetas)}
    td_idx = {v: i for i, v in enumerate(theta_dots)}
    for _, row in df.iterrows():
        i = th_idx[row["theta"]]
        j = td_idx[row["theta_dot"]]
        heatmap[i, j] = row["predicted"]  # 1=success, 0=failure, -1=separatrix

    # Remap CSV labels to our colormap: -1→red(failure), 0→yellow(separatrix), 1→green(success)
    # CSV: 0=failure → -1, -1=separatrix → 0
    remapped = np.full_like(heatmap, np.nan)
    remapped[heatmap == 1] = 1     # success → success
    remapped[heatmap == 0] = -1    # failure → red
    remapped[heatmap == -1] = 0    # separatrix → yellow
    heatmap = remapped

    colors = ["#d73027", "#fee08b", "#1a9850"]
    cmap = mcolors.ListedColormap(colors)
    bounds = [-1.5, -0.5, 0.5, 1.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(4, 3), constrained_layout=True)
    ax.imshow(
        heatmap.T,
        origin="lower",
        aspect="auto",
        cmap=cmap,
        norm=norm,
        extent=[thetas[0], thetas[-1], theta_dots[0], theta_dots[-1]],
        interpolation="nearest",
    )

    ax.set_title("Lyapunov NN Baseline (1,000 traj.)", fontsize=10)
    ax.set_xlabel(r"$\theta$ (rad)", fontsize=9)
    ax.set_ylabel(r"$\dot{\theta}$ (rad/s)", fontsize=9)

    ax.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    ax.set_xticklabels(
        [r"$-\pi$", r"$-\frac{\pi}{2}$", "0", r"$\frac{\pi}{2}$", r"$\pi$"],
        fontsize=7,
    )
    ax.set_yticks([-2 * np.pi, -np.pi, 0, np.pi, 2 * np.pi])
    ax.set_yticklabels(
        [r"$-2\pi$", r"$-\pi$", "0", r"$\pi$", r"$2\pi$"],
        fontsize=7,
    )

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
        bbox_to_anchor=(0.5, -0.08),
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ["pdf", "png"]:
        out = str(OUTPUT_DIR / f"qualitative_pendulum_lyapunov_baseline.{ext}")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Saved {out}")
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Qualitative pendulum ROA heatmap")
    parser.add_argument("--device", default="cuda:0", help="Device for inference")
    parser.add_argument("--batch_size", type=int, default=MC_BATCH_SIZE, help="MC batch size for inference")
    parser.add_argument(
        "--recompute", action="store_true",
        help="Force recomputation even if cache exists",
    )
    parser.add_argument(
        "--epochs", type=int, nargs="+", default=None,
        help="Epochs to compute and plot (default: all 0-19). "
             "E.g. --epochs 0 2 4 6",
    )
    parser.add_argument(
        "--plot_epochs", type=int, nargs="+", default=None,
        help="Epochs to plot (subset of computed). Uses cache if available. "
             "E.g. --plot_epochs 0 4 8 12 16",
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
        "--no_baseline", action="store_true",
        help="Skip standalone Lyapunov NN baseline heatmap",
    )
    parser.add_argument(
        "--no_probabilities", action="store_true",
        help="Skip continuous p(success) probability heatmap",
    )
    args = parser.parse_args()

    if not args.no_baseline:
        plot_baseline()

    compute_epochs = args.epochs or ALL_EPOCHS

    if CACHE_FILE.exists() and not args.recompute:
        print(f"Loading from cache: {CACHE_FILE}")
        results, theta, theta_dot = load_cache()
        # Check if requested epochs are missing from cache
        first_method = list(RUNS.keys())[0]
        missing = [e for e in compute_epochs if e not in results.get(first_method, {})]
        if missing:
            print(f"Cache missing epochs {missing}, computing...")
            new_results, theta, theta_dot = compute_all_grids(args.device, missing, args.batch_size)
            for m in new_results:
                results.setdefault(m, {}).update(new_results[m])
            save_cache(results, theta, theta_dot)
    else:
        results, theta, theta_dot = compute_all_grids(args.device, compute_epochs, args.batch_size)
        save_cache(results, theta, theta_dot)

    # Apply fixed threshold if requested
    fixed_lambda = args.fixed_lambda
    fixed_delta = args.fixed_delta if args.fixed_delta is not None else 0.05
    if fixed_lambda is not None:
        print(f"Reclassifying with fixed λ*={fixed_lambda}, δ={fixed_delta}")
        results = reclassify_with_fixed_threshold(results, fixed_lambda, fixed_delta)

    plot_epochs = args.plot_epochs or compute_epochs
    plot_figure(results, theta, theta_dot, plot_epochs,
                fixed_lambda=fixed_lambda,
                fixed_delta=fixed_delta if fixed_lambda is not None else None)

    if not args.no_probabilities:
        plot_probabilities(results, theta, theta_dot, plot_epochs)


if __name__ == "__main__":
    main()
