#!/usr/bin/env python3
"""
Qualitative 2D CartPole Heatmap: ROA estimation on 2D slices of 4D state space.

Since CartPole has a 4D state (x, θ, ẋ, θ̇), we fix 2 dimensions and sweep
the other 2 on a grid. Supported slices:
  - theta_thetadot: (θ, θ̇) at x=0, ẋ=0  [comparable to pendulum]
  - x_xdot:         (x, ẋ) at θ=0, θ̇=0   [translational ROA]

Outputs:
  - Bare single images per (method, slice): no axis, no title, just the heatmap
  - Combined 2×2 horizontal figure (methods × slices) with --all_slices

Usage:
    python scripts/plot_qualitative_cartpole.py --all_slices --device cuda:0
    python scripts/plot_qualitative_cartpole.py --slice theta_thetadot --device cuda:0
"""

import argparse
import json
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
    "adaptive_cartpole_pybullet/outputs"
)
RUNS = {
    "Adaptive (d2=1.0)": _BASE / (
        "training_index_0_d2_ratio_1.0_warm_start_False_manifold_False"
        "_threshold_mode_dynamic_adapt_iter_20_alpha_0.1_sampling_mode_direct_w_0.9"
    ) / "2026-02-18_14-26-32",
    "Non-adaptive": _BASE / (
        "training_index_0_d2_ratio_0.0_warm_start_False_manifold_False"
        "_threshold_mode_dynamic_adapt_iter_20_alpha_0.1_sampling_mode_direct_w_0.9"
    ) / "2026-02-26_15-40-14",
}
DEFAULT_EPOCHS = [15]

# ── Ground truth CSVs ────────────────────────────────────────────────────────
_GT_DIR = Path(
    "/common/users/shared/pracsys/genMoPlan/data_trajectories/deterministic/cartpole_pybullet"
)

# ── Slice definitions ────────────────────────────────────────────────────────
# State order: [x, θ, ẋ, θ̇] = indices [0, 1, 2, 3]
SLICES = {
    "theta_thetadot": {
        "sweep_indices": (1, 3),           # θ, θ̇
        "sweep_ranges": ((-np.pi, np.pi), (-8.5, 8.5)),
        "fixed_values": {0: 0.0, 2: 0.0}, # x=0, ẋ=0
        "xlabel": r"$\theta$ (rad)",
        "ylabel": r"$\dot{\theta}$ (rad/s)",
        "xticks": [-np.pi, -np.pi/2, 0, np.pi/2, np.pi],
        "xticklabels": [r"$-\pi$", r"$-\frac{\pi}{2}$", "0", r"$\frac{\pi}{2}$", r"$\pi$"],
        "yticks": [-2*np.pi, -np.pi, 0, np.pi, 2*np.pi],
        "yticklabels": [r"$-2\pi$", r"$-\pi$", "0", r"$\pi$", r"$2\pi$"],
        "xticks_compact": [-np.pi, 0, np.pi],
        "xticklabels_compact": [r"$-\pi$", "0", r"$\pi$"],
        "yticks_compact": [-2*np.pi, 0, 2*np.pi],
        "yticklabels_compact": [r"$-2\pi$", "0", r"$2\pi$"],
        "gt_csv": _GT_DIR / "viz_theta_vs_thetadot.csv",
        "gt_ax0_col": "theta",      # CSV column for x-axis
        "gt_ax1_col": "theta_dot",  # CSV column for y-axis
    },
    "x_xdot": {
        "sweep_indices": (0, 2),           # x, ẋ
        "sweep_ranges": ((-6.0, 6.0), (-7.0, 7.0)),
        "fixed_values": {1: 0.0, 3: 0.0}, # θ=0, θ̇=0
        "xlabel": r"$x$ (m)",
        "ylabel": r"$\dot{x}$ (m/s)",
        "xticks": [-6, -3, 0, 3, 6],
        "xticklabels": ["-6", "-3", "0", "3", "6"],
        "yticks": [-6, -3, 0, 3, 6],
        "yticklabels": ["-6", "-3", "0", "3", "6"],
        "xticks_compact": [-6, 0, 6],
        "xticklabels_compact": ["-6", "0", "6"],
        "yticks_compact": [-6, 0, 6],
        "yticklabels_compact": ["-6", "0", "6"],
        "gt_csv": _GT_DIR / "viz_x_vs_xdot.csv",
        "gt_ax0_col": "x",
        "gt_ax1_col": "x_dot",
    },
}

# ── Grid / MC parameters ────────────────────────────────────────────────────
NUM_MC_SAMPLES = 20
MC_BATCH_SIZE = 2048
ATTRACTOR_RADIUS = 0.2
DECISION_RULE = "one_sided"

OUTPUT_DIR = Path("results/figures")
FLAGSHIP_DIR = OUTPUT_DIR / "flagship"


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_checkpoint(run_dir: Path, epoch: int, device: str):
    """Load CartPole flow matcher from best checkpoint in epoch directory."""
    import glob as _glob
    from adaptive_roa.flow_matching.cartpole.latent_conditional.flow_matcher import (
        CartPoleLatentConditionalFlowMatcher,
    )

    epoch_dir = run_dir / f"epoch_{epoch:03d}"
    ckpt_dir = epoch_dir / "checkpoints"
    ckpts = _glob.glob(str(ckpt_dir / "best*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint in {ckpt_dir}")
    return CartPoleLatentConditionalFlowMatcher.load_from_checkpoint(ckpts[0], device=device)


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


def load_slice_data(slice_name: str):
    """Load the ground truth CSV and return grid + labels.

    The CSV has columns: x, theta, x_dot, theta_dot, label
    on a dense 400×400 grid for the given slice.

    Returns:
        grid_np: [N, 4] array of 4D states
        gt_labels: [N] array (1=success, 0=failure)
        axis0_vals: sorted unique values for swept dim 0
        axis1_vals: sorted unique values for swept dim 1
        n0, n1: grid dimensions
    """
    import pandas as pd
    sl = SLICES[slice_name]
    df = pd.read_csv(sl["gt_csv"])

    ax0_col = sl["gt_ax0_col"]
    ax1_col = sl["gt_ax1_col"]

    axis0 = np.sort(df[ax0_col].unique())
    axis1 = np.sort(df[ax1_col].unique())
    n0, n1 = len(axis0), len(axis1)

    # Build 4D grid in the same order as the sorted axes
    # Sort the dataframe by (ax0, ax1) so it matches meshgrid indexing="ij"
    df = df.sort_values([ax0_col, ax1_col]).reset_index(drop=True)

    grid_np = df[["x", "theta", "x_dot", "theta_dot"]].values.astype(np.float32)
    gt_labels = df["label"].values.astype(np.int8)

    print(f"  Loaded {sl['gt_csv'].name}: {n0}×{n1} = {len(df)} points")
    return grid_np, gt_labels, axis0, axis1, n0, n1


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

def cache_file(slice_name: str) -> Path:
    return OUTPUT_DIR / f"qualitative_cartpole_{slice_name}_cache.npz"


def compute_all_grids(device: str, slice_name: str, epochs: list[int], batch_size: int = MC_BATCH_SIZE):
    """Compute classification grids for all (method, epoch) combos.

    Uses the same 400×400 grid from the ground truth CSV.
    """
    grid_np, gt_labels, axis0, axis1, n0, n1 = load_slice_data(slice_name)
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

            del model
            torch.cuda.empty_cache()

    return results, gt_labels, axis0, axis1, n0, n1


def save_cache(results, gt_labels, axis0, axis1, n0, n1, slice_name: str):
    """Save computed grids to .npz for fast re-plotting."""
    save_dict = {
        "axis0": axis0, "axis1": axis1,
        "gt_labels": gt_labels,
        "n0": np.array(n0), "n1": np.array(n1),
    }
    for method_name in results:
        for epoch in results[method_name]:
            prefix = f"{method_name}_epoch{epoch}"
            r = results[method_name][epoch]
            save_dict[f"{prefix}_pred"] = r["pred_labels"]
            save_dict[f"{prefix}_psuc"] = r["p_success"]
            save_dict[f"{prefix}_lam"] = np.array(r["lambda_star"])
            save_dict[f"{prefix}_delta"] = np.array(r["delta_star"])
            save_dict[f"{prefix}_ntraj"] = np.array(r["n_traj"])
    cf = cache_file(slice_name)
    cf.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(cf), **save_dict)
    print(f"\nCache saved to {cf}")


def load_cache_file(slice_name: str):
    """Load cached grids from .npz."""
    data = np.load(str(cache_file(slice_name)), allow_pickle=False)
    axis0 = data["axis0"]
    axis1 = data["axis1"]
    gt_labels = data["gt_labels"]
    n0 = int(data["n0"])
    n1 = int(data["n1"])
    results = {}
    for method_name in RUNS:
        results[method_name] = {}
        for epoch in range(20):
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
    return results, gt_labels, axis0, axis1, n0, n1


# ── Plotting helpers ──────────────────────────────────────────────────────────

def _discrete_cmap():
    # TP (success) / Uncertain / TN (failure) — matches error-analysis palette
    colors = ["#440154", "#D3D3D3", "#FDE725"]
    cmap = mcolors.ListedColormap(colors)
    bounds = [-1.5, -0.5, 0.5, 1.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    return cmap, norm


def _prob_cmap():
    return mcolors.LinearSegmentedColormap.from_list(
        "roa_prob", ["#440154", "#31688E", "#21918C", "#5EC962", "#FDE725"]
    )


def _pred_to_heatmap(pred, n0, n1):
    heatmap = np.zeros_like(pred, dtype=float)
    heatmap[pred == 1] = 1     # success
    heatmap[pred == 0] = -1    # failure
    return heatmap.reshape(n0, n1).T


def _gt_to_heatmap(gt_labels, n0, n1):
    """Convert GT labels (1=success, 0=failure) to heatmap values (+1/-1)."""
    heatmap = np.where(gt_labels == 1, 1.0, -1.0)
    return heatmap.reshape(n0, n1).T


def _style_ax(ax, sl):
    ax.set_xticks(sl["xticks"])
    ax.set_xticklabels(sl["xticklabels"], fontsize=12, fontweight="bold")
    ax.set_yticks(sl["yticks"])
    ax.set_yticklabels(sl["yticklabels"], fontsize=12, fontweight="bold")
    ax.tick_params(labelsize=12)


def _safe_method_name(method_name: str) -> str:
    return method_name.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "")


# ── Bare single images (no axis, no title, nothing — just the heatmap) ───────

def _save_bare(heatmap_2d, extent, cmap, name, norm=None, vmin=None, vmax=None, interp="nearest"):
    """Save a single bare image — no axis, no title, nothing."""
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(
        heatmap_2d, origin="lower", aspect="auto",
        cmap=cmap, norm=norm, vmin=vmin, vmax=vmax,
        extent=extent, interpolation=interp,
    )
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FLAGSHIP_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(OUTPUT_DIR / f"{name}.pdf"), dpi=300, bbox_inches="tight", pad_inches=0)
    fig.savefig(str(FLAGSHIP_DIR / f"{name}.png"), dpi=300, bbox_inches="tight", pad_inches=0)
    print(f"  Saved bare: {name}")
    plt.close(fig)


def plot_bare_singles(results, gt_labels, n0, n1, slice_name: str, epochs: list[int] = None):
    """Save bare images for GT + each (method, epoch) — discrete and probability."""
    sl = SLICES[slice_name]
    cmap_d, norm_d = _discrete_cmap()
    cmap_p = _prob_cmap()
    extent = [*sl["sweep_ranges"][0], *sl["sweep_ranges"][1]]
    method_names = list(RUNS.keys())
    if epochs is None:
        epochs = sorted(results[method_names[0]].keys())

    # Ground truth
    gt_heatmap = _gt_to_heatmap(gt_labels, n0, n1)
    _save_bare(gt_heatmap, extent, cmap_d, f"cartpole_{slice_name}_ground_truth", norm=norm_d)

    # Model predictions
    for method_name in method_names:
        safe = _safe_method_name(method_name)
        for epoch in epochs:
            r = results[method_name][epoch]

            # Discrete classification
            hm = _pred_to_heatmap(r["pred_labels"], n0, n1)
            _save_bare(hm, extent, cmap_d, f"cartpole_{slice_name}_{safe}_ep{epoch}", norm=norm_d)

            # Probability
            p_suc = r["p_success"]
            if hasattr(p_suc, "cpu"):
                p_suc = p_suc.cpu().numpy()
            p_2d = p_suc.reshape(n0, n1).T
            _save_bare(p_2d, extent, cmap_p, f"cartpole_{slice_name}_prob_{safe}_ep{epoch}",
                        vmin=0, vmax=1, interp="bilinear")


# ── Combined horizontal figure: rows=slices, cols=(GT, methods) ──────────

def plot_combined(all_results: dict, epoch: int):
    """Create 2×3 combined figure: rows=slices, cols=(GT, Adaptive, Non-adaptive)."""
    slice_names = list(all_results.keys())
    method_names = list(RUNS.keys())
    col_labels = ["Ground Truth"] + method_names
    n_rows = len(slice_names)
    n_cols = len(col_labels)

    cmap, norm = _discrete_cmap()

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.5 * n_cols, 3.0 * n_rows),
        squeeze=False,
        constrained_layout=True,
    )

    for row, slice_name in enumerate(slice_names):
        sl = SLICES[slice_name]
        results, gt_labels, _, _, n0, n1 = all_results[slice_name]
        extent = [*sl["sweep_ranges"][0], *sl["sweep_ranges"][1]]

        # Col 0: Ground truth
        ax = axes[row, 0]
        gt_hm = _gt_to_heatmap(gt_labels, n0, n1)
        ax.imshow(gt_hm, origin="lower", aspect="auto",
                  cmap=cmap, norm=norm, extent=extent, interpolation="nearest")
        ax.set_xlabel(sl["xlabel"], fontsize=13, fontweight="bold")
        ax.set_ylabel(sl["ylabel"], fontsize=13, fontweight="bold")
        _style_ax(ax, sl)

        # Cols 1+: model predictions
        for col_offset, method_name in enumerate(method_names):
            ax = axes[row, col_offset + 1]
            r = results[method_name][epoch]
            hm = _pred_to_heatmap(r["pred_labels"], n0, n1)
            ax.imshow(hm, origin="lower", aspect="auto",
                      cmap=cmap, norm=norm, extent=extent, interpolation="nearest")
            ax.set_xlabel(sl["xlabel"], fontsize=13, fontweight="bold")
            ax.set_ylabel(sl["ylabel"], fontsize=13, fontweight="bold")
            _style_ax(ax, sl)

    # Column titles
    for col, label in enumerate(col_labels):
        axes[0, col].set_title(label, fontsize=14, fontweight="bold")

    # Color legend for success / uncertain / failure
    from matplotlib.patches import Patch
    legend_patches = [
        Patch(facecolor="#FDE725", label="Success"),
        Patch(facecolor="#D3D3D3", label="Uncertain"),
        Patch(facecolor="#440154", label="Failure"),
    ]
    fig.legend(
        handles=legend_patches, loc="lower center",
        ncol=3, fontsize=12, frameon=False, prop={"weight": "bold"},
        bbox_to_anchor=(0.5, -0.04),
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FLAGSHIP_DIR.mkdir(parents=True, exist_ok=True)
    for ext, d in [("pdf", OUTPUT_DIR), ("png", FLAGSHIP_DIR)]:
        out = str(d / f"qualitative_cartpole_combined_ep{epoch}.{ext}")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Saved {out}")
    plt.close(fig)


def plot_prob_combined(all_results: dict, epoch: int):
    """Create 2×2 combined probability figure: rows=slices, cols=methods."""
    slice_names = list(all_results.keys())
    method_names = list(RUNS.keys())
    n_rows = len(slice_names)
    n_cols = len(method_names)

    cmap = _prob_cmap()

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.8 * n_cols, 3.0 * n_rows),
        squeeze=False,
        constrained_layout=True,
    )

    im = None
    for row, slice_name in enumerate(slice_names):
        sl = SLICES[slice_name]
        results, _, _, _, n0, n1 = all_results[slice_name]
        extent = [*sl["sweep_ranges"][0], *sl["sweep_ranges"][1]]

        for col, method_name in enumerate(method_names):
            ax = axes[row, col]
            r = results[method_name][epoch]
            p_suc = r["p_success"]
            if hasattr(p_suc, "cpu"):
                p_suc = p_suc.cpu().numpy()
            p_2d = p_suc.reshape(n0, n1).T

            im = ax.imshow(
                p_2d, origin="lower", aspect="auto",
                cmap=cmap, vmin=0, vmax=1,
                extent=extent, interpolation="bilinear",
            )

            ax.set_xlabel(sl["xlabel"], fontsize=13, fontweight="bold")
            ax.set_ylabel(sl["ylabel"], fontsize=13, fontweight="bold")
            _style_ax(ax, sl)

    # Column titles
    for col, label in enumerate(method_names):
        axes[0, col].set_title(label, fontsize=14, fontweight="bold")

    cbar = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
    cbar.set_label(r"$p(\mathrm{success} \mid x)$", fontsize=13, fontweight="bold")
    cbar.ax.tick_params(labelsize=11)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FLAGSHIP_DIR.mkdir(parents=True, exist_ok=True)
    for ext, d in [("pdf", OUTPUT_DIR), ("png", FLAGSHIP_DIR)]:
        out = str(d / f"qualitative_cartpole_prob_combined_ep{epoch}.{ext}")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Saved {out}")
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Qualitative CartPole ROA slice heatmap")
    parser.add_argument("--device", default="cuda:0", help="Device for inference")
    parser.add_argument("--batch_size", type=int, default=MC_BATCH_SIZE)
    parser.add_argument("--recompute", action="store_true")
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
        "--epochs", type=int, nargs="+", default=DEFAULT_EPOCHS,
        help="Epochs to compute/plot (default: [15])",
    )
    parser.add_argument("--no_probabilities", action="store_true")
    args = parser.parse_args()

    slice_names = list(SLICES.keys()) if args.all_slices else [args.slice]
    epochs = args.epochs

    all_results = {}

    for slice_name in slice_names:
        print(f"\n{'#'*70}")
        print(f"# Slice: {slice_name}")
        print(f"{'#'*70}")

        cf = cache_file(slice_name)

        if cf.exists() and not args.recompute:
            print(f"Loading from cache: {cf}")
            results, gt_labels, axis0, axis1, n0, n1 = load_cache_file(slice_name)
            first_method = list(RUNS.keys())[0]
            missing = [e for e in epochs if e not in results.get(first_method, {})]
            if missing:
                print(f"Cache missing epochs {missing}, computing...")
                new_results, gt_labels, axis0, axis1, n0, n1 = compute_all_grids(
                    args.device, slice_name, missing, args.batch_size)
                for m in new_results:
                    results.setdefault(m, {}).update(new_results[m])
                save_cache(results, gt_labels, axis0, axis1, n0, n1, slice_name)
        else:
            results, gt_labels, axis0, axis1, n0, n1 = compute_all_grids(
                args.device, slice_name, epochs, args.batch_size)
            save_cache(results, gt_labels, axis0, axis1, n0, n1, slice_name)

        all_results[slice_name] = (results, gt_labels, axis0, axis1, n0, n1)

        # Bare single images (no axis, no title, nothing) — GT + model predictions
        plot_bare_singles(results, gt_labels, n0, n1, slice_name, epochs)

    # Combined horizontal figure (if multiple slices)
    if len(slice_names) > 1:
        for epoch in epochs:
            plot_combined(all_results, epoch)
            if not args.no_probabilities:
                plot_prob_combined(all_results, epoch)


if __name__ == "__main__":
    main()
