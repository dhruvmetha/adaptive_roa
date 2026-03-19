#!/usr/bin/env python3
"""
Quadrotor 3D ROA video: per-method, per-slice videos + static ground truth images.

State: (x, y, z, qw, qx, qy, qz, x_dot, y_dot, z_dot, p, q, r) — 13D
CSVs also include Euler angles (phi, theta_euler, psi) for sweep axes.

Available slices (6):
  theta_vs_q    — pitch angle vs pitch rate
  phi_vs_p      — roll angle vs roll rate
  psi_vs_r      — yaw angle vs yaw rate
  x_vs_xdot     — x position vs x velocity
  y_vs_ydot     — y position vs y velocity
  z_vs_zdot     — z position vs z velocity

Default slices: theta_vs_q, z_vs_zdot

Usage:
    python scripts/plot_qualitative_quadrotor3d_video.py --gt_only
    python scripts/plot_qualitative_quadrotor3d_video.py --cache_only --device cuda:0 --methods adaptive
    python scripts/plot_qualitative_quadrotor3d_video.py --device cuda:0
    python scripts/plot_qualitative_quadrotor3d_video.py --device cpu --slices phi_vs_p psi_vs_r
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import glob as _glob
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.patches import Patch

from adaptive_roa.conformal.config import ConformalConfig
from adaptive_roa.conformal.probability_estimator import ProbabilityEstimator
from adaptive_roa.adaptive_v2.eval.full_roa import _predict_lambda_delta

# ── Run definitions ──────────────────────────────────────────────────────────
_BASE = Path(
    "/common/users/shared/pracsys/adaptive_roa_experiments/dhruv/"
    "adaptive_quadrotor3d/outputs"
)
_EVAL_SUBDIR = "radius_0.3_alpha_0.1_mc_10_batch_100000_calk_1000"

RUNS = {
    "adaptive": {
        "label": "Adaptive",
        "dir": _BASE / (
            "training_index_0_d2_ratio_1.0_warm_start_False_manifold_False"
            "_threshold_mode_dynamic_adapt_iter_20_alpha_0.1"
            "_sampling_mode_direct"
        ) / "2026-02-26_12-02-36",
        "epochs": list(range(16)),  # 0-15 (eval artifacts available)
    },
    "nonadaptive": {
        "label": "Non-adaptive",
        "dir": _BASE / (
            "training_index_0_d2_ratio_0.0_warm_start_False_manifold_False"
            "_threshold_mode_dynamic_adapt_iter_15_alpha_0.1"
            "_sampling_mode_direct"
        ) / "2026-02-28_11-49-54",
        "epochs": list(range(15)),  # 0-14 (eval artifacts available)
    },
}

# Common epoch range for combined videos; per-method videos use RUNS[key]["epochs"]
ALL_EPOCHS = list(range(16))  # 0-15 → 10k-25k trajectories
DEFAULT_SLICES = ["theta_vs_q", "phi_vs_p", "psi_vs_r", "x_vs_xdot", "y_vs_ydot", "z_vs_zdot"]
OUTPUT_DIR = Path("results/videos_and_images/quadrotor3d")

# ── Ground truth CSVs ────────────────────────────────────────────────────────
_GT_DIR = Path(
    "/common/users/shared/pracsys/genMoPlan/data_trajectories/quadrotor3D_lqr"
)

# CSV columns: x, y, z, qw, qx, qy, qz, x_dot, y_dot, z_dot, p, q, r,
#              phi, theta_euler, psi, label
_STATE_COLS = ["x", "y", "z", "qw", "qx", "qy", "qz",
               "x_dot", "y_dot", "z_dot", "p", "q", "r"]

# ── Slice definitions ────────────────────────────────────────────────────────
SLICES = {
    "theta_vs_q": {
        "gt_csv": _GT_DIR / "viz_theta_vs_q.csv",
        "gt_ax0_col": "theta_euler",
        "gt_ax1_col": "q",
        "sweep_ranges": ((-np.pi, np.pi), (-39.2, 39.2)),
        "xlabel": r"$\theta$ (rad)",
        "ylabel": r"$q$ (rad/s)",
        "xticks": [-np.pi, -np.pi/2, 0, np.pi/2, np.pi],
        "xticklabels": [r"$-\pi$", r"$-\frac{\pi}{2}$", "0", r"$\frac{\pi}{2}$", r"$\pi$"],
        "yticks": [-30, -15, 0, 15, 30],
        "yticklabels": ["-30", "-15", "0", "15", "30"],
        "xticks_compact": [-np.pi, 0, np.pi],
        "xticklabels_compact": [r"$-\pi$", "0", r"$\pi$"],
        "yticks_compact": [-30, 0, 30],
        "yticklabels_compact": ["-30", "0", "30"],
    },
    "phi_vs_p": {
        "gt_csv": _GT_DIR / "viz_phi_vs_p.csv",
        "gt_ax0_col": "phi",
        "gt_ax1_col": "p",
        "sweep_ranges": ((-np.pi, np.pi), (-39.2, 39.2)),
        "xlabel": r"$\phi$ (rad)",
        "ylabel": r"$p$ (rad/s)",
        "xticks": [-np.pi, -np.pi/2, 0, np.pi/2, np.pi],
        "xticklabels": [r"$-\pi$", r"$-\frac{\pi}{2}$", "0", r"$\frac{\pi}{2}$", r"$\pi$"],
        "yticks": [-30, -15, 0, 15, 30],
        "yticklabels": ["-30", "-15", "0", "15", "30"],
        "xticks_compact": [-np.pi, 0, np.pi],
        "xticklabels_compact": [r"$-\pi$", "0", r"$\pi$"],
        "yticks_compact": [-30, 0, 30],
        "yticklabels_compact": ["-30", "0", "30"],
    },
    "psi_vs_r": {
        "gt_csv": _GT_DIR / "viz_psi_vs_r.csv",
        "gt_ax0_col": "psi",
        "gt_ax1_col": "r",
        "sweep_ranges": ((-np.pi, np.pi), (-39.7, 39.7)),
        "xlabel": r"$\psi$ (rad)",
        "ylabel": r"$r$ (rad/s)",
        "xticks": [-np.pi, -np.pi/2, 0, np.pi/2, np.pi],
        "xticklabels": [r"$-\pi$", r"$-\frac{\pi}{2}$", "0", r"$\frac{\pi}{2}$", r"$\pi$"],
        "yticks": [-30, -15, 0, 15, 30],
        "yticklabels": ["-30", "-15", "0", "15", "30"],
        "xticks_compact": [-np.pi, 0, np.pi],
        "xticklabels_compact": [r"$-\pi$", "0", r"$\pi$"],
        "yticks_compact": [-30, 0, 30],
        "yticklabels_compact": ["-30", "0", "30"],
    },
    "x_vs_xdot": {
        "gt_csv": _GT_DIR / "viz_x_vs_xdot.csv",
        "gt_ax0_col": "x",
        "gt_ax1_col": "x_dot",
        "sweep_ranges": ((-1.85, 1.85), (-3.25, 3.25)),
        "xlabel": r"$x$ (m)",
        "ylabel": r"$\dot{x}$ (m/s)",
        "xticks": [-1.5, -0.75, 0, 0.75, 1.5],
        "xticklabels": ["-1.5", "-0.75", "0", "0.75", "1.5"],
        "yticks": [-3, -1.5, 0, 1.5, 3],
        "yticklabels": ["-3", "-1.5", "0", "1.5", "3"],
        "xticks_compact": [-1.5, 0, 1.5],
        "xticklabels_compact": ["-1.5", "0", "1.5"],
        "yticks_compact": [-3, 0, 3],
        "yticklabels_compact": ["-3", "0", "3"],
    },
    "y_vs_ydot": {
        "gt_csv": _GT_DIR / "viz_y_vs_ydot.csv",
        "gt_ax0_col": "y",
        "gt_ax1_col": "y_dot",
        "sweep_ranges": ((-1.85, 1.85), (-3.25, 3.25)),
        "xlabel": r"$y$ (m)",
        "ylabel": r"$\dot{y}$ (m/s)",
        "xticks": [-1.5, -0.75, 0, 0.75, 1.5],
        "xticklabels": ["-1.5", "-0.75", "0", "0.75", "1.5"],
        "yticks": [-3, -1.5, 0, 1.5, 3],
        "yticklabels": ["-3", "-1.5", "0", "1.5", "3"],
        "xticks_compact": [-1.5, 0, 1.5],
        "xticklabels_compact": ["-1.5", "0", "1.5"],
        "yticks_compact": [-3, 0, 3],
        "yticklabels_compact": ["-3", "0", "3"],
    },
    "z_vs_zdot": {
        "gt_csv": _GT_DIR / "viz_z_vs_zdot.csv",
        "gt_ax0_col": "z",
        "gt_ax1_col": "z_dot",
        "sweep_ranges": ((0.06, 3.05), (-3.35, 3.05)),
        "xlabel": r"$z$ (m)",
        "ylabel": r"$\dot{z}$ (m/s)",
        "xticks": [0.5, 1.0, 1.5, 2.0, 2.5],
        "xticklabels": ["0.5", "1.0", "1.5", "2.0", "2.5"],
        "yticks": [-3, -1.5, 0, 1.5, 3],
        "yticklabels": ["-3", "-1.5", "0", "1.5", "3"],
        "xticks_compact": [0.5, 1.5, 2.5],
        "xticklabels_compact": ["0.5", "1.5", "2.5"],
        "yticks_compact": [-3, 0, 3],
        "yticklabels_compact": ["-3", "0", "3"],
    },
}
ALL_SLICE_NAMES = list(SLICES.keys())

# ── Grid / MC parameters ────────────────────────────────────────────────────
NUM_MC_SAMPLES = 10
MC_BATCH_SIZE = 2048
ATTRACTOR_RADIUS = 0.3
DECISION_RULE = "one_sided"


# ── Plotting helpers ─────────────────────────────────────────────────────────

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
    heatmap[pred == 1] = 1
    heatmap[pred == 0] = -1
    return heatmap.reshape(n0, n1).T


def _gt_to_heatmap(gt_labels, n0, n1):
    heatmap = np.where(gt_labels == 1, 1.0, -1.0)
    return heatmap.reshape(n0, n1).T


# ── Data loading helpers ────────────────────────────────────────────────────

def load_checkpoint(run_dir: Path, epoch: int, device: str):
    from adaptive_roa.flow_matching.quadrotor_3d.latent_conditional.flow_matcher import (
        Quadrotor3DLatentConditionalFlowMatcher,
    )
    epoch_dir = run_dir / f"epoch_{epoch:03d}"
    ckpt_dir = epoch_dir / "checkpoints"
    ckpts = _glob.glob(str(ckpt_dir / "best*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint in {ckpt_dir}")
    return Quadrotor3DLatentConditionalFlowMatcher.load_from_checkpoint(ckpts[0], device=device)


def load_eval_artifacts(run_dir, epoch):
    """Load lambda_star, delta_star, F1, and unc% from evaluation artifacts."""
    arts = run_dir / "evaluations" / _EVAL_SUBDIR / f"epoch_{epoch:03d}" / "artifacts_v2.json"
    with open(arts) as f:
        d = json.load(f)
    ts = d["threshold_state"]
    ld = d["eval_metrics"]["lambda_delta"]
    return {
        "lambda_star": float(ts["lambda_star"]),
        "delta_star": float(ts["delta_star"]),
        "f1": float(ld["f1"]),
        "unc_pct": float(ld["separatrix_pct"]) * 100.0,
    }


def load_train_trajectories_count(run_dir: Path, epoch: int) -> int:
    arts = run_dir / f"epoch_{epoch:03d}" / "artifacts_v2.json"
    with open(arts) as f:
        d = json.load(f)
    return int(d["train_trajectories"])


def load_slice_data(slice_name: str):
    """Load GT CSV. Returns grid as 13D state (quaternion, not Euler).

    The CSV has Euler angles for the sweep axes but we feed the
    quaternion columns (x,y,z,qw,qx,qy,qz,x_dot,y_dot,z_dot,p,q,r)
    to the model.
    """
    sl = SLICES[slice_name]
    df = pd.read_csv(sl["gt_csv"])

    ax0_col = sl["gt_ax0_col"]
    ax1_col = sl["gt_ax1_col"]

    axis0 = np.sort(df[ax0_col].unique())
    axis1 = np.sort(df[ax1_col].unique())
    n0, n1 = len(axis0), len(axis1)

    df = df.sort_values([ax0_col, ax1_col]).reset_index(drop=True)

    grid_np = df[_STATE_COLS].values.astype(np.float32)
    gt_labels = df["label"].values.astype(np.int8)

    print(f"  Loaded {sl['gt_csv'].name}: {n0}x{n1} = {len(df)} points")
    return grid_np, gt_labels, axis0, axis1, n0, n1


def classify_grid(model, system, grid_np, lambda_star, delta_star, device,
                  batch_size=MC_BATCH_SIZE):
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

# ── Per-method computation ───────────────────────────────────────────────────

def compute_single_method(method_key, slice_name, epochs, device, batch_size,
                          existing_results=None):
    """Load model per epoch, classify grid, return results dict.

    Uses lambda/delta from evaluation artifacts. Saves cache incrementally.
    """
    run = RUNS[method_key]
    run_dir = run["dir"]
    grid_np, gt_labels, axis0, axis1, n0, n1 = load_slice_data(slice_name)
    results = dict(existing_results) if existing_results else {}

    for epoch in epochs:
        print(f"\n  {run['label']} | {slice_name} | epoch {epoch}")
        eval_arts = load_eval_artifacts(run_dir, epoch)
        lam = eval_arts["lambda_star"]
        delta = eval_arts["delta_star"]
        print(f"    lambda*={lam:.4f}, delta*={delta:.4f}")
        print(f"    eval F1={eval_arts['f1']:.4f}, eval unc%={eval_arts['unc_pct']:.1f}%")

        model = load_checkpoint(run_dir, epoch, device)
        pred_labels, p_success, _, _ = classify_grid(
            model, model.system, grid_np, lam, delta, device, batch_size
        )
        n_traj = load_train_trajectories_count(run_dir, epoch)

        results[epoch] = {
            "pred_labels": pred_labels,
            "p_success": p_success,
            "lambda_star": lam,
            "delta_star": delta,
            "n_traj": n_traj,
            "eval_f1": eval_arts["f1"],
            "eval_unc_pct": eval_arts["unc_pct"],
        }

        del model
        torch.cuda.empty_cache()

        # Save incrementally after each epoch
        save_cache(results, gt_labels, axis0, axis1, n0, n1, method_key, slice_name)

    return results, gt_labels, axis0, axis1, n0, n1


# ── Caching ──────────────────────────────────────────────────────────────────

def _cache_path(method_key, slice_name):
    return OUTPUT_DIR / f"cache_{method_key}_{slice_name}.npz"


def save_cache(results, gt_labels, axis0, axis1, n0, n1, method_key, slice_name):
    save_dict = {
        "axis0": axis0, "axis1": axis1,
        "gt_labels": gt_labels,
        "n0": np.array(n0), "n1": np.array(n1),
    }
    for epoch, r in results.items():
        prefix = f"epoch{epoch}"
        save_dict[f"{prefix}_pred"] = r["pred_labels"]
        save_dict[f"{prefix}_psuc"] = r["p_success"]
        save_dict[f"{prefix}_lam"] = np.array(r["lambda_star"])
        save_dict[f"{prefix}_delta"] = np.array(r["delta_star"])
        save_dict[f"{prefix}_ntraj"] = np.array(r["n_traj"])
        save_dict[f"{prefix}_eval_f1"] = np.array(r["eval_f1"])
        save_dict[f"{prefix}_eval_unc"] = np.array(r["eval_unc_pct"])
    cf = _cache_path(method_key, slice_name)
    cf.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(cf), **save_dict)
    print(f"  Cache saved: {cf}")


def load_cache(method_key, slice_name):
    data = np.load(str(_cache_path(method_key, slice_name)), allow_pickle=False)
    axis0 = data["axis0"]
    axis1 = data["axis1"]
    gt_labels = data["gt_labels"]
    n0 = int(data["n0"])
    n1 = int(data["n1"])
    run_dir = RUNS[method_key]["dir"]
    results = {}
    needs_resave = False
    for epoch in range(max(ALL_EPOCHS) + 5):
        key = f"epoch{epoch}_pred"
        if key not in data:
            continue
        r = {
            "pred_labels": data[f"epoch{epoch}_pred"],
            "p_success": data[f"epoch{epoch}_psuc"],
            "lambda_star": float(data[f"epoch{epoch}_lam"]),
            "delta_star": float(data[f"epoch{epoch}_delta"]),
            "n_traj": int(data[f"epoch{epoch}_ntraj"]),
        }
        if f"epoch{epoch}_eval_f1" in data:
            r["eval_f1"] = float(data[f"epoch{epoch}_eval_f1"])
            r["eval_unc_pct"] = float(data[f"epoch{epoch}_eval_unc"])
        else:
            eval_arts = load_eval_artifacts(run_dir, epoch)
            r["eval_f1"] = eval_arts["f1"]
            r["eval_unc_pct"] = eval_arts["unc_pct"]
            needs_resave = True
        results[epoch] = r
    if needs_resave:
        save_cache(results, gt_labels, axis0, axis1, n0, n1, method_key, slice_name)
    return results, gt_labels, axis0, axis1, n0, n1


def load_or_compute(method_key, slice_name, epochs, device, batch_size, recompute):
    cf = _cache_path(method_key, slice_name)
    if cf.exists() and not recompute:
        print(f"Loading cache: {cf}")
        results, gt_labels, axis0, axis1, n0, n1 = load_cache(method_key, slice_name)
        missing = [e for e in epochs if e not in results]
        if missing:
            print(f"  Cache missing epochs {missing}, computing...")
            results, gt_labels, axis0, axis1, n0, n1 = compute_single_method(
                method_key, slice_name, missing, device, batch_size,
                existing_results=results,
            )
    else:
        results, gt_labels, axis0, axis1, n0, n1 = compute_single_method(
            method_key, slice_name, epochs, device, batch_size
        )
    return results, gt_labels, axis0, axis1, n0, n1


# ── Ground truth image ───────────────────────────────────────────────────────

def render_gt_image(slice_name):
    grid_np, gt_labels, axis0, axis1, n0, n1 = load_slice_data(slice_name)
    sl = SLICES[slice_name]
    cmap_d, norm_d = _discrete_cmap()
    extent = [*sl["sweep_ranges"][0], *sl["sweep_ranges"][1]]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(
        _gt_to_heatmap(gt_labels, n0, n1),
        origin="lower", aspect="auto",
        cmap=cmap_d, norm=norm_d, extent=extent, interpolation="nearest",
    )
    ax.set_xlabel(sl["xlabel"], fontsize=12, fontweight="bold")
    ax.set_ylabel(sl["ylabel"], fontsize=12, fontweight="bold")
    ax.set_xticks(sl["xticks"])
    ax.set_xticklabels(sl["xticklabels"], fontsize=10)
    ax.set_yticks(sl["yticks"])
    ax.set_yticklabels(sl["yticklabels"], fontsize=10)
    ax.set_title("Ground Truth", fontsize=14, fontweight="bold")

    legend_elements = [
        Patch(facecolor="#FDE725", edgecolor="none", label="Success"),
        Patch(facecolor="#440154", edgecolor="none", label="Failure"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=10, framealpha=0.9)

    fig.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        out = OUTPUT_DIR / f"gt_{slice_name}.{ext}"
        fig.savefig(str(out), dpi=300, bbox_inches="tight")
        print(f"  Saved: {out}")
    plt.close(fig)


# ── Video frame rendering ───────────────────────────────────────────────────

def _style_axes(ax, sl, fontsize=9, fontweight="normal"):
    ax.set_xticks(sl["xticks"])
    ax.set_xticklabels(sl["xticklabels"], fontsize=fontsize, fontweight=fontweight)
    ax.set_yticks(sl["yticks"])
    ax.set_yticklabels(sl["yticklabels"], fontsize=fontsize, fontweight=fontweight)
    ax.tick_params(labelsize=fontsize)


def render_video_frame(epoch_idx, epochs, method_label, slice_name,
                       results, n0, n1, metrics, frame_path,
                       unc_ylim, mode="classification"):
    """Render a single frame for one method and one slice.

    mode: "classification" or "probability"
    Layout: top row = heatmap, bottom row = unc% | F1 line plots.
    """
    epoch = epochs[epoch_idx]
    sl = SLICES[slice_name]
    r = results[epoch]
    extent = [*sl["sweep_ranges"][0], *sl["sweep_ranges"][1]]

    fig = plt.figure(figsize=(16, 9))
    right_margin = 0.88 if mode == "probability" else 0.93
    gs = fig.add_gridspec(
        2, 2,
        height_ratios=[1.2, 0.8],
        hspace=0.40, wspace=0.35,
        left=0.07, right=right_margin, top=0.88, bottom=0.10,
    )

    fig.suptitle(
        f"{method_label}\n"
        f"Epoch {epoch}  |  {r['n_traj']} traj  |  "
        rf"$\lambda^*$={r['lambda_star']:.3f}  $\delta$={r['delta_star']:.3f}",
        fontsize=14, fontweight="bold",
    )

    # ── Top: heatmap spanning both columns ──
    ax_map = fig.add_subplot(gs[0, :])
    if mode == "classification":
        cmap_d, norm_d = _discrete_cmap()
        ax_map.imshow(
            _pred_to_heatmap(r["pred_labels"], n0, n1),
            origin="lower", aspect="auto",
            cmap=cmap_d, norm=norm_d, extent=extent, interpolation="nearest",
        )
        ax_map.set_title("Classification", fontsize=12, fontweight="bold")
        legend_elements = [
            Patch(facecolor="#FDE725", edgecolor="none", label="Success"),
            Patch(facecolor="#D3D3D3", edgecolor="none", label="Uncertain"),
            Patch(facecolor="#440154", edgecolor="none", label="Failure"),
        ]
        ax_map.legend(
            handles=legend_elements, loc="lower center", ncol=3,
            fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, -0.12),
        )
    else:
        cmap_p = _prob_cmap()
        p_suc = r["p_success"]
        if hasattr(p_suc, "cpu"):
            p_suc = p_suc.cpu().numpy()
        im = ax_map.imshow(
            p_suc.reshape(n0, n1).T,
            origin="lower", aspect="auto",
            cmap=cmap_p, vmin=0, vmax=1, extent=extent, interpolation="bilinear",
        )
        ax_map.set_title(r"$p(\mathrm{success} \mid x)$", fontsize=12, fontweight="bold")
        cbar_ax = fig.add_axes([0.90, 0.52, 0.015, 0.36])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label(r"$p(\mathrm{success})$", fontsize=9)
        cbar.ax.tick_params(labelsize=8)

    _style_axes(ax_map, sl)
    ax_map.set_xlabel(sl["xlabel"], fontsize=10)
    ax_map.set_ylabel(sl["ylabel"], fontsize=10)

    # ── Bottom: line plots ──
    ax_unc = fig.add_subplot(gs[1, 0])
    unc_vals = [metrics[e]["unc_pct"] for e in epochs]
    ax_unc.plot(epochs, unc_vals, "o-", color="#2166ac", ms=4, lw=1.5)
    ax_unc.plot(epoch, metrics[epoch]["unc_pct"], "o",
                color="#2166ac", ms=12, mec="black", mew=1.5, zorder=5)
    ax_unc.axvline(epoch, color="gray", ls="--", alpha=0.5, lw=1)
    ax_unc.set_xlabel("Epoch", fontsize=10)
    ax_unc.set_ylabel("Sep %", fontsize=10)
    ax_unc.set_title("Separatrix % vs Epoch", fontsize=11, fontweight="bold")
    ax_unc.set_xlim(-0.5, max(epochs) + 0.5)
    ax_unc.set_ylim(0, unc_ylim)
    ax_unc.grid(True, alpha=0.3)
    ax_unc.tick_params(labelsize=9)

    ax_f1 = fig.add_subplot(gs[1, 1])
    f1_vals = [metrics[e]["f1"] for e in epochs]
    ax_f1.plot(epochs, f1_vals, "o-", color="#b2182b", ms=4, lw=1.5)
    ax_f1.plot(epoch, metrics[epoch]["f1"], "o",
               color="#b2182b", ms=12, mec="black", mew=1.5, zorder=5)
    ax_f1.axvline(epoch, color="gray", ls="--", alpha=0.5, lw=1)
    ax_f1.set_xlabel("Epoch", fontsize=10)
    ax_f1.set_ylabel("F1", fontsize=10)
    ax_f1.set_title("F1 Score vs Epoch", fontsize=11, fontweight="bold")
    ax_f1.set_xlim(-0.5, max(epochs) + 0.5)
    ax_f1.set_ylim(0, 1.05)
    ax_f1.grid(True, alpha=0.3)
    ax_f1.tick_params(labelsize=9)

    fig.savefig(str(frame_path), dpi=120)
    plt.close(fig)


# ── Video assembly ───────────────────────────────────────────────────────────

def _stitch_video(frame_dir, out_path, fps):
    subprocess.run(
        ["ffmpeg", "-y", "-framerate", str(fps),
         "-i", str(Path(frame_dir) / "frame_%04d.png"),
         "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
         str(out_path)],
        check=True,
    )


def make_video(method_key, slice_name, epochs, results, gt_labels, n0, n1,
               metrics, fps):
    """Render classification and probability videos for one (method, slice)."""
    method_label = RUNS[method_key]["label"]
    max_unc = max(metrics[e]["unc_pct"] for e in epochs)
    unc_ylim = max_unc * 1.1 + 1
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for mode in ("classification", "probability"):
        out_path = OUTPUT_DIR / f"{method_key}_{slice_name}_{mode}.mp4"
        frame_dir = OUTPUT_DIR / "frames" / method_key / slice_name / mode
        frame_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nRendering {len(epochs)} frames: {method_key} / {slice_name} / {mode}")
        with tempfile.TemporaryDirectory() as tmpdir:
            for i, e in enumerate(epochs):
                fp = Path(tmpdir) / f"frame_{i:04d}.png"
                render_video_frame(
                    i, epochs, method_label, slice_name,
                    results, n0, n1, metrics, fp, unc_ylim, mode=mode,
                )
                # Save permanent copy
                import shutil
                shutil.copy2(str(fp), str(frame_dir / f"epoch_{e:03d}.png"))
                print(f"  Frame {i+1}/{len(epochs)} (epoch {e})")
            _stitch_video(tmpdir, out_path, fps)
        print(f"  Video saved: {out_path}")
        print(f"  Frames saved: {frame_dir}")


# ── Combined video ───────────────────────────────────────────────────────────

METHOD_COLORS = {"adaptive": "#2166ac", "nonadaptive": "#b2182b"}
SLICE_LABELS = {
    "theta_vs_q": r"$(\theta,\,q)$",
    "phi_vs_p": r"$(\phi,\,p)$",
    "psi_vs_r": r"$(\psi,\,r)$",
    "x_vs_xdot": r"$(x,\,\dot{x})$",
    "y_vs_ydot": r"$(y,\,\dot{y})$",
    "z_vs_zdot": r"$(z,\,\dot{z})$",
}


def render_combined_frame(epoch_idx, epochs, slice_names, all_data,
                          all_metrics, gt_slices, frame_path, unc_ylim,
                          f1_ylim=None, mode="classification"):
    """Render combined frame: rows=(GT, Non-adapt, Adaptive), cols=slices.

    Flagship styling: tight spacing, compact ticks, tuned fonts.
    mode: "classification" or "probability"
    """
    epoch = epochs[epoch_idx]
    cmap_d, norm_d = _discrete_cmap()
    cmap_p = _prob_cmap()
    method_keys = ["nonadaptive", "adaptive"]
    n_slices = len(slice_names)
    n_rows = 1 + len(method_keys)
    gc = n_slices * 2
    last_heatmap_row = n_rows - 1

    # Fixed resolution: 2560x1440 @ 120 dpi
    fig_w = 2560 / 120.0
    fig_h = 1440 / 120.0
    cell_w = fig_w / n_slices

    # Chrome fonts (title, legend, line plots) — fixed across all scripts
    chrome_title_fs = 28   # system name
    chrome_subtitle_fs = 22  # trajectories count
    legend_fs = 18
    line_title_fs = 16
    line_label_fs = 14
    line_tick_fs = 12

    # Heatmap fonts — scale down for many columns (6 slices → smaller cells)
    title_fs = 22
    label_fs = 20
    tick_fs = 18
    method_fs = 20
    if n_slices > 3:
        s = min(1.0, cell_w / 5.5)
        title_fs = max(13, int(title_fs * s))
        label_fs = max(12, int(label_fs * s))
        tick_fs = max(11, int(tick_fs * s))
        method_fs = max(13, int(method_fs * s))

    right_margin = 0.92 if mode == "probability" else 0.99
    fig = plt.figure(figsize=(fig_w, fig_h))

    gs_top = fig.add_gridspec(
        n_rows, gc,
        hspace=0.40, wspace=0.65,
        left=0.06, right=right_margin, top=0.88, bottom=0.34,
    )
    gs_bot = fig.add_gridspec(
        1, 2,
        hspace=0, wspace=0.35,
        left=0.15, right=right_margin - 0.07, top=0.24, bottom=0.02,
    )

    def _strip(ax, sl, row, col):
        ax.set_xticks(sl["xticks_compact"])
        ax.set_xticklabels(sl["xticklabels_compact"], fontsize=tick_fs)
        ax.set_yticks(sl["yticks_compact"])
        ax.set_yticklabels(sl["yticklabels_compact"], fontsize=tick_fs)
        ax.tick_params(length=2, width=0.8, pad=1)
        ax.set_ylabel(sl["ylabel"], fontsize=label_fs, fontweight="bold", labelpad=1)
        ax.set_xlabel(sl["xlabel"], fontsize=label_fs, fontweight="bold", labelpad=1)

    all_axes = {}

    for ci, sn in enumerate(slice_names):
        sl = SLICES[sn]
        extent = [*sl["sweep_ranges"][0], *sl["sweep_ranges"][1]]
        c0, c1 = ci * 2, (ci + 1) * 2

        gt_labels_arr, gt_n0, gt_n1 = gt_slices[sn]
        ax = fig.add_subplot(gs_top[0, c0:c1])
        all_axes[(0, ci)] = ax
        ax.imshow(
            _gt_to_heatmap(gt_labels_arr, gt_n0, gt_n1),
            origin="lower", aspect="auto",
            cmap=cmap_d, norm=norm_d, extent=extent, interpolation="nearest",
        )
        _strip(ax, sl, 0, ci)
        ax.set_title(SLICE_LABELS[sn], fontsize=title_fs, fontweight="bold",
                     pad=4)

        for mi, mk in enumerate(method_keys):
            results, _, n0, n1 = all_data[mk][sn]
            r = results[epoch]
            row = mi + 1
            ax = fig.add_subplot(gs_top[row, c0:c1])
            all_axes[(row, ci)] = ax
            if mode == "classification":
                ax.imshow(
                    _pred_to_heatmap(r["pred_labels"], n0, n1),
                    origin="lower", aspect="auto",
                    cmap=cmap_d, norm=norm_d, extent=extent,
                    interpolation="nearest",
                )
            else:
                p_suc = r["p_success"]
                if hasattr(p_suc, "cpu"):
                    p_suc = p_suc.cpu().numpy()
                im = ax.imshow(
                    p_suc.reshape(n0, n1).T,
                    origin="lower", aspect="auto",
                    cmap=cmap_p, vmin=0, vmax=1, extent=extent,
                    interpolation="bilinear",
                )
                if mi == len(method_keys) - 1 and ci == n_slices - 1:
                    cbar_ax = fig.add_axes([right_margin + 0.015, 0.30, 0.012, 0.55])
                    cbar = fig.colorbar(im, cax=cbar_ax)
                    cbar.set_label(r"$p(\mathrm{success})$", fontsize=label_fs)
                    cbar.ax.tick_params(labelsize=tick_fs)
            _strip(ax, sl, row, ci)

    # Row labels — small, tucked left of first column
    row_labels = ["Ground Truth"] + [RUNS[mk]["label"] for mk in method_keys]
    for ri, label in enumerate(row_labels):
        all_axes[(ri, 0)].annotate(
            label, xy=(0, 0.5), xytext=(-0.22, 0.5),
            xycoords="axes fraction", textcoords="axes fraction",
            fontsize=method_fs + 4, fontweight="bold",
            ha="right", va="center", rotation=90,
            clip_on=False,
        )

    # Title — system name + trajectory count
    r = all_data[method_keys[0]][slice_names[0]][0][epoch]
    fig.text(0.5, 0.98, "Spatial Quadrotor", fontsize=chrome_title_fs,
             fontweight="bold", ha="center", va="top")
    fig.text(0.5, 0.935, f"Trajectories: {r['n_traj']}", fontsize=chrome_subtitle_fs,
             fontweight="bold", ha="center", va="top")

    # Legend in the gap between heatmaps and line plots
    if mode == "classification":
        legend_elements = [
            Patch(facecolor="#FDE725", edgecolor="0.4", linewidth=0.5,
                  label="Success"),
            Patch(facecolor="#D3D3D3", edgecolor="0.4", linewidth=0.5,
                  label="Uncertain"),
            Patch(facecolor="#440154", edgecolor="0.4", linewidth=0.5,
                  label="Failure"),
        ]
        fig.legend(
            handles=legend_elements, loc="lower center", ncol=3,
            fontsize=legend_fs, frameon=False,
            prop={"weight": "bold", "size": legend_fs},
            bbox_to_anchor=(0.53, 0.255),
            handletextpad=0.4, columnspacing=0.8,
            handleheight=1.2, handlelength=1.5,
        )

    # Line plots
    ax_unc = fig.add_subplot(gs_bot[0, 0])
    ax_f1 = fig.add_subplot(gs_bot[0, 1])

    sn0 = slice_names[0]
    # Collect per-method line data
    line_data = {}
    for mk in method_keys:
        line_data[mk] = {
            "traj": [all_data[mk][sn0][0][e]["n_traj"] for e in epochs],
            "unc": [all_metrics[mk][sn0][e]["unc_pct"] for e in epochs],
            "f1": [all_metrics[mk][sn0][e]["f1"] for e in epochs],
        }
    # Align first point across methods
    ref = method_keys[0]
    for mk in method_keys[1:]:
        line_data[mk]["traj"][0] = line_data[ref]["traj"][0]
        line_data[mk]["unc"][0] = line_data[ref]["unc"][0]
        line_data[mk]["f1"][0] = line_data[ref]["f1"][0]

    ei = epochs.index(epoch)
    for mk in method_keys:
        color = METHOD_COLORS[mk]
        label = RUNS[mk]["label"]
        traj_counts = line_data[mk]["traj"]
        unc_vals = line_data[mk]["unc"]
        f1_vals = line_data[mk]["f1"]

        ax_unc.plot(traj_counts, unc_vals, "o-", color=color, label=label,
                    ms=5, lw=2.5)
        ax_unc.plot(traj_counts[ei], unc_vals[ei], "o",
                    color=color, ms=11, mec="black", mew=1.5, zorder=5)

        ax_f1.plot(traj_counts, f1_vals, "o-", color=color, label=label,
                   ms=5, lw=2.5)
        ax_f1.plot(traj_counts[ei], f1_vals[ei], "o",
                   color=color, ms=11, mec="black", mew=1.5, zorder=5)

    all_traj = line_data[method_keys[0]]["traj"]
    cur_traj = all_traj[ei]
    for ax, title, ylabel in [(ax_unc, "Uncertain %", "Unc %"),
                               (ax_f1, "F1 Score", "F1")]:
        ax.axvline(cur_traj, color="gray", ls="--", alpha=0.5, lw=1)
        ax.set_xlabel("Trajectories", fontsize=line_label_fs, fontweight="bold", labelpad=2)
        ax.set_ylabel(ylabel, fontsize=line_label_fs + 4, fontweight="bold", labelpad=8)
        ax.set_title(title, fontsize=line_title_fs + 4, fontweight="bold")
        ax.legend(fontsize=line_label_fs, loc="best", prop={"weight": "bold"})
        ax.set_xlim(min(all_traj) - 500, max(all_traj) + 500)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=line_tick_fs, pad=2)

    ax_unc.set_ylim(0, unc_ylim)
    if f1_ylim is not None:
        ax_f1.set_ylim(f1_ylim)
    else:
        ax_f1.set_ylim(0, 1.05)

    fig.savefig(str(frame_path), dpi=120)
    plt.close(fig)


def make_combined_video(epochs, slice_names, all_data, all_metrics, fps):
    """Render classification and probability combined comparison videos."""
    all_f1 = []
    all_unc = []
    for mk in all_metrics:
        for sn in all_metrics[mk]:
            for e in epochs:
                all_unc.append(all_metrics[mk][sn][e]["unc_pct"])
                all_f1.append(all_metrics[mk][sn][e]["f1"])
    unc_ylim = max(all_unc) * 1.1 + 1
    # Zoom F1 axis: pad below min, cap at 1.02
    f1_min = min(all_f1)
    f1_lo = max(0, f1_min - 0.05 * (1.0 - f1_min) - 0.02)
    f1_lo = round(f1_lo, 2)
    f1_ylim = (f1_lo, 1.02)

    gt_slices = {}
    for sn in slice_names:
        _, gt_labels, _, _, n0, n1 = load_slice_data(sn)
        gt_slices[sn] = (gt_labels, n0, n1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for mode in ("classification", "probability"):
        out_path = OUTPUT_DIR / f"combined_{mode}.mp4"
        frame_dir = OUTPUT_DIR / "frames" / "combined" / mode
        frame_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nRendering {len(epochs)} combined {mode} frames")
        with tempfile.TemporaryDirectory() as tmpdir:
            for i, e in enumerate(epochs):
                fp = Path(tmpdir) / f"frame_{i:04d}.png"
                render_combined_frame(i, epochs, slice_names, all_data,
                                      all_metrics, gt_slices, fp, unc_ylim,
                                      f1_ylim=f1_ylim, mode=mode)
                import shutil
                shutil.copy2(str(fp), str(frame_dir / f"epoch_{e:03d}.png"))
                print(f"  Frame {i+1}/{len(epochs)} (epoch {e})")
            _stitch_video(tmpdir, out_path, fps)
        print(f"  Video saved: {out_path}")
        print(f"  Frames saved: {frame_dir}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Quadrotor 3D ROA video: per-method, per-slice videos + GT images"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=MC_BATCH_SIZE)
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--epochs", type=int, nargs="+", default=None,
                        help="Epochs to include (default: per-method max)")
    parser.add_argument("--fps", type=int, default=2)
    parser.add_argument("--methods", nargs="+", default=list(RUNS.keys()),
                        choices=list(RUNS.keys()))
    parser.add_argument("--slices", nargs="+", default=DEFAULT_SLICES,
                        choices=ALL_SLICE_NAMES,
                        help="Slices to process (default: theta_vs_q z_vs_zdot)")
    parser.add_argument("--gt_only", action="store_true")
    parser.add_argument("--cache_only", action="store_true")
    parser.add_argument("--combined_only", action="store_true",
                        help="Only generate combined comparison videos (skip per-method)")
    args = parser.parse_args()

    if not args.cache_only:
        print("=" * 60)
        print("Ground truth images")
        print("=" * 60)
        for sn in args.slices:
            render_gt_image(sn)

        if args.gt_only:
            print("\n--gt_only: skipping video generation.")
            return

    all_data = {}
    all_metrics = {}

    for method_key in args.methods:
        # Per-method epoch list: CLI override or method-specific default
        if args.epochs is not None:
            epochs = sorted(args.epochs)
        else:
            epochs = RUNS[method_key]["epochs"]

        all_data[method_key] = {}
        all_metrics[method_key] = {}
        for sn in args.slices:
            print(f"\n{'=' * 60}")
            print(f"Method: {RUNS[method_key]['label']}  |  Slice: {sn}")
            print(f"  Epochs: {epochs[0]}-{epochs[-1]} ({len(epochs)} total)")
            print(f"{'=' * 60}")

            results, gt_labels, axis0, axis1, n0, n1 = load_or_compute(
                method_key, sn, epochs, args.device, args.batch_size,
                args.recompute,
            )
            all_data[method_key][sn] = (results, gt_labels, n0, n1)

            metrics = {}
            for e in epochs:
                metrics[e] = {
                    "unc_pct": results[e]["eval_unc_pct"],
                    "f1": results[e]["eval_f1"],
                }
                print(f"  ep{e:2d}: unc={metrics[e]['unc_pct']:5.1f}%  F1={metrics[e]['f1']:.3f}")
            all_metrics[method_key][sn] = metrics

    if args.cache_only:
        print(f"\n--cache_only: caches saved to {OUTPUT_DIR}")
        return

    # Per-method videos use each method's own epoch list
    if not args.combined_only:
        for method_key in args.methods:
            if args.epochs is not None:
                epochs = sorted(args.epochs)
            else:
                epochs = RUNS[method_key]["epochs"]
            for sn in args.slices:
                results, gt_labels, n0, n1 = all_data[method_key][sn]
                make_video(
                    method_key, sn, epochs, results, gt_labels, n0, n1,
                    all_metrics[method_key][sn], args.fps,
                )

    # Fill missing epochs by repeating last available (e.g. nonadaptive lacks epoch 15)
    # Use the other method's n_traj so subtitle/line-plots show the correct count
    common_epochs = ALL_EPOCHS if args.epochs is None else sorted(args.epochs)
    for method_key in args.methods:
        method_epochs = RUNS[method_key]["epochs"]
        last_ep = max(method_epochs)
        # Find a reference method that has all epochs (for n_traj)
        ref_mk = [mk for mk in args.methods if mk != method_key and
                   set(common_epochs).issubset(RUNS[mk]["epochs"])]
        for e in common_epochs:
            if e not in method_epochs:
                print(f"  {method_key}: repeating epoch {last_ep} data for epoch {e}")
                for sn in args.slices:
                    results, gt_labels, n0, n1 = all_data[method_key][sn]
                    repeated = dict(results[last_ep])  # shallow copy
                    # Use reference method's n_traj if available
                    if ref_mk:
                        ref_results = all_data[ref_mk[0]][sn][0]
                        if e in ref_results:
                            repeated["n_traj"] = ref_results[e]["n_traj"]
                    results[e] = repeated
                    all_metrics[method_key][sn][e] = all_metrics[method_key][sn][last_ep]

    # Combined video uses common epoch range
    both_methods = set(args.methods) == set(RUNS.keys())
    if both_methods and len(args.slices) >= 2:
        print(f"\n{'=' * 60}")
        print(f"Combined comparison video (epochs {common_epochs[0]}-{common_epochs[-1]})")
        print(f"{'=' * 60}")
        make_combined_video(common_epochs, args.slices, all_data, all_metrics, args.fps)
    else:
        print("\nSkipping combined video (needs both methods and 2+ slices).")

    print(f"\nAll outputs in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
