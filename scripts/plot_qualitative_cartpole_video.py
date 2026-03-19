#!/usr/bin/env python3
"""
CartPole ROA video: per-method, per-slice videos + static ground truth images.

Generates:
  Ground truth images (2):
    gt_theta_thetadot.pdf/.png
    gt_x_xdot.pdf/.png

  Per-method videos (4):
    adaptive_theta_thetadot.mp4
    adaptive_x_xdot.mp4
    nonadaptive_theta_thetadot.mp4
    nonadaptive_x_xdot.mp4

  Combined video (1):
    combined_comparison.mp4

  Caches (4):
    cache_adaptive_theta_thetadot.npz, etc.

Per-method frame layout (2x2):
  +---------------------------+---------------------------+
  |   Classification          |   p(success|x)     [cbar] |
  +---------------------------+---------------------------+
  |   Unc % vs Epoch          |   Macro F1 vs Epoch       |
  +---------------------------+---------------------------+

Combined frame layout (3x2):
  +---------------------------+---------------------------+
  |  Non-adaptive (theta,dth) |  Adaptive (theta,dth)     |
  +---------------------------+---------------------------+
  |  Non-adaptive (x, dx)     |  Adaptive (x, dx)         |
  +---------------------------+---------------------------+
  |  Unc % vs Epoch           |  Macro F1 vs Epoch        |
  +---------------------------+---------------------------+

Usage:
    # GT images only (no GPU)
    python scripts/plot_qualitative_cartpole_video.py --gt_only

    # Single epoch test
    python scripts/plot_qualitative_cartpole_video.py --epochs 15 --device cuda:0

    # Full 20-epoch run
    python scripts/plot_qualitative_cartpole_video.py --device cuda:0

    # Custom options
    python scripts/plot_qualitative_cartpole_video.py --device cuda:0 --fps 3 \\
        --methods adaptive --slices theta_thetadot
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_qualitative_cartpole import (
    SLICES,
    MC_BATCH_SIZE,
    _discrete_cmap,
    _prob_cmap,
    _pred_to_heatmap,
    _gt_to_heatmap,
    load_checkpoint,
    load_train_trajectories_count,
    load_slice_data,
    classify_grid,
)

# ── Run definitions (different from plot_qualitative_cartpole.py) ────────────
_BASE = Path(
    "/common/users/shared/pracsys/adaptive_roa_experiments/dhruv/"
    "adaptive_cartpole_pybullet/outputs"
)

_EVAL_SUBDIR = "radius_0.2_alpha_0.1_mc_10_batch_100000"

RUNS = {
    "adaptive": {
        "label": "Adaptive",
        "dir": _BASE / (
            "training_index_0_d2_ratio_1.0_warm_start_False_manifold_False"
            "_threshold_mode_dynamic_adapt_iter_20_alpha_0.1"
            "_sampling_mode_direct_w_0.9"
        ) / "2026-02-18_14-26-32",
    },
    "nonadaptive": {
        "label": "Non-adaptive",
        "dir": _BASE / (
            "training_index_0_d2_ratio_0.0_warm_start_False_manifold_False"
            "_threshold_mode_dynamic_adapt_iter_20_alpha_0.1"
            "_sampling_mode_ranked_w_0.9"
        ) / "2026-02-18_14-26-34",
    },
}

ALL_EPOCHS = list(range(15))  # 0-14 = 300-1000 traj
SLICE_NAMES = ["theta_thetadot", "x_xdot"]
OUTPUT_DIR = Path("results/videos_and_images/cartpole")


# ── Evaluation artifacts ─────────────────────────────────────────────────────

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


# ── Per-method computation ───────────────────────────────────────────────────

def compute_single_method(method_key, slice_name, epochs, device, batch_size,
                          existing_results=None):
    """Load model per epoch, classify grid, return results dict.

    Uses lambda/delta from evaluation artifacts. Saves cache incrementally.
    """
    import torch

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
    """Save computed grids to .npz."""
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
    """Load cached grids from .npz. Returns (results, gt_labels, axis0, axis1, n0, n1)."""
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
        # Backfill eval metrics from artifacts if missing from cache
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
    """Orchestrate caching with missing-epoch detection."""
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
    """Render and save static ground truth image for a slice."""
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
    traj_counts = [results[e]["n_traj"] for e in epochs]
    cur_traj = r["n_traj"]

    ax_unc = fig.add_subplot(gs[1, 0])
    unc_vals = [metrics[e]["unc_pct"] for e in epochs]
    ax_unc.plot(traj_counts, unc_vals, "o-", color="#2166ac", ms=4, lw=1.5)
    ax_unc.plot(cur_traj, metrics[epoch]["unc_pct"], "o",
                color="#2166ac", ms=12, mec="black", mew=1.5, zorder=5)
    ax_unc.axvline(cur_traj, color="gray", ls="--", alpha=0.5, lw=1)
    ax_unc.set_xlabel("Trajectories", fontsize=10)
    ax_unc.set_ylabel("Unc %", fontsize=10)
    ax_unc.set_title("Uncertain %", fontsize=11, fontweight="bold")
    traj_pad = max(50, int((max(traj_counts) - min(traj_counts)) * 0.05))
    ax_unc.set_xlim(min(traj_counts) - traj_pad, max(traj_counts) + traj_pad)
    ax_unc.set_ylim(0, unc_ylim)
    ax_unc.grid(True, alpha=0.3)
    ax_unc.tick_params(labelsize=9)

    ax_f1 = fig.add_subplot(gs[1, 1])
    f1_vals = [metrics[e]["f1"] for e in epochs]
    ax_f1.plot(traj_counts, f1_vals, "o-", color="#b2182b", ms=4, lw=1.5)
    ax_f1.plot(cur_traj, metrics[epoch]["f1"], "o",
               color="#b2182b", ms=12, mec="black", mew=1.5, zorder=5)
    ax_f1.axvline(cur_traj, color="gray", ls="--", alpha=0.5, lw=1)
    ax_f1.set_xlabel("Trajectories", fontsize=10)
    ax_f1.set_ylabel("F1", fontsize=10)
    ax_f1.set_title("F1 Score", fontsize=11, fontweight="bold")
    ax_f1.set_xlim(min(traj_counts) - traj_pad, max(traj_counts) + traj_pad)
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
                import shutil
                shutil.copy2(str(fp), str(frame_dir / f"epoch_{e:03d}.png"))
                print(f"  Frame {i+1}/{len(epochs)} (epoch {e})")
            _stitch_video(tmpdir, out_path, fps)
        print(f"  Video saved: {out_path}")
        print(f"  Frames saved: {frame_dir}")


# ── Combined video (both methods × both slices) ─────────────────────────────

# Style constants for combined line plots
METHOD_COLORS = {"adaptive": "#2166ac", "nonadaptive": "#b2182b"}
SLICE_STYLES = {"theta_thetadot": "-", "x_xdot": "--"}
SLICE_LABELS = {
    "theta_thetadot": r"$(\theta,\,\dot{\theta})$",
    "x_xdot": r"$(x,\,\dot{x})$",
}


def render_combined_frame(epoch_idx, epochs, all_data, all_metrics,
                          gt_slices, frame_path, unc_ylim, f1_ylim=None,
                          mode="classification"):
    """Render combined frame: rows=(GT, Adaptive, Non-adapt), cols=slices.

    Flagship styling: tight spacing, compact ticks, scaled bold fonts.
    mode: "classification" or "probability"
    """
    epoch = epochs[epoch_idx]
    cmap_d, norm_d = _discrete_cmap()
    cmap_p = _prob_cmap()
    method_keys = ["nonadaptive", "adaptive"]
    n_slices = len(SLICE_NAMES)
    n_rows = 1 + len(method_keys)  # GT + methods
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

    # Heatmap fonts — scale down for many columns
    title_fs = 22
    label_fs = 14
    tick_fs = 12
    method_fs = 20
    if n_slices > 3:
        s = min(1.0, cell_w / 5.5)
        title_fs = max(13, int(title_fs * s))
        label_fs = max(12, int(label_fs * s))
        tick_fs = max(11, int(tick_fs * s))
        method_fs = max(13, int(method_fs * s))

    # Heatmap margins — add padding to make 2-column heatmaps squarer
    margin_frac = 230 / 2560.0  # ~0.09

    right_margin = 0.92 if mode == "probability" else 0.99
    fig = plt.figure(figsize=(fig_w, fig_h))

    m = margin_frac
    gs_top = fig.add_gridspec(
        n_rows, gc,
        hspace=0.30, wspace=0.65,
        left=m + 0.16, right=(1.0 - m) - 0.07, top=0.85, bottom=0.35,
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

    for ci, sn in enumerate(SLICE_NAMES):
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
            label, xy=(0, 0.5), xytext=(-0.25, 0.5),
            xycoords="axes fraction", textcoords="axes fraction",
            fontsize=method_fs, fontweight="bold",
            ha="right", va="center", rotation=90,
            clip_on=False,
        )

    # Title — system name + trajectory count
    r = all_data[method_keys[0]][SLICE_NAMES[0]][0][epoch]
    fig.text(0.5, 0.98, "CartPole", fontsize=chrome_title_fs,
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

    sn0 = SLICE_NAMES[0]
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
    traj_pad = max(50, int((max(all_traj) - min(all_traj)) * 0.05))
    for ax, title, ylabel in [(ax_unc, "Uncertain %", "Unc %"),
                               (ax_f1, "F1 Score", "F1")]:
        ax.axvline(cur_traj, color="gray", ls="--", alpha=0.5, lw=1)
        ax.set_xlabel("Trajectories", fontsize=line_label_fs, fontweight="bold", labelpad=2)
        ax.set_ylabel(ylabel, fontsize=line_label_fs + 4, fontweight="bold", labelpad=8)
        ax.set_title(title, fontsize=line_title_fs + 4, fontweight="bold")
        ax.legend(fontsize=line_label_fs, loc="best", prop={"weight": "bold"})
        ax.set_xlim(min(all_traj) - traj_pad, max(all_traj) + traj_pad)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=line_tick_fs, pad=2)

    ax_unc.set_ylim(0, unc_ylim)
    if f1_ylim is not None:
        ax_f1.set_ylim(f1_ylim)
    else:
        ax_f1.set_ylim(0, 1.05)

    fig.savefig(str(frame_path), dpi=120)
    plt.close(fig)


def make_combined_video(epochs, all_data, all_metrics, fps):
    """Render classification and probability combined comparison videos."""
    all_f1 = []
    all_unc = []
    for mk in all_metrics:
        for sn in all_metrics[mk]:
            for e in epochs:
                all_unc.append(all_metrics[mk][sn][e]["unc_pct"])
                all_f1.append(all_metrics[mk][sn][e]["f1"])
    unc_ylim = max(all_unc) * 1.1 + 1
    # Zoom F1 axis: pad 5% below min, cap at 1.02
    f1_min = min(all_f1)
    f1_lo = max(0, f1_min - 0.05 * (1.0 - f1_min) - 0.02)
    f1_lo = round(f1_lo, 2)
    f1_ylim = (f1_lo, 1.02)

    gt_slices = {}
    for sn in SLICE_NAMES:
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
                render_combined_frame(i, epochs, all_data, all_metrics,
                                      gt_slices, fp, unc_ylim,
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
        description="CartPole ROA video: per-method, per-slice videos + GT images"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=MC_BATCH_SIZE)
    parser.add_argument("--recompute", action="store_true",
                        help="Force recomputation of all epochs")
    parser.add_argument("--epochs", type=int, nargs="+", default=ALL_EPOCHS,
                        help="Epochs to include (default: 0-19)")
    parser.add_argument("--fps", type=int, default=2,
                        help="Video frame rate (default: 2)")
    parser.add_argument("--methods", nargs="+", default=list(RUNS.keys()),
                        choices=list(RUNS.keys()),
                        help="Methods to process (default: all)")
    parser.add_argument("--slices", nargs="+", default=SLICE_NAMES,
                        choices=SLICE_NAMES,
                        help="Slices to process (default: all)")
    parser.add_argument("--gt_only", action="store_true",
                        help="Only generate ground truth images (no videos)")
    parser.add_argument("--cache_only", action="store_true",
                        help="Only compute and cache grids (no videos/images)")
    parser.add_argument("--combined_only", action="store_true",
                        help="Only generate combined comparison videos (skip per-method)")
    args = parser.parse_args()
    epochs = sorted(args.epochs)

    if not args.cache_only:
        # ── Ground truth images ──
        print("=" * 60)
        print("Ground truth images")
        print("=" * 60)
        for sn in args.slices:
            render_gt_image(sn)

        if args.gt_only:
            print("\n--gt_only: skipping video generation.")
            return

    # ── Load / compute all requested (method, slice) combos ──
    # all_data[method][slice] = (results, gt_labels, n0, n1)
    # all_metrics[method][slice][epoch] = {unc_pct, f1}
    all_data = {}
    all_metrics = {}

    for method_key in args.methods:
        all_data[method_key] = {}
        all_metrics[method_key] = {}
        for sn in args.slices:
            print(f"\n{'=' * 60}")
            print(f"Method: {RUNS[method_key]['label']}  |  Slice: {sn}")
            print(f"{'=' * 60}")

            results, gt_labels, axis0, axis1, n0, n1 = load_or_compute(
                method_key, sn, epochs, args.device, args.batch_size,
                args.recompute,
            )
            all_data[method_key][sn] = (results, gt_labels, n0, n1)

            # Use eval metrics from artifacts (same for all slices of a method)
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

    # ── Per-(method, slice) videos ──
    if not args.combined_only:
        for method_key in args.methods:
            for sn in args.slices:
                results, gt_labels, n0, n1 = all_data[method_key][sn]
                make_video(
                    method_key, sn, epochs, results, gt_labels, n0, n1,
                    all_metrics[method_key][sn], args.fps,
                )

    # ── Combined video (when both methods and both slices are requested) ──
    both_methods = set(args.methods) == set(RUNS.keys())
    both_slices = set(args.slices) == set(SLICE_NAMES)
    if both_methods and both_slices:
        print(f"\n{'=' * 60}")
        print("Combined comparison video")
        print(f"{'=' * 60}")
        make_combined_video(epochs, all_data, all_metrics, args.fps)
    else:
        print("\nSkipping combined video (needs both methods and both slices).")

    print(f"\nAll outputs in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
