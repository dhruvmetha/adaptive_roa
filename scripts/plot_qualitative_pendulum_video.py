#!/usr/bin/env python3
"""
Pendulum ROA video: per-method videos + combined comparison + GT image.

Uses eval_states.txt for ground truth and evaluation artifacts for metrics.

Generates:
  Ground truth image:
    gt_pendulum.pdf/.png

  Per-method videos (2 modes x 2 methods = 4):
    adaptive_classification.mp4, adaptive_probability.mp4
    nonadaptive_classification.mp4, nonadaptive_probability.mp4

  Combined videos (2 modes):
    combined_classification.mp4, combined_probability.mp4

  Caches:
    cache_adaptive.npz, cache_nonadaptive.npz

Per-method frame layout:
  +--------------------------------------------------+
  |          Heatmap (classification or probability)  |
  +-------------------------+-------------------------+
  |   Sep % vs Epoch        |   F1 Score vs Epoch     |
  +-------------------------+-------------------------+

Combined frame layout (1x3 + line plots):
  +------------------+------------------+------------------+
  |  Ground Truth    |  Non-adaptive    |  Adaptive        |
  +------------------+------------------+------------------+
  |  Sep % vs Epoch                | F1 Score vs Epoch    |
  +--------------------------------+----------------------+

Usage:
    # GT image only (no GPU)
    python scripts/plot_qualitative_pendulum_video.py --gt_only

    # Full 20-epoch run
    python scripts/plot_qualitative_pendulum_video.py --device cuda:0

    # Custom options
    python scripts/plot_qualitative_pendulum_video.py --device cuda:0 --fps 3 \\
        --methods adaptive
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_qualitative_pendulum import (
    ATTRACTOR_RADIUS,
    GRID_RES,
    MC_BATCH_SIZE,
    NUM_MC_SAMPLES,
    THETA_DOT_RANGE,
    THETA_RANGE,
    classify_grid,
    load_checkpoint,
    make_grid,
)

# ── Paths & Run definitions ──────────────────────────────────────────────────
_BASE = Path(
    "/common/users/shared/pracsys/adaptive_roa_experiments/dhruv/"
    "adaptive_pendulum_dhruv/outputs"
)

_EVAL_SUBDIR = "radius_0.075_alpha_0.1_mc_20_batch_100000"

GT_FILE = Path(
    "/common/users/shared/pracsys/genMoPlan/data_trajectories/"
    "deterministic/pendulum_lqr_50k/eval_states.txt"
)

RUNS = {
    "runc0": {
        "label": "Non-Adaptive\n" + r"($r_{\mathrm{unc}} = 0$)",
        "short_label": "Non-Adaptive",
        "dir": _BASE / (
            "training_index_0_d2_ratio_0.0_warm_start_False_threshold_mode_dynamic"
            "_adapt_iter_20_alpha_0.1_sampling_mode_ranked"
        ) / "2026-02-20_01-56-20",
    },
    "runc05": {
        "label": "Adaptive\n" + r"($r_{\mathrm{unc}} = 0.5$)",
        "short_label": r"$r_{\mathrm{unc}} = 0.5$",
        "dir": _BASE / (
            "training_index_0_d2_ratio_0.5_warm_start_False_threshold_mode_dynamic"
            "_adapt_iter_20_alpha_0.1_sampling_mode_direct"
        ) / "2026-03-01_19-50-39",
    },
    "runc1": {
        "label": "Adaptive\n" + r"($r_{\mathrm{unc}} = 1$)",
        "short_label": r"$r_{\mathrm{unc}} = 1$",
        "dir": _BASE / (
            "training_index_0_d2_ratio_1.0_warm_start_False_threshold_mode_dynamic"
            "_adapt_iter_20_alpha_0.1_sampling_mode_direct"
        ) / "2026-02-20_01-56-07",
    },
}
METHOD_ORDER = ["runc0", "runc05", "runc1"]

ALL_EPOCHS = list(range(20))  # 0-19
OUTPUT_DIR = Path("results/videos_and_images/pendulum")
EXTENT = [*THETA_RANGE, *THETA_DOT_RANGE]

# ── Training sample overlay paths ────────────────────────────────────────────
_TRAJ_BASE = Path(
    "/common/users/shared/pracsys/genMoPlan/data_trajectories/deterministic/pendulum_lqr_50k"
)
SHUFFLED_INDICES_FILE = _TRAJ_BASE / "train_test_splits" / "shuffled_indices_0.txt"
SHUFFLED_LABELS_FILE = _TRAJ_BASE / "train_test_splits" / "shuffled_labels_0.txt"
TRAJECTORIES_DIR = _TRAJ_BASE / "trajectories"

# ── Axis styling ──────────────────────────────────────────────────────────────
AX_CFG = {
    "xlabel": r"$\theta$ (rad)",
    "ylabel": r"$\dot{\theta}$ (rad/s)",
    "xticks": [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
    "xticklabels": [r"$-\pi$", r"$-\frac{\pi}{2}$", "0",
                    r"$\frac{\pi}{2}$", r"$\pi$"],
    "yticks": [-2 * np.pi, -np.pi, 0, np.pi, 2 * np.pi],
    "yticklabels": [r"$-2\pi$", r"$-\pi$", "0", r"$\pi$", r"$2\pi$"],
    "xticks_compact": [-np.pi, 0, np.pi],
    "xticklabels_compact": [r"$-\pi$", "0", r"$\pi$"],
    "yticks_compact": [-2 * np.pi, 0, 2 * np.pi],
    "yticklabels_compact": [r"$-2\pi$", "0", r"$2\pi$"],
}


# ── Colormaps ─────────────────────────────────────────────────────────────────

def _discrete_cmap():
    """Failure (#440154) / Uncertain (gray) / Success (#FDE725)."""
    colors = ["#440154", "#D3D3D3", "#FDE725"]
    cmap = mcolors.ListedColormap(colors)
    bounds = [-1.5, -0.5, 0.5, 1.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    return cmap, norm


def _prob_cmap():
    return mcolors.LinearSegmentedColormap.from_list(
        "roa_prob", ["#440154", "#31688E", "#21918C", "#5EC962", "#FDE725"]
    )


def _pred_to_heatmap(pred):
    """Map pred labels (1=success, 0=failure, -1=uncertain) to +1/0/-1."""
    heatmap = np.zeros_like(pred, dtype=float)
    heatmap[pred == 1] = 1      # success
    heatmap[pred == 0] = -1     # failure
    return heatmap.reshape(GRID_RES, GRID_RES).T


# ── Ground truth from eval_states.txt ─────────────────────────────────────────

GT_GRID_RES = 80  # coarser than model grid for smoother GT


def load_gt_binned():
    """Bin eval_states.txt into GT_GRID_RES x GT_GRID_RES grid.

    Returns heatmap with +1 (success), -1 (failure), NaN (no data).
    Uses a coarser grid than model predictions to avoid noise from
    sparse bins near the separatrix.
    """
    data = np.loadtxt(str(GT_FILE), delimiter=",")
    theta = data[:, 0]
    thetadot = data[:, 1]
    labels = data[:, 4].astype(int)  # 0=failure, 1=success

    n = GT_GRID_RES
    theta_edges = np.linspace(*THETA_RANGE, n + 1)
    thetadot_edges = np.linspace(*THETA_DOT_RANGE, n + 1)

    ti = np.clip(np.digitize(theta, theta_edges) - 1, 0, n - 1)
    tdi = np.clip(np.digitize(thetadot, thetadot_edges) - 1, 0, n - 1)

    count = np.zeros((n, n), dtype=int)
    success_count = np.zeros((n, n), dtype=int)
    np.add.at(count, (ti, tdi), 1)
    np.add.at(success_count, (ti, tdi), labels)

    gt = np.full((n, n), np.nan)
    has_data = count > 0
    majority_success = success_count[has_data] > (count[has_data] / 2)
    gt[has_data] = np.where(majority_success, 1.0, -1.0)
    return gt


# ── Evaluation artifacts ──────────────────────────────────────────────────────

def load_eval_artifacts(run_dir, epoch):
    """Load lambda_star, delta_star, F1, and sep% from evaluation artifacts."""
    arts = (run_dir / "evaluations" / _EVAL_SUBDIR
            / f"epoch_{epoch:03d}" / "artifacts_v2.json")
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


def load_train_trajectories_count(run_dir, epoch):
    """Return number of training trajectories at this epoch."""
    arts = run_dir / f"epoch_{epoch:03d}" / "artifacts_v2.json"
    with open(arts) as f:
        d = json.load(f)
    return int(d["train_trajectories"])


# ── Per-method computation ────────────────────────────────────────────────────

def compute_single_method(method_key, epochs, device, batch_size,
                          existing_results=None):
    import torch

    run = RUNS[method_key]
    run_dir = run["dir"]
    grid_np, _, _ = make_grid()
    results = dict(existing_results) if existing_results else {}

    for epoch in epochs:
        print(f"\n  {run['label']} | epoch {epoch}")
        eval_arts = load_eval_artifacts(run_dir, epoch)
        lam = eval_arts["lambda_star"]
        delta = eval_arts["delta_star"]
        print(f"    lambda*={lam:.4f}, delta*={delta:.4f}")
        print(f"    eval F1={eval_arts['f1']:.4f}, "
              f"eval sep%={eval_arts['unc_pct']:.1f}%")

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
        save_cache(results, method_key)

    return results


# ── Caching ───────────────────────────────────────────────────────────────────

def _cache_path(method_key):
    return OUTPUT_DIR / f"cache_{method_key}.npz"


def save_cache(results, method_key):
    save_dict = {}
    for epoch, r in results.items():
        pfx = f"epoch{epoch}"
        save_dict[f"{pfx}_pred"] = r["pred_labels"]
        save_dict[f"{pfx}_psuc"] = r["p_success"]
        save_dict[f"{pfx}_lam"] = np.array(r["lambda_star"])
        save_dict[f"{pfx}_delta"] = np.array(r["delta_star"])
        save_dict[f"{pfx}_ntraj"] = np.array(r["n_traj"])
        save_dict[f"{pfx}_eval_f1"] = np.array(r["eval_f1"])
        save_dict[f"{pfx}_eval_unc"] = np.array(r["eval_unc_pct"])
    cf = _cache_path(method_key)
    cf.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(cf), **save_dict)
    print(f"  Cache saved: {cf}")


def load_cache(method_key):
    data = np.load(str(_cache_path(method_key)), allow_pickle=False)
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
        save_cache(results, method_key)
    return results


def load_or_compute(method_key, epochs, device, batch_size, recompute):
    cf = _cache_path(method_key)
    if cf.exists() and not recompute:
        print(f"Loading cache: {cf}")
        results = load_cache(method_key)
        missing = [e for e in epochs if e not in results]
        if missing:
            print(f"  Cache missing epochs {missing}, computing...")
            results = compute_single_method(
                method_key, missing, device, batch_size,
                existing_results=results,
            )
    else:
        results = compute_single_method(method_key, epochs, device, batch_size)
    return results


# ── Training sample overlay ───────────────────────────────────────────────────

def preload_start_states(pool_indices):
    """Read first line (θ,θ̇) of trajectory files for given pool indices.

    pool_indices maps pool position → filename (e.g. 'sequence_40950.txt').
    Returns dict[pool_idx → (θ, θ̇)].
    """
    filenames = open(SHUFFLED_INDICES_FILE).read().splitlines()
    states = {}
    for idx in pool_indices:
        fname = filenames[idx].strip()
        path = TRAJECTORIES_DIR / fname
        line = open(path).readline().strip()
        theta, thetadot = line.split(",")
        states[idx] = (float(theta), float(thetadot))
    return states


def load_sample_data_for_run(run_dir, epochs):
    """Load per-epoch cumulative and new sample indices from artifacts.

    Returns dict[epoch → {cumulative_indices: set, new_indices: list}].
    """
    sample_data = {}
    # Detect seed size from epoch 0
    with open(run_dir / "epoch_000" / "artifacts_v2.json") as f:
        seed_size = json.load(f)["train_trajectories"]
    cumulative = set(range(seed_size))

    for epoch in sorted(epochs):
        arts_path = run_dir / f"epoch_{epoch:03d}" / "artifacts_v2.json"
        with open(arts_path) as f:
            acq = json.load(f)["acquisition"]
        new = acq["d1_indices"] + acq["d2_indices"]

        sample_data[epoch] = {
            "cumulative_indices": set(cumulative),
            "new_indices": new,
        }
        cumulative.update(new)

    return sample_data


def load_all_sample_data(methods, epochs):
    """Load sample data for all methods.

    Returns (all_sample_data, start_states, labels) where:
      all_sample_data: dict[method → dict[epoch → {cumulative, new}]]
      start_states: dict[pool_idx → (θ, θ̇)]
      labels: dict[pool_idx → int]
    """
    all_sample_data = {}
    all_pool_indices = set()

    for mk in methods:
        run_dir = RUNS[mk]["dir"]
        sd = load_sample_data_for_run(run_dir, epochs)
        all_sample_data[mk] = sd
        for epoch_data in sd.values():
            all_pool_indices.update(epoch_data["cumulative_indices"])
            all_pool_indices.update(epoch_data["new_indices"])

    print(f"  Loading {len(all_pool_indices)} start states from trajectory files...")
    start_states = preload_start_states(all_pool_indices)

    raw_labels = open(SHUFFLED_LABELS_FILE).read().splitlines()
    labels = {idx: int(raw_labels[idx]) for idx in all_pool_indices}

    return all_sample_data, start_states, labels


def _overlay_samples(ax, cumulative_indices, new_indices, start_states):
    """Scatter cumulative (gray) and new (red diamond) training samples."""
    if cumulative_indices:
        cum_pts = np.array([start_states[i] for i in cumulative_indices
                            if i in start_states])
        if len(cum_pts) > 0:
            ax.scatter(
                cum_pts[:, 0], cum_pts[:, 1],
                s=10, alpha=0.45, c="black", edgecolors="black",
                linewidths=0.4, zorder=3,
            )
    if new_indices:
        new_pts = np.array([start_states[i] for i in new_indices
                            if i in start_states])
        if len(new_pts) > 0:
            ax.scatter(
                new_pts[:, 0], new_pts[:, 1],
                s=30, alpha=0.8, c="red", edgecolors="darkred",
                linewidths=0.5, marker="D", zorder=4,
            )


# ── Ground truth image ────────────────────────────────────────────────────────

def render_gt_image():
    gt = load_gt_binned()
    cmap_d, norm_d = _discrete_cmap()
    cmap_d.set_bad(color="white")

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(
        gt.T, origin="lower", aspect="auto",
        cmap=cmap_d, norm=norm_d, extent=EXTENT, interpolation="nearest",
    )
    ax.set_xlabel(AX_CFG["xlabel"], fontsize=12, fontweight="bold")
    ax.set_ylabel(AX_CFG["ylabel"], fontsize=12, fontweight="bold")
    ax.set_xticks(AX_CFG["xticks"])
    ax.set_xticklabels(AX_CFG["xticklabels"], fontsize=10)
    ax.set_yticks(AX_CFG["yticks"])
    ax.set_yticklabels(AX_CFG["yticklabels"], fontsize=10)
    ax.set_title("Ground Truth", fontsize=14, fontweight="bold")

    legend_elements = [
        Patch(facecolor="#FDE725", edgecolor="none", label="Success"),
        Patch(facecolor="#440154", edgecolor="none", label="Failure"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=10,
              framealpha=0.9)

    fig.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        out = OUTPUT_DIR / f"gt_pendulum.{ext}"
        fig.savefig(str(out), dpi=300, bbox_inches="tight")
        print(f"  Saved: {out}")
    plt.close(fig)


# ── Per-method video frame ────────────────────────────────────────────────────

def render_video_frame(epoch_idx, epochs, method_label, results,
                       metrics, frame_path, unc_ylim, mode="classification",
                       sample_data=None, start_states=None):
    epoch = epochs[epoch_idx]
    r = results[epoch]

    fig = plt.figure(figsize=(16, 9))
    right_margin = 0.88 if mode == "probability" else 0.93
    gs = fig.add_gridspec(
        2, 2, height_ratios=[1.2, 0.8],
        hspace=0.40, wspace=0.35,
        left=0.07, right=right_margin, top=0.88, bottom=0.10,
    )

    fig.suptitle(
        f"{method_label}\n"
        f"Epoch {epoch}  |  {r['n_traj']} traj  |  "
        rf"$\lambda^*$={r['lambda_star']:.3f}  "
        rf"$\delta$={r['delta_star']:.3f}",
        fontsize=14, fontweight="bold",
    )

    # ── Heatmap (top, spans both columns) ──
    ax_map = fig.add_subplot(gs[0, :])
    if mode == "classification":
        cmap_d, norm_d = _discrete_cmap()
        ax_map.imshow(
            _pred_to_heatmap(r["pred_labels"]),
            origin="lower", aspect="auto",
            cmap=cmap_d, norm=norm_d, extent=EXTENT, interpolation="nearest",
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
            p_suc.reshape(GRID_RES, GRID_RES).T,
            origin="lower", aspect="auto",
            cmap=cmap_p, vmin=0, vmax=1, extent=EXTENT,
            interpolation="bilinear",
        )
        ax_map.set_title(r"$p(\mathrm{success} \mid x)$",
                         fontsize=12, fontweight="bold")
        cbar_ax = fig.add_axes([0.90, 0.52, 0.015, 0.36])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label(r"$p(\mathrm{success})$", fontsize=9)
        cbar.ax.tick_params(labelsize=8)

    ax_map.set_xticks(AX_CFG["xticks"])
    ax_map.set_xticklabels(AX_CFG["xticklabels"], fontsize=9)
    ax_map.set_yticks(AX_CFG["yticks"])
    ax_map.set_yticklabels(AX_CFG["yticklabels"], fontsize=9)
    ax_map.set_xlabel(AX_CFG["xlabel"], fontsize=10)
    ax_map.set_ylabel(AX_CFG["ylabel"], fontsize=10)

    if sample_data is not None and start_states is not None:
        sd = sample_data.get(epoch)
        if sd:
            _overlay_samples(ax_map, sd["cumulative_indices"],
                             sd["new_indices"], start_states)

    # ── Line plots (bottom) ──
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


# ── Video assembly ────────────────────────────────────────────────────────────

def _stitch_video(frame_dir, out_path, fps):
    # Use fraction notation for ffmpeg framerate (e.g. 9/10 for 0.9 fps)
    from fractions import Fraction
    frac = Fraction(fps).limit_denominator(1000)
    framerate_str = f"{frac.numerator}/{frac.denominator}"
    subprocess.run(
        ["ffmpeg", "-y", "-framerate", framerate_str,
         "-i", str(Path(frame_dir) / "frame_%04d.png"),
         "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
         "-r", "30",  # output at 30fps for smooth playback
         str(out_path)],
        check=True,
    )


def make_video(method_key, epochs, results, metrics, fps,
               sample_data=None, start_states=None):
    method_label = RUNS[method_key]["label"]
    max_unc = max(metrics[e]["unc_pct"] for e in epochs)
    unc_ylim = max_unc * 1.1 + 1
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for mode in ("classification", "probability"):
        out_path = OUTPUT_DIR / f"{method_key}_{mode}.mp4"
        print(f"\nRendering {len(epochs)} frames: {method_key} / {mode}")
        with tempfile.TemporaryDirectory() as tmpdir:
            for i, e in enumerate(epochs):
                fp = Path(tmpdir) / f"frame_{i:04d}.png"
                render_video_frame(
                    i, epochs, method_label, results,
                    metrics, fp, unc_ylim, mode=mode,
                    sample_data=sample_data, start_states=start_states,
                )
                print(f"  Frame {i+1}/{len(epochs)} (epoch {e})")
            _stitch_video(tmpdir, out_path, fps)
        print(f"  Video saved: {out_path}")


# ── Combined video ────────────────────────────────────────────────────────────

METHOD_COLORS = {"runc0": "#b2182b", "runc05": "#1b7837", "runc1": "#2166ac"}


def render_combined_frame(epoch_map, traj_count, all_data,
                          gt_heatmap, frame_path,
                          mode="classification",
                          all_sample_data=None, start_states=None,
                          is_last_frame=False,
                          samples_bg_mode=None,
                          all_metrics=None, all_traj_counts=None,
                          unc_ylim=None, f1_ylim=None):
    """Render combined frame: 4 rows (GT + 3 r_unc methods) × 2 cols.

    Col 0: heatmap only. Col 1: heatmap + training sample overlay.
    GT row uses col 0 only. Cartpole-style aspect ratio with row labels.

    samples_bg_mode: if set, col 1 uses this mode for the background heatmap
        (e.g. "probability" while col 0 uses "classification").

    epoch_map: dict[method_key → epoch] for trajectory-aligned rendering.
    """
    cmap_d, norm_d = _discrete_cmap()
    cmap_d_gt, norm_d_gt = _discrete_cmap()
    cmap_d_gt.set_bad(color="white")
    cmap_p = _prob_cmap()

    # Font sizes (matching cartpole style)
    title_fs = 28
    subtitle_fs = 22
    label_fs = 14
    tick_fs = 12
    method_fs = 20
    legend_fs = 18
    col_header_fs = 18

    # 2560×1440 @ 120 dpi (same as cartpole)
    fig_w = 2560 / 120.0
    fig_h = 1440 / 120.0

    fig = plt.figure(figsize=(fig_w, fig_h))

    # Line plot fonts (matching cartpole)
    line_title_fs = 16
    line_label_fs = 14
    line_tick_fs = 12

    ac = AX_CFG
    method_axes = {}  # (row, col) → ax for method rows only

    # Layout: heatmaps on left, line plots stacked on right, centered overall
    left_m, right_m = 0.14, 0.53  # heatmap area
    lines_l, lines_r = 0.66, 0.88  # line plot area
    content_cx = (left_m + lines_r) / 2  # visual center of full content

    # GT gridspec: single centered plot (within heatmap area)
    hm_cx = (left_m + right_m) / 2
    gt_half_w = (right_m - left_m) / 4
    gs_gt = fig.add_gridspec(
        1, 1, left=hm_cx - gt_half_w, right=hm_cx + gt_half_w,
        top=0.85, bottom=0.71,
    )

    # Methods gridspec: 3 rows × 2 cols
    gs_methods = fig.add_gridspec(
        3, 2, hspace=0.45, wspace=0.18,
        left=left_m, right=right_m, top=0.62, bottom=0.06,
    )

    # Line plots gridspec: 2 rows × 1 col, stacked vertically on right
    gs_lines = fig.add_gridspec(
        2, 1, hspace=0.50,
        left=lines_l, right=lines_r, top=0.78, bottom=0.10,
    )

    def _style_ax(ax, col):
        ax.set_xticks(ac["xticks_compact"])
        ax.set_xticklabels(ac["xticklabels_compact"], fontsize=tick_fs)
        ax.set_yticks(ac["yticks_compact"])
        ax.tick_params(length=2, width=0.8, pad=1)
        ax.set_xlabel(ac["xlabel"], fontsize=label_fs, fontweight="bold",
                      labelpad=1)
        ax.set_ylabel(ac["ylabel"], fontsize=label_fs, fontweight="bold",
                      labelpad=1)
        if col == 0:
            ax.set_yticklabels(ac["yticklabels_compact"], fontsize=tick_fs)
        else:
            ax.yaxis.set_label_position("right")
            ax.yaxis.tick_right()
            ax.set_yticklabels(ac["yticklabels_compact"], fontsize=tick_fs)

    def _draw_heatmap(ax, r, alpha=1.0, override_mode=None):
        m = override_mode or mode
        if m == "classification":
            ax.imshow(
                _pred_to_heatmap(r["pred_labels"]),
                origin="lower", aspect="auto",
                cmap=cmap_d, norm=norm_d, extent=EXTENT,
                interpolation="nearest", alpha=alpha,
            )
        else:
            p_suc = r["p_success"]
            if hasattr(p_suc, "cpu"):
                p_suc = p_suc.cpu().numpy()
            ax.imshow(
                p_suc.reshape(GRID_RES, GRID_RES).T,
                origin="lower", aspect="auto",
                cmap=cmap_p, vmin=0, vmax=1, extent=EXTENT,
                interpolation="bilinear", alpha=alpha,
            )

    # ── Row 0: GT (centered) ──
    ax_gt = fig.add_subplot(gs_gt[0, 0])
    ax_gt.imshow(
        gt_heatmap.T, origin="lower", aspect="auto",
        cmap=cmap_d_gt, norm=norm_d_gt, extent=EXTENT,
        interpolation="nearest",
    )
    ax_gt.set_xticks(ac["xticks_compact"])
    ax_gt.set_xticklabels(ac["xticklabels_compact"], fontsize=tick_fs)
    ax_gt.set_yticks(ac["yticks_compact"])
    ax_gt.set_yticklabels(ac["yticklabels_compact"], fontsize=tick_fs)
    ax_gt.tick_params(length=2, width=0.8, pad=1)
    ax_gt.set_xlabel(ac["xlabel"], fontsize=label_fs, fontweight="bold",
                     labelpad=1)
    ax_gt.set_ylabel(ac["ylabel"], fontsize=label_fs, fontweight="bold",
                     labelpad=1)

    # "Ground Truth" — horizontal, above GT plot
    gt_pos = gs_gt[0, 0].get_position(fig)
    fig.text((gt_pos.x0 + gt_pos.x1) / 2, gt_pos.y1 + 0.005,
             "Ground Truth", fontsize=method_fs, fontweight="bold",
             ha="center", va="bottom")

    # ── Column headers between GT and method rows ──
    col_centers = [
        (gs_methods[0, 0].get_position(fig).x0
         + gs_methods[0, 0].get_position(fig).x1) / 2,
        (gs_methods[0, 1].get_position(fig).x0
         + gs_methods[0, 1].get_position(fig).x1) / 2,
    ]
    header_y = gs_methods[0, 0].get_position(fig).y1 + 0.015
    for cx, label in zip(col_centers, ["Prediction", "Training Samples"]):
        fig.text(cx, header_y, label, fontsize=col_header_fs,
                 fontweight="bold", ha="center", va="bottom")

    # ── Rows 0-2 (methods) ──
    im_last = None
    for mi, mk in enumerate(METHOD_ORDER):
        results = all_data[mk]
        r = results[epoch_map[mk]]

        # Col 0: plain heatmap
        ax_left = fig.add_subplot(gs_methods[mi, 0])
        method_axes[(mi, 0)] = ax_left
        _draw_heatmap(ax_left, r)
        _style_ax(ax_left, 0)

        # Col 1: heatmap (faded) + samples
        ax_right = fig.add_subplot(gs_methods[mi, 1])
        method_axes[(mi, 1)] = ax_right
        _draw_heatmap(ax_right, r, alpha=0.4, override_mode=samples_bg_mode)
        _style_ax(ax_right, 1)

        if all_sample_data is not None and start_states is not None:
            sd = all_sample_data.get(mk, {}).get(epoch_map[mk])
            if sd:
                if is_last_frame:
                    all_indices = set(sd["cumulative_indices"]) | set(sd["new_indices"])
                    _overlay_samples(ax_right, all_indices, [], start_states)
                else:
                    _overlay_samples(ax_right, sd["cumulative_indices"],
                                     sd["new_indices"], start_states)

        if mode == "probability" or samples_bg_mode == "probability":
            im_last = ax_right.images[0]

    # ── Method row labels (left side, horizontal) ──
    for mi, mk in enumerate(METHOD_ORDER):
        method_axes[(mi, 0)].annotate(
            RUNS[mk]["label"], xy=(0, 0.5), xytext=(-0.18, 0.5),
            xycoords="axes fraction", textcoords="axes fraction",
            fontsize=method_fs - 4, fontweight="bold",
            ha="right", va="center", rotation=0,
            clip_on=False,
        )

    # ── Colorbar for probability mode ──
    if (mode == "probability" or samples_bg_mode == "probability") and im_last is not None:
        cbar_ax = fig.add_axes([0.545, 0.06, 0.010, 0.56])
        cbar = fig.colorbar(im_last, cax=cbar_ax)
        cbar.set_label(r"$p(\mathrm{success})$", fontsize=label_fs)
        cbar.ax.tick_params(labelsize=tick_fs)

    # ── Line plots (right side) ──
    if all_metrics is not None and all_traj_counts is not None:
        ax_unc = fig.add_subplot(gs_lines[0, 0])
        ax_f1 = fig.add_subplot(gs_lines[1, 0])

        # Build filtered data per method, then align first points
        line_data = {}
        for mk in METHOD_ORDER:
            traj_vals = all_traj_counts[mk]
            sorted_epochs = sorted(all_metrics[mk].keys())
            unc_vals = [all_metrics[mk][e]["unc_pct"] for e in sorted_epochs]
            f1_vals = [all_metrics[mk][e]["f1"] for e in sorted_epochs]
            mask = [(100 <= t <= 500) for t in traj_vals]
            line_data[mk] = {
                "traj": [t for t, m in zip(traj_vals, mask) if m],
                "unc": [u for u, m in zip(unc_vals, mask) if m],
                "f1": [f for f, m in zip(f1_vals, mask) if m],
            }

        # Align first points across methods (use runc0 as reference)
        ref = METHOD_ORDER[0]
        for mk in METHOD_ORDER[1:]:
            if line_data[mk]["traj"] and line_data[ref]["traj"]:
                line_data[mk]["traj"][0] = line_data[ref]["traj"][0]
                line_data[mk]["unc"][0] = line_data[ref]["unc"][0]
                line_data[mk]["f1"][0] = line_data[ref]["f1"][0]

        for mk in METHOD_ORDER:
            color = METHOD_COLORS[mk]
            label = RUNS[mk]["short_label"]
            tv = line_data[mk]["traj"]
            uv = line_data[mk]["unc"]
            fv = line_data[mk]["f1"]

            ax_unc.plot(tv, uv, "o-", color=color, label=label, ms=4, lw=2)
            ax_f1.plot(tv, fv, "o-", color=color, label=label, ms=4, lw=2)

            # Current position marker
            if traj_count in tv:
                idx = tv.index(traj_count)
                ax_unc.plot(traj_count, uv[idx], "o", color=color,
                            ms=10, mec="black", mew=1.5, zorder=5)
                ax_f1.plot(traj_count, fv[idx], "o", color=color,
                           ms=10, mec="black", mew=1.5, zorder=5)

        for ax, title, ylabel in [(ax_unc, "Uncertain %", "Unc %"),
                                   (ax_f1, "F1 Score", "F1")]:
            ax.axvline(traj_count, color="gray", ls="--", alpha=0.4, lw=1)
            ax.set_xlabel("Trajectories", fontsize=line_label_fs - 1,
                          fontweight="bold", labelpad=2)
            ax.set_ylabel(ylabel, fontsize=line_label_fs - 1,
                          fontweight="bold", labelpad=4)
            ax.set_title(title, fontsize=line_title_fs - 1, fontweight="bold",
                         pad=6)
            ax.legend(fontsize=line_label_fs - 4, loc="best",
                      prop={"weight": "bold"}, framealpha=0.8)
            ax.set_xlim(80, 520)
            ax.grid(True, alpha=0.25, linewidth=0.5)
            ax.tick_params(labelsize=line_tick_fs - 1)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        if unc_ylim is not None:
            ax_unc.set_ylim(0, unc_ylim)
        if f1_ylim is not None:
            ax_f1.set_ylim(f1_ylim)
        else:
            ax_f1.set_ylim(0, 1.05)

    # ── Title ──
    fig.text(content_cx, 0.98, "Pendulum", fontsize=title_fs,
             fontweight="bold", ha="center", va="top")
    fig.text(content_cx, 0.93, f"Trajectories: {traj_count}",
             fontsize=subtitle_fs, fontweight="bold", ha="center", va="top")

    fig.savefig(str(frame_path), dpi=120)
    plt.close(fig)


def make_combined_video(epochs, all_data, all_metrics, fps,
                        all_sample_data=None, start_states=None,
                        traj_range=(100, 500)):
    """Render combined videos aligned by trajectory count."""
    # Build traj_count → epoch mapping per method
    traj_to_epoch = {}
    for mk in METHOD_ORDER:
        traj_to_epoch[mk] = {}
        for e in epochs:
            if e in all_data[mk]:
                n = all_data[mk][e]["n_traj"]
                traj_to_epoch[mk][n] = e

    # Find common trajectory counts in range
    common = sorted(set.intersection(
        *[set(m.keys()) for m in traj_to_epoch.values()]
    ))
    common = [t for t in common if traj_range[0] <= t <= traj_range[1]]

    if not common:
        print("WARNING: No common trajectory counts found in range "
              f"{traj_range}. Skipping combined video.")
        return

    print(f"  Aligned frames: {len(common)} "
          f"({common[0]}–{common[-1]} trajectories)")

    gt_heatmap = load_gt_binned()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Build per-method trajectory count lists and compute y-limits
    all_traj_counts = {}
    all_unc = []
    all_f1 = []
    for mk in METHOD_ORDER:
        sorted_epochs = sorted(all_data[mk].keys())
        all_traj_counts[mk] = [all_data[mk][e]["n_traj"] for e in sorted_epochs]
        for e in sorted_epochs:
            all_unc.append(all_metrics[mk][e]["unc_pct"])
            all_f1.append(all_metrics[mk][e]["f1"])
    unc_ylim = max(all_unc) * 1.1 + 1
    f1_min = min(all_f1)
    f1_lo = max(0, f1_min - 0.05 * (1.0 - f1_min) - 0.02)
    f1_ylim = (round(f1_lo, 2), 1.02)

    # (mode, samples_bg_mode, suffix) for each combined video variant
    variants = [
        ("classification", None, "classification"),
        ("probability", None, "probability"),
        ("classification", "probability", "classification_prob_samples"),
    ]

    for mode, sbg, suffix in variants:
        out_path = OUTPUT_DIR / f"combined_{suffix}.mp4"
        frame_dir = OUTPUT_DIR / "frames" / "combined" / suffix
        frame_dir.mkdir(parents=True, exist_ok=True)
        label = f"{mode}" + (f" (samples bg: {sbg})" if sbg else "")
        print(f"\nRendering {len(common)} combined {label} frames")
        with tempfile.TemporaryDirectory() as tmpdir:
            for i, traj_count in enumerate(common):
                epoch_map = {mk: traj_to_epoch[mk][traj_count]
                             for mk in METHOD_ORDER}
                fp = Path(tmpdir) / f"frame_{i:04d}.png"
                render_combined_frame(
                    epoch_map, traj_count, all_data,
                    gt_heatmap, fp, mode=mode,
                    all_sample_data=all_sample_data,
                    start_states=start_states,
                    is_last_frame=(i == len(common) - 1),
                    samples_bg_mode=sbg,
                    all_metrics=all_metrics,
                    all_traj_counts=all_traj_counts,
                    unc_ylim=unc_ylim, f1_ylim=f1_ylim,
                )
                import shutil
                shutil.copy2(str(fp), str(frame_dir / f"traj_{traj_count:04d}.png"))
                print(f"  Frame {i+1}/{len(common)} (traj={traj_count})")
            _stitch_video(tmpdir, out_path, fps)
        print(f"  Video saved: {out_path}")
        print(f"  Frames saved: {frame_dir}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Pendulum ROA video: per-method videos + GT images"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=MC_BATCH_SIZE)
    parser.add_argument("--recompute", action="store_true",
                        help="Force recomputation of all epochs")
    parser.add_argument("--epochs", type=int, nargs="+", default=ALL_EPOCHS,
                        help="Epochs to include (default: 0-19)")
    parser.add_argument("--fps", type=float, default=0.9,
                        help="Video frame rate (default: 0.9 ≈ 10s for 9 frames)")
    parser.add_argument("--methods", nargs="+", default=list(RUNS.keys()),
                        choices=list(RUNS.keys()),
                        help="Methods to process (default: all)")
    parser.add_argument("--gt_only", action="store_true",
                        help="Only generate ground truth images (no videos)")
    parser.add_argument("--cache_only", action="store_true",
                        help="Only compute and cache grids (no videos/images)")
    parser.add_argument("--combined_only", action="store_true",
                        help="Only generate combined videos (skip per-method)")
    parser.add_argument("--show_samples", action="store_true",
                        help="Overlay training sample locations on heatmaps")
    args = parser.parse_args()
    epochs = sorted(args.epochs)

    if not args.cache_only:
        print("=" * 60)
        print("Ground truth image")
        print("=" * 60)
        render_gt_image()

        if args.gt_only:
            print("\n--gt_only: skipping video generation.")
            return

    # ── Load / compute all requested methods ──
    all_data = {}      # method -> {epoch: {...}}
    all_metrics = {}   # method -> {epoch: {unc_pct, f1}}

    for method_key in args.methods:
        print(f"\n{'=' * 60}")
        print(f"Method: {RUNS[method_key]['label']}")
        print(f"{'=' * 60}")

        results = load_or_compute(
            method_key, epochs, args.device, args.batch_size, args.recompute,
        )
        all_data[method_key] = results

        metrics = {}
        for e in epochs:
            metrics[e] = {
                "unc_pct": results[e]["eval_unc_pct"],
                "f1": results[e]["eval_f1"],
            }
            print(f"  ep{e:2d}: unc={metrics[e]['unc_pct']:5.1f}%  "
                  f"F1={metrics[e]['f1']:.3f}")
        all_metrics[method_key] = metrics

    if args.cache_only:
        print(f"\n--cache_only: caches saved to {OUTPUT_DIR}")
        return

    # ── Load sample data ──
    all_sample_data = None
    start_states = None
    all_methods = set(args.methods) == set(RUNS.keys())
    if args.show_samples or all_methods:
        print(f"\n{'=' * 60}")
        print("Loading training sample data for overlay")
        print(f"{'=' * 60}")
        all_sample_data, start_states, _ = load_all_sample_data(
            args.methods, epochs,
        )

    # ── Per-method videos ──
    if not args.combined_only:
        for method_key in args.methods:
            mk_samples = None
            mk_states = None
            if args.show_samples:
                mk_samples = (all_sample_data or {}).get(method_key)
                mk_states = start_states
            make_video(method_key, epochs, all_data[method_key],
                       all_metrics[method_key], args.fps,
                       sample_data=mk_samples, start_states=mk_states)

    # ── Combined video (needs all methods) ──
    if all_methods:
        print(f"\n{'=' * 60}")
        print("Combined comparison video")
        print(f"{'=' * 60}")
        make_combined_video(epochs, all_data, all_metrics, args.fps,
                            all_sample_data=all_sample_data,
                            start_states=start_states)
    else:
        print("\nSkipping combined video (needs all methods).")

    print(f"\nAll outputs in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
