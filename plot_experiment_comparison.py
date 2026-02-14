"""Compare adaptive ROA experiments across different configurations."""
import json
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'legend.fontsize': 9,
    'figure.dpi': 150,
})

# ── Experiment definitions ──────────────────────────────────────────────
EXPERIMENTS = {
    "adaptive, d2=0.75": {
        "base": "/common/users/shared/pracsys/adaptive_roa_experiments/adaptive_cartpole_pybullet/outputs/"
                "training_index_0_d2_ratio_0.75_warm_start_True_manifold_False_threshold_mode_dynamic_adapt_iter_10_alpha_0.1_sampling_mode_ranked/"
                "2026-01-30_13-48-14",
        "format": "v1",
        "color": "#1f77b4",
        "marker": "o",
    },
    "adaptive, d2=0.5": {
        "base": "/common/users/shared/pracsys/adaptive_roa_experiments/adaptive_cartpole_pybullet/outputs/"
                "training_index_0_d2_ratio_0.5_warm_start_True_manifold_False_threshold_mode_dynamic_adapt_iter_10_alpha_0.1_sampling_mode_ranked/"
                "2026-01-30_13-48-34",
        "format": "v1",
        "color": "#ff7f0e",
        "marker": "s",
    },
    "(new) adaptive, d2=0.5": {
        "base": "/common/users/shared/pracsys/adaptive_roa_experiments/dhruv/adaptive_cartpole_pybullet/outputs/"
                "training_index_0_d2_ratio_0.5_warm_start_False_manifold_False_threshold_mode_dynamic_adapt_iter_15_alpha_0.1_sampling_mode_ranked_w_0.9/"
                "2026-02-12_11-21-03",
        "format": "v2",
        "color": "#2ca02c",
        "marker": "^",
    },
    "(new) non-adaptive": {
        "base": "/common/users/shared/pracsys/adaptive_roa_experiments/dhruv/adaptive_cartpole_pybullet/outputs/"
                "training_index_0_d2_ratio_0.5_warm_start_False_manifold_False_threshold_mode_dynamic_adapt_iter_1_alpha_0.1_sampling_mode_ranked_w_0.9/"
                "2026-02-12_11-52-02",
        "format": "v2",
        "color": "#d62728",
        "marker": "D",
    },
}


def load_epoch_results(base_dir):
    """Load per-epoch results.json files from an experiment directory."""
    epoch_dirs = sorted(glob.glob(os.path.join(base_dir, "epoch_*")))
    results = []
    for ed in epoch_dirs:
        rpath = os.path.join(ed, "results.json")
        if os.path.exists(rpath):
            with open(rpath) as f:
                results.append(json.load(f))
    return results


def extract_metrics(results_list, fmt):
    """Extract comparable metrics from a list of per-epoch result dicts.

    Returns dict of numpy arrays keyed by metric name.
    """
    metrics = {
        "epoch": [],
        "train_trajectories": [],
        "f1": [],
        "accuracy": [],
        "precision": [],
        "recall": [],
        "separatrix_pct": [],
        "invalid_pct": [],
        "uncertain_pct": [],
        "roa_error_full": [],
        "roa_error_certain": [],
        "roa_error_certain_success": [],
        "endpoint_mae_overall": [],
        "endpoint_mae_success": [],
        "endpoint_mae_failure": [],
        "lambda_star": [],
    }

    # Threshold key mapping
    if fmt == "v1":
        roa_key = "conformal_thresholds"
        # Try the v1 keys; fall back if needed
        roa_fallback = "lambda_delta"
    else:
        roa_key = "lambda_delta"
        roa_fallback = None

    for r in results_list:
        metrics["epoch"].append(r["epoch"])
        metrics["train_trajectories"].append(r.get("train_trajectories", 0))
        metrics["lambda_star"].append(r.get("lambda_star", 0))

        # Endpoint error from validation
        ee = r.get("endpoint_error", {})
        metrics["endpoint_mae_overall"].append(ee.get("overall_mae", np.nan))
        metrics["endpoint_mae_success"].append(ee.get("success_mae", np.nan))
        metrics["endpoint_mae_failure"].append(ee.get("failure_mae", np.nan))

        # Full ROA evaluation
        roa = r.get("full_roa", {})
        # Try primary key, then fallback
        block = roa.get(roa_key, roa.get(roa_fallback, {})) if roa_fallback else roa.get(roa_key, {})
        if not block:
            # Try all possible keys
            for k in ["conformal_thresholds", "lambda_delta", "notebook_thresholds"]:
                block = roa.get(k, {})
                if block:
                    break

        metrics["f1"].append(block.get("f1", np.nan))
        metrics["accuracy"].append(block.get("accuracy", np.nan))
        metrics["precision"].append(block.get("precision", np.nan))
        metrics["recall"].append(block.get("recall", np.nan))
        metrics["separatrix_pct"].append(block.get("separatrix_pct", np.nan))
        metrics["invalid_pct"].append(block.get("invalid_pct", np.nan))
        metrics["uncertain_pct"].append(block.get("uncertain_pct", np.nan))

        # Endpoint errors from ROA eval
        ep_err = roa.get("endpoint_errors", {})
        full_err = ep_err.get("full", {})
        certain_err = ep_err.get("certain", {})
        cs_err = ep_err.get("certain_success", {})
        metrics["roa_error_full"].append(full_err.get("mean", np.nan))
        metrics["roa_error_certain"].append(certain_err.get("mean", np.nan))
        metrics["roa_error_certain_success"].append(cs_err.get("mean", np.nan))

    return {k: np.array(v) for k, v in metrics.items()}


# ── Load all data ───────────────────────────────────────────────────────
all_data = {}
for name, cfg in EXPERIMENTS.items():
    results = load_epoch_results(cfg["base"])
    if not results:
        print(f"WARNING: No epoch results found for {name}")
        continue
    all_data[name] = extract_metrics(results, cfg["format"])
    print(f"Loaded {name}: {len(results)} epochs, "
          f"trajectories {all_data[name]['train_trajectories'][0]}->{all_data[name]['train_trajectories'][-1]}")

# ── Plot: Dataset Size vs F1 and Dataset Size vs Separatrix % ──────────
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5.5))
fig.suptitle("Adaptive ROA — CartPole (PyBullet)", fontsize=15, fontweight="bold")

for name, cfg in EXPERIMENTS.items():
    if name not in all_data:
        continue
    d = all_data[name]
    traj = d["train_trajectories"]
    mask = (traj >= 400) & (traj <= 1000)

    if len(traj[mask]) == 1:
        # Single point: draw as horizontal dashed line spanning the full x range
        for ax, vals in [(ax1, d["f1"]), (ax2, d["separatrix_pct"] * 100), (ax3, d["roa_error_full"])]:
            ax.axhline(y=vals[mask][0], color=cfg["color"], linestyle="--",
                       linewidth=2, label=name, alpha=0.85)
    else:
        ax1.plot(traj[mask], d["f1"][mask], color=cfg["color"], marker=cfg["marker"],
                 markersize=6, linewidth=2, label=name, alpha=0.85)
        ax2.plot(traj[mask], d["separatrix_pct"][mask] * 100, color=cfg["color"], marker=cfg["marker"],
                 markersize=6, linewidth=2, label=name, alpha=0.85)
        ax3.plot(traj[mask], d["roa_error_full"][mask], color=cfg["color"], marker=cfg["marker"],
                 markersize=6, linewidth=2, label=name, alpha=0.85)

ax1.set_xlabel("Dataset Size (# trajectories)")
ax1.set_ylabel("F1 Score")
ax1.set_title("F1 Score vs Dataset Size")
ax1.set_xlim(400, 1000)
ax1.grid(True, alpha=0.3)
ax1.legend(loc="lower right")

ax2.set_xlabel("Dataset Size (# trajectories)")
ax2.set_ylabel("Separatrix %")
ax2.set_title("Separatrix % vs Dataset Size")
ax2.set_xlim(400, 1000)
ax2.set_ylim(0, 10)
ax2.grid(True, alpha=0.3)
ax2.legend(loc="upper right")

ax3.set_xlabel("Dataset Size (# trajectories)")
ax3.set_ylabel("Mean Geodesic Error")
ax3.set_title("ROA Geodesic Error vs Dataset Size")
ax3.set_xlim(400, 1000)
ax3.grid(True, alpha=0.3)
ax3.legend(loc="upper right")

plt.tight_layout(rect=[0, 0, 1, 0.94])
out_path = "/common/home/dm1487/robotics_research/tripods/olympics-classifier/dataset_vs_f1_sep.pdf"
plt.savefig(out_path, bbox_inches="tight")
print(f"\nSaved to {out_path}")

# ── Also make a summary table ──────────────────────────────────────────
print("\n" + "="*100)
print("FINAL EPOCH SUMMARY")
print("="*100)
header = f"{'Experiment':<40} {'F1':>8} {'Acc':>8} {'Sep%':>8} {'Inv%':>8} {'ROA Err':>10} {'ROA CS':>10} {'Val MAE':>10} {'Traj':>6}"
print(header)
print("-"*100)
for name in EXPERIMENTS:
    if name not in all_data:
        continue
    d = all_data[name]
    i = -1  # last epoch
    print(f"{name:<40} {d['f1'][i]:8.4f} {d['accuracy'][i]:8.4f} {d['separatrix_pct'][i]:8.4f} "
          f"{d['invalid_pct'][i]:8.4f} {d['roa_error_full'][i]:10.4f} {d['roa_error_certain_success'][i]:10.4f} "
          f"{d['endpoint_mae_overall'][i]:10.6f} {int(d['train_trajectories'][i]):6d}")
