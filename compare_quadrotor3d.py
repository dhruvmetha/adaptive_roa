#!/usr/bin/env python3
"""Compare evaluation metrics across adaptive sampling strategies for Quadrotor3D."""

import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

RUNS = [
    {
        "label": "Non-adaptive baseline",
        "base": (
            "/common/users/shared/pracsys/adaptive_roa_experiments/dhruv/adaptive_quadrotor3d/outputs/"
            "training_index_0_d2_ratio_0.0_warm_start_False_manifold_False_threshold_mode_dynamic_adapt_iter_10_alpha_0.1_sampling_mode_ranked/"
            "2026-02-18_14-25-10"
        ),
        "color": "blue",
        "marker": "o",
        "linestyle": "-",
    },
    {
        "label": "Adaptive direct",
        "base": (
            "/common/users/shared/pracsys/adaptive_roa_experiments/dhruv/adaptive_quadrotor3d/outputs/"
            "training_index_0_d2_ratio_1.0_warm_start_False_manifold_False_threshold_mode_dynamic_adapt_iter_10_alpha_0.1_sampling_mode_direct/"
            "2026-02-18_14-25-13"
        ),
        "color": "red",
        "marker": "s",
        "linestyle": "-",
    },
    {
        "label": "Adaptive ranked",
        "base": (
            "/common/users/shared/pracsys/adaptive_roa_experiments/dhruv/adaptive_quadrotor3d/outputs/"
            "training_index_0_d2_ratio_1.0_warm_start_False_manifold_False_threshold_mode_dynamic_adapt_iter_10_alpha_0.1_sampling_mode_ranked/"
            "2026-02-18_14-25-29"
        ),
        "color": "green",
        "marker": "^",
        "linestyle": "-",
    },
]

EVAL_SUBDIR = "evaluations/radius_0.3_alpha_0.1_mc_10_batch_100000"
OUTPUT_PATH = "/common/home/dm1487/robotics_research/tripods/olympics-classifier/quadrotor3d_comparison.pdf"


def load_run_data(base_path, eval_subdir):
    eval_base = f"{base_path}/{eval_subdir}"
    eval_epochs = sorted([d for d in os.listdir(eval_base) if d.startswith("epoch_")])

    data = {
        "train_trajectories": [], "epochs": [],
        "qhat_f1": [], "qhat_sep": [],
        "ld_f1": [], "ld_sep": [],
    }

    for ep_dir in eval_epochs:
        e = int(ep_dir.split("_")[1])
        with open(f"{base_path}/epoch_{e:03d}/artifacts_v2.json") as f:
            root_data = json.load(f)
        with open(f"{eval_base}/{ep_dir}/artifacts_v2.json") as f:
            eval_data = json.load(f)

        data["train_trajectories"].append(root_data["train_trajectories"])
        data["epochs"].append(e)

        qps = eval_data["eval_metrics"]["qhat_prediction_sets"]
        data["qhat_f1"].append(qps["f1"])
        data["qhat_sep"].append(qps["separatrix_pct"] * 100)

        ld = eval_data["eval_metrics"]["lambda_delta"]
        data["ld_f1"].append(ld["f1"])
        data["ld_sep"].append(ld["separatrix_pct"] * 100)

    return {k: np.array(v) for k, v in data.items()}


def main():
    run_data = []
    for run in RUNS:
        print(f"Loading {run['label']}...")
        d = load_run_data(run["base"], EVAL_SUBDIR)
        run_data.append(d)

    fig, axes = plt.subplots(2, 4, figsize=(24, 10))

    plot_configs = [
        (axes[0, 0], "epochs", "ld_f1", "Epoch", "F1 Score", r"Epoch vs F1: $\lambda \pm \delta$"),
        (axes[0, 1], "epochs", "qhat_f1", "Epoch", "F1 Score", r"Epoch vs F1: $\hat{q}$ pred sets"),
        (axes[0, 2], "train_trajectories", "ld_f1", "Trajectories", "F1 Score", r"Trajs vs F1: $\lambda \pm \delta$"),
        (axes[0, 3], "train_trajectories", "qhat_f1", "Trajectories", "F1 Score", r"Trajs vs F1: $\hat{q}$ pred sets"),
        (axes[1, 0], "epochs", "ld_sep", "Epoch", "Separatrix %", r"Epoch vs Sep%: $\lambda \pm \delta$"),
        (axes[1, 1], "epochs", "qhat_sep", "Epoch", "Separatrix %", r"Epoch vs Sep%: $\hat{q}$ pred sets"),
        (axes[1, 2], "train_trajectories", "ld_sep", "Trajectories", "Separatrix %", r"Trajs vs Sep%: $\lambda \pm \delta$"),
        (axes[1, 3], "train_trajectories", "qhat_sep", "Trajectories", "Separatrix %", r"Trajs vs Sep%: $\hat{q}$ pred sets"),
    ]

    for ax, x_key, y_key, xlabel, ylabel, title in plot_configs:
        for run, d in zip(RUNS, run_data):
            ax.plot(
                d[x_key], d[y_key],
                color=run["color"], marker=run["marker"], linestyle=run["linestyle"],
                label=run["label"], linewidth=2, markersize=5,
            )
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Quadrotor3D: Sampling Strategy Comparison", fontsize=16, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved to: {OUTPUT_PATH}")

    print("\n" + "=" * 120)
    print("Summary Table (Final Epoch)")
    print("=" * 120)
    print(f"{'Method':<25} {'Trajs':<7} {'ld_F1':<8} {'ld_Sep%':<9} {'qhat_F1':<9} {'qhat_Sep%':<10}")
    print("-" * 120)
    for run, d in zip(RUNS, run_data):
        i = -1
        print(
            f"{run['label']:<25} "
            f"{d['train_trajectories'][i]:<7} "
            f"{d['ld_f1'][i]:<8.4f} "
            f"{d['ld_sep'][i]:<9.2f} "
            f"{d['qhat_f1'][i]:<9.4f} "
            f"{d['qhat_sep'][i]:<10.2f}"
        )


if __name__ == "__main__":
    main()
