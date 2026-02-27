#!/usr/bin/env python3
"""Compare joint vs delta optimization mode for each sampling strategy on Quadrotor2D.

Layout: 2 rows (F1, Sep%) x 3 columns (Non-adaptive, Direct, Ranked)
Each subplot shows joint (solid) vs delta (dashed) for one strategy.
"""

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Paired runs: each strategy has a joint and delta variant
STRATEGIES = [
    {
        "name": "Non-adaptive",
        "joint": (
            "/common/users/shared/pracsys/adaptive_roa_experiments/dhruv/adaptive_quadrotor2d/outputs/"
            "training_index_0_d2_ratio_0.0_warm_start_False_manifold_False_threshold_mode_dynamic_adapt_iter_10_alpha_0.1_sampling_mode_ranked/"
            "2026-02-17_14-07-33"
        ),
        "delta": (
            "/common/users/shared/pracsys/adaptive_roa_experiments/dhruv/adaptive_quadrotor2d/outputs/"
            "training_index_0_d2_ratio_0.0_warm_start_False_manifold_False_threshold_mode_dynamic_adapt_iter_10_alpha_0.1_sampling_mode_ranked/"
            "2026-02-19_13-50-18"
        ),
    },
    {
        "name": "Adaptive direct",
        "joint": (
            "/common/users/shared/pracsys/adaptive_roa_experiments/dhruv/adaptive_quadrotor2d/outputs/"
            "training_index_0_d2_ratio_1.0_warm_start_False_manifold_False_threshold_mode_dynamic_adapt_iter_10_alpha_0.1_sampling_mode_direct/"
            "2026-02-17_14-12-47"
        ),
        "delta": (
            "/common/users/shared/pracsys/adaptive_roa_experiments/dhruv/adaptive_quadrotor2d/outputs/"
            "training_index_0_d2_ratio_1.0_warm_start_False_manifold_False_threshold_mode_dynamic_adapt_iter_10_alpha_0.1_sampling_mode_direct/"
            "2026-02-19_13-51-06"
        ),
    },
    {
        "name": "Adaptive ranked",
        "joint": (
            "/common/users/shared/pracsys/adaptive_roa_experiments/dhruv/adaptive_quadrotor2d/outputs/"
            "training_index_0_d2_ratio_1.0_warm_start_False_manifold_False_threshold_mode_dynamic_adapt_iter_10_alpha_0.1_sampling_mode_ranked/"
            "2026-02-18_14-09-00"
        ),
        "delta": (
            "/common/users/shared/pracsys/adaptive_roa_experiments/dhruv/adaptive_quadrotor2d/outputs/"
            "training_index_0_d2_ratio_1.0_warm_start_False_manifold_False_threshold_mode_dynamic_adapt_iter_10_alpha_0.1_sampling_mode_ranked/"
            "2026-02-19_13-50-09"
        ),
    },
]

EVAL_SUBDIR = "evaluations/radius_0.3_alpha_0.1_mc_10_batch_100000"
EPOCHS = list(range(10))
OUTPUT_PATH = "/common/home/dm1487/robotics_research/tripods/olympics-classifier/quadrotor2d_opt_mode_comparison.pdf"

JOINT_COLOR = "#2196F3"  # blue
DELTA_COLOR = "#FF5722"  # red-orange


def load_run_data(base_path, eval_subdir, epochs):
    data = {"train_trajectories": [], "epochs": [], "f1": [], "sep": []}
    for e in epochs:
        with open(f"{base_path}/epoch_{e:03d}/artifacts_v2.json") as f:
            root_data = json.load(f)
        with open(f"{base_path}/{eval_subdir}/epoch_{e:03d}/artifacts_v2.json") as f:
            eval_data = json.load(f)
        data["train_trajectories"].append(root_data["train_trajectories"])
        data["epochs"].append(e)
        qps = eval_data["eval_metrics"]["qhat_prediction_sets"]
        data["f1"].append(qps["f1"])
        data["sep"].append(qps["separatrix_pct"] * 100)
    return {k: np.array(v) for k, v in data.items()}


def main():
    # Load all data
    all_data = {}
    for strat in STRATEGIES:
        for mode in ["joint", "delta"]:
            key = f"{strat['name']}_{mode}"
            print(f"Loading {key}...")
            all_data[key] = load_run_data(strat[mode], EVAL_SUBDIR, EPOCHS)

    # 2 rows (F1, Sep%) x 3 cols (strategies)
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))

    for col, strat in enumerate(STRATEGIES):
        joint = all_data[f"{strat['name']}_joint"]
        delta = all_data[f"{strat['name']}_delta"]

        # F1 row
        ax = axes[0, col]
        ax.plot(joint["epochs"], joint["f1"], color=JOINT_COLOR, marker="o",
                linestyle="-", linewidth=2, markersize=6, label="Joint opt")
        ax.plot(delta["epochs"], delta["f1"], color=DELTA_COLOR, marker="s",
                linestyle="--", linewidth=2, markersize=6, label="Delta opt")
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("F1 Score", fontsize=11)
        ax.set_title(f"{strat['name']}: F1", fontsize=13, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # Sep% row
        ax = axes[1, col]
        ax.plot(joint["epochs"], joint["sep"], color=JOINT_COLOR, marker="o",
                linestyle="-", linewidth=2, markersize=6, label="Joint opt")
        ax.plot(delta["epochs"], delta["sep"], color=DELTA_COLOR, marker="s",
                linestyle="--", linewidth=2, markersize=6, label="Delta opt")
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("Separatrix %", fontsize=11)
        ax.set_title(f"{strat['name']}: Sep%", fontsize=13, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Quadrotor2D: Joint vs Delta Optimization", fontsize=16, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved to: {OUTPUT_PATH}")

    # Summary
    print("\n" + "=" * 100)
    print("Summary (Final Epoch)")
    print("=" * 100)
    print(f"{'Strategy':<25} {'Mode':<8} {'Trajs':<7} {'F1':<8} {'Sep%':<8}")
    print("-" * 100)
    for strat in STRATEGIES:
        for mode in ["joint", "delta"]:
            d = all_data[f"{strat['name']}_{mode}"]
            print(f"{strat['name']:<25} {mode:<8} {d['train_trajectories'][-1]:<7} "
                  f"{d['f1'][-1]:<8.4f} {d['sep'][-1]:<8.2f}")


if __name__ == "__main__":
    main()
