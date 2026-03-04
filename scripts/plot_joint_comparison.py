#!/usr/bin/env python3
"""Plot joint optimization comparison: Trajectories vs F1 and Sep% for all three systems.

One PDF per system:
  - Quadrotor 2D: lambda-delta only (F1 + Sep%)
  - CartPole: lambda-delta only, full + zoomed (F1 + Sep%)
  - Quadrotor 3D: lambda-delta + qhat (F1 + Sep%)
"""

import json
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({
    "font.size": 13,
    "axes.labelsize": 14,
    "axes.titlesize": 15,
    "legend.fontsize": 11,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

RUNS = {
    "Pendulum": {
        "non_adaptive": "/common/users/shared/pracsys/adaptive_roa_experiments/dhruv/adaptive_pendulum_dhruv/outputs/training_index_0_d2_ratio_0.0_warm_start_False_threshold_mode_dynamic_adapt_iter_20_alpha_0.1_sampling_mode_ranked/2026-02-20_01-56-20",
        "adaptive_direct": "/common/users/shared/pracsys/adaptive_roa_experiments/dhruv/adaptive_pendulum_dhruv/outputs/training_index_0_d2_ratio_1.0_warm_start_False_threshold_mode_dynamic_adapt_iter_20_alpha_0.1_sampling_mode_direct/2026-02-20_01-56-07",
        "adaptive_ranked": "/common/users/shared/pracsys/adaptive_roa_experiments/dhruv/adaptive_pendulum_dhruv/outputs/training_index_0_d2_ratio_1.0_warm_start_False_threshold_mode_dynamic_adapt_iter_20_alpha_0.1_sampling_mode_ranked/2026-02-20_01-55-55",
        "eval_config": "radius_0.075_alpha_0.1_mc_10_batch_100000",
        "classifier_f1": 0.985,
        "classifier_sep": 0.19,
        "classifier_traj": 1000,
        "deepreach_f1": 0.9752,
        "deepreach_sep": None,
        "morals_f1": 0.939,
        "morals_sep": 7.3,
        "neuromancer_f1": 0.755,
        "neuromancer_sep": 0.0,
        "nongenerative_f1": 0.644,
        "nongenerative_sep": None,
    },
    "Quadrotor 2D": {
        "non_adaptive": "/common/users/shared/pracsys/adaptive_roa_experiments/dhruv/adaptive_quadrotor2d/outputs/training_index_0_d2_ratio_0.0_warm_start_False_manifold_False_threshold_mode_dynamic_adapt_iter_10_alpha_0.1_sampling_mode_ranked/2026-02-17_14-07-33",
        "adaptive_direct": "/common/users/shared/pracsys/adaptive_roa_experiments/dhruv/adaptive_quadrotor2d/outputs/training_index_0_d2_ratio_1.0_warm_start_False_manifold_False_threshold_mode_dynamic_adapt_iter_10_alpha_0.1_sampling_mode_direct/2026-02-17_14-12-47",
        "adaptive_ranked": "/common/users/shared/pracsys/adaptive_roa_experiments/dhruv/adaptive_quadrotor2d/outputs/training_index_0_d2_ratio_1.0_warm_start_False_manifold_False_threshold_mode_dynamic_adapt_iter_10_alpha_0.1_sampling_mode_ranked/2026-02-18_14-09-00",
        "eval_config": "radius_0.3_alpha_0.1_mc_10_batch_100000",
        "classifier_f1": 0.7767,
        "classifier_sep": 0.46,
        "classifier_traj": 12000,
        "deepreach_f1": 0.2548,
        "deepreach_sep": 0.0,
        "max_traj": None,
    },
    "CartPole": {
        "non_adaptive": "/common/users/shared/pracsys/adaptive_roa_experiments/dhruv/adaptive_cartpole_pybullet/outputs/training_index_0_d2_ratio_0.0_warm_start_False_manifold_False_threshold_mode_dynamic_adapt_iter_20_alpha_0.1_sampling_mode_ranked_w_0.9/2026-02-18_14-26-34",
        "adaptive_direct": "/common/users/shared/pracsys/adaptive_roa_experiments/dhruv/adaptive_cartpole_pybullet/outputs/training_index_0_d2_ratio_1.0_warm_start_False_manifold_False_threshold_mode_dynamic_adapt_iter_20_alpha_0.1_sampling_mode_direct_w_0.9/2026-02-18_14-26-32",
        "adaptive_ranked": "/common/users/shared/pracsys/adaptive_roa_experiments/dhruv/adaptive_cartpole_pybullet/outputs/training_index_0_d2_ratio_1.0_warm_start_False_manifold_False_threshold_mode_dynamic_adapt_iter_20_alpha_0.1_sampling_mode_ranked_w_0.9/2026-02-18_19-22-14",
        "eval_config": "radius_0.2_alpha_0.1_mc_10_batch_100000",
        "classifier_f1": 0.945,
        "classifier_sep": 0.05,
        "classifier_traj": 1000,
        "deepreach_f1": 0.7988,
        "deepreach_sep": 0.0,
        "morals_f1": 0.4438,
        "morals_sep": 0.0,
        "neuromancer_f1": 0.734,
        "neuromancer_sep": 0.0,
        "nongenerative_f1": 0.999,
        "nongenerative_sep": 31.6,
        "max_traj": 1000,
    },
    "Quadrotor 3D": {
        "non_adaptive": "/common/users/shared/pracsys/adaptive_roa_experiments/dhruv/adaptive_quadrotor3d/outputs/training_index_0_d2_ratio_0.0_warm_start_False_manifold_False_threshold_mode_dynamic_adapt_iter_10_alpha_0.1_sampling_mode_ranked/2026-02-18_14-25-10",
        "adaptive_direct": "/common/users/shared/pracsys/adaptive_roa_experiments/dhruv/adaptive_quadrotor3d/outputs/training_index_0_d2_ratio_1.0_warm_start_False_manifold_False_threshold_mode_dynamic_adapt_iter_10_alpha_0.1_sampling_mode_direct/2026-02-18_14-25-13",
        "adaptive_ranked": "/common/users/shared/pracsys/adaptive_roa_experiments/dhruv/adaptive_quadrotor3d/outputs/training_index_0_d2_ratio_1.0_warm_start_False_manifold_False_threshold_mode_dynamic_adapt_iter_10_alpha_0.1_sampling_mode_ranked/2026-02-18_14-25-29",
        "eval_config": "radius_0.3_alpha_0.1_mc_10_batch_100000",
        "classifier_f1": 0.7633,
        "classifier_sep": 0.57,
        "classifier_traj": 12000,
        "deepreach_f1": 0.6183,
        "deepreach_sep": 0.0,
        "max_epochs": 8,
    },
}

# Quad3D qhat uses alpha=0.2 (stable, unlike alpha=0.1 which has degenerate F1=0 epochs)
# Adaptive ranked is a different run; direct/non-adaptive are same runs, different eval config
RUNS_QUAD3D_QHAT = {
    "non_adaptive": RUNS["Quadrotor 3D"]["non_adaptive"],
    "adaptive_direct": RUNS["Quadrotor 3D"]["adaptive_direct"],
    "adaptive_ranked": "/common/users/shared/pracsys/adaptive_roa_experiments/dhruv/adaptive_quadrotor3d/outputs/training_index_0_d2_ratio_1.0_warm_start_False_manifold_False_threshold_mode_dynamic_adapt_iter_10_alpha_0.1_sampling_mode_ranked/2026-02-19_13-52-12",
    "eval_config": "radius_0.3_alpha_0.2_mc_10_batch_100000",
    "classifier_f1": 0.7633,
    "classifier_sep": 0.57,
    "classifier_traj": 12000,
    "deepreach_f1": 0.6183,
    "deepreach_sep": 0.0,
    "max_epochs": 8,
}

METHODS = {
    "non_adaptive": {"label": "Non-adaptive", "color": "#1f77b4", "marker": "o", "ls": "-"},
    "adaptive_direct": {"label": "Adaptive (direct)", "color": "#ff7f0e", "marker": "s", "ls": "-"},
    "adaptive_ranked": {"label": "Adaptive (ranked)", "color": "#2ca02c", "marker": "^", "ls": "-"},
}


def extract_data(run_path, eval_config, eval_type, max_traj=None, max_epochs=None):
    """Extract (trajectories, f1, sep_pct) from artifacts."""
    eval_base = os.path.join(run_path, "evaluations", eval_config)
    if not os.path.exists(eval_base):
        print(f"  WARNING: No eval dir at {eval_base}")
        return [], [], []

    epoch_dirs = sorted(d for d in os.listdir(eval_base) if d.startswith("epoch_"))
    if max_epochs is not None:
        epoch_dirs = epoch_dirs[:max_epochs]

    trajs, f1s, seps = [], [], []
    for epoch_dir in epoch_dirs:
        epoch_num = int(epoch_dir.split("_")[1])
        results_file = os.path.join(run_path, f"epoch_{epoch_num:03d}", "results.json")
        if not os.path.exists(results_file):
            continue
        with open(results_file) as f:
            results = json.load(f)
        n_traj = results.get("train_trajectories", None)
        if n_traj is None:
            continue
        if max_traj is not None and n_traj > max_traj:
            break

        artifacts_file = os.path.join(eval_base, epoch_dir, "artifacts_v2.json")
        if not os.path.exists(artifacts_file):
            continue
        with open(artifacts_file) as f:
            artifacts = json.load(f)

        metrics = artifacts["eval_metrics"].get(eval_type, {})
        f1 = metrics.get("f1", None)
        sep = metrics.get("separatrix_pct", None)
        if f1 is not None and sep is not None:
            trajs.append(n_traj)
            f1s.append(f1)
            seps.append(sep * 100)

    return trajs, f1s, seps


def plot_metric(system_cfg, eval_type, ax, metric="f1",
                ylim=None, title="", ylabel=""):
    """Plot a single metric (f1 or sep%) on the given axes."""
    eval_config = system_cfg["eval_config"]
    max_traj = system_cfg.get("max_traj")
    max_epochs = system_cfg.get("max_epochs")

    # Extract all data first to unify the first point (same seed)
    all_data = {}
    for method_key in METHODS:
        run_path = system_cfg[method_key]
        trajs, f1s, seps = extract_data(
            run_path, eval_config, eval_type,
            max_traj=max_traj, max_epochs=max_epochs,
        )
        if trajs:
            all_data[method_key] = (trajs, f1s, seps)

    # Use non-adaptive first point as the shared seed point
    if "non_adaptive" in all_data:
        seed_f1 = all_data["non_adaptive"][1][0]
        seed_sep = all_data["non_adaptive"][2][0]
        for method_key in all_data:
            all_data[method_key][1][0] = seed_f1
            all_data[method_key][2][0] = seed_sep

    for method_key, style in METHODS.items():
        if method_key not in all_data:
            continue
        trajs, f1s, seps = all_data[method_key]
        values = f1s if metric == "f1" else seps
        ax.plot(
            trajs, values,
            label=style["label"],
            color=style["color"],
            marker=style["marker"],
            linewidth=2, markersize=6,
            linestyle=style["ls"],
        )

    # Classifier baseline
    clf_traj = system_cfg.get("classifier_traj", "?")
    if metric == "f1":
        clf_val = system_cfg.get("classifier_f1")
        if clf_val is not None:
            ax.axhline(y=clf_val, color="#d62728", linestyle="--", linewidth=1.8,
                        label=f"Classifier @{clf_traj} (F1={clf_val:.3f})")
    else:
        clf_val = system_cfg.get("classifier_sep")
        if clf_val is not None:
            ax.axhline(y=clf_val, color="#d62728", linestyle="--", linewidth=1.8,
                        label=f"Classifier @{clf_traj} (Sep={clf_val:.2f}%)")

    # DeepReach baseline
    if metric == "f1":
        dr_val = system_cfg.get("deepreach_f1")
        if dr_val is not None:
            ax.axhline(y=dr_val, color="#9467bd", linestyle=":", linewidth=1.8,
                        label=f"DeepReach (F1={dr_val:.3f})")
    else:
        dr_val = system_cfg.get("deepreach_sep")
        if dr_val is not None:
            ax.axhline(y=dr_val, color="#9467bd", linestyle=":", linewidth=1.8,
                        label=f"DeepReach (Sep={dr_val:.2f}%)")

    # MORALS baseline
    if metric == "f1":
        morals_val = system_cfg.get("morals_f1")
        if morals_val is not None:
            ax.axhline(y=morals_val, color="#8c564b", linestyle="-.", linewidth=1.8,
                        label=f"MORALS (F1={morals_val:.3f})")
    else:
        morals_val = system_cfg.get("morals_sep")
        if morals_val is not None:
            ax.axhline(y=morals_val, color="#8c564b", linestyle="-.", linewidth=1.8,
                        label=f"MORALS (Sep={morals_val:.2f}%)")

    # Lyapunov NN (Neural Lyapunov) baseline
    if metric == "f1":
        nm_val = system_cfg.get("neuromancer_f1")
        if nm_val is not None:
            ax.axhline(y=nm_val, color="#e377c2", linestyle="--", linewidth=1.8,
                        label=f"Lyapunov NN (F1={nm_val:.3f})")
    else:
        nm_val = system_cfg.get("neuromancer_sep")
        if nm_val is not None:
            ax.axhline(y=nm_val, color="#e377c2", linestyle="--", linewidth=1.8,
                        label=f"Lyapunov NN (Sep={nm_val:.2f}%)")

    # Non-generative Final State Predictor baseline
    if metric == "f1":
        ng_val = system_cfg.get("nongenerative_f1")
        if ng_val is not None:
            ax.axhline(y=ng_val, color="#7f7f7f", linestyle="-.", linewidth=1.8,
                        label=f"Non-generative (F1={ng_val:.3f})")
    else:
        ng_val = system_cfg.get("nongenerative_sep")
        if ng_val is not None:
            ax.axhline(y=ng_val, color="#7f7f7f", linestyle="-.", linewidth=1.8,
                        label=f"Non-generative (Sep={ng_val:.2f}%)")

    ax.set_xlabel("Number of Trajectories")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15),
              ncol=2, framealpha=0.9, fontsize=9)


def plot_pendulum():
    """Pendulum: 2 rows × 2 cols — full + zoomed, F1 + Sep%."""
    cfg = RUNS["Pendulum"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 13))

    plot_metric(cfg, "lambda_delta", axes[0, 0], metric="f1",
                ylim=(0.4, 1.02),
                title="Pendulum — F1", ylabel="F1 Score")
    plot_metric(cfg, "lambda_delta", axes[0, 1], metric="sep",
                title="Pendulum — Separatrix %", ylabel="Separatrix %")
    plot_metric(cfg, "lambda_delta", axes[1, 0], metric="f1",
                ylim=(0.94, 1.002),
                title="Pendulum — F1 (zoomed)", ylabel="F1 Score")
    plot_metric(cfg, "lambda_delta", axes[1, 1], metric="sep",
                ylim=(0, 15),
                title="Pendulum — Separatrix % (zoomed)", ylabel="Separatrix %")

    fig.suptitle("Pendulum: Joint Optimization (Lambda-Delta)",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95], h_pad=4.0)
    out = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                        "results", "figures", "joint_comparison_pendulum.pdf"))
    fig.savefig(out)
    print(f"Saved: {out}")
    plt.close(fig)


def plot_quadrotor2d():
    """Quad 2D: 1 row, 2 cols — lambda-delta F1 | lambda-delta Sep%."""
    cfg = RUNS["Quadrotor 2D"]
    fig, (ax_f1, ax_sep) = plt.subplots(1, 2, figsize=(14, 8))

    plot_metric(cfg, "lambda_delta", ax_f1, metric="f1",
                ylim=(0.4, 1.02),
                title="Quadrotor 2D — F1", ylabel="F1 Score")
    plot_metric(cfg, "lambda_delta", ax_sep, metric="sep",
                title="Quadrotor 2D — Separatrix %", ylabel="Separatrix %")

    fig.suptitle("Quadrotor 2D: Joint Optimization (Lambda-Delta)",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93], w_pad=3.0)
    out = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                        "results", "figures", "joint_comparison_quadrotor_2d.pdf"))
    fig.savefig(out)
    print(f"Saved: {out}")
    plt.close(fig)


def plot_cartpole():
    """CartPole: 2 rows × 2 cols — full + zoomed, F1 + Sep%."""
    cfg = RUNS["CartPole"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 13))

    plot_metric(cfg, "lambda_delta", axes[0, 0], metric="f1",
                ylim=(0.4, 1.02),
                title="CartPole — F1", ylabel="F1 Score")
    plot_metric(cfg, "lambda_delta", axes[0, 1], metric="sep",
                title="CartPole — Separatrix %", ylabel="Separatrix %")
    plot_metric(cfg, "lambda_delta", axes[1, 0], metric="f1",
                ylim=(0.93, 1.002),
                title="CartPole — F1 (zoomed)", ylabel="F1 Score")
    plot_metric(cfg, "lambda_delta", axes[1, 1], metric="sep",
                ylim=(0, 40),
                title="CartPole — Separatrix % (zoomed)", ylabel="Separatrix %")

    fig.suptitle("CartPole: Joint Optimization (Lambda-Delta)",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95], h_pad=4.0)
    out = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                        "results", "figures", "joint_comparison_cartpole.pdf"))
    fig.savefig(out)
    print(f"Saved: {out}")
    plt.close(fig)


def plot_quadrotor3d_lambda_delta():
    """Quad 3D lambda-delta: 2 rows × 2 cols — full + zoomed, F1 + Sep%."""
    cfg = RUNS["Quadrotor 3D"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 13))

    plot_metric(cfg, "lambda_delta", axes[0, 0], metric="f1",
                ylim=(0.4, 1.02),
                title="Quadrotor 3D — F1", ylabel="F1 Score")
    plot_metric(cfg, "lambda_delta", axes[0, 1], metric="sep",
                title="Quadrotor 3D — Separatrix %", ylabel="Separatrix %")
    plot_metric(cfg, "lambda_delta", axes[1, 0], metric="f1",
                ylim=(0.82, 0.96),
                title="Quadrotor 3D — F1 (zoomed)", ylabel="F1 Score")
    plot_metric(cfg, "lambda_delta", axes[1, 1], metric="sep",
                ylim=(20, 50),
                title="Quadrotor 3D — Separatrix % (zoomed)", ylabel="Separatrix %")

    fig.suptitle("Quadrotor 3D: Joint Optimization (Lambda-Delta)",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95], h_pad=4.0)
    out = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                        "results", "figures", "joint_comparison_quadrotor_3d_lambda_delta.pdf"))
    fig.savefig(out)
    print(f"Saved: {out}")
    plt.close(fig)


def plot_quadrotor3d_qhat():
    """Quad 3D qhat (alpha=0.2): 2 rows × 2 cols — full + zoomed, F1 + Sep%."""
    cfg = RUNS_QUAD3D_QHAT
    fig, axes = plt.subplots(2, 2, figsize=(14, 13))

    plot_metric(cfg, "qhat_prediction_sets", axes[0, 0], metric="f1",
                ylim=(0.0, 1.02),
                title="Quadrotor 3D — F1", ylabel="F1 Score")
    plot_metric(cfg, "qhat_prediction_sets", axes[0, 1], metric="sep",
                title="Quadrotor 3D — Separatrix %", ylabel="Separatrix %")
    plot_metric(cfg, "qhat_prediction_sets", axes[1, 0], metric="f1",
                ylim=(0.82, 0.98),
                title="Quadrotor 3D — F1 (zoomed)", ylabel="F1 Score")
    plot_metric(cfg, "qhat_prediction_sets", axes[1, 1], metric="sep",
                ylim=(25, 65),
                title="Quadrotor 3D — Separatrix % (zoomed)", ylabel="Separatrix %")

    fig.suptitle("Quadrotor 3D: Joint Optimization (Q-hat, α=0.2)",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95], h_pad=4.0)
    out = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                        "results", "figures", "joint_comparison_quadrotor_3d_qhat.pdf"))
    fig.savefig(out)
    print(f"Saved: {out}")
    plt.close(fig)


def main():
    plot_pendulum()
    plot_quadrotor2d()
    plot_cartpole()
    plot_quadrotor3d_lambda_delta()
    plot_quadrotor3d_qhat()


if __name__ == "__main__":
    main()
