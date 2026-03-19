#!/usr/bin/env python3
"""Extract conformal coverage, prediction set sizes, and endpoint errors
from the LAST evaluated epoch for each system/run."""

import json
import os
from pathlib import Path

BASE = "/common/users/shared/pracsys/adaptive_roa_experiments/dhruv"

# Define all runs: (system_label, run_label, training_dir, eval_config, last_epoch)
RUNS = [
    # ─── Pendulum (no manifold_False in dir names) ───
    ("Pendulum", "Non-adaptive (ranked)",
     f"{BASE}/adaptive_pendulum_dhruv/outputs/training_index_0_d2_ratio_0.0_warm_start_False_threshold_mode_dynamic_adapt_iter_20_alpha_0.1_sampling_mode_ranked/2026-02-20_01-56-20",
     "radius_0.075_alpha_0.1_mc_20_batch_100000", "epoch_019"),
    ("Pendulum", "Adaptive direct (d2=1.0)",
     f"{BASE}/adaptive_pendulum_dhruv/outputs/training_index_0_d2_ratio_1.0_warm_start_False_threshold_mode_dynamic_adapt_iter_20_alpha_0.1_sampling_mode_direct/2026-02-20_01-56-07",
     "radius_0.075_alpha_0.1_mc_20_batch_100000", "epoch_019"),
    ("Pendulum", "Adaptive ranked (d2=1.0)",
     f"{BASE}/adaptive_pendulum_dhruv/outputs/training_index_0_d2_ratio_1.0_warm_start_False_threshold_mode_dynamic_adapt_iter_20_alpha_0.1_sampling_mode_ranked/2026-02-20_01-55-55",
     "radius_0.075_alpha_0.1_mc_20_batch_100000", "epoch_019"),
    ("Pendulum", "Adaptive direct (d2=0.75)",
     f"{BASE}/adaptive_pendulum_dhruv/outputs/training_index_0_d2_ratio_0.75_warm_start_False_threshold_mode_dynamic_adapt_iter_20_alpha_0.1_sampling_mode_direct/2026-02-25_12-11-22",
     "radius_0.075_alpha_0.1_mc_20_batch_100000", "epoch_019"),
    ("Pendulum", "Adaptive direct (d2=0.5)",
     f"{BASE}/adaptive_pendulum_dhruv/outputs/training_index_0_d2_ratio_0.5_warm_start_False_threshold_mode_dynamic_adapt_iter_20_alpha_0.1_sampling_mode_direct/2026-02-25_12-11-15",
     "radius_0.075_alpha_0.1_mc_20_batch_100000", "epoch_019"),
    ("Pendulum", "Manifold (d2=0.5, opt=loss)",
     f"{BASE}/adaptive_pendulum_dhruv/outputs/training_index_0_d2_ratio_0.5_warm_start_False_threshold_mode_dynamic_adapt_iter_20_alpha_0.1_sampling_mode_direct/2026-02-26_15-22-40",
     "radius_0.075_alpha_0.1_mc_20_batch_100000", "epoch_019"),
    ("Pendulum", "Non-adaptive (direct)",
     f"{BASE}/adaptive_pendulum_dhruv/outputs/training_index_0_d2_ratio_0.0_warm_start_False_threshold_mode_dynamic_adapt_iter_20_alpha_0.1_sampling_mode_direct/2026-02-28_13-15-25",
     "radius_0.075_alpha_0.1_mc_20_batch_100000", "epoch_019"),
    ("Pendulum", "Adaptive direct (d2=1.0, v2)",
     f"{BASE}/adaptive_pendulum_dhruv/outputs/training_index_0_d2_ratio_1.0_warm_start_False_threshold_mode_dynamic_adapt_iter_20_alpha_0.1_sampling_mode_direct/2026-03-01_11-11-14",
     "radius_0.075_alpha_0.1_mc_20_batch_100000", "epoch_019"),

    # ─── CartPole ───
    ("CartPole", "Non-adaptive (d2=0)",
     f"{BASE}/adaptive_cartpole_pybullet/outputs/training_index_0_d2_ratio_0.0_warm_start_False_manifold_False_threshold_mode_dynamic_adapt_iter_20_alpha_0.1_sampling_mode_ranked_w_0.9/2026-02-18_14-26-34",
     "radius_0.2_alpha_0.1_mc_20_batch_100000", "epoch_019"),
    ("CartPole", "Adaptive direct (d2=1.0)",
     f"{BASE}/adaptive_cartpole_pybullet/outputs/training_index_0_d2_ratio_1.0_warm_start_False_manifold_False_threshold_mode_dynamic_adapt_iter_20_alpha_0.1_sampling_mode_direct_w_0.9/2026-02-18_14-26-32",
     "radius_0.2_alpha_0.1_mc_20_batch_100000", "epoch_019"),
    ("CartPole", "Adaptive ranked (d2=1.0)",
     f"{BASE}/adaptive_cartpole_pybullet/outputs/training_index_0_d2_ratio_1.0_warm_start_False_manifold_False_threshold_mode_dynamic_adapt_iter_20_alpha_0.1_sampling_mode_ranked_w_0.9/2026-02-18_19-22-14",
     "radius_0.2_alpha_0.1_mc_20_batch_100000", "epoch_019"),
    ("CartPole", "Adaptive direct (d2=0.75)",
     f"{BASE}/adaptive_cartpole_pybullet/outputs/training_index_0_d2_ratio_0.75_warm_start_False_manifold_False_threshold_mode_dynamic_adapt_iter_20_alpha_0.1_sampling_mode_direct_w_0.9/2026-02-25_12-10-13",
     "radius_0.2_alpha_0.1_mc_20_batch_100000", "epoch_019"),
    ("CartPole", "Adaptive direct (d2=0.5)",
     f"{BASE}/adaptive_cartpole_pybullet/outputs/training_index_0_d2_ratio_0.5_warm_start_False_manifold_False_threshold_mode_dynamic_adapt_iter_20_alpha_0.1_sampling_mode_direct_w_0.9/2026-02-25_12-10-05",
     "radius_0.2_alpha_0.1_mc_20_batch_100000", "epoch_019"),

    # ─── Quadrotor 2D (Large) ───
    ("Quad2D", "Non-adaptive (d2=0)",
     f"{BASE}/adaptive_quadrotor2d/outputs/training_index_0_d2_ratio_0.0_warm_start_False_manifold_False_threshold_mode_dynamic_adapt_iter_10_alpha_0.1_sampling_mode_direct/2026-02-28_10-17-55",
     "radius_0.3_alpha_0.1_mc_10_batch_100000_calk_1000", "epoch_009"),
    ("Quad2D", "Adaptive direct (d2=0.5)",
     f"{BASE}/adaptive_quadrotor2d/outputs/training_index_0_d2_ratio_0.5_warm_start_False_manifold_False_threshold_mode_dynamic_adapt_iter_10_alpha_0.1_sampling_mode_direct/2026-02-28_10-17-39",
     "radius_0.3_alpha_0.1_mc_10_batch_100000_calk_1000", "epoch_009"),
    ("Quad2D", "Adaptive direct (d2=0.75)",
     f"{BASE}/adaptive_quadrotor2d/outputs/training_index_0_d2_ratio_0.75_warm_start_False_manifold_False_threshold_mode_dynamic_adapt_iter_10_alpha_0.1_sampling_mode_direct/2026-02-28_10-16-05",
     "radius_0.3_alpha_0.1_mc_10_batch_100000_calk_1000", "epoch_009"),
    ("Quad2D", "Adaptive direct (d2=1.0)",
     f"{BASE}/adaptive_quadrotor2d/outputs/training_index_0_d2_ratio_1.0_warm_start_False_manifold_False_threshold_mode_dynamic_adapt_iter_10_alpha_0.1_sampling_mode_direct/2026-02-28_10-05-32",
     "radius_0.3_alpha_0.1_mc_10_batch_100000_calk_1000", "epoch_009"),
    ("Quad2D", "Adaptive direct (d2=1.0, fixed)",
     f"{BASE}/adaptive_quadrotor2d/outputs/training_index_0_d2_ratio_1.0_warm_start_False_manifold_False_threshold_mode_dynamic_adapt_iter_10_alpha_0.1_sampling_mode_direct/2026-02-28_12-09-02",
     "radius_0.3_alpha_0.1_mc_10_batch_100000_calk_1000", "epoch_009"),

    # ─── Quadrotor 3D (Large) ───
    ("Quad3D", "Non-adaptive (d2=0, direct)",
     f"{BASE}/adaptive_quadrotor3d/outputs/training_index_0_d2_ratio_0.0_warm_start_False_manifold_False_threshold_mode_dynamic_adapt_iter_15_alpha_0.1_sampling_mode_direct/2026-02-28_11-49-54",
     "radius_0.3_alpha_0.1_mc_10_batch_100000_calk_1000", "epoch_014"),
    ("Quad3D", "Adaptive direct (d2=0.5)",
     f"{BASE}/adaptive_quadrotor3d/outputs/training_index_0_d2_ratio_0.5_warm_start_False_manifold_False_threshold_mode_dynamic_adapt_iter_15_alpha_0.1_sampling_mode_direct/2026-02-28_11-49-55",
     "radius_0.3_alpha_0.1_mc_10_batch_100000_calk_1000", "epoch_014"),
    ("Quad3D", "Adaptive direct (d2=0.75)",
     f"{BASE}/adaptive_quadrotor3d/outputs/training_index_0_d2_ratio_0.75_warm_start_False_manifold_False_threshold_mode_dynamic_adapt_iter_15_alpha_0.1_sampling_mode_direct/2026-02-28_11-50-16",
     "radius_0.3_alpha_0.1_mc_10_batch_100000_calk_1000", "epoch_014"),
    ("Quad3D", "Adaptive direct (d2=1.0, 20iter)",
     f"{BASE}/adaptive_quadrotor3d/outputs/training_index_0_d2_ratio_1.0_warm_start_False_manifold_False_threshold_mode_dynamic_adapt_iter_20_alpha_0.1_sampling_mode_direct/2026-02-26_12-02-36",
     "radius_0.3_alpha_0.1_mc_10_batch_100000_calk_1000", "epoch_015"),
    ("Quad3D", "Adaptive ranked (d2=1.0, 20iter)",
     f"{BASE}/adaptive_quadrotor3d/outputs/training_index_0_d2_ratio_1.0_warm_start_False_manifold_False_threshold_mode_dynamic_adapt_iter_20_alpha_0.1_sampling_mode_ranked/2026-02-26_12-02-36",
     "radius_0.3_alpha_0.1_mc_10_batch_100000_calk_1000", "epoch_015"),
    ("Quad3D", "Adaptive direct (d2=1.0, fixed)",
     f"{BASE}/adaptive_quadrotor3d/outputs/training_index_0_d2_ratio_1.0_warm_start_False_manifold_False_threshold_mode_dynamic_adapt_iter_15_alpha_0.1_sampling_mode_direct/2026-02-28_12-01-36",
     "radius_0.3_alpha_0.1_mc_10_batch_100000_calk_1000", "epoch_014"),
]


def safe_get(d, *keys, default="—"):
    """Nested dict access with default."""
    for k in keys:
        if isinstance(d, dict):
            d = d.get(k, None)
        else:
            return default
        if d is None:
            return default
    return d


def fmt(val, decimals=4):
    if isinstance(val, (int, float)):
        return f"{val:.{decimals}f}"
    return str(val)


def extract_run(system, label, training_dir, eval_config, last_epoch):
    """Extract metrics from the last epoch's artifacts_v2.json."""
    artifacts_path = os.path.join(
        training_dir, "evaluations", eval_config, last_epoch, "artifacts_v2.json"
    )

    if not os.path.exists(artifacts_path):
        # Try to find the actual last epoch
        eval_dir = os.path.join(training_dir, "evaluations", eval_config)
        if os.path.exists(eval_dir):
            epochs = sorted([d for d in os.listdir(eval_dir) if d.startswith("epoch_")])
            if epochs:
                last_epoch = epochs[-1]
                artifacts_path = os.path.join(eval_dir, last_epoch, "artifacts_v2.json")
            else:
                return None
        else:
            return None

    if not os.path.exists(artifacts_path):
        return None

    with open(artifacts_path) as f:
        data = json.load(f)

    em = data.get("eval_metrics", {})

    # --- Lambda-delta classification ---
    ld = em.get("lambda_delta", {})

    # --- Q-hat prediction sets ---
    qhat = em.get("qhat_prediction_sets", {})

    # --- Conservative ---
    cons_ld = em.get("conservative_lambda_delta", {})
    cons_qhat = em.get("conservative_qhat", {})

    # --- Endpoint errors ---
    ee = em.get("endpoint_errors", {})
    ee_qhat = em.get("endpoint_errors_qhat_regions", {})

    # --- MC sample errors ---
    mc_err = em.get("mc_sample_errors", {})

    # --- Threshold state ---
    ts = data.get("threshold_state", {})

    result = {
        "system": system,
        "label": label,
        "epoch": last_epoch,
        # Threshold
        "lambda_star": safe_get(ts, "lambda_star"),
        "delta_star": safe_get(ts, "delta_star"),
        "q_hat": safe_get(ts, "q_hat_eval", default=safe_get(em, "q_hat")),
        # Lambda-delta
        "ld_f1": safe_get(ld, "f1"),
        "ld_precision": safe_get(ld, "precision"),
        "ld_recall": safe_get(ld, "recall"),
        "ld_specificity": safe_get(ld, "specificity"),
        "ld_sep_pct": safe_get(ld, "separatrix_pct"),
        "ld_n_confident": safe_get(ld, "n_confident"),
        "ld_n_invalid": safe_get(ld, "n_invalid"),
        "ld_n_uncertain": safe_get(ld, "n_uncertain"),
        # Q-hat prediction sets
        "qhat_f1": safe_get(qhat, "f1"),
        "qhat_precision": safe_get(qhat, "precision"),
        "qhat_recall": safe_get(qhat, "recall"),
        "qhat_specificity": safe_get(qhat, "specificity"),
        "qhat_coverage": safe_get(qhat, "coverage"),
        "qhat_avg_set_size": safe_get(qhat, "avg_set_size"),
        "qhat_median_set_size": safe_get(qhat, "median_set_size"),
        "qhat_sep_pct": safe_get(qhat, "separatrix_pct"),
        "qhat_n_pred_success": safe_get(qhat, "n_pred_success"),
        "qhat_n_pred_failure": safe_get(qhat, "n_pred_failure"),
        # Conservative
        "cons_ld_f1": safe_get(cons_ld, "f1"),
        "cons_qhat_f1": safe_get(cons_qhat, "f1"),
        # Endpoint errors (lambda-delta regions)
        "ee_full_mean": safe_get(ee, "full", "mean"),
        "ee_full_median": safe_get(ee, "full", "median"),
        "ee_full_n": safe_get(ee, "full", "n_points"),
        "ee_success_mean": safe_get(ee, "certain_success", "mean"),
        "ee_success_median": safe_get(ee, "certain_success", "median"),
        "ee_success_n": safe_get(ee, "certain_success", "n_points"),
        "ee_failure_mean": safe_get(ee, "certain_failure", "mean"),
        "ee_failure_median": safe_get(ee, "certain_failure", "median"),
        "ee_failure_n": safe_get(ee, "certain_failure", "n_points"),
        "ee_uncertain_mean": safe_get(ee, "uncertain", "mean"),
        "ee_uncertain_median": safe_get(ee, "uncertain", "median"),
        "ee_uncertain_n": safe_get(ee, "uncertain", "n_points"),
        "ee_invalid_mean": safe_get(ee, "invalid", "mean"),
        "ee_invalid_median": safe_get(ee, "invalid", "median"),
        "ee_invalid_n": safe_get(ee, "invalid", "n_points"),
        # Endpoint errors (qhat regions)
        "ee_qhat_full_mean": safe_get(ee_qhat, "full", "mean"),
        "ee_qhat_full_median": safe_get(ee_qhat, "full", "median"),
        "ee_qhat_success_mean": safe_get(ee_qhat, "certain_success", "mean"),
        "ee_qhat_success_median": safe_get(ee_qhat, "certain_success", "median"),
        "ee_qhat_failure_mean": safe_get(ee_qhat, "certain_failure", "mean"),
        "ee_qhat_failure_median": safe_get(ee_qhat, "certain_failure", "median"),
        # MC sample variability
        "mc_full_mean": safe_get(mc_err, "full", "mean"),
        "mc_full_median": safe_get(mc_err, "full", "median"),
        "mc_full_p90": safe_get(mc_err, "full", "p90"),
        # Total points
        "n_total": safe_get(em, "n_total"),
        "num_mc_samples": safe_get(em, "num_mc_samples"),
    }
    return result


def print_system_table(system_name, results):
    """Print formatted tables for a single system."""
    sys_results = [r for r in results if r and r["system"] == system_name]
    if not sys_results:
        print(f"\n## {system_name} — NO DATA\n")
        return

    print(f"\n## {system_name} (Last Epoch)\n")

    # --- Table 1: Classification + Conformal ---
    print("### Classification & Conformal Coverage\n")
    print("| Run | Epoch | λ* | δ* | q̂ | F1 (λδ) | Sep% | Coverage | Avg Set | Med Set | Cons F1 (λδ) | Cons F1 (q̂) |")
    print("|-----|-------|----|----|-----|---------|------|----------|---------|---------|-------------|-------------|")
    for r in sys_results:
        print(f"| {r['label']} | {r['epoch']} "
              f"| {fmt(r['lambda_star'])} | {fmt(r['delta_star'])} | {fmt(r['q_hat'])} "
              f"| {fmt(r['ld_f1'])} | {fmt(r['ld_sep_pct'], 2)}% "
              f"| {fmt(r['qhat_coverage'])} | {fmt(r['qhat_avg_set_size'], 2)} | {fmt(r['qhat_median_set_size'], 1)} "
              f"| {fmt(r['cons_ld_f1'])} | {fmt(r['cons_qhat_f1'])} |")

    # --- Table 2: Prediction Set Details ---
    print("\n### Prediction Set Details\n")
    print("| Run | q̂ F1 | q̂ Prec | q̂ Recall | q̂ Spec | N pred success | N pred failure | N total |")
    print("|-----|-------|--------|----------|--------|----------------|----------------|---------|")
    for r in sys_results:
        print(f"| {r['label']} "
              f"| {fmt(r['qhat_f1'])} | {fmt(r['qhat_precision'])} | {fmt(r['qhat_recall'])} | {fmt(r['qhat_specificity'])} "
              f"| {r['qhat_n_pred_success']} | {r['qhat_n_pred_failure']} | {r['n_total']} |")

    # --- Table 3: Endpoint Errors (lambda-delta regions) ---
    print("\n### Endpoint Errors (λδ regions)\n")
    print("| Run | Full Mean | Full Med | Success Mean | Success Med | Failure Mean | Failure Med | Uncertain Mean | Uncertain Med |")
    print("|-----|-----------|----------|-------------|-------------|-------------|-------------|----------------|---------------|")
    for r in sys_results:
        print(f"| {r['label']} "
              f"| {fmt(r['ee_full_mean'])} | {fmt(r['ee_full_median'])} "
              f"| {fmt(r['ee_success_mean'])} | {fmt(r['ee_success_median'])} "
              f"| {fmt(r['ee_failure_mean'])} | {fmt(r['ee_failure_median'])} "
              f"| {fmt(r['ee_uncertain_mean'])} | {fmt(r['ee_uncertain_median'])} |")

    # --- Table 4: Endpoint Errors (q-hat regions) ---
    print("\n### Endpoint Errors (q̂ regions)\n")
    print("| Run | Full Mean | Full Med | Success Mean | Success Med | Failure Mean | Failure Med |")
    print("|-----|-----------|----------|-------------|-------------|-------------|-------------|")
    for r in sys_results:
        print(f"| {r['label']} "
              f"| {fmt(r['ee_qhat_full_mean'])} | {fmt(r['ee_qhat_full_median'])} "
              f"| {fmt(r['ee_qhat_success_mean'])} | {fmt(r['ee_qhat_success_median'])} "
              f"| {fmt(r['ee_qhat_failure_mean'])} | {fmt(r['ee_qhat_failure_median'])} |")

    # --- Table 5: MC Sample Variability ---
    print("\n### MC Sample Variability\n")
    print("| Run | MC Full Mean | MC Full Median | MC Full P90 | N MC samples |")
    print("|-----|-------------|----------------|-------------|-------------|")
    for r in sys_results:
        print(f"| {r['label']} "
              f"| {fmt(r['mc_full_mean'])} | {fmt(r['mc_full_median'])} | {fmt(r['mc_full_p90'])} "
              f"| {r['num_mc_samples']} |")


def main():
    results = []
    for system, label, tdir, eval_cfg, last_ep in RUNS:
        r = extract_run(system, label, tdir, eval_cfg, last_ep)
        if r is None:
            print(f"WARNING: No data for {system} / {label} at {tdir}")
        results.append(r)

    for sys_name in ["Pendulum", "CartPole", "Quad2D", "Quad3D"]:
        print_system_table(sys_name, results)

    # Also dump raw JSON for programmatic use
    valid = [r for r in results if r is not None]
    out_path = os.path.join(os.path.dirname(__file__), "..", "results", "last_epoch_details.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(valid, f, indent=2, default=str)
    print(f"\n\nRaw JSON saved to: {out_path}")


if __name__ == "__main__":
    main()
