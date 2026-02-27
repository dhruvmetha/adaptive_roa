#!/usr/bin/env python3
"""Compare metrics across three adaptive quadrotor2d experiment runs."""

import json
import os

# ── Run definitions ──────────────────────────────────────────────────────────
RUNS = {
    "Run A (d2_ratio=1.0, Feb 14)": (
        "/common/users/shared/pracsys/adaptive_roa_experiments/dhruv/adaptive_quadrotor2d/"
        "outputs/training_index_0_d2_ratio_1.0_warm_start_False_manifold_False_"
        "threshold_mode_dynamic_adapt_iter_10_alpha_0.1_sampling_mode_ranked/"
        "2026-02-14_12-42-40/evaluations/radius_0.3_alpha_0.1_mc_10_batch_100000"
    ),
    "Run B (d2_ratio=0, Feb 13)": (
        "/common/users/shared/pracsys/adaptive_roa_experiments/dhruv/adaptive_quadrotor2d/"
        "outputs/training_index_0_d2_ratio_0_warm_start_False_manifold_False_"
        "threshold_mode_dynamic_adapt_iter_10_alpha_0.1_sampling_mode_ranked/"
        "2026-02-13_23-33-57/evaluations/radius_0.3_alpha_0.1_mc_10_batch_100000"
    ),
    "Run C (d2_ratio=0.0, Feb 14)": (
        "/common/users/shared/pracsys/adaptive_roa_experiments/dhruv/adaptive_quadrotor2d/"
        "outputs/training_index_0_d2_ratio_0.0_warm_start_False_manifold_False_"
        "threshold_mode_dynamic_adapt_iter_10_alpha_0.1_sampling_mode_ranked/"
        "2026-02-14_19-23-42/evaluations/radius_0.3_alpha_0.1_mc_10_batch_100000"
    ),
}

# ── Column definitions ───────────────────────────────────────────────────────
# Each tuple: (header, width, format_spec, extractor_function)
COLUMNS = [
    ("epoch",          5,  "d",   lambda d: d["epoch"]),
    ("lam*",           6,  ".3f", lambda d: d["eval_metrics"]["lambda_delta"]["lambda_star"]),
    ("del*",           6,  ".3f", lambda d: d["eval_metrics"]["lambda_delta"]["delta"]),
    ("accuracy",       9,  ".4f", lambda d: d["eval_metrics"]["lambda_delta"]["accuracy"]),
    ("precision",      9,  ".4f", lambda d: d["eval_metrics"]["lambda_delta"]["precision"]),
    ("recall",         8,  ".4f", lambda d: d["eval_metrics"]["lambda_delta"]["recall"]),
    ("f1",             8,  ".4f", lambda d: d["eval_metrics"]["lambda_delta"]["f1"]),
    ("sep_pct",        8,  ".4f", lambda d: d["eval_metrics"]["lambda_delta"]["separatrix_pct"]),
    ("n_conf",         8,  "d",   lambda d: d["eval_metrics"]["lambda_delta"]["n_confident"]),
    ("n_inv",          8,  "d",   lambda d: d["eval_metrics"]["lambda_delta"]["n_invalid"]),
    ("n_unc",          8,  "d",   lambda d: d["eval_metrics"]["lambda_delta"]["n_uncertain"]),
    ("lam*(ts)",       8,  ".3f", lambda d: d["threshold_state"]["lambda_star"]),
    ("del*(ts)",       8,  ".3f", lambda d: d["threshold_state"]["delta_star"]),
    ("q_hat",          8,  ".4f", lambda d: d["threshold_state"]["q_hat_eval"]),
    ("ep_full",        9,  ".4f", lambda d: d["eval_metrics"]["endpoint_errors"]["full"]["mean"]),
    ("ep_csuc",        9,  ".4f", lambda d: d["eval_metrics"]["endpoint_errors"]["certain_success"]["mean"]),
]


def load_epoch_data(base_dir, epoch_idx):
    """Load artifacts_v2.json for a given epoch."""
    path = os.path.join(base_dir, f"epoch_{epoch_idx:03d}", "artifacts_v2.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def print_header():
    """Print the column header row and separator."""
    parts = []
    for name, width, _, _ in COLUMNS:
        parts.append(f"{name:>{width}}")
    header = " | ".join(parts)
    print(header)
    print("-" * len(header))


def print_row(data):
    """Print a single data row."""
    parts = []
    for name, width, fmt, extractor in COLUMNS:
        try:
            val = extractor(data)
            formatted = f"{val:{fmt}}"
            parts.append(f"{formatted:>{width}}")
        except (KeyError, TypeError):
            parts.append(f"{'N/A':>{width}}")
    print(" | ".join(parts))


def main():
    for run_name, base_dir in RUNS.items():
        print()
        print("=" * 120)
        print(f"  {run_name}")
        print(f"  {base_dir}")
        print("=" * 120)

        # Discover available epochs
        epoch_dirs = sorted([
            d for d in os.listdir(base_dir)
            if d.startswith("epoch_") and os.path.isdir(os.path.join(base_dir, d))
        ])
        epoch_indices = [int(d.split("_")[1]) for d in epoch_dirs]

        if not epoch_indices:
            print("  (no epoch directories found)")
            continue

        print(f"  Available epochs: {epoch_indices}")
        print()

        print_header()

        for idx in epoch_indices:
            data = load_epoch_data(base_dir, idx)
            if data is None:
                print(f"  epoch {idx:03d}: artifacts_v2.json not found")
                continue
            print_row(data)

        print()

    # ── Summary: best F1 per run ─────────────────────────────────────────────
    print()
    print("=" * 120)
    print("  SUMMARY: Best F1 per run")
    print("=" * 120)
    print(f"  {'Run':<40s}  {'Epoch':>5s}  {'F1':>8s}  {'Acc':>9s}  {'Prec':>9s}  {'Rec':>8s}  {'sep_pct':>8s}  {'ep_full':>9s}  {'ep_csuc':>9s}")
    print("  " + "-" * 108)

    for run_name, base_dir in RUNS.items():
        epoch_dirs = sorted([
            d for d in os.listdir(base_dir)
            if d.startswith("epoch_") and os.path.isdir(os.path.join(base_dir, d))
        ])
        epoch_indices = [int(d.split("_")[1]) for d in epoch_dirs]

        best_f1 = -1
        best_data = None
        best_epoch = -1
        for idx in epoch_indices:
            data = load_epoch_data(base_dir, idx)
            if data is None:
                continue
            f1 = data["eval_metrics"]["lambda_delta"]["f1"]
            if f1 > best_f1:
                best_f1 = f1
                best_data = data
                best_epoch = idx

        if best_data is not None:
            ld = best_data["eval_metrics"]["lambda_delta"]
            ee = best_data["eval_metrics"]["endpoint_errors"]
            print(
                f"  {run_name:<40s}  {best_epoch:>5d}  {ld['f1']:>8.4f}  "
                f"{ld['accuracy']:>9.4f}  {ld['precision']:>9.4f}  {ld['recall']:>8.4f}  "
                f"{ld['separatrix_pct']:>8.4f}  {ee['full']['mean']:>9.4f}  "
                f"{ee['certain_success']['mean']:>9.4f}"
            )

    # ── Summary: last epoch per run ──────────────────────────────────────────
    print()
    print("=" * 120)
    print("  SUMMARY: Last epoch (epoch 9) per run")
    print("=" * 120)
    print(f"  {'Run':<40s}  {'Epoch':>5s}  {'F1':>8s}  {'Acc':>9s}  {'Prec':>9s}  {'Rec':>8s}  {'sep_pct':>8s}  {'ep_full':>9s}  {'ep_csuc':>9s}")
    print("  " + "-" * 108)

    for run_name, base_dir in RUNS.items():
        data = load_epoch_data(base_dir, 9)
        if data is not None:
            ld = data["eval_metrics"]["lambda_delta"]
            ee = data["eval_metrics"]["endpoint_errors"]
            print(
                f"  {run_name:<40s}  {9:>5d}  {ld['f1']:>8.4f}  "
                f"{ld['accuracy']:>9.4f}  {ld['precision']:>9.4f}  {ld['recall']:>8.4f}  "
                f"{ld['separatrix_pct']:>8.4f}  {ee['full']['mean']:>9.4f}  "
                f"{ee['certain_success']['mean']:>9.4f}"
            )

    print()


if __name__ == "__main__":
    main()
