#!/usr/bin/env python3
"""
Extract evaluation results from an adaptive ROA training run and output
markdown tables ready to paste into results documentation.

Usage:
    # From a training run directory (auto-discovers evaluations)
    python scripts/extract_results_md.py /path/to/training/run

    # From a specific evaluations folder
    python scripts/extract_results_md.py /path/to/training/run/evaluations/radius_0.3_alpha_0.1_mc_10_batch_100000

    # Specify metric type (default: lambda_delta)
    python scripts/extract_results_md.py /path/to/run --metric lambda_delta

    # Show all available metric types
    python scripts/extract_results_md.py /path/to/run --metric all

    # Custom label for the table header
    python scripts/extract_results_md.py /path/to/run --label "Adaptive direct (d2=0.5)"
"""

import argparse
import json
import re
import sys
from pathlib import Path


METRIC_TYPES = [
    "lambda_delta", "qhat_prediction_sets", "lambda_only", "fixed_threshold",
    "conservative_lambda_delta", "conservative_qhat",
]


def parse_run_info(training_dir: Path) -> dict:
    """Extract run parameters from the directory name.

    Training dirs typically look like:
        .../training_index_0_d2_ratio_0.5_..._sampling_mode_direct_w_0.9/2026-02-25_12-10-05
    So we search the full path string for parameters.
    """
    # Use the full path string to find params — they may be in any ancestor dir
    path_str = str(training_dir)

    info = {}
    # Parse d2_ratio
    m = re.search(r"d2_ratio_([\d.]+)", path_str)
    if m:
        info["d2_ratio"] = float(m.group(1))

    # Parse sampling_mode
    m = re.search(r"sampling_mode_(\w+?)(?:_w_|/|$)", path_str)
    if m:
        info["sampling_mode"] = m.group(1)

    # Parse adapt_iter
    m = re.search(r"adapt_iter_(\d+)", path_str)
    if m:
        info["adapt_iter"] = int(m.group(1))

    # Parse alpha
    m = re.search(r"(?<![a-z_])alpha_([\d.]+)", path_str)
    if m:
        info["alpha"] = float(m.group(1))

    return info


def find_eval_dir(training_dir: Path) -> Path | None:
    """Find the evaluations directory, trying various locations."""
    evals_dir = training_dir / "evaluations"
    if evals_dir.exists():
        # Find subdirectories (e.g., radius_0.3_alpha_0.1_mc_10_batch_100000)
        subdirs = sorted([d for d in evals_dir.iterdir() if d.is_dir()])
        if len(subdirs) == 1:
            return subdirs[0]
        elif len(subdirs) > 1:
            print(f"Multiple evaluation configs found:", file=sys.stderr)
            for d in subdirs:
                print(f"  - {d.name}", file=sys.stderr)
            print(f"Using: {subdirs[-1].name}", file=sys.stderr)
            return subdirs[-1]
    return None


def get_trajectory_counts(training_dir: Path) -> dict[int, int]:
    """Get trajectory counts per epoch from results.json files."""
    traj_map = {}
    for epoch_dir in training_dir.iterdir():
        if epoch_dir.is_dir() and re.match(r"epoch_\d+", epoch_dir.name):
            results_file = epoch_dir / "results.json"
            if results_file.exists():
                try:
                    with open(results_file) as f:
                        data = json.load(f)
                    epoch_num = int(re.search(r"\d+", epoch_dir.name).group())
                    traj_map[epoch_num] = data.get("train_trajectories", 0)
                except (json.JSONDecodeError, PermissionError):
                    pass
    return traj_map


def extract_epoch_metrics(artifacts_path: Path, metric_type: str) -> dict | None:
    """Extract metrics from a single epoch's artifacts_v2.json."""
    try:
        with open(artifacts_path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, PermissionError) as e:
        print(f"  Warning: {e} for {artifacts_path}", file=sys.stderr)
        return None

    epoch = data.get("epoch", 0)
    threshold_state = data.get("threshold_state", {})
    eval_metrics = data.get("eval_metrics", {})

    metrics = eval_metrics.get(metric_type, {})
    if not metrics:
        return None

    # Also extract cross-metric fields (conservative F1, coverage) regardless of metric_type
    conservative_key = f"conservative_{metric_type}" if not metric_type.startswith("conservative_") else None
    conservative_metrics = eval_metrics.get(conservative_key, {}) if conservative_key else {}
    qhat_metrics = eval_metrics.get("qhat_prediction_sets", {})

    # Extract endpoint errors
    endpoint_errors = eval_metrics.get("endpoint_errors", {})
    endpoint_error_data = {}
    if endpoint_errors:
        endpoint_error_data["component_names"] = endpoint_errors.get("component_names", [])
        for region in ["full", "certain", "certain_success", "certain_failure", "uncertain", "invalid"]:
            region_data = endpoint_errors.get(region, {})
            if region_data:
                endpoint_error_data[region] = {
                    "n_points": region_data.get("n_points", 0),
                    "mean": region_data.get("mean", 0),
                    "median": region_data.get("median", 0),
                    "variance": region_data.get("variance", 0),
                    "mean_per_dim": region_data.get("mean_per_dim", []),
                    "median_per_dim": region_data.get("median_per_dim", []),
                }

    return {
        "epoch": epoch,
        "f1": metrics.get("f1", 0),
        "precision": metrics.get("precision", 0),
        "recall": metrics.get("recall", 0),
        "specificity": metrics.get("specificity", 0),
        "separatrix_pct": metrics.get("separatrix_pct", 0),
        "accuracy": metrics.get("accuracy", 0),
        "n_confident": metrics.get("n_confident", 0),
        "n_invalid": metrics.get("n_invalid", 0),
        "n_uncertain": metrics.get("n_uncertain", 0),
        "lambda_star": threshold_state.get("lambda_star", 0),
        "delta_star": threshold_state.get("delta_star", 0),
        "q_hat": threshold_state.get("q_hat_eval", 0),
        "is_conservative": metric_type.startswith("conservative_"),
        "conservative_f1": conservative_metrics.get("f1"),
        "coverage": qhat_metrics.get("coverage"),
        "avg_set_size": qhat_metrics.get("avg_set_size"),
        "median_set_size": qhat_metrics.get("median_set_size"),
        "endpoint_errors": endpoint_error_data,
    }


def extract_all_epochs(eval_dir: Path, metric_type: str) -> list[dict]:
    """Extract metrics from all epochs in an evaluation directory."""
    epoch_dirs = sorted(
        [d for d in eval_dir.iterdir() if d.is_dir() and re.match(r"epoch_\d+", d.name)],
        key=lambda x: int(re.search(r"\d+", x.name).group()),
    )

    results = []
    for epoch_dir in epoch_dirs:
        artifacts = epoch_dir / "artifacts_v2.json"
        if artifacts.exists():
            m = extract_epoch_metrics(artifacts, metric_type)
            if m is not None:
                results.append(m)

    return results


def format_markdown_table(
    epochs: list[dict],
    traj_map: dict[int, int],
    label: str | None = None,
    metric_type: str = "lambda_delta",
) -> str:
    """Format results as a markdown table."""
    lines = []

    is_conservative = epochs and epochs[0].get("is_conservative", False)
    has_cons_f1 = any(m.get("conservative_f1") is not None for m in epochs)
    has_coverage = any(m.get("coverage") is not None for m in epochs)
    has_set_size = any(m.get("avg_set_size") is not None for m in epochs)

    # Header comment
    if label:
        lines.append(f"#### {label}")
    lines.append("")

    if is_conservative:
        lines.append("| Epoch | Trajectories | F1 | Precision | Recall | Specificity |")
        lines.append("|-------|-------------|------|-----------|--------|-------------|")
    else:
        header = "| Epoch | Trajectories | F1 | Sep%"
        sep = "|-------|-------------|------|------"
        if has_cons_f1:
            header += " | Cons. F1"
            sep += "|----------"
        if has_coverage:
            header += " | Coverage"
            sep += "|----------"
        if has_set_size:
            header += " | Avg Set | Med Set"
            sep += "|---------|---------"
        header += " | Precision | Recall | Specificity |"
        sep += "|-----------|--------|-------------|"
        lines.append(header)
        lines.append(sep)

    for m in epochs:
        epoch = m["epoch"]
        traj = traj_map.get(epoch, "N/A")
        if is_conservative:
            lines.append(
                f"| {epoch} | {traj} | {m['f1']:.4f} | "
                f"{m['precision']:.4f} | {m['recall']:.4f} | {m['specificity']:.4f} |"
            )
        else:
            sep_pct = m["separatrix_pct"] * 100
            row = f"| {epoch} | {traj} | {m['f1']:.4f} | {sep_pct:.2f}%"
            if has_cons_f1:
                cf1 = m.get("conservative_f1")
                row += f" | {cf1:.4f}" if cf1 is not None else " | —"
            if has_coverage:
                cov = m.get("coverage")
                row += f" | {cov:.4f}" if cov is not None else " | —"
            if has_set_size:
                avg_ss = m.get("avg_set_size")
                med_ss = m.get("median_set_size")
                row += f" | {avg_ss:.2f}" if avg_ss is not None else " | —"
                row += f" | {med_ss:.1f}" if med_ss is not None else " | —"
            row += f" | {m['precision']:.4f} | {m['recall']:.4f} | {m['specificity']:.4f} |"
            lines.append(row)

    return "\n".join(lines)


def format_summary_line(
    epochs: list[dict],
    traj_map: dict[int, int],
    label: str,
    target_traj: int | None = None,
) -> str:
    """Format a single summary line for the final epoch or a target trajectory count."""
    if target_traj is not None:
        # Find the epoch closest to the target trajectory count
        best = None
        for m in epochs:
            traj = traj_map.get(m["epoch"], 0)
            if traj == target_traj:
                best = m
                best_traj = traj
                break
            if best is None or abs(traj - target_traj) < abs(best_traj - target_traj):
                best = m
                best_traj = traj
    else:
        best = epochs[-1]
        best_traj = traj_map.get(best["epoch"], "N/A")

    sep_pct = best["separatrix_pct"] * 100
    cf1 = best.get("conservative_f1")
    cf1_str = f"{cf1:.4f}" if cf1 is not None else "—"
    cov = best.get("coverage")
    cov_str = f"{cov:.4f}" if cov is not None else "—"
    avg_ss = best.get("avg_set_size")
    avg_ss_str = f"{avg_ss:.2f}" if avg_ss is not None else "—"
    med_ss = best.get("median_set_size")
    med_ss_str = f"{med_ss:.1f}" if med_ss is not None else "—"
    return (
        f"| {label} | {best_traj} | {best['f1']:.4f} | {sep_pct:.2f}% | {cf1_str} | {cov_str} | "
        f"{avg_ss_str} | {med_ss_str} | {best['precision']:.4f} | {best['recall']:.4f} |"
    )


def format_endpoint_errors_table(
    epochs: list[dict],
    traj_map: dict[int, int],
    label: str | None = None,
    regions: list[str] | None = None,
    per_dim: bool = False,
) -> str:
    """Format endpoint errors as a markdown table.

    Args:
        regions: Which regions to show. Default: ["full", "certain_success", "certain_failure"]
        per_dim: If True, show per-dimension breakdown instead of aggregate.
    """
    if regions is None:
        regions = ["full", "certain_success", "certain_failure"]

    lines = []
    if label:
        lines.append(f"#### {label} — Endpoint Errors")
    lines.append("")

    # Get component names from first epoch that has them
    component_names = []
    for m in epochs:
        ee = m.get("endpoint_errors", {})
        if ee.get("component_names"):
            component_names = ee["component_names"]
            break

    if per_dim and component_names:
        # Per-dimension table for each region
        for region in regions:
            region_label = region.replace("_", " ").title()
            lines.append(f"**{region_label}:**")
            lines.append("")

            header = "| Trajectories | " + " | ".join(component_names) + " |"
            sep_line = "|-------------|" + "|".join(["--------"] * len(component_names)) + "|"
            lines.append(header)
            lines.append(sep_line)

            for m in epochs:
                traj = traj_map.get(m["epoch"], "N/A")
                ee = m.get("endpoint_errors", {})
                region_data = ee.get(region, {})
                dim_means = region_data.get("mean_per_dim", [])
                if dim_means:
                    cells = " | ".join(f"{v:.4f}" for v in dim_means)
                    lines.append(f"| {traj} | {cells} |")
                else:
                    lines.append(f"| {traj} | " + " | ".join(["—"] * len(component_names)) + " |")

            lines.append("")
    else:
        # Aggregate table: mean and median for each region
        header = "| Trajectories"
        sep_line = "|-------------"
        for region in regions:
            region_label = region.replace("_", " ").title()
            header += f" | {region_label} Mean | {region_label} Med"
            sep_line += "|--------|--------"
        header += " |"
        sep_line += "|"
        lines.append(header)
        lines.append(sep_line)

        for m in epochs:
            traj = traj_map.get(m["epoch"], "N/A")
            ee = m.get("endpoint_errors", {})
            row = f"| {traj}"
            for region in regions:
                region_data = ee.get(region, {})
                mean = region_data.get("mean")
                median = region_data.get("median")
                row += f" | {mean:.4f}" if mean is not None else " | —"
                row += f" | {median:.4f}" if median is not None else " | —"
            row += " |"
            lines.append(row)

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Extract evaluation results as markdown tables"
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Path to training run directory or specific evaluations folder",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="lambda_delta",
        help=f"Metric type to extract: {', '.join(METRIC_TYPES)}, or 'all' (default: lambda_delta)",
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="Custom label for the table header (auto-detected if not provided)",
    )
    parser.add_argument(
        "--summary-at",
        type=int,
        default=None,
        help="Also print a summary line at this trajectory count",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Only print the compact table (no epoch column)",
    )
    parser.add_argument(
        "--errors",
        action="store_true",
        help="Also print endpoint error tables (mean/median geodesic distance)",
    )
    parser.add_argument(
        "--errors-only",
        action="store_true",
        help="Only print endpoint error tables (skip classification metrics)",
    )
    parser.add_argument(
        "--errors-per-dim",
        action="store_true",
        help="Show per-dimension endpoint error breakdown",
    )
    parser.add_argument(
        "--errors-regions",
        type=str,
        default=None,
        help="Comma-separated regions to show (default: full,certain_success,certain_failure)",
    )
    args = parser.parse_args()

    input_path = args.path.resolve()

    # Determine training_dir and eval_dir
    if input_path.name.startswith("radius_") or input_path.name.startswith("mc_"):
        # User passed the eval config dir directly
        eval_dir = input_path
        training_dir = input_path.parent.parent  # evaluations/../
    elif (input_path / "evaluations").exists():
        # User passed the training run directory
        training_dir = input_path
        eval_dir = find_eval_dir(training_dir)
        if eval_dir is None:
            print("ERROR: No evaluation directories found", file=sys.stderr)
            return 1
    else:
        print(f"ERROR: Cannot find evaluations in {input_path}", file=sys.stderr)
        return 1

    # Parse run info
    run_info = parse_run_info(training_dir)

    # Auto-generate label
    if args.label is None:
        d2 = run_info.get("d2_ratio", "?")
        mode = run_info.get("sampling_mode", "?")
        if d2 == 0 or d2 == 0.0:
            label = "Non-adaptive"
        else:
            label = f"Adaptive ({mode}, d2={d2})"
    else:
        label = args.label

    # Get eval config info
    eval_config_path = eval_dir / "eval_config.json"
    eval_params = {}
    if eval_config_path.exists():
        with open(eval_config_path) as f:
            eval_config = json.load(f)
        eval_params = eval_config.get("parameters", {})

    # Get trajectory counts
    traj_map = get_trajectory_counts(training_dir)

    # Print header info
    print(f"<!-- Run: {training_dir} -->")
    print(f"<!-- Eval: {eval_dir.name} -->")
    print(f"<!-- Params: radius={eval_params.get('attractor_radius')}, "
          f"alpha={eval_params.get('alpha_eval')}, "
          f"mc={eval_params.get('num_mc_samples')}, "
          f"batch={eval_params.get('batch_size')} -->")
    print(f"<!-- d2_ratio={run_info.get('d2_ratio')}, "
          f"sampling_mode={run_info.get('sampling_mode')} -->")
    print()

    # Parse error regions
    error_regions = None
    if args.errors_regions:
        error_regions = [r.strip() for r in args.errors_regions.split(",")]

    # Extract and format
    metric_types = METRIC_TYPES if args.metric == "all" else [args.metric]

    for metric_type in metric_types:
        epochs = extract_all_epochs(eval_dir, metric_type)
        if not epochs:
            print(f"No data found for metric type: {metric_type}", file=sys.stderr)
            continue

        if args.metric == "all":
            section_label = f"{label} — {metric_type}"
        else:
            section_label = label

        if not args.errors_only:
            if args.compact:
                # Compact table without epoch column
                has_cons_f1 = any(m.get("conservative_f1") is not None for m in epochs)
                has_coverage = any(m.get("coverage") is not None for m in epochs)
                has_set_size = any(m.get("avg_set_size") is not None for m in epochs)

                print(f"#### {section_label}")
                print()
                header = "| Trajectories | Sep% | F1"
                sep_line = "|-------------|------|------"
                if has_cons_f1:
                    header += " | Cons. F1"
                    sep_line += "|----------"
                if has_coverage:
                    header += " | Coverage"
                    sep_line += "|----------"
                if has_set_size:
                    header += " | Avg Set | Med Set"
                    sep_line += "|---------|---------"
                header += " | Precision | Recall | Specificity |"
                sep_line += "|-----------|--------|-------------|"
                print(header)
                print(sep_line)
                for m in epochs:
                    traj = traj_map.get(m["epoch"], "N/A")
                    sep_pct = m["separatrix_pct"] * 100
                    row = f"| {traj} | {sep_pct:.2f}% | {m['f1']:.4f}"
                    if has_cons_f1:
                        cf1 = m.get("conservative_f1")
                        row += f" | {cf1:.4f}" if cf1 is not None else " | —"
                    if has_coverage:
                        cov = m.get("coverage")
                        row += f" | {cov:.4f}" if cov is not None else " | —"
                    if has_set_size:
                        avg_ss = m.get("avg_set_size")
                        med_ss = m.get("median_set_size")
                        row += f" | {avg_ss:.2f}" if avg_ss is not None else " | —"
                        row += f" | {med_ss:.1f}" if med_ss is not None else " | —"
                    row += f" | {m['precision']:.4f} | {m['recall']:.4f} | {m['specificity']:.4f} |"
                    print(row)
            else:
                table = format_markdown_table(epochs, traj_map, section_label, metric_type)
                print(table)

            print()

            # Summary line
            if args.summary_at is not None:
                summary = format_summary_line(epochs, traj_map, section_label, args.summary_at)
                print("**Summary line (for summary table):**")
                print()
                print("| Method | Trajectories | F1 | Sep% | Cons. F1 | Coverage | Avg Set | Med Set | Precision | Recall |")
                print("|--------|-------------|------|------|----------|----------|---------|---------|-----------|--------|")
                print(summary)
                print()

        # Endpoint error tables
        if args.errors or args.errors_only:
            has_errors = any(m.get("endpoint_errors", {}).get("full") for m in epochs)
            if has_errors:
                err_table = format_endpoint_errors_table(
                    epochs, traj_map, section_label,
                    regions=error_regions,
                    per_dim=args.errors_per_dim,
                )
                print(err_table)
                print()
            else:
                print(f"No endpoint error data found for {section_label}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
