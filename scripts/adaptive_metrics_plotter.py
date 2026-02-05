"""
Adaptive vs Non-Adaptive Metrics Plotter

Plots adaptive experiment metrics (from metrics directories) alongside
non-adaptive baseline scores (manually specified) for comparison.

Usage:
    python src/visualization/adaptive_metrics_plotter.py --config path/to/config.yaml
"""

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import yaml


def load_config(config_path: str) -> dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_adaptive_metrics(metrics_dir: str) -> pd.DataFrame:
    """Load metrics from an adaptive experiment's metrics directory."""
    metrics_file = Path(metrics_dir) / "metrics_summary.csv"
    if not metrics_file.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_file}")
    return pd.read_csv(metrics_file)


def get_color(index: int, is_adaptive: bool) -> tuple:
    """
    Get color for an experiment.

    Adaptive experiments use distinct colors from tab10.
    Non-adaptive experiments use lighter variants.
    """
    # Use tab10 for more distinct colors
    cmap = plt.cm.tab10
    if is_adaptive:
        color_idx = index % 10
    else:
        color_idx = index % 10
    return cmap(color_idx)


def sync_first_epoch(
    adaptive_data: list[tuple[str, pd.DataFrame]],
    metric_name: str,
) -> None:
    """
    Sync first epoch values across all experiments by using the max value.

    Modifies dataframes in place.
    """
    if len(adaptive_data) < 2:
        return

    # Find the max value at the first epoch for this metric
    first_values = []
    for name, df in adaptive_data:
        if metric_name in df.columns and len(df) > 0:
            val = df[metric_name].iloc[0]
            if not pd.isna(val):
                first_values.append(val)

    if not first_values:
        return

    max_val = max(first_values)

    # Set first epoch value to max for all experiments
    for name, df in adaptive_data:
        if metric_name in df.columns and len(df) > 0:
            df.loc[df.index[0], metric_name] = max_val


def plot_metric(
    config: dict[str, Any],
    metric_name: str,
    adaptive_data: list[tuple[str, pd.DataFrame]],
    output_dir: Path,
    output_format: str = "png",
) -> None:
    """
    Plot a single metric comparing adaptive and non-adaptive experiments.

    Args:
        config: Configuration dictionary
        metric_name: Name of the metric to plot (column in CSV)
        adaptive_data: List of (name, dataframe) tuples for adaptive experiments
        output_dir: Directory to save the plot
        output_format: Output format (png, pdf, etc.)
    """
    # Sync first epoch if requested
    if config.get("sync_first_epoch", False):
        sync_first_epoch(adaptive_data, metric_name)

    # Get figure size from config (default 8x8)
    figsize = config.get("figsize", [8, 8])
    fig, ax = plt.subplots(figsize=tuple(figsize))

    x_min, x_max = float("inf"), float("-inf")

    # Plot adaptive experiments (solid lines with markers)
    for idx, (name, df) in enumerate(adaptive_data):
        if metric_name not in df.columns:
            print(f"Warning: Metric '{metric_name}' not found in {name}, skipping")
            continue

        x = df["train_trajectories"]
        y = df[metric_name]

        # Filter out NaN values
        mask = ~y.isna()
        x, y = x[mask], y[mask]

        if len(x) == 0:
            continue

        x_min = min(x_min, x.min())
        x_max = max(x_max, x.max())

        color = get_color(idx, is_adaptive=True)
        ax.plot(x, y, "-o", label=name, color=color, markersize=8, linewidth=3.5)

    # Plot non-adaptive experiments (horizontal dotted lines)
    non_adaptive = config.get("non_adaptive_experiments", [])
    for idx, exp in enumerate(non_adaptive):
        name = exp["name"]
        scores = exp.get("scores", {})

        if metric_name not in scores:
            print(f"Warning: Metric '{metric_name}' not specified for {name}, skipping")
            continue

        score = scores[metric_name]
        color = get_color(idx, is_adaptive=False)
        ax.axhline(y=score, linestyle="--", label=name, color=color, linewidth=3)

    # Configure axes
    ax.set_xlabel("Number of Training Trajectories", fontsize=20)
    # Use custom y-axis label if specified, otherwise use metric name
    y_labels = config.get("y_labels", {})
    y_label = y_labels.get(metric_name, metric_name)
    ax.set_ylabel(y_label, fontsize=20)
    ax.tick_params(axis='both', labelsize=18)
    ax.grid(True, alpha=0.3)

    # Only show legend on specified metrics (default: all)
    legend_on_metrics = config.get("legend_on_metrics", None)
    if legend_on_metrics is None or metric_name in legend_on_metrics:
        ax.legend(loc="best", fontsize=20)

    # Remove top and right spines (only keep x and y axes)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Adjust x-axis limits with some padding
    if x_min != float("inf") and x_max != float("-inf"):
        padding = (x_max - x_min) * 0.02
        # Apply config-specified x limits if provided
        x_min_cfg = config.get("x_min")
        x_max_cfg = config.get("x_max")
        ax.set_xlim(
            (x_min_cfg - padding) if x_min_cfg is not None else (x_min - padding),
            (x_max_cfg + padding) if x_max_cfg is not None else (x_max + padding),
        )

    # Apply fixed y-axis limits if specified
    y_min = config.get("y_min")
    y_max = config.get("y_max")
    if y_min is not None or y_max is not None:
        current_ylim = ax.get_ylim()
        ax.set_ylim(
            y_min if y_min is not None else current_ylim[0],
            y_max if y_max is not None else current_ylim[1],
        )

    plt.tight_layout()

    # Save the plot
    output_path = output_dir / f"{metric_name}.{output_format}"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot adaptive vs non-adaptive experiment metrics"
    )
    parser.add_argument(
        "--config", "-c",
        required=True,
        help="Path to YAML configuration file"
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Load adaptive experiment data
    adaptive_data = []
    for exp in config.get("adaptive_experiments", []):
        name = exp["name"]
        metrics_dir = exp["metrics_dir"]
        try:
            df = load_adaptive_metrics(metrics_dir)
            adaptive_data.append((name, df))
            print(f"Loaded adaptive experiment: {name}")
        except FileNotFoundError as e:
            print(f"Warning: {e}")

    # Prepare output directory
    output_dir = Path(config.get("output_dir", "./plots"))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_format = config.get("output_format", "png")

    # Plot each metric
    metrics_to_plot = config.get("metrics_to_plot", [])
    if not metrics_to_plot:
        print("Warning: No metrics specified in 'metrics_to_plot'")
        return

    for metric_name in metrics_to_plot:
        print(f"Plotting metric: {metric_name}")
        plot_metric(config, metric_name, adaptive_data, output_dir, output_format)

    print(f"\nAll plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
