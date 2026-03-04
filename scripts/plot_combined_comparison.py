#!/usr/bin/env python3
"""
Combined CartPole comparison: Ground Truth vs Olympics-Classifier vs baselines.

Reads cached results from pipelines and composes unified figures.
  - Olympics-classifier cache: results/figures/qualitative_cartpole_<slice>_cache.npz
  - DeepReach cache: <deepreach_cache_dir>/cache_<exp>_<slice>_r<res>_t<t>.npz
  - Lyapunov NN CSVs: lyap_classified_<slice>.csv

Two layout variants:
  Horizontal:  rows = slices,  cols = methods  (wide, good for 2-column / full-width)
  Vertical:    rows = methods, cols = slices   (tall/narrow, fits IROS single column ~3.5in)

Usage:
    # Olympics-only (no baselines)
    python scripts/plot_combined_comparison.py --all_slices

    # With Lyapunov baseline
    python scripts/plot_combined_comparison.py --all_slices --baselines lyapunov

    # With both baselines
    python scripts/plot_combined_comparison.py --all_slices --baselines lyapunov deepreach

    # Skip error analysis
    python scripts/plot_combined_comparison.py --all_slices --baselines lyapunov --no_errors
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# ── Olympics-classifier cache (same paths as plot_qualitative_cartpole.py) ────
OLYMPICS_CACHE_DIR = Path("results/figures")
OLYMPICS_RUNS = {
    "Adaptive (Ours)": None,
    "Non-adaptive (Ours)": None,
}
# Map from cache keys (old names) to display labels
_CACHE_KEY_MAP = {
    "Adaptive (d2=1.0)": "Adaptive (Ours)",
    "Non-adaptive": "Non-adaptive (Ours)",
}
DEFAULT_EPOCH = 15

# ── Lyapunov NN baseline ─────────────────────────────────────────────────────
LYAPUNOV_DIR = Path(
    "/common/users/shared/pracsys/adaptive_roa_experiments/dhruv"
    "/lyapunov neural network"
)
LYAPUNOV_CSVS = {
    "theta_thetadot": LYAPUNOV_DIR / "lyap_classified_theta_thetadot.csv",
    "x_xdot": LYAPUNOV_DIR / "lyap_classified_x_xdot.csv",
}
# (ax0_col, ax1_col) matching SLICES sweep_indices ordering
LYAPUNOV_AXES = {
    "theta_thetadot": ("s1", "s3"),   # theta, thetadot
    "x_xdot": ("s0", "s2"),           # x, xdot
}

# ── DeepReach CSV baseline ────────────────────────────────────────────────────
DEEPREACH_DIR = Path(
    "/common/users/shared/pracsys/adaptive_roa_experiments/dhruv/deepreach"
)
DEEPREACH_CSVS = {
    "theta_thetadot": DEEPREACH_DIR / "deepreach_classified_theta_thetadot.csv",
    "x_xdot": DEEPREACH_DIR / "deepreach_classified_x_xdot.csv",
}
DEEPREACH_AXES = {
    "theta_thetadot": ("s1", "s3"),   # theta, thetadot
    "x_xdot": ("s0", "s2"),           # x, xdot
}

# ── DeepReach thresholds (calibrated on 1000 best) ──────────────────────────
DEEPREACH_C_LOW = -0.264
DEEPREACH_C_HIGH = -0.194

# ── Slice definitions (must match plot_qualitative_cartpole.py) ──────────────
SLICES = {
    "theta_thetadot": {
        "sweep_indices": (1, 3),
        "sweep_ranges": ((-np.pi, np.pi), (-8.5, 8.5)),
        "fixed_values": {0: 0.0, 2: 0.0},
        "xlabel": r"$\theta$ (rad)",
        "ylabel": r"$\dot{\theta}$ (rad/s)",
        "xticks": [-np.pi, -np.pi/2, 0, np.pi/2, np.pi],
        "xticklabels": [r"$-\pi$", r"$-\frac{\pi}{2}$", "0", r"$\frac{\pi}{2}$", r"$\pi$"],
        "yticks": [-2*np.pi, -np.pi, 0, np.pi, 2*np.pi],
        "yticklabels": [r"$-2\pi$", r"$-\pi$", "0", r"$\pi$", r"$2\pi$"],
    },
    "x_xdot": {
        "sweep_indices": (0, 2),
        "sweep_ranges": ((-6.0, 6.0), (-7.0, 7.0)),
        "fixed_values": {1: 0.0, 3: 0.0},
        "xlabel": r"$x$ (m)",
        "ylabel": r"$\dot{x}$ (m/s)",
        "xticks": [-6, -3, 0, 3, 6],
        "xticklabels": ["-6", "-3", "0", "3", "6"],
        "yticks": [-6, -3, 0, 3, 6],
        "yticklabels": ["-6", "-3", "0", "3", "6"],
    },
}

# ── Output ───────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path("results/figures")
FLAGSHIP_DIR = OUTPUT_DIR / "flagship"


# ═══════════════════════════════════════════════════════════════════════════════
# Data loaders
# ═══════════════════════════════════════════════════════════════════════════════

def load_olympics_cache(slice_name: str, epoch: int):
    """Load olympics-classifier cache for one slice.

    Returns: (gt_labels, gt_n0, gt_n1, methods_dict)
    where methods_dict[method_name] = {pred_labels, p_success, ...}
    """
    cf = OLYMPICS_CACHE_DIR / f"qualitative_cartpole_{slice_name}_cache.npz"
    if not cf.exists():
        raise FileNotFoundError(
            f"Olympics cache not found: {cf}\n"
            "Run: python scripts/plot_qualitative_cartpole.py --all_slices"
        )
    data = np.load(str(cf), allow_pickle=False)
    gt_labels = data["gt_labels"]
    n0 = int(data["n0"])
    n1 = int(data["n1"])

    methods = {}
    for cache_key, display_name in _CACHE_KEY_MAP.items():
        prefix = f"{cache_key}_epoch{epoch}"
        key = f"{prefix}_pred"
        if key not in data:
            print(f"  Warning: {cache_key} epoch {epoch} not in cache, skipping")
            continue
        methods[display_name] = {
            "pred_labels": data[f"{prefix}_pred"],
            "p_success": data[f"{prefix}_psuc"],
            "lambda_star": float(data[f"{prefix}_lam"]),
            "delta_star": float(data[f"{prefix}_delta"]),
        }
    return gt_labels, n0, n1, methods


def load_lyapunov_csv(slice_name: str):
    """Load Lyapunov NN CSV for one slice.

    Returns: (labels_flat, values_flat, n0, n1)
    Labels: {-1, 0, 1} = {failure, uncertain, success}
    """
    import pandas as pd

    csv_path = LYAPUNOV_CSVS[slice_name]
    if not csv_path.exists():
        raise FileNotFoundError(f"Lyapunov CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    ax0_col, ax1_col = LYAPUNOV_AXES[slice_name]

    # Sort by (ax0, ax1) to match grid ordering (meshgrid indexing="ij")
    df = df.sort_values([ax0_col, ax1_col]).reset_index(drop=True)

    n0 = df[ax0_col].nunique()
    n1 = df[ax1_col].nunique()

    labels = df["label"].values.astype(np.int8)
    values = df["value"].values.astype(np.float32)

    print(f"  Loaded Lyapunov {csv_path.name}: {n0}x{n1} = {len(df)} points")
    n_safe = np.sum(labels == 1)
    n_fail = np.sum(labels == -1)
    n_unc = np.sum(labels == 0)
    total = len(labels)
    print(f"    {n_safe} safe ({100*n_safe/total:.1f}%), "
          f"{n_fail} fail ({100*n_fail/total:.1f}%), "
          f"{n_unc} uncertain ({100*n_unc/total:.1f}%)")
    return labels, values, n0, n1


def load_deepreach_csv(slice_name: str):
    """Load DeepReach CSV for one slice.

    Returns: (pred_labels_flat, values_flat, n0, n1)
    pred_labels: {-1, 0, 1} = {failure, uncertain, success}
    """
    import pandas as pd

    csv_path = DEEPREACH_CSVS[slice_name]
    if not csv_path.exists():
        raise FileNotFoundError(f"DeepReach CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    ax0_col, ax1_col = DEEPREACH_AXES[slice_name]

    # Sort by (ax0, ax1) to match grid ordering (meshgrid indexing="ij")
    df = df.sort_values([ax0_col, ax1_col]).reset_index(drop=True)

    n0 = df[ax0_col].nunique()
    n1 = df[ax1_col].nunique()

    labels = df["pred_label"].values.astype(np.int8)
    values = df["value"].values.astype(np.float32)

    print(f"  Loaded DeepReach {csv_path.name}: {n0}x{n1} = {len(df)} points")
    n_safe = np.sum(labels == 1)
    n_fail = np.sum(labels == -1)
    n_unc = np.sum(labels == 0)
    total = len(labels)
    print(f"    {n_safe} safe ({100*n_safe/total:.1f}%), "
          f"{n_fail} fail ({100*n_fail/total:.1f}%), "
          f"{n_unc} uncertain ({100*n_unc/total:.1f}%)")
    return labels, values, n0, n1


# ── DeepReach loaders (legacy .npz) ─────────────────────────────────────────

def find_deepreach_cache(cache_dir: str, slice_name: str):
    """Find a DeepReach .npz cache for the given slice in cache_dir."""
    cache_dir = Path(cache_dir)
    # Pattern: cache_<exp>_<slice>_t<t>.npz
    candidates = sorted(cache_dir.glob(f"cache_*_{slice_name}_t*.npz"))
    if not candidates:
        return None
    if len(candidates) > 1:
        print(f"  Multiple DeepReach caches for {slice_name}, using latest: {candidates[-1].name}")
    return candidates[-1]


def load_deepreach_cache(cache_path: Path):
    """Load DeepReach cache. Returns (values_2d, axis0, axis1, gt_labels)."""
    data = np.load(str(cache_path))
    gt_labels = data["gt_labels"] if "gt_labels" in data else None
    return data["values"], data["axis0"], data["axis1"], gt_labels


def _print_stats(heatmap):
    n_safe = np.sum(heatmap == 1.0)
    n_unsafe = np.sum(heatmap == -1.0)
    n_sep = np.sum(heatmap == 0.0)
    total = heatmap.size
    print(f"      {n_safe} safe ({100*n_safe/total:.1f}%), "
          f"{n_unsafe} fail ({100*n_unsafe/total:.1f}%), "
          f"{n_sep} sep ({100*n_sep/total:.1f}%)")


def deepreach_to_heatmap(values_2d, c_low, c_high):
    """Classify DeepReach V values into 3-class heatmap.

    V <= c_low  -> +1 (success / safe)
    V >= c_high -> -1 (failure / unsafe)
    else        ->  0 (separatrix / uncertain)
    """
    heatmap = np.zeros_like(values_2d, dtype=float)
    heatmap[values_2d <= c_low] = 1.0
    heatmap[values_2d >= c_high] = -1.0
    return heatmap


def deepreach_to_labels(values_2d, c_low, c_high):
    """Convert DeepReach V values to flat 3-class labels {-1, 0, 1}."""
    hm = deepreach_to_heatmap(values_2d, c_low, c_high)
    return hm.ravel().astype(np.int8)


# ═══════════════════════════════════════════════════════════════════════════════
# Heatmap / styling helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _discrete_cmap():
    """Red / yellow / green for failure / uncertain / success."""
    colors = ["#d73027", "#fee08b", "#1a9850"]
    cmap = mcolors.ListedColormap(colors)
    bounds = [-1.5, -0.5, 0.5, 1.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    return cmap, norm


def _gt_to_heatmap(gt_labels, n0, n1):
    return np.where(gt_labels == 1, 1.0, -1.0).reshape(n0, n1).T


def _pred_to_heatmap(pred, n0, n1):
    """Binary pred {0,1} -> discrete heatmap {-1, 1}."""
    heatmap = np.zeros_like(pred, dtype=float)
    heatmap[pred == 1] = 1
    heatmap[pred == 0] = -1
    return heatmap.reshape(n0, n1).T


def _3class_to_heatmap(labels, n0, n1):
    """3-class labels {-1, 0, 1} -> 2D heatmap (values already correct)."""
    return labels.astype(float).reshape(n0, n1).T


def _style_ax(ax, sl, fontsize=12):
    ax.set_xticks(sl["xticks"])
    ax.set_xticklabels(sl["xticklabels"], fontsize=fontsize, fontweight="bold")
    ax.set_yticks(sl["yticks"])
    ax.set_yticklabels(sl["yticklabels"], fontsize=fontsize, fontweight="bold")
    ax.tick_params(labelsize=fontsize)


def _save_bare(heatmap_2d, extent, cmap, name, norm=None, vmin=None, vmax=None, interp="nearest"):
    """Save a single bare image -- no axis, no title."""
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(
        heatmap_2d, origin="lower", aspect="auto",
        cmap=cmap, norm=norm, vmin=vmin, vmax=vmax,
        extent=extent, interpolation=interp,
    )
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FLAGSHIP_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(OUTPUT_DIR / f"{name}.pdf"), dpi=300, bbox_inches="tight", pad_inches=0)
    fig.savefig(str(FLAGSHIP_DIR / f"{name}.png"), dpi=300, bbox_inches="tight", pad_inches=0)
    print(f"  Saved bare: {name}")
    plt.close(fig)


def _ensure_dirs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FLAGSHIP_DIR.mkdir(parents=True, exist_ok=True)


def _save_fig(fig, base_name):
    """Save figure to both OUTPUT_DIR (.pdf) and FLAGSHIP_DIR (.png)."""
    _ensure_dirs()
    for ext, d in [("pdf", OUTPUT_DIR), ("png", FLAGSHIP_DIR)]:
        out = str(d / f"{base_name}.{ext}")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Saved {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# Combined discrete plots (horizontal + vertical)
# ═══════════════════════════════════════════════════════════════════════════════

def _build_col_labels(olympics_methods, baselines):
    """Build ordered column labels: GT + olympics methods + baselines."""
    labels = ["Ground Truth"] + list(olympics_methods)
    for bl in baselines:
        labels.append(bl["label"])
    return labels


def _plot_row(axes_row, slice_name, d, olympics_methods,
              baselines, cmap, norm, label_fs, tick_fs):
    """Fill one row of subplots (one slice)."""
    sl = SLICES[slice_name]
    extent = [*sl["sweep_ranges"][0], *sl["sweep_ranges"][1]]
    col = 0

    # Ground truth
    ax = axes_row[col]
    ax.imshow(d["gt_heatmap"], origin="lower", aspect="auto",
              cmap=cmap, norm=norm, extent=extent, interpolation="nearest")
    ax.set_xlabel(sl["xlabel"], fontsize=label_fs, fontweight="bold")
    ax.set_ylabel(sl["ylabel"], fontsize=label_fs, fontweight="bold")
    _style_ax(ax, sl, fontsize=tick_fs)
    col += 1

    # Olympics-classifier columns
    for method_name in olympics_methods:
        ax = axes_row[col]
        hm = _pred_to_heatmap(d["olympics"][method_name]["pred_labels"],
                               d["n0"], d["n1"])
        ax.imshow(hm, origin="lower", aspect="auto",
                  cmap=cmap, norm=norm, extent=extent, interpolation="nearest")
        ax.set_xlabel(sl["xlabel"], fontsize=label_fs, fontweight="bold")
        ax.set_ylabel(sl["ylabel"], fontsize=label_fs, fontweight="bold")
        _style_ax(ax, sl, fontsize=tick_fs)
        col += 1

    # Baseline columns
    for bl in baselines:
        ax = axes_row[col]
        ax.imshow(d[bl["heatmap_key"]], origin="lower", aspect="auto",
                  cmap=cmap, norm=norm, extent=extent, interpolation="nearest")
        ax.set_xlabel(sl["xlabel"], fontsize=label_fs, fontweight="bold")
        ax.set_ylabel(sl["ylabel"], fontsize=label_fs, fontweight="bold")
        _style_ax(ax, sl, fontsize=tick_fs)
        col += 1


# ── Horizontal layout: rows=slices, cols=methods (wide) ─────────────────────

def plot_discrete_horizontal(all_data: dict, epoch: int, baselines: list,
                              suffix: str = ""):
    """Horizontal: rows=slices, cols=(GT, Adaptive, Non-adaptive, [baselines...])."""
    slice_names = list(all_data.keys())
    olympics_methods = [m for m in OLYMPICS_RUNS if m in all_data[slice_names[0]]["olympics"]]
    col_labels = _build_col_labels(olympics_methods, baselines)
    n_rows = len(slice_names)
    n_cols = len(col_labels)

    cmap, norm = _discrete_cmap()

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.5 * n_cols, 3.0 * n_rows),
        squeeze=False,
        constrained_layout=True,
    )

    for row, slice_name in enumerate(slice_names):
        _plot_row(axes[row], slice_name, all_data[slice_name],
                  olympics_methods, baselines, cmap, norm,
                  label_fs=13, tick_fs=12)

    # Column titles
    for col, label in enumerate(col_labels):
        axes[0, col].set_title(label, fontsize=14, fontweight="bold")

    # Legend
    from matplotlib.patches import Patch
    legend_patches = [
        Patch(facecolor="#1a9850", label="Success"),
        Patch(facecolor="#fee08b", label="Uncertain"),
        Patch(facecolor="#d73027", label="Failure"),
    ]
    fig.legend(
        handles=legend_patches, loc="lower center",
        ncol=3, fontsize=12, frameon=False, prop={"weight": "bold"},
        bbox_to_anchor=(0.5, -0.04),
    )

    tag = f"_{suffix}" if suffix else ""
    _save_fig(fig, f"combined_comparison_discrete_horiz{tag}_ep{epoch}")
    plt.close(fig)


# ── Vertical layout: rows=methods, cols=slices (IROS single-column) ─────────

IROS_COL_WIDTH = 3.5   # inches, single-column

def plot_discrete_vertical(all_data: dict, epoch: int, baselines: list,
                            suffix: str = ""):
    """Vertical (IROS single-col): rows=(GT, Adaptive, Non-adaptive, [baselines...]), cols=slices."""
    slice_names = list(all_data.keys())
    olympics_methods = [m for m in OLYMPICS_RUNS if m in all_data[slice_names[0]]["olympics"]]
    row_labels = _build_col_labels(olympics_methods, baselines)
    n_rows = len(row_labels)
    n_cols = len(slice_names)

    # Sizing: fit IROS single column
    cell_w = IROS_COL_WIDTH / n_cols
    cell_h = cell_w * 0.9  # slightly shorter than wide
    fig_w = IROS_COL_WIDTH
    fig_h = cell_h * n_rows

    cmap, norm = _discrete_cmap()

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(fig_w, fig_h),
        squeeze=False,
        constrained_layout=True,
    )

    # Font sizes scaled for single-column
    title_fs = 8
    label_fs = 7
    tick_fs = 6
    legend_fs = 7

    for col_idx, slice_name in enumerate(slice_names):
        sl = SLICES[slice_name]
        d = all_data[slice_name]
        extent = [*sl["sweep_ranges"][0], *sl["sweep_ranges"][1]]
        row = 0

        # Row 0: Ground truth
        ax = axes[row, col_idx]
        ax.imshow(d["gt_heatmap"], origin="lower", aspect="auto",
                  cmap=cmap, norm=norm, extent=extent, interpolation="nearest")
        ax.set_xlabel(sl["xlabel"], fontsize=label_fs, fontweight="bold")
        ax.set_ylabel(sl["ylabel"], fontsize=label_fs, fontweight="bold")
        _style_ax(ax, sl, fontsize=tick_fs)
        row += 1

        # Olympics methods
        for method_name in olympics_methods:
            ax = axes[row, col_idx]
            hm = _pred_to_heatmap(d["olympics"][method_name]["pred_labels"],
                                   d["n0"], d["n1"])
            ax.imshow(hm, origin="lower", aspect="auto",
                      cmap=cmap, norm=norm, extent=extent, interpolation="nearest")
            ax.set_xlabel(sl["xlabel"], fontsize=label_fs, fontweight="bold")
            ax.set_ylabel(sl["ylabel"], fontsize=label_fs, fontweight="bold")
            _style_ax(ax, sl, fontsize=tick_fs)
            row += 1

        # Baselines
        for bl in baselines:
            ax = axes[row, col_idx]
            ax.imshow(d[bl["heatmap_key"]], origin="lower", aspect="auto",
                      cmap=cmap, norm=norm, extent=extent, interpolation="nearest")
            ax.set_xlabel(sl["xlabel"], fontsize=label_fs, fontweight="bold")
            ax.set_ylabel(sl["ylabel"], fontsize=label_fs, fontweight="bold")
            _style_ax(ax, sl, fontsize=tick_fs)
            row += 1

    # Slice titles on top row
    for col_idx, slice_name in enumerate(slice_names):
        short = r"$(\theta,\,\dot{\theta})$" if "theta_thetadot" in slice_name else r"$(x,\,\dot{x})$"
        axes[0, col_idx].set_title(short, fontsize=title_fs, fontweight="bold")

    # Row labels on left edge
    for row, label in enumerate(row_labels):
        axes[row, 0].annotate(
            label, xy=(0, 0.5), xytext=(-0.45, 0.5),
            xycoords="axes fraction", textcoords="axes fraction",
            fontsize=label_fs, fontweight="bold",
            ha="right", va="center", rotation=90,
        )

    # Legend
    from matplotlib.patches import Patch
    legend_patches = [
        Patch(facecolor="#1a9850", label="Success"),
        Patch(facecolor="#fee08b", label="Uncertain"),
        Patch(facecolor="#d73027", label="Failure"),
    ]
    fig.legend(
        handles=legend_patches, loc="lower center",
        ncol=3, fontsize=legend_fs, frameon=False, prop={"weight": "bold"},
        bbox_to_anchor=(0.5, -0.03),
    )

    tag = f"_{suffix}" if suffix else ""
    _save_fig(fig, f"combined_comparison_discrete_vert{tag}_ep{epoch}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Error analysis (FP/FN plots)
# ═══════════════════════════════════════════════════════════════════════════════

def _error_cmap():
    """5-color discrete colormap for error analysis.

    0 = TP  (green)      : GT=1, pred=1
    1 = TN  (dark red)   : GT=0, pred=-1 (3-class) or pred=0 (binary)
    2 = FP  (blue)       : GT=0, pred=1
    3 = FN  (orange)     : GT=1, pred=-1 (3-class) or pred=0 (binary)
    4 = Separatrix       : pred=0, any GT (3-class only)
        (light yellow)
    """
    colors = ["#FDE725", "#440154", "#35B779", "#31688E", "#D3D3D3"]
    cmap = mcolors.ListedColormap(colors)
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    return cmap, norm


def compute_error_map(gt_binary, pred_labels, failure_value):
    """Compute per-point error classification.

    Args:
        gt_binary:     flat {0, 1} array (0=failure, 1=success)
        pred_labels:   flat array of predictions
        failure_value: the prediction value that means "definitive failure"
                       0  for Olympics  (pred: 1=success, 0=failure, <0=uncertain)
                       -1 for Lyapunov/DeepReach (pred: 1=success, -1=failure, 0=uncertain)

    Returns:
        error_codes: flat int8 array
            0 = TP         (GT=1, pred=1)
            1 = TN         (GT=0, pred=failure_value)
            2 = FP         (GT=0, pred=1)
            3 = FN         (GT=1, pred=failure_value)
            4 = Separatrix (pred is neither 1 nor failure_value, any GT)
    """
    gt_success = (gt_binary == 1)
    pred_success = (pred_labels == 1)
    pred_failure = (pred_labels == failure_value)
    sep_mask = ~pred_success & ~pred_failure

    codes = np.empty(len(gt_binary), dtype=np.int8)
    codes[sep_mask] = 4                              # Separatrix
    codes[gt_success & pred_success] = 0             # TP
    codes[~gt_success & pred_failure] = 1            # TN
    codes[~gt_success & pred_success] = 2            # FP
    codes[gt_success & pred_failure] = 3             # FN

    return codes


def _print_error_stats(label, codes, total):
    """Print error analysis statistics for one method."""
    tp = int(np.sum(codes == 0))
    tn = int(np.sum(codes == 1))
    fp = int(np.sum(codes == 2))
    fn = int(np.sum(codes == 3))
    sep = int(np.sum(codes == 4))

    # FPR/FNR only on definitive predictions (excluding separatrix)
    gt_pos = tp + fn              # GT=1 with definitive pred
    gt_neg = tn + fp              # GT=0 with definitive pred

    fpr = fp / gt_neg * 100 if gt_neg > 0 else 0
    fnr = fn / gt_pos * 100 if gt_pos > 0 else 0

    print(f"    {label:25s}: TP={tp:6d} TN={tn:6d} FP={fp:6d} FN={fn:6d} "
          f"Sep={sep:6d} | FPR={fpr:.1f}% FNR={fnr:.1f}%")


def _build_methods_for_errors(all_data, slice_names, baselines):
    """Build unified method list for error analysis.

    Each entry: {"label": str, "get_labels": callable(d)->flat_labels, "failure_value": int}
    failure_value: pred value meaning "definitive failure" (0 for Olympics, -1 for baselines)
    """
    first = all_data[slice_names[0]]
    olympics_methods = [m for m in OLYMPICS_RUNS if m in first["olympics"]]

    methods = []
    for m in olympics_methods:
        methods.append({
            "label": m,
            "get_labels": lambda d, _m=m: d["olympics"][_m]["pred_labels"],
            "failure_value": 0,   # Olympics: 1=success, 0=failure, <0=uncertain
        })
    for bl in baselines:
        methods.append({
            "label": bl["label"],
            "get_labels": lambda d, _k=bl["labels_key"]: d[_k],
            "failure_value": bl["failure_value"],
        })
    return methods


def plot_error_horizontal(all_data: dict, epoch: int, baselines: list,
                           suffix: str = ""):
    """Error analysis (horizontal): rows=slices, cols=(GT + methods).

    Compressed for IROS: x-labels/ticks only on bottom row,
    y-labels/ticks only on left column.
    """
    slice_names = list(all_data.keys())
    methods = _build_methods_for_errors(all_data, slice_names, baselines)
    col_labels = ["Ground Truth"] + [m["label"] for m in methods]

    n_rows = len(slice_names)
    n_cols = len(col_labels)
    is_bottom = lambda r: r == n_rows - 1

    cmap, norm = _error_cmap()
    label_fs = 8
    tick_fs = 7
    title_fs = 10

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(1.8 * n_cols, 1.6 * n_rows),
        squeeze=False,
        gridspec_kw={"wspace": 0.05, "hspace": 0.05},
    )
    fig.subplots_adjust(left=0.10, right=0.98, top=0.90, bottom=0.14)

    for row, slice_name in enumerate(slice_names):
        sl = SLICES[slice_name]
        d = all_data[slice_name]
        extent = [*sl["sweep_ranges"][0], *sl["sweep_ranges"][1]]
        n0, n1 = d["n0"], d["n1"]
        gt = d["gt_labels"]

        print(f"\n  Error analysis -- {slice_name}:")

        def _strip_horiz(ax, r, c):
            _style_ax(ax, sl, fontsize=tick_fs)
            # y-axis: only on leftmost column
            if c == 0:
                ax.set_ylabel(sl["ylabel"], fontsize=label_fs, fontweight="bold")
            else:
                ax.set_ylabel("")
                ax.set_yticklabels([])
                ax.tick_params(left=False)
            # x-axis: ticks on bottom row, label only on leftmost bottom
            if is_bottom(r):
                if c == 0:
                    ax.set_xlabel(sl["xlabel"], fontsize=label_fs, fontweight="bold")
                else:
                    ax.set_xlabel("")
            else:
                ax.set_xlabel("")
                ax.set_xticklabels([])
                ax.tick_params(bottom=False)

        # GT column (trivially all TP/TN)
        gt_codes = np.where(gt == 1, 0, 1).astype(np.int8)
        ax = axes[row, 0]
        ax.imshow(gt_codes.reshape(n0, n1).T, origin="lower", aspect="auto",
                  cmap=cmap, norm=norm, extent=extent, interpolation="nearest")
        _strip_horiz(ax, row, 0)
        _print_error_stats("Ground Truth", gt_codes, gt.size)

        # Method columns
        for col, m in enumerate(methods, start=1):
            pred = m["get_labels"](d)
            codes = compute_error_map(gt, pred, m["failure_value"])
            ax = axes[row, col]
            ax.imshow(codes.reshape(n0, n1).T, origin="lower", aspect="auto",
                      cmap=cmap, norm=norm, extent=extent, interpolation="nearest")
            _strip_horiz(ax, row, col)
            _print_error_stats(m["label"], codes, gt.size)

    # Column titles
    for col, label in enumerate(col_labels):
        axes[0, col].set_title(label, fontsize=title_fs, fontweight="bold")

    # Legend
    from matplotlib.patches import Patch
    legend_patches = [
        Patch(facecolor="#FDE725", label="TP"),
        Patch(facecolor="#440154", label="TN"),
        Patch(facecolor="#35B779", label="FP"),
        Patch(facecolor="#31688E", label="FN"),
        Patch(facecolor="#D3D3D3", label="Separatrix"),
    ]
    fig.legend(
        handles=legend_patches, loc="lower center",
        ncol=5, fontsize=9, frameon=False, prop={"weight": "bold"},
        bbox_to_anchor=(0.5, -0.06),
    )

    tag = f"_{suffix}" if suffix else ""
    _save_fig(fig, f"combined_comparison_errors_horiz{tag}_ep{epoch}")
    plt.close(fig)


def plot_error_vertical(all_data: dict, epoch: int, baselines: list,
                         suffix: str = ""):
    """Error analysis (vertical / IROS): rows=(GT + methods), cols=slices.

    Compressed: x-labels/ticks only on bottom row,
    y-labels/ticks only on left column.
    """
    slice_names = list(all_data.keys())
    methods_raw = _build_methods_for_errors(all_data, slice_names, baselines)
    # Reorder: baselines first, then Olympics reversed (non-adaptive, adaptive)
    bl_methods = [m for m in methods_raw if m["failure_value"] != 0]
    ol_methods = [m for m in methods_raw if m["failure_value"] == 0]
    methods = bl_methods + list(reversed(ol_methods))
    row_labels = ["Ground Truth"] + [m["label"] for m in methods]

    n_rows = len(row_labels)
    n_cols = len(slice_names)
    is_bottom = n_rows - 1

    fig_w = 12
    cell_w = fig_w / n_cols
    cell_h = cell_w * 0.65
    fig_h = cell_h * n_rows

    cmap, norm = _error_cmap()

    title_fs = 20
    label_fs = 18
    tick_fs = 16
    legend_fs = 12

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(fig_w, fig_h),
        squeeze=False,
        gridspec_kw={"wspace": 0.05, "hspace": 0.05},
    )
    fig.subplots_adjust(left=0.20, right=0.92, top=0.95, bottom=0.09)

    for col_idx, slice_name in enumerate(slice_names):
        sl = SLICES[slice_name]
        d = all_data[slice_name]
        extent = [*sl["sweep_ranges"][0], *sl["sweep_ranges"][1]]
        n0, n1 = d["n0"], d["n1"]
        gt = d["gt_labels"]

        row = 0
        is_last_col = (col_idx == n_cols - 1)

        # helper: strip ticks/labels on interior edges
        def _strip_interior(ax, r, c):
            _style_ax(ax, sl, fontsize=tick_fs)
            # y-axis
            if c == 0:
                ax.set_ylabel(sl["ylabel"], fontsize=label_fs, fontweight="bold")
            elif c == n_cols - 1:
                ax.yaxis.set_label_position("right")
                ax.yaxis.tick_right()
                ax.set_ylabel(sl["ylabel"], fontsize=label_fs, fontweight="bold")
            else:
                ax.set_ylabel("")
                ax.set_yticklabels([])
                ax.tick_params(left=False, right=False)
            # x-axis
            if r == is_bottom:
                ax.set_xlabel(sl["xlabel"], fontsize=label_fs, fontweight="bold")
            else:
                ax.set_xlabel("")
                ax.set_xticklabels([])
                ax.tick_params(bottom=False)

        # GT row
        gt_codes = np.where(gt == 1, 0, 1).astype(np.int8)
        ax = axes[row, col_idx]
        ax.imshow(gt_codes.reshape(n0, n1).T, origin="lower", aspect="auto",
                  cmap=cmap, norm=norm, extent=extent, interpolation="nearest")
        _strip_interior(ax, row, col_idx)
        row += 1

        # Method rows
        for i, m in enumerate(methods):
            pred = m["get_labels"](d)
            codes = compute_error_map(gt, pred, m["failure_value"])
            ax = axes[row, col_idx]
            ax.imshow(codes.reshape(n0, n1).T, origin="lower", aspect="auto",
                      cmap=cmap, norm=norm, extent=extent, interpolation="nearest")
            _strip_interior(ax, row, col_idx)
            row += 1

    # Slice titles
    for col_idx, slice_name in enumerate(slice_names):
        short = r"$(\theta,\,\dot{\theta})$" if "theta_thetadot" in slice_name else r"$(x,\,\dot{x})$"
        axes[0, col_idx].set_title(short, fontsize=title_fs, fontweight="bold")

    # Row labels
    method_fs = 20
    for row, label in enumerate(row_labels):
        axes[row, 0].annotate(
            label, xy=(0, 0.5), xytext=(-0.28, 0.5),
            xycoords="axes fraction", textcoords="axes fraction",
            fontsize=method_fs, fontweight="bold",
            ha="right", va="center", rotation=90,
        )

    from matplotlib.patches import Patch
    legend_patches = [
        Patch(facecolor="#FDE725", edgecolor="0.4", linewidth=0.5, label="TP"),
        Patch(facecolor="#440154", edgecolor="0.4", linewidth=0.5, label="TN"),
        Patch(facecolor="#35B779", edgecolor="0.4", linewidth=0.5, label="FP"),
        Patch(facecolor="#31688E", edgecolor="0.4", linewidth=0.5, label="FN"),
        Patch(facecolor="#D3D3D3", edgecolor="0.4", linewidth=0.5, label="Separatrix"),
    ]
    fig.legend(
        handles=legend_patches, loc="lower center",
        ncol=5, fontsize=18, frameon=False, prop={"weight": "bold", "size": 18},
        bbox_to_anchor=(0.56, -0.01), handletextpad=0.5, columnspacing=1.0,
        handleheight=1.5, handlelength=2.0,
    )

    tag = f"_{suffix}" if suffix else ""
    _save_fig(fig, f"combined_comparison_errors_vert{tag}_ep{epoch}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Combined raw V figure (DeepReach only -- diverging colormap)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_value_combined(all_data: dict, deepreach_label: str, vmin=None, vmax=None):
    """Standalone DeepReach V(x,t) heatmap: rows=slices, single column."""
    from matplotlib.colors import LinearSegmentedColormap
    slice_names = list(all_data.keys())
    value_cmap = LinearSegmentedColormap.from_list(
        "value_div", ["#d73027", "#fc8d59", "#ffffbf", "#91bfdb", "#4575b4"]
    )

    n_rows = len(slice_names)
    fig, axes = plt.subplots(
        n_rows, 1, figsize=(4.5, 3.0 * n_rows),
        squeeze=False, constrained_layout=True,
    )

    im = None
    for row, slice_name in enumerate(slice_names):
        sl = SLICES[slice_name]
        d = all_data[slice_name]
        vals_2d = d["deepreach_values"]
        ext = [*sl["sweep_ranges"][0], *sl["sweep_ranges"][1]]

        vlim = max(abs(vmin or vals_2d.min()), abs(vmax or vals_2d.max()))
        v_lo = vmin if vmin is not None else -vlim
        v_hi = vmax if vmax is not None else vlim

        ax = axes[row, 0]
        im = ax.imshow(
            vals_2d.T, origin="lower", extent=ext, aspect="auto",
            cmap=value_cmap, vmin=v_lo, vmax=v_hi, interpolation="bilinear",
        )
        # Zero-level contour
        axis0, axis1 = d["deepreach_axis0"], d["deepreach_axis1"]
        try:
            ax.contour(axis0, axis1, vals_2d.T, levels=[0.0], colors="k", linewidths=1.0)
        except ValueError:
            pass
        # Threshold contours
        for level, ls in [(DEEPREACH_C_LOW, "--"), (DEEPREACH_C_HIGH, ":")]:
            try:
                ax.contour(axis0, axis1, vals_2d.T, levels=[level], colors="w",
                           linewidths=0.8, linestyles=ls)
            except ValueError:
                pass
        ax.set_xlabel(sl["xlabel"], fontsize=13, fontweight="bold")
        ax.set_ylabel(sl["ylabel"], fontsize=13, fontweight="bold")
        _style_ax(ax, sl)

    axes[0, 0].set_title(deepreach_label, fontsize=14, fontweight="bold")

    if im is not None:
        cbar = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
        cbar.set_label(r"$V(x, t)$", fontsize=13, fontweight="bold")
        cbar.ax.tick_params(labelsize=11)

    _save_fig(fig, "combined_comparison_value")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Combined CartPole comparison: GT vs Olympics-Classifier vs baselines"
    )
    parser.add_argument(
        "--deepreach_cache_dir", type=str,
        default=str(Path(__file__).resolve().parents[1].parent
                    / "deepreach" / "results" / "figures"),
        help="Directory containing DeepReach .npz caches (default: ../deepreach/results/figures)",
    )
    parser.add_argument("--epoch", type=int, default=DEFAULT_EPOCH,
                        help="Olympics-classifier epoch to use (default: 15)")
    parser.add_argument("--slice", default="theta_thetadot",
                        choices=list(SLICES.keys()))
    parser.add_argument("--all_slices", action="store_true",
                        help="Use both slices")
    parser.add_argument(
        "--baselines", nargs="*", default=None,
        choices=["lyapunov", "deepreach", "none"],
        help="Baselines to include (default: olympics-only). "
             "E.g. --baselines lyapunov deepreach",
    )
    # Legacy flag (deprecated, use --baselines instead)
    parser.add_argument("--no_deepreach", action="store_true",
                        help="(Deprecated) Skip DeepReach. Use --baselines instead.")
    parser.add_argument("--deepreach_label", default="DeepReach",
                        help="Column label for DeepReach (default: DeepReach)")
    parser.add_argument("--c_low", type=float, default=DEEPREACH_C_LOW,
                        help=f"Calibrated safe threshold V <= c_low (default: {DEEPREACH_C_LOW})")
    parser.add_argument("--c_high", type=float, default=DEEPREACH_C_HIGH,
                        help=f"Calibrated unsafe threshold V >= c_high (default: {DEEPREACH_C_HIGH})")
    parser.add_argument("--c_low_alt", type=float, default=-0.1,
                        help="Alt safe threshold V <= c_low_alt (default: -0.1)")
    parser.add_argument("--c_high_alt", type=float, default=0.0,
                        help="Alt unsafe threshold V >= c_high_alt (default: 0.0)")
    parser.add_argument("--vmin", type=float, default=None,
                        help="Value heatmap colorbar lower bound")
    parser.add_argument("--vmax", type=float, default=None,
                        help="Value heatmap colorbar upper bound")
    parser.add_argument("--no_value", action="store_true",
                        help="Skip raw V heatmap figure")
    parser.add_argument("--no_errors", action="store_true",
                        help="Skip FP/FN error analysis plots")
    args = parser.parse_args()

    # ── Resolve which baselines are active ──────────────────────────────
    if args.baselines is None:
        # --baselines not specified at all → olympics-only
        active_baselines = set()
    else:
        active_baselines = set(args.baselines) - {"none"}

    if args.no_deepreach:
        active_baselines.discard("deepreach")

    use_lyapunov = "lyapunov" in active_baselines
    use_deepreach = "deepreach" in active_baselines

    slice_names = list(SLICES.keys()) if args.all_slices else [args.slice]

    # ══════════════════════════════════════════════════════════════════════
    # Load all data
    # ══════════════════════════════════════════════════════════════════════
    all_data = {}
    has_deepreach = use_deepreach  # may be set False if cache missing

    for slice_name in slice_names:
        print(f"\n{'='*60}")
        print(f"Slice: {slice_name}")
        print(f"{'='*60}")

        # Olympics-classifier
        print("  Loading olympics-classifier cache...")
        gt_labels, n0, n1, olympics_methods = load_olympics_cache(slice_name, args.epoch)
        gt_heatmap = _gt_to_heatmap(gt_labels, n0, n1)

        entry = {
            "gt_heatmap": gt_heatmap,
            "gt_labels": gt_labels,   # flat, binary {0, 1}
            "n0": n0, "n1": n1,
            "olympics": olympics_methods,
        }

        # Lyapunov NN (optional)
        if use_lyapunov:
            print("  Loading Lyapunov NN...")
            lyap_labels, lyap_values, ln0, ln1 = load_lyapunov_csv(slice_name)
            assert ln0 == n0 and ln1 == n1, (
                f"Grid mismatch: Olympics {n0}x{n1} vs Lyapunov {ln0}x{ln1}"
            )
            entry["lyapunov_heatmap"] = _3class_to_heatmap(lyap_labels, n0, n1)
            entry["lyapunov_labels"] = lyap_labels

        # DeepReach (optional) — load from CSV
        if has_deepreach:
            print("  Loading DeepReach CSV...")
            try:
                dr_labels, dr_values, dn0, dn1 = load_deepreach_csv(slice_name)
                assert dn0 == n0 and dn1 == n1, (
                    f"Grid mismatch: Olympics {n0}x{n1} vs DeepReach {dn0}x{dn1}"
                )
                entry["deepreach_heatmap"] = _3class_to_heatmap(dr_labels, n0, n1)
                entry["deepreach_labels"] = dr_labels
            except FileNotFoundError as e:
                print(f"  Warning: {e}, skipping DeepReach")
                has_deepreach = False

        all_data[slice_name] = entry

    # ══════════════════════════════════════════════════════════════════════
    # Build baselines list for figures
    # ══════════════════════════════════════════════════════════════════════
    baselines = []

    if use_lyapunov:
        baselines.append({
            "label": "Lyapunov NN",
            "heatmap_key": "lyapunov_heatmap",
            "labels_key": "lyapunov_labels",
            "failure_value": -1,   # Lyapunov: 1=success, -1=failure, 0=uncertain
        })

    if has_deepreach:
        baselines.append({
            "label": args.deepreach_label,
            "heatmap_key": "deepreach_heatmap",
            "labels_key": "deepreach_labels",
            "failure_value": -1,   # DeepReach: 1=success, -1=failure, 0=uncertain
        })

    # Build file-name suffix from active baselines
    bl_names = []
    if use_lyapunov:
        bl_names.append("lyap")
    if has_deepreach:
        bl_names.append("dr")
    bl_suffix = "_".join(bl_names) if bl_names else "olympics"

    # ══════════════════════════════════════════════════════════════════════
    # Combined discrete figures
    # ══════════════════════════════════════════════════════════════════════
    print(f"\nGenerating horizontal discrete figure ({bl_suffix})...")
    plot_discrete_horizontal(all_data, args.epoch, baselines, suffix=bl_suffix)

    print(f"Generating vertical discrete figure ({bl_suffix})...")
    plot_discrete_vertical(all_data, args.epoch, baselines, suffix=bl_suffix)

    # ══════════════════════════════════════════════════════════════════════
    # Error analysis (FP/FN)
    # ══════════════════════════════════════════════════════════════════════
    if not args.no_errors:
        print(f"\nGenerating error analysis (FP/FN) figures...")
        plot_error_horizontal(all_data, args.epoch, baselines, suffix=bl_suffix)
        plot_error_vertical(all_data, args.epoch, baselines, suffix=bl_suffix)

    # ══════════════════════════════════════════════════════════════════════
    # DeepReach V heatmap (standalone)
    # ══════════════════════════════════════════════════════════════════════
    # V heatmap only available with legacy .npz caches (not CSV)
    first_slice = next(iter(all_data.values()))
    if has_deepreach and not args.no_value and "deepreach_values" in first_slice:
        print(f"Generating DeepReach V heatmap figure...")
        plot_value_combined(all_data, args.deepreach_label,
                            vmin=args.vmin, vmax=args.vmax)

    print("\nDone.")


if __name__ == "__main__":
    main()
