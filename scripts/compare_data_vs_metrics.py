"""
Compare adaptive-run metrics vs data used across TWO standalone run directories.

Loads per-epoch:
  - epoch_XXX/results.json (train_trajectories, n_d1_added, n_d2_added)
  - epoch_XXX/full_roa_evaluation.json (conformal_thresholds + notebook_thresholds)

Computes "data used" at epoch start:
  data_used = train_trajectories(end_of_epoch) - n_d1_added - n_d2_added

Produces 4 overlay plots with a matched x-axis using the common overlap range:
  - data_used vs F1 (varying λ*±δ*)
  - data_used vs Separatrix% (varying λ*±δ*)
  - data_used vs F1 (fixed notebook thresholds)
  - data_used vs Separatrix% (fixed notebook thresholds)

Usage:
  python scripts/compare_data_vs_metrics.py /abs/runA /abs/runB [--out-dir /abs/out] [--label-a A] [--label-b B]
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class Series:
    label: str
    color: str
    xs: List[int]
    varying_f1: List[float]
    varying_sep: List[float]
    fixed_f1: List[float]
    fixed_sep: List[float]


def _read_json(path: Path) -> Dict:
    return json.loads(path.read_text())


def _safe_get(d: Dict, keys: Tuple[str, ...]) -> Optional[float]:
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def load_series(run_dir: Path, label: str, color: str) -> Series:
    xs: List[int] = []
    varying_f1: List[float] = []
    varying_sep: List[float] = []
    fixed_f1: List[float] = []
    fixed_sep: List[float] = []

    epoch_dirs = sorted(
        [p for p in run_dir.glob("epoch_*") if p.is_dir()],
        key=lambda p: int(p.name.split("_")[1]),
    )
    for ep_dir in epoch_dirs:
        results_path = ep_dir / "results.json"
        full_roa_path = ep_dir / "full_roa_evaluation.json"
        if not results_path.exists() or not full_roa_path.exists():
            continue

        res = _read_json(results_path)
        roa = _read_json(full_roa_path)

        train_end = int(res["train_trajectories"])
        n_d1 = int(res.get("n_d1_added", 0))
        n_d2 = int(res.get("n_d2_added", 0))
        data_used = train_end - n_d1 - n_d2

        vf1 = _safe_get(roa, ("conformal_thresholds", "f1"))
        vsep = _safe_get(roa, ("conformal_thresholds", "separatrix_pct"))
        ff1 = _safe_get(roa, ("notebook_thresholds", "f1"))
        fsep = _safe_get(roa, ("notebook_thresholds", "separatrix_pct"))

        if None in (vf1, vsep, ff1, fsep):
            continue

        xs.append(int(data_used))
        varying_f1.append(float(vf1))
        varying_sep.append(float(vsep))
        fixed_f1.append(float(ff1))
        fixed_sep.append(float(fsep))

    return Series(
        label=label,
        color=color,
        xs=xs,
        varying_f1=varying_f1,
        varying_sep=varying_sep,
        fixed_f1=fixed_f1,
        fixed_sep=fixed_sep,
    )


def _overlap_xlim(series_list: List[Series]) -> Tuple[int, int]:
    mins = [min(s.xs) for s in series_list if s.xs]
    maxs = [max(s.xs) for s in series_list if s.xs]
    if not mins or not maxs:
        raise ValueError("No x values found in one or more series.")
    # Use common overlap (intersection) so axes match for partial runs.
    lo = max(mins)
    hi = min(maxs)  # "use min" (of maxima)
    if lo > hi:
        raise ValueError(f"No overlap in x-range across runs: lo={lo}, hi={hi}")
    return lo, hi


def _filter_to_xlim(s: Series, lo: int, hi: int) -> Series:
    keep = [i for i, x in enumerate(s.xs) if lo <= x <= hi]
    return Series(
        label=s.label,
        color=s.color,
        xs=[s.xs[i] for i in keep],
        varying_f1=[s.varying_f1[i] for i in keep],
        varying_sep=[s.varying_sep[i] for i in keep],
        fixed_f1=[s.fixed_f1[i] for i in keep],
        fixed_sep=[s.fixed_sep[i] for i in keep],
    )


def _plot_overlay(
    *,
    series_list: List[Series],
    y_getter,
    xlabel: str,
    ylabel: str,
    title: str,
    out_path: Path,
    xlim: Tuple[int, int],
    y_scale: float = 1.0,
    hlines: Optional[List[Tuple[float, str, str]]] = None,
    ylim: Optional[Tuple[float, float]] = None,
):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7.8, 5.2))
    for s in series_list:
        ys = [y_scale * v for v in y_getter(s)]
        plt.plot(s.xs, ys, marker="o", linewidth=2, label=s.label, color=s.color)

    if hlines:
        for y, label, color in hlines:
            plt.axhline(
                y=y,
                linestyle="--",
                linewidth=2.0,
                alpha=0.9,
                color=color,
                label=label,
            )

    plt.grid(True, alpha=0.3)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    plt.legend()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare data_used vs metrics across two adaptive run directories.")
    ap.add_argument("run_a", type=str)
    ap.add_argument("run_b", type=str)
    ap.add_argument("--label-a", type=str, default=None)
    ap.add_argument("--label-b", type=str, default=None)
    ap.add_argument("--out-dir", type=str, default=None)
    ap.add_argument(
        "--ref-classification-f1",
        type=float,
        default=None,
        help="Reference F1 baseline. Accepts either fraction (0-1) or percent (0-100).",
    )
    ap.add_argument(
        "--ref-classification-sep",
        type=float,
        default=None,
        help="Reference separatrix baseline. Accepts either fraction (0-1) or percent (0-100).",
    )
    ap.add_argument(
        "--ref-nonadaptive-f1",
        type=float,
        default=None,
        help="Reference F1 baseline. Accepts either fraction (0-1) or percent (0-100).",
    )
    ap.add_argument(
        "--ref-nonadaptive-sep",
        type=float,
        default=None,
        help="Reference separatrix baseline. Accepts either fraction (0-1) or percent (0-100).",
    )
    args = ap.parse_args()

    run_a = Path(args.run_a).expanduser().resolve()
    run_b = Path(args.run_b).expanduser().resolve()
    label_a = args.label_a or run_a.name
    label_b = args.label_b or run_b.name

    if not run_a.exists():
        raise SystemExit(f"Run dir A not found: {run_a}")
    if not run_b.exists():
        raise SystemExit(f"Run dir B not found: {run_b}")

    # Explicit colors so the meaning is stable across reruns.
    # Default matplotlib cycle uses blue for first series and orange for second.
    # User requested flipping those, so we set:
    #   run_a (label-a) -> orange, run_b (label-b) -> blue
    s1 = load_series(run_a, label_a, color="#ff7f0e")  # orange
    s2 = load_series(run_b, label_b, color="#1f77b4")  # blue
    if not s1.xs or not s2.xs:
        raise SystemExit("One of the runs has no epochs with both results.json and full_roa_evaluation.json")

    lo, hi = _overlap_xlim([s1, s2])
    s1 = _filter_to_xlim(s1, lo, hi)
    s2 = _filter_to_xlim(s2, lo, hi)

    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (run_a.parent / "comparisons" / f"{run_a.name}__vs__{run_b.name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    xlabel = "Dataset Size"
    xlim = (lo, hi)
    pct_ylim = (0.0, 100.0)

    # Use consistent, high-contrast colors for baseline reference lines.
    # (Chosen to be distinct from the default series colors and from each other.)
    baseline_colors = {
        "classification": "#000000",  # black
        "nonadaptive": "#7A1FA2",      # purple
    }

    def _to_fraction(v: Optional[float]) -> Optional[float]:
        """Interpret either 0-1 fraction or 0-100 percent; return fraction."""
        if v is None:
            return None
        v = float(v)
        return v / 100.0 if v > 1.0 else v

    ref_cf1 = _to_fraction(args.ref_classification_f1)
    ref_nf1 = _to_fraction(args.ref_nonadaptive_f1)
    ref_csep = _to_fraction(args.ref_classification_sep)
    ref_nsep = _to_fraction(args.ref_nonadaptive_sep)

    f1_hlines: List[Tuple[float, str, str]] = []
    sep_hlines: List[Tuple[float, str, str]] = []
    if ref_cf1 is not None:
        f1_hlines.append((100.0 * ref_cf1, "Classification@500 (F1)", baseline_colors["classification"]))
    if ref_nf1 is not None:
        f1_hlines.append((100.0 * ref_nf1, "Non-adaptive@500 (F1)", baseline_colors["nonadaptive"]))
    if ref_csep is not None:
        sep_hlines.append((100.0 * ref_csep, "Classification@500 (Sep%)", baseline_colors["classification"]))
    if ref_nsep is not None:
        sep_hlines.append((100.0 * ref_nsep, "Non-adaptive@500 (Sep%)", baseline_colors["nonadaptive"]))

    _plot_overlay(
        series_list=[s1, s2],
        y_getter=lambda s: s.varying_f1,
        xlabel=xlabel,
        ylabel="F1",
        title="Cartpole: F1 v/s Dataset Size (Varying Thresholds)",
        out_path=out_dir / "compare_data_used_vs_f1_varying.png",
        xlim=xlim,
        y_scale=100.0,
        hlines=f1_hlines if f1_hlines else None,
        ylim=pct_ylim,
    )
    _plot_overlay(
        series_list=[s1, s2],
        y_getter=lambda s: s.varying_sep,
        xlabel=xlabel,
        ylabel="Separatrix (%)",
        title="Cartpole: Separatrix v/s Dataset Size (Varying Thresholds)",
        out_path=out_dir / "compare_data_used_vs_sep_varying.png",
        xlim=xlim,
        y_scale=100.0,
        hlines=sep_hlines if sep_hlines else None,
        ylim=pct_ylim,
    )
    _plot_overlay(
        series_list=[s1, s2],
        y_getter=lambda s: s.fixed_f1,
        xlabel=xlabel,
        ylabel="F1",
        title="Cartpole: F1 v/s Dataset Size (Fixed Thresholds)",
        out_path=out_dir / "compare_data_used_vs_f1_fixed.png",
        xlim=xlim,
        y_scale=100.0,
        hlines=f1_hlines if f1_hlines else None,
        ylim=pct_ylim,
    )
    _plot_overlay(
        series_list=[s1, s2],
        y_getter=lambda s: s.fixed_sep,
        xlabel=xlabel,
        ylabel="Separatrix (%)",
        title="Cartpole: Separatrix v/s Dataset Size (Fixed Thresholds)",
        out_path=out_dir / "compare_data_used_vs_sep_fixed.png",
        xlim=xlim,
        y_scale=100.0,
        hlines=sep_hlines if sep_hlines else None,
        ylim=pct_ylim,
    )

    print(f"Saved comparison plots to: {out_dir}")
    print(f"Matched x-axis (overlap) range: [{lo}, {hi}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

