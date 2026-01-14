"""
Plot adaptive-run metrics vs data used.

For a standalone adaptive run directory (with epoch_XXX subfolders), this script:
- Reads epoch_XXX/results.json (train_trajectories, n_d1_added, n_d2_added)
- Reads epoch_XXX/full_roa_evaluation.json (conformal_thresholds + notebook_thresholds)
- Computes "data used" for training at epoch start:
    data_used = train_trajectories(end_of_epoch) - n_d1_added - n_d2_added
- Produces four plots:
    1) data_used vs F1 (varying λ*±δ*)
    2) data_used vs separatrix% (varying λ*±δ*)
    3) data_used vs F1 (fixed notebook thresholds)
    4) data_used vs separatrix% (fixed notebook thresholds)

Usage:
  python scripts/plot_data_vs_metrics.py /abs/path/to/run_dir
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class EpochRow:
    epoch: int
    data_used: int
    data_end: int
    lambda_star: float
    delta: float
    varying_f1: float
    varying_sep: float
    fixed_f1: float
    fixed_sep: float


def _read_json(path: Path) -> Dict:
    return json.loads(path.read_text())


def _safe_get(d: Dict, keys: Tuple[str, ...]) -> Optional[float]:
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def load_epoch_rows(run_dir: Path) -> List[EpochRow]:
    rows: List[EpochRow] = []
    epoch_dirs = sorted(
        [p for p in run_dir.glob("epoch_*") if p.is_dir()],
        key=lambda p: int(p.name.split("_")[1]),
    )
    for ep_dir in epoch_dirs:
        epoch = int(ep_dir.name.split("_")[1])
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

        lambda_star = float(roa.get("lambda_star", float("nan")))
        delta = float(roa.get("delta", float("nan")))

        varying_f1 = _safe_get(roa, ("conformal_thresholds", "f1"))
        varying_sep = _safe_get(roa, ("conformal_thresholds", "separatrix_pct"))
        fixed_f1 = _safe_get(roa, ("notebook_thresholds", "f1"))
        fixed_sep = _safe_get(roa, ("notebook_thresholds", "separatrix_pct"))

        # Require all metrics for the requested plots
        if None in (varying_f1, varying_sep, fixed_f1, fixed_sep):
            continue

        rows.append(
            EpochRow(
                epoch=epoch,
                data_used=int(data_used),
                data_end=int(train_end),
                lambda_star=lambda_star,
                delta=delta,
                varying_f1=float(varying_f1),
                varying_sep=float(varying_sep),
                fixed_f1=float(fixed_f1),
                fixed_sep=float(fixed_sep),
            )
        )

    # Sort by epoch (stable) then by data_used
    rows.sort(key=lambda r: r.epoch)
    return rows


def _plot_xy(
    *,
    xs: List[int],
    ys: List[float],
    xlabel: str,
    ylabel: str,
    title: str,
    out_path: Path,
    annotate_epochs: Optional[List[int]] = None,
):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7.5, 5.0))
    plt.plot(xs, ys, marker="o", linewidth=2)
    plt.grid(True, alpha=0.3)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    if annotate_epochs is not None:
        for x, y, ep in zip(xs, ys, annotate_epochs):
            plt.annotate(str(ep), (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8, alpha=0.8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def main(argv: List[str]) -> int:
    if len(argv) != 2:
        print("Usage: python scripts/plot_data_vs_metrics.py /abs/path/to/run_dir", file=sys.stderr)
        return 2

    run_dir = Path(argv[1]).expanduser().resolve()
    if not run_dir.exists():
        print(f"Run dir not found: {run_dir}", file=sys.stderr)
        return 2

    rows = load_epoch_rows(run_dir)
    if not rows:
        print(f"No epochs with both results.json and full_roa_evaluation.json found in: {run_dir}", file=sys.stderr)
        return 1

    xs = [r.data_used for r in rows]
    epochs = [r.epoch for r in rows]

    plots_dir = run_dir / "plots"

    _plot_xy(
        xs=xs,
        ys=[r.varying_f1 for r in rows],
        xlabel="Training data used (start-of-epoch trajectories)",
        ylabel="F1",
        title="Data used vs F1 (varying λ*±δ*)",
        out_path=plots_dir / "data_used_vs_f1_varying.png",
        annotate_epochs=epochs,
    )
    _plot_xy(
        xs=xs,
        ys=[100.0 * r.varying_sep for r in rows],
        xlabel="Training data used (start-of-epoch trajectories)",
        ylabel="Separatrix (%)",
        title="Data used vs Separatrix (varying λ*±δ*)",
        out_path=plots_dir / "data_used_vs_sep_varying.png",
        annotate_epochs=epochs,
    )
    _plot_xy(
        xs=xs,
        ys=[r.fixed_f1 for r in rows],
        xlabel="Training data used (start-of-epoch trajectories)",
        ylabel="F1",
        title="Data used vs F1 (fixed notebook thresholds)",
        out_path=plots_dir / "data_used_vs_f1_fixed.png",
        annotate_epochs=epochs,
    )
    _plot_xy(
        xs=xs,
        ys=[100.0 * r.fixed_sep for r in rows],
        xlabel="Training data used (start-of-epoch trajectories)",
        ylabel="Separatrix (%)",
        title="Data used vs Separatrix (fixed notebook thresholds)",
        out_path=plots_dir / "data_used_vs_sep_fixed.png",
        annotate_epochs=epochs,
    )

    print(f"Saved plots to: {plots_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

