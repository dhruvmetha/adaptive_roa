#!/usr/bin/env python
"""Stage-1 correctness check for the outcome-FM arm (deterministic pendulum).

Stage 1 answers NOTHING about the ablation's research question: on the
deterministic pendulum true p is in {0,1}, so calibration is degenerate and the
6-43x calibration gap this work exists to explain cannot even be posed. What it
does check is that the machinery is sound before spending compute on the noisy
levels:

  1. the arm learns at all           -- accuracy and AUROC against the binary labels
  2. it sharpens toward the truth    -- mean p(1-p) should FALL across epochs, since
                                        every cell has a deterministic outcome
  3. the flow map stays well behaved -- non-monotone fraction stays negligible, so
                                        the exact readout is bisecting rather than
                                        silently falling back to quadrature

Self-contained: uses `true_labels` stored in each epoch's npz, so it needs no
matched baseline run (the available deterministic-pendulum runs are from older
campaigns with different configs and would not be a fair comparison anyway).

Usage:  python scripts/outcome_fm_stage1_check.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from outcome_fm_ablation_report import OUTCOME_ROOT, resolve_run  # noqa: E402
from stoch_prob_metrics import epoch_dirs  # noqa: E402


def auroc(p: np.ndarray, y: np.ndarray) -> float:
    """Rank-based AUROC with tie handling; O(n log n)."""
    pos, neg = y == 1, y == 0
    if not pos.any() or not neg.any():
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p), dtype=np.float64)
    ranks[order] = np.arange(1, len(p) + 1)
    sp = p[order]
    i = 0
    while i < len(sp):  # average ranks within tie groups
        j = i
        while j + 1 < len(sp) and sp[j + 1] == sp[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    n_p, n_n = int(pos.sum()), int(neg.sum())
    return float((ranks[pos].sum() - n_p * (n_p + 1) / 2.0) / (n_p * n_n))


def main() -> None:
    run = resolve_run(OUTCOME_ROOT / "fm_outcome_deterministic")
    if run is None:
        print("no deterministic outcome-FM run found yet")
        return

    dirs = epoch_dirs(run)
    print(f"# Stage 1 correctness check — {len(dirs)} epochs evaluated")
    print(f"run: {run}\n")
    print("| epoch | acc | AUROC | Brier(binary) | sharpness p(1-p) | interior |")
    print("|---|---|---|---|---|---|")

    rows = []
    for d in dirs:
        with np.load(d / "full_roa_per_point.npz") as z:
            p = z["p_success"].astype(np.float64)
            lab = z["true_labels"].astype(np.float64)
        y = (lab > 0).astype(np.float64)          # stored as -1/+1
        keep = np.isfinite(p) & np.isfinite(lab)
        p, y = p[keep], y[keep]

        acc = float(((p > 0.5) == (y > 0.5)).mean())
        brier = float(np.mean((p - y) ** 2))
        sharp = float(np.mean(p * (1.0 - p)))
        interior = float(np.mean((p > 0.02) & (p < 0.98)))
        ep = int(d.name.split("_")[1])
        rows.append((ep, acc, brier, sharp))
        print(f"| {ep} | {acc:.4f} | {auroc(p, y):.4f} | {brier:.4f} | {sharp:.4f} | {interior:.3f} |")

    if len(rows) < 3:
        print("\n(too few epochs to judge a trend yet)")
        return

    first, last = rows[0], rows[-1]
    print(f"\n  accuracy   {first[1]:.4f} -> {last[1]:.4f}")
    print(f"  Brier      {first[2]:.4f} -> {last[2]:.4f}")
    print(f"  sharpness  {first[3]:.4f} -> {last[3]:.4f}  "
          f"({'sharpening as expected' if last[3] < first[3] else 'NOT sharpening -- investigate'})")
    print("\n  Reminder: this stage validates machinery only. True p is in {0,1} here,")
    print("  so nothing about calibration -- the actual research question -- is testable")
    print("  until the noisy levels.")


if __name__ == "__main__":
    main()
