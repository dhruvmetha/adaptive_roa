#!/usr/bin/env python
"""Figures for the stochastic-pendulum arm comparison.

Learning curves are the headline: debiased Brier and skill score against
acquisition epoch, one line per arm. Epoch index is the data budget, so a
vertical slice is a matched-budget comparison and the gap between the
non-adaptive line and an adaptive one is exactly what the campaign asks.

Reliability and risk-coverage curves are drawn at the matched epoch, where every
arm of a (predictor, level) pair has the same amount of data.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from stoch_compare_report import ARM_LABEL, ARM_ORDER, DATA, LEVELS, discover
from stoch_prob_metrics import (
    load_ground_truth,
    match_to_truth,
    reliability_curve,
    rc_curve,
)

COLOR = {
    "dir00": "#444444",
    "ent05": "#1f77b4",
    "ent10": "#d62728",
    "tb05": "#2ca02c",
    "tb10": "#ff7f0e",
}
STYLE = {"dir00": "--", "ent05": "-", "ent10": "-", "tb05": "-", "tb10": "-"}


def learning_curves(rows: list[dict], out: Path, key: str, ylabel: str,
                    logy: bool = False) -> None:
    for pred in ("fm", "clf"):
        sub = [r for r in rows if r["predictor"] == pred]
        if not sub:
            continue
        fig, axes = plt.subplots(1, 4, figsize=(19, 4.2), sharey=True)
        for ax, lvl in zip(axes, LEVELS):
            any_line = False
            for arm in ARM_ORDER:
                pts = sorted((r["epoch"], r[key]) for r in sub
                             if r["level"] == lvl and r["arm"] == arm and r.get(key) is not None)
                if not pts:
                    continue
                x, y = zip(*pts)
                ax.plot(x, y, STYLE[arm], color=COLOR[arm], marker="o", ms=3.5,
                        lw=2 if arm == "dir00" else 1.6, label=ARM_LABEL[arm])
                any_line = True
            ax.set_title(f"{lvl} noise")
            ax.set_xlabel("acquisition epoch")
            ax.grid(alpha=0.3)
            if logy and any_line:
                ax.set_yscale("log")
        axes[0].set_ylabel(ylabel)
        axes[-1].legend(fontsize=8, loc="best")
        fig.suptitle(f"{'Flow matching' if pred == 'fm' else 'Classifier'} — {ylabel} "
                     f"vs acquisition budget", y=1.02)
        fig.tight_layout()
        fig.savefig(out / f"learning_{key}_{pred}.png", dpi=140, bbox_inches="tight")
        plt.close(fig)


def curves_at_matched(arms: list[dict], depth: dict, out: Path) -> None:
    for pred in ("fm", "clf"):
        for kind in ("reliability", "risk_coverage"):
            fig, axes = plt.subplots(1, 4, figsize=(19, 4.2))
            drew = False
            for ax, lvl in zip(axes, LEVELS):
                peers = [d for (p, l, _), d in depth.items() if p == pred and l == lvl]
                if not peers:
                    continue
                matched = min(peers)
                gt = load_ground_truth(DATA / lvl)
                for arm in ARM_ORDER:
                    a = next((x for x in arms if x["predictor"] == pred
                              and x["level"] == lvl and x["arm"] == arm), None)
                    if a is None:
                        continue
                    ed = a["run_dir"] / f"epoch_{matched:03d}"
                    f = ed / "full_roa_per_point.npz"
                    if not f.exists():
                        continue
                    with np.load(f) as z:
                        states, p_hat = z["start_states"], z["p_success"].astype(np.float64)
                    idx = match_to_truth(states, gt[0])
                    p_true = gt[1][idx]
                    if kind == "reliability":
                        c = reliability_curve(p_hat, p_true)
                        # Marker area tracks how many cells are in each bin. Without
                        # it the eye reads a deep sag in a bin holding 0.1% of the
                        # data as catastrophic miscalibration, when the debiased
                        # REL term (which is count-weighted) is near zero.
                        frac = c["count"] / c["count"].sum()
                        ax.plot(c["p_hat"], c["p_true"], "-", color=COLOR[arm],
                                lw=1.4, label=ARM_LABEL[arm], alpha=0.85)
                        ax.scatter(c["p_hat"], c["p_true"], s=8 + 400 * frac,
                                   color=COLOR[arm], alpha=0.55, edgecolors="none")
                    else:
                        k = 100.0 if pred == "fm" else None
                        c = rc_curve(p_hat, p_true, k, 90.0)
                        ax.plot(c["coverage"], c["risk"], color=COLOR[arm], lw=1.6,
                                label=ARM_LABEL[arm])
                    drew = True
                if kind == "reliability":
                    ax.plot([0, 1], [0, 1], ":", color="gray", lw=1)
                    ax.set_xlabel("predicted p(success)  (marker area ∝ bin count)")
                    ax.set_ylabel("observed p(success)")
                else:
                    ax.set_xlabel("coverage")
                    ax.set_ylabel("debiased selective risk")
                ax.set_title(f"{lvl} noise (epoch {matched})")
                ax.grid(alpha=0.3)
            if not drew:
                plt.close(fig)
                continue
            axes[-1].legend(fontsize=8, loc="best")
            fig.suptitle(f"{'Flow matching' if pred == 'fm' else 'Classifier'} — "
                         f"{kind.replace('_', ' ')} at matched epoch", y=1.02)
            fig.tight_layout()
            fig.savefig(out / f"{kind}_{pred}.png", dpi=140, bbox_inches="tight")
            plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", type=Path, default=Path("docs/stoch_compare/metrics.json"))
    ap.add_argument("--out", type=Path, default=Path("docs/stoch_compare"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rows = json.loads(args.metrics.read_text())
    arms = discover()
    depth = {}
    for a in arms:
        eps = [int(d.name.split("_")[1]) for d in sorted(a["run_dir"].glob("epoch_*"))
               if (d / "full_roa_per_point.npz").exists()]
        if eps:
            depth[(a["predictor"], a["level"], a["arm"])] = max(eps)

    learning_curves(rows, args.out, "brier_debiased", "debiased Brier")
    learning_curves(rows, args.out, "skill_score", "skill score")
    learning_curves(rows, args.out, "sAUROC", "soft AUROC")
    curves_at_matched(arms, depth, args.out)
    print(f"wrote figures to {args.out}")


if __name__ == "__main__":
    main()
