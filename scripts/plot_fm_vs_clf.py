#!/usr/bin/env python
"""Presentation figures comparing flow matching against the classifier.

Three figures, each answering one question:

  1. summary   — how do they differ across noise levels? (the calibration-vs-ranking point)
  2. curves    — how does that difference develop with acquisition budget?
  3. divergence— does entropy acquisition help or hurt, per predictor?

All comparisons use the non-adaptive arm unless stated, so the acquisition
strategy is held fixed and only the predictor varies, and every panel is drawn at
an epoch both predictors have reached.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LEVELS = ["low", "med", "high", "xhigh"]
LEVEL_LABEL = {"low": "low", "med": "medium", "high": "high", "xhigh": "extreme"}
FM_C, CLF_C = "#1f77b4", "#d62728"

plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12,
    "legend.fontsize": 11, "figure.dpi": 140,
})


def load(metrics: Path) -> list[dict]:
    return json.loads(metrics.read_text())


def series(rows, pred, lvl, arm, key):
    d = {r["epoch"]: r[key] for r in rows
         if r["predictor"] == pred and r["level"] == lvl and r["arm"] == arm
         and r.get(key) is not None}
    return d


def matched_epoch(rows, lvl, arm="dir00"):
    f = set(series(rows, "fm", lvl, arm, "brier_debiased"))
    c = set(series(rows, "clf", lvl, arm, "brier_debiased"))
    both = f & c
    return max(both) if both else None


def fig_summary(rows, out: Path):
    """Brier / skill / sAUROC across noise levels. The headline figure."""
    eps = {lvl: matched_epoch(rows, lvl) for lvl in LEVELS}
    lv = [l for l in LEVELS if eps[l] is not None]
    x = np.arange(len(lv))
    w = 0.36

    panels = [
        ("brier_debiased", "debiased Brier  (lower better)", True),
        ("skill_score", "skill score  (higher better)", False),
        ("sAUROC", "soft AUROC  (higher better)", False),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for ax, (key, label, logy) in zip(axes, panels):
        fm = [series(rows, "fm", l, "dir00", key)[eps[l]] for l in lv]
        cl = [series(rows, "clf", l, "dir00", key)[eps[l]] for l in lv]
        ax.bar(x - w / 2, fm, w, label="flow matching", color=FM_C)
        ax.bar(x + w / 2, cl, w, label="classifier", color=CLF_C)
        ax.set_xticks(x)
        ax.set_xticklabels([LEVEL_LABEL[l] for l in lv])
        ax.set_xlabel("process noise")
        ax.set_ylabel(label)
        if logy:
            ax.set_yscale("log")
            # Headroom first, then annotate: on a log axis the default top is only
            # just above the tallest bar, so a label placed above it lands outside
            # the axes and silently disappears -- which happened to the two largest
            # ratios, the ones that matter most.
            lo, hi = min(fm + cl), max(fm + cl)
            ax.set_ylim(lo / 2.5, hi * 4.0)
            for xi, (a, b) in enumerate(zip(fm, cl)):
                ax.annotate(f"{b / a:.0f}× worse", (xi, max(a, b) * 1.45), ha="center",
                            fontsize=11, color="0.2", fontweight="bold")
        ax.grid(alpha=0.3, axis="y")
    axes[2].set_ylim(0.8, 1.02)
    axes[0].legend(loc="upper left")
    axes[0].set_title("Probability accuracy")
    axes[1].set_title("Skill vs the true field")
    axes[2].set_title("Ranking ability")
    fig.suptitle("Flow matching vs classifier — the gap is calibration, not ranking "
                 f"(non-adaptive arm, epoch {eps[lv[0]]}–{eps[lv[-1]]})", y=1.02)
    fig.tight_layout()
    fig.savefig(out / "fm_vs_clf_summary.png", bbox_inches="tight")
    plt.close(fig)


def fig_curves(rows, out: Path):
    """Debiased Brier against acquisition budget, both predictors, per level."""
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.3), sharey=True)
    for ax, lvl in zip(axes, LEVELS):
        for pred, colour, name in (("fm", FM_C, "flow matching"),
                                   ("clf", CLF_C, "classifier")):
            d = series(rows, pred, lvl, "dir00", "brier_debiased")
            if not d:
                continue
            xs = sorted(d)
            ax.plot(xs, [d[e] for e in xs], "-o", ms=3.5, color=colour, label=name)
        ax.set_yscale("log")
        ax.set_title(f"{LEVEL_LABEL[lvl]} noise")
        ax.set_xlabel("acquisition epoch")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("debiased Brier (log)")
    axes[0].legend()
    fig.suptitle("Probability error vs acquisition budget (non-adaptive arm) — "
                 "flow matching stays an order of magnitude lower", y=1.03)
    fig.tight_layout()
    fig.savefig(out / "fm_vs_clf_curves.png", bbox_inches="tight")
    plt.close(fig)


def noise_floor(rows, pred, lvl):
    """Run-to-run spread, measured at epoch 0 where all arms are the same model."""
    v = np.array([r["brier_debiased"] for r in rows
                  if r["predictor"] == pred and r["level"] == lvl and r["epoch"] == 0])
    return float(v.std(ddof=1)) if len(v) >= 2 else 0.0


def fig_divergence(rows, out: Path):
    """Effect of entropy acquisition on each predictor: the headline divergence.

    Plotted as a percentage change, which is what makes the split legible — but a
    percentage hides scale. FM's Brier is ~0.002, so a 50% swing there is 0.001,
    far inside its own run-to-run noise, while the classifier's percentages sit on
    a Brier 30x larger and are real. Bars that do not clear 2x the run-to-run floor
    are therefore drawn hollow and labelled n.s., so an insignificant wobble cannot
    read as an effect.
    """
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    x = np.arange(len(LEVELS))
    w = 0.36
    notes = []
    for pred, off, colour, name in (("fm", -w / 2, FM_C, "flow matching"),
                                    ("clf", +w / 2, CLF_C, "classifier")):
        rel, sig = [], []
        for lvl in LEVELS:
            base = series(rows, pred, lvl, "dir00", "brier_debiased")
            adap = series(rows, pred, lvl, "ent10", "brier_debiased")
            both = set(base) & set(adap)
            if not both:
                rel.append(np.nan); sig.append(False); continue
            ep = max(both)
            delta = adap[ep] - base[ep]
            rel.append(100.0 * delta / base[ep])
            sig.append(abs(delta) > 2 * noise_floor(rows, pred, lvl))
            notes.append(f"{pred} {lvl} ep{ep} delta={delta:+.5f} sig={sig[-1]}")
        for xi, (val, is_sig) in enumerate(zip(rel, sig)):
            if not np.isfinite(val):
                continue
            ax.bar(xi + off, val, w, color=colour if is_sig else "none",
                   edgecolor=colour, hatch=None if is_sig else "///",
                   linewidth=1.6,
                   label=name if xi == 0 else None)
            if not is_sig:
                ax.annotate("n.s.", (xi + off, val + (6 if val >= 0 else -14)),
                            ha="center", fontsize=10, color=colour)
    ax.axhline(0, color="0.3", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([LEVEL_LABEL[l] for l in LEVELS])
    ax.set_xlabel("process noise")
    ax.set_ylabel("change in debiased Brier from fully-adaptive\nacquisition (%, negative = better)")
    ax.set_title("Same acquisition strategy, opposite effect by model class\n"
                 "(hollow = within run-to-run noise, not a real effect)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3, axis="y")
    # Significance here uses the epoch-0 floor, which mixes GPU architectures and
    # is deliberately conservative. Seed replicates hold hardware fixed and are
    # sharper; where they exist they are the better test, and they disagree in one
    # place. Saying so on the figure is better than letting it contradict the
    # write-up.
    fig.text(0.5, -0.06,
             "Significance vs the conservative cross-run floor (epoch-0 spread). "
             "Seed replicates, which hold hardware fixed, are sharper: under those, "
             "flow matching at medium noise IS a real improvement (−0.00094, 2×SD 0.00028).",
             ha="center", fontsize=9.5, color="0.3", wrap=True)
    fig.tight_layout()
    fig.savefig(out / "fm_vs_clf_divergence.png", bbox_inches="tight")
    plt.close(fig)
    return notes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", type=Path, default=Path("docs/stoch_compare/metrics.json"))
    ap.add_argument("--out", type=Path, default=Path("docs/stoch_compare"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = load(args.metrics)
    fig_summary(rows, args.out)
    fig_curves(rows, args.out)
    notes = fig_divergence(rows, args.out)
    print("wrote fm_vs_clf_summary.png, fm_vs_clf_curves.png, fm_vs_clf_divergence.png")
    print("divergence epochs:", ", ".join(sorted(set(notes))))


if __name__ == "__main__":
    main()
