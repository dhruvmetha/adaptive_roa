#!/usr/bin/env python
"""Is the adaptive-vs-random gap bigger than ordinary run-to-run noise?

Every arm in the main campaign is a single run, so a small gap between two arms
could just be the difference between two random initialisations. This scores
seed replicates of the non-adaptive and the fully-adaptive arm at the same
level, and puts the arm gap next to the seed-to-seed spread.

For the non-adaptive arm the seed changes only training stochasticity (its data
order is fixed by the shuffle file), so its spread is a clean training-noise
floor. For the adaptive arm the seed also perturbs which states get acquired,
so its spread is the full run-to-run variability of the procedure.

Usage:
    python scripts/seed_variance.py --level med --out docs/stoch_compare/seed_variance.md
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from stoch_compare_report import DATA, EXP
from stoch_prob_metrics import epoch_dirs, load_ground_truth, score_epoch

SEED_ROOT = EXP / "stoch_compare_seeds"
# Minimum replicates per arm before a comparison epoch is usable.
MIN_SEEDS = 3
MAIN_SEED = 42


def collect(pred: str, level: str, arm: str) -> dict[int, Path]:
    """Map seed -> run dir, taking the main campaign run as the seed-42 member.

    The seed-42 member is taken from Amarel when available, even if the iLab copy
    is deeper. Every replicate runs on Amarel, and this comparison exists to
    isolate seed-to-seed variation — folding in a run from a different GPU
    architecture would put hardware differences into the "seed" SD and make the
    significance verdict mean something other than what it says.
    """
    runs: dict[int, Path] = {}
    for root in (EXP / "stoch_compare_amarel", EXP / "stoch_compare"):
        d = root / f"{pred}_{level}_{arm}"
        if d.exists() and epoch_dirs(d) and MAIN_SEED not in runs:
            runs[MAIN_SEED] = d
    if SEED_ROOT.exists():
        for d in SEED_ROOT.glob(f"{pred}_{level}_{arm}_s*"):
            if epoch_dirs(d):
                runs[int(d.name.rsplit("_s", 1)[1])] = d
    return runs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", nargs="*", default=["med", "high", "xhigh"])
    ap.add_argument("--arms", nargs="*", default=["dir00", "ent10"])
    ap.add_argument("--out", type=Path, default=Path("docs/stoch_compare/seed_variance.md"))
    args = ap.parse_args()

    md = ["# Seed replicates", "",
          "A difference between arms only means something if it is larger than the "
          "spread you get by rerunning the same arm with a different seed. "
          "`dir00` replicates change training stochasticity only (its data order is "
          "fixed), so their spread is the training-noise floor.", ""]
    for level in args.levels:
        md += section_for_level(level, args.arms)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(md))
    print(f"wrote {args.out}")
    print("\n".join(md[-8:]))


def section_for_level(level: str, arm_names: list[str]) -> list[str]:
    gt = load_ground_truth(DATA / level)
    md: list[str] = []
    args = argparse.Namespace(level=level, arms=arm_names)

    table: dict[tuple[str, str], dict[int, dict[int, float]]] = {}
    for pred in ("fm", "clf"):
        for arm in args.arms:
            runs = collect(pred, args.level, arm)
            if not runs:
                continue
            per_seed: dict[int, dict[int, float]] = {}
            for seed, d in sorted(runs.items()):
                scores = {}
                for ed in epoch_dirs(d):
                    try:
                        r = score_epoch(ed, gt, None, None)
                    except Exception:
                        continue
                    scores[int(ed.name.split("_")[1])] = r["brier_debiased"]
                if scores:
                    per_seed[seed] = scores
            if per_seed:
                table[(pred, arm)] = per_seed

    for pred in ("fm", "clf"):
        arms = {a: table[(pred, a)] for a in args.arms if (pred, a) in table}
        if not arms:
            continue
        # Use the deepest epoch where every arm has at least MIN_SEEDS replicates,
        # rather than an epoch shared by ALL seeds. A replicate that is behind, or
        # one whose early artifacts were lost, would otherwise empty the
        # intersection and silently suppress the whole comparison -- the seeds
        # that do have the epoch still carry a perfectly good variance estimate.
        depths = [max(e for s in v.values() for e in s) for v in arms.values()]
        usable = [c for c in range(max(depths), -1, -1)
                  if all(sum(c in s for s in v.values()) >= MIN_SEEDS for v in arms.values())]
        if not usable:
            continue
        ep = usable[0]
        md += [f"## {level} — {'Flow matching' if pred == 'fm' else 'Classifier'} "
               f"— epoch {ep}", "",
               "| arm | seeds | debiased Brier per seed | mean | SD |", "|---|---|---|---|---|"]
        stats = {}
        for arm, per_seed in arms.items():
            vals = np.array([per_seed[s][ep] for s in sorted(per_seed) if ep in per_seed[s]])
            stats[arm] = vals
            md.append(f"| {arm} | {len(vals)} | "
                      + ", ".join(f"{v:.5f}" for v in vals)
                      + f" | {vals.mean():.5f} | {vals.std(ddof=1) if len(vals) > 1 else float('nan'):.5f} |")
        md.append("")
        if len(stats) == 2:
            a, b = args.arms[0], args.arms[1]
            if a in stats and b in stats:
                gap = stats[b].mean() - stats[a].mean()
                replicated = [v for v in stats.values() if len(v) > 1]
                if not replicated:
                    # No verdict is possible without at least two seeds: comparing
                    # against a NaN SD silently reads as "not significant".
                    md += [f"Gap ({b} − {a}) = **{gap:+.5f}**. No seed SD yet "
                           f"(need >=2 seeds per arm; replicates still running), so "
                           f"**no significance claim can be made**.", ""]
                else:
                    pooled = float(np.sqrt(np.mean([v.var(ddof=1) for v in replicated])))
                    sig = abs(gap) > 2 * pooled
                    md += [f"Gap ({b} − {a}) = **{gap:+.5f}**; pooled seed SD = "
                           f"{pooled:.5f} (2 SD = {2 * pooled:.5f}). The gap is "
                           + ("larger than 2 seed SDs, so it is distinguishable from "
                              "run-to-run noise." if sig else
                              "smaller than 2 seed SDs, so it is **not** distinguishable "
                              "from run-to-run noise."), ""]
                md += _stability(arms, usable[:4], args.arms)
    return md


def _stability(arms: dict, epochs: list[int], arm_names: list[str]) -> list[str]:
    """Repeat the verdict at the last few usable epochs.

    A verdict computed at one epoch with three seeds can flip on the next one: FM
    at high noise was significant at epoch 2 and not at epoch 3, because a single
    replicate landed three times higher than its siblings. Showing consecutive
    epochs makes a fragile verdict visible instead of letting whichever epoch
    happened to be deepest speak for the result.
    """
    a, b = arm_names[0], arm_names[1]
    if a not in arms or b not in arms or len(epochs) < 2:
        return []
    out = ["Stability across the last usable epochs:", "",
           "The 2xSD test uses a pooled standard deviation, which one outlier seed can "
           "inflate enough to hide a real separation. `ranges disjoint?` is the "
           "non-parametric companion: it asks whether every seed of one arm beats every "
           "seed of the other, which no single outlier can fake in the favourable "
           "direction. Trust a verdict that both columns agree on.", "",
           "| epoch | gap | 2×seed SD | significant? | ranges disjoint? |",
           "|---|---|---|---|---|"]
    for ep in sorted(epochs):
        va = np.array([s[ep] for s in arms[a].values() if ep in s])
        vb = np.array([s[ep] for s in arms[b].values() if ep in s])
        if len(va) < 2 or len(vb) < 2:
            continue
        gap = vb.mean() - va.mean()
        pooled = float(np.sqrt(np.mean([va.var(ddof=1), vb.var(ddof=1)])))
        disjoint = vb.max() < va.min() or va.max() < vb.min()
        out.append(f"| {ep} | {gap:+.5f} | {2 * pooled:.5f} | "
                   f"{'**yes**' if abs(gap) > 2 * pooled else 'no'} | "
                   f"{'**yes**' if disjoint else 'no'} |")
    return out + [""] if len(out) > 6 else []


if __name__ == "__main__":
    main()
