#!/usr/bin/env python
"""CSV of every metric at the deepest epoch ALL methods share, per noise level.

The matched epoch is the minimum over every (predictor, arm) present at that
level, so each row is a genuinely equal-budget comparison: same number of
acquired training trajectories, same eval grid. Flow matching is the shallower
family, so it usually sets the epoch.

Usage:
    python scripts/export_matched_epoch_csv.py --out docs/stoch_compare/matched_epoch.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from stoch_compare_report import ARM_LABEL, LEVELS, discover
from stoch_prob_metrics import epoch_dirs

# Ordered so the CSV reads left-to-right: identity, budget, headline, then detail.
COLS = [
    "brier_debiased", "brier_debiased_se", "brier_raw", "skill_score",
    "REL_debiased", "RES", "UNC_debiased", "VAR_debiased", "decomp_gap",
    "sAUROC", "SHARP", "SHARP_star",
    "AURC", "risk@0.2", "risk@0.5", "risk@1",
    "KL", "log_score", "log_score_oracle", "log_score_climatology", "MAE", "bias",
    "acc@0.5", "prec@0.5", "rec@0.5", "f1@0.5", "f1@0.25", "f1@0.75",
    "roa_frac_true@0.5", "roa_frac_pred@0.5",
    "mean_p_invalid", "n_points", "K", "M",
]


def train_trajectories(run_dir: Path, epoch: int) -> int | None:
    """Acquired pool size at this epoch, read from the run's own artifact."""
    f = run_dir / f"epoch_{epoch:03d}" / "artifacts_v2.json"
    if not f.exists():
        return None
    try:
        return int(json.loads(f.read_text()).get("train_trajectories"))
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", type=Path, default=Path("docs/stoch_compare/metrics.json"))
    ap.add_argument("--out", type=Path, default=Path("docs/stoch_compare/matched_epoch.csv"))
    ap.add_argument("--per-predictor", action="store_true",
                    help="match within each predictor separately instead of across both")
    args = ap.parse_args()

    rows = json.loads(args.metrics.read_text())
    arms = {(a["predictor"], a["level"], a["arm"]): a for a in discover()}

    depth: dict[tuple[str, str, str], int] = {}
    for key, a in arms.items():
        eps = [int(d.name.split("_")[1]) for d in epoch_dirs(a["run_dir"])]
        if eps:
            depth[key] = max(eps)

    out_rows = []
    for lvl in LEVELS:
        groups = ({"fm": [], "clf": []} if args.per_predictor else {"all": []})
        for (pred, l, arm), d in depth.items():
            if l != lvl:
                continue
            groups[pred if args.per_predictor else "all"].append(d)
        for gname, depths in groups.items():
            if not depths:
                continue
            ep = min(depths)
            for (pred, l, arm), a in sorted(arms.items()):
                if l != lvl or (args.per_predictor and pred != gname):
                    continue
                r = next((x for x in rows if x["predictor"] == pred and x["level"] == lvl
                          and x["arm"] == arm and x["epoch"] == ep), None)
                if r is None:
                    continue
                rec = {
                    "level": lvl,
                    "predictor": pred,
                    "arm": arm,
                    "arm_label": ARM_LABEL[arm],
                    "matched_epoch": ep,
                    "train_trajectories": train_trajectories(a["run_dir"], ep),
                }
                for c in COLS:
                    rec[c] = r.get(c)
                out_rows.append(rec)

    if not out_rows:
        print("no rows — no epoch is shared by every method yet")
        return
    header = (["level", "predictor", "arm", "arm_label", "matched_epoch",
               "train_trajectories"] + COLS)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=header)
        w.writeheader()
        w.writerows(out_rows)
    print(f"wrote {args.out}  ({len(out_rows)} rows)")
    # Report every distinct (level, epoch) group, not just the first row: with
    # --per-predictor the two predictors match at different epochs, and printing
    # only the first would claim they agree when they do not.
    seen: dict[tuple, tuple] = {}
    for r in out_rows:
        seen.setdefault((r["level"], r["predictor"] if args.per_predictor else "all"),
                        (r["matched_epoch"], r["train_trajectories"], 0))
    counts: dict[tuple, int] = {}
    for r in out_rows:
        k = (r["level"], r["predictor"] if args.per_predictor else "all")
        counts[k] = counts.get(k, 0) + 1
    for k in sorted(seen):
        ep, ntr, _ = seen[k]
        print(f"  {k[0]:6s} {k[1]:4s} matched epoch {ep:2d}  "
              f"({ntr} train trajectories)  {counts[k]} methods")


if __name__ == "__main__":
    main()
