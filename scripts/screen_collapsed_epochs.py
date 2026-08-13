#!/usr/bin/env python
"""Flag scored epochs where the model collapsed to chance.

A collapsed epoch is a training failure, not a data point. It has a distinctive
signature -- ``sAUROC`` at chance together with ``RES`` (resolution) at zero --
and it corrupts results in two different directions depending on where it lands:

* inside a **floor pool** it inflates the run-to-run floor, which manufactures
  false *nulls* (every arm suddenly sits "within noise");
* inside an **arm** it manufactures a false *catastrophe*.

Both have already happened in this campaign. ``clf_low_total`` at epoch 10 has
``sAUROC=0.5002``/``RES=0.00001`` and was written up as a +896x total-entropy
failure before the screen existed; and four collapsed floor-seed epochs sit
exactly one epoch outside their level's shared range, so the floors were clean
only by luck.

Run this before every scoring pass, and again whenever a floor seed advances --
the margin that kept those four out of the pools is a single epoch.

Usage:
    python scripts/screen_collapsed_epochs.py --metrics <dir-or-csv> [...]
    python scripts/screen_collapsed_epochs.py --metrics m.csv --fail-on-hit

Exit status is 1 with --fail-on-hit if anything is flagged, so this can gate a
scoring pipeline.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# A collapsed model is at chance AND has no resolution. Requiring both avoids
# flagging a saturated-but-working epoch, where sAUROC can look odd while RES
# stays healthy.
DEFAULT_AUROC_MAX = 0.6
DEFAULT_RES_MAX = 0.01


def _resolve(path: Path) -> Path:
    """Accept either the CSV itself or the directory stoch_prob_metrics writes.

    ``stoch_prob_metrics.py --out`` takes a DIRECTORY and writes metrics.csv
    inside it, which is easy to get wrong at the call site.
    """
    return path / "metrics.csv" if path.is_dir() else path


def screen(rows, auroc_max=DEFAULT_AUROC_MAX, res_max=DEFAULT_RES_MAX):
    """Return (hits, totals_by_predictor, collapsed_by_predictor)."""
    hits, totals, collapsed = [], {}, {}
    for r in rows:
        try:
            auroc = float(r["sAUROC"])
            res = float(r["RES"])
            epoch = int(r["epoch"])
        except (KeyError, ValueError, TypeError):
            continue
        pred = r.get("predictor", "?")
        totals[pred] = totals.get(pred, 0) + 1
        if auroc <= auroc_max and res <= res_max:
            collapsed[pred] = collapsed.get(pred, 0) + 1
            hits.append({
                "predictor": pred, "level": r.get("level", "?"),
                "arm": r.get("arm", "?"), "epoch": epoch,
                "sAUROC": auroc, "RES": res,
                "is_floor_seed": r.get("arm", "").startswith("dir00"),
            })
    hits.sort(key=lambda h: (h["predictor"], h["level"], h["arm"], h["epoch"]))
    return hits, totals, collapsed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metrics", nargs="+", required=True,
                    help="metrics.csv files, or directories containing one")
    ap.add_argument("--auroc-max", type=float, default=DEFAULT_AUROC_MAX)
    ap.add_argument("--res-max", type=float, default=DEFAULT_RES_MAX)
    ap.add_argument("--fail-on-hit", action="store_true",
                    help="exit 1 if anything is flagged (for gating a pipeline)")
    a = ap.parse_args()

    all_hits, all_totals, all_collapsed = [], {}, {}
    for raw in a.metrics:
        path = _resolve(Path(raw))
        if not path.exists():
            print(f"  {raw}: NOT FOUND", file=sys.stderr)
            continue
        rows = list(csv.DictReader(open(path)))
        hits, totals, collapsed = screen(rows, a.auroc_max, a.res_max)
        print(f"  {path}: {len(rows)} rows, {len(hits)} collapsed")
        for h in hits:
            role = "FLOOR SEED" if h["is_floor_seed"] else "arm"
            print(f"      {h['predictor']} {h['level']} {h['arm']} ep{h['epoch']}  "
                  f"sAUROC={h['sAUROC']:.4f} RES={h['RES']:.5f}  [{role}]")
        all_hits += hits
        for k, v in totals.items():
            all_totals[k] = all_totals.get(k, 0) + v
        for k, v in collapsed.items():
            all_collapsed[k] = all_collapsed.get(k, 0) + v

    if all_totals:
        print(f"\n  {'predictor':<10}{'scored':>9}{'collapsed':>11}{'rate':>9}")
        for p in sorted(all_totals):
            n, c = all_totals[p], all_collapsed.get(p, 0)
            print(f"  {p:<10}{n:>9}{c:>11}{c / n * 100:>8.2f}%")

    seeds = [h for h in all_hits if h["is_floor_seed"]]
    if seeds:
        print(f"\n  {len(seeds)} collapsed FLOOR-SEED epoch(s). Check whether each falls inside "
              f"its level's\n  3-seed shared range -- if so that floor is inflated and its "
              f"nulls are not real.")
    return 1 if (a.fail_on_hit and all_hits) else 0


if __name__ == "__main__":
    sys.exit(main())
