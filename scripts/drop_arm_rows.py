#!/usr/bin/env python
"""Delete one arm's rows from the all-levels CSVs so the scorer re-projects them.

score_stoch_incremental.py is incremental: `scored_epochs` treats an (arm, epoch)
pair already present with a non-blank auc_bal_acc as done, so an arm whose epoch
artifacts are REWRITTEN IN PLACE is skipped forever and the CSV keeps the old
numbers with no error anywhere. Deleting the arm's rows is what makes those
epochs look unscored again; the next ordinary scoring pass rebuilds them, and
merge_rows is key-based with NEW winning, so nothing is duplicated.

`--rescore` on the scorer does the same job without CSV surgery, but it
re-projects EVERY arm in the campaign -- hundreds of epochs at 990k eval points
for quad3D. This is the surgical version.

    ./env/bin/python scripts/drop_arm_rows.py partx_faithful --dry-run
    ./env/bin/python scripts/drop_arm_rows.py partx_faithful

Touches both the wide CSV and its _levelsets twin, since the level-set figures
read the twin and a half-dropped pair would draw one readout against the other.
Writes atomically (tmp + replace) so an interrupted run cannot leave a truncated
CSV behind a live figure job.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

DOCS = Path("/common/users/shared/pracsys/genMoPlan/docs/stochastic")
DEFAULT_ROOTS = [DOCS / "timeout_fix", DOCS / "timeout_fix_40k"]


def drop(path: Path, arm: str, dry: bool) -> str:
    with path.open() as f:
        rows = list(csv.DictReader(f))
        fields = csv.DictReader(path.open()).fieldnames
    if not rows:
        return f"[skip] {path.name}: empty"
    if "arm" not in (fields or []):
        return f"[skip] {path.name}: no arm column"
    keep = [r for r in rows if r.get("arm") != arm]
    n = len(rows) - len(keep)
    if n == 0:
        return f"[none] {path.name}: no {arm} rows"
    if dry:
        return f"[dry]  {path.name}: would drop {n} of {len(rows)}"
    tmp = path.with_suffix(".csv.tmp")
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(keep)
    tmp.replace(path)
    return f"[ok]   {path.name}: dropped {n} of {len(rows)}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("arm")
    ap.add_argument("roots", nargs="*", type=Path, default=DEFAULT_ROOTS)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    for root in a.roots:
        for wide in sorted(root.glob("*_all_levels.csv")):
            print(drop(wide, a.arm, a.dry_run))
            lv = wide.with_name(wide.stem + "_levelsets" + wide.suffix)
            if lv.exists():
                print(drop(lv, a.arm, a.dry_run))


if __name__ == "__main__":
    main()
