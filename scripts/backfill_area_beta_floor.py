#!/usr/bin/env python
"""Recompute every auc_* column over beta >= AREA_BETA_MIN, in place.

The area scalars used to average a rate over all ten beta levels; they now
average over beta >= stoch_prob_metrics.AREA_BETA_MIN (see the comment there for
why). Already-scored epochs would otherwise keep the old definition forever,
because score_stoch_incremental.py is incremental and never revisits a row it
has.

A rescore from full_roa_per_point.npz is NOT needed: auc_* is a pure function of
the per-level table, and that table is already persisted next to each wide CSV as
<stem>_levelsets.csv, ten rows per (level, arm, epoch). This reads those rows and
rewrites the wide CSV's auc_* columns from them, so the result is identical to
what the new scorer would produce.

    ./env/bin/python scripts/backfill_area_beta_floor.py [--dry-run] [dir ...]

Default roots are the campaign docs dirs. Writes atomically (tmp + replace), so
an interrupted run cannot leave a half-written CSV behind a live figure job.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Plain import, not spec_from_file_location: the module defines a dataclass, and
# a module loaded without a sys.modules entry makes dataclasses' type resolution
# fail with a bare AttributeError on NoneType.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import stoch_prob_metrics as M  # noqa: E402

DOCS = Path("/common/users/shared/pracsys/genMoPlan/docs/stochastic")
DEFAULT_ROOTS = [DOCS / "timeout_fix", DOCS / "timeout_fix_40k"]
KEY = ["level", "arm", "epoch"]


def recompute(wide: Path, dry: bool) -> str:
    lv = wide.with_name(wide.stem + "_levelsets" + wide.suffix)
    if not lv.exists():
        return f"[skip] {wide.name}: no levelsets sibling"
    w, l = pd.read_csv(wide), pd.read_csv(lv)
    missing = set(map(tuple, w[KEY].values)) - set(map(tuple, l[KEY].values))
    if missing:
        # Refuse rather than write NaN over a real number: a row with no per-level
        # source cannot be recomputed, and silently blanking it would delete a
        # scored result that only the npz could restore.
        return f"[REFUSE] {wide.name}: {len(missing)} wide rows have no levelsets rows"

    area = l[l["beta"] >= M.AREA_BETA_MIN]
    stats = [s for s in M.LEVEL_STATS if s in area.columns]
    agg = {}
    for s in stats:
        agg[f"auc_{s}"] = (s, "mean")                    # pandas mean skips NaN
        if f"{s}_oracle" in area.columns:
            agg[f"auc_{s}_oracle"] = (f"{s}_oracle", "mean")
    new = area.groupby(KEY, dropna=False).agg(**agg).reset_index()
    n_def = (area.assign(_d=area["bal_acc"].notna())
                 .groupby(KEY, dropna=False)["_d"].sum().reset_index(name="n_levels_defined"))
    n_tot = area.groupby(KEY, dropna=False).size().reset_index(name="n_levels")
    new = new.merge(n_def, on=KEY).merge(n_tot, on=KEY)

    cols = [c for c in new.columns if c not in KEY and c in w.columns]
    before = w[cols[0]].copy() if cols else None
    out = w.drop(columns=cols).merge(new[KEY + cols], on=KEY, how="left")[w.columns]
    moved = int((~np.isclose(before, out[cols[0]], equal_nan=True)).sum()) if cols else 0
    if dry:
        return (f"[dry] {wide.name}: {len(out)} rows, {len(cols)} cols, "
                f"{moved} changed in {cols[0]}")
    tmp = wide.with_suffix(".csv.tmp")
    out.to_csv(tmp, index=False)
    tmp.replace(wide)
    return f"[ok]  {wide.name}: {len(out)} rows, {len(cols)} cols, {moved} changed in {cols[0]}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("roots", nargs="*", type=Path, default=DEFAULT_ROOTS)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    print(f"AREA_BETA_MIN = {M.AREA_BETA_MIN}")
    for root in a.roots:
        for wide in sorted(root.glob("*_all_levels.csv")):
            print(recompute(wide, a.dry_run))


if __name__ == "__main__":
    main()
