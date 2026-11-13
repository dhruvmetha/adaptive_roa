"""Render every ground-truth slice and a GT-only contact sheet per setting.

    python slices/lib/render_gt_all.py [--setting S]

reads   slices/data/<setting>/*.npz
writes  slices/figs_gt/<setting>/<plane>.{png,pdf}
        slices/sheets/<setting>_gt.png      (sorted by in-box gt_clarity, best first)
and prints one line per slice. No GPU, no model: ground truth only.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SLICES = Path(__file__).resolve().parents[1]
if str(SLICES) not in sys.path:
    sys.path.insert(0, str(SLICES))

from lib import metrics, sheet, style  # noqa: E402


def render_setting(setting: str, data_root: Path, figs_root: Path, sheets_root: Path) -> list[dict]:
    files = sorted((data_root / setting).glob("*.npz"))
    if not files:
        print(f"{setting}: no slices under {data_root / setting}")
        return []
    rows, pngs = [], []
    for f in files:
        sl = style.load_slice(f)
        if sl["setting"] != setting:
            raise ValueError(f"{f}: setting {sl['setting']!r} != directory {setting!r}")
        if f.stem != sl["plane"]:
            print(f"  note: {f.name} holds plane {sl['plane']!r}; figure named after the plane")
        png, pdf = style.render_gt(sl, figs_root / setting / sl["plane"])
        gt_m, _ = metrics.slice_metrics(sl)
        row = sheet.index_row(sl, gt_m, extra={"png": png, "pdf": pdf})
        rows.append(row)
        pngs.append(png)
        print(f"  {sheet.gt_caption(row)}")
    order = sorted(range(len(rows)), key=lambda k: -(rows[k]["gt_clarity"] if rows[k]["gt_clarity"] == rows[k]["gt_clarity"] else -1))
    out = sheet.contact_sheet([pngs[k] for k in order], [sheet.gt_caption(rows[k]) for k in order],
                              sheets_root / f"{setting}_gt.png", ncols=3, tile_width=900)
    print(f"{setting}: {len(rows)} slices -> {figs_root / setting}, sheet {out}")
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--setting", help="one setting (default: every directory under data/)")
    p.add_argument("--data-root", type=Path, default=SLICES / "data")
    p.add_argument("--figs-root", type=Path, default=SLICES / "figs_gt")
    p.add_argument("--sheets-root", type=Path, default=SLICES / "sheets")
    a = p.parse_args(argv)
    settings = [a.setting] if a.setting else sorted(d.name for d in a.data_root.iterdir() if d.is_dir())
    for s in settings:
        render_setting(s, a.data_root, a.figs_root, a.sheets_root)


if __name__ == "__main__":
    main()
