"""Contact sheets and index rows for the slice figures."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import metrics as M

_FONT = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"

# Primary (in-box) metrics, then their full-slice copies.
_GT = M.GT_KEYS + ["in_box_frac"]
_TERA = M.TERA_KEYS
INDEX_COLUMNS = (
    ["setting", "system", "controller", "plane", "dim1", "dim2", "x_range", "y_range", "box_x", "box_y",
     "fixed_values", "resolution", "succ_frac", "source", "source_note", "gt_check"]
    + [k for k in _GT if k != "succ_frac"]
    + [f"{k}_full" for k in M.GT_KEYS]
    + ["run", "seed", "K", "t_s", "t_f", "u_s", "u_f"]
    + _TERA + [f"{k}_full" for k in _TERA]
    + ["rank_score", "rank_score_full", "png", "pdf"]
)


def _rng(a) -> str:
    return "" if a is None else json.dumps([None if math.isnan(float(v)) else round(float(v), 6) for v in a])


def index_row(sl: dict, gt_m: dict, tera_m: dict | None = None, run=None, extra: dict | None = None) -> dict:
    """One index row (keys = INDEX_COLUMNS; missing values are "").

    ``gt_m`` / ``tera_m``: ``metrics.slice_metrics`` output (in-box primary plus ``_full``).
    ``run``: a run name (seed, K = 100 and c6c thresholds are then looked up) or a dict with any of
    run, seed, K, t_s, t_f, u_s, u_f. ``extra``: more columns, e.g. png and pdf; unknown keys raise.
    """
    row = {k: "" for k in INDEX_COLUMNS}
    row.update(setting=sl["setting"], system=sl["system_label"], controller=sl["controller"], plane=sl["plane"],
               dim1=sl["dim_names"][0], dim2=sl["dim_names"][1],
               x_range=_rng([sl["xs"][0], sl["xs"][-1]]), y_range=_rng([sl["ys"][0], sl["ys"][-1]]),
               box_x=_rng(sl.get("box_x")), box_y=_rng(sl.get("box_y")),
               fixed_values=sl["fixed_json"], resolution=f"{sl['nx']}x{sl['ny']}",
               source=sl["source"], source_note=sl["source_note"], gt_check=sl["gt_check_json"])
    for k, v in gt_m.items():
        if k in row:
            row[k] = v
    if tera_m is not None:
        for k, v in tera_m.items():
            if k in row:
                row[k] = v
        if "rank_score" not in tera_m:
            row["rank_score"] = M.rank_score(gt_m, tera_m)
    if isinstance(run, str):
        from . import infer
        thr = infer.thresholds(run)
        seed = [json.loads(l).get("seed") for l in infer.MATRIX_ROWS.read_text().splitlines()
                if l.strip() and json.loads(l).get("name") == run]
        row.update(run=run, seed=seed[0] if seed else "", K=100, **thr)
    elif isinstance(run, dict):
        for k, v in run.items():
            if k not in ("run", "seed", "K", "t_s", "t_f", "u_s", "u_f"):
                raise KeyError(f"unknown run key {k}")
            row[k] = v
    for k, v in (extra or {}).items():
        if k not in row:
            raise KeyError(f"{k} is not an index column")
        row[k] = v
    return row


def write_index(rows, path):
    """Write rows (index_row output) to a CSV with INDEX_COLUMNS, NaN as empty."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=INDEX_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if isinstance(v, float) and math.isnan(v) else v) for k, v in r.items()})
    return str(path)


def _f(v, fmt):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if math.isnan(v) else format(v, fmt)


def pair_caption(row: dict) -> str:
    """plane, rank_score, gt_clarity, R_s, R_f, abstain, FP, FN (in-box numbers)."""
    return (f"{row['plane']}   rank {_f(row.get('rank_score'), '.3f')}   clarity {_f(row.get('gt_clarity'), '.3f')}   "
            f"R_s {_f(row.get('R_s'), '.3f')}   R_f {_f(row.get('R_f'), '.3f')}   "
            f"abstain {_f(row.get('abstain_frac'), '.1%')}   FP {int(row.get('n_wrong_s') or 0)}   "
            f"FN {int(row.get('n_wrong_f') or 0)}")


def gt_caption(row: dict) -> str:
    """plane, gt_clarity, succ_frac, components, speckle (in-box numbers)."""
    return (f"{row['plane']}   clarity {_f(row.get('gt_clarity'), '.3f')}   succ {_f(row.get('succ_frac'), '.1%')}   "
            f"cc {row.get('n_cc_succ')}+{row.get('n_cc_fail')}   speckle {_f(row.get('speckle_frac'), '.2%')}   "
            f"in-box {_f(row.get('in_box_frac'), '.0%')}")


def contact_sheet(png_paths, captions, out_path, ncols=2, tile_width=1300):
    """Tile PNGs (scaled to ``tile_width`` px) in ``ncols`` columns, a one-line caption over each."""
    if len(png_paths) != len(captions):
        raise ValueError("png_paths and captions differ in length")
    if not png_paths:
        raise ValueError("no images")
    font = ImageFont.truetype(str(_FONT), size=max(14, tile_width // 52))
    cap_h = int(font.size * 1.8)
    tiles = []
    for p in png_paths:
        im = Image.open(p).convert("RGB")
        h = round(im.height * tile_width / im.width)
        tiles.append(im.resize((tile_width, h), Image.LANCZOS))
    th = max(t.height for t in tiles) + cap_h
    pad = 12
    nrows = math.ceil(len(tiles) / ncols)
    sheet = Image.new("RGB", (ncols * (tile_width + pad) + pad, nrows * (th + pad) + pad), "white")
    draw = ImageDraw.Draw(sheet)
    for k, (t, cap) in enumerate(zip(tiles, captions)):
        r, c = divmod(k, ncols)
        x0, y0 = pad + c * (tile_width + pad), pad + r * (th + pad)
        text = cap
        while draw.textlength(text, font=font) > tile_width - 8 and len(text) > 4:
            text = text[:-2]
        if text != cap:
            text = text[:-1] + "…"
        draw.text((x0 + 4, y0 + (cap_h - font.size) // 2), text, fill="black", font=font)
        sheet.paste(t, (x0, y0 + cap_h))
        draw.rectangle([x0 - 1, y0 - 1, x0 + tile_width, y0 + th], outline="#BBBBBB")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    return str(out_path)
