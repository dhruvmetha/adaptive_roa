"""Crop a slice and one of its preds to the D_test start box (cells whose centres lie inside it).

    python composites/crop_to_box.py <setting> <plane> <rK>

Writes composites/_crops/data/<setting>/<plane>.npz and composites/_crops/pred/<setting>/<plane>__<rK>.npz.
Preview helper for the Figure 1 candidates only; the originals are read, never written.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
setting, plane, rk = sys.argv[1:4]
src_d = ROOT / "data" / setting / f"{plane}.npz"
src_p = ROOT / "pred" / setting / f"{plane}__{rk}.npz"
d = dict(np.load(src_d, allow_pickle=False))
p = dict(np.load(src_p, allow_pickle=False))
xs, ys = d["xs"], d["ys"]
nx, ny = len(xs), len(ys)


def keep(v, box):
    lo, hi = box
    return (np.isnan(lo) | (v >= lo - 1e-9)) & (np.isnan(hi) | (v <= hi + 1e-9))


ix, iy = np.where(keep(xs, d["box_x"]))[0], np.where(keep(ys, d["box_y"]))[0]
sx, sy = slice(ix[0], ix[-1] + 1), slice(iy[0], iy[-1] + 1)


def crop(a):
    a = np.asarray(a)
    if a.shape[:2] == (ny, nx):
        return a[sy, sx]
    if a.ndim >= 1 and a.shape[0] == ny * nx:
        return a.reshape((ny, nx) + a.shape[1:])[sy, sx].reshape((-1,) + a.shape[1:])
    return a


d2 = {k: crop(v) for k, v in d.items()}
d2["xs"], d2["ys"] = xs[sx], ys[sy]
d2["source_note"] = np.array(str(d["source_note"]) + " | PREVIEW CROP to the D_test start box")
p2 = {k: crop(v) for k, v in p.items()}
for sub, obj, name in (("data", d2, f"{plane}.npz"), ("pred", p2, f"{plane}__{rk}.npz")):
    out = ROOT / "composites" / "_crops" / sub / setting
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / name, **obj)
print(f"{setting} {plane} {rk}: {nx}x{ny} -> {len(ix)}x{len(iy)}")
