"""Re-render every full-resolution pair figure from its pred npz, in the current style.

    $PY slices/lib/rerender_pairs.py [--setting S]

For each pred/<setting>/<plane>__r{1,2,3}.npz (symlinks followed; `_s2` and `.tmp` files skipped),
re-reads data/<setting>/<plane>.npz (so a rewritten box or label set is picked up), computes
metrics.slice_metrics on the stored decision, and renders figs/<setting>/<plane>__rK.{png,pdf}. Each
figure is written under a temporary name in the same directory and moved into place with os.replace,
so a reader never sees a half-written file. Pred files are only read, never written, so this cannot
collide with workers that write their figure first and their npz last. No GPU, no detv3 import.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

SLICES = Path(__file__).resolve().parents[1]
if str(SLICES) not in sys.path:
    sys.path.insert(0, str(SLICES))
from lib import metrics, style  # noqa: E402

PRED_RE = re.compile(r"^(?P<plane>.+)__(?P<rk>r[123])\.npz$")


def rerender_one(pred_path: Path) -> tuple[str, str]:
    m = PRED_RE.match(pred_path.name)
    setting, plane, rk = pred_path.parent.name, m["plane"], m["rk"]
    sl = style.load_slice(SLICES / "data" / setting / f"{plane}.npz")
    with np.load(pred_path, allow_pickle=False) as z:
        dec = np.array(z["decision"]).astype(np.int8)
        stride = int(np.asarray(z["stride"]).item()) if "stride" in z.files else 1
    if stride != 1 or dec.shape != sl["labels"].shape:
        raise ValueError(f"stride {stride}, decision {dec.shape} vs labels {sl['labels'].shape}")
    _, tera_m = metrics.slice_metrics(sl, dec)
    final = SLICES / "figs" / setting / f"{plane}__{rk}"
    tmp = final.parent / f".{final.name}.tmp{os.getpid()}"
    png, pdf = style.render_pair(sl, dec, tmp, metrics=tera_m)
    os.replace(png, f"{final}.png")
    os.replace(pdf, f"{final}.pdf")
    return f"{final}.png", f"{final}.pdf"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--setting", help="one setting (default: all under pred/)")
    a = p.parse_args(argv)
    dirs = [SLICES / "pred" / a.setting] if a.setting else sorted(d for d in (SLICES / "pred").iterdir() if d.is_dir())
    t0, n_ok, failed = time.time(), 0, []
    for d in dirs:
        for pred_path in sorted(d.glob("*__r*.npz")):
            m = PRED_RE.match(pred_path.name)
            if not m or m["plane"].endswith("_s2") or ".tmp" in pred_path.name:
                continue
            if not (SLICES / "data" / d.name / f"{m['plane']}.npz").exists():
                failed.append(f"{pred_path}: no data npz")
                continue
            try:
                rerender_one(pred_path)
                n_ok += 1
            except Exception as e:  # noqa: BLE001  report and carry on
                failed.append(f"{pred_path}: {e}")
    print(f"re-rendered {n_ok} pair figures in {time.time() - t0:.0f} s; {len(failed)} skipped")
    for f in failed:
        print("  skipped", f, file=sys.stderr)


if __name__ == "__main__":
    main()
