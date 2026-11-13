"""Candidate Figure 1 triplets (coordinator). Usage: python composites/make_triplets.py <name> <c_slice> <c_pred> <p_slice> <p_pred> <s_slice> <s_pred>.

Paths are relative to slices/. Writes composites/<name>_2x3.* and composites/<name>_1x3.*.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib import composite  # noqa: E402

name, *paths = sys.argv[1:]
entries = [{"slice": str(ROOT / paths[i]), "pred": str(ROOT / paths[i + 1])} for i in (0, 2, 4)]
for layout in ("2x3", "1x3"):
    print(composite.render_triplet(entries, str(ROOT / "composites" / f"{name}_{layout}"), layout=layout))
