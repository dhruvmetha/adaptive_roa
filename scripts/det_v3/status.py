#!/usr/bin/env python
"""det_v3 entry point: the det_v2 tool of the same name with DET_CAMPAIGN=det_v3 (root, ledger, manifest)."""
import os
import runpy
import sys
from pathlib import Path

os.environ["DET_CAMPAIGN"] = "det_v3"
target = Path(__file__).resolve().parents[1] / "det_v2" / Path(__file__).name
sys.argv[0] = str(target)
sys.path.insert(0, str(target.parent))
runpy.run_path(str(target), run_name="__main__")
