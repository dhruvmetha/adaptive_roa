#!/usr/bin/env python
"""Print a det_v2 run's output_dir, composed from its own config.

    run_dir.py <name> [--code <snapshot or checkout>]

launch.sh calls this on the commit's snapshot, so the run dir is the one that commit's
config sets (${exp_dir}/det_v2/<setting>/<name>_s<seed>, with exp_dir from user.env).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import det_v2_lib as lib  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--code", default=None, help="code tree whose configs to compose (default: this repo)")
    a = ap.parse_args()
    env = lib.load_user_env()
    row = lib.manifest_row(env, a.name)
    code = a.code or env["DET_V2_REPO"]
    cfg = lib.compose_row(env, row, code)
    out = cfg.get("output_dir")
    if not out:
        raise SystemExit(f"{a.name}: the composed config sets no output_dir")
    print(out)


if __name__ == "__main__":
    main()
