#!/usr/bin/env python
"""Check det-rerun run dirs against their own artifacts (tracker C08).

Usage: check_prediction_store.py RUN_DIR [RUN_DIR ...]
Exits 1 if any run has a problem; a cell counts as done only when this passes.
"""
import sys

from adaptive_roa.adaptive_v2.eval.store_check import check_run


def main(argv):
    if not argv:
        print(__doc__)
        return 2
    bad = 0
    for run in argv:
        problems = check_run(run)
        print(f"{'OK  ' if not problems else 'FAIL'} {run}")
        for p in problems:
            print(f"     {p}")
        bad += bool(problems)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
