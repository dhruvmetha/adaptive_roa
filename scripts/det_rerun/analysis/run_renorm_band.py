#!/usr/bin/env python
"""Band rule on the renormalised p' = (n_s + 0.5) / (n_s + n_f + 1), no separate invalid veto.

Success if p' >= hi, failure if p' <= lo, else outcome-uncertain; (hi, lo) fit on the
run's fit set (predictions_thrfit, K = 10) with the optimizer's loss and w, via the margin
rule on (p', 1 - p'). Count arms only (FM, regressors); test-scored.
Usage: run_renorm_band.py [--mode ranked] [--out PATH]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import calib  # noqa: E402
import margin_rule  # noqa: E402
from cells import SETTINGS, SHORT, as_run, load_cell, matched_epoch, run_dirs, score  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="ranked")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    dirs, rows = run_dirs(), []
    for s in SETTINGS:
        E = matched_epoch(dirs, s, a.mode)
        for arm in ("fm_det", "det_mlp_reg", "det_bnn_reg"):
            if E is None or (s, arm, a.mode) not in dirs:
                continue
            c = load_cell(dirs[(s, arm, a.mode)], E)
            ps, pf, _, y = c["test"]
            ts_, tf_, _, ty = c["thrfit"]
            pt, pfit = calib.renorm(ps, pf, c["k_test"]), calib.renorm(ts_, tf_, 10)
            f = margin_rule.fit(pfit, 1 - pfit, ty, float(c["threshold"].get("w", 0.9)))
            after = score(margin_rule.predict(pt, 1 - pt, f["t_s"], f["t_f"]), y)
            r = {"setting": s, "arm": arm, "epoch": E, "hi": (1 + f["t_s"]) / 2, "lo": (1 - f["t_f"]) / 2,
                 "as_run": score(as_run(c), y), "band": after}
            rows.append(r)
            b, o = r["band"], r["as_run"]
            print(f"{s:7s} {SHORT[arm]:8s} ep{E:<2d} band [{r['lo']:.3f}, {r['hi']:.3f}]  "
                  f"F1c {o['F1c']:.3f}->{b['F1c']:.3f} F1all {o['F1all']:.3f}->{b['F1all']:.3f} "
                  f"unc {o['unc']:.1f}->{b['unc']:.1f} FPR {o['FPR']:.4f}->{b['FPR']:.4f} FNR {o['FNR']:.4f}->{b['FNR']:.4f}")
    if a.out:
        Path(a.out).write_text(json.dumps(rows, indent=1, default=float))


if __name__ == "__main__":
    main()
