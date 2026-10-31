#!/usr/bin/env python
"""Compare the run's (lambda, delta) rule with the (tau_s, tau_f) rule on every setting.

For each setting, at the latest epoch every ranked learned method has complete:
  as-run      the run's lambda*/delta* (fit on predictions_thrfit, K = 10), test-scored
  ld@val      the same (lambda, delta) optimizer refit on val (eval K), test-scored
  tau@thrfit  (tau_s, tau_f) fit on the run's fit set, test-scored
  tau@val     (tau_s, tau_f) fit on val, test-scored
Writes a JSON of all rows and prints compact tables.  Usage: run_tau_rule.py [--mode ranked] [--out PATH]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cells import LEARNED, SETTINGS, SHORT, as_run, load_cell, matched_epoch, run_dirs, score  # noqa: E402
import tau_rule  # noqa: E402

from adaptive_roa.adaptive_v2.eval.full_roa import _predict_lambda_delta  # noqa: E402
from adaptive_roa.adaptive_v2.eval.store_threshold import refit  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="ranked")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    dirs = run_dirs()
    rows = []
    for s in SETTINGS:
        E = matched_epoch(dirs, s, a.mode)
        if E is None:
            print(f"{s}: no common epoch"); continue
        for arm in LEARNED:
            if (s, arm, a.mode) not in dirs:
                continue
            c = load_cell(dirs[(s, arm, a.mode)], E)
            w = float(c["threshold"].get("w", 0.9))
            y = c["test"][3]
            before = score(as_run(c), y)
            ld = c["eval_metrics"]["lambda_delta"]
            assert abs(before["F1c"] - ld["f1"]) < 1e-9 and abs(before["inv"] / 100 - ld["invalid_pct"]) < 1e-9, \
                f"as-run scoring does not reproduce the stored eval for {s} {arm}"
            vs, vf, vi, vy = c["val"]
            lam_v, d_v = refit(c["threshold"], w, vs, vf, vi, vy)
            pred_ldv, _ = _predict_lambda_delta(*c["test"][:3], lam_v, d_v, c["rule"], None)
            row = {"setting": s, "arm": arm, "mode": a.mode, "epoch": E, "w": w,
                   "lam": c["lam"], "delta": c["delta"], "as_run": before,
                   "ld_val": {"lam": lam_v, "delta": d_v, **score(pred_ldv, y)}}
            for name, fs in (("tau_thrfit", c["thrfit"]), ("tau_val", c["val"])):
                f = tau_rule.fit(*fs, w)
                pred, n_both = tau_rule.predict(*c["test"][:3], f["tau_s"], f["tau_f"])
                row[name] = {**f, **score(pred, y), "n_both": n_both, "pct_both": 100 * n_both / len(y)}
            rows.append(row)
            _print(row)
    if a.out:
        Path(a.out).write_text(json.dumps(rows, indent=1, default=float))


def _fmt(m):
    return (f"F1c {m['F1c']:.3f} F1all {m['F1all']:.3f} unc {m['unc']:5.1f} inv {m['inv']:5.1f} "
            f"ounc {m['ounc']:5.1f} FPR {m['FPR']:.4f} FNR {m['FNR']:.4f}")


def _print(r):
    print(f"\n== {r['setting']} {SHORT[r['arm']]} ({r['mode']}, epoch {r['epoch']})")
    print(f"  as-run     lam {r['lam']:.3f} d {r['delta']:.3f}            {_fmt(r['as_run'])}")
    v = r["ld_val"]
    print(f"  ld@val     lam {v['lam']:.3f} d {v['delta']:.3f}            {_fmt(v)}")
    for k in ("tau_thrfit", "tau_val"):
        t = r[k]
        print(f"  {k:10s} ts {t['tau_s']:.2f} tf {t['tau_f']:.2f} sum<=1 {str(t['sum_le_1'])[0]} both {t['pct_both']:4.1f}% "
              f"{_fmt(t)}")


if __name__ == "__main__":
    main()
