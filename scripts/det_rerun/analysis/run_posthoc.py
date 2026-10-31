#!/usr/bin/env python
"""KL variants and the margin decision rule for every method, val-fit and test-scored.

At the latest epoch every learned method of the setting has complete (per --mode).
Writes a JSON of all rows.  Usage: run_posthoc.py [--mode ranked] [--out PATH]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import calib  # noqa: E402
import margin_rule  # noqa: E402
from cells import LEARNED, SETTINGS, SHORT, as_run, load_cell, matched_epoch, run_dirs, score  # noqa: E402

COV = (0.5, 0.7, 0.8, 0.9, 1.0)


def risk_coverage(ps, pf, y):
    pred, conf = ps > pf, np.abs(ps - pf)
    o = np.argsort(-conf, kind="stable")
    err = (pred[o] != (y[o] == 1)).astype(float)
    cum = np.cumsum(err) / np.arange(1, len(err) + 1)
    return [100 * cum[max(0, int(c * len(err)) - 1)] for c in COV], 100 * float(cum.mean())


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
            continue
        for arm in LEARNED:
            if (s, arm, a.mode) not in dirs:
                continue
            c = load_cell(dirs[(s, arm, a.mode)], E)
            ps, pf, pi, y = c["test"]
            vs, vf, vi, vy = c["val"]
            k = c["k_test"]
            kv = 100 if c["counts"] else None   # val is stored at the eval K
            kl = {"raw": calib.nll(calib.raw(ps, k), y)}
            stored = c["eval_metrics"]["threshold_free"]["log_score"]
            assert abs(kl["raw"] - stored) < 1e-6, f"KL does not reproduce for {s} {arm}: {kl['raw']} vs {stored}"
            p_rn, v_rn = calib.renorm(ps, pf, k), calib.renorm(vs, vf, kv)
            kl["renorm"] = calib.nll(p_rn, y)
            for base, pt, pv in (("raw", calib.raw(ps, k), calib.raw(vs, kv)), ("renorm", p_rn, v_rn)):
                t = calib.fit_temp(pv, vy)
                kl[f"{base}+temp"] = calib.nll(calib.apply_temp(pt, t), y)
                ab = calib.fit_platt(pv, vy)
                kl[f"{base}+platt"] = calib.nll(calib.apply_platt(pt, ab), y)
                kl[f"{base}+iso"] = calib.nll(calib.fit_iso(pv, vy).predict(pt), y)
            w = float(c["threshold"].get("w", 0.9))
            row = {"setting": s, "arm": arm, "mode": a.mode, "epoch": E, "kl": kl,
                   "as_run": score(as_run(c), y)}
            rc, aurc = risk_coverage(ps, pf, y)
            row["risk_cov"], row["aurc"] = rc, aurc
            for name, fs in (("margin_thrfit", c["thrfit"]), ("margin_val", c["val"])):
                f = margin_rule.fit(fs[0], fs[1], fs[3], w)
                row[name] = {**f, **score(margin_rule.predict(ps, pf, f["t_s"], f["t_f"]), y)}
            rows.append(row)
            _print(row)
    if a.out:
        Path(a.out).write_text(json.dumps(rows, indent=1, default=float))


def _fmt(m):
    return (f"F1c {m['F1c']:.3f} F1all {m['F1all']:.3f} unc {m['unc']:5.1f} inv {m['inv']:5.1f} "
            f"ounc {m['ounc']:5.1f} FPR {m['FPR']:.4f} FNR {m['FNR']:.4f}")


def _print(r):
    k = r["kl"]
    print(f"\n== {r['setting']} {SHORT[r['arm']]} ({r['mode']}, epoch {r['epoch']})  AURC {r['aurc']:.2f}")
    print("  KL " + " ".join(f"{n} {v:.4f}" for n, v in k.items()))
    print(f"  as-run                    {_fmt(r['as_run'])}")
    for n in ("margin_thrfit", "margin_val"):
        m = r[n]
        print(f"  {n:13s} ts {m['t_s']:.2f} tf {m['t_f']:.2f}  {_fmt(m)}")


if __name__ == "__main__":
    main()
