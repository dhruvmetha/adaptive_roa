#!/usr/bin/env python
"""Area under the learning curve, one number per arm per level.

The per-epoch figures are read at their right-hand end because arms converge at
different rates, and a vertical slice mid-run flatters whichever arm happened to
converge first. The area under metric-vs-budget rewards getting there sooner,
which is the actual active-learning claim, and gives a single number per arm
that the 3-seed control turns into a floor in the usual way.

The x axis is training trajectories, not epoch index, for the reason given in
plot_stoch_all_levels.xs: an arm that under-spends its allowance stops where its
data stops. The area is normalised by the budget span so it reads in the
metric's own units -- an ALC of 0.05 on KL means "KL averaged 0.05 over the
run". Epoch 0 is included: it is the shared pre-acquisition start and drops out
of any arm-vs-arm difference.

Usage:
    python scripts/stoch_alc.py                # every campaign in plot_stoch_all_levels
    python scripts/stoch_alc.py pendulum cartpole
"""
from __future__ import annotations
import argparse
import csv
import math
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_stoch_all_levels import CAMPAIGNS, DOCS, SEEDS, within_budget  # noqa: E402

ALC_METRICS = ("KL", "brier_debiased", "sAUROC",
               "auc_bal_acc", "auc_f05", "auc_tpr", "auc_tnr", "worst_overclaim")
_trapz = getattr(np, "trapezoid", None) or np.trapz


def alc(x, y) -> float:
    """Trapezoid area of y over x divided by the span of x. NaN below two points."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2:
        return float("nan")
    if np.any(np.diff(x) <= 0):
        raise ValueError("budget must be strictly increasing along the curve")
    return float(_trapz(y, x) / (x[-1] - x[0]))


def alc_table(rows: list[dict], level: str, metrics=ALC_METRICS) -> list[dict]:
    """One row per arm at `level`, plus `dir00_mean` and `dir00_2sd` when the
    three uniform seeds are present. A metric an arm lacks (blank column, or a
    NaN epoch) gives NaN for that cell rather than killing the table."""
    by_arm: dict[str, dict[int, dict]] = defaultdict(dict)
    pred: dict[str, str] = {}
    for r in rows:
        if r["level"] != level:
            continue
        by_arm[r["arm"]][int(r["epoch"])] = r
        pred[r["arm"]] = r["predictor"]
    out = []
    for arm in sorted(by_arm):
        eps = sorted(by_arm[arm])
        rs = [by_arm[arm][e] for e in eps]
        try:
            x = [int(r["train_trajectories"]) for r in rs]
        except ValueError as exc:
            raise ValueError(f"{level}/{arm}: train_trajectories missing on a row") from exc
        row = {"level": level, "predictor": pred[arm], "arm": arm, "n_epochs": len(eps),
               "budget_lo": x[0], "budget_hi": x[-1]}
        for m in metrics:
            try:
                y = [float(r[m]) for r in rs]
            except (KeyError, ValueError):
                row[f"alc_{m}"] = float("nan")
                continue
            row[f"alc_{m}"] = float("nan") if any(math.isnan(v) for v in y) else alc(x, y)
        out.append(row)

    seeds = [r for r in out if r["arm"] in SEEDS]
    if len(seeds) == 3:
        base = {"level": level, "predictor": seeds[0]["predictor"],
                "n_epochs": min(s["n_epochs"] for s in seeds),
                "budget_lo": seeds[0]["budget_lo"], "budget_hi": seeds[0]["budget_hi"]}
        mean_row = dict(base, arm="dir00_mean")
        sd_row = dict(base, arm="dir00_2sd")
        for m in metrics:
            v = [s[f"alc_{m}"] for s in seeds]
            if any(math.isnan(t) for t in v):
                mean_row[f"alc_{m}"] = sd_row[f"alc_{m}"] = float("nan")
            else:
                mean_row[f"alc_{m}"] = st.mean(v)
                sd_row[f"alc_{m}"] = 2 * st.stdev(v)
        out += [mean_row, sd_row]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("campaigns", nargs="*", default=list(CAMPAIGNS))
    a = ap.parse_args()
    for name in (a.campaigns or list(CAMPAIGNS)):
        c = CAMPAIGNS[name]
        src = DOCS / c["csv"]
        if not src.exists():
            print(f"  SKIP {name}: {c['csv']} absent")
            continue
        # The campaign's reporting budget bounds the curve: an arm that ran past
        # it contributes area only up to the cap, so ALC stays budget-matched.
        rows = within_budget(list(csv.DictReader(src.open())), c)
        table = []
        for lv, _lab in c["panels"]:
            table += alc_table(rows, lv)
        if not table:
            print(f"  SKIP {name}: no rows")
            continue
        out = src.with_name(src.stem + "_alc" + src.suffix)
        fields = ["level", "predictor", "arm", "n_epochs", "budget_lo", "budget_hi"] + \
                 [f"alc_{m}" for m in ALC_METRICS]
        with out.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            for r in table:
                w.writerow(r)
        print(f"  wrote {out.relative_to(DOCS)}  ({len(table)} rows)")
        # Console standings on the two headline columns, floor beside the mean.
        for lv, _lab in c["panels"]:
            sub = [r for r in table if r["level"] == lv]
            floor = next((r for r in sub if r["arm"] == "dir00_2sd"), None)
            print(f"    {lv}:  arm{'':20s} alc_KL   alc_bal_acc   (n)")
            for r in sorted(sub, key=lambda r: (r["arm"] in SEEDS, r["arm"])):
                if r["arm"] in SEEDS:
                    continue
                print(f"      {r['arm']:22s} {r['alc_KL']:.4f}   {r['alc_auc_bal_acc']:.4f}"
                      f"      {r['n_epochs']}")
            if floor:
                print(f"      {'FM 2·SD floor':22s} {floor['alc_KL']:.4f}   "
                      f"{floor['alc_auc_bal_acc']:.4f}")


if __name__ == "__main__":
    main()
