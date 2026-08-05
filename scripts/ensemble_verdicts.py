#!/usr/bin/env python
"""Verdict table for the ensemble-epistemic campaign.

Every analysis in this campaign was ad-hoc until now, which meant the pooled-floor
rule and the two-consecutive-epoch rule lived only in throwaway snippets. Both were
learned the hard way and both change conclusions, so they belong in code:

  * POOLED FLOOR. With three seeds, a single epoch's SD has two degrees of freedom.
    The deterministic floor was measured varying 15.6x across epochs (0.00024 to
    0.00380) -- enough to flip a verdict depending on which slice was used. So the
    floor is 2*sqrt(mean(var_e)) over every epoch the three seeds share, excluding
    epoch 0 (all seeds hold identical data there by construction, so its spread is
    structurally zero and would make any gap look significant).

  * TWO CONSECUTIVE EPOCHS. Single-epoch reads lie. On the only complete curve,
    `aleat` was +10.7x the floor WORSE at epoch 4 and the BEST arm by epoch 18.

  * POST-RECALIBRATION METRIC. Brier = REL - RES + UNC, so perfect recalibration
    drives REL to 0 and leaves UNC - RES. Raw Brier gaps at high noise turned out
    to be almost entirely REL, which recalibration repairs; UNC - RES is the part
    it cannot.

Usage:
    python scripts/ensemble_verdicts.py --metrics <score_dir>/metrics.csv
    python scripts/ensemble_verdicts.py --metrics ... --metric sAUROC

The metrics CSV comes from scripts/stoch_prob_metrics.py. Deterministic runs are not
in it and cannot be: those metrics need rollout ground truth, which a deterministic
system does not have -- score det from artifacts_v2.json label metrics instead.
"""
from __future__ import annotations

import argparse
import collections
import csv
import math
import statistics
from pathlib import Path

ARMS = ["epi_bald", "epi_var", "total", "aleat"]
CONTROL = "dir00"
SEED_SUFFIXES = ["_s43", "_s44"]


def _load(path: Path, metric: str):
    """metric='recal' -> UNC_debiased - RES; otherwise a raw column name."""
    out = collections.defaultdict(dict)
    with open(path) as fh:
        for r in csv.DictReader(fh):
            if r["predictor"] != "clf" and r["predictor"] != "fm":
                continue
            try:
                v = (float(r["UNC_debiased"]) - float(r["RES"])) if metric == "recal" else float(r[metric])
                out[(r["predictor"], r["level"], r["arm"])][int(r["epoch"])] = v
            except (ValueError, KeyError):
                continue
    return out


def pooled_floor(series, min_epoch: int = 1):
    """2*sqrt(mean per-epoch variance) over epochs shared by all seeds.

    Epoch 0 is excluded by default: pre-acquisition every seed trains on identical
    data, so its spread is structurally ~0 and using it makes every gap significant.
    """
    shared = sorted({e for e in set.intersection(*[set(s) for s in series]) if e >= min_epoch})
    if len(shared) < 2:
        return None, 0, shared
    var = [statistics.variance([s[e] for s in series]) for e in shared]
    return 2 * math.sqrt(sum(var) / len(var)), len(shared), shared


def verdicts(data, predictor: str, level: str):
    key = lambda arm: (predictor, level, arm)
    seeds = [key(CONTROL)] + [key(CONTROL + s) for s in SEED_SUFFIXES]
    if not all(k in data for k in seeds):
        return None, f"floor incomplete (need {CONTROL} + {' + '.join(SEED_SUFFIXES)})"
    floor, n_ep, _ = pooled_floor([data[k] for k in seeds])
    if floor is None:
        return None, "fewer than 2 shared floor epochs"
    present = [a for a in ARMS if key(a) in data]
    if not present:
        return None, "no arms scored"
    common = sorted(set.intersection(*[set(data[key(a)]) for a in present]) & set(data[key(CONTROL)]))
    if len(common) < 2:
        return None, f"fewer than 2 epochs shared by all arms (have {len(common)})"
    eps = common[-2:]
    rows = []
    for a in present:
        # Full-trajectory sign check, not just the two-epoch window. At [CLF] high the
        # least-harmful arm rotated almost every epoch and ALL FOUR arms held that slot,
        # so two adjacent epochs agreed by luck -- even across independent metrics. A
        # two-epoch window cannot detect a quantity oscillating on a two-epoch timescale.
        traj = [data[key(a)][e] - data[key(CONTROL)][e] for e in common if e >= 1]
        sign_stable = len({g > 0 for g in traj}) == 1 if traj else False
        n_traj = len(traj)
        gaps = [data[key(a)][e] - data[key(CONTROL)][e] for e in eps]
        mult = [g / floor for g in gaps]
        # Three outcomes, not two. Collapsing them loses the campaign's most
        # important result: at xhigh the epistemic arms sit BELOW the floor at both
        # epochs, which is a clean null (they tie random sampling) and reads as a
        # finding -- not the same as an arm whose sign flips between epochs.
        over = [abs(m) > 1 for m in mult]
        if all(over) and (mult[0] > 0) == (mult[1] > 0):
            label = "DISTINGUISHABLE"
        elif not any(over):
            label = "within noise (null)"
        else:
            label = "not stable"
        if label == "DISTINGUISHABLE" and not sign_stable:
            label += f"  [!] sign flips over {n_traj} epochs -- treat as unstable"
        elif label == "DISTINGUISHABLE":
            label += f"  (sign stable {n_traj}/{n_traj} ep)"
        rows.append((a, eps, gaps, mult, label))
    return (floor, n_ep, rows), None


def load_det(exp_root: Path, field: str = "brier"):
    """Deterministic runs, from artifacts_v2.json label metrics.

    The probability metrics cannot cover det at all: they score against a
    ground-truth p_success estimated from repeated rollouts, and a deterministic
    system has no such thing (its p is degenerate {0,1}). So det is judged on the
    label metrics the evaluator already writes. Same floor and stability rules.
    """
    import json

    out = collections.defaultdict(dict)
    for run in sorted(exp_root.glob("clf_det_*")):
        arm = run.name[len("clf_det_"):]
        for f in sorted(run.glob("epoch_*/artifacts_v2.json")):
            try:
                v = json.load(open(f))["eval_metrics"]["threshold_free"][field]
                out[("clf", "det", arm)][int(f.parent.name.split("_")[1])] = v
            except Exception:
                continue
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", type=Path, required=True)
    ap.add_argument("--exp-root", type=Path,
                    default=Path("/common/users/shared/pracsys/adaptive_roa_experiments/ensemble_epistemic"),
                    help="run tree, used for the deterministic level (label metrics)")
    ap.add_argument("--det-field", default="brier",
                    help="threshold_free field for det: brier (default), log_score, auc")
    ap.add_argument("--metric", default="recal",
                    help="'recal' (UNC-RES, default) or a column such as sAUROC / brier_debiased")
    ap.add_argument("--levels", nargs="*", default=["low", "med", "high", "xhigh"])
    ap.add_argument("--predictors", nargs="*", default=["clf", "fm"])
    a = ap.parse_args()

    data = _load(a.metrics, a.metric)
    print(f"metric: {a.metric}   (pooled floor, 2 consecutive epochs, epoch 0 excluded)\n")

    # det first: different metric source, same rules.
    if "det" in a.levels or a.levels == ["low", "med", "high", "xhigh"]:
        det = load_det(a.exp_root, a.det_field)
        if det:
            res, why = verdicts(det, "clf", "det")
            if res is None:
                print(f"clf det ({a.det_field}): -- {why}\n")
            else:
                floor, n_ep, rows = res
                print(f"clf det ({a.det_field}, label metric): pooled 2*SD = {floor:.5f} over {n_ep} epochs")
                for arm, eps, gaps, mult, label in rows:
                    cells = "  ".join(f"ep{e}: {g:+.5f} ({m:+.1f}x)" for e, g, m in zip(eps, gaps, mult))
                    print(f"   {arm:<10} {cells}   -> {label}")
                print()

    for pred in a.predictors:
        for lvl in a.levels:
            res, why = verdicts(data, pred, lvl)
            if res is None:
                print(f"{pred} {lvl}: -- {why}")
                continue
            floor, n_ep, rows = res
            print(f"{pred} {lvl}: pooled 2*SD = {floor:.5f} over {n_ep} epochs")
            for arm, eps, gaps, mult, label in rows:
                cells = "  ".join(f"ep{e}: {g:+.5f} ({m:+.1f}x)" for e, g, m in zip(eps, gaps, mult))
                print(f"   {arm:<10} {cells}   -> {label}")
            print()


if __name__ == "__main__":
    main()
