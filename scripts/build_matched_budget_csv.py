#!/usr/bin/env python3
"""Matched-budget comparison CSV: campaign-B arms vs every prior arm.

One row per (system, arm) at a SINGLE target trajectory budget per system, so
every row is directly comparable:

    pendulum   500     cartpole  1000     quad2d  12000

Arms covered
------------
campaign B (matched schedule)  entropy, entropy_tiebreak, mode_sep  x d2 {0.5,1.0}
FM baselines                   fm_ranked, fm_direct, fm_nonadapt
classifiers                    clf_adaptive, clf_nonadapt
other                          partx, and the campaign-A dispersion arms

Columns
-------
f1, sep_pct, invalid_pct, uncertain_pct, tpr, fpr, tnr, fnr
  from eval_metrics.lambda_delta. TPR=tp/(tp+fn), TNR=tn/(tn+fp), etc.

<class>_<outer>_of_<inner>   (27 = 3 classes x 3 outer x 3 inner)
  from eval_metrics.mc_sample_errors. Each eval point has K MC samples, so the
  final-state error is aggregated twice: INNER over the K samples for a point,
  then OUTER over the eval points. e.g. full_p90_of_medians = 90th pct across
  points of (median across that point's K samples).
  classes: full / certain_success / certain_failure (predicted class).

`train_traj` records the budget actually used and `budget_exact` flags whether it
hit the target on the nose -- arms are on different epoch grids, so a few land on
a neighbouring epoch instead. Nothing is silently snapped.

Usage:  python scripts/build_matched_budget_csv.py [-o out.csv]
"""
from __future__ import annotations

import argparse, csv, glob, json, os, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "docs" / "plots"))
import plot_clf_vs_fm as P                                          # noqa: E402
from adaptive_roa.adaptive_v2.eval.full_roa import (                # noqa: E402
    _predict_lambda_delta,
    _threshold_free_metrics,
)

# system -> the one budget every arm is compared at
TARGET = {"pendulum": 500, "cartpole": 1000, "quad2d": 12000, "quad3d": 29000}
# system -> (experiment root, adapt_iter token identifying the campaign-B runs)
NEW = {"pendulum": ("adaptive_pendulum_lqr", "adapt_iter_20"),
       "cartpole": ("adaptive_cartpole_pybullet", "adapt_iter_20"),
       "quad2d": ("adaptive_quadrotor2d", "adapt_iter_10"),
       "quad3d": ("adaptive_quadrotor3d", "adapt_iter_20")}
NEW_ARMS = ("entropy", "entropy_tiebreak", "mode_sep")
CAMPAIGN_B_TS = "2026-07-30_18"          # campaign B launched at/after this
ERR_CLASSES = ("full", "certain_success", "certain_failure")
OUTER = ("mean", "median", "p90")        # over eval points
INNER = ("means", "medians", "p90s")     # over the K MC samples

ERR_COLS = [f"{c}_{o}_of_{i}" for c in ERR_CLASSES for o in OUTER for i in INNER]
# Threshold-free scores, recomputed from the stored per-point probabilities.
# num_mc_samples is carried alongside because it sets the resolution of p: a
# K-sample arm can only emit k/K, which bounds how good Brier/log score can get.
TF_COLS = ["auc", "auprc", "brier", "log_score", "num_mc_samples",
           "log_score_smoothing", "n_saturated", "base_rate",
           "tf_source", "tf_validated"]
COLS = (["system", "arm", "d2_ratio", "target_traj", "train_traj", "budget_exact",
         "epoch", "n_total", "f1", "sep_pct", "invalid_pct", "uncertain_pct",
         "tpr", "fpr", "tnr", "fnr"] + TF_COLS +
        ["f1_win3_median", "invalid_pct_win3_median", "win3_n_epochs"] + ERR_COLS)


# ── threshold-free scores ────────────────────────────────────────────────
# artifacts_v2.json stores only aggregates, and every run here predates the
# threshold_free block, so AUC/AUPRC/Brier/log-score are recomputed from the
# per-point probabilities that each eval already wrote to disk. Two layouts
# exist; both are checked against the artifact's own confusion matrix before
# their numbers are used (see `_validate`), so a mismatched eval set or MC
# budget shows up as tf_validated=FAIL rather than a plausible wrong number.
_LABELS: dict[str, list] = {}


def _label_bank(system: str) -> list:
    """[(start_states, true_labels)] harvested from any per-point npz for `system`.

    mc_cache stores probabilities but no ground truth, so labels are transferred
    from a run whose start_states are bit-identical -- i.e. literally the same
    eval file in the same order. Exact equality is required, never a tolerance.
    """
    if system in _LABELS:
        return _LABELS[system]
    bank, seen = [], set()
    roots = [f"{P.EXP}/{NEW[system][0]}/outputs/*/*/epoch_*/full_roa_per_point.npz"]
    for arm in ("clf_adaptive", "clf_nonadapt"):
        spec = P.CLF_DIR.get((system, arm))
        if spec:
            roots.append(f"{spec}/epoch_*/full_roa_per_point.npz")
    for pat in roots:
        for p in sorted(glob.glob(pat)):
            try:
                z = np.load(p)
                s, y = z["start_states"], z["true_labels"]
            except Exception:
                continue
            key = (s.shape, float(s[0, 0]), float(s[-1, -1]))
            if key in seen:
                continue
            seen.add(key)
            bank.append((s, y))
    _LABELS[system] = bank
    return bank


def _probs(art: dict, system: str):
    """(p_success, p_failure, p_invalid, y_true, source) for one epoch, or None."""
    path = art.get("__path__")
    if not path:
        return None
    d = os.path.dirname(path)

    pp = os.path.join(d, "full_roa_per_point.npz")
    if os.path.exists(pp):
        z = np.load(pp)
        return (z["p_success"], z["p_failure"], z["p_invalid"],
                z["true_labels"], "per_point")

    # Older FM runs evaluated into {run}/evaluations/{cfg}/epoch_N and kept the
    # raw MC draws in {run}/mc_cache instead.
    run, ep = d, os.path.basename(d)
    for _ in range(4):
        run = os.path.dirname(run)
        cache = os.path.join(run, "mc_cache", f"{ep}_test.npz")
        if os.path.exists(cache):
            z = np.load(cache)
            ml, K = z["mc_labels"], int(z["num_mc_samples"])
            s = z["start_states"]
            y = next((yy for ss, yy in _label_bank(system)
                      if ss.shape == s.shape and np.array_equal(ss, s)), None)
            if y is None:
                return None
            return ((ml == 1).sum(1) / K, (ml == -1).sum(1) / K,
                    (ml == 0).sum(1) / K, y, f"mc_cache_K{K}")
    return None


def _validate(ps, pf, pi, y, ld) -> str | None:
    """Rule whose lambda/delta confusion matrix reproduces the artifact, else None.

    The stored decision_rule isn't in the artifact, so both are tried. Matching
    tp/tn/fp/fn on ~50k points is the proof that these probabilities are the
    exact array the recorded metrics were computed from.
    """
    want = {k: ld.get(k) for k in ("tp", "tn", "fp", "fn")}
    if any(v is None for v in want.values()):
        return None
    for rule in ("one_sided", "two_sided"):
        pred, _ = _predict_lambda_delta(ps, pf, pi, ld["lambda_star"], ld["delta"],
                                        rule, ld.get("invalid_threshold"))
        c = (pred == 1) | (pred == 0)
        yp, yt = np.where(pred[c] == 1, 1, -1), y[c]
        got = {"tp": int(((yp == 1) & (yt == 1)).sum()),
               "tn": int(((yp == -1) & (yt == -1)).sum()),
               "fp": int(((yp == 1) & (yt == -1)).sum()),
               "fn": int(((yp == -1) & (yt == 1)).sum())}
        if got == want:
            return rule
    return None


def threshold_free(art: dict, system: str) -> dict:
    """AUC/AUPRC/Brier/log-score for one epoch, or Nones if unrecoverable."""
    em = art.get("eval_metrics") or {}
    K = em.get("num_mc_samples")
    blank = dict.fromkeys(TF_COLS) | {"num_mc_samples": K}
    got = _probs(art, system)
    if got is None:
        return {**blank, "tf_source": "none"}
    ps, pf, pi, y, source = got
    rule = _validate(ps, pf, pi, y, em.get("lambda_delta") or {})
    if rule is None:
        return {**blank, "tf_source": source, "tf_validated": "FAIL"}
    m = _threshold_free_metrics(ps, y, num_mc_samples=K, p_invalid=pi)
    return {k: m[k] for k in ("auc", "auprc", "brier", "log_score",
                              "log_score_smoothing", "n_saturated", "base_rate")
            } | {"num_mc_samples": K, "tf_source": source, "tf_validated": rule}


def metrics(art: dict) -> dict | None:
    """Flatten one artifacts_v2.json into the column set (None if unusable)."""
    em = art.get("eval_metrics") or {}
    ld = em.get("lambda_delta") or {}
    if not ld or ld.get("f1") is None:
        return None
    tp, fp, tn, fn = (ld.get(k) for k in ("tp", "fp", "tn", "fn"))
    rate = lambda a, b: (a / b) if (a is not None and b) else None   # noqa: E731
    pos = (tp + fn) if None not in (tp, fn) else None
    neg = (tn + fp) if None not in (tn, fp) else None
    n_conf, n_unc, n_inv = ld.get("n_confident"), ld.get("n_uncertain"), ld.get("n_invalid")

    row = {
        "epoch": art.get("epoch"),
        "n_total": em.get("n_total") or (
            sum(x for x in (n_conf, n_unc, n_inv) if x is not None) or None),
        "f1": ld.get("f1"),
        "sep_pct": ld.get("separatrix_pct"),
        "invalid_pct": ld.get("invalid_pct"),
        "uncertain_pct": ld.get("uncertain_pct"),
        "tpr": rate(tp, pos), "fnr": rate(fn, pos),
        "tnr": rate(tn, neg), "fpr": rate(fp, neg),
    }
    mc = em.get("mc_sample_errors") or {}
    for c in ERR_CLASSES:
        blk = mc.get(c) or {}
        for o in OUTER:
            for i in INNER:
                row[f"{c}_{o}_of_{i}"] = blk.get(f"{o}_of_{i}")
    return row


def hydra_schedule(run_dir: str):
    """(initial_train_size, samples_per_epoch) from the run's hydra snapshot.

    The pre-newcrit FM runs never recorded train_trajectories in their artifacts,
    so without this every pendulum/cartpole baseline drops out of the comparison
    silently. The schedule is linear, so the config reconstructs it exactly.
    """
    import yaml
    d = run_dir
    for _ in range(4):
        cfg = os.path.join(d, ".hydra", "config.yaml")
        if os.path.exists(cfg):
            try:
                c = yaml.safe_load(open(cfg))
                i, s = c.get("initial_train_size"), c.get("samples_per_epoch")
                if i is not None and s is not None:
                    return int(i), int(s)
            except Exception:
                return None
        d = os.path.dirname(d)
    return None


def load_epochs(paths: list[str], run_dir: str | None = None) -> list[tuple[int, dict]]:
    """[(train_trajectories, artifact)] for every readable epoch."""
    sched = None
    out = []
    for p in paths:
        try:
            art = json.load(open(p))
        except Exception:
            continue
        art["__path__"] = p          # so the per-point probabilities can be found
        ep = art.get("epoch")
        if ep is None:                       # old layout: epoch lives in the dir name
            base = os.path.basename(os.path.dirname(p))
            if base.startswith("epoch_"):
                ep = int(base.split("_")[1])
                art["epoch"] = ep
        tt = art.get("train_trajectories")
        if tt is None:                       # old layout: sibling results.json has it
            try:
                tt = json.load(open(os.path.join(os.path.dirname(p), "results.json"))
                               ).get("train_trajectories")
            except Exception:
                pass
        if tt is None and run_dir is not None and ep is not None:
            if sched is None:
                sched = hydra_schedule(run_dir) or False
            if sched:
                tt = sched[0] + sched[1] * ep
        out.append((tt, art))
    return out


def pick(entries, target):
    """Epoch at the target budget, else the nearest one. Returns (tt, art, exact)."""
    usable = [(tt, a) for tt, a in entries if tt is not None]
    if not usable:
        return None
    exact = [x for x in usable if x[0] == target]
    if exact:
        return exact[0][0], exact[0][1], True
    tt, art = min(usable, key=lambda x: abs(x[0] - target))
    return tt, art, False


def window_stats(entries, tt):
    """Median f1 / invalid_pct over the 3 epochs centred on the chosen budget.

    invalid_pct is bistable -- it swings between ~0.09 and ~0.40 on adjacent
    checkpoints, and a high-invalid epoch shrinks the committed set to easy
    points, mechanically inflating F1 (within-run corr(F1, invalid) is ~+0.8).
    Pinning to a single budget therefore samples that coin flip. These columns
    give the local median so a spiked epoch can be spotted and discounted.
    """
    usable = sorted([(t, a) for t, a in entries if t is not None], key=lambda x: x[0])
    idx = next((i for i, (t, _) in enumerate(usable) if t == tt), None)
    if idx is None:
        return None, None, None
    # Slide the window to keep 3 samples at the ends rather than truncating to 2.
    # The target budget is the LAST epoch for most arms, so a centred window would
    # otherwise degrade to 2 points for 17 of 18 quad2d rows -- and an even-length
    # window has no unique middle, which biased the statistic upward.
    n = len(usable)
    lo = min(max(0, idx - 1), max(0, n - 3))
    win = usable[lo: lo + 3]
    f1s, invs = [], []
    for _, a in win:
        ld = ((a.get("eval_metrics") or {}).get("lambda_delta") or {})
        if ld.get("f1") is not None:
            f1s.append(ld["f1"])
        if ld.get("invalid_pct") is not None:
            invs.append(ld["invalid_pct"])

    def med(v):
        if not v:
            return None
        s = sorted(v)
        m = len(s) // 2
        return s[m] if len(s) % 2 else (s[m - 1] + s[m]) / 2.0

    return med(f1s), med(invs), len(win)


def emit(rows, system, arm, d2, entries):
    got = pick(entries, TARGET[system])
    if not got:
        return
    tt, art, exact = got
    m = metrics(art)
    if not m:
        return
    f1w, invw, nw = window_stats(entries, tt)
    rows.append({"system": system, "arm": arm, "d2_ratio": d2,
                 "target_traj": TARGET[system], "train_traj": tt,
                 "budget_exact": exact, **m, **threshold_free(art, system),
                 "f1_win3_median": f1w, "invalid_pct_win3_median": invw,
                 "win3_n_epochs": nw})


def campaign_b(system, rows):
    root, tag = NEW[system]
    for run in sorted(glob.glob(f"{P.EXP}/{root}/outputs/*{tag}_sampling_mode_*")):
        b = os.path.basename(run)
        if "d2_ratio_" not in b:
            continue
        arm = b.split("sampling_mode_")[1].split("_selection")[0]
        if arm not in NEW_ARMS:
            continue
        d2 = b.split("d2_ratio_")[1].split("_")[0]
        for ts in sorted(glob.glob(run + "/*")):
            if os.path.basename(ts) < CAMPAIGN_B_TS or not os.path.isdir(ts):
                continue
            eps = sorted(glob.glob(ts + "/epoch_*/artifacts_v2.json"))
            if eps:
                emit(rows, system, arm, d2, load_epochs(eps))


def fm_baselines(system, rows):
    inline = system in P.NEWCRIT_SYSTEMS
    for arm in ("fm_ranked", "fm_direct", "fm_nonadapt"):
        d = P.resolve_fm_newcrit(system, arm) if inline else P.resolve_fm(system, arm)
        if not d:
            continue
        eps = (sorted(glob.glob(f"{d}/epoch_*/artifacts_v2.json")) if inline else
               sorted(glob.glob(f"{d}/evaluations/{P.FM_EVAL.get(system)}/epoch_*/artifacts_v2.json")))
        if eps:
            emit(rows, system, arm, "baseline", load_epochs(eps, run_dir=d))


def dispersion(system, rows):
    for arm in ("fm_disp_greedy_d05", "fm_disp_greedy_d10", "fm_disp_gdiv_d05",
                "fm_disp_gdiv_d10", "fm_disp_prop_d05", "fm_disp_prop_d10"):
        d = P.resolve_fm_dispersion(system, arm)
        if not d:
            continue
        eps = sorted(glob.glob(f"{d}/epoch_*/artifacts_v2.json"))
        if eps:
            emit(rows, system, arm, "campaignA", load_epochs(eps, run_dir=d))


def classifiers(system, rows):
    for arm in ("clf_adaptive", "clf_nonadapt"):
        spec = P.CLF_DIR.get((system, arm))
        if not spec:
            continue
        for d in (sorted(glob.glob(spec)) if any(c in spec for c in "*?") else [spec]):
            eps = sorted(glob.glob(f"{d}/epoch_*/artifacts_v2.json"))
            if eps:
                emit(rows, system, arm, "-", load_epochs(eps, run_dir=d))
            break


def partx(system, rows):
    d = getattr(P, "PARTX_DIR", {}).get(system)
    if not d:
        return
    eps = sorted(glob.glob(f"{d}/epoch_*/artifacts_v2.json"))
    if eps:
        emit(rows, system, "partx", "-", load_epochs(eps, run_dir=d))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default="docs/plots/matched_budget_metrics.csv")
    a = ap.parse_args()

    rows = []
    for s in ("pendulum", "cartpole", "quad2d", "quad3d"):
        campaign_b(s, rows)
        fm_baselines(s, rows)
        classifiers(s, rows)
        dispersion(s, rows)
        partx(s, rows)

    order = {s: i for i, s in enumerate(("pendulum", "cartpole", "quad2d", "quad3d"))}
    rows.sort(key=lambda r: (order[r["system"]], -(r["f1"] or 0)))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {a.out}  ({len(rows)} rows, {len(COLS)} columns)\n")
    for s in ("pendulum", "cartpole", "quad2d", "quad3d"):
        sub = [r for r in rows if r["system"] == s]
        inexact = [r for r in sub if not r["budget_exact"]]
        print(f"{s}: {len(sub)} arms @ target {TARGET[s]}"
              + (f"   [{len(inexact)} off-target: "
                 + ", ".join(f"{r['arm']}@{r['train_traj']}" for r in inexact) + "]"
                 if inexact else "   [all exact]"))

    ok = [r for r in rows if r["tf_validated"] in ("one_sided", "two_sided")]
    bad = [r for r in rows if r["tf_validated"] == "FAIL"]
    gap = [r for r in rows if r["tf_source"] == "none"]
    print(f"\nthreshold-free: {len(ok)}/{len(rows)} recovered and validated against "
          f"the artifact's own confusion matrix")
    if bad:
        print("  FAILED validation (numbers withheld): "
              + ", ".join(f"{r['system']}/{r['arm']}" for r in bad))
    if gap:
        print("  no per-point probabilities on disk: "
              + ", ".join(f"{r['system']}/{r['arm']}(d2={r['d2_ratio']})" for r in gap))


if __name__ == "__main__":
    main()
