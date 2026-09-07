#!/usr/bin/env python
"""Probability-comparison metrics for the stochastic pendulum ROA models.

The eval grid gives a Monte-Carlo ground truth: each cell's success probability
p_i is itself estimated from M rollouts, and the model's p_hat_i from K samples.
Both sides therefore carry binomial noise, and a plain Brier score charges the
model for noise it did not create. Every estimator here that can be debiased is
debiased on both sides:

    E[(p_hat - p)^2] = (p_hat_true - p_true)^2 + p_hat(1-p_hat)/(K-1)_noise
                                              + p(1-p)/(M-1)_noise

so subtracting the two unbiased variance estimates leaves the signal.

Discriminative predictors (the classifier arm) emit a continuous p_hat from a
single forward pass, i.e. K = infinity, and their sampling-noise term vanishes.
Pass k=None for those.

Usage:
    python scripts/stoch_prob_metrics.py --spec runs.json --out docs/stoch_compare
    python scripts/stoch_prob_metrics.py --selftest
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

EPS = 1e-12
# Above this many distinct predicted values, treat p_hat as continuous and bin it
# rather than grouping on exact values (see decomposition()).
_MAX_EXACT_LEVELS = 200
_DEFAULT_BINS = 20


# --------------------------------------------------------------------------
# core estimators
# --------------------------------------------------------------------------
def _noise(p: np.ndarray, n: float | None) -> np.ndarray:
    """Unbiased estimate of the sampling variance of a proportion from n draws.

    n <= 1 means the value is not a proportion over draws at all (a classifier
    emits p directly, and the pipeline records num_mc_samples=1 for it), so
    there is no sampling noise to remove.
    """
    if n is None or not np.isfinite(n) or n <= 1:
        return np.zeros_like(p)
    return p * (1.0 - p) / (n - 1.0)


def debiased_brier(p_hat: np.ndarray, p: np.ndarray, k: float | None, m: float) -> float:
    """Doubly debiased Brier score (MSE against the noiseless field)."""
    return float(np.mean((p_hat - p) ** 2 - _noise(p_hat, k) - _noise(p, m)))


def pointwise_debiased_error(p_hat: np.ndarray, p: np.ndarray, k: float | None,
                             m: float) -> np.ndarray:
    """Per-cell debiased squared error; its mean is the debiased Brier score."""
    return (p_hat - p) ** 2 - _noise(p_hat, k) - _noise(p, m)


def paired_delta(p_hat_a: np.ndarray, p_hat_b: np.ndarray, p: np.ndarray,
                 k_a: float | None, k_b: float | None, m: float) -> dict:
    """Paired arm-vs-arm comparison on the same eval cells.

    Both arms see the same grid, so pairing removes the cell-to-cell variance
    that dominates an unpaired comparison — and the ground-truth noise term
    cancels identically, so the result does not depend on M at all. Negative
    delta means arm A beats arm B.
    """
    d = (pointwise_debiased_error(p_hat_a, p, k_a, m)
         - pointwise_debiased_error(p_hat_b, p, k_b, m))
    n = len(d)
    se = float(np.std(d, ddof=1) / np.sqrt(n))
    mean = float(np.mean(d))
    return {"delta": mean, "se": se, "z": mean / se if se > 0 else float("nan"), "n": n}


def debiased_variance(p: np.ndarray, m: float) -> float:
    """Debiased variance of the true field — the reference a skill score divides by."""
    return float(np.mean((p - p.mean()) ** 2 - _noise(p, m)))


def skill_score(p_hat: np.ndarray, p: np.ndarray, k: float | None, m: float) -> float:
    var = debiased_variance(p, m)
    if abs(var) < EPS:
        return float("nan")
    return 1.0 - debiased_brier(p_hat, p, k, m) / var


def decomposition(p_hat: np.ndarray, p: np.ndarray, k: float | None, m: float,
                  n_bins: int | None = None) -> dict:
    """Murphy REL/RES/UNC decomposition of the Brier score.

    With K MC samples p_hat lives on the K+1 grid {l/K}, and binning on those
    exact levels makes the identity BRIER = REL - RES + UNC exact. A continuous
    p_hat (the classifier) is binned into quantiles instead, which leaves a
    within-bin residual reported as `decomp_gap`. Grouping a continuous predictor
    on exact values is not merely imprecise -- it is degenerate, giving one bin
    per cell so that REL equals the Brier score and RES equals UNC.
    """
    if n_bins is None:
        levels, inv = np.unique(p_hat, return_inverse=True)
        n_groups = len(levels)
        if n_groups > _MAX_EXACT_LEVELS:
            # A continuous predictor (the classifier emits p directly) would get
            # roughly one bin per cell, and the decomposition collapses: every
            # bin's mean prediction equals its single member, so REL becomes the
            # Brier score and RES becomes UNC. Bin instead.
            n_bins = _DEFAULT_BINS
    if n_bins is not None:
        edges = np.quantile(p_hat, np.linspace(0, 1, n_bins + 1))
        edges = np.unique(edges)
        inv = np.clip(np.searchsorted(edges, p_hat, side="right") - 1, 0, len(edges) - 2)
        n_groups = len(edges) - 1

    n = len(p)
    counts = np.bincount(inv, minlength=n_groups).astype(float)
    nz = counts > 0
    sum_hat = np.bincount(inv, weights=p_hat, minlength=n_groups)
    sum_true = np.bincount(inv, weights=p, minlength=n_groups)
    mean_hat = np.divide(sum_hat, counts, out=np.zeros_like(sum_hat), where=nz)
    mean_true = np.divide(sum_true, counts, out=np.zeros_like(sum_true), where=nz)
    pbar = float(p.mean())

    rel = float(np.sum(counts[nz] * (mean_hat[nz] - mean_true[nz]) ** 2) / n)
    res = float(np.sum(counts[nz] * (mean_true[nz] - pbar) ** 2) / n)
    unc = float(np.mean((p - pbar) ** 2))
    brier = float(np.mean((p_hat - p) ** 2))

    return {
        "REL": rel,
        "RES": res,
        "UNC": unc,
        # Debiasing splits cleanly across the identity: the model-noise term sits
        # in REL, the ground-truth-noise term in UNC.
        "REL_debiased": rel - float(np.mean(_noise(p_hat, k))),
        "UNC_debiased": unc - float(np.mean(_noise(p, m))),
        "decomp_gap": brier - (rel - res + unc),
        "n_bins_used": int(np.count_nonzero(nz)),
    }


def soft_auroc(p_hat: np.ndarray, p: np.ndarray) -> float:
    """Probability-weighted AUROC: every cell contributes p_i positive mass and
    (1-p_i) negative mass, so no thresholding of the ground truth is needed."""
    order = np.argsort(p_hat, kind="mergesort")
    ph, pp = p_hat[order], p[order]
    w = 1.0 - pp
    cw = np.concatenate(([0.0], np.cumsum(w)))

    _, inv, counts = np.unique(ph, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)
    starts = np.concatenate(([0], ends[:-1]))
    w_less = cw[starts][inv]
    w_tie = (cw[ends] - cw[starts])[inv]

    num = float(np.sum(pp * (w_less + 0.5 * (w_tie - w))))
    den = float(p.sum() * (1.0 - p).sum() - np.sum(p * (1.0 - p)))
    return num / den if abs(den) > EPS else float("nan")


def sharpness(p_hat: np.ndarray, p: np.ndarray, k: float | None, m: float) -> dict:
    """Debiased spread of the predicted field and of the true field."""
    if k is None or not np.isfinite(k) or k <= 1:
        sharp = float(np.mean(p_hat * (1.0 - p_hat)))
    else:
        sharp = float(k / (k - 1.0) * np.mean(p_hat * (1.0 - p_hat)))
    # Same guard as the k branch above. m is the GT trials per eval cell, and a
    # DETERMINISTIC level ships m = 1 (one rollout is enough with no noise), which
    # made m/(m-1) a ZeroDivisionError that killed every epoch of the level rather
    # than just this one metric. With a single draw there is no within-cell
    # sampling variance to remove, so the raw spread IS the debiased spread --
    # and for a deterministic field it is legitimately 0, since p is 0 or 1
    # everywhere. Exposed by cartpole safe_explorer_ppo/baseline (trials = 1);
    # also applies to any f_0.000-style level.
    if not np.isfinite(m) or m <= 1:
        sharp_star = float(np.mean(p * (1.0 - p)))
    else:
        sharp_star = float(m / (m - 1.0) * np.mean(p * (1.0 - p)))
    return {
        "SHARP": sharp,
        "SHARP_star": sharp_star,
    }


def risk_coverage(p_hat: np.ndarray, p: np.ndarray, k: float | None, m: float,
                  coverages=(0.2, 0.5, 0.8, 1.0)) -> dict:
    """Debiased selective risk over the most-confident fraction, plus its area.

    Confidence is distance from the p = 1/2 decision point, so r(c) answers
    "how wrong is the model where it is most sure".
    """
    err = (p_hat - p) ** 2 - _noise(p_hat, k) - _noise(p, m)
    order = np.argsort(-np.abs(p_hat - 0.5), kind="mergesort")
    e = err[order]
    n = len(e)
    r = np.cumsum(e) / np.arange(1, n + 1)
    out = {"AURC": float(r.mean())}
    for c in coverages:
        j = max(1, int(np.ceil(c * n)))
        out[f"risk@{c:g}"] = float(r[j - 1])
    return out


def reliability_curve(p_hat: np.ndarray, p: np.ndarray, n_bins: int = 20) -> dict:
    """Binned predicted-vs-true probability, the curve REL measures deviation from.

    Bins are fixed-width on [0,1] so curves from different arms are comparable;
    empty bins are dropped rather than plotted at zero.
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.searchsorted(edges, p_hat, side="right") - 1, 0, n_bins - 1)
    counts = np.bincount(idx, minlength=n_bins).astype(float)
    nz = counts > 0
    mean_hat = np.bincount(idx, weights=p_hat, minlength=n_bins)[nz] / counts[nz]
    mean_true = np.bincount(idx, weights=p, minlength=n_bins)[nz] / counts[nz]
    return {"p_hat": mean_hat, "p_true": mean_true, "count": counts[nz]}


def rc_curve(p_hat: np.ndarray, p: np.ndarray, k: float | None, m: float,
             n_points: int = 200) -> dict:
    """Debiased selective risk as a function of coverage."""
    err = (p_hat - p) ** 2 - _noise(p_hat, k) - _noise(p, m)
    order = np.argsort(-np.abs(p_hat - 0.5), kind="mergesort")
    r = np.cumsum(err[order]) / np.arange(1, len(err) + 1)
    n = len(r)
    js = np.unique(np.clip((np.linspace(0, 1, n_points) * n).astype(int), 1, n))
    return {"coverage": js / n, "risk": r[js - 1]}


def thresholded_roa(p_hat: np.ndarray, p: np.ndarray, beta: float) -> dict:
    """Classification view: both sides thresholded at the same beta."""
    y = (p >= beta).astype(int)
    yh = (p_hat >= beta).astype(int)
    tp = int(np.sum((y == 1) & (yh == 1)))
    fp = int(np.sum((y == 0) & (yh == 1)))
    fn = int(np.sum((y == 1) & (yh == 0)))
    tn = int(np.sum((y == 0) & (yh == 0)))
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if prec + rec and np.isfinite(prec + rec) else float("nan")
    return {
        f"acc@{beta:g}": (tp + tn) / len(y),
        f"prec@{beta:g}": prec,
        f"rec@{beta:g}": rec,
        f"f1@{beta:g}": f1,
        f"roa_frac_true@{beta:g}": float(y.mean()),
        f"roa_frac_pred@{beta:g}": float(yh.mean()),
    }


def calibration_scores(p_hat: np.ndarray, p: np.ndarray, successes: np.ndarray,
                       trials: np.ndarray) -> dict:
    """KL and per-rollout log score, with the oracle/climatology floors that say
    how much of the score is irreducible sampling noise."""
    q = np.clip(p_hat, 1e-3, 1 - 1e-3)
    pc = np.clip(p, EPS, 1 - EPS)
    kl = float(np.mean(pc * np.log(pc / q) + (1 - pc) * np.log((1 - pc) / (1 - q))))

    def ls(qq):
        qq = np.clip(qq, 1e-3, 1 - 1e-3)
        return float(np.mean(-(successes * np.log(qq) + (trials - successes) * np.log(1 - qq)) / trials))

    return {
        "KL": kl,
        "log_score": ls(q),
        "log_score_oracle": ls(p),
        "log_score_climatology": ls(np.full_like(p, p.mean())),
        "MAE": float(np.mean(np.abs(p_hat - p))),
        "bias": float(np.mean(p_hat - p)),
    }


def all_metrics(p_hat: np.ndarray, p: np.ndarray, successes: np.ndarray,
                trials: np.ndarray, k: float | None, n_bins: int | None = None,
                betas=(0.25, 0.5, 0.75)) -> dict:
    m = float(np.mean(trials))
    out: dict = {"n_points": int(len(p)), "K": k, "M": m}
    out["brier_raw"] = float(np.mean((p_hat - p) ** 2))
    out["brier_debiased"] = debiased_brier(p_hat, p, k, m)
    e = pointwise_debiased_error(p_hat, p, k, m)
    out["brier_debiased_se"] = float(np.std(e, ddof=1) / np.sqrt(len(e)))
    out["VAR_debiased"] = debiased_variance(p, m)
    out["skill_score"] = skill_score(p_hat, p, k, m)
    out.update(decomposition(p_hat, p, k, m, n_bins))
    out["sAUROC"] = soft_auroc(p_hat, p)
    out.update(sharpness(p_hat, p, k, m))
    out.update(risk_coverage(p_hat, p, k, m))
    for b in betas:
        out.update(thresholded_roa(p_hat, p, b))
    out.update(calibration_scores(p_hat, p, successes, trials))
    return out


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
@dataclass
class Arm:
    predictor: str
    level: str
    arm: str
    run_dir: Path


def load_ground_truth(dataset_root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(dataset_root / "eval_success_prob.npz") as z:
        return (z["starts"].astype(np.float64), z["p_success"].astype(np.float64),
                z["successes"].astype(np.float64), z["trials"].astype(np.float64))


def match_to_truth(states: np.ndarray, gt_starts: np.ndarray, tol: float = 1e-3) -> np.ndarray:
    d, idx = cKDTree(gt_starts).query(states.astype(np.float64), k=1)
    if d.max() > tol:
        raise ValueError(f"eval states do not match the ground-truth grid (max dist {d.max():.3g})")
    return idx


def epoch_dirs(run_dir: Path) -> list[Path]:
    return sorted(d for d in run_dir.glob("epoch_*")
                  if (d / "full_roa_per_point.npz").exists())


def score_epoch(epoch_dir: Path, gt: tuple, k_override: float | None,
                n_bins: int | None, collapse_invalid_to_failure: bool = False) -> dict:
    gt_starts, gt_p, gt_s, gt_t = gt
    with np.load(epoch_dir / "full_roa_per_point.npz") as z:
        states = z["start_states"]
        p_hat = z["p_success"].astype(np.float64)
        p_invalid_raw = z["p_invalid"].astype(np.float64)
    p_invalid = np.zeros_like(p_invalid_raw) if collapse_invalid_to_failure else p_invalid_raw
    idx = match_to_truth(states, gt_starts)
    k = k_override
    art = epoch_dir / "artifacts_v2.json"
    meta = {}
    if art.exists():
        try:
            meta = json.loads(art.read_text())
        except (ValueError, OSError):
            # A run copied from another user's scratch can leave artifacts_v2.json
            # unreadable while full_roa_per_point.npz is not (seen on the q3dppo
            # runs, 2026-09-07). Only K comes from here; fall back as if missing.
            meta = {}
    if k is None:
        try:
            k = float(meta["eval_metrics"]["num_mc_samples"])
        except (KeyError, ValueError, TypeError):
            k = None
    out = all_metrics(p_hat, gt_p[idx], gt_s[idx], gt_t[idx], k, n_bins)
    out["mean_p_invalid"] = float(p_invalid.mean())
    out["mean_p_unresolved_raw"] = float(p_invalid_raw.mean())
    out["collapse_invalid_to_failure"] = bool(collapse_invalid_to_failure)
    # Training-set size the epoch's model was fit on. Recorded per row because it
    # is the honest x axis for a budget comparison: epoch index is only a proxy,
    # and it stops being a fair one the moment two campaigns use different
    # samples_per_epoch (quad2D 500 vs quad3D 5000).
    tt = meta.get("train_trajectories")
    out["train_trajectories"] = int(tt) if tt is not None else ""
    return out


# --------------------------------------------------------------------------
# selftest
# --------------------------------------------------------------------------
def selftest() -> None:
    rng = np.random.default_rng(0)

    # soft AUROC against the literal O(n^2) definition
    n = 150
    ph = rng.integers(0, 8, n) / 7.0
    pt = rng.uniform(size=n)
    num = 0.0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            ind = 1.0 if ph[i] > ph[j] else (0.5 if ph[i] == ph[j] else 0.0)
            num += pt[i] * (1 - pt[j]) * ind
    den = pt.sum() * (1 - pt).sum() - np.sum(pt * (1 - pt))
    assert abs(soft_auroc(ph, pt) - num / den) < 1e-10, "soft_auroc mismatch"

    # exact-level decomposition reproduces the Brier score
    d = decomposition(ph, pt, k=7, m=90)
    assert abs(d["decomp_gap"]) < 1e-12, f"decomposition identity broken: {d['decomp_gap']}"

    # a continuous predictor must NOT be grouped on exact values: that gives one
    # bin per point, collapsing REL onto the Brier score and RES onto UNC
    cont = rng.uniform(size=5000)
    truth = np.clip(cont + rng.normal(0, 0.1, 5000), 0, 1)
    dc = decomposition(cont, truth, k=None, m=90)
    brier_c = float(np.mean((cont - truth) ** 2))
    assert dc["n_bins_used"] <= _DEFAULT_BINS, f"continuous p_hat not binned: {dc['n_bins_used']}"
    assert abs(dc["REL"] - brier_c) > 1e-6, "REL collapsed onto the Brier score"
    assert abs(dc["RES"] - dc["UNC"]) > 1e-6, "RES collapsed onto UNC"

    # debiased identity: MSE_hat == REL_hat - RES + UNC_hat
    lhs = debiased_brier(ph, pt, 7, 90)
    rhs = d["REL_debiased"] - d["RES"] + d["UNC_debiased"]
    assert abs(lhs - rhs) < 1e-12, f"debiased decomposition broken: {lhs} vs {rhs}"

    # the debiasing actually removes binomial noise: a perfect model scores ~0
    n, K, M = 20000, 100, 90
    p_true = rng.uniform(0.05, 0.95, n)
    p_hat = rng.binomial(K, p_true) / K
    p_obs = rng.binomial(M, p_true) / M
    naive = float(np.mean((p_hat - p_obs) ** 2))
    deb = debiased_brier(p_hat, p_obs, K, M)
    assert naive > 0.003, f"expected visible naive noise, got {naive}"
    assert abs(deb) < 5e-4, f"debiased Brier should be ~0 for a perfect model, got {deb}"

    # skill score of a perfect model is ~1, of climatology ~0
    ss = skill_score(p_hat, p_obs, K, M)
    assert ss > 0.99, f"perfect-model skill score too low: {ss}"
    clim = np.full(n, p_obs.mean())
    assert abs(skill_score(clim, p_obs, None, M)) < 0.02, "climatology skill should be ~0"

    # risk-coverage: full coverage equals the debiased Brier
    rc = risk_coverage(p_hat, p_obs, K, M)
    assert abs(rc["risk@1"] - deb) < 1e-12, "risk@1 must equal the debiased Brier"

    # paired delta is exactly the difference of the two debiased Brier scores,
    # and an arm compared with itself is exactly zero
    p_other = np.clip(p_hat + rng.normal(0, 0.05, n), 0, 1)
    pd = paired_delta(p_other, p_hat, p_obs, K, K, M)
    expect = debiased_brier(p_other, p_obs, K, M) - debiased_brier(p_hat, p_obs, K, M)
    assert abs(pd["delta"] - expect) < 1e-12, f"paired delta mismatch: {pd['delta']} vs {expect}"
    self_cmp = paired_delta(p_hat, p_hat, p_obs, K, K, M)
    assert abs(self_cmp["delta"]) < 1e-15 and self_cmp["se"] < 1e-15, "self-comparison must be 0"

    # sharpness: both estimators target the same quantity, mean p_true(1-p_true).
    # SHARP reads it off the model's K samples, SHARP* off the grid's M rollouts,
    # so if either factor were wrong they would disagree with the truth.
    target = float(np.mean(p_true * (1 - p_true)))
    sh = sharpness(p_hat, p_obs, K, M)
    assert abs(sh["SHARP"] - target) < 5e-4, f"SHARP biased: {sh['SHARP']} vs {target}"
    assert abs(sh["SHARP_star"] - target) < 5e-4, f"SHARP* biased: {sh['SHARP_star']} vs {target}"
    naive_sharp = float(np.mean(p_hat * (1 - p_hat)))
    assert naive_sharp < target - 1e-4, "undebiased sharpness should underestimate"

    # A deterministic arm can have exactly one rollout per state. Its empirical
    # probabilities are exact 0/1 outcomes, so the ground-truth sharpness is
    # finite (zero) rather than an m/(m-1) division by zero.
    sh_det = sharpness(np.array([0.1, 0.9]), np.array([0.0, 1.0]), None, 1.0)
    assert sh_det["SHARP_star"] == 0.0, "M=1 deterministic sharpness must be finite"

    # thresholded RoA against sklearn
    try:
        from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
        beta = 0.5
        tr = thresholded_roa(p_hat, p_obs, beta)
        y, yh = (p_obs >= beta).astype(int), (p_hat >= beta).astype(int)
        assert abs(tr[f"f1@{beta:g}"] - f1_score(y, yh)) < 1e-12, "F1 mismatch vs sklearn"
        assert abs(tr[f"prec@{beta:g}"] - precision_score(y, yh)) < 1e-12, "precision mismatch"
        assert abs(tr[f"rec@{beta:g}"] - recall_score(y, yh)) < 1e-12, "recall mismatch"
        assert abs(tr[f"acc@{beta:g}"] - accuracy_score(y, yh)) < 1e-12, "accuracy mismatch"
        sk = "vs sklearn OK"
    except ImportError:
        sk = "sklearn absent, skipped"

    print("selftest OK  "
          f"(naive brier {naive:.5f} -> debiased {deb:+.6f}, SS {ss:.4f}, sAUROC verified, "
          f"sharpness unbiased, thresholded RoA {sk})")


# --------------------------------------------------------------------------
# cli
# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", type=Path, help="JSON list of {predictor, level, arm, run_dir}")
    ap.add_argument("--data-root", type=Path,
                    default=Path("/common/users/shared/pracsys/genMoPlan/data_trajectories/noisy/pendulum/lqr"))
    ap.add_argument("--out", type=Path, default=Path("docs/stoch_compare"))
    ap.add_argument("--epochs", default="all", help="'all', 'last', or a comma list of ints")
    ap.add_argument("--n-bins", type=int, default=None,
                    help="quantile bins for the decomposition (default: exact p_hat levels)")
    ap.add_argument("--collapse-invalid-to-failure", action="store_true",
                    help="binary outcome semantics: report unresolved endpoint mass as failure")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    if args.spec is None:
        ap.error("--spec is required unless --selftest")

    arms = [Arm(**a) if not isinstance(a, Arm) else a
            for a in (dict(x, run_dir=Path(x["run_dir"])) for x in json.loads(args.spec.read_text()))]
    gt_cache: dict[str, tuple] = {}
    rows = []
    for a in arms:
        if a.level not in gt_cache:
            gt_cache[a.level] = load_ground_truth(args.data_root / a.level)
        eds = epoch_dirs(a.run_dir)
        if args.epochs == "last":
            eds = eds[-1:]
        elif args.epochs != "all":
            keep = {int(e) for e in args.epochs.split(",")}
            eds = [d for d in eds if int(d.name.split("_")[1]) in keep]
        for ed in eds:
            try:
                row = score_epoch(
                    ed, gt_cache[a.level], None, args.n_bins,
                    collapse_invalid_to_failure=args.collapse_invalid_to_failure,
                )
            except Exception as exc:  # a half-written epoch must not kill the sweep
                print(f"  !! {a.predictor}/{a.level}/{a.arm}/{ed.name}: {exc}", flush=True)
                continue
            row.update(predictor=a.predictor, level=a.level, arm=a.arm,
                       epoch=int(ed.name.split("_")[1]), run_dir=str(a.run_dir))
            rows.append(row)
            print(f"  {a.predictor:3s} {a.level:5s} {a.arm:22s} {ed.name} "
                  f"brier_deb={row['brier_debiased']:+.5f} SS={row['skill_score']:.4f} "
                  f"sAUROC={row['sAUROC']:.4f}", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "metrics.json").write_text(json.dumps(rows, indent=2))
    if rows:
        keys = sorted({k for r in rows for k in r})
        lead = ["predictor", "level", "arm", "epoch"]
        keys = lead + [k for k in keys if k not in lead]
        with (args.out / "metrics.csv").open("w") as fh:
            fh.write(",".join(keys) + "\n")
            for r in rows:
                fh.write(",".join(str(r.get(k, "")) for k in keys) + "\n")
    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
