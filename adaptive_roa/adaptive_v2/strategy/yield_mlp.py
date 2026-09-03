"""Yield-aware acquisition with an MLP length model refit every epoch.

Why this exists (pendulum i100 sweep, 2026-08-17):

``yield_aware`` prices candidates by ``score(x) * E[L(x)]**alpha`` where E[L] is
a two-point mixture routed through the ensemble's p_bar. p_bar is the badly
miscalibrated quantity -- at epoch 0 it produced a 7.2x yield overestimate --
so the weight is effectively inert until epoch 6-8 in every seed.

``yield_knn`` replaced that with a distance-weighted kNN regression on the true
lengths of acquired trajectories, which never entered the inert regime. This arm
replaces the kNN with a small MLP, refit from scratch each epoch.

Measured head-to-head, R^2 for predicting length from the start state on 20,000
held-out pool trajectories (3 restarts, mean +- sd):

    n_train   kNN k=15        MLP 64-64       MLP 128-128-64   RF-200
      100     0.681+-0.011    0.757+-0.001    0.736+-0.012     0.501+-0.034
      300     0.773+-0.003    0.797+-0.008    0.776+-0.014     0.703+-0.016
      600     0.799+-0.001    0.809+-0.005    0.807+-0.005     0.764+-0.001
     1200     0.809+-0.004    0.817+-0.008    0.818+-0.004     0.786+-0.006
     2000     0.814+-0.004    0.820+-0.007    0.823+-0.002     0.792+-0.003

Three things that decided the design:

1. The MLP's advantage is LARGEST at n=100 (+0.076 R^2, an 11% relative gain)
   and shrinks to +0.006 by n=2000. That is exactly the right shape: the yield
   weight is broken early and fine late, so the estimator that helps early is
   the one worth having.
2. 64-64 beats 128-128-64 at n=100 (0.757 vs 0.736) and only ties it later, so
   the smaller net is used -- capacity hurts at the sample sizes that matter.
3. The MLP is also far more STABLE at small n (sd 0.001 vs the kNN's 0.011),
   which matters because a single bad early batch propagates through the whole
   run.

Random forest was tested and is clearly worse than both (0.501 at n=100); it is
not offered as an option.

Ceiling for reference: a two-point mixture with the ORACLE label reaches
R^2 0.987, because 98.7% of Var(L) on this pool is explained by the outcome
alone. Every learned estimator here is solving the outcome-prediction problem
in disguise, and none is close to that ceiling.

The kNN prediction is computed alongside the MLP every epoch and logged, so the
two estimators can be compared WITHIN this run rather than across runs.

alpha=0 still reduces EXACTLY to plain epi_var.

A NEW module rather than a flag on ``yield_knn`` on purpose: sibling runs
re-import strategy modules from disk every adaptive epoch.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from adaptive_roa.adaptive_v2.strategy.class_quota import embed_states
from adaptive_roa.adaptive_v2.strategy.uncertainty_scores import (
    SCORE_MODES, aleatoric_uncertainty, epistemic_bald, score_by_mode,
)
from adaptive_roa.adaptive_v2.strategy.yield_aware import expected_length
from adaptive_roa.adaptive_v2.strategy.yield_knn import knn_length
from adaptive_roa.adaptive_v2.types import AcquisitionResult, ThresholdState


def fit_mlp_length(X: np.ndarray, L: np.ndarray, hidden: tuple[int, ...],
                   seed: int, max_iter: int = 3000):
    """Fit an MLP to predict trajectory length; returns a predict callable.

    The target is standardised because raw lengths span 2..1001 and an
    unstandardised target makes the net spend its capacity on the scale rather
    than the shape. Early stopping is on by default -- with 100 training points
    an unregularised 64-64 net memorises instantly.

    Returns ``None`` if the fit fails for any reason; the caller falls back.
    """
    try:
        from sklearn.neural_network import MLPRegressor
    except Exception:
        return None
    if len(X) < 4:
        return None
    mu, sg = float(np.mean(L)), float(np.std(L)) + 1e-9
    try:
        m = MLPRegressor(hidden_layer_sizes=tuple(hidden), max_iter=max_iter,
                         random_state=int(seed), early_stopping=True,
                         n_iter_no_change=40, learning_rate_init=3e-3)
        m.fit(X, (L - mu) / sg)
    except Exception:
        return None
    return lambda Q: m.predict(Q) * sg + mu


def choose_length_model(E: np.ndarray, L: np.ndarray, hidden: tuple[int, ...],
                        knn_k: int, seed: int,
                        holdout: float = 0.25) -> tuple[str, float, float]:
    """Pick MLP or kNN by held-out MSE on the ALREADY-ACQUIRED data.

    Why this exists rather than always using the MLP: the pool-level benchmark
    said MLP > kNN at every sample size, but that was measured with hundreds of
    training points drawn uniformly. At the smallest sizes this arm actually
    sees -- and with sklearn's ``early_stopping`` carving another 10% off an
    already tiny set -- the MLP can fit badly enough to INVERT the ordering. A
    unit test caught exactly that: on 40 points forming two clean clusters at
    lengths 130 and 1001, the MLP predicted 430 and 224 (wrong order) while the
    kNN returned 130 and 1001 exactly.

    Trusting the benchmark blindly would therefore have shipped a strategy that
    is worse than ``yield_knn`` at precisely the depths the arm exists to fix.
    Selecting per-epoch on measured error costs a fraction of a second at
    n <= 2000 and bounds the arm below by the kNN.

    Returns ``(winner, mse_mlp, mse_knn)``; MSEs are NaN when not evaluable.
    """
    n = len(E)
    n_val = int(round(holdout * n))
    if n_val < 3 or n - n_val < 4:
        return "knn", float("nan"), float("nan")
    rs = np.random.default_rng(int(seed))
    perm = rs.permutation(n)
    va, tr = perm[:n_val], perm[n_val:]

    knn_pred = knn_length(E[tr], L[tr], E[va], knn_k)
    mse_knn = float(np.mean((knn_pred - L[va]) ** 2))

    predict = fit_mlp_length(E[tr], L[tr], hidden, seed)
    if predict is None:
        return "knn", float("nan"), mse_knn
    mlp_pred = np.asarray(predict(E[va]), dtype=np.float64)
    if not np.isfinite(mlp_pred).all():
        return "knn", float("nan"), mse_knn
    mse_mlp = float(np.mean((mlp_pred - L[va]) ** 2))
    return ("mlp" if mse_mlp < mse_knn else "knn"), mse_mlp, mse_knn


class YieldMLPAcquisitionStrategy:
    """yield_aware whose length model is an MLP refit each epoch."""

    mode = "decomposition"

    def __init__(self, cfg: Any):
        self.score_mode = str(cfg.score)
        if self.score_mode not in SCORE_MODES:
            raise ValueError(
                f"unknown score mode {self.score_mode!r}; expected one of {SCORE_MODES}")
        self.d2_ratio = float(cfg.d2_ratio)
        self.n_candidates = int(cfg.n_candidates)
        self.alpha = float(cfg.get("alpha", 1.0))
        self.hidden = tuple(cfg.get("hidden", (64, 64)))
        self.knn_k = int(cfg.get("knn_k", 15))
        self.min_labeled = int(cfg.get("min_labeled", 20))
        self.mlp_seed = int(cfg.get("mlp_seed", 0))
        self.prior_len_success = float(cfg.get("prior_len_success", 150.0))
        self.prior_len_failure = float(cfg.get("prior_len_failure", 1000.0))
        self.min_per_class = int(cfg.get("min_per_class", 5))
        self.verbose = bool(cfg.verbose)
        self._epoch = 0

    # ------------------------------------------------------- acquired truth
    @staticmethod
    def _acquired(pool: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(start_states, labels in {0,1}, lengths) for everything already bought.

        Labels normalised HERE: ``get_label`` returns {-1, +1} and a ``y == 0``
        failure test silently disabled the class_quota gate on its first run.
        """
        builder = getattr(pool, "dataset_builder", None)
        source = getattr(builder, "data_source", None)
        split = getattr(builder, "train_split", None)
        if source is None or split is None:
            return np.empty((0, 0)), np.empty(0), np.empty(0)
        X, y, L = [], [], []
        for idx in list(getattr(split, "indices", []) or []):
            try:
                X.append(np.asarray(source.get_start_state(int(idx)), dtype=np.float64))
                y.append(1.0 if int(source.get_label(int(idx))) == 1 else 0.0)
                L.append(float(source.get_trajectory_length(int(idx))))
            except Exception:
                continue
        if not X:
            return np.empty((0, 0)), np.empty(0), np.empty(0)
        return np.stack(X), np.asarray(y), np.asarray(L)

    @staticmethod
    def _geometry(system: Any, state_dim: int) -> tuple[np.ndarray, np.ndarray]:
        scales, mask = np.ones(state_dim), np.zeros(state_dim, dtype=bool)
        if system is not None:
            s = system.get_normalization_scales()
            scales = s.cpu().numpy().astype(np.float64) if hasattr(s, "cpu") else np.asarray(s, dtype=np.float64)
            idx = system.get_circular_indices()
            if idx:
                mask[np.asarray(idx, dtype=int)] = True
        return scales, mask

    def _fallback_lengths(self, y: np.ndarray, L: np.ndarray) -> tuple[float, float]:
        ls = float(L[y == 1].mean()) if int((y == 1).sum()) >= self.min_per_class else self.prior_len_success
        lf = float(L[y == 0].mean()) if int((y == 0).sum()) >= self.min_per_class else self.prior_len_failure
        return ls, lf

    # ----------------------------------------------------------------- select
    def select(
        self,
        pool: Any,
        probability_backend: Any,
        threshold_backend: Any,
        threshold_state: ThresholdState,
        target_count: int,
        exclude: set[int] | None = None,
    ) -> AcquisitionResult:
        if target_count <= 0:
            return self._skip("target_count_zero")
        if not hasattr(probability_backend, "estimate_members"):
            raise RuntimeError(
                "YieldMLPAcquisitionStrategy needs a backend exposing "
                f"estimate_members(); got {type(probability_backend).__name__}.")

        states, indices = pool.sample_candidates_without_marking(
            self.n_candidates, exclude=exclude)
        n_actual = len(indices)
        if n_actual == 0:
            return self._skip("pool_exhausted")

        p_members = probability_backend.estimate_members(states, verbose=self.verbose)
        k = getattr(probability_backend, "member_sample_size", None)
        base = np.asarray(score_by_mode(self.score_mode, p_members, k), dtype=np.float64)
        p_bar = np.asarray(p_members, dtype=np.float64).mean(axis=0)

        X_lab, y_lab, L_lab = self._acquired(pool)
        states_arr = np.asarray(states, dtype=np.float64)
        ls, lf = (self._fallback_lengths(y_lab, L_lab) if len(y_lab)
                  else (self.prior_len_success, self.prior_len_failure))
        len_pbar = expected_length(p_bar, ls, lf)

        len_knn = None
        len_mlp = None
        winner, mse_mlp, mse_knn = "pbar", float("nan"), float("nan")
        if len(X_lab) >= self.min_labeled:
            scales, mask = self._geometry(getattr(pool, "system", None), states_arr.shape[1])
            E_lab = embed_states(X_lab, scales, mask)
            E_cand = embed_states(states_arr, scales, mask)
            len_knn = knn_length(E_lab, L_lab, E_cand, self.knn_k)
            seed = self.mlp_seed + self._epoch
            # Choose on measured held-out error, not on the pool-level benchmark:
            # the MLP can invert the ordering at the smallest sample sizes.
            winner, mse_mlp, mse_knn = choose_length_model(
                E_lab, L_lab, self.hidden, self.knn_k, seed)
            if winner == "mlp":
                predict = fit_mlp_length(E_lab, L_lab, self.hidden, seed)
                if predict is not None:
                    cand = np.asarray(predict(E_cand), dtype=np.float64)
                    if np.isfinite(cand).all():
                        len_mlp = cand

        if len_mlp is not None:
            len_hat, source_used = len_mlp, "mlp"
        elif len_knn is not None:
            len_hat, source_used = len_knn, "knn"
        else:
            len_hat, source_used = len_pbar, "pbar_fallback"

        weight = np.maximum(len_hat, 1.0) ** self.alpha
        score = np.sign(base) * np.abs(base) * weight

        key = np.where(np.isfinite(score), score, -np.inf)
        positions = np.argsort(-key, kind="stable")[:target_count]
        selected = [indices[int(p)] for p in positions]

        diagnostics = self._diagnostics(
            p_members, base, score, p_bar, positions, n_actual, k,
            len_hat, len_knn, len_pbar, ls, lf, y_lab, source_used,
            winner, mse_mlp, mse_knn)
        if self.verbose:
            kn = diagnostics["len_knn_mean_selected"]
            print(f"    [YieldMLP/{self.score_mode}] {n_actual} cand, alpha={self.alpha}, "
                  f"src={source_used} (n_lab={len(X_lab)}, hidden={self.hidden}); "
                  f"selected {len(selected)}; L_mlp={diagnostics['len_hat_mean_selected']:.0f} "
                  f"L_knn={kn if kn is None else round(kn)} "
                  f"L_pbar={diagnostics['len_pbar_mean_selected']:.0f}")
        self._epoch += 1

        return AcquisitionResult(
            d1_indices=[], d2_indices=selected,
            n_candidates_evaluated=n_actual,
            n_certain_discarded=n_actual - len(selected),
            n_invalid_added=0, diagnostics=diagnostics,
        )

    def _diagnostics(self, p_members, base, score, p_bar, positions, n_actual, k,
                     len_hat, len_knn, len_pbar, ls, lf, y_lab, source_used,
                     winner, mse_mlp, mse_knn) -> dict[str, Any]:
        epi = epistemic_bald(p_members)
        ale = aleatoric_uncertainty(p_members)
        sel = positions
        m = lambda a: (float(np.nanmean(a[sel])) if a is not None and len(sel) else None)
        return {
            "score_mode": self.score_mode,
            "alpha": self.alpha,
            "length_model": source_used,
            # the per-epoch bake-off that picked it, so the choice is auditable
            "length_model_winner": winner,
            "holdout_mse_mlp": None if mse_mlp != mse_mlp else mse_mlp,
            "holdout_mse_knn": None if mse_knn != mse_knn else mse_knn,
            "mlp_hidden": list(self.hidden),
            "n_labeled": int(len(y_lab)),
            "n_members": int(np.asarray(p_members).shape[0]),
            "member_sample_size": k,
            "score_mean_selected": m(score),
            "base_score_mean_selected": m(base),
            "epistemic_mean_selected": m(epi),
            "aleatoric_mean_selected": m(ale),
            "p_bar_mean_selected": m(p_bar),
            # three length models on the SAME selected set -> in-run comparison
            "len_hat_mean_selected": m(len_hat),
            "len_knn_mean_selected": m(len_knn),
            "len_pbar_mean_selected": m(len_pbar),
            "len_success_est": float(ls),
            "len_failure_est": float(lf),
            "expected_pairs_selected": float(np.nansum(len_hat[sel])) if len(sel) else 0.0,
            "n_candidates_evaluated": int(n_actual),
            "selection_rule": f"mlp_length_weighted_alpha{self.alpha:g}",
        }

    @staticmethod
    def _skip(reason: str) -> AcquisitionResult:
        return AcquisitionResult(
            d1_indices=[], d2_indices=[], n_candidates_evaluated=0,
            n_certain_discarded=0, n_invalid_added=0,
            diagnostics={"skipped_reason": reason},
        )
