"""Yield-aware acquisition whose length model is LEARNED from the start state.

Why this exists (pendulum i100 sweep, 2026-08-16):

``yield_aware`` ranks by ``score(x) * E[L(x)]**alpha`` and beats uniform sampling
on KL across three seeds. But its length model is a two-point mixture routed
through the ensemble's own p_bar:

    E[L(x)] = p_bar(x) * L_success + (1 - p_bar(x)) * L_failure

and p_bar is exactly the quantity that is miscalibrated early. Measured: at
epoch 0 the arm expected 82,002 pairs and bought 11,374 (7.2x over), because
p_bar assigned ~0.21 to states whose true success probability was ~0.99. The
weight stays effectively inert until epoch 6-8 in every seed -- roughly a third
of the campaign spent with the mechanism switched off.

The lengths themselves were never the problem. Measured on the pool:

    L_failure = 1001.0 with sd 0.0   (every failure hits the timeout cap)
    L_success = 136.5   with sd 76.5
    98.7% of Var(L) is explained by the OUTCOME LABEL alone

So predicting length IS predicting the outcome, and the two-point mixture is
already near-optimal *given a good p*. The bottleneck is the p.

This strategy replaces p_bar with a distance-weighted kNN regression fitted on
the TRUE lengths of trajectories already acquired -- the same estimator family
``class_quota`` uses for its gate, which reaches soft-AUROC 0.956 against
continuous ground truth from the initial 100 trajectories alone. Measured
head-to-head on held-out pool data (R^2 for predicting L from the start state):

    n=100 acquired   direct kNN length 0.687   vs   kNN-outcome two-point 0.674
    n=600            0.800                     vs   0.790
    n=2000           0.808                     vs   0.801
    oracle label two-point                     0.987

Direct regression and the kNN two-point form correlate at 0.997, so the choice
between them is immaterial; what matters is that BOTH are far better calibrated
than the ensemble's p_bar early on. The direct form is used here because it is
one estimator instead of two and needs no L_success/L_failure bookkeeping.

alpha=0 still reduces EXACTLY to plain epi_var, so this remains a single-knob
change relative to the unweighted arm, and a single-estimator change relative to
``yield_aware``.

A NEW module rather than a flag on ``yield_aware`` on purpose: sibling runs
re-import strategy modules from disk every adaptive epoch, so editing a module
they use could change their behaviour mid-run.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from adaptive_roa.adaptive_v2.strategy.class_quota import embed_states
from adaptive_roa.adaptive_v2.strategy.uncertainty_scores import (
    SCORE_MODES, aleatoric_uncertainty, epistemic_bald, score_by_mode,
)
from adaptive_roa.adaptive_v2.strategy.yield_aware import expected_length
from adaptive_roa.adaptive_v2.types import AcquisitionResult, ThresholdState


def knn_length(X_labeled: np.ndarray, L_labeled: np.ndarray,
               X_query: np.ndarray, k: int) -> np.ndarray:
    """Distance-weighted kNN regression of trajectory length from start state.

    Identical estimator to ``class_quota.knn_success_prob`` with lengths in
    place of labels, so the two arms share their notion of locality.
    """
    k = int(min(k, len(X_labeled)))
    d, idx = cKDTree(X_labeled).query(X_query, k=k)
    if k == 1:
        d, idx = d[:, None], idx[:, None]
    w = 1.0 / np.maximum(d, 1e-6)
    return (w * L_labeled[idx]).sum(axis=1) / w.sum(axis=1)


class YieldKNNAcquisitionStrategy:
    """yield_aware with a learned length model instead of a p_bar mixture."""

    mode = "decomposition"

    def __init__(self, cfg: Any):
        self.score_mode = str(cfg.score)
        if self.score_mode not in SCORE_MODES:
            raise ValueError(
                f"unknown score mode {self.score_mode!r}; expected one of {SCORE_MODES}")
        self.d2_ratio = float(cfg.d2_ratio)
        self.n_candidates = int(cfg.n_candidates)
        self.alpha = float(cfg.get("alpha", 1.0))
        self.knn_k = int(cfg.get("knn_k", 15))
        # Below this many acquired trajectories the kNN has too little support;
        # fall back to the p_bar two-point mixture (i.e. to yield_aware).
        self.min_labeled = int(cfg.get("min_labeled", 20))
        self.prior_len_success = float(cfg.get("prior_len_success", 150.0))
        self.prior_len_failure = float(cfg.get("prior_len_failure", 1000.0))
        self.min_per_class = int(cfg.get("min_per_class", 5))
        self.verbose = bool(cfg.verbose)

    # ------------------------------------------------------- acquired truth
    @staticmethod
    def _acquired(pool: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(start_states, labels in {0,1}, lengths) for everything already bought.

        Labels are normalised HERE: ``get_label`` returns the internal {-1, +1}
        convention, and testing ``y == 0`` for failure silently disabled the
        class_quota gate on its first live run.
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
                "YieldKNNAcquisitionStrategy needs a backend exposing "
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
        ls, lf = self._fallback_lengths(y_lab, L_lab) if len(y_lab) else (
            self.prior_len_success, self.prior_len_failure)
        len_pbar = expected_length(p_bar, ls, lf)          # what yield_aware would use

        knn_on = len(X_lab) >= self.min_labeled
        if knn_on:
            scales, mask = self._geometry(getattr(pool, "system", None), states_arr.shape[1])
            E_lab = embed_states(X_lab, scales, mask)
            E_cand = embed_states(states_arr, scales, mask)
            len_hat = knn_length(E_lab, L_lab, E_cand, self.knn_k)
        else:
            len_hat = len_pbar

        weight = np.maximum(len_hat, 1.0) ** self.alpha
        score = np.sign(base) * np.abs(base) * weight

        key = np.where(np.isfinite(score), score, -np.inf)
        positions = np.argsort(-key, kind="stable")[:target_count]
        selected = [indices[int(p)] for p in positions]

        diagnostics = self._diagnostics(
            p_members, base, score, p_bar, positions, n_actual, k,
            len_hat, len_pbar, ls, lf, y_lab, knn_on)
        if self.verbose:
            print(f"    [YieldKNN/{self.score_mode}] {n_actual} cand, alpha={self.alpha}, "
                  f"knn={'on' if knn_on else 'OFF'} (n_lab={len(X_lab)}); selected {len(selected)}; "
                  f"L_hat mean sel={diagnostics['len_hat_mean_selected']:.0f} "
                  f"(p_bar model would say {diagnostics['len_pbar_mean_selected']:.0f}); "
                  f"E[pairs]={diagnostics['expected_pairs_selected']:.0f}")

        return AcquisitionResult(
            d1_indices=[], d2_indices=selected,
            n_candidates_evaluated=n_actual,
            n_certain_discarded=n_actual - len(selected),
            n_invalid_added=0, diagnostics=diagnostics,
        )

    def _diagnostics(self, p_members, base, score, p_bar, positions, n_actual, k,
                     len_hat, len_pbar, ls, lf, y_lab, knn_on) -> dict[str, Any]:
        epi = epistemic_bald(p_members)
        ale = aleatoric_uncertainty(p_members)
        sel = positions
        return {
            "score_mode": self.score_mode,
            "alpha": self.alpha,
            "knn_active": bool(knn_on),
            "knn_k": self.knn_k,
            "n_labeled": int(len(y_lab)),
            "n_members": int(np.asarray(p_members).shape[0]),
            "member_sample_size": k,
            "score_mean_selected": float(np.nanmean(score[sel])) if len(sel) else None,
            "base_score_mean_selected": float(np.nanmean(base[sel])) if len(sel) else None,
            "epistemic_mean_selected": float(np.nanmean(epi[sel])) if len(sel) else None,
            "aleatoric_mean_selected": float(np.nanmean(ale[sel])) if len(sel) else None,
            "p_bar_mean_selected": float(np.nanmean(p_bar[sel])) if len(sel) else None,
            # the head-to-head that motivates this arm: learned vs p_bar length
            "len_hat_mean_selected": float(np.nanmean(len_hat[sel])) if len(sel) else None,
            "len_pbar_mean_selected": float(np.nanmean(len_pbar[sel])) if len(sel) else None,
            "len_hat_over_pbar": (float(np.nanmean(len_hat[sel]) / np.nanmean(len_pbar[sel]))
                                  if len(sel) and np.nanmean(len_pbar[sel]) else None),
            "len_success_est": float(ls),
            "len_failure_est": float(lf),
            "expected_pairs_selected": float(np.nansum(len_hat[sel])) if len(sel) else 0.0,
            "n_candidates_evaluated": int(n_actual),
            "selection_rule": f"knn_length_weighted_alpha{self.alpha:g}",
        }

    @staticmethod
    def _skip(reason: str) -> AcquisitionResult:
        return AcquisitionResult(
            d1_indices=[], d2_indices=[], n_candidates_evaluated=0,
            n_certain_discarded=0, n_invalid_added=0,
            diagnostics={"skipped_reason": reason},
        )
