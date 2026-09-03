"""Class-quota epistemic acquisition: buy composition by MEASURED outcome, not p_bar.

Why this exists (pendulum i100 sweep, 2026-08-16):

The yield-aware arm (`yield_aware.py`) priced candidates by
``epi_var * E[L]`` with ``E[L]`` routed through the ensemble's own p_bar.
Epoch 0 falsified it: it expected 82k pairs and bought 11.4k -- fewer than
plain epi_var -- because p_bar called 99%-true-success states likely-failures
(p_bar 0.21 vs true p 0.99), i.e. the yield model inherited exactly the
miscalibration it was built to work around. The completed 20-epoch epi_var run
shows this never self-heals: mean p_bar on acquired batches climbs 0.16 -> ~0.6
and plateaus while true p stays 0.90-0.99; the gap asymptotes at ~0.3 after
2000 trajectories.

What DOES estimate outcome correctly is already in hand: the true labels of
every trajectory acquired so far. A distance-weighted kNN fit on acquired
(start_state -> label) reaches soft-AUROC 0.956 against continuous ground
truth with only the initial 100 uniform trajectories, and its
predicted-failure precision is 0.93-0.98 from epoch 0 onward (measured by
replaying the completed epi_var run's acquisitions).

So instead of multiplying the score by a fragile expected-length weight, this
strategy makes the batch COMPOSITION explicit and guaranteed:

    1. score all candidates with the unchanged uncertainty score (epi_var);
    2. gate candidates into predicted-failure / predicted-success pots with the
       kNN fit on acquired ground truth (p_bar appears nowhere in this path);
    3. spend ``quota_fail`` of the trajectory budget on the top-scored
       candidates of the failure pot and the rest on the top-scored candidates
       of the success pot, each pot selected with the same greedy_diverse rule
       the plain arm uses.

``quota_fail="pool"`` tracks the measured class marginal of everything
acquired so far (~0.60 failures on pendulum-high), which makes the arm's pair
accrual match the uniform control's by construction (simulated worst case --
epi_var ranking adding nothing within a pot -- buys 68-75k pairs/epoch vs the
control's ~65k) while every slot is still an epistemic pick within its class.
Bounded downside: if the score is useless inside a pot the batch degrades to a
control-composition random batch, not to a starved one.

The single knob vs the plain epi_var arm is the quota. With the gate disabled
(a class not yet seen ``min_labeled_per_class`` times) the strategy runs the
plain arm's selection unchanged.

A NEW module on purpose: sibling runs are in flight and mp.spawn children
re-import strategy modules from disk every adaptive epoch, so editing
decomposition.py / yield_aware.py would change their behaviour mid-run.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from adaptive_roa.adaptive_v2.strategy.dispersion_score import (
    select_greedy, select_greedy_diverse,
)
from adaptive_roa.adaptive_v2.strategy.uncertainty_scores import (
    SCORE_MODES, aleatoric_uncertainty, epistemic_bald, score_by_mode,
)
from adaptive_roa.adaptive_v2.types import AcquisitionResult, ThresholdState

_SELECTION_RULES = ("greedy", "greedy_diverse")


def embed_states(states: np.ndarray, scales: np.ndarray,
                 circular_mask: np.ndarray) -> np.ndarray:
    """Map states into the metric the kNN measures distance in.

    Circular dims become (cos, sin) so theta=+pi and theta=-pi coincide;
    Euclidean dims are divided by the system's normalization scale so one
    radian and one rad/s are commensurate.
    """
    states = np.asarray(states, dtype=np.float64)
    cols = []
    for j in range(states.shape[1]):
        if circular_mask[j]:
            cols.append(np.cos(states[:, j]))
            cols.append(np.sin(states[:, j]))
        else:
            cols.append(states[:, j] / float(scales[j]))
    return np.column_stack(cols)


def knn_success_prob(X_labeled: np.ndarray, y_labeled: np.ndarray,
                     X_query: np.ndarray, k: int) -> np.ndarray:
    """Distance-weighted kNN estimate of p(success|x) from acquired truth."""
    k = int(min(k, len(X_labeled)))
    d, idx = cKDTree(X_labeled).query(X_query, k=k)
    if k == 1:
        d, idx = d[:, None], idx[:, None]
    w = 1.0 / np.maximum(d, 1e-6)
    return (w * y_labeled[idx]).sum(axis=1) / w.sum(axis=1)


class ClassQuotaDecompositionAcquisitionStrategy:
    """DecompositionAcquisitionStrategy with a measured-outcome class quota.

    Identical candidate sampling, identical uncertainty score, identical
    within-pot selection rule; the only difference is that the batch's
    predicted-class composition is fixed by ``quota_fail`` instead of being
    whatever the score happens to buy.
    """

    mode = "decomposition"

    def __init__(self, cfg: Any):
        self.score_mode = str(cfg.score)
        if self.score_mode not in SCORE_MODES:
            raise ValueError(
                f"unknown score mode {self.score_mode!r}; expected one of {SCORE_MODES}")
        self.d2_ratio = float(cfg.d2_ratio)
        self.n_candidates = int(cfg.n_candidates)
        self.selection_rule = str(cfg.get("selection_rule", "greedy_diverse"))
        if self.selection_rule not in _SELECTION_RULES:
            raise ValueError(
                f"selection_rule must be one of {_SELECTION_RULES}, got {self.selection_rule!r}")
        self.diversity_pool_multiplier = int(cfg.get("diversity_pool_multiplier", 5))
        # "pool" -> failure fraction measured from acquired labels; else a float
        self.quota_fail = cfg.get("quota_fail", "pool")
        if self.quota_fail != "pool":
            q = float(self.quota_fail)
            if not 0.0 <= q <= 1.0:
                raise ValueError(f"quota_fail must be in [0,1] or 'pool', got {q}")
        self.knn_k = int(cfg.get("knn_k", 15))
        self.min_labeled_per_class = int(cfg.get("min_labeled_per_class", 10))
        self.verbose = bool(cfg.verbose)

    # ------------------------------------------------------- acquired truth
    def _acquired_truth(self, pool: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(start_states, labels, lengths) of every trajectory already paid for.

        Labels are normalised to {0, 1} HERE, at the only point they enter this
        module. ``data_source.get_label`` returns the INTERNAL convention, which
        ``_load_shuffled_labels`` builds with ``label_mapping = {0: -1, 1: 1}``:
        failure is **-1**, not 0. Testing ``y == 0`` therefore matches nothing,
        which silently disabled the gate on the first live run (epoch 0 reported
        n_labeled_failure=0 out of 57 real failures, so the strategy fell back to
        plain epi_var and bought 13,273 pairs instead of the predicted ~68k).
        The same bug corrupts ``knn_success_prob``, which averages labels: on
        {-1, 1} it returns a value in [-1, 1] and the ``knn_p < 0.5`` split is
        meaningless. Normalising once here fixes both, and ``(y == 1)`` is
        correct under either convention.
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
                L.append(int(source.get_trajectory_length(int(idx))))
            except Exception:
                continue
        if not X:
            return np.empty((0, 0)), np.empty(0), np.empty(0)
        return np.stack(X), np.asarray(y, dtype=np.float64), np.asarray(L, dtype=np.float64)

    # ----------------------------------------------------------- geometry
    @staticmethod
    def _system_geometry(system: Any, state_dim: int) -> tuple[np.ndarray, np.ndarray]:
        scales = np.ones(state_dim)
        mask = np.zeros(state_dim, dtype=bool)
        if system is not None:
            s = system.get_normalization_scales()
            scales = s.cpu().numpy().astype(np.float64) if hasattr(s, "cpu") else np.asarray(s, dtype=np.float64)
            idx = system.get_circular_indices()
            if idx:
                mask[np.asarray(idx, dtype=int)] = True
        return scales, mask

    def _select_within(self, score: np.ndarray, states: np.ndarray,
                       scales: np.ndarray, mask: np.ndarray, n: int) -> np.ndarray:
        if n <= 0 or len(score) == 0:
            return np.empty(0, dtype=int)
        if self.selection_rule == "greedy":
            return select_greedy(score, n)
        return select_greedy_diverse(score, states, scales, mask, n,
                                     pool_multiplier=self.diversity_pool_multiplier)

    # ---------------------------------------------------------------- select
    def select(
        self,
        pool: Any,
        probability_backend: Any,
        threshold_backend: Any,
        threshold_state: ThresholdState,
        target_count: int,
        exclude: set[int] | None = None,
    ) -> AcquisitionResult:
        # threshold_state is deliberately unread, as in the plain arm.
        if target_count <= 0:
            return self._skip("target_count_zero")
        if not hasattr(probability_backend, "estimate_members"):
            raise RuntimeError(
                "ClassQuotaDecompositionAcquisitionStrategy needs a backend exposing "
                f"estimate_members(); got {type(probability_backend).__name__}.")
        system = getattr(probability_backend, "system", None)

        states, indices = pool.sample_candidates_without_marking(
            self.n_candidates, exclude=exclude)
        n_actual = len(indices)
        if n_actual == 0:
            return self._skip("pool_exhausted")
        states = np.asarray(states)

        p_members = probability_backend.estimate_members(states, verbose=self.verbose)
        k = getattr(probability_backend, "member_sample_size", None)
        score = np.asarray(score_by_mode(self.score_mode, p_members, k), dtype=np.float64)
        score = np.where(np.isfinite(score), score, np.nan)
        p_bar = np.asarray(p_members, dtype=np.float64).mean(axis=0)  # diagnostics ONLY

        scales, mask = self._system_geometry(system, states.shape[1])

        # ---- the gate: kNN on acquired ground truth, p_bar nowhere in sight
        X_lab, y_lab, L_lab = self._acquired_truth(pool)
        n_succ_lab = int((y_lab == 1).sum())
        n_fail_lab = int((y_lab == 0).sum())
        gate_on = (min(n_succ_lab, n_fail_lab) >= self.min_labeled_per_class
                   and X_lab.shape[1] == states.shape[1])

        if not gate_on:
            # plain-arm behaviour, exactly: single-knob null preserved
            positions = self._select_within(score, states, scales, mask, target_count)
            knn_p = np.full(n_actual, np.nan)
            quota = float("nan")
            n_fail_quota = 0
        else:
            E_lab = embed_states(X_lab, scales, mask)
            E_cand = embed_states(states, scales, mask)
            knn_p = knn_success_prob(E_lab, y_lab, E_cand, self.knn_k)
            quota = (float((y_lab == 0).mean()) if self.quota_fail == "pool"
                     else float(self.quota_fail))
            quota = float(np.clip(quota, 0.0, 1.0))
            n_fail_quota = int(round(quota * target_count))

            fail_pot = np.where(knn_p < 0.5)[0]
            succ_pot = np.where(knn_p >= 0.5)[0]
            n_fail_take = min(n_fail_quota, len(fail_pot))
            n_succ_take = min(target_count - n_fail_take, len(succ_pot))
            # backfill whichever pot ran short from the other
            n_fail_take = min(n_fail_take + (target_count - n_fail_take - n_succ_take),
                              len(fail_pot))

            picks = []
            for pot, n_take in ((fail_pot, n_fail_take), (succ_pot, n_succ_take)):
                if n_take > 0:
                    local = self._select_within(score[pot], states[pot], scales, mask, n_take)
                    picks.append(pot[local])
            positions = (np.concatenate(picks) if picks else np.empty(0, dtype=int))

        selected = [indices[int(p)] for p in positions]
        diagnostics = self._diagnostics(
            p_members, score, knn_p, positions, n_actual, k, p_bar,
            gate_on, quota, n_fail_quota, n_succ_lab, n_fail_lab, y_lab, L_lab)
        if self.verbose:
            print(f"    [ClassQuota/{self.score_mode}] {n_actual} candidates, "
                  f"gate={'on' if gate_on else 'OFF'} quota_fail={quota:.2f} "
                  f"labeled S/F={n_succ_lab}/{n_fail_lab}; selected {len(selected)} "
                  f"({diagnostics['n_selected_pred_fail']} pred-fail); "
                  f"E[pairs]={diagnostics['expected_pairs_selected']:.0f}")

        return AcquisitionResult(
            d1_indices=[], d2_indices=selected,
            n_candidates_evaluated=n_actual,
            n_certain_discarded=n_actual - len(selected),
            n_invalid_added=0, diagnostics=diagnostics,
        )

    def _diagnostics(self, p_members, score, knn_p, positions, n_actual, k, p_bar,
                     gate_on, quota, n_fail_quota, n_succ_lab, n_fail_lab,
                     y_lab, L_lab) -> dict[str, Any]:
        finite = score[np.isfinite(score)]
        epi = epistemic_bald(p_members)
        ale = aleatoric_uncertainty(p_members)
        sel_knn = knn_p[positions] if len(positions) else np.empty(0)
        # expected pairs from MEASURED class mean lengths of acquired truth;
        # diagnostics only, never used for ranking
        ls = float(L_lab[y_lab == 1].mean()) if n_succ_lab else float("nan")
        lf = float(L_lab[y_lab == 0].mean()) if n_fail_lab else float("nan")
        if len(sel_knn) and np.isfinite(sel_knn).all() and np.isfinite([ls, lf]).all():
            exp_pairs = float(np.sum(sel_knn * (ls - 1) + (1 - sel_knn) * (lf - 1)))
        else:
            exp_pairs = float("nan")
        return {
            "score_mode": self.score_mode,
            "selection_rule": f"class_quota_{self.selection_rule}",
            "n_members": int(np.asarray(p_members).shape[0]),
            "member_sample_size": k,
            "score_mean": float(finite.mean()) if len(finite) else None,
            "score_mean_selected": float(np.nanmean(score[positions])) if len(positions) else None,
            "epistemic_mean": float(np.nanmean(epi)),
            "aleatoric_mean": float(np.nanmean(ale)),
            "epistemic_mean_selected": float(np.nanmean(epi[positions])) if len(positions) else None,
            "aleatoric_mean_selected": float(np.nanmean(ale[positions])) if len(positions) else None,
            # p_bar is recorded so the miscalibration stays visible, but the
            # selection above never reads it
            "p_bar_mean_candidates": float(np.nanmean(p_bar)),
            "p_bar_mean_selected": float(np.nanmean(p_bar[positions])) if len(positions) else None,
            "gate_on": bool(gate_on),
            "quota_fail_target": float(quota) if np.isfinite(quota) else None,
            "n_fail_quota": int(n_fail_quota),
            "n_labeled_success": int(n_succ_lab),
            "n_labeled_failure": int(n_fail_lab),
            "knn_k": int(self.knn_k),
            "knn_p_mean_candidates": (float(np.nanmean(knn_p))
                                      if np.isfinite(knn_p).any() else None),
            "knn_p_mean_selected": (float(np.nanmean(sel_knn))
                                    if np.isfinite(sel_knn).any() else None),
            "n_selected_pred_fail": int((sel_knn < 0.5).sum()) if len(sel_knn) else 0,
            "len_success_measured": ls,
            "len_failure_measured": lf,
            "expected_pairs_selected": exp_pairs,
            "n_candidates_evaluated": int(n_actual),
        }

    @staticmethod
    def _skip(reason: str) -> AcquisitionResult:
        return AcquisitionResult(
            d1_indices=[], d2_indices=[], n_candidates_evaluated=0,
            n_certain_discarded=0, n_invalid_added=0,
            diagnostics={"skipped_reason": reason},
        )
