"""Rank candidates by expected trajectory length ALONE -- the yield-weight control.

Why this exists (pendulum i100 sweep, 2026-08-16):

``yield_aware`` ranks by ``score(x) * E[L(x)]`` and beats uniform sampling at
matched trajectory budget (+1.87 floor units on KL, pooled ep15-19, against a
3-seed control floor). But that arm confounds two factors, and the campaign has
no arm that separates them:

  1. the epistemic score -- WHICH states are informative, and
  2. the expected-length weight -- how many training pairs each state buys.

On ``noisy/pendulum/lqr/high`` the budget is trajectories while the model trains
on pairs, and success rollouts stop at the goal (mean 136.5 steps) while
failures run to the 1001-step cap. So a strategy that did nothing but buy the
longest expected rollouts would also accumulate pairs far faster than uniform.
If THAT alone reproduces the win, the epistemic component contributes nothing
and the honest description of the result is "buy long trajectories", not
"informativeness per unit budget".

This strategy is that null: it ranks by

    E[L(x)] = p_bar(x) * L_success + (1 - p_bar(x)) * L_failure

and ignores the uncertainty score entirely for SELECTION. It still computes the
score and reports it in diagnostics, so we can measure directly whether ranking
by length incidentally selects uncertain points -- i.e. whether the two factors
are correlated on this system rather than independent.

Note what this arm does dynamically: with L_failure >> L_success, ranking by
E[L] is monotone DECREASING in p_bar, so it greedily buys the model's predicted
FAILURE region. That is the opposite pole from every scored arm run so far
(which bought the uncertain band = the true-success region), which makes it a
useful bracket on the acquisition space regardless of how it scores.

A NEW module rather than a flag on ``yield_aware`` on purpose: sibling runs
(yield_s43, yield_s44) are in flight and ``mp.spawn`` trainer children re-import
strategy modules from disk every adaptive epoch, so editing a module they touch
could change their behaviour mid-run.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from adaptive_roa.adaptive_v2.strategy.uncertainty_scores import (
    SCORE_MODES, aleatoric_uncertainty, epistemic_bald, score_by_mode,
)
from adaptive_roa.adaptive_v2.strategy.yield_aware import expected_length
from adaptive_roa.adaptive_v2.types import AcquisitionResult, ThresholdState


class LengthOnlyAcquisitionStrategy:
    """Greedy top-N by expected pairs purchased, with no uncertainty term.

    The diagnostic score mode is still evaluated and reported; it never affects
    which candidates are selected.
    """

    mode = "decomposition"

    def __init__(self, cfg: Any):
        # Kept only so diagnostics stay comparable with the scored arms.
        self.score_mode = str(cfg.score)
        if self.score_mode not in SCORE_MODES:
            raise ValueError(
                f"unknown score mode {self.score_mode!r}; expected one of {SCORE_MODES}")
        self.d2_ratio = float(cfg.d2_ratio)
        self.n_candidates = int(cfg.n_candidates)
        self.prior_len_success = float(cfg.get("prior_len_success", 150.0))
        self.prior_len_failure = float(cfg.get("prior_len_failure", 1000.0))
        self.min_per_class = int(cfg.get("min_per_class", 5))
        self.verbose = bool(cfg.verbose)

    # ------------------------------------------------------------ length model
    def _length_stats(self, pool: Any) -> tuple[float, float, int, int]:
        """Mean length of acquired success / failure trajectories.

        Same estimator as ``yield_aware``: measured from trajectories already
        paid for, never from the candidate, whose length is unknown until it is
        rolled out. Falls back to the prior for a class with too few examples.
        """
        builder = getattr(pool, "dataset_builder", None)
        source = getattr(builder, "data_source", None)
        split = getattr(builder, "train_split", None)
        if source is None or split is None:
            return self.prior_len_success, self.prior_len_failure, 0, 0

        lens_s: list[int] = []
        lens_f: list[int] = []
        for idx in list(getattr(split, "indices", []) or []):
            try:
                L = int(source.get_trajectory_length(int(idx)))
                y = int(source.get_label(int(idx)))
            except Exception:
                continue
            (lens_s if y == 1 else lens_f).append(L)

        ls = float(np.mean(lens_s)) if len(lens_s) >= self.min_per_class else self.prior_len_success
        lf = float(np.mean(lens_f)) if len(lens_f) >= self.min_per_class else self.prior_len_failure
        return ls, lf, len(lens_s), len(lens_f)

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
        # threshold_state is deliberately unread, as in the sibling strategies.
        if target_count <= 0:
            return self._skip("target_count_zero")
        if not hasattr(probability_backend, "estimate_members"):
            raise RuntimeError(
                "LengthOnlyAcquisitionStrategy needs a backend exposing "
                f"estimate_members(); got {type(probability_backend).__name__}.")

        states, indices = pool.sample_candidates_without_marking(
            self.n_candidates, exclude=exclude)
        n_actual = len(indices)
        if n_actual == 0:
            return self._skip("pool_exhausted")

        p_members = probability_backend.estimate_members(states, verbose=self.verbose)
        k = getattr(probability_backend, "member_sample_size", None)
        p_bar = np.asarray(p_members, dtype=np.float64).mean(axis=0)

        len_s, len_f, n_s, n_f = self._length_stats(pool)
        exp_len = expected_length(p_bar, len_s, len_f)

        # Selection uses ONLY expected length. The score below is diagnostic.
        key = np.where(np.isfinite(exp_len), exp_len, -np.inf)
        positions = np.argsort(-key, kind="stable")[:target_count]
        selected = [indices[int(p)] for p in positions]

        base = np.asarray(score_by_mode(self.score_mode, p_members, k), dtype=np.float64)
        diagnostics = self._diagnostics(
            p_members, base, exp_len, p_bar, positions, n_actual, k, len_s, len_f, n_s, n_f)
        if self.verbose:
            print(f"    [LengthOnly] {n_actual} candidates, L_succ={len_s:.0f} "
                  f"L_fail={len_f:.0f}; selected {len(selected)}; "
                  f"p_bar mean cand={diagnostics['p_bar_mean_candidates']:.4f} "
                  f"sel={diagnostics['p_bar_mean_selected']:.4f}; "
                  f"E[pairs]={diagnostics['expected_pairs_selected']:.0f}")

        return AcquisitionResult(
            d1_indices=[], d2_indices=selected,
            n_candidates_evaluated=n_actual,
            n_certain_discarded=n_actual - len(selected),
            n_invalid_added=0, diagnostics=diagnostics,
        )

    def _diagnostics(self, p_members, base, exp_len, p_bar, positions,
                     n_actual, k, len_s, len_f, n_s, n_f) -> dict[str, Any]:
        epi = epistemic_bald(p_members)
        ale = aleatoric_uncertainty(p_members)
        # Does length-ranking incidentally buy uncertain points? Compare the
        # selected set's score against the candidate pool's mean: a ratio near
        # 1.0 means the two factors are independent on this system.
        base_all = float(np.nanmean(base))
        base_sel = float(np.nanmean(base[positions])) if len(positions) else None
        return {
            "score_mode": self.score_mode,
            "n_members": int(np.asarray(p_members).shape[0]),
            "member_sample_size": k,
            "score_mean": base_all,
            # named to match the scored arms so the monitoring tooling lines up
            "score_mean_selected": base_sel,
            "base_score_mean_selected": base_sel,
            "score_selected_over_candidates": (
                base_sel / base_all if base_sel is not None and base_all else None),
            "epistemic_mean": float(np.nanmean(epi)),
            "aleatoric_mean": float(np.nanmean(ale)),
            "epistemic_mean_selected": float(np.nanmean(epi[positions])) if len(positions) else None,
            "aleatoric_mean_selected": float(np.nanmean(ale[positions])) if len(positions) else None,
            "p_bar_mean_candidates": float(np.nanmean(p_bar)),
            "p_bar_mean_selected": float(np.nanmean(p_bar[positions])) if len(positions) else None,
            "len_success_est": float(len_s),
            "len_failure_est": float(len_f),
            "n_acquired_success": int(n_s),
            "n_acquired_failure": int(n_f),
            "expected_pairs_selected": (
                float(exp_len[positions].sum()) if len(positions) else 0.0),
            "n_candidates_evaluated": int(n_actual),
            "selection_rule": "expected_length_only",
        }

    @staticmethod
    def _skip(reason: str) -> AcquisitionResult:
        return AcquisitionResult(
            d1_indices=[], d2_indices=[], n_candidates_evaluated=0,
            n_certain_discarded=0, n_invalid_added=0,
            diagnostics={"skipped_reason": reason},
        )
