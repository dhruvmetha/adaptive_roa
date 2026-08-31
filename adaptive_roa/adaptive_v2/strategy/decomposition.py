"""Acquisition on one component of an ensemble's uncertainty.

Threshold-free, like the entropy strategy: no lambda*, delta*, q_hat or decision
rule is read, so the training loop never depends on a fitted boundary. The only
difference between arms is which component of the uncertainty is scored, which
is what makes `aleatoric` a usable negative control -- it is the same code path
with one word changed.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from adaptive_roa.adaptive_v2.strategy.dispersion_score import (
    select_greedy, select_greedy_diverse,
)
from adaptive_roa.adaptive_v2.strategy.uncertainty_scores import (
    SCORE_MODES, aleatoric_uncertainty, epistemic_bald, score_by_mode,
)
from adaptive_roa.adaptive_v2.types import AcquisitionResult, ThresholdState

_SELECTION_RULES = ("greedy", "greedy_diverse")


class DecompositionAcquisitionStrategy:
    mode = "decomposition"

    def __init__(self, cfg: Any):
        self.score_mode = str(cfg.score)
        if self.score_mode not in SCORE_MODES:
            raise ValueError(
                f"unknown score mode {self.score_mode!r}; expected one of {SCORE_MODES}")
        self.d2_ratio = float(cfg.d2_ratio)
        self.n_candidates = int(cfg.n_candidates)
        self.selection_rule = str(cfg.selection_rule)
        if self.selection_rule not in _SELECTION_RULES:
            raise ValueError(
                f"selection_rule must be one of {_SELECTION_RULES}, got {self.selection_rule!r}")
        self.diversity_pool_multiplier = int(cfg.diversity_pool_multiplier)
        self.verbose = bool(cfg.verbose)

    def select(
        self,
        pool: Any,
        probability_backend: Any,
        threshold_backend: Any,
        threshold_state: ThresholdState,
        target_count: int,
        exclude: set[int] | None = None,
    ) -> AcquisitionResult:
        # threshold_state is deliberately unread; see module docstring.
        if target_count <= 0:
            return self._skip("target_count_zero")

        system = getattr(probability_backend, "system", None)
        if system is None or not hasattr(probability_backend, "estimate_members"):
            raise RuntimeError(
                "DecompositionAcquisitionStrategy needs a backend exposing system and "
                f"estimate_members(); got {type(probability_backend).__name__}. A "
                "non-ensemble backend cannot separate aleatoric from epistemic."
            )

        states, indices = pool.sample_candidates_without_marking(
            self.n_candidates, exclude=exclude)
        n_actual = len(indices)
        if n_actual == 0:
            return self._skip("pool_exhausted")

        p_members = probability_backend.estimate_members(states, verbose=self.verbose)
        k = getattr(probability_backend, "member_sample_size", None)
        score = np.asarray(score_by_mode(self.score_mode, p_members, k), dtype=np.float64)
        score = np.where(np.isfinite(score), score, np.nan)

        positions = self._apply_selection_rule(score, np.asarray(states), system, target_count)
        selected = [indices[int(p)] for p in positions]

        diagnostics = self._diagnostics(p_members, score, positions, n_actual, k)
        if self.verbose:
            print(f"    [Decomposition/{self.score_mode}] {n_actual} candidates, "
                  f"M={diagnostics['n_members']}, selected {len(selected)}; "
                  f"epistemic_mean={diagnostics['epistemic_mean']:.5f} "
                  f"aleatoric_mean={diagnostics['aleatoric_mean']:.5f}")

        return AcquisitionResult(
            d1_indices=[], d2_indices=selected,
            n_candidates_evaluated=n_actual,
            n_certain_discarded=n_actual - len(selected),
            n_invalid_added=0, diagnostics=diagnostics,
        )

    @staticmethod
    def _circular_mask(system) -> np.ndarray:
        n = int(system.state_dim)
        mask = np.zeros(n, dtype=bool)
        idx = system.get_circular_indices()
        if idx:
            mask[np.asarray(idx, dtype=int)] = True
        return mask

    def _apply_selection_rule(self, score, states, system, target_count) -> np.ndarray:
        if self.selection_rule == "greedy":
            return select_greedy(score, target_count)
        return select_greedy_diverse(
            score, states,
            system.get_normalization_scales().cpu().numpy().astype(np.float64),
            self._circular_mask(system), target_count,
            pool_multiplier=self.diversity_pool_multiplier,
        )

    def _diagnostics(self, p_members, score, positions, n_actual, k) -> dict[str, Any]:
        finite = score[np.isfinite(score)]
        # Both components are recorded on every arm, not just the one being
        # scored: that is what lets the report show WHY an arm chose what it did.
        epi = epistemic_bald(p_members)
        ale = aleatoric_uncertainty(p_members)
        return {
            "score_mode": self.score_mode,
            "n_members": int(np.asarray(p_members).shape[0]),
            "member_sample_size": k,
            "score_min": float(finite.min()) if len(finite) else None,
            "score_max": float(finite.max()) if len(finite) else None,
            "score_mean": float(finite.mean()) if len(finite) else None,
            "score_mean_selected": float(np.nanmean(score[positions])) if len(positions) else None,
            "epistemic_mean": float(np.nanmean(epi)),
            "aleatoric_mean": float(np.nanmean(ale)),
            "epistemic_mean_selected": float(np.nanmean(epi[positions])) if len(positions) else None,
            "aleatoric_mean_selected": float(np.nanmean(ale[positions])) if len(positions) else None,
            "n_candidates_evaluated": int(n_actual),
            "selection_rule": self.selection_rule,
        }

    @staticmethod
    def _skip(reason: str) -> AcquisitionResult:
        return AcquisitionResult(
            d1_indices=[], d2_indices=[], n_candidates_evaluated=0,
            n_certain_discarded=0, n_invalid_added=0,
            diagnostics={"skipped_reason": reason},
        )
