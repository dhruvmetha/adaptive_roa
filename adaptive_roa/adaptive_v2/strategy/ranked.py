"""Ranked acquisition strategy (D2 only)."""

from __future__ import annotations

from typing import Any

from adaptive_roa.adaptive.balanced_sampler import UncertainSampler
from adaptive_roa.adaptive_v2.types import AcquisitionResult, ThresholdState


class RankedAcquisitionStrategy:
    mode = "ranked"

    def __init__(self, cfg: Any):
        self.d2_ratio = float(cfg.d2_ratio)
        self.batch_size_sampling = int(cfg.batch_size_sampling)
        self.max_samples_per_epoch = int(cfg.max_samples_per_epoch)
        self.n_ranked_candidates = int(cfg.n_ranked_candidates)
        self.decision_rule = str(cfg.decision_rule)
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
        if target_count <= 0:
            return AcquisitionResult(
                d1_indices=[], d2_indices=[], n_candidates_evaluated=0,
                n_certain_discarded=0, n_invalid_added=0,
                diagnostics={"skipped_reason": "target_count_zero"},
            )
        sampler = UncertainSampler(
            dataset_builder=pool.dataset_builder,
            target_count=target_count,
            batch_size=self.batch_size_sampling,
            max_candidates=self.max_samples_per_epoch,
        )
        ranked_result = sampler.sample_ranked(
            prob_estimator=probability_backend.estimator,
            calibrator=threshold_backend.predictor.calibrator,
            lambda_star=threshold_state.lambda_star,
            delta_star=threshold_state.delta_star,
            decision_rule=self.decision_rule,
            n_candidates=self.n_ranked_candidates,
            n_select=target_count,
            verbose=self.verbose,
        )
        return AcquisitionResult(
            d1_indices=[],
            d2_indices=list(ranked_result.selected_indices),
            n_candidates_evaluated=int(ranked_result.n_candidates_evaluated),
            n_certain_discarded=int(ranked_result.n_candidates_evaluated - len(ranked_result.selected_indices)),
            n_invalid_added=0,
            diagnostics={
                "ranked_score_threshold": float(ranked_result.score_threshold),
                "n_ranked_candidates_evaluated": int(ranked_result.n_candidates_evaluated),
            },
        )
