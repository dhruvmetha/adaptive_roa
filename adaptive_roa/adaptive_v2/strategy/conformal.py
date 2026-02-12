"""Conformal acquisition strategy (D2 selection with q-hat)."""

from __future__ import annotations

from typing import Any

from adaptive_roa.adaptive.balanced_sampler import UncertainSampler
from adaptive_roa.adaptive_v2.types import AcquisitionResult, ThresholdState


class ConformalAcquisitionStrategy:
    def select(
        self,
        pool: Any,
        probability_backend: Any,
        threshold_backend: Any,
        threshold_state: ThresholdState,
        cfg: Any,
        target_count: int,
        exclude: set[int] | None = None,
    ) -> AcquisitionResult:
        if threshold_state.q_hat is None:
            raise ValueError("Conformal strategy requires q_hat in ThresholdState")

        sampler = UncertainSampler(
            dataset_builder=pool.dataset_builder,
            target_count=target_count,
            batch_size=cfg.get("batch_size_sampling", 50),
            max_candidates=cfg.get("max_samples_per_epoch", 50000),
        )

        sample_result = sampler.sample(
            prob_estimator=probability_backend.estimator,
            calibrator=threshold_backend.predictor.calibrator,
            lambda_star=threshold_state.lambda_star,
            delta_star=threshold_state.delta_star,
            q_hat=threshold_state.q_hat,
            decision_rule=cfg.conformal.get("decision_rule", "two_sided"),
            exclude=exclude or set(),
            verbose=cfg.conformal.get("verbose", True),
        )

        return AcquisitionResult(
            d1_indices=[],
            d2_indices=list(sample_result.uncertain_indices),
            n_candidates_evaluated=int(sample_result.n_candidates_evaluated),
            n_certain_discarded=int(sample_result.n_certain_discarded),
            n_invalid_added=int(sample_result.n_invalid_added),
            diagnostics={
                "n_batches": int(sample_result.n_batches),
            },
        )
