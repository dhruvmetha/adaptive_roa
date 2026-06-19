"""Factory selecting the probability estimator by predictor type.

Both the generative flow matcher and the discriminative classifier expose the
same ``estimate(states) -> (p_success, p_failure, p_invalid)`` contract, so every
consumer (acquisition backend, conformal threshold optimizer, evaluator) can be
predictor-agnostic by constructing its estimator through this factory.
"""
from __future__ import annotations

from typing import Any

from adaptive_roa.conformal.classifier_probability_estimator import ClassifierProbabilityEstimator
from adaptive_roa.conformal.probability_estimator import ProbabilityEstimator


def build_probability_estimator(
    predictor_type: str,
    model: Any,
    system: Any,
    config: Any,
    device: str = "cuda",
):
    """Return a probability estimator matching ``predictor_type``.

    Args:
        predictor_type: "classifier" -> forward-pass estimator; anything else
            (e.g. "generative") -> Monte-Carlo flow-matching estimator.
        model: trained model handle (ClassifierModule or flow matcher).
        system: dynamical system.
        config: ConformalConfig.
        device: torch device string.
    """
    if str(predictor_type) == "classifier":
        return ClassifierProbabilityEstimator(model, system, config, device)
    return ProbabilityEstimator(model, system, config, device)
