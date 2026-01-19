"""
Calibrator for Conformal Prediction.

Computes the calibration threshold q_hat that provides coverage guarantees
and constructs prediction sets.

Supports two decision rules:
- one_sided: Uses only p_success (default, for pendulum-like systems)
- two_sided: Uses both p_success and p_failure (for CartPole-like systems)
"""
import numpy as np
from typing import List, Set, Optional
from adaptive_roa.conformal.config import ConformalConfig


def nonconformity_score_one_sided(
    p_success: float,
    y_candidate: int,
    lambda_star: float,
    delta: float
) -> float:
    """
    One-sided nonconformity score using only p_success.
    
    - FAILURE (y=-1): High p_s makes FAILURE strange
    - SUCCESS (y=1): Low p_s makes SUCCESS strange  
    - UNKNOWN (y=0): Being outside [λ-δ, λ+δ] is strange
    """
    lower = lambda_star - delta
    upper = lambda_star + delta

    if y_candidate == -1:  # FAILURE
        return max(0.0, p_success - lower)
    elif y_candidate == 1:  # SUCCESS
        return max(0.0, upper - p_success)
    else:  # UNKNOWN (y_candidate == 0)
        if p_success < lower:
            return lower - p_success
        elif p_success > upper:
            return p_success - upper
        else:
            return 0.0


def nonconformity_score_two_sided(
    p_success: float,
    p_failure: float,
    y_candidate: int,
    lambda_star: float,
    delta: float
) -> float:
    """
    Two-sided nonconformity score using both p_success and p_failure.
    
    Decision regions (with u = λ+δ, v = 1-λ+δ):
    - SUCCESS: p_s >= u and p_f <= v
    - FAILURE: p_f >= v and p_s <= u  
    - UNKNOWN/SEPARATRIX: neither confident (p_s < u and p_f < v)
    
    Score = max of constraint violations (L_inf distance to region).
    """
    u = lambda_star + delta       # success threshold on p_s
    v = 1 - lambda_star + delta   # failure threshold on p_f

    if y_candidate == 1:  # SUCCESS
        # Require: p_s >= u and p_f <= v
        # Score = max(0, u - p_s, p_f - v)
        return max(0.0, u - p_success, p_failure - v)
    elif y_candidate == -1:  # FAILURE
        # Require: p_f >= v and p_s <= u
        # Score = max(0, v - p_f, p_s - u)
        return max(0.0, v - p_failure, p_success - u)
    else:  # UNKNOWN (y_candidate == 0)
        # Require: p_s < u and p_f < v (separatrix region)
        # Score = max(0, p_s - u, p_f - v)
        return max(0.0, p_success - u, p_failure - v)


def nonconformity_scores_batch_one_sided(
    p_success: np.ndarray,
    y_candidates: np.ndarray,
    lambda_star: float,
    delta: float
) -> np.ndarray:
    """Batch one-sided nonconformity scores."""
    lower = lambda_star - delta
    upper = lambda_star + delta

    scores = np.zeros(len(p_success))

    # FAILURE candidates
    failure_mask = y_candidates == -1
    scores[failure_mask] = np.maximum(0, p_success[failure_mask] - lower)

    # SUCCESS candidates
    success_mask = y_candidates == 1
    scores[success_mask] = np.maximum(0, upper - p_success[success_mask])

    # UNKNOWN candidates
    unknown_mask = y_candidates == 0
    unknown_p = p_success[unknown_mask]
    unknown_scores = np.zeros(np.sum(unknown_mask))
    unknown_scores[unknown_p < lower] = lower - unknown_p[unknown_p < lower]
    unknown_scores[unknown_p > upper] = unknown_p[unknown_p > upper] - upper
    scores[unknown_mask] = unknown_scores

    return scores


def nonconformity_scores_batch_two_sided(
    p_success: np.ndarray,
    p_failure: np.ndarray,
    y_candidates: np.ndarray,
    lambda_star: float,
    delta: float
) -> np.ndarray:
    """Batch two-sided nonconformity scores using p_s and p_f."""
    u = lambda_star + delta       # success threshold on p_s
    v = 1 - lambda_star + delta   # failure threshold on p_f

    scores = np.zeros(len(p_success))

    # SUCCESS candidates: require p_s >= u and p_f <= v
    success_mask = y_candidates == 1
    scores[success_mask] = np.maximum(
        0,
        np.maximum(u - p_success[success_mask], p_failure[success_mask] - v)
    )

    # FAILURE candidates: require p_f >= v and p_s <= u
    failure_mask = y_candidates == -1
    scores[failure_mask] = np.maximum(
        0,
        np.maximum(v - p_failure[failure_mask], p_success[failure_mask] - u)
    )

    # UNKNOWN candidates: require p_s < u and p_f < v
    unknown_mask = y_candidates == 0
    scores[unknown_mask] = np.maximum(
        0,
        np.maximum(p_success[unknown_mask] - u, p_failure[unknown_mask] - v)
    )

    return scores


class Calibrator:
    """
    Calibrate conformal predictor and construct prediction sets.

    Uses non-conformity scores to calibrate and provides coverage guarantee:
    P(true_label ∈ prediction_set) ≥ 1 - α

    Supports two decision rules:
    - one_sided: Uses only p_success (default)
        - FAILURE (y=-1): s = max(0, p_s - (λ-δ))
        - SUCCESS (y=1):  s = max(0, (λ+δ) - p_s)
        - UNKNOWN (y=0):  s = max(0, (λ-δ) - p_s, p_s - (λ+δ))
    
    - two_sided: Uses both p_success and p_failure
        - SUCCESS (y=1):  s = max(0, (λ+δ) - p_s, p_f - (1-λ+δ))
        - FAILURE (y=-1): s = max(0, (1-λ+δ) - p_f, p_s - (λ+δ))
        - UNKNOWN (y=0):  s = max(0, p_s - (λ+δ), p_f - (1-λ+δ))

    Attributes:
        config: ConformalConfig with delta, alpha, decision_rule
    """

    def __init__(self, config: ConformalConfig):
        """
        Initialize calibrator.

        Args:
            config: ConformalConfig with calibration parameters.
        """
        self.config = config
        self.decision_rule = getattr(config, 'decision_rule', 'one_sided')

    def non_conformity_score(
        self,
        p_success: float,
        y_candidate: int,
        lambda_star: float,
        delta: float = None,
        p_failure: Optional[float] = None
    ) -> float:
        """
        Compute non-conformity score for a single (probability, label) pair.

        The score measures how "strange" it is to claim y_candidate is the
        true label given the estimated probability.

        Args:
            p_success: Estimated p(success|x)
            y_candidate: Candidate label (-1, 0, or 1)
            lambda_star: Optimal decision boundary
            delta: Uncertainty half-width (uses config.delta if None)
            p_failure: Estimated p(failure|x), required for two_sided rule

        Returns:
            Non-conformity score (lower = more conforming)
        """
        if delta is None:
            delta = self.config.delta

        if self.decision_rule == "two_sided":
            if p_failure is None:
                raise ValueError("p_failure required for two_sided decision rule")
            return nonconformity_score_two_sided(
                p_success, p_failure, y_candidate, lambda_star, delta
            )
        else:
            return nonconformity_score_one_sided(
                p_success, y_candidate, lambda_star, delta
            )

    def non_conformity_scores_batch(
        self,
        p_success: np.ndarray,
        y_candidates: np.ndarray,
        lambda_star: float,
        delta: float = None,
        p_failure: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute non-conformity scores for a batch of (probability, label) pairs.

        Args:
            p_success: [N] array of estimated p(success|x)
            y_candidates: [N] array of candidate labels (-1, 0, or 1)
            lambda_star: Optimal decision boundary
            delta: Uncertainty half-width (uses config.delta if None)
            p_failure: [N] array of estimated p(failure|x), required for two_sided

        Returns:
            [N] array of non-conformity scores
        """
        if delta is None:
            delta = self.config.delta

        if self.decision_rule == "two_sided":
            if p_failure is None:
                raise ValueError("p_failure required for two_sided decision rule")
            return nonconformity_scores_batch_two_sided(
                p_success, p_failure, y_candidates, lambda_star, delta
            )
        else:
            return nonconformity_scores_batch_one_sided(
                p_success, y_candidates, lambda_star, delta
            )

    def calibrate(
        self,
        p_success: np.ndarray,
        y_true: np.ndarray,
        lambda_star: float,
        delta: float = None,
        p_failure: Optional[np.ndarray] = None
    ) -> float:
        """
        Compute calibration threshold q_hat from calibration set.

        Uses the non-conformity scores of true labels to find the threshold
        that provides (1-α) coverage.

        Args:
            p_success: [N] array of estimated p(success|x) for calibration set
            y_true: [N] array of TRUE labels for calibration set
            lambda_star: Optimal decision boundary (from lambda optimizer)
            delta: Uncertainty half-width (uses config.delta if None)
            p_failure: [N] array of estimated p(failure|x), required for two_sided

        Returns:
            q_hat: Calibration threshold. Prediction sets include labels
                   with non-conformity score ≤ q_hat.
        """
        n = len(y_true)
        alpha = self.config.alpha

        # Compute non-conformity score for each calibration point using TRUE label
        scores = self.non_conformity_scores_batch(
            p_success, y_true, lambda_star, delta, p_failure
        )

        # Compute the (1-α)(n+1)/n quantile
        # This is the finite-sample correction for conformal prediction
        quantile_level = (1 - alpha) * (n + 1) / n
        quantile_level = min(quantile_level, 1.0)  # Cap at 1.0

        q_hat = np.quantile(scores, quantile_level)

        return q_hat

    def get_prediction_set(
        self,
        p_success: float,
        lambda_star: float,
        q_hat: float,
        delta: float = None,
        p_failure: Optional[float] = None
    ) -> Set[int]:
        """
        Construct prediction set for a single point.

        Includes all labels whose non-conformity score is ≤ q_hat.

        Args:
            p_success: Estimated p(success|x)
            lambda_star: Optimal decision boundary
            q_hat: Calibration threshold
            delta: Uncertainty half-width (uses config.delta if None)
            p_failure: Estimated p(failure|x), required for two_sided

        Returns:
            Set of predicted labels (subset of {-1, 0, 1})
        """
        prediction_set = set()

        for label in [-1, 0, 1]:
            score = self.non_conformity_score(
                p_success, label, lambda_star, delta, p_failure
            )
            if score <= q_hat:
                prediction_set.add(label)

        return prediction_set

    def get_prediction_sets_batch(
        self,
        p_success: np.ndarray,
        lambda_star: float,
        q_hat: float,
        delta: float = None,
        p_failure: Optional[np.ndarray] = None
    ) -> List[Set[int]]:
        """
        Construct prediction sets for a batch of points.

        Args:
            p_success: [N] array of estimated p(success|x)
            lambda_star: Optimal decision boundary
            q_hat: Calibration threshold
            delta: Uncertainty half-width (uses config.delta if None)
            p_failure: [N] array of estimated p(failure|x), required for two_sided

        Returns:
            List of N prediction sets (each is a subset of {-1, 0, 1})
        """
        n = len(p_success)
        prediction_sets = []

        for i in range(n):
            pf_i = p_failure[i] if p_failure is not None else None
            pred_set = self.get_prediction_set(
                p_success[i], lambda_star, q_hat, delta, pf_i
            )
            prediction_sets.append(pred_set)

        return prediction_sets

    def is_uncertain(
        self,
        p_success: float,
        lambda_star: float,
        q_hat: float,
        delta: float = None,
        p_failure: Optional[float] = None
    ) -> bool:
        """
        Check if a point is uncertain (prediction set contains multiple labels).

        A point is uncertain if its prediction set contains more than one label,
        meaning we can't confidently predict SUCCESS or FAILURE.

        Args:
            p_success: Estimated p(success|x)
            lambda_star: Optimal decision boundary
            q_hat: Calibration threshold
            delta: Uncertainty half-width (uses config.delta if None)
            p_failure: Estimated p(failure|x), required for two_sided

        Returns:
            True if point is uncertain (prediction set size > 1)
        """
        pred_set = self.get_prediction_set(
            p_success, lambda_star, q_hat, delta, p_failure
        )
        return len(pred_set) > 1

    def get_uncertain_mask(
        self,
        p_success: np.ndarray,
        lambda_star: float,
        q_hat: float,
        delta: float = None,
        p_failure: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Get mask of uncertain points in a batch.

        Args:
            p_success: [N] array of estimated p(success|x)
            lambda_star: Optimal decision boundary
            q_hat: Calibration threshold
            delta: Uncertainty half-width (uses config.delta if None)
            p_failure: [N] array of estimated p(failure|x), required for two_sided

        Returns:
            [N] boolean array where True = uncertain (needs simulation)
        """
        prediction_sets = self.get_prediction_sets_batch(
            p_success, lambda_star, q_hat, delta, p_failure
        )
        uncertain = np.array([len(ps) > 1 for ps in prediction_sets])
        return uncertain
