"""
Calibrator for Conformal Prediction.

Computes the calibration threshold q_hat that provides coverage guarantees
and constructs prediction sets.
"""
import numpy as np
from typing import List, Set
from src.conformal.config import ConformalConfig


class Calibrator:
    """
    Calibrate conformal predictor and construct prediction sets.

    Uses non-conformity scores to calibrate and provides coverage guarantee:
    P(true_label ∈ prediction_set) ≥ 1 - α

    Non-conformity scores measure how "strange" a label is given the probability:
    - For FAILURE (y=-1): s = max(0, p - (λ-δ))  # High p makes FAILURE strange
    - For SUCCESS (y=1):  s = max(0, (λ+δ) - p)  # Low p makes SUCCESS strange
    - For UNKNOWN (y=0):  s = max(0, (λ-δ) - p, p - (λ+δ))  # Outside [λ-δ, λ+δ] is strange

    Attributes:
        config: ConformalConfig with delta, alpha
    """

    def __init__(self, config: ConformalConfig):
        """
        Initialize calibrator.

        Args:
            config: ConformalConfig with calibration parameters.
        """
        self.config = config

    def non_conformity_score(
        self,
        p_success: float,
        y_candidate: int,
        lambda_star: float,
        delta: float = None
    ) -> float:
        """
        Compute non-conformity score for a single (probability, label) pair.

        The score measures how "strange" it is to claim y_candidate is the
        true label given the estimated probability p_success.

        Args:
            p_success: Estimated p(success|x)
            y_candidate: Candidate label (-1, 0, or 1)
            lambda_star: Optimal decision boundary
            delta: Uncertainty half-width (uses config.delta if None)

        Returns:
            Non-conformity score (lower = more conforming)
        """
        if delta is None:
            delta = self.config.delta
        lower = lambda_star - delta
        upper = lambda_star + delta

        if y_candidate == -1:  # FAILURE
            # High p makes FAILURE claim strange
            return max(0.0, p_success - lower)
        elif y_candidate == 1:  # SUCCESS
            # Low p makes SUCCESS claim strange
            return max(0.0, upper - p_success)
        else:  # UNKNOWN (y_candidate == 0)
            # Being outside [lower, upper] makes UNKNOWN strange
            if p_success < lower:
                return lower - p_success
            elif p_success > upper:
                return p_success - upper
            else:
                return 0.0

    def non_conformity_scores_batch(
        self,
        p_success: np.ndarray,
        y_candidates: np.ndarray,
        lambda_star: float,
        delta: float = None
    ) -> np.ndarray:
        """
        Compute non-conformity scores for a batch of (probability, label) pairs.

        Args:
            p_success: [N] array of estimated p(success|x)
            y_candidates: [N] array of candidate labels (-1, 0, or 1)
            lambda_star: Optimal decision boundary
            delta: Uncertainty half-width (uses config.delta if None)

        Returns:
            [N] array of non-conformity scores
        """
        if delta is None:
            delta = self.config.delta
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

    def calibrate(
        self,
        p_success: np.ndarray,
        y_true: np.ndarray,
        lambda_star: float,
        delta: float = None
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

        Returns:
            q_hat: Calibration threshold. Prediction sets include labels
                   with non-conformity score ≤ q_hat.
        """
        n = len(y_true)
        alpha = self.config.alpha

        # Compute non-conformity score for each calibration point using TRUE label
        scores = self.non_conformity_scores_batch(p_success, y_true, lambda_star, delta)

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
        delta: float = None
    ) -> Set[int]:
        """
        Construct prediction set for a single point.

        Includes all labels whose non-conformity score is ≤ q_hat.

        Args:
            p_success: Estimated p(success|x)
            lambda_star: Optimal decision boundary
            q_hat: Calibration threshold
            delta: Uncertainty half-width (uses config.delta if None)

        Returns:
            Set of predicted labels (subset of {-1, 0, 1})
        """
        prediction_set = set()

        for label in [-1, 0, 1]:
            score = self.non_conformity_score(p_success, label, lambda_star, delta)
            if score <= q_hat:
                prediction_set.add(label)

        return prediction_set

    def get_prediction_sets_batch(
        self,
        p_success: np.ndarray,
        lambda_star: float,
        q_hat: float,
        delta: float = None
    ) -> List[Set[int]]:
        """
        Construct prediction sets for a batch of points.

        Args:
            p_success: [N] array of estimated p(success|x)
            lambda_star: Optimal decision boundary
            q_hat: Calibration threshold
            delta: Uncertainty half-width (uses config.delta if None)

        Returns:
            List of N prediction sets (each is a subset of {-1, 0, 1})
        """
        n = len(p_success)
        prediction_sets = []

        for i in range(n):
            pred_set = self.get_prediction_set(p_success[i], lambda_star, q_hat, delta)
            prediction_sets.append(pred_set)

        return prediction_sets

    def is_uncertain(
        self,
        p_success: float,
        lambda_star: float,
        q_hat: float,
        delta: float = None
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

        Returns:
            True if point is uncertain (prediction set size > 1)
        """
        pred_set = self.get_prediction_set(p_success, lambda_star, q_hat, delta)
        return len(pred_set) > 1

    def get_uncertain_mask(
        self,
        p_success: np.ndarray,
        lambda_star: float,
        q_hat: float,
        delta: float = None
    ) -> np.ndarray:
        """
        Get mask of uncertain points in a batch.

        Args:
            p_success: [N] array of estimated p(success|x)
            lambda_star: Optimal decision boundary
            q_hat: Calibration threshold
            delta: Uncertainty half-width (uses config.delta if None)

        Returns:
            [N] boolean array where True = uncertain (needs simulation)
        """
        prediction_sets = self.get_prediction_sets_batch(p_success, lambda_star, q_hat, delta)
        uncertain = np.array([len(ps) > 1 for ps in prediction_sets])
        return uncertain
