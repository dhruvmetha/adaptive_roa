"""
Lambda/Delta Optimizer for Conformal Prediction.

Finds the optimal decision boundary via grid search:
- Mode "lambda": Optimize λ* with fixed δ
- Mode "delta": Optimize δ* with fixed λ=0.5

Decision rules:
- "one_sided": Uses only p_success (default, for pendulum/mountain car)
- "two_sided": Uses both p_success and p_failure (for CartPole)
"""
import numpy as np
from typing import Tuple, Optional
from adaptive_roa.conformal.config import ConformalConfig


# =============================================================================
# Decision Rule Functions
# =============================================================================

def apply_one_sided_rule(
    p_success: np.ndarray,
    p_failure: Optional[np.ndarray],
    lambda_star: float,
    delta: float
) -> np.ndarray:
    """
    One-sided decision rule using only p_success.
    
    - SUCCESS if p_success > λ + δ
    - FAILURE if p_success < λ - δ
    - UNKNOWN otherwise
    
    Used for systems where failure is simply "not success" (pendulum, mountain car).
    """
    n = len(p_success)
    pred = np.zeros(n)
    pred[p_success > lambda_star + delta] = 1
    pred[p_success < lambda_star - delta] = -1
    return pred


def apply_two_sided_rule(
    p_success: np.ndarray,
    p_failure: np.ndarray,
    lambda_star: float,
    delta: float
) -> np.ndarray:
    """
    Two-sided decision rule using both p_success and p_failure.
    
    - SUCCESS if p_success > λ + δ
    - FAILURE if (1 - p_failure) < λ - δ  (equivalently p_failure > 1 - (λ - δ))
    - UNKNOWN otherwise
    - If both conditions trigger, treat as UNKNOWN (rare edge case)
    
    Used for systems with explicit failure conditions (CartPole).
    """
    n = len(p_success)
    pred = np.zeros(n)
    
    success_thresh = lambda_star + delta
    failure_thresh = lambda_star - delta  # threshold for (1 - p_f)
    
    pred[p_success > success_thresh] = 1
    pred[(1.0 - p_failure) < failure_thresh] = -1
    
    # # If both trigger, treat as unknown
    # both = (p_success > success_thresh) & ((1.0 - p_failure) < failure_thresh)
    # pred[both] = 0
    
    return pred


class LambdaOptimizer:
    """
    Find optimal decision boundary via grid search.

    Two modes:
    1. optimize_mode="lambda": Find λ* with fixed δ
       - Search λ ∈ [δ, 1-δ]

    2. optimize_mode="delta": Find δ* with fixed λ=0.5
       - Search δ ∈ [delta_min, delta_max]

    Two decision rules (config.decision_rule):
    1. "one_sided": Uses only p_success
       - SUCCESS if p_s > λ+δ, FAILURE if p_s < λ-δ
    2. "two_sided": Uses both p_success and p_failure
       - SUCCESS if p_s > λ+δ, FAILURE if (1-p_f) < λ-δ

    Loss function:
        Loss = w × MisclassificationRate + (1-w) × UnknownRate

    Attributes:
        config: ConformalConfig with optimization parameters
    """

    def __init__(self, config: ConformalConfig):
        """
        Initialize optimizer.

        Args:
            config: ConformalConfig with optimization parameters.
        """
        self.config = config

    def optimize(
        self,
        p_success: np.ndarray,
        y_true: np.ndarray,
        p_failure: Optional[np.ndarray] = None
    ) -> Tuple[float, float, dict]:
        """
        Find optimal parameters via grid search based on optimize_mode.

        Args:
            p_success: [N] array of estimated p(success|x) for training states
            y_true: [N] array of true labels (-1 for failure, 1 for success)
            p_failure: [N] array of estimated p(failure|x). Required if decision_rule="two_sided".

        Returns:
            Tuple of:
                lambda_star: Optimal λ (0.5 if mode="delta")
                delta_star: Optimal δ (config.delta if mode="lambda")
                info: Dict with optimization details
        """
        # Validate p_failure for two_sided rule
        if self.config.decision_rule == "two_sided" and p_failure is None:
            raise ValueError("p_failure is required when decision_rule='two_sided'")
        
        # Default p_failure to zeros for one_sided (unused but allows uniform code)
        if p_failure is None:
            p_failure = np.zeros_like(p_success)
        
        if self.config.optimize_mode == "delta":
            return self.optimize_delta(p_success, y_true, p_failure)
        else:
            return self.optimize_lambda(p_success, y_true, p_failure)

    def optimize_lambda(
        self,
        p_success: np.ndarray,
        y_true: np.ndarray,
        p_failure: np.ndarray
    ) -> Tuple[float, float, dict]:
        """
        Find optimal λ* via grid search.

        For each λ in [δ, 1-δ], computes the loss and returns the λ
        that minimizes it.

        Args:
            p_success: [N] array of estimated p(success|x) for training states
            y_true: [N] array of true labels (-1 for failure, 1 for success)
            p_failure: [N] array of estimated p(failure|x) for training states

        Returns:
            Tuple of:
                lambda_star: Optimal decision boundary
                info: Dict with optimization details (losses, best_loss, etc.)
        """
        delta = self.config.delta
        w = self.config.w
        grid_size = self.config.lambda_grid_size
        decision_rule = self.config.decision_rule

        # Grid search over λ ∈ [δ, 1-δ]
        # We need λ-δ ≥ 0 and λ+δ ≤ 1, so λ ∈ [δ, 1-δ]
        lambdas = np.linspace(delta, 1 - delta, grid_size)

        # Debug: print distribution of inputs
        print(f"      [λ Debug] decision_rule: {decision_rule}")
        print(f"      [λ Debug] y_true distribution: {np.sum(y_true == 1)} success, {np.sum(y_true == -1)} failure")
        print(f"      [λ Debug] p_success stats: min={p_success.min():.4f}, max={p_success.max():.4f}, mean={p_success.mean():.4f}, median={np.median(p_success):.4f}")
        if decision_rule == "two_sided":
            print(f"      [λ Debug] p_failure stats: min={p_failure.min():.4f}, max={p_failure.max():.4f}, mean={p_failure.mean():.4f}, median={np.median(p_failure):.4f}")

        best_lambda = 0.5
        best_loss = float('inf')
        losses = []
        misclass_rates = []
        unknown_rates = []

        n_samples = len(y_true)

        for lam in lambdas:
            # Apply decision rule based on config
            if decision_rule == "two_sided":
                predictions = apply_two_sided_rule(p_success, p_failure, lam, delta)
            else:
                predictions = apply_one_sided_rule(p_success, p_failure, lam, delta)

            unknown_mask = predictions == 0

            # Count misclassifications (only among confident predictions)
            confident_mask = ~unknown_mask
            n_confident = np.sum(confident_mask)

            if n_confident > 0:
                misclassified = np.sum(predictions[confident_mask] != y_true[confident_mask])
                misclass_rate = misclassified / n_confident
            else:
                # All unknown - no misclassifications but max unknown rate
                misclass_rate = 0.0

            # Unknown rate
            unknown_rate = np.sum(unknown_mask) / n_samples

            # Weighted loss
            loss = w * misclass_rate + (1 - w) * unknown_rate

            losses.append(loss)
            misclass_rates.append(misclass_rate)
            unknown_rates.append(unknown_rate)

            if loss < best_loss:
                best_loss = loss
                best_lambda = lam

        info = {
            'optimize_mode': 'lambda',
            'decision_rule': decision_rule,
            'lambdas': lambdas,
            'losses': np.array(losses),
            'misclass_rates': np.array(misclass_rates),
            'unknown_rates': np.array(unknown_rates),
            'best_loss': best_loss,
            'best_misclass_rate': misclass_rates[np.argmin(losses)],
            'best_unknown_rate': unknown_rates[np.argmin(losses)],
        }

        # Return (lambda_star, delta, info) - delta is fixed from config
        return best_lambda, delta, info

    def optimize_delta(
        self,
        p_success: np.ndarray,
        y_true: np.ndarray,
        p_failure: np.ndarray
    ) -> Tuple[float, float, dict]:
        """
        Find optimal δ* via grid search with fixed λ=0.5.

        For each δ in [delta_min, delta_max], computes the loss and returns
        the δ that minimizes it.

        Args:
            p_success: [N] array of estimated p(success|x) for training states
            y_true: [N] array of true labels (-1 for failure, 1 for success)
            p_failure: [N] array of estimated p(failure|x) for training states

        Returns:
            Tuple of:
                lambda_star: Fixed at 0.5
                delta_star: Optimal uncertainty half-width
                info: Dict with optimization details
        """
        lam = 0.5  # Fixed lambda
        w = self.config.w
        grid_size = self.config.delta_grid_size
        delta_min = self.config.delta_min
        delta_max = self.config.delta_max
        decision_rule = self.config.decision_rule

        # Grid search over δ ∈ [delta_min, delta_max]
        deltas = np.linspace(delta_min, delta_max, grid_size)

        # Debug: print distribution of inputs
        print(f"      [δ Debug] Optimizing δ with fixed λ=0.5, decision_rule={decision_rule}")
        print(f"      [δ Debug] y_true distribution: {np.sum(y_true == 1)} success, {np.sum(y_true == -1)} failure")
        print(f"      [δ Debug] p_success stats: min={p_success.min():.4f}, max={p_success.max():.4f}, mean={p_success.mean():.4f}, median={np.median(p_success):.4f}")
        if decision_rule == "two_sided":
            print(f"      [δ Debug] p_failure stats: min={p_failure.min():.4f}, max={p_failure.max():.4f}, mean={p_failure.mean():.4f}, median={np.median(p_failure):.4f}")

        best_delta = 0.05
        best_loss = float('inf')
        losses = []
        misclass_rates = []
        unknown_rates = []

        n_samples = len(y_true)

        for delta in deltas:
            # Apply decision rule based on config
            if decision_rule == "two_sided":
                predictions = apply_two_sided_rule(p_success, p_failure, lam, delta)
            else:
                predictions = apply_one_sided_rule(p_success, p_failure, lam, delta)

            unknown_mask = predictions == 0

            # Count misclassifications (only among confident predictions)
            confident_mask = ~unknown_mask
            n_confident = np.sum(confident_mask)

            if n_confident > 0:
                misclassified = np.sum(predictions[confident_mask] != y_true[confident_mask])
                misclass_rate = misclassified / n_confident
            else:
                # All unknown - no misclassifications but max unknown rate
                misclass_rate = 0.0

            # Unknown rate
            unknown_rate = np.sum(unknown_mask) / n_samples

            # Weighted loss
            loss = w * misclass_rate + (1 - w) * unknown_rate

            losses.append(loss)
            misclass_rates.append(misclass_rate)
            unknown_rates.append(unknown_rate)

            if loss < best_loss:
                best_loss = loss
                best_delta = delta

        info = {
            'optimize_mode': 'delta',
            'decision_rule': decision_rule,
            'deltas': deltas,
            'losses': np.array(losses),
            'misclass_rates': np.array(misclass_rates),
            'unknown_rates': np.array(unknown_rates),
            'best_loss': best_loss,
            'best_misclass_rate': misclass_rates[np.argmin(losses)],
            'best_unknown_rate': unknown_rates[np.argmin(losses)],
        }

        print(f"      [δ Debug] Best δ={best_delta:.4f}, loss={best_loss:.4f}, misclass={info['best_misclass_rate']:.4f}, unknown={info['best_unknown_rate']:.4f}")

        # Return (lambda=0.5, delta_star, info)
        return lam, best_delta, info

    def get_prediction_region(
        self,
        p_success: float,
        lambda_star: float,
        delta: float = None
    ) -> int:
        """
        Classify a single point based on its probability.

        Args:
            p_success: Estimated p(success|x) for the point
            lambda_star: Decision boundary
            delta: Uncertainty half-width (uses config.delta if None)

        Returns:
            Predicted label:
                1: SUCCESS (confident p > λ+δ)
               -1: FAILURE (confident p < λ-δ)
                0: UNKNOWN (uncertain, λ-δ ≤ p ≤ λ+δ)
        """
        if delta is None:
            delta = self.config.delta
        lower = lambda_star - delta
        upper = lambda_star + delta

        if p_success < lower:
            return -1  # FAILURE
        elif p_success > upper:
            return 1   # SUCCESS
        else:
            return 0   # UNKNOWN

    def get_prediction_regions_batch(
        self,
        p_success: np.ndarray,
        lambda_star: float,
        delta: float = None
    ) -> np.ndarray:
        """
        Classify a batch of points based on their probabilities.

        Args:
            p_success: [N] array of estimated p(success|x)
            lambda_star: Optimal decision boundary
            delta: Uncertainty half-width (uses config.delta if None)

        Returns:
            [N] array of predicted labels (-1, 0, or 1)
        """
        if delta is None:
            delta = self.config.delta
        lower = lambda_star - delta
        upper = lambda_star + delta

        predictions = np.zeros(len(p_success), dtype=int)
        predictions[p_success < lower] = -1  # FAILURE
        predictions[p_success > upper] = 1   # SUCCESS
        # Remaining are already 0 (UNKNOWN)

        return predictions
