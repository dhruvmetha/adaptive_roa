"""
Lambda/Delta Optimizer for Conformal Prediction.

Finds the optimal decision boundary via grid search:
- Mode "lambda": Optimize λ* with fixed δ
- Mode "delta": Optimize δ* with fixed λ=0.5

Two-Probability Mode (for CartPole):
When both success_rate and failure_rate are provided, uses:
- SUCCESS: success_rate > λ+δ
- FAILURE: (1 - failure_rate) < λ-δ  (equivalently: failure_rate > 1 - (λ-δ))
- SEPARATRIX: neither condition met

Points with separatrix majority (neither success_rate > 0.5 nor failure_rate > 0.5)
are excluded from optimization.
"""
import numpy as np
from typing import Tuple, Optional
from src.conformal.config import ConformalConfig


class LambdaOptimizer:
    """
    Find optimal decision boundary via grid search.

    Two modes:
    1. optimize_mode="lambda": Find λ* with fixed δ
       - Search λ ∈ [δ, 1-δ]
       - Decision: p < λ-δ → FAILURE, p > λ+δ → SUCCESS

    2. optimize_mode="delta": Find δ* with fixed λ=0.5
       - Search δ ∈ [delta_min, delta_max]
       - Decision: p < 0.5-δ → FAILURE, p > 0.5+δ → SUCCESS

    Two-Probability Mode (CartPole):
    When both success_rate and failure_rate are provided:
    - SUCCESS: success_rate > λ+δ
    - FAILURE: (1 - failure_rate) < λ-δ
    - Uses symmetric thresholds on different probability transformations

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
            p_failure: [N] array of estimated p(failure|x) for training states (optional)
                       If provided, uses two-probability mode for CartPole

        Returns:
            Tuple of:
                lambda_star: Optimal λ (0.5 if mode="delta")
                delta_star: Optimal δ (config.delta if mode="lambda")
                info: Dict with optimization details
        """
        if self.config.optimize_mode == "delta":
            return self.optimize_delta(p_success, y_true, p_failure)
        else:
            return self.optimize_lambda(p_success, y_true, p_failure)

    def optimize_lambda(
        self,
        p_success: np.ndarray,
        y_true: np.ndarray,
        p_failure: Optional[np.ndarray] = None
    ) -> Tuple[float, float, dict]:
        """
        Find optimal λ* via grid search.

        For each λ in [δ, 1-δ], computes the loss and returns the λ
        that minimizes it.

        Two modes:
        1. Single probability (p_failure=None):
           - SUCCESS: p_success > λ+δ
           - FAILURE: p_success < λ-δ

        2. Two probabilities (p_failure provided, CartPole):
           - Filter out separatrix-majority points (neither p_success > 0.5 nor p_failure > 0.5)
           - SUCCESS: p_success > λ+δ
           - FAILURE: (1 - p_failure) < λ-δ

        Args:
            p_success: [N] array of estimated p(success|x) for training states
            y_true: [N] array of true labels (-1 for failure, 1 for success)
            p_failure: [N] array of estimated p(failure|x) (optional, for CartPole)

        Returns:
            Tuple of:
                lambda_star: Optimal decision boundary
                delta: Fixed delta from config
                info: Dict with optimization details (losses, best_loss, etc.)
        """
        delta = self.config.delta
        w = self.config.w
        grid_size = self.config.lambda_grid_size

        two_prob_mode = p_failure is not None

        if two_prob_mode:
            # Two-probability mode (CartPole): filter out separatrix-majority points
            success_majority = p_success > 0.5
            failure_majority = p_failure > 0.5
            valid_mask = success_majority | failure_majority

            p_success_filtered = p_success[valid_mask]
            p_failure_filtered = p_failure[valid_mask]
            y_true_filtered = y_true[valid_mask]

            n_excluded = np.sum(~valid_mask)
            n_valid = np.sum(valid_mask)

            print(f"      [λ Debug] Two-probability mode (CartPole)")
            print(f"      [λ Debug] Excluded {n_excluded} separatrix-majority points, using {n_valid} points")
            print(f"      [λ Debug] y_true distribution: {np.sum(y_true_filtered == 1)} success, {np.sum(y_true_filtered == -1)} failure")
            print(f"      [λ Debug] p_success stats: min={p_success_filtered.min():.4f}, max={p_success_filtered.max():.4f}, mean={p_success_filtered.mean():.4f}")
            print(f"      [λ Debug] p_failure stats: min={p_failure_filtered.min():.4f}, max={p_failure_filtered.max():.4f}, mean={p_failure_filtered.mean():.4f}")
        else:
            # Single probability mode (original)
            p_success_filtered = p_success
            y_true_filtered = y_true
            print(f"      [λ Debug] Single-probability mode")
            print(f"      [λ Debug] y_true distribution: {np.sum(y_true == 1)} success, {np.sum(y_true == -1)} failure")
            print(f"      [λ Debug] p_success stats: min={p_success.min():.4f}, max={p_success.max():.4f}, mean={p_success.mean():.4f}, median={np.median(p_success):.4f}")

        # Grid search over λ ∈ [δ, 1-δ]
        # Need λ-δ ≥ 0 and λ+δ ≤ 1, so λ ∈ [δ, 1-δ]
        lambdas = np.linspace(delta, 1 - delta, grid_size)

        best_lambda = 0.5
        best_loss = float('inf')
        losses = []
        misclass_rates = []
        unknown_rates = []

        n_samples = len(y_true_filtered)

        for lam in lambdas:
            # Thresholds for two-prob mode
            upper = lam + delta  # For success: p_success > λ+δ
            lower = lam - delta  # For failure: (1 - p_failure) < λ-δ

            predictions = np.zeros(n_samples)
            unknown_mask = np.zeros(n_samples, dtype=bool)

            if two_prob_mode:
                # Two-probability mode:
                # SUCCESS: p_success > λ+δ
                # FAILURE: (1 - p_failure) < λ-δ
                for i in range(n_samples):
                    if p_success_filtered[i] > upper:
                        predictions[i] = 1   # SUCCESS
                    elif (1 - p_failure_filtered[i]) < lower:
                        predictions[i] = -1  # FAILURE
                    else:
                        predictions[i] = 0   # UNKNOWN
                        unknown_mask[i] = True
            else:
                # Single probability mode: use λ-δ and λ+δ
                for i in range(n_samples):
                    if p_success_filtered[i] < lower:
                        predictions[i] = -1  # FAILURE
                    elif p_success_filtered[i] > upper:
                        predictions[i] = 1   # SUCCESS
                    else:
                        predictions[i] = 0   # UNKNOWN
                        unknown_mask[i] = True

            # Count misclassifications (only among confident predictions)
            confident_mask = ~unknown_mask
            n_confident = np.sum(confident_mask)

            if n_confident > 0:
                misclassified = np.sum(predictions[confident_mask] != y_true_filtered[confident_mask])
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
            'two_prob_mode': two_prob_mode,
            'lambdas': lambdas,
            'losses': np.array(losses),
            'misclass_rates': np.array(misclass_rates),
            'unknown_rates': np.array(unknown_rates),
            'best_loss': best_loss,
            'best_misclass_rate': misclass_rates[np.argmin(losses)],
            'best_unknown_rate': unknown_rates[np.argmin(losses)],
        }

        if two_prob_mode:
            info['n_excluded_separatrix'] = int(n_excluded)
            info['n_valid'] = int(n_valid)

        # Return (lambda_star, delta, info) - delta is fixed from config
        return best_lambda, delta, info

    def optimize_delta(
        self,
        p_success: np.ndarray,
        y_true: np.ndarray,
        p_failure: Optional[np.ndarray] = None
    ) -> Tuple[float, float, dict]:
        """
        Find optimal δ* via grid search with fixed λ=0.5.

        For each δ in [delta_min, delta_max], computes the loss and returns
        the δ that minimizes it.

        Two modes:
        1. Single probability (p_failure=None):
           - SUCCESS: p_success > λ+δ
           - FAILURE: p_success < λ-δ

        2. Two probabilities (p_failure provided, CartPole):
           - Filter out separatrix-majority points
           - SUCCESS: p_success > λ+δ
           - FAILURE: (1 - p_failure) < λ-δ

        Args:
            p_success: [N] array of estimated p(success|x) for training states
            y_true: [N] array of true labels (-1 for failure, 1 for success)
            p_failure: [N] array of estimated p(failure|x) (optional, for CartPole)

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

        two_prob_mode = p_failure is not None

        if two_prob_mode:
            # Two-probability mode (CartPole): filter out separatrix-majority points
            success_majority = p_success > 0.5
            failure_majority = p_failure > 0.5
            valid_mask = success_majority | failure_majority

            p_success_filtered = p_success[valid_mask]
            p_failure_filtered = p_failure[valid_mask]
            y_true_filtered = y_true[valid_mask]

            n_excluded = np.sum(~valid_mask)
            n_valid = np.sum(valid_mask)

            print(f"      [δ Debug] Two-probability mode (CartPole) with fixed λ=0.5")
            print(f"      [δ Debug] Excluded {n_excluded} separatrix-majority points, using {n_valid} points")
            print(f"      [δ Debug] y_true distribution: {np.sum(y_true_filtered == 1)} success, {np.sum(y_true_filtered == -1)} failure")
            print(f"      [δ Debug] p_success stats: min={p_success_filtered.min():.4f}, max={p_success_filtered.max():.4f}, mean={p_success_filtered.mean():.4f}")
            print(f"      [δ Debug] p_failure stats: min={p_failure_filtered.min():.4f}, max={p_failure_filtered.max():.4f}, mean={p_failure_filtered.mean():.4f}")
        else:
            # Single probability mode (original)
            p_success_filtered = p_success
            y_true_filtered = y_true
            print(f"      [δ Debug] Optimizing δ with fixed λ=0.5")
            print(f"      [δ Debug] y_true distribution: {np.sum(y_true == 1)} success, {np.sum(y_true == -1)} failure")
            print(f"      [δ Debug] p_success stats: min={p_success.min():.4f}, max={p_success.max():.4f}, mean={p_success.mean():.4f}, median={np.median(p_success):.4f}")

        # Grid search over δ ∈ [delta_min, delta_max]
        deltas = np.linspace(delta_min, delta_max, grid_size)

        best_delta = 0.05
        best_loss = float('inf')
        losses = []
        misclass_rates = []
        unknown_rates = []

        n_samples = len(y_true_filtered)

        for delta in deltas:
            # Thresholds for two-prob mode
            upper = lam + delta  # For success: p_success > λ+δ
            lower = lam - delta  # For failure: (1 - p_failure) < λ-δ

            predictions = np.zeros(n_samples)
            unknown_mask = np.zeros(n_samples, dtype=bool)

            if two_prob_mode:
                # Two-probability mode:
                # SUCCESS: p_success > λ+δ
                # FAILURE: (1 - p_failure) < λ-δ
                for i in range(n_samples):
                    if p_success_filtered[i] > upper:
                        predictions[i] = 1   # SUCCESS
                    elif (1 - p_failure_filtered[i]) < lower:
                        predictions[i] = -1  # FAILURE
                    else:
                        predictions[i] = 0   # UNKNOWN
                        unknown_mask[i] = True
            else:
                # Single probability mode: use λ-δ and λ+δ
                for i in range(n_samples):
                    if p_success_filtered[i] < lower:
                        predictions[i] = -1  # FAILURE
                    elif p_success_filtered[i] > upper:
                        predictions[i] = 1   # SUCCESS
                    else:
                        predictions[i] = 0   # UNKNOWN
                        unknown_mask[i] = True

            # Count misclassifications (only among confident predictions)
            confident_mask = ~unknown_mask
            n_confident = np.sum(confident_mask)

            if n_confident > 0:
                misclassified = np.sum(predictions[confident_mask] != y_true_filtered[confident_mask])
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
            'two_prob_mode': two_prob_mode,
            'deltas': deltas,
            'losses': np.array(losses),
            'misclass_rates': np.array(misclass_rates),
            'unknown_rates': np.array(unknown_rates),
            'best_loss': best_loss,
            'best_misclass_rate': misclass_rates[np.argmin(losses)],
            'best_unknown_rate': unknown_rates[np.argmin(losses)],
        }

        if two_prob_mode:
            info['n_excluded_separatrix'] = int(n_excluded)
            info['n_valid'] = int(n_valid)

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

    def get_prediction_regions_batch_two_prob(
        self,
        p_success: np.ndarray,
        p_failure: np.ndarray,
        lambda_star: float,
        delta: float = None
    ) -> np.ndarray:
        """
        Classify a batch of points using two-probability mode (CartPole).

        Uses:
        - SUCCESS: p_success > λ+δ
        - FAILURE: (1 - p_failure) < λ-δ

        Args:
            p_success: [N] array of estimated p(success|x)
            p_failure: [N] array of estimated p(failure|x)
            lambda_star: Optimal decision boundary
            delta: Uncertainty half-width (uses config.delta if None)

        Returns:
            [N] array of predicted labels (-1, 0, or 1)
        """
        if delta is None:
            delta = self.config.delta
        upper = lambda_star + delta
        lower = lambda_star - delta

        predictions = np.zeros(len(p_success), dtype=int)
        # SUCCESS: p_success > λ+δ
        predictions[p_success > upper] = 1
        # FAILURE: (1 - p_failure) < λ-δ (and not already success)
        failure_mask = ((1 - p_failure) < lower) & (predictions != 1)
        predictions[failure_mask] = -1
        # Remaining are already 0 (UNKNOWN/SEPARATRIX)

        return predictions
