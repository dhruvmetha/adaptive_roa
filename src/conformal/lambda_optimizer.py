"""
Lambda/Delta Optimizer for Conformal Prediction.

Finds the optimal decision boundary via grid search:
- Mode "lambda": Optimize λ* with fixed δ
- Mode "delta": Optimize δ* with fixed λ=0.5
"""
import numpy as np
from typing import Tuple
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
        y_true: np.ndarray
    ) -> Tuple[float, float, dict]:
        """
        Find optimal parameters via grid search based on optimize_mode.

        Args:
            p_success: [N] array of estimated p(success|x) for training states
            y_true: [N] array of true labels (-1 for failure, 1 for success)

        Returns:
            Tuple of:
                lambda_star: Optimal λ (0.5 if mode="delta")
                delta_star: Optimal δ (config.delta if mode="lambda")
                info: Dict with optimization details
        """
        if self.config.optimize_mode == "delta":
            return self.optimize_delta(p_success, y_true)
        else:
            return self.optimize_lambda(p_success, y_true)

    def optimize_lambda(
        self,
        p_success: np.ndarray,
        y_true: np.ndarray
    ) -> Tuple[float, float, dict]:
        """
        Find optimal λ* via grid search.

        For each λ in [δ, 1-δ], computes the loss and returns the λ
        that minimizes it.

        Args:
            p_success: [N] array of estimated p(success|x) for training states
            y_true: [N] array of true labels (-1 for failure, 1 for success)

        Returns:
            Tuple of:
                lambda_star: Optimal decision boundary
                info: Dict with optimization details (losses, best_loss, etc.)
        """
        delta = self.config.delta
        w = self.config.w
        grid_size = self.config.lambda_grid_size

        # Grid search over λ ∈ [δ, 1-δ]
        # We need λ-δ ≥ 0 and λ+δ ≤ 1, so λ ∈ [δ, 1-δ]
        lambdas = np.linspace(delta, 1 - delta, grid_size)

        # Debug: print distribution of inputs
        print(f"      [λ Debug] y_true distribution: {np.sum(y_true == 1)} success, {np.sum(y_true == -1)} failure")
        print(f"      [λ Debug] p_success stats: min={p_success.min():.4f}, max={p_success.max():.4f}, mean={p_success.mean():.4f}, median={np.median(p_success):.4f}")

        best_lambda = 0.5
        best_loss = float('inf')
        losses = []
        misclass_rates = []
        unknown_rates = []

        n_samples = len(y_true)

        for lam in lambdas:
            # Classify each point based on current λ
            lower = lam - delta
            upper = lam + delta

            predictions = np.zeros(n_samples)
            unknown_mask = np.zeros(n_samples, dtype=bool)

            for i in range(n_samples):
                if p_success[i] < lower:
                    predictions[i] = -1  # FAILURE
                elif p_success[i] > upper:
                    predictions[i] = 1   # SUCCESS
                else:
                    predictions[i] = 0   # UNKNOWN
                    unknown_mask[i] = True

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
        y_true: np.ndarray
    ) -> Tuple[float, float, dict]:
        """
        Find optimal δ* via grid search with fixed λ=0.5.

        For each δ in [delta_min, delta_max], computes the loss and returns
        the δ that minimizes it.

        Args:
            p_success: [N] array of estimated p(success|x) for training states
            y_true: [N] array of true labels (-1 for failure, 1 for success)

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

        # Grid search over δ ∈ [delta_min, delta_max]
        deltas = np.linspace(delta_min, delta_max, grid_size)

        # Debug: print distribution of inputs
        print(f"      [δ Debug] Optimizing δ with fixed λ=0.5")
        print(f"      [δ Debug] y_true distribution: {np.sum(y_true == 1)} success, {np.sum(y_true == -1)} failure")
        print(f"      [δ Debug] p_success stats: min={p_success.min():.4f}, max={p_success.max():.4f}, mean={p_success.mean():.4f}, median={np.median(p_success):.4f}")

        best_delta = 0.05
        best_loss = float('inf')
        losses = []
        misclass_rates = []
        unknown_rates = []

        n_samples = len(y_true)

        for delta in deltas:
            # Classify each point based on λ=0.5 and current δ
            lower = lam - delta
            upper = lam + delta

            predictions = np.zeros(n_samples)
            unknown_mask = np.zeros(n_samples, dtype=bool)

            for i in range(n_samples):
                if p_success[i] < lower:
                    predictions[i] = -1  # FAILURE
                elif p_success[i] > upper:
                    predictions[i] = 1   # SUCCESS
                else:
                    predictions[i] = 0   # UNKNOWN
                    unknown_mask[i] = True

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
