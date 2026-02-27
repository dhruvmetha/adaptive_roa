"""
Lambda/Delta Optimizer for Conformal Prediction.

Finds the optimal decision boundary via grid search:
- Mode "lambda": Optimize λ* with fixed δ
- Mode "delta": Optimize δ* with fixed λ=0.5
- Mode "joint": Optimize both λ* and δ* simultaneously via 2D grid search

Decision rules:
- "one_sided": Uses only p_success (default, for pendulum/mountain car)
- "two_sided": Uses both p_success and p_failure (for CartPole)

Points with high p_invalid (>= λ-δ) are treated as UNKNOWN, matching
the evaluation logic in full_roa.py.
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
    delta: float,
    p_invalid: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    One-sided decision rule using only p_success.

    - UNKNOWN if p_invalid >= λ - δ
    - SUCCESS if p_success > λ + δ
    - FAILURE if p_success < λ - δ
    - UNKNOWN otherwise

    Used for systems where failure is simply "not success" (pendulum, mountain car).
    """
    n = len(p_success)
    pred = np.zeros(n)

    failure_thresh = lambda_star - delta

    # Mark high-invalid points as UNKNOWN first
    if p_invalid is not None:
        invalid_mask = p_invalid >= failure_thresh
    else:
        invalid_mask = np.zeros(n, dtype=bool)

    non_invalid = ~invalid_mask
    pred[(p_success > lambda_star + delta) & non_invalid] = 1
    pred[(p_success < failure_thresh) & non_invalid] = -1
    return pred


def apply_two_sided_rule(
    p_success: np.ndarray,
    p_failure: np.ndarray,
    lambda_star: float,
    delta: float,
    p_invalid: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Two-sided decision rule using both p_success and p_failure.

    - UNKNOWN if p_invalid >= λ - δ
    - SUCCESS if p_success > λ + δ
    - FAILURE if (1 - p_failure) < λ - δ  (equivalently p_failure > 1 - (λ - δ))
    - UNKNOWN otherwise

    Used for systems with explicit failure conditions (CartPole).
    """
    n = len(p_success)
    pred = np.zeros(n)

    success_thresh = lambda_star + delta
    failure_thresh = lambda_star - delta

    # Mark high-invalid points as UNKNOWN first
    if p_invalid is not None:
        invalid_mask = p_invalid >= failure_thresh
    else:
        invalid_mask = np.zeros(n, dtype=bool)

    non_invalid = ~invalid_mask
    pred[(p_success > success_thresh) & non_invalid] = 1
    pred[((1.0 - p_failure) < failure_thresh) & non_invalid] = -1

    return pred


def compute_jstat(predictions: np.ndarray, y_true: np.ndarray, confident_mask: np.ndarray) -> float:
    """
    Compute Youden's J-statistic among confident predictions.

    J = Sensitivity + Specificity - 1 = TPR + TNR - 1

    Returns a value in [-1, 1] where 1 is perfect and 0 is random.
    Returns 0.0 if no confident predictions or if a class is absent.
    """
    y_pred = predictions[confident_mask]
    y_true_conf = y_true[confident_mask]

    n_pos = np.sum(y_true_conf == 1)
    n_neg = np.sum(y_true_conf == -1)

    if n_pos == 0 or n_neg == 0:
        return 0.0

    tpr = np.sum((y_pred == 1) & (y_true_conf == 1)) / n_pos
    tnr = np.sum((y_pred == -1) & (y_true_conf == -1)) / n_neg

    return tpr + tnr - 1.0


class LambdaOptimizer:
    """
    Find optimal decision boundary via grid search.

    Three modes:
    1. optimize_mode="lambda": Find λ* with fixed δ
       - Search λ ∈ [δ, 1-δ]

    2. optimize_mode="delta": Find δ* with fixed λ=0.5
       - Search δ ∈ [delta_min, delta_max]

    3. optimize_mode="joint": Find both λ* and δ* simultaneously
       - 2D grid search over λ ∈ [delta_min, 1-delta_min] and δ ∈ [delta_min, delta_max]
       - Only evaluates valid combinations where λ-δ > 0 and λ+δ < 1

    Two decision rules (config.decision_rule):
    1. "one_sided": Uses only p_success
       - SUCCESS if p_s > λ+δ, FAILURE if p_s < λ-δ
    2. "two_sided": Uses both p_success and p_failure
       - SUCCESS if p_s > λ+δ, FAILURE if (1-p_f) < λ-δ

    In both rules, points with p_invalid >= λ-δ are treated as UNKNOWN.

    Loss functions (selected by config.optimize_objective):
    - "loss":  Loss = w × MisclassificationRate + (1-w) × UnknownRate
    - "jstat": Loss = w × (1 - J_statistic) + (1-w) × UnknownRate
      where J = TPR + TNR - 1 (Youden's J-statistic, class-balanced)

    Attributes:
        config: ConformalConfig with optimization parameters
    """

    def __init__(self, config: ConformalConfig):
        self.config = config

    def _compute_loss(
        self,
        predictions: np.ndarray,
        y_true: np.ndarray,
        n_samples: int,
    ) -> Tuple[float, float, float]:
        """
        Compute the optimization loss for a given set of predictions.

        Dispatches based on config.optimize_objective:
        - "loss":  w * misclass_rate + (1-w) * unknown_rate
        - "jstat": w * (1 - J_statistic) + (1-w) * unknown_rate

        Returns:
            Tuple of (loss, error_term, unknown_rate) where error_term is
            misclass_rate for "loss" or (1 - J) for "jstat".
        """
        w = self.config.w
        objective = self.config.optimize_objective

        unknown_mask = predictions == 0
        confident_mask = ~unknown_mask
        n_confident = np.sum(confident_mask)
        unknown_rate = np.sum(unknown_mask) / n_samples

        if objective == "jstat":
            if n_confident > 0:
                j = compute_jstat(predictions, y_true, confident_mask)
                error_term = 1.0 - j  # in [0, 2], 0 is perfect
            else:
                error_term = 1.0  # no confident preds → J=0 → error=1
        else:
            # Default: misclassification rate
            if n_confident > 0:
                misclassified = np.sum(predictions[confident_mask] != y_true[confident_mask])
                error_term = misclassified / n_confident
            else:
                error_term = 0.0

        loss = w * error_term + (1 - w) * unknown_rate
        return loss, error_term, unknown_rate

    def optimize(
        self,
        p_success: np.ndarray,
        y_true: np.ndarray,
        p_failure: Optional[np.ndarray] = None,
        p_invalid: Optional[np.ndarray] = None,
    ) -> Tuple[float, float, dict]:
        """
        Find optimal parameters via grid search based on optimize_mode.

        Args:
            p_success: [N] array of estimated p(success|x) for training states
            y_true: [N] array of true labels (-1 for failure, 1 for success)
            p_failure: [N] array of estimated p(failure|x). Required if decision_rule="two_sided".
            p_invalid: [N] array of estimated p(invalid|x). Points with high p_invalid
                       are treated as UNKNOWN during optimization.

        Returns:
            Tuple of:
                lambda_star: Optimal λ (0.5 if mode="delta")
                delta_star: Optimal δ (config.delta if mode="lambda")
                info: Dict with optimization details
        """
        if self.config.decision_rule == "two_sided" and p_failure is None:
            raise ValueError("p_failure is required when decision_rule='two_sided'")

        if p_failure is None:
            p_failure = np.zeros_like(p_success)

        if self.config.optimize_mode == "delta":
            return self.optimize_delta(p_success, y_true, p_failure, p_invalid)
        elif self.config.optimize_mode == "joint":
            return self.optimize_joint(p_success, y_true, p_failure, p_invalid)
        else:
            return self.optimize_lambda(p_success, y_true, p_failure, p_invalid)

    def optimize_lambda(
        self,
        p_success: np.ndarray,
        y_true: np.ndarray,
        p_failure: np.ndarray,
        p_invalid: Optional[np.ndarray] = None,
    ) -> Tuple[float, float, dict]:
        """
        Find optimal λ* via grid search.

        For each λ in [δ, 1-δ], computes the loss and returns the λ
        that minimizes it.
        """
        delta = self.config.delta
        grid_size = self.config.lambda_grid_size
        decision_rule = self.config.decision_rule
        objective = self.config.optimize_objective

        lambdas = np.linspace(delta, 1 - delta, grid_size)

        print(f"      [λ opt] objective={objective}, decision_rule={decision_rule}, N={len(y_true)}, "
              f"success={np.sum(y_true == 1)}, failure={np.sum(y_true == -1)}")
        print(f"      [λ opt] p_success: min={p_success.min():.4f}, max={p_success.max():.4f}, "
              f"mean={p_success.mean():.4f}, median={np.median(p_success):.4f}")
        if p_invalid is not None:
            print(f"      [λ opt] p_invalid: min={p_invalid.min():.4f}, max={p_invalid.max():.4f}, "
                  f"mean={p_invalid.mean():.4f}, median={np.median(p_invalid):.4f}")

        best_lambda = 0.5
        best_loss = float('inf')
        losses = []
        error_terms = []
        unknown_rates = []

        n_samples = len(y_true)

        for lam in lambdas:
            if decision_rule == "two_sided":
                predictions = apply_two_sided_rule(p_success, p_failure, lam, delta, p_invalid)
            else:
                predictions = apply_one_sided_rule(p_success, p_failure, lam, delta, p_invalid)

            loss, error_term, unknown_rate = self._compute_loss(predictions, y_true, n_samples)

            losses.append(loss)
            error_terms.append(error_term)
            unknown_rates.append(unknown_rate)

            if loss < best_loss:
                best_loss = loss
                best_lambda = lam

        best_idx = np.argmin(losses)
        info = {
            'optimize_mode': 'lambda',
            'optimize_objective': objective,
            'decision_rule': decision_rule,
            'lambdas': lambdas,
            'losses': np.array(losses),
            'misclass_rates': np.array(error_terms),
            'unknown_rates': np.array(unknown_rates),
            'best_loss': best_loss,
            'best_misclass_rate': error_terms[best_idx],
            'best_unknown_rate': unknown_rates[best_idx],
        }

        return best_lambda, delta, info

    def optimize_delta(
        self,
        p_success: np.ndarray,
        y_true: np.ndarray,
        p_failure: np.ndarray,
        p_invalid: Optional[np.ndarray] = None,
    ) -> Tuple[float, float, dict]:
        """
        Find optimal δ* via grid search with fixed λ=0.5.

        For each δ in [delta_min, delta_max], computes the loss and returns
        the δ that minimizes it.
        """
        lam = 0.5
        grid_size = self.config.delta_grid_size
        delta_min = self.config.delta_min
        delta_max = self.config.delta_max
        decision_rule = self.config.decision_rule
        objective = self.config.optimize_objective

        deltas = np.linspace(delta_min, delta_max, grid_size)

        print(f"      [δ opt] objective={objective}, λ=0.5, decision_rule={decision_rule}, N={len(y_true)}, "
              f"success={np.sum(y_true == 1)}, failure={np.sum(y_true == -1)}")
        print(f"      [δ opt] p_success: min={p_success.min():.4f}, max={p_success.max():.4f}, "
              f"mean={p_success.mean():.4f}, median={np.median(p_success):.4f}")
        if p_invalid is not None:
            print(f"      [δ opt] p_invalid: min={p_invalid.min():.4f}, max={p_invalid.max():.4f}, "
                  f"mean={p_invalid.mean():.4f}, median={np.median(p_invalid):.4f}")

        best_delta = 0.05
        best_loss = float('inf')
        losses = []
        error_terms = []
        unknown_rates = []

        n_samples = len(y_true)

        for delta in deltas:
            if decision_rule == "two_sided":
                predictions = apply_two_sided_rule(p_success, p_failure, lam, delta, p_invalid)
            else:
                predictions = apply_one_sided_rule(p_success, p_failure, lam, delta, p_invalid)

            loss, error_term, unknown_rate = self._compute_loss(predictions, y_true, n_samples)

            losses.append(loss)
            error_terms.append(error_term)
            unknown_rates.append(unknown_rate)

            if loss < best_loss:
                best_loss = loss
                best_delta = delta

        best_idx = np.argmin(losses)
        error_label = "1-J" if objective == "jstat" else "misclass"
        info = {
            'optimize_mode': 'delta',
            'optimize_objective': objective,
            'decision_rule': decision_rule,
            'deltas': deltas,
            'losses': np.array(losses),
            'misclass_rates': np.array(error_terms),
            'unknown_rates': np.array(unknown_rates),
            'best_loss': best_loss,
            'best_misclass_rate': error_terms[best_idx],
            'best_unknown_rate': unknown_rates[best_idx],
        }

        print(f"      [δ opt] Best δ={best_delta:.4f}, loss={best_loss:.4f}, "
              f"{error_label}={info['best_misclass_rate']:.4f}, unknown={info['best_unknown_rate']:.4f}")

        return lam, best_delta, info

    def optimize_joint(
        self,
        p_success: np.ndarray,
        y_true: np.ndarray,
        p_failure: np.ndarray,
        p_invalid: Optional[np.ndarray] = None,
    ) -> Tuple[float, float, dict]:
        """
        Find optimal λ* and δ* jointly via 2D grid search.

        Searches over all valid (λ, δ) pairs where λ-δ > 0 and λ+δ < 1.
        Uses lambda_grid_size × delta_grid_size evaluations.
        """
        lambda_grid_size = self.config.lambda_grid_size
        delta_grid_size = self.config.delta_grid_size
        delta_min = self.config.delta_min
        delta_max = self.config.delta_max
        decision_rule = self.config.decision_rule
        objective = self.config.optimize_objective

        lambdas = np.linspace(delta_min, 1 - delta_min, lambda_grid_size)
        deltas = np.linspace(delta_min, delta_max, delta_grid_size)

        print(f"      [joint opt] objective={objective}, decision_rule={decision_rule}, N={len(y_true)}, "
              f"success={np.sum(y_true == 1)}, failure={np.sum(y_true == -1)}")
        print(f"      [joint opt] λ grid: {lambda_grid_size} points in [{delta_min:.3f}, {1-delta_min:.3f}]")
        print(f"      [joint opt] δ grid: {delta_grid_size} points in [{delta_min:.3f}, {delta_max:.3f}]")
        print(f"      [joint opt] p_success: min={p_success.min():.4f}, max={p_success.max():.4f}, "
              f"mean={p_success.mean():.4f}, median={np.median(p_success):.4f}")
        if p_invalid is not None:
            print(f"      [joint opt] p_invalid: min={p_invalid.min():.4f}, max={p_invalid.max():.4f}, "
                  f"mean={p_invalid.mean():.4f}, median={np.median(p_invalid):.4f}")

        best_lambda = 0.5
        best_delta = 0.05
        best_loss = float('inf')
        best_error_term = 0.0
        best_unknown_rate = 0.0

        n_samples = len(y_true)
        n_evaluated = 0

        # 2D grid: store losses for diagnostics
        loss_grid = np.full((lambda_grid_size, delta_grid_size), np.nan)

        for i, lam in enumerate(lambdas):
            for j, delta in enumerate(deltas):
                # Skip invalid combinations
                if lam - delta <= 0 or lam + delta >= 1:
                    continue

                n_evaluated += 1

                if decision_rule == "two_sided":
                    predictions = apply_two_sided_rule(p_success, p_failure, lam, delta, p_invalid)
                else:
                    predictions = apply_one_sided_rule(p_success, p_failure, lam, delta, p_invalid)

                loss, error_term, unknown_rate = self._compute_loss(predictions, y_true, n_samples)
                loss_grid[i, j] = loss

                if loss < best_loss:
                    best_loss = loss
                    best_lambda = lam
                    best_delta = delta
                    best_error_term = error_term
                    best_unknown_rate = unknown_rate

        error_label = "1-J" if objective == "jstat" else "misclass"
        info = {
            'optimize_mode': 'joint',
            'optimize_objective': objective,
            'decision_rule': decision_rule,
            'lambdas': lambdas,
            'deltas': deltas,
            'loss_grid': loss_grid,
            'n_evaluated': n_evaluated,
            'best_loss': best_loss,
            'best_misclass_rate': best_error_term,
            'best_unknown_rate': best_unknown_rate,
        }

        print(f"      [joint opt] Evaluated {n_evaluated}/{lambda_grid_size * delta_grid_size} valid (λ,δ) pairs")
        print(f"      [joint opt] Best λ={best_lambda:.4f}, δ={best_delta:.4f}, loss={best_loss:.4f}, "
              f"{error_label}={best_error_term:.4f}, unknown={best_unknown_rate:.4f}")

        return best_lambda, best_delta, info

    def get_prediction_region(
        self,
        p_success: float,
        lambda_star: float,
        delta: float = None
    ) -> int:
        """
        Classify a single point based on its probability.

        Returns:
            1: SUCCESS, -1: FAILURE, 0: UNKNOWN
        """
        if delta is None:
            delta = self.config.delta
        lower = lambda_star - delta
        upper = lambda_star + delta

        if p_success < lower:
            return -1
        elif p_success > upper:
            return 1
        else:
            return 0

    def get_prediction_regions_batch(
        self,
        p_success: np.ndarray,
        lambda_star: float,
        delta: float = None
    ) -> np.ndarray:
        """
        Classify a batch of points based on their probabilities.

        Returns:
            [N] array of predicted labels (-1, 0, or 1)
        """
        if delta is None:
            delta = self.config.delta
        lower = lambda_star - delta
        upper = lambda_star + delta

        predictions = np.zeros(len(p_success), dtype=int)
        predictions[p_success < lower] = -1
        predictions[p_success > upper] = 1

        return predictions
