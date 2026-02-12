"""
Main Conformal Predictor class.

Combines probability estimation, lambda optimization, and calibration
to provide predictions with coverage guarantees.
"""
import torch
import numpy as np
from typing import Union, Tuple, List, Set, Dict, Optional
from adaptive_roa.conformal.config import ConformalConfig
from adaptive_roa.conformal.probability_estimator import ProbabilityEstimator
from adaptive_roa.conformal.lambda_optimizer import LambdaOptimizer
from adaptive_roa.conformal.calibrator import Calibrator


class ConformalPredictor:
    """
    Conformal predictor for dynamical system classification.

    Provides three-way classification (SUCCESS/FAILURE/UNKNOWN) with
    coverage guarantees using conformal prediction framework.

    Workflow:
    1. fit(X_train, y_train, X_cal, y_cal):
       - Estimate probabilities for training and calibration data
       - Optimize λ* on training data
       - Calibrate q_hat on calibration data
    2. predict(X):
       - Estimate probabilities for new data
       - Return prediction sets with coverage guarantee

    Attributes:
        flow_matcher: Trained flow matching model
        system: Dynamical system instance
        config: ConformalConfig
        device: Device for computation
        lambda_star: Optimal decision boundary (set after fit)
        delta_star: Optimal uncertainty half-width (set after fit)
        q_hat: Calibration threshold (set after fit)
    """

    def __init__(
        self,
        flow_matcher,
        system,
        config: ConformalConfig,
        device: str = "cuda"
    ):
        """
        Initialize conformal predictor.

        Args:
            flow_matcher: Trained flow matcher (any system)
            system: Dynamical system with classify_attractor()
            config: ConformalConfig
            device: Device for computation
        """
        self.flow_matcher = flow_matcher
        self.system = system
        self.config = config
        self.device = device

        # Components
        self.prob_estimator = ProbabilityEstimator(
            flow_matcher, system, config, device
        )
        self.lambda_optimizer = LambdaOptimizer(config)
        self.calibrator = Calibrator(config)

        # State (set after fit)
        self.lambda_star: Optional[float] = None
        self.delta_star: Optional[float] = None
        self.q_hat: Optional[float] = None
        self.fit_info: Optional[Dict] = None

    def optimize_thresholds(
        self,
        X_train: Union[torch.Tensor, np.ndarray],
        y_train: Union[torch.Tensor, np.ndarray],
        verbose: bool = True
    ) -> Tuple[float, float, Dict]:
        """
        Optimize λ* and δ* on training data. NO q_hat calibration.

        This is the first step of the new two-phase fitting:
        1. optimize_thresholds() - find optimal λ*/δ* on training data
        2. calibrate_qhat() - calibrate q_hat on separate calibration data (e.g., D1)

        Args:
            X_train: [N_train, state_dim] training states
            y_train: [N_train] training labels (-1 or 1)
            verbose: Print optimization progress

        Returns:
            Tuple of (lambda_star, delta_star, optimization_info)
        """
        # Convert to numpy if needed
        if isinstance(y_train, torch.Tensor):
            y_train = y_train.cpu().numpy()

        optimize_mode = self.config.optimize_mode
        decision_rule = self.config.decision_rule

        if verbose:
            print("=" * 60)
            print("OPTIMIZING THRESHOLDS (λ*/δ*)")
            print("=" * 60)
            print(f"Training set: {len(y_train)} points")
            print(f"Optimization mode: {optimize_mode}")
            print(f"Decision rule: {decision_rule}")

        # Step 1: Estimate probabilities for training set
        if verbose:
            print(f"\n[1/2] Estimating probabilities for training set ({self.config.num_mc_samples} MC samples)...")
        p_train_success, p_train_failure, p_train_invalid = self.prob_estimator.estimate(X_train)

        # Step 2: Optimize λ* or δ* depending on mode
        p_failure_for_opt = p_train_failure if decision_rule == "two_sided" else None

        if optimize_mode == "delta":
            if verbose:
                print(f"[2/2] Optimizing δ* via grid search with fixed λ=0.5 ({self.config.delta_grid_size} points)...")
        elif optimize_mode == "joint":
            if verbose:
                print(f"[2/2] Optimizing λ* and δ* jointly via 2D grid search "
                      f"({self.config.lambda_grid_size}×{self.config.delta_grid_size} grid)...")
        else:
            if verbose:
                print(f"[2/2] Optimizing λ* via grid search with fixed δ={self.config.delta} ({self.config.lambda_grid_size} points)...")

        self.lambda_star, self.delta_star, opt_info = self.lambda_optimizer.optimize(
            p_train_success, y_train, p_failure=p_failure_for_opt, p_invalid=p_train_invalid
        )

        if verbose:
            print(f"      → λ* = {self.lambda_star:.4f}" + (" (fixed)" if optimize_mode == "delta" else ""))
            print(f"      → δ* = {self.delta_star:.4f}" + (" (fixed)" if optimize_mode == "lambda" else ""))
            print(f"      → Best loss = {opt_info['best_loss']:.4f}")
            print(f"      → Misclass rate = {opt_info['best_misclass_rate']:.2%}")
            print(f"      → Unknown rate = {opt_info['best_unknown_rate']:.2%}")
            print("=" * 60)
            print("THRESHOLD OPTIMIZATION COMPLETE (q_hat NOT yet calibrated)")
            print("=" * 60)

        return self.lambda_star, self.delta_star, opt_info

    def calibrate_qhat(
        self,
        X_cal: Union[torch.Tensor, np.ndarray],
        y_cal: Union[torch.Tensor, np.ndarray],
        verbose: bool = True
    ) -> float:
        """
        Calibrate q_hat on calibration data using stored λ*, δ*.

        Must call optimize_thresholds() first to set λ* and δ*.

        This is the second step of the new two-phase fitting:
        1. optimize_thresholds() - find optimal λ*/δ* on training data
        2. calibrate_qhat() - calibrate q_hat on separate calibration data (e.g., D1)

        Args:
            X_cal: [N_cal, state_dim] calibration states
            y_cal: [N_cal] calibration labels (-1 or 1)
            verbose: Print calibration progress

        Returns:
            q_hat (calibration threshold)
        """
        if self.lambda_star is None or self.delta_star is None:
            raise RuntimeError("Must call optimize_thresholds() before calibrate_qhat()")

        # Convert to numpy if needed
        if isinstance(y_cal, torch.Tensor):
            y_cal = y_cal.cpu().numpy()

        decision_rule = self.config.decision_rule

        if verbose:
            print("=" * 60)
            print("CALIBRATING q_hat")
            print("=" * 60)
            print(f"Calibration set: {len(y_cal)} points")
            print(f"Using λ* = {self.lambda_star:.4f}, δ* = {self.delta_star:.4f}")
            print(f"Decision rule: {decision_rule}")

        # Step 1: Estimate probabilities for calibration set
        if verbose:
            print(f"\n[1/2] Estimating probabilities for calibration set ({self.config.num_mc_samples} MC samples)...")
        p_cal_success, p_cal_failure, p_cal_invalid = self.prob_estimator.estimate(X_cal)

        # Step 2: Calibrate q_hat
        p_failure_for_cal = p_cal_failure if decision_rule == "two_sided" else None

        if verbose:
            print(f"[2/2] Calibrating q_hat (α={self.config.alpha}, coverage={1-self.config.alpha:.0%})...")
        self.q_hat = self.calibrator.calibrate(
            p_cal_success, y_cal, self.lambda_star, self.delta_star, p_failure_for_cal,
            verbose=verbose
        )
        if verbose:
            print(f"      → q_hat = {self.q_hat:.4f}")

        # Update fit_info
        self.fit_info = {
            'lambda_star': self.lambda_star,
            'delta_star': self.delta_star,
            'q_hat': self.q_hat,
            'n_train': 0,  # Not tracked in two-phase approach
            'n_cal': len(y_cal),
            'optimization_info': None,  # Already done in optimize_thresholds()
            'optimize_mode': self.config.optimize_mode,
            'decision_rule': decision_rule,
        }

        if verbose:
            print("=" * 60)
            print("q_hat CALIBRATION COMPLETE")
            print("=" * 60)

        return self.q_hat

    def fit(
        self,
        X_train: Union[torch.Tensor, np.ndarray],
        y_train: Union[torch.Tensor, np.ndarray],
        X_cal: Union[torch.Tensor, np.ndarray],
        y_cal: Union[torch.Tensor, np.ndarray],
        verbose: bool = True
    ) -> Dict:
        """
        Fit conformal predictor on labeled data.

        Steps:
        1. Estimate p(success|x) for training states
        2. Optimize λ* to minimize weighted loss
        3. Estimate p(success|x) for calibration states
        4. Calibrate q_hat for coverage guarantee

        Args:
            X_train: [N_train, state_dim] training states
            y_train: [N_train] training labels (-1 or 1)
            X_cal: [N_cal, state_dim] calibration states
            y_cal: [N_cal] calibration labels (-1 or 1)
            verbose: Print fitting progress

        Returns:
            Dict with fitting information
        """
        # Convert to numpy if needed
        if isinstance(y_train, torch.Tensor):
            y_train = y_train.cpu().numpy()
        if isinstance(y_cal, torch.Tensor):
            y_cal = y_cal.cpu().numpy()

        optimize_mode = self.config.optimize_mode
        decision_rule = self.config.decision_rule

        if verbose:
            print("=" * 60)
            print("FITTING CONFORMAL PREDICTOR")
            print("=" * 60)
            print(f"Training set: {len(y_train)} points")
            print(f"Calibration set: {len(y_cal)} points")
            print(f"Optimization mode: {optimize_mode}")
            print(f"Decision rule: {decision_rule}")

        # Step 1: Estimate probabilities for training set
        if verbose:
            print(f"\n[1/4] Estimating probabilities for training set ({self.config.num_mc_samples} MC samples)...")
        p_train_success, p_train_failure, p_train_invalid = self.prob_estimator.estimate(X_train)

        # Step 2: Optimize λ* or δ* depending on mode
        # Pass p_failure for two_sided decision rule (CartPole), None for one_sided (pendulum)
        p_failure_for_opt = p_train_failure if decision_rule == "two_sided" else None

        if optimize_mode == "delta":
            if verbose:
                print(f"[2/4] Optimizing δ* via grid search with fixed λ=0.5 ({self.config.delta_grid_size} points)...")
        elif optimize_mode == "joint":
            if verbose:
                print(f"[2/4] Optimizing λ* and δ* jointly via 2D grid search "
                      f"({self.config.lambda_grid_size}×{self.config.delta_grid_size} grid)...")
        else:
            if verbose:
                print(f"[2/4] Optimizing λ* via grid search with fixed δ={self.config.delta} ({self.config.lambda_grid_size} points)...")

        self.lambda_star, self.delta_star, opt_info = self.lambda_optimizer.optimize(
            p_train_success, y_train, p_failure=p_failure_for_opt, p_invalid=p_train_invalid
        )

        if verbose:
            print(f"      → λ* = {self.lambda_star:.4f}" + (" (fixed)" if optimize_mode == "delta" else ""))
            print(f"      → δ* = {self.delta_star:.4f}" + (" (fixed)" if optimize_mode == "lambda" else ""))
            print(f"      → Best loss = {opt_info['best_loss']:.4f}")
            print(f"      → Misclass rate = {opt_info['best_misclass_rate']:.2%}")
            print(f"      → Unknown rate = {opt_info['best_unknown_rate']:.2%}")

        # Step 3: Estimate probabilities for calibration set
        if verbose:
            print(f"[3/4] Estimating probabilities for calibration set...")
        p_cal_success, p_cal_failure, _ = self.prob_estimator.estimate(X_cal)

        # Step 4: Calibrate q_hat
        # Pass p_failure for two_sided decision rule (CartPole), None for one_sided (pendulum)
        p_failure_for_cal = p_cal_failure if decision_rule == "two_sided" else None
        
        if verbose:
            print(f"[4/4] Calibrating q_hat (α={self.config.alpha}, coverage={1-self.config.alpha:.0%})...")
        self.q_hat = self.calibrator.calibrate(
            p_cal_success, y_cal, self.lambda_star, self.delta_star, p_failure_for_cal,
            verbose=verbose
        )
        if verbose:
            print(f"      → q_hat = {self.q_hat:.4f}")

        # Store fit info
        self.fit_info = {
            'lambda_star': self.lambda_star,
            'delta_star': self.delta_star,
            'q_hat': self.q_hat,
            'n_train': len(y_train),
            'n_cal': len(y_cal),
            'optimization_info': opt_info,
            'optimize_mode': optimize_mode,
            'decision_rule': decision_rule,
        }

        if verbose:
            print("=" * 60)
            print("FITTING COMPLETE")
            print("=" * 60)

        return self.fit_info

    def fit_with_fixed_thresholds(
        self,
        X_cal: Union[torch.Tensor, np.ndarray],
        y_cal: Union[torch.Tensor, np.ndarray],
        lambda_star: float,
        delta_star: float,
        verbose: bool = True
    ) -> Dict:
        """
        Fit conformal predictor with fixed thresholds (skip optimization).

        Only calibrates q_hat using the fixed λ* and δ* values.
        Use this when you want to skip the threshold optimization step
        and use predetermined thresholds.

        Args:
            X_cal: [N_cal, state_dim] calibration states
            y_cal: [N_cal] calibration labels (-1 or 1)
            lambda_star: Fixed decision boundary
            delta_star: Fixed uncertainty half-width
            verbose: Print fitting progress

        Returns:
            Dict with fitting information
        """
        if isinstance(y_cal, torch.Tensor):
            y_cal = y_cal.cpu().numpy()

        decision_rule = self.config.decision_rule

        if verbose:
            print("=" * 60)
            print("FITTING WITH FIXED THRESHOLDS (no optimization)")
            print("=" * 60)
            print(f"Calibration set: {len(y_cal)} points")
            print(f"Fixed λ* = {lambda_star:.4f}")
            print(f"Fixed δ* = {delta_star:.4f}")
            print(f"Decision rule: {decision_rule}")

        # Set fixed thresholds
        self.lambda_star = lambda_star
        self.delta_star = delta_star

        # Step 1: Estimate probabilities for calibration set
        if verbose:
            print(f"\n[1/2] Estimating probabilities for calibration set ({self.config.num_mc_samples} MC samples)...")
        p_cal_success, p_cal_failure, _ = self.prob_estimator.estimate(X_cal)

        # Step 2: Calibrate q_hat with fixed thresholds
        p_failure_for_cal = p_cal_failure if decision_rule == "two_sided" else None

        if verbose:
            print(f"[2/2] Calibrating q_hat (α={self.config.alpha}, coverage={1-self.config.alpha:.0%})...")
        self.q_hat = self.calibrator.calibrate(
            p_cal_success, y_cal, self.lambda_star, self.delta_star, p_failure_for_cal,
            verbose=verbose
        )
        if verbose:
            print(f"      → q_hat = {self.q_hat:.4f}")

        # Store fit info
        self.fit_info = {
            'lambda_star': self.lambda_star,
            'delta_star': self.delta_star,
            'q_hat': self.q_hat,
            'n_train': 0,  # No training data used (no optimization)
            'n_cal': len(y_cal),
            'optimization_info': None,  # No optimization performed
            'optimize_mode': 'fixed',
            'decision_rule': decision_rule,
        }

        if verbose:
            print("=" * 60)
            print("FITTING COMPLETE")
            print("=" * 60)

        return self.fit_info

    def predict(
        self,
        X: Union[torch.Tensor, np.ndarray]
    ) -> Tuple[List[Set[int]], np.ndarray, np.ndarray]:
        """
        Make predictions with coverage guarantees.

        Args:
            X: [N, state_dim] states to predict

        Returns:
            Tuple of:
                prediction_sets: List of N prediction sets (each subset of {-1, 0, 1})
                p_success: [N] array of estimated p(success|x)
                p_failure: [N] array of estimated p(failure|x)
        """
        if self.lambda_star is None or self.q_hat is None:
            raise RuntimeError("ConformalPredictor must be fit before predicting")

        # Estimate probabilities
        p_success, p_failure, _ = self.prob_estimator.estimate(X)

        # Pass p_failure for two_sided decision rule
        p_failure_for_cal = p_failure if self.config.decision_rule == "two_sided" else None

        # Get prediction sets
        prediction_sets = self.calibrator.get_prediction_sets_batch(
            p_success, self.lambda_star, self.q_hat, self.delta_star, p_failure_for_cal
        )

        return prediction_sets, p_success, p_failure

    def select_uncertain(
        self,
        X: Union[torch.Tensor, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Select uncertain points that need simulation.

        Uncertain points have prediction sets with multiple labels,
        meaning we can't confidently classify them.

        Args:
            X: [N, state_dim] candidate states

        Returns:
            Tuple of:
                uncertain_mask: [N] boolean array (True = uncertain)
                uncertain_indices: Indices of uncertain points
                p_success: [N] array of estimated p(success|x)
                p_failure: [N] array of estimated p(failure|x)
        """
        if self.lambda_star is None or self.q_hat is None:
            raise RuntimeError("ConformalPredictor must be fit before selecting")

        # Estimate probabilities
        p_success, p_failure, _ = self.prob_estimator.estimate(X)

        # Pass p_failure for two_sided decision rule
        p_failure_for_cal = p_failure if self.config.decision_rule == "two_sided" else None

        # Get uncertainty mask
        uncertain_mask = self.calibrator.get_uncertain_mask(
            p_success, self.lambda_star, self.q_hat, self.delta_star, p_failure_for_cal
        )

        uncertain_indices = np.where(uncertain_mask)[0]

        return uncertain_mask, uncertain_indices, p_success, p_failure

    def evaluate(
        self,
        X: Union[torch.Tensor, np.ndarray],
        y_true: Union[torch.Tensor, np.ndarray],
        verbose: bool = True
    ) -> Dict:
        """
        Evaluate conformal predictor on test data.

        Computes coverage, prediction set sizes, and classification metrics.

        Args:
            X: [N, state_dim] test states
            y_true: [N] true labels
            verbose: Print evaluation results

        Returns:
            Dict with evaluation metrics
        """
        if self.lambda_star is None or self.q_hat is None:
            raise RuntimeError("ConformalPredictor must be fit before evaluating")

        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()

        # Get predictions
        prediction_sets, p_success, p_failure = self.predict(X)

        n = len(y_true)

        # Coverage: fraction of times true label is in prediction set
        coverage_count = sum(
            y_true[i] in prediction_sets[i]
            for i in range(n)
        )
        coverage = coverage_count / n

        # Prediction set sizes
        set_sizes = np.array([len(ps) for ps in prediction_sets])
        avg_set_size = np.mean(set_sizes)
        singleton_rate = np.mean(set_sizes == 1)  # Confident predictions

        # For confident predictions, compute accuracy
        confident_mask = set_sizes == 1
        n_confident = np.sum(confident_mask)

        if n_confident > 0:
            confident_preds = np.array([
                list(prediction_sets[i])[0] if len(prediction_sets[i]) == 1 else 0
                for i in range(n)
            ])
            correct_confident = np.sum(
                confident_preds[confident_mask] == y_true[confident_mask]
            )
            confident_accuracy = correct_confident / n_confident
        else:
            confident_accuracy = 0.0

        # Unknown rate
        unknown_rate = 1 - singleton_rate

        # F1 score (treating UNKNOWN as abstention)
        # Only compute among confident predictions
        if n_confident > 0:
            y_pred_confident = confident_preds[confident_mask]
            y_true_confident = y_true[confident_mask]

            # For binary classification (1 vs -1)
            tp = np.sum((y_pred_confident == 1) & (y_true_confident == 1))
            fp = np.sum((y_pred_confident == 1) & (y_true_confident == -1))
            fn = np.sum((y_pred_confident == -1) & (y_true_confident == 1))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        else:
            precision, recall, f1 = 0.0, 0.0, 0.0

        metrics = {
            'coverage': coverage,
            'avg_set_size': avg_set_size,
            'singleton_rate': singleton_rate,
            'unknown_rate': unknown_rate,
            'confident_accuracy': confident_accuracy,
            'n_confident': n_confident,
            'n_total': n,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'lambda_star': self.lambda_star,
            'delta_star': self.delta_star,
            'q_hat': self.q_hat,
        }

        if verbose:
            print("\n" + "=" * 60)
            print("CONFORMAL PREDICTOR EVALUATION")
            print("=" * 60)
            print(f"Test set size: {n}")
            print(f"λ* = {self.lambda_star:.4f}, δ* = {self.delta_star:.4f}, q_hat = {self.q_hat:.4f}")
            print("-" * 60)
            print(f"Coverage:           {coverage:.2%} (target: {1-self.config.alpha:.0%})")
            print(f"Avg set size:       {avg_set_size:.2f}")
            print(f"Confident rate:     {singleton_rate:.2%}")
            print(f"Unknown rate:       {unknown_rate:.2%}")
            print("-" * 60)
            print(f"Confident accuracy: {confident_accuracy:.2%} ({n_confident} points)")
            print(f"Precision:          {precision:.2%}")
            print(f"Recall:             {recall:.2%}")
            print(f"F1 Score:           {f1:.2%}")
            print("=" * 60)

        return metrics

    def get_state(self) -> Dict:
        """Get current state of the predictor (for saving)."""
        return {
            'lambda_star': self.lambda_star,
            'delta_star': self.delta_star,
            'q_hat': self.q_hat,
            'fit_info': self.fit_info,
            'config': {
                'delta': self.config.delta,
                'w': self.config.w,
                'alpha': self.config.alpha,
                'num_mc_samples': self.config.num_mc_samples,
                'attractor_radius': self.config.attractor_radius,
                'optimize_mode': self.config.optimize_mode,
                'decision_rule': self.config.decision_rule,
            }
        }

    def load_state(self, state: Dict):
        """Load state of the predictor (after loading)."""
        self.lambda_star = state['lambda_star']
        self.delta_star = state.get('delta_star', self.config.delta)
        self.q_hat = state['q_hat']
        self.fit_info = state.get('fit_info')
