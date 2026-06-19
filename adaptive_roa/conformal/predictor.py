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
        device: str = "cuda",
        probability_estimator=None,
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

        # Components — allow an injected estimator (e.g. a classifier) so the
        # predictor stays agnostic to how probabilities are produced.
        self.prob_estimator = (
            probability_estimator
            if probability_estimator is not None
            else ProbabilityEstimator(flow_matcher, system, config, device)
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
        X_val: Union[torch.Tensor, np.ndarray],
        y_val: Union[torch.Tensor, np.ndarray],
        verbose: bool = True
    ) -> Tuple[float, float, Dict]:
        """
        Optimize λ* and δ* on held-out val data. NO q_hat calibration.

        The caller is responsible for passing val-only data (data the FM
        never trained on). No internal split is performed here.

        When ``config.optimize_objective == "f1"``, the optimization maximizes F1
        subject to a minimum separatrix% using
        ``optimize_lambda_delta_for_f1_targets`` (joint λ+δ search).

        When ``config.optimize_objective == "loss"`` (default), the existing
        w-weighted loss grid search runs on the val data.

        Args:
            X_val: [N, state_dim] val states for threshold optimization
            y_val: [N] labels (-1 or 1)
            verbose: Print optimization progress

        Returns:
            Tuple of (lambda_star, delta_star, optimization_info)
        """
        # Convert to numpy if needed
        if isinstance(y_val, torch.Tensor):
            y_val = y_val.cpu().numpy()

        optimize_mode = self.config.optimize_mode
        decision_rule = self.config.decision_rule
        optimize_objective = self.config.optimize_objective

        if verbose:
            obj_labels = {
                "f1": "F1-based", "jstat": "J-statistic-based",
                "loss": "loss-based", "fixed": "fixed (no optimization)",
            }
            obj_label = obj_labels.get(optimize_objective, "loss-based")
            print("=" * 60)
            print(f"OPTIMIZING THRESHOLDS ({obj_label})")
            print("=" * 60)

        # Fast path: fixed thresholds — no MC sampling, no grid search
        if optimize_objective == "fixed":
            opt_info = self._optimize_fixed(y_val, verbose)
        else:
            if verbose:
                n_pos = int(np.sum(y_val == 1))
                n_neg = int(np.sum(y_val == -1))
                print(f"Val data: {len(y_val)} points (pos={n_pos}, neg={n_neg})")
                print(f"Optimization mode: {optimize_mode}")
                print(f"Decision rule: {decision_rule}")
                print(f"Objective: {optimize_objective}")

            # Step 1: Estimate probabilities for val data (one MC pass)
            if verbose:
                print(f"\n[1/2] Estimating probabilities for val data ({self.config.num_mc_samples} MC samples)...")
            p_success, p_failure, p_invalid = self.prob_estimator.estimate(X_val)

            # Step 2: Dispatch based on optimize_objective
            if optimize_objective == "f1":
                opt_info = self._optimize_f1(
                    p_success, p_failure, p_invalid, y_val, verbose
                )
            else:
                opt_info = self._optimize_loss(
                    p_success, p_failure, p_invalid, y_val, verbose
                )

        if verbose:
            print("=" * 60)
            print("THRESHOLD OPTIMIZATION COMPLETE (q_hat NOT yet calibrated)")
            print("=" * 60)

        return self.lambda_star, self.delta_star, opt_info

    def _optimize_fixed(self, y_val: np.ndarray, verbose: bool) -> Dict:
        """Use fixed thresholds from config — no MC sampling, no grid search."""
        self.lambda_star = self.config.fixed_lambda_star
        self.delta_star = self.config.fixed_delta_star

        if verbose:
            print(f"Using fixed thresholds: λ* = {self.lambda_star:.4f}, δ* = {self.delta_star:.4f}")

        return {
            "objective": "fixed",
            "best_loss": 0.0,
            "best_misclass_rate": 0.0,
            "best_unknown_rate": 0.0,
        }

    def _optimize_loss(
        self,
        p_success_val: np.ndarray,
        p_failure_val: np.ndarray,
        p_invalid_val: np.ndarray,
        y_val: np.ndarray,
        verbose: bool,
    ) -> Dict:
        """Run the w-weighted loss grid search on the val split.

        Works for both "loss" (misclass-based) and "jstat" (J-statistic-based)
        objectives — the LambdaOptimizer dispatches internally.
        """
        optimize_mode = self.config.optimize_mode
        decision_rule = self.config.decision_rule
        optimize_objective = self.config.optimize_objective

        p_failure_for_opt = p_failure_val if decision_rule == "two_sided" else None

        if optimize_mode == "delta":
            if verbose:
                print(f"[3/3] Optimizing δ* via grid search with fixed λ=0.5 ({self.config.delta_grid_size} points)...")
        elif optimize_mode == "joint":
            if verbose:
                print(f"[3/3] Optimizing λ* and δ* jointly via 2D grid search "
                      f"({self.config.lambda_grid_size}×{self.config.delta_grid_size} grid)...")
        else:
            if verbose:
                print(f"[3/3] Optimizing λ* via grid search with fixed δ={self.config.delta} ({self.config.lambda_grid_size} points)...")

        p_invalid_for_opt = p_invalid_val if self.config.use_p_invalid_veto else None

        self.lambda_star, self.delta_star, opt_info = self.lambda_optimizer.optimize(
            p_success_val, y_val, p_failure=p_failure_for_opt, p_invalid=p_invalid_for_opt
        )

        if verbose:
            if not self.config.use_p_invalid_veto:
                print("      (p_invalid veto disabled)")
            print(f"      → λ* = {self.lambda_star:.4f}" + (" (fixed)" if optimize_mode == "delta" else ""))
            print(f"      → δ* = {self.delta_star:.4f}" + (" (fixed)" if optimize_mode == "lambda" else ""))
            print(f"      → Best loss = {opt_info['best_loss']:.4f}")
            error_label = "1-J (Youden)" if optimize_objective == "jstat" else "Misclass rate"
            print(f"      → {error_label} = {opt_info['best_misclass_rate']:.4f}")
            print(f"      → Unknown rate = {opt_info['best_unknown_rate']:.2%}")

        return opt_info

    def _optimize_f1(
        self,
        p_success_val: np.ndarray,
        p_failure_val: np.ndarray,
        p_invalid_val: np.ndarray,
        y_val: np.ndarray,
        verbose: bool,
    ) -> Dict:
        """Run F1-based joint λ+δ optimization on the val split."""
        # Lazy import to avoid circular dependency:
        # conformal.predictor → adaptive_v2.eval.full_roa → adaptive_v2.__init__
        # → engine → conformal_threshold → conformal.__init__ → conformal.predictor
        from adaptive_roa.adaptive_v2.eval.full_roa import optimize_lambda_delta_for_f1_targets

        target_f1 = self.config.target_f1

        if verbose:
            print(f"[3/3] Optimizing λ* and δ* for F1 ≥ {target_f1:.2f} "
                  f"({self.config.lambda_grid_size}×{self.config.delta_grid_size} grid)...")

        results = optimize_lambda_delta_for_f1_targets(
            p_success=p_success_val,
            p_failure=p_failure_val,
            p_invalid=p_invalid_val,
            y_true=y_val,
            target_f1s=[target_f1],
            decision_rule=self.config.decision_rule,
            lambda_range=(self.config.delta_min, 1 - self.config.delta_min),
            delta_range=(self.config.delta_min, self.config.delta_max),
            n_lambda_steps=self.config.lambda_grid_size,
            n_delta_steps=self.config.delta_grid_size,
        )

        key = f"{target_f1:.2f}"
        best = results[key]

        self.lambda_star = best["lambda_star"]
        self.delta_star = best["delta"]

        if verbose:
            attainable_str = "YES" if best["attainable"] else "NO (using best available)"
            print(f"      → Target F1 ≥ {target_f1:.2f} attainable: {attainable_str}")
            print(f"      → λ* = {self.lambda_star:.4f}")
            print(f"      → δ* = {self.delta_star:.4f}")
            print(f"      → Achieved F1 = {best['f1']:.4f}")
            print(f"      → Separatrix% = {best['separatrix_pct']:.2%}")
            print(f"      → Precision = {best['precision']:.4f}")
            print(f"      → Recall = {best['recall']:.4f}")

        opt_info = {
            "objective": "f1",
            "target_f1": target_f1,
            "attainable": best["attainable"],
            "best_f1": best["f1"],
            "best_separatrix_pct": best["separatrix_pct"],
            "best_loss": 1.0 - best["f1"],  # for compatibility with callers expecting best_loss
            "best_misclass_rate": 1.0 - best.get("accuracy", 0.0),
            "best_unknown_rate": best["separatrix_pct"],
            "precision": best["precision"],
            "recall": best["recall"],
        }
        return opt_info

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
