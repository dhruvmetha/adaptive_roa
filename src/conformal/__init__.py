"""
Conformal Prediction module for flow matching uncertainty quantification.

This module provides conformal prediction with coverage guarantees for
classifying dynamical system trajectories into SUCCESS/FAILURE/UNKNOWN.

Main components:
- ConformalConfig: Configuration dataclass (instantiable via Hydra)
- ProbabilityEstimator: MC sampling for p(success|x) estimation
- LambdaOptimizer: Find optimal decision boundary λ*
- Calibrator: Compute q_hat for coverage guarantee
- ConformalPredictor: Main class combining all components
"""

from src.conformal.config import ConformalConfig
from src.conformal.probability_estimator import ProbabilityEstimator
from src.conformal.lambda_optimizer import LambdaOptimizer
from src.conformal.calibrator import Calibrator
from src.conformal.predictor import ConformalPredictor

__all__ = [
    "ConformalConfig",
    "ProbabilityEstimator",
    "LambdaOptimizer",
    "Calibrator",
    "ConformalPredictor",
]
