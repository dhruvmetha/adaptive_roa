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

from adaptive_roa.conformal.config import ConformalConfig
from adaptive_roa.conformal.probability_estimator import ProbabilityEstimator
from adaptive_roa.conformal.lambda_optimizer import LambdaOptimizer
from adaptive_roa.conformal.calibrator import Calibrator
from adaptive_roa.conformal.predictor import ConformalPredictor
from adaptive_roa.conformal.refinement import RefinementStats, refine_invalid_endpoints

__all__ = [
    "ConformalConfig",
    "ProbabilityEstimator",
    "LambdaOptimizer",
    "Calibrator",
    "ConformalPredictor",
    "RefinementStats",
    "refine_invalid_endpoints",
]
