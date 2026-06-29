"""
Configuration dataclass for Conformal Prediction.

This class holds all hyperparameters for conformal prediction and can be
instantiated via Hydra from configs/conformal/default.yaml.
"""
from dataclasses import dataclass, field


@dataclass
class ConformalConfig:
    """
    Configuration for conformal prediction with coverage guarantees.

    Attributes:
        delta: Unknown region half-width. Points with p(success) in
               [λ-δ, λ+δ] are classified as UNKNOWN.
        w: Weight for misclassification vs unknown in loss function.
           Loss = w * misclass_rate + (1-w) * unknown_rate
           Higher w = penalize wrong predictions more than "I don't know".
        alpha: Significance level for coverage guarantee.
               α=0.1 means 90% of prediction sets contain the true label.
        num_mc_samples: Number of Monte Carlo samples (latent z vectors)
                       to estimate p(success|x).
        mc_batch_size: Batch size for MC sampling to manage GPU memory.
        attractor_radius: Radius passed to system.classify_attractor()
                         to determine if endpoint is in attractor basin.
        optimize_mode: "lambda" (optimize λ with fixed δ), "delta" (optimize δ
                      with fixed λ=0.5), or "joint" (optimize both via 2D grid search).
        decision_rule: Either "one_sided" (uses only p_success) or
                      "two_sided" (uses both p_success and p_failure).
                      Use "two_sided" for systems with explicit failure conditions (e.g. CartPole).
        lambda_grid_size: Number of grid points for searching optimal λ*.
        delta_grid_size: Number of grid points for searching optimal δ*.
        delta_min: Minimum δ value for grid search (when optimize_mode="delta").
        delta_max: Maximum δ value for grid search (when optimize_mode="delta").
    """
    # Decision boundary parameters
    delta: float = 0.05
    w: float = 0.9

    # Coverage parameters
    alpha: float = 0.1

    # Monte Carlo sampling
    num_mc_samples: int = 100
    mc_batch_size: int = 1024

    # Classification
    attractor_radius: float = 0.2

    # Optimization mode: "lambda" or "delta"
    optimize_mode: str = "lambda"

    # Decision rule: "one_sided" (p_success only) or "two_sided" (p_success and p_failure)
    decision_rule: str = "one_sided"

    # Lambda optimization (when optimize_mode="lambda")
    lambda_grid_size: int = 100

    # Delta optimization (when optimize_mode="delta")
    delta_grid_size: int = 100
    delta_min: float = 0.01
    delta_max: float = 0.49

    # p_invalid veto during threshold optimization
    use_p_invalid_veto: bool = True

    # Threshold optimization objective:
    #   "loss"  - w × misclass_rate + (1-w) × unknown_rate
    #   "jstat" - w × (1-J) + (1-w) × unknown_rate  (J = Youden's J-statistic)
    #   "f1"    - F1 ≥ target_f1, then minimize separatrix%
    #   "fixed" - no optimization, use fixed_lambda_star / fixed_delta_star
    optimize_objective: str = "loss"

    # Target F1 for F1-based optimization (only used when optimize_objective="f1")
    target_f1: float = 0.90

    # Fixed thresholds (only used when optimize_objective="fixed")
    fixed_lambda_star: float = 0.5
    fixed_delta_star: float = 0.1

    # Trajectory checking: when True and using local prediction mode,
    # classify trajectories at every timestep (first outcome wins)
    # instead of only checking the endpoint
    trajectory_checking: bool = False

    # Invalid endpoint refinement
    refine_invalids: bool = False
    refine_t_min: float = 0.7
    refine_t_max: float = 0.9
    refine_num_steps: int = 100
    refine_max_attempts: int = 5

    @staticmethod
    def _resolve_objective(c) -> str:
        """Resolve optimize_objective, with backward compat for threshold_mode."""
        obj = c.get("optimize_objective", None)
        if obj is not None:
            return obj
        # Backward compat: threshold_mode: fixed → optimize_objective: fixed
        tm = c.get("threshold_mode", "dynamic")
        return "fixed" if tm == "fixed" else "loss"

    def __post_init__(self):
        """Validate configuration parameters."""
        assert 0 < self.delta < 0.5, f"delta must be in (0, 0.5), got {self.delta}"
        assert 0 < self.w <= 1, f"w must be in (0, 1], got {self.w}"
        assert 0 < self.alpha < 1, f"alpha must be in (0, 1), got {self.alpha}"
        assert self.num_mc_samples > 0, f"num_mc_samples must be positive, got {self.num_mc_samples}"
        assert self.mc_batch_size > 0, f"mc_batch_size must be positive, got {self.mc_batch_size}"
        assert self.attractor_radius > 0, f"attractor_radius must be positive, got {self.attractor_radius}"
        assert self.optimize_mode in ["lambda", "delta", "joint"], f"optimize_mode must be 'lambda', 'delta', or 'joint', got {self.optimize_mode}"
        assert self.decision_rule in ["one_sided", "two_sided"], f"decision_rule must be 'one_sided' or 'two_sided', got {self.decision_rule}"
        assert self.lambda_grid_size > 1, f"lambda_grid_size must be > 1, got {self.lambda_grid_size}"
        assert self.delta_grid_size > 1, f"delta_grid_size must be > 1, got {self.delta_grid_size}"
        assert 0 < self.delta_min < self.delta_max < 0.5, f"delta_min/max must be in (0, 0.5) with min < max"
        assert self.optimize_objective in ["loss", "f1", "jstat", "fixed"], f"optimize_objective must be 'loss', 'f1', 'jstat', or 'fixed', got {self.optimize_objective}"
        assert 0 < self.target_f1 <= 1, f"target_f1 must be in (0, 1], got {self.target_f1}"
        if self.refine_invalids:
            assert 0 < self.refine_t_min < self.refine_t_max < 1.0, (
                f"refine_t_min/max must be in (0, 1) with min < max, "
                f"got [{self.refine_t_min}, {self.refine_t_max}]"
            )
            assert self.refine_num_steps > 0, f"refine_num_steps must be positive, got {self.refine_num_steps}"
            assert self.refine_max_attempts >= 1, f"refine_max_attempts must be >= 1, got {self.refine_max_attempts}"
