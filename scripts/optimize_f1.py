#!/usr/bin/env python3
"""
Find optimal lambda/delta to achieve target F1 scores with minimum separatrix%.

This script evaluates trained models and performs grid search over lambda and delta
to find parameters that achieve target F1 scores while minimizing separatrix percentage.
Also computes q_hat conformal prediction sets using the optimized thresholds.
Useful for model comparison and understanding separatrix sharpness.

Usage:
    # Find lambda/delta for 90% and 95% F1 (with q_hat conformal)
    python scripts/optimize_f1.py /path/to/training/output --target_f1 "0.90,0.95"

    # Single target
    python scripts/optimize_f1.py /path/to/training/output --target_f1 "0.90"

    # Evaluate specific epoch
    python scripts/optimize_f1.py /path/to/training/output --target_f1 "0.90" --epoch 5

    # Custom search ranges
    python scripts/optimize_f1.py /path/to/training/output \
        --target_f1 "0.90" \
        --lambda_range "0.4,0.6" \
        --delta_range "0.02,0.2"

    # With different attractor radius
    python scripts/optimize_f1.py /path/to/training/output \
        --target_f1 "0.90,0.95" \
        --attractor_radius 0.25
"""

import argparse
import glob
import importlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from adaptive_roa.adaptive_v2.eval.full_roa import (
    _classification_metrics_from_predictions,
    _predict_lambda_delta,
    _predict_qhat_prediction_sets,
    optimize_lambda_delta_for_f1_targets,
)
from adaptive_roa.adaptive_v2.eval.system_hooks import resolve_system_hook
from adaptive_roa.adaptive.data_source import load_eval_states
from adaptive_roa.conformal import ConformalConfig
from adaptive_roa.conformal.calibrator import Calibrator
from adaptive_roa.conformal.probability_estimator import ProbabilityEstimator
from adaptive_roa.flow_matching.base.checkpoint_utils import (
    load_hydra_config as _load_hydra_config_from_dir,
)

_SYSTEM_REGISTRY = {
    "PendulumSystem": (
        "adaptive_roa.systems.pendulum",
        "PendulumSystem",
        "adaptive_roa.flow_matching.pendulum.latent_conditional.flow_matcher",
        "PendulumLatentConditionalFlowMatcher",
    ),
    "CartPoleSystem": (
        "adaptive_roa.systems.cartpole",
        "CartPoleSystem",
        "adaptive_roa.flow_matching.cartpole.latent_conditional.flow_matcher",
        "CartPoleLatentConditionalFlowMatcher",
    ),
    "Quadrotor2DSystem": (
        "adaptive_roa.systems.quadrotor2d",
        "Quadrotor2DSystem",
        "adaptive_roa.flow_matching.quadrotor_2d.latent_conditional.flow_matcher",
        "Quadrotor2DLatentConditionalFlowMatcher",
    ),
    "Quadrotor3DSystem": (
        "adaptive_roa.systems.quadrotor3d",
        "Quadrotor3DSystem",
        "adaptive_roa.flow_matching.quadrotor_3d.latent_conditional.flow_matcher",
        "Quadrotor3DLatentConditionalFlowMatcher",
    ),
}


def _resolve_system_classes(hydra_config: dict):
    """Return (SystemClass, FlowMatcherClass, system_name) from Hydra config."""
    target = hydra_config.get("system", {}).get("_target_", "")
    class_name = target.rsplit(".", 1)[-1] if target else ""

    if class_name not in _SYSTEM_REGISTRY:
        supported = ", ".join(_SYSTEM_REGISTRY.keys())
        raise ValueError(
            f"Unknown system._target_: {target}\n"
            f"Supported systems: {supported}"
        )

    sys_mod, sys_cls, fm_mod, fm_cls = _SYSTEM_REGISTRY[class_name]
    SystemClass = getattr(importlib.import_module(sys_mod), sys_cls)
    FlowMatcherClass = getattr(importlib.import_module(fm_mod), fm_cls)
    return SystemClass, FlowMatcherClass, class_name


def load_hydra_config(training_dir: Path) -> dict:
    """Load the Hydra config from training directory (top-level .hydra/)."""
    cfg = _load_hydra_config_from_dir(training_dir)
    if cfg is None:
        raise FileNotFoundError(
            f"Hydra config not found in {training_dir} or parent directories"
        )
    return cfg


def find_epoch_dirs(training_dir: Path) -> list[Path]:
    """Find all epoch directories in training output."""
    return sorted(
        [d for d in training_dir.iterdir() if d.is_dir() and re.match(r"epoch_\d+", d.name)],
        key=lambda x: int(re.search(r"\d+", x.name).group()),
    )


def load_checkpoint(FlowMatcherClass, epoch_dir: Path, device: str):
    """Load flow matcher from best checkpoint in epoch directory."""
    checkpoint_dir = epoch_dir / "checkpoints"
    ckpts = glob.glob(str(checkpoint_dir / "best*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")
    return FlowMatcherClass.load_from_checkpoint(ckpts[0], device=device)


def load_epoch_threshold_state(epoch_dir: Path) -> tuple[float, float]:
    """Load lambda_star and delta from v2 epoch artifacts."""
    artifacts_path = epoch_dir / "artifacts_v2.json"
    if not artifacts_path.exists():
        raise FileNotFoundError(f"v2 artifacts not found: {artifacts_path}")

    with open(artifacts_path) as f:
        artifacts = json.load(f)

    ts = artifacts.get("threshold_state", {})
    lambda_star = ts.get("lambda_star")
    delta_star = ts.get("delta_star")
    if lambda_star is None or delta_star is None:
        raise ValueError(f"Missing threshold_state.lambda_star/delta_star in {artifacts_path}")

    return float(lambda_star), float(delta_star)


def estimate_probabilities(
    flow_matcher,
    system,
    data_file: str,
    num_mc_samples: int,
    attractor_radius: float,
    device: str,
    batch_size: int = 2048,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate p_success, p_failure, p_invalid on a dataset.

    Returns:
        Tuple of (X, p_success, p_failure, p_invalid, y_true)
    """
    X, _, y = load_eval_states(data_file)

    prob_config = ConformalConfig(
        delta=0.05,
        alpha=0.1,
        num_mc_samples=num_mc_samples,
        mc_batch_size=batch_size,
        attractor_radius=attractor_radius,
    )
    prob_estimator = ProbabilityEstimator(flow_matcher, system, prob_config, device)

    X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
    p_success, p_failure, p_invalid = prob_estimator.estimate(
        X_tensor,
        verbose=True,
    )

    return X, p_success, p_failure, p_invalid, y


def compute_training_metrics(
    p_success: np.ndarray,
    p_failure: np.ndarray,
    p_invalid: np.ndarray,
    y_true: np.ndarray,
    lambda_star: float,
    delta: float,
    decision_rule: str,
) -> dict:
    """Compute metrics using training lambda/delta."""
    pred, _ = _predict_lambda_delta(
        p_success, p_failure, p_invalid,
        lambda_star=lambda_star,
        delta=delta,
        decision_rule=decision_rule,
        invalid_threshold=None,
    )
    return _classification_metrics_from_predictions(pred, y_true)


def compute_qhat(
    p_success: np.ndarray,
    p_failure: np.ndarray,
    y_true: np.ndarray,
    lambda_star: float,
    delta: float,
    alpha: float,
    decision_rule: str,
) -> float:
    """Compute q_hat on calibration set."""
    cal_config = ConformalConfig(
        delta=delta,
        alpha=alpha,
        decision_rule=decision_rule,
    )
    calibrator = Calibrator(cal_config)
    q_hat = calibrator.calibrate(
        p_success,
        y_true,
        lambda_star,
        delta,
        p_failure=p_failure,
        verbose=False,
    )
    return q_hat


def compute_qhat_metrics(
    p_success: np.ndarray,
    p_failure: np.ndarray,
    p_invalid: np.ndarray,
    y_true: np.ndarray,
    lambda_star: float,
    delta: float,
    q_hat: float,
    decision_rule: str,
) -> dict:
    """Compute metrics using q_hat conformal prediction sets."""
    pred, extras = _predict_qhat_prediction_sets(
        p_success, p_failure, p_invalid, y_true,
        lambda_star=lambda_star,
        delta=delta,
        q_hat=q_hat,
        decision_rule=decision_rule,
        invalid_threshold=None,
    )
    metrics = _classification_metrics_from_predictions(pred, y_true)
    metrics.update(extras)
    return metrics


def save_results(
    output_dir: Path,
    epoch_num: int,
    target_f1s: list[float],
    optimization_results: dict,
    training_metrics: dict,
    training_lambda: float,
    training_delta: float,
    alpha: float,
) -> Path:
    """Save optimization results for an epoch."""
    epoch_dir = output_dir / f"epoch_{epoch_num:03d}"
    epoch_dir.mkdir(parents=True, exist_ok=True)

    artifacts = {
        "epoch": epoch_num,
        "training_thresholds": {
            "lambda_star": training_lambda,
            "delta_star": training_delta,
        },
        "training_metrics": training_metrics,
        "alpha": alpha,
        "f1_optimization": {
            "target_f1s": target_f1s,
            "results": optimization_results,
        },
        "source": "scripts/optimize_f1.py",
    }

    with open(epoch_dir / "f1_optimization.json", "w") as f:
        json.dump(artifacts, f, indent=2)

    return epoch_dir


def main():
    parser = argparse.ArgumentParser(
        description="Find optimal lambda/delta for target F1 scores"
    )
    parser.add_argument(
        "training_dir", type=Path,
        help="Path to training output directory containing epoch_XXX folders",
    )
    parser.add_argument(
        "--target_f1", type=str, required=True,
        help="Comma-separated F1 targets (e.g., '0.90,0.95')"
    )
    parser.add_argument(
        "--output_dir", type=Path, default=None,
        help="Custom output directory (default: training_dir/f1_optimization)"
    )
    parser.add_argument(
        "--attractor_radius", type=float, default=None,
        help="Attractor radius for success classification"
    )
    parser.add_argument(
        "--alpha", type=float, default=0.1,
        help="Significance level for q_hat calibration (default: 0.1)"
    )
    parser.add_argument(
        "--num_mc_samples", type=int, default=20,
        help="Number of MC samples per point (default: 20)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=2048,
        help="GPU batch size for MC sampling"
    )
    parser.add_argument(
        "--lambda_range", type=str, default="0.3,0.7",
        help="Lambda search range as 'min,max' (default: 0.3,0.7)"
    )
    parser.add_argument(
        "--delta_range", type=str, default="0.01,0.3",
        help="Delta search range as 'min,max' (default: 0.01,0.3)"
    )
    parser.add_argument(
        "--n_lambda_steps", type=int, default=41,
        help="Number of lambda grid points (default: 41)"
    )
    parser.add_argument(
        "--n_delta_steps", type=int, default=30,
        help="Number of delta grid points (default: 30)"
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0",
        help="Device for evaluation"
    )
    parser.add_argument(
        "--epoch", type=int, default=None,
        help="Specific epoch to evaluate (default: all epochs)"
    )
    parser.add_argument(
        "--max_epoch", type=int, default=None,
        help="Maximum epoch to evaluate"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Show tqdm progress bars"
    )
    args = parser.parse_args()

    training_dir = args.training_dir
    if not training_dir.exists():
        print(f"ERROR: Training directory not found: {training_dir}")
        return 1

    target_f1s = [float(x.strip()) for x in args.target_f1.split(",")]
    lambda_range = tuple(float(x) for x in args.lambda_range.split(","))
    delta_range = tuple(float(x) for x in args.delta_range.split(","))

    hydra_config = load_hydra_config(training_dir)
    SystemClass, FlowMatcherClass, system_class_name = _resolve_system_classes(hydra_config)
    system = SystemClass()
    hook = resolve_system_hook(system)
    decision_rule = hydra_config.get("conformal", {}).get("decision_rule", hook.decision_rule)

    conformal_cfg = hydra_config.get("conformal", {})
    attractor_radius = args.attractor_radius if args.attractor_radius is not None else \
        conformal_cfg.get("attractor_radius", hook.attractor_radius_default)

    print("=" * 70)
    print(f"F1 OPTIMIZATION - {system_class_name}")
    print("=" * 70)
    print(f"Target F1s: {target_f1s}")
    print(f"Lambda range: {lambda_range} ({args.n_lambda_steps} steps)")
    print(f"Delta range: {delta_range} ({args.n_delta_steps} steps)")
    print(f"Attractor radius: {attractor_radius}")
    print(f"MC samples: {args.num_mc_samples}")
    print(f"Alpha (for q_hat): {args.alpha}")
    print(f"Decision rule: {decision_rule}")

    data_source = hydra_config.get("data_source", {})
    cal_set_file = data_source.get("cal_set_file")
    test_set_file = data_source.get("test_set_file")
    if not test_set_file:
        print("ERROR: data_source.test_set_file missing from Hydra config")
        return 1

    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = training_dir / "f1_optimization"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    epoch_dirs = find_epoch_dirs(training_dir)
    if not epoch_dirs:
        print(f"ERROR: No epoch directories found in {training_dir}")
        return 1

    if args.epoch is not None:
        epoch_dirs = [d for d in epoch_dirs if int(re.search(r"\d+", d.name).group()) == args.epoch]
        if not epoch_dirs:
            print(f"ERROR: Epoch {args.epoch} not found")
            return 1
    elif args.max_epoch is not None:
        epoch_dirs = [d for d in epoch_dirs if int(re.search(r"\d+", d.name).group()) <= args.max_epoch]

    print(f"\nFound {len(epoch_dirs)} epochs to evaluate")

    all_results = {}

    for epoch_dir in epoch_dirs:
        epoch_num = int(re.search(r"\d+", epoch_dir.name).group())
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch_num}")
        print(f"{'='*60}")

        try:
            print("  Loading checkpoint...")
            flow_matcher = load_checkpoint(FlowMatcherClass, epoch_dir, args.device)

            print("  Loading training thresholds...")
            training_lambda, training_delta = load_epoch_threshold_state(epoch_dir)
            print(f"    Training: lambda={training_lambda:.4f}, delta={training_delta:.4f}")

            print("  Estimating probabilities on test set...")
            X_test, p_success_test, p_failure_test, p_invalid_test, y_test = estimate_probabilities(
                flow_matcher, system, test_set_file,
                args.num_mc_samples, attractor_radius, args.device, args.batch_size
            )

            print("  Computing training metrics...")
            training_metrics = compute_training_metrics(
                p_success_test, p_failure_test, p_invalid_test, y_test,
                training_lambda, training_delta, decision_rule
            )
            print(f"    Training F1={training_metrics['f1']:.2%}, Sep={training_metrics['separatrix_pct']:.1%}")

            print(f"  Optimizing for target F1s: {target_f1s}...")
            opt_results = optimize_lambda_delta_for_f1_targets(
                p_success_test, p_failure_test, p_invalid_test, y_test,
                target_f1s=target_f1s,
                decision_rule=decision_rule,
                lambda_range=lambda_range,
                delta_range=delta_range,
                n_lambda_steps=args.n_lambda_steps,
                n_delta_steps=args.n_delta_steps,
                invalid_threshold=None,
            )

            if cal_set_file:
                print("  Estimating probabilities on calibration set...")
                X_cal, p_success_cal, p_failure_cal, p_invalid_cal, y_cal = estimate_probabilities(
                    flow_matcher, system, cal_set_file,
                    args.num_mc_samples, attractor_radius, args.device, args.batch_size
                )

            for target in sorted(opt_results.keys(), key=lambda x: float(x), reverse=True):
                r = opt_results[target]
                opt_lambda = r["lambda_star"]
                opt_delta = r["delta"]

                r["training_metrics"] = compute_training_metrics(
                    p_success_test, p_failure_test, p_invalid_test, y_test,
                    opt_lambda, opt_delta, decision_rule
                )

                if cal_set_file:
                    r["q_hat"] = compute_qhat(
                        p_success_cal, p_failure_cal, y_cal,
                        opt_lambda, opt_delta, args.alpha, decision_rule
                    )
                    r["qhat_metrics"] = compute_qhat_metrics(
                        p_success_test, p_failure_test, p_invalid_test, y_test,
                        opt_lambda, opt_delta, r["q_hat"], decision_rule
                    )

            print("\n  " + "=" * 95)
            print("  RESULTS")
            print("  " + "=" * 95)
            print(f"  {'Target':<8} {'Lambda':<7} {'Delta':<7} {'F1':<7} {'Sep%':<7} │ {'q_hat':<7} {'qF1':<7} {'qSep%':<7} {'qCov':<7} │ {'Status'}")
            print("  " + "-" * 95)
            for target in sorted(opt_results.keys(), key=lambda x: float(x), reverse=True):
                r = opt_results[target]
                status = "OK" if r["attainable"] else "BEST"
                qhat_str = f"{r['q_hat']:.3f}" if "q_hat" in r else "N/A"
                qf1_str = f"{r['qhat_metrics']['f1']:.1%}" if "qhat_metrics" in r else "N/A"
                qsep_str = f"{r['qhat_metrics']['separatrix_pct']:.1f}" if "qhat_metrics" in r else "N/A"
                qcov_str = f"{r['qhat_metrics']['coverage']:.1%}" if "qhat_metrics" in r else "N/A"
                print(f"  {float(target):<8.0%} {r['lambda_star']:<7.3f} {r['delta']:<7.3f} "
                      f"{r['f1']:<7.1%} {r['separatrix_pct']:<7.1f} │ {qhat_str:<7} {qf1_str:<7} {qsep_str:<7} {qcov_str:<7} │ {status}")
            print("  " + "-" * 95)
            print(f"  {'Train':<8} {training_lambda:<7.3f} {training_delta:<7.3f} "
                  f"{training_metrics['f1']:<7.1%} {training_metrics['separatrix_pct']:<7.1f} │ (training thresholds)")

            epoch_output = save_results(
                output_dir, epoch_num, target_f1s, opt_results,
                training_metrics, training_lambda, training_delta, args.alpha
            )
            print(f"\n  Results saved to: {epoch_output}")

            all_results[f"epoch_{epoch_num:03d}"] = {
                "training_lambda": training_lambda,
                "training_delta": training_delta,
                "training_f1": training_metrics["f1"],
                "training_sep": training_metrics["separatrix_pct"],
                "optimized": opt_results,
            }

        except Exception as e:
            import traceback
            print(f"  ERROR: {e}")
            traceback.print_exc()
            continue

    summary = {
        "source_training_dir": str(training_dir),
        "system": system_class_name,
        "decision_rule": decision_rule,
        "target_f1s": target_f1s,
        "parameters": {
            "attractor_radius": attractor_radius,
            "alpha": args.alpha,
            "num_mc_samples": args.num_mc_samples,
            "lambda_range": lambda_range,
            "delta_range": delta_range,
            "n_lambda_steps": args.n_lambda_steps,
            "n_delta_steps": args.n_delta_steps,
        },
        "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        "per_epoch": all_results,
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*70}")
    print("OPTIMIZATION COMPLETE")
    print(f"{'='*70}")
    print(f"Summary saved to: {output_dir / 'summary.json'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())