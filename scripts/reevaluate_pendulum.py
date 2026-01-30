#!/usr/bin/env python3
"""
Re-evaluate Pendulum models from all epochs with configurable parameters.

This script allows re-evaluating trained models with different evaluation parameters
(attractor_radius, alpha_eval, num_mc_samples, batch_size) while keeping the trained
model weights and conformal parameters (lambda_star, delta) from training.

Usage:
    # DEFAULT MODE: Use stored metrics (no re-evaluation)
    python scripts/reevaluate_pendulum.py /path/to/training/output

    # RE-EVALUATION MODE: Change attractor radius
    python scripts/reevaluate_pendulum.py /path/to/training/output \
        --attractor_radius 0.15

    # Multiple params changed
    python scripts/reevaluate_pendulum.py /path/to/training/output \
        --attractor_radius 0.15 \
        --alpha_eval 0.05 \
        --num_mc_samples 20
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from omegaconf import OmegaConf

from adaptive_roa.flow_matching.pendulum.latent_conditional.flow_matcher import (
    PendulumLatentConditionalFlowMatcher,
)
from adaptive_roa.systems.pendulum import PendulumSystem
from adaptive_roa.conformal import ConformalConfig
from adaptive_roa.conformal.calibrator import Calibrator
from adaptive_roa.conformal.probability_estimator import ProbabilityEstimator
from adaptive_roa.adaptive.data_source import load_eval_states

# Import evaluate function from training script
from run_adaptive_pendulum import evaluate_full_roa_fast


def load_hydra_config(training_dir: Path) -> dict:
    """Load the Hydra config from training directory."""
    config_path = training_dir / ".hydra" / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Hydra config not found: {config_path}")

    # Register OmegaConf resolvers if needed
    from adaptive_roa.utils.env_config import get_net_id, get_exp_dir, get_data_dir

    if not OmegaConf.has_resolver("net_id"):
        OmegaConf.register_new_resolver("net_id", lambda: get_net_id())
    if not OmegaConf.has_resolver("exp_dir"):
        OmegaConf.register_new_resolver("exp_dir", lambda: get_exp_dir())
    if not OmegaConf.has_resolver("data_dir"):
        OmegaConf.register_new_resolver("data_dir", lambda: get_data_dir())
    # Register 'now' resolver (returns empty string since we don't need timestamp)
    if not OmegaConf.has_resolver("now"):
        OmegaConf.register_new_resolver("now", lambda fmt="": "")

    cfg = OmegaConf.load(config_path)
    return OmegaConf.to_container(cfg, resolve=True)


def find_epoch_dirs(training_dir: Path) -> list:
    """Find all epoch directories in training output."""
    epoch_dirs = sorted(
        [d for d in training_dir.iterdir() if d.is_dir() and re.match(r"epoch_\d+", d.name)],
        key=lambda x: int(re.search(r"\d+", x.name).group()),
    )
    return epoch_dirs


def load_checkpoint(epoch_dir: Path, device: str):
    """Load flow matcher from checkpoint."""
    import glob

    checkpoint_dir = epoch_dir / "checkpoints"
    ckpts = glob.glob(str(checkpoint_dir / "best*.ckpt"))

    if not ckpts:
        raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")

    return PendulumLatentConditionalFlowMatcher.load_from_checkpoint(
        ckpts[0], device=device
    )


def load_conformal_state(epoch_dir: Path) -> tuple:
    """Load lambda_star and delta from conformal_state.json."""
    conformal_state_path = epoch_dir / "conformal_state.json"
    if not conformal_state_path.exists():
        raise FileNotFoundError(f"Conformal state not found: {conformal_state_path}")

    with open(conformal_state_path) as f:
        state = json.load(f)

    return state["lambda_star"], state["delta_star"]


def recompute_qhat(
    flow_matcher,
    system,
    cal_set_file: str,
    lambda_star: float,
    delta: float,
    alpha_eval: float,
    num_mc_samples: int,
    attractor_radius: float,
    device: str,
    invalid_threshold: float = None,
    filter_invalid_by_conformal: bool = False,
) -> tuple:
    """Recompute q_hat using alpha_eval on calibration set.

    Args:
        invalid_threshold: If provided, exclude calibration points where
            p_invalid >= invalid_threshold from the non-conformity score computation.
        filter_invalid_by_conformal: If True, exclude calibration points where
            p_invalid >= (lambda_star - delta). This takes precedence over invalid_threshold.

    Returns:
        Tuple of (q_hat, n_cal_total, n_cal_used)
    """
    # Load calibration set
    X_cal, _, y_cal = load_eval_states(cal_set_file)
    n_cal_total = len(X_cal)

    # Create probability estimator with custom num_mc_samples
    prob_config = ConformalConfig(
        delta=delta,
        alpha=alpha_eval,
        num_mc_samples=num_mc_samples,
        mc_batch_size=1024,
        attractor_radius=attractor_radius,
    )
    prob_estimator = ProbabilityEstimator(flow_matcher, system, prob_config, device)

    # Estimate probabilities on calibration set
    X_cal_tensor = torch.tensor(X_cal, dtype=torch.float32, device=device)
    p_success, p_failure, p_invalid = prob_estimator.estimate(X_cal_tensor)

    # Convert to numpy if needed
    if hasattr(p_success, 'cpu'):
        p_success = p_success.cpu().numpy()
    if hasattr(p_failure, 'cpu'):
        p_failure = p_failure.cpu().numpy()
    if hasattr(p_invalid, 'cpu'):
        p_invalid = p_invalid.cpu().numpy()

    # Filter out invalid points if threshold is provided
    # filter_invalid_by_conformal takes precedence over invalid_threshold
    if filter_invalid_by_conformal:
        conformal_threshold = lambda_star - delta
        valid_mask = p_invalid < conformal_threshold
        p_success = p_success[valid_mask]
        p_failure = p_failure[valid_mask]
        y_cal = y_cal[valid_mask]
        n_cal_used = int(np.sum(valid_mask))
    elif invalid_threshold is not None:
        valid_mask = p_invalid < invalid_threshold
        p_success = p_success[valid_mask]
        p_failure = p_failure[valid_mask]
        y_cal = y_cal[valid_mask]
        n_cal_used = int(np.sum(valid_mask))
    else:
        n_cal_used = n_cal_total

    # Check if we have any valid calibration points
    if n_cal_used == 0:
        print("    WARNING: No valid calibration points after filtering!")
        return None, n_cal_total, 0

    # Create calibrator with alpha_eval and calibrate
    # Pendulum uses one_sided decision rule
    cal_config = ConformalConfig(
        delta=delta,
        alpha=alpha_eval,
        decision_rule="one_sided",
    )
    calibrator = Calibrator(cal_config)

    q_hat = calibrator.calibrate(
        p_success,
        y_cal,
        lambda_star,
        delta,
        p_failure=p_failure,
        verbose=False,
    )

    return q_hat, n_cal_total, n_cal_used


def save_epoch_results(output_dir: Path, epoch_num: int, metrics: dict,
                       lambda_star: float, delta: float, q_hat: float):
    """Save re-evaluation results for an epoch."""
    epoch_dir = output_dir / f"epoch_{epoch_num:03d}"
    epoch_dir.mkdir(parents=True, exist_ok=True)

    # Create results.json in the same format as training
    results = {
        "epoch": epoch_num,
        "lambda_star": lambda_star,
        "delta_star": delta,
        "q_hat_eval": q_hat,
        "full_roa": metrics,
    }

    with open(epoch_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    return epoch_dir


def main():
    parser = argparse.ArgumentParser(
        description="Re-evaluate Pendulum models with configurable parameters"
    )
    parser.add_argument(
        "training_dir",
        type=Path,
        help="Path to training output directory containing epoch_XXX folders",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Custom output directory (overrides auto-naming)",
    )
    parser.add_argument(
        "--attractor_radius",
        type=float,
        default=None,
        help="Attractor radius for success classification",
    )
    parser.add_argument(
        "--alpha_eval",
        type=float,
        default=None,
        help="Significance level for q_hat calibration",
    )
    parser.add_argument(
        "--num_mc_samples",
        type=int,
        default=None,
        help="Number of MC samples per point",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="GPU batch size for evaluation",
    )
    parser.add_argument(
        "--invalid_threshold",
        type=float,
        default=None,
        help="If provided, exclude calibration points with p_invalid >= threshold from q_hat computation",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device for evaluation",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-evaluation even with no parameter changes",
    )
    parser.add_argument(
        "--no_filter_invalid_by_conformal",
        action="store_true",
        help="Disable default filtering of calibration points where p_invalid >= (lambda_star - delta)",
    )
    args = parser.parse_args()

    training_dir = args.training_dir
    if not training_dir.exists():
        print(f"ERROR: Training directory not found: {training_dir}")
        return 1

    # Determine if any params changed
    params_changed = any([
        args.attractor_radius is not None,
        args.alpha_eval is not None,
        args.num_mc_samples is not None,
        args.batch_size is not None,
        args.invalid_threshold is not None,
    ])

    if not params_changed and not args.force:
        # DEFAULT MODE - no re-evaluation needed
        print("=" * 70)
        print("USING DEFAULT STORED METRICS")
        print("=" * 70)
        print("No evaluation parameters specified - using existing results.json")
        print(f"Default metrics are in: {training_dir}/epoch_XXX/results.json")
        print("")
        print("To re-evaluate, specify at least one parameter:")
        print("  --attractor_radius, --alpha_eval, --num_mc_samples, --batch_size")
        print("Or use --force to re-evaluate with training defaults")
        print("")
        print("To compile default metrics, run:")
        print(f"  python scripts/compile_adaptive_metrics.py {training_dir}")
        return 0

    # RE-EVALUATION MODE
    print("=" * 70)
    print("RE-EVALUATION MODE - Pendulum")
    print("=" * 70)
    print("Parameters (None = use training default):")
    print(f"  attractor_radius: {args.attractor_radius}")
    print(f"  alpha_eval: {args.alpha_eval}")
    print(f"  num_mc_samples: {args.num_mc_samples}")
    print(f"  batch_size: {args.batch_size}")
    print(f"  invalid_threshold: {args.invalid_threshold}")
    print(f"  device: {args.device}")

    # Load training config for defaults
    hydra_config = load_hydra_config(training_dir)

    # Set defaults from training config if not specified
    training_defaults = {
        "attractor_radius": hydra_config["conformal"]["attractor_radius"],
        "alpha_eval": hydra_config["conformal"]["alpha_eval"],
        "num_mc_samples_eval": hydra_config["conformal"]["num_mc_samples_eval"],
        "val_batch_size": hydra_config.get("val_batch_size", 2048),
    }

    attractor_radius = args.attractor_radius if args.attractor_radius is not None else training_defaults["attractor_radius"]
    alpha_eval = args.alpha_eval if args.alpha_eval is not None else training_defaults["alpha_eval"]
    num_mc_samples = args.num_mc_samples if args.num_mc_samples is not None else training_defaults["num_mc_samples_eval"]
    batch_size = args.batch_size if args.batch_size is not None else training_defaults["val_batch_size"]

    print("\nEffective parameters:")
    print(f"  attractor_radius: {attractor_radius}")
    print(f"  alpha_eval: {alpha_eval}")
    print(f"  num_mc_samples: {num_mc_samples}")
    print(f"  batch_size: {batch_size}")

    # Generate output directory name from changed parameters
    if args.output_dir:
        output_dir = args.output_dir
    else:
        name_parts = []
        if args.attractor_radius is not None:
            name_parts.append(f"radius_{args.attractor_radius}")
        if args.alpha_eval is not None:
            name_parts.append(f"alpha_{args.alpha_eval}")
        if args.num_mc_samples is not None:
            name_parts.append(f"mc_{args.num_mc_samples}")
        if args.batch_size is not None:
            name_parts.append(f"batch_{args.batch_size}")
        if args.invalid_threshold is not None:
            name_parts.append(f"invthresh_{args.invalid_threshold}")
        eval_name = "_".join(name_parts) if name_parts else "custom"
        output_dir = training_dir / "evaluations" / eval_name

    print(f"\nOutput directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find epoch directories
    epoch_dirs = find_epoch_dirs(training_dir)
    if not epoch_dirs:
        print(f"ERROR: No epoch directories found in {training_dir}")
        return 1

    # Reorder: even epochs first, then odd epochs
    even_epochs = [d for d in epoch_dirs if int(re.search(r"\d+", d.name).group()) % 2 == 0]
    odd_epochs = [d for d in epoch_dirs if int(re.search(r"\d+", d.name).group()) % 2 == 1]
    epoch_dirs = even_epochs + odd_epochs

    print(f"\nFound {len(epoch_dirs)} epochs to re-evaluate (even first, then odd)")

    # Initialize system
    system = PendulumSystem()

    # Get data file paths from config
    cal_set_file = hydra_config["data_source"]["cal_set_file"]
    test_set_file = hydra_config["data_source"]["test_set_file"]

    # Track per-epoch info for eval_config.json
    per_epoch_info = {}

    # Re-evaluate each epoch
    for epoch_dir in epoch_dirs:
        epoch_num = int(re.search(r"\d+", epoch_dir.name).group())
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch_num}")
        print(f"{'='*60}")

        # 1. Load checkpoint
        print("  Loading checkpoint...")
        flow_matcher = load_checkpoint(epoch_dir, args.device)

        # 2. Load conformal state (lambda_star, delta from training)
        print("  Loading conformal state...")
        lambda_star, delta = load_conformal_state(epoch_dir)
        print(f"    lambda_star: {lambda_star}, delta: {delta}")

        # 3. Recompute q_hat using alpha_eval on calibration set
        # Determine filtering mode: conformal (default) or fixed threshold
        filter_invalid_by_conformal = not args.no_filter_invalid_by_conformal
        conformal_threshold = lambda_star - delta if filter_invalid_by_conformal else None

        print(f"  Recomputing q_hat (alpha_eval={alpha_eval})...")
        q_hat, n_cal_total, n_cal_used = recompute_qhat(
            flow_matcher,
            system,
            cal_set_file,
            lambda_star,
            delta,
            alpha_eval,
            num_mc_samples,
            attractor_radius,
            args.device,
            invalid_threshold=args.invalid_threshold,
            filter_invalid_by_conformal=filter_invalid_by_conformal,
        )

        # Skip epoch if no valid calibration points
        if q_hat is None:
            print(f"  SKIPPING epoch {epoch_num}: No valid calibration points")
            continue

        if filter_invalid_by_conformal:
            print(f"    q_hat: {q_hat:.4f} (from {n_cal_used}/{n_cal_total} cal points, {n_cal_total - n_cal_used} excluded by conformal filter: p_invalid >= {conformal_threshold:.4f})")
        elif args.invalid_threshold is not None:
            print(f"    q_hat: {q_hat:.4f} (from {n_cal_used}/{n_cal_total} cal points, {n_cal_total - n_cal_used} excluded by invalid_threshold)")
        else:
            print(f"    q_hat: {q_hat:.4f} (from {n_cal_used} calibration points)")

        # 4. Run full ROA evaluation
        print("  Running full ROA evaluation...")
        metrics = evaluate_full_roa_fast(
            flow_matcher,
            system,
            test_set_file,
            num_mc_samples=num_mc_samples,
            batch_size=batch_size,
            lambda_star=lambda_star,
            delta=delta,
            q_hat=q_hat,
            attractor_radius=attractor_radius,
            device=args.device,
            output_file=None,
            verbose=False,
        )

        # 5. Save results
        epoch_output_dir = save_epoch_results(
            output_dir, epoch_num, metrics, lambda_star, delta, q_hat
        )
        print(f"  Results saved to: {epoch_output_dir}")

        # Print summary metrics
        conf_m = metrics.get("conformal_thresholds", {})
        print(f"  [lambda*+/-delta] F1={conf_m.get('f1', 0):.2%}, Acc={conf_m.get('accuracy', 0):.2%}, Sep%={conf_m.get('separatrix_pct', 0):.1%}")

        # Track info for eval_config
        per_epoch_info[f"epoch_{epoch_num:03d}"] = {
            "lambda_star": lambda_star,
            "delta_star": delta,
            "q_hat_recomputed": q_hat,
            "n_cal_total": n_cal_total,
            "n_cal_used": n_cal_used,
            "conformal_threshold": conformal_threshold,
        }

    # Save eval_config.json
    eval_config = {
        "source_training_dir": str(training_dir),
        "evaluation_timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        "mode": "re-evaluated",
        "parameters": {
            "attractor_radius": attractor_radius,
            "alpha_eval": alpha_eval,
            "num_mc_samples": num_mc_samples,
            "batch_size": batch_size,
            "invalid_threshold": args.invalid_threshold,
            "filter_invalid_by_conformal": not args.no_filter_invalid_by_conformal,
            "device": args.device,
        },
        "training_defaults": training_defaults,
        "per_epoch": per_epoch_info,
    }

    with open(output_dir / "eval_config.json", "w") as f:
        json.dump(eval_config, f, indent=2)

    print(f"\n{'='*70}")
    print("RE-EVALUATION COMPLETE")
    print(f"{'='*70}")
    print(f"Results saved to: {output_dir}")
    print("\nTo compile metrics and generate plots, run:")
    print(f"  python scripts/compile_adaptive_metrics.py {training_dir} --eval_dir {output_dir.name}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
