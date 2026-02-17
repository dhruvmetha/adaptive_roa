#!/usr/bin/env python3
"""
Unified re-evaluation script for all systems (Pendulum, CartPole, Quadrotor2D, Quadrotor3D).

Re-evaluates trained models from all epochs with configurable parameters.
The system type is auto-detected from the training directory's Hydra config.

Usage:
    # DEFAULT MODE: Use stored metrics (no re-evaluation)
    python scripts/reevaluate.py /path/to/training/output

    # RE-EVALUATION MODE: Change attractor radius
    python scripts/reevaluate.py /path/to/training/output --attractor_radius 0.25

    # Multiple params changed
    python scripts/reevaluate.py /path/to/training/output \
        --attractor_radius 0.25 --alpha_eval 0.05 --num_mc_samples 20

    # Evaluate only a specific epoch
    python scripts/reevaluate.py /path/to/training/output --epoch 10

    # Evaluate epochs up to a maximum
    python scripts/reevaluate.py /path/to/training/output --max_epoch 5

    # Evaluate only even or odd epochs (useful for split/parallel runs)
    python scripts/reevaluate.py /path/to/training/output --even_only --force
    python scripts/reevaluate.py /path/to/training/output --odd_only --force

    # Custom output dir
    python scripts/reevaluate.py /path/to/training/output \
        --attractor_radius 0.25 --output_dir /custom/path
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

from adaptive_roa.adaptive_v2.eval.full_roa import evaluate_full_roa_fast
from adaptive_roa.adaptive_v2.eval.system_hooks import resolve_system_hook
from adaptive_roa.adaptive.data_source import load_eval_states
from adaptive_roa.conformal import ConformalConfig
from adaptive_roa.conformal.calibrator import Calibrator
from adaptive_roa.conformal.probability_estimator import ProbabilityEstimator
from adaptive_roa.flow_matching.base.checkpoint_utils import (
    load_hydra_config as _load_hydra_config_from_dir,
)

# ── System registry ──────────────────────────────────────────────────────────
# Maps system _target_ class name → (system_module, system_class, fm_module, fm_class)
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


# ── Hydra config loading ─────────────────────────────────────────────────────

def load_hydra_config(training_dir: Path) -> dict:
    """Load the Hydra config from training directory (top-level .hydra/)."""
    cfg = _load_hydra_config_from_dir(training_dir)
    if cfg is None:
        raise FileNotFoundError(
            f"Hydra config not found in {training_dir} or parent directories"
        )
    return cfg


# ── Epoch discovery ───────────────────────────────────────────────────────────

def find_epoch_dirs(training_dir: Path) -> list[Path]:
    """Find all epoch directories in training output."""
    return sorted(
        [d for d in training_dir.iterdir() if d.is_dir() and re.match(r"epoch_\d+", d.name)],
        key=lambda x: int(re.search(r"\d+", x.name).group()),
    )


# ── Checkpoint loading ────────────────────────────────────────────────────────

def load_checkpoint(FlowMatcherClass, epoch_dir: Path, device: str):
    """Load flow matcher from best checkpoint in epoch directory."""
    checkpoint_dir = epoch_dir / "checkpoints"
    ckpts = glob.glob(str(checkpoint_dir / "best*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")
    return FlowMatcherClass.load_from_checkpoint(ckpts[0], device=device)


# ── Threshold state loading ──────────────────────────────────────────────────

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


# ── q_hat recomputation ──────────────────────────────────────────────────────

def recompute_qhat(
    flow_matcher,
    system,
    cal_set_file: str,
    lambda_star: float,
    delta: float,
    alpha_eval: float,
    num_mc_samples: int,
    attractor_radius: float,
    decision_rule: str,
    device: str,
    invalid_threshold: float = None,
    filter_invalid_by_conformal: bool = False,
    refine_invalids: bool = False,
    refine_t_range: tuple = (0.7, 0.9),
    refine_num_steps: int = 100,
    refine_max_attempts: int = 5,
) -> tuple:
    """Recompute q_hat (global and per-class) using alpha_eval on calibration set.

    Returns:
        Tuple of (q_hat, q_hat_success, q_hat_failure, n_cal_total, n_cal_used).
        q_hat is None if no valid calibration points remain after filtering.
        q_hat_success/q_hat_failure are None if no cal points exist for that class.
    """
    X_cal, _, y_cal = load_eval_states(cal_set_file)
    n_cal_total = len(X_cal)

    prob_config = ConformalConfig(
        delta=delta,
        alpha=alpha_eval,
        num_mc_samples=num_mc_samples,
        mc_batch_size=1024,
        attractor_radius=attractor_radius,
    )
    prob_estimator = ProbabilityEstimator(flow_matcher, system, prob_config, device)

    X_cal_tensor = torch.tensor(X_cal, dtype=torch.float32, device=device)
    p_success, p_failure, p_invalid = prob_estimator.estimate(
        X_cal_tensor,
        refine_invalids=refine_invalids,
        refine_t_range=refine_t_range,
        refine_num_steps=refine_num_steps,
        refine_max_attempts=refine_max_attempts,
    )

    if hasattr(p_success, "cpu"):
        p_success = p_success.cpu().numpy()
    if hasattr(p_failure, "cpu"):
        p_failure = p_failure.cpu().numpy()
    if hasattr(p_invalid, "cpu"):
        p_invalid = p_invalid.cpu().numpy()

    # Filter out invalid points
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

    if n_cal_used == 0:
        print("    WARNING: No valid calibration points after filtering!")
        return None, None, None, n_cal_total, 0

    cal_config = ConformalConfig(
        delta=delta,
        alpha=alpha_eval,
        decision_rule=decision_rule,
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

    # Compute per-class q_hat values
    q_hat_success, q_hat_failure, _mc_info = calibrator.calibrate_per_class(
        p_success,
        y_cal,
        lambda_star,
        delta,
        p_failure=p_failure,
        verbose=False,
    )

    return q_hat, q_hat_success, q_hat_failure, n_cal_total, n_cal_used


# ── Results saving ────────────────────────────────────────────────────────────

def save_epoch_results(
    output_dir: Path, epoch_num: int, metrics: dict,
    lambda_star: float, delta: float, q_hat: float,
    q_hat_success: float | None = None,
    q_hat_failure: float | None = None,
) -> Path:
    """Save re-evaluation results for an epoch."""
    epoch_dir = output_dir / f"epoch_{epoch_num:03d}"
    epoch_dir.mkdir(parents=True, exist_ok=True)

    artifacts = {
        "epoch": epoch_num,
        "threshold_state": {
            "lambda_star": lambda_star,
            "delta_star": delta,
            "q_hat_eval": q_hat,
            "q_hat_success_eval": q_hat_success,
            "q_hat_failure_eval": q_hat_failure,
        },
        "eval_metrics": metrics,
        "extra": {
            "source": "scripts/reevaluate.py",
        },
    }

    with open(epoch_dir / "artifacts_v2.json", "w") as f:
        json.dump(artifacts, f, indent=2)

    return epoch_dir


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Re-evaluate trained models with configurable parameters (all systems)"
    )
    parser.add_argument(
        "training_dir", type=Path,
        help="Path to training output directory containing epoch_XXX folders",
    )
    parser.add_argument("--output_dir", type=Path, default=None,
                        help="Custom output directory (overrides auto-naming)")
    parser.add_argument("--attractor_radius", type=float, default=None,
                        help="Attractor radius for success classification")
    parser.add_argument("--alpha_eval", type=float, default=None,
                        help="Significance level for q_hat calibration")
    parser.add_argument("--num_mc_samples", type=int, default=None,
                        help="Number of MC samples per point")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="GPU batch size for evaluation")
    parser.add_argument("--invalid_threshold", type=float, default=None,
                        help="Exclude calibration points with p_invalid >= threshold from q_hat")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device for evaluation")
    parser.add_argument("--epoch", type=int, default=None,
                        help="Specific epoch to evaluate")
    parser.add_argument("--max_epoch", type=int, default=None,
                        help="Maximum epoch to evaluate (all epochs <= max_epoch)")
    parity_group = parser.add_mutually_exclusive_group()
    parity_group.add_argument(
        "--even_only",
        action="store_true",
        help="Evaluate only even-numbered epochs",
    )
    parity_group.add_argument(
        "--odd_only",
        action="store_true",
        help="Evaluate only odd-numbered epochs",
    )
    parser.add_argument("--force", action="store_true",
                        help="Force re-evaluation even with no parameter changes")
    parser.add_argument("--no_filter_invalid_by_conformal", action="store_true",
                        help="Disable default conformal filtering of calibration points")
    parser.add_argument("--refine_invalids", action="store_true",
                        help="Enable refinement of invalid endpoints via late-time ODE")
    parser.add_argument("--refine_t_min", type=float, default=0.7,
                        help="Min t_start for refinement interval (default: 0.7)")
    parser.add_argument("--refine_t_max", type=float, default=0.9,
                        help="Max t_start for refinement interval (default: 0.9)")
    parser.add_argument("--refine_num_steps", type=int, default=100,
                        help="ODE steps for refinement interval (default: 100)")
    parser.add_argument("--refine_max_attempts", type=int, default=5,
                        help="Max refinement iterations per invalid endpoint (default: 5)")
    parser.add_argument("--verbose", action="store_true",
                        help="Show tqdm progress bars during evaluation")
    args = parser.parse_args()

    training_dir = args.training_dir
    if not training_dir.exists():
        print(f"ERROR: Training directory not found: {training_dir}")
        return 1

    # ── Load Hydra config & detect system ─────────────────────────────────
    hydra_config = load_hydra_config(training_dir)
    SystemClass, FlowMatcherClass, system_class_name = _resolve_system_classes(hydra_config)
    system = SystemClass()
    hook = resolve_system_hook(system)
    decision_rule = hydra_config.get("conformal", {}).get("decision_rule", hook.decision_rule)

    print(f"Detected system: {system_class_name} (decision_rule={decision_rule})")

    # ── Check if any params changed ───────────────────────────────────────
    params_changed = any([
        args.attractor_radius is not None,
        args.alpha_eval is not None,
        args.num_mc_samples is not None,
        args.batch_size is not None,
        args.invalid_threshold is not None,
        args.refine_invalids,
    ])

    if not params_changed and not args.force:
        print("=" * 70)
        print("USING DEFAULT STORED METRICS")
        print("=" * 70)
        print("No evaluation parameters specified - using existing artifacts_v2.json")
        print(f"Default metrics are in: {training_dir}/epoch_XXX/artifacts_v2.json")
        print("")
        print("To re-evaluate, specify at least one parameter:")
        print("  --attractor_radius, --alpha_eval, --num_mc_samples, --batch_size")
        print("Or use --force to re-evaluate with training defaults")
        print("")
        print("To compile default metrics, run:")
        print(f"  python scripts/compile_adaptive_metrics.py {training_dir}")
        return 0

    # ── RE-EVALUATION MODE ────────────────────────────────────────────────
    print("=" * 70)
    print(f"RE-EVALUATION MODE - {system_class_name}")
    print("=" * 70)
    print("Parameters (None = use training default):")
    print(f"  attractor_radius: {args.attractor_radius}")
    print(f"  alpha_eval: {args.alpha_eval}")
    print(f"  num_mc_samples: {args.num_mc_samples}")
    print(f"  batch_size: {args.batch_size}")
    print(f"  invalid_threshold: {args.invalid_threshold}")
    print(f"  device: {args.device}")
    if args.refine_invalids:
        print(f"  refine_invalids: True (t~U[{args.refine_t_min}, {args.refine_t_max}], "
              f"steps={args.refine_num_steps}, max_attempts={args.refine_max_attempts})")

    # Extract training defaults (safe .get() chains)
    conformal_cfg = hydra_config.get("conformal", {})
    training_defaults = {
        "attractor_radius": conformal_cfg.get("attractor_radius", hook.attractor_radius_default),
        "alpha_eval": conformal_cfg.get("alpha_eval", 0.1),
        "num_mc_samples_eval": conformal_cfg.get("num_mc_samples_eval", 10),
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
    print(f"  decision_rule: {decision_rule}")

    # ── Output directory ──────────────────────────────────────────────────
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
        if args.refine_invalids:
            name_parts.append(f"refine_t{args.refine_t_min}-{args.refine_t_max}_x{args.refine_max_attempts}")
        eval_name = "_".join(name_parts) if name_parts else "custom"
        output_dir = training_dir / "evaluations" / eval_name

    print(f"\nOutput directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Find epoch directories ────────────────────────────────────────────
    epoch_dirs = find_epoch_dirs(training_dir)
    if not epoch_dirs:
        print(f"ERROR: No epoch directories found in {training_dir}")
        return 1

    if args.epoch is not None:
        if args.even_only and args.epoch % 2 != 0:
            print(f"ERROR: --epoch {args.epoch} conflicts with --even_only")
            return 1
        if args.odd_only and args.epoch % 2 == 0:
            print(f"ERROR: --epoch {args.epoch} conflicts with --odd_only")
            return 1

        epoch_dirs = [d for d in epoch_dirs if int(re.search(r"\d+", d.name).group()) == args.epoch]
        if not epoch_dirs:
            print(f"ERROR: Epoch {args.epoch} not found in {training_dir}")
            return 1
        print(f"\nEvaluating single epoch: {args.epoch}")
    else:
        if args.max_epoch is not None:
            epoch_dirs = [d for d in epoch_dirs if int(re.search(r"\d+", d.name).group()) <= args.max_epoch]
            if not epoch_dirs:
                print(f"ERROR: No epochs <= {args.max_epoch} found in {training_dir}")
                return 1

        # Optional parity filtering for split runs.
        if args.even_only:
            epoch_dirs = [d for d in epoch_dirs if int(re.search(r"\d+", d.name).group()) % 2 == 0]
            order_info = "even only"
        elif args.odd_only:
            epoch_dirs = [d for d in epoch_dirs if int(re.search(r"\d+", d.name).group()) % 2 == 1]
            order_info = "odd only"
        else:
            # Default ordering: even epochs first, then odd epochs.
            even_epochs = [d for d in epoch_dirs if int(re.search(r"\d+", d.name).group()) % 2 == 0]
            odd_epochs = [d for d in epoch_dirs if int(re.search(r"\d+", d.name).group()) % 2 == 1]
            epoch_dirs = even_epochs + odd_epochs
            order_info = "even first, then odd"

        if not epoch_dirs:
            max_info = f" with max_epoch={args.max_epoch}" if args.max_epoch is not None else ""
            parity = "even" if args.even_only else "odd"
            print(f"ERROR: No {parity} epochs found in {training_dir}{max_info}")
            return 1

        max_info = f" (max_epoch={args.max_epoch})" if args.max_epoch is not None else ""
        print(f"\nFound {len(epoch_dirs)} epochs to re-evaluate{max_info} ({order_info})")

    # ── Data file paths from config ───────────────────────────────────────
    data_source = hydra_config.get("data_source", {})
    cal_set_file = data_source.get("cal_set_file")
    test_set_file = data_source.get("test_set_file")

    if not cal_set_file or not test_set_file:
        print(f"ERROR: data_source.cal_set_file or test_set_file missing from Hydra config")
        return 1

    # ── Evaluate epochs ───────────────────────────────────────────────────
    per_epoch_info = {}
    skipped_epochs = []

    for epoch_dir in epoch_dirs:
        epoch_num = int(re.search(r"\d+", epoch_dir.name).group())
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch_num}")
        print(f"{'='*60}")

        try:
            # 1. Load checkpoint
            print("  Loading checkpoint...")
            flow_matcher = load_checkpoint(FlowMatcherClass, epoch_dir, args.device)

            # 2. Load threshold state (lambda_star, delta from training)
            print("  Loading threshold state from v2 artifacts...")
            lambda_star, delta = load_epoch_threshold_state(epoch_dir)
            print(f"    lambda_star: {lambda_star}, delta: {delta}")

            # 3. Recompute q_hat
            filter_invalid_by_conformal = not args.no_filter_invalid_by_conformal
            conformal_threshold = lambda_star - delta if filter_invalid_by_conformal else None

            print(f"  Recomputing q_hat (alpha_eval={alpha_eval})...")
            q_hat, q_hat_success, q_hat_failure, n_cal_total, n_cal_used = recompute_qhat(
                flow_matcher, system, cal_set_file,
                lambda_star, delta, alpha_eval, num_mc_samples,
                attractor_radius, decision_rule, args.device,
                invalid_threshold=args.invalid_threshold,
                filter_invalid_by_conformal=filter_invalid_by_conformal,
                refine_invalids=args.refine_invalids,
                refine_t_range=(args.refine_t_min, args.refine_t_max),
                refine_num_steps=args.refine_num_steps,
                refine_max_attempts=args.refine_max_attempts,
            )

            if q_hat is None:
                print(f"  SKIPPING epoch {epoch_num}: q_hat is None (no valid calibration points)")
                skipped_epochs.append(epoch_num)
                continue

            if filter_invalid_by_conformal:
                print(f"    q_hat: {q_hat:.4f} (from {n_cal_used}/{n_cal_total} cal points, "
                      f"{n_cal_total - n_cal_used} excluded by conformal filter: "
                      f"p_invalid >= {conformal_threshold:.4f})")
            elif args.invalid_threshold is not None:
                print(f"    q_hat: {q_hat:.4f} (from {n_cal_used}/{n_cal_total} cal points, "
                      f"{n_cal_total - n_cal_used} excluded by invalid_threshold)")
            else:
                print(f"    q_hat: {q_hat:.4f} (from {n_cal_used} calibration points)")

            if q_hat_success is not None and q_hat_failure is not None:
                print(f"    q_hat_success: {q_hat_success:.4f}, q_hat_failure: {q_hat_failure:.4f}")
            else:
                print(f"    q_hat per-class: skipped (success={q_hat_success}, failure={q_hat_failure})")

            # 4. Run full ROA evaluation
            print("  Running full ROA evaluation...")
            metrics = evaluate_full_roa_fast(
                flow_matcher, system, test_set_file,
                num_mc_samples=num_mc_samples,
                batch_size=batch_size,
                lambda_star=lambda_star,
                delta=delta,
                q_hat=q_hat,
                q_hat_success=q_hat_success,
                q_hat_failure=q_hat_failure,
                attractor_radius=attractor_radius,
                device=args.device,
                output_dir=None,
                verbose=args.verbose,
                decision_rule=decision_rule,
                refine_invalids=args.refine_invalids,
                refine_t_range=(args.refine_t_min, args.refine_t_max),
                refine_num_steps=args.refine_num_steps,
                refine_max_attempts=args.refine_max_attempts,
            )

            # 5. Save results
            epoch_output_dir = save_epoch_results(
                output_dir, epoch_num, metrics, lambda_star, delta, q_hat,
                q_hat_success=q_hat_success,
                q_hat_failure=q_hat_failure,
            )
            print(f"  Results saved to: {epoch_output_dir}")

            # Print summary metrics
            qhat_m = metrics.get("qhat_prediction_sets", {})
            print(f"  [q_hat conformal] F1={qhat_m.get('f1', 0):.2%}, "
                  f"Acc={qhat_m.get('accuracy', 0):.2%}, "
                  f"Sep%={qhat_m.get('invalid_pct', 0):.1%}, "
                  f"Coverage={qhat_m.get('coverage', 0):.2%}")
            for mode in ("min", "max", "skip"):
                mc_key = f"qhat_multi_class_{mode}"
                mc_m = metrics.get(mc_key)
                if mc_m:
                    print(f"  [mc_{mode:4s} conformal] F1={mc_m.get('f1', 0):.2%}, "
                          f"Acc={mc_m.get('accuracy', 0):.2%}, "
                          f"Sep%={mc_m.get('invalid_pct', 0):.1%}, "
                          f"Coverage={mc_m.get('coverage', 0):.2%}")

            per_epoch_info[f"epoch_{epoch_num:03d}"] = {
                "lambda_star": lambda_star,
                "delta_star": delta,
                "q_hat_recomputed": q_hat,
                "q_hat_success": q_hat_success,
                "q_hat_failure": q_hat_failure,
                "n_cal_total": n_cal_total,
                "n_cal_used": n_cal_used,
                "conformal_threshold": conformal_threshold,
            }

        except Exception as e:
            print(f"  ERROR evaluating epoch {epoch_num}: {e}")
            skipped_epochs.append(epoch_num)
            continue

    # ── Save eval_config.json ─────────────────────────────────────────────
    eval_config = {
        "source_training_dir": str(training_dir),
        "system": system_class_name,
        "decision_rule": decision_rule,
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
            "refine_invalids": args.refine_invalids,
            "refine_t_min": args.refine_t_min,
            "refine_t_max": args.refine_t_max,
            "refine_num_steps": args.refine_num_steps,
            "refine_max_attempts": args.refine_max_attempts,
        },
        "training_defaults": training_defaults,
        "per_epoch": per_epoch_info,
    }
    if skipped_epochs:
        eval_config["skipped_epochs"] = skipped_epochs

    with open(output_dir / "eval_config.json", "w") as f:
        json.dump(eval_config, f, indent=2)

    print(f"\n{'='*70}")
    print("RE-EVALUATION COMPLETE")
    print(f"{'='*70}")
    print(f"Results saved to: {output_dir}")
    if skipped_epochs:
        print(f"Skipped epochs: {skipped_epochs}")
    print(f"\nTo compile metrics and generate plots, run:")
    print(f"  python scripts/compile_adaptive_metrics.py {training_dir} --eval_dir {output_dir.name}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
