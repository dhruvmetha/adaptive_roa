"""Canonical full-ROA evaluator for adaptive v2."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from adaptive_roa.adaptive.data_source import load_eval_states
from adaptive_roa.adaptive_v2.eval.system_hooks import resolve_system_hook
from adaptive_roa.adaptive_v2.types import ThresholdState
from adaptive_roa.conformal.calibrator import (
    nonconformity_scores_batch_one_sided,
    nonconformity_scores_batch_two_sided,
)
from adaptive_roa.conformal.refinement import RefinementStats, refine_invalid_endpoints


def _classification_metrics_from_predictions(pred_labels: np.ndarray, y_true: np.ndarray) -> dict[str, Any]:
    n_total = len(y_true)
    n_invalid = int(np.sum(pred_labels == -2))
    n_uncertain = int(np.sum(pred_labels == -1))
    confident_mask = (pred_labels == 1) | (pred_labels == 0)
    n_confident = int(np.sum(confident_mask))

    y_pred_conf = np.where(pred_labels[confident_mask] == 1, 1, -1)
    y_true_conf = y_true[confident_mask]

    tp = int(np.sum((y_pred_conf == 1) & (y_true_conf == 1)))
    tn = int(np.sum((y_pred_conf == -1) & (y_true_conf == -1)))
    fp = int(np.sum((y_pred_conf == 1) & (y_true_conf == -1)))
    fn = int(np.sum((y_pred_conf == -1) & (y_true_conf == 1)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / n_confident if n_confident > 0 else 0.0

    return {
        "n_confident": n_confident,
        "n_invalid": n_invalid,
        "n_uncertain": n_uncertain,
        "invalid_pct": n_invalid / n_total,
        "uncertain_pct": n_uncertain / n_total,
        "separatrix_pct": (n_invalid + n_uncertain) / n_total,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def _conservative_metrics(pred_labels: np.ndarray, y_true: np.ndarray) -> dict[str, Any]:
    """Compute full-coverage metrics by classifying separatrix points as failure.

    This gives a safety-aware F1: uncertain/invalid points are treated as
    "not in ROA", which is the conservative default for a controller that
    should not attempt stabilization from uncertain states.

    Precision is unchanged from the original (only committed positives count).
    Recall drops for every true-positive point hiding in the separatrix.
    """
    # Map: success(1) stays 1, everything else (0, -1, -2) becomes failure
    y_pred_full = np.where(pred_labels == 1, 1, -1)

    tp = int(np.sum((y_pred_full == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred_full == -1) & (y_true == -1)))
    fp = int(np.sum((y_pred_full == 1) & (y_true == -1)))
    fn = int(np.sum((y_pred_full == -1) & (y_true == 1)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def optimize_lambda_delta_for_f1_targets(
    p_success: np.ndarray,
    p_failure: np.ndarray,
    p_invalid: np.ndarray,
    y_true: np.ndarray,
    target_f1s: list[float],
    decision_rule: str,
    lambda_range: tuple[float, float] = (0.3, 0.7),
    delta_range: tuple[float, float] = (0.01, 0.3),
    n_lambda_steps: int = 41,
    n_delta_steps: int = 30,
    invalid_threshold: float | None = None,
) -> dict[str, dict[str, Any]]:
    """
    Find (lambda, delta) achieving target F1 with minimum separatrix%.

    Grid search over lambda and delta to find parameters that achieve
    F1 >= target_f1 while minimizing separatrix percentage.

    Args:
        p_success: Array of success probabilities
        p_failure: Array of failure probabilities
        p_invalid: Array of invalid probabilities
        y_true: True labels (1 for success, -1 for failure)
        target_f1s: List of target F1 scores to optimize for
        decision_rule: "one_sided" or "two_sided"
        lambda_range: (min, max) for lambda search
        delta_range: (min, max) for delta search
        n_lambda_steps: Number of lambda grid points
        n_delta_steps: Number of delta grid points
        invalid_threshold: Optional threshold for invalid classification

    Returns:
        dict mapping str(target_f1) -> result dict with keys:
            - lambda_star: optimal lambda
            - delta: optimal delta
            - f1: achieved F1
            - separatrix_pct: separatrix percentage
            - accuracy, precision, recall, specificity
            - attainable: bool whether target_f1 was achievable
    """
    lambda_vals = np.linspace(lambda_range[0], lambda_range[1], n_lambda_steps)
    delta_vals = np.linspace(delta_range[0], delta_range[1], n_delta_steps)

    all_candidates = []

    for lam in lambda_vals:
        for d in delta_vals:
            pred, _ = _predict_lambda_delta(
                p_success, p_failure, p_invalid,
                lambda_star=float(lam),
                delta=float(d),
                decision_rule=decision_rule,
                invalid_threshold=invalid_threshold,
            )
            metrics = _classification_metrics_from_predictions(pred, y_true)
            all_candidates.append({
                "lambda_star": float(lam),
                "delta": float(d),
                "f1": metrics["f1"],
                "separatrix_pct": metrics["separatrix_pct"],
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "specificity": metrics["specificity"],
            })

    all_candidates.sort(key=lambda x: (-x["f1"], x["separatrix_pct"]))

    results = {}
    for target in sorted(target_f1s, reverse=True):
        matching = [c for c in all_candidates if c["f1"] >= target]

        if matching:
            best = min(matching, key=lambda x: x["separatrix_pct"])
            best["attainable"] = True
        else:
            best = all_candidates[0].copy()
            best["attainable"] = False

        results[f"{target:.2f}"] = best

    return results


def _predict_lambda_delta(
    p_success: np.ndarray,
    p_failure: np.ndarray,
    p_invalid: np.ndarray,
    lambda_star: float,
    delta: float,
    decision_rule: str,
    invalid_threshold: float | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    success_thresh = float(lambda_star + delta)
    failure_thresh = float(lambda_star - delta)
    effective_invalid_threshold = failure_thresh if invalid_threshold is None else float(invalid_threshold)

    pred = np.full(len(p_success), -1, dtype=np.int8)
    invalid_mask = p_invalid >= effective_invalid_threshold
    pred[invalid_mask] = -2

    non_invalid = ~invalid_mask
    pred[(p_success > success_thresh) & non_invalid] = 1

    if decision_rule == "two_sided":
        failure_mask = ((1.0 - p_failure) < failure_thresh) & non_invalid
    else:
        failure_mask = (p_success < failure_thresh) & non_invalid
    pred[failure_mask] = 0

    if decision_rule == "two_sided":
        both = (p_success > success_thresh) & ((1.0 - p_failure) < failure_thresh) & non_invalid
        pred[both] = -1

    return pred, {
        "lambda_star": float(lambda_star),
        "delta": float(delta),
        "success_threshold": success_thresh,
        "failure_threshold": failure_thresh,
        "invalid_threshold": effective_invalid_threshold,
    }


def _predict_fixed_threshold(
    p_success: np.ndarray,
    p_failure: np.ndarray,
    p_invalid: np.ndarray,
    threshold: float = 0.6,
    invalid_threshold: float = 0.5,
) -> tuple[np.ndarray, dict[str, Any]]:
    pred = np.full(len(p_success), -1, dtype=np.int8)
    invalid_mask = p_invalid >= invalid_threshold
    pred[invalid_mask] = -2

    non_invalid = ~invalid_mask
    pred[(p_success > threshold) & non_invalid] = 1
    pred[(p_failure > threshold) & non_invalid] = 0

    both = (p_success > threshold) & (p_failure > threshold) & non_invalid
    pred[both] = -1

    return pred, {
        "threshold": float(threshold),
        "invalid_threshold": float(invalid_threshold),
    }


def _predict_lambda_only(
    p_success: np.ndarray,
    p_failure: np.ndarray,
    p_invalid: np.ndarray,
    lambda_star: float,
    decision_rule: str,
    invalid_threshold: float | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    effective_invalid_threshold = float(lambda_star if invalid_threshold is None else invalid_threshold)
    pred = np.full(len(p_success), -1, dtype=np.int8)
    invalid_mask = p_invalid >= effective_invalid_threshold
    pred[invalid_mask] = -2

    non_invalid = ~invalid_mask
    is_success = (p_success > lambda_star) & non_invalid
    if decision_rule == "two_sided":
        failure_threshold_pf = 1.0 - lambda_star
        is_failure = (p_failure > failure_threshold_pf) & non_invalid
        both = is_success & is_failure
        pred[is_success & ~both] = 1
        pred[is_failure & ~both] = 0
    else:
        pred[is_success] = 1
        pred[(p_success < lambda_star) & non_invalid] = 0

    return pred, {
        "lambda_star": float(lambda_star),
        "delta": 0.0,
        "invalid_threshold": effective_invalid_threshold,
    }


def _predict_qhat_prediction_sets(
    p_success: np.ndarray,
    p_failure: np.ndarray,
    p_invalid: np.ndarray,
    y_true: np.ndarray,
    lambda_star: float,
    delta: float,
    q_hat: float,
    decision_rule: str,
    invalid_threshold: float | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    n = len(p_success)
    effective_invalid_threshold = float((lambda_star - delta) if invalid_threshold is None else invalid_threshold)

    if decision_rule == "two_sided":
        score_success = nonconformity_scores_batch_two_sided(
            p_success, p_failure, np.ones(n, dtype=int), lambda_star, delta
        )
        score_failure = nonconformity_scores_batch_two_sided(
            p_success, p_failure, -np.ones(n, dtype=int), lambda_star, delta
        )
        score_unknown = nonconformity_scores_batch_two_sided(
            p_success, p_failure, np.zeros(n, dtype=int), lambda_star, delta
        )
    else:
        score_success = nonconformity_scores_batch_one_sided(
            p_success, np.ones(n, dtype=int), lambda_star, delta
        )
        score_failure = nonconformity_scores_batch_one_sided(
            p_success, -np.ones(n, dtype=int), lambda_star, delta
        )
        score_unknown = nonconformity_scores_batch_one_sided(
            p_success, np.zeros(n, dtype=int), lambda_star, delta
        )

    in_set_success = score_success <= q_hat
    in_set_failure = score_failure <= q_hat
    in_set_unknown = score_unknown <= q_hat

    pred = np.full(n, -1, dtype=np.int8)  # default: uncertain
    invalid_mask = p_invalid >= effective_invalid_threshold
    pred[invalid_mask] = -2  # truly invalid (high p_invalid)

    non_invalid = ~invalid_mask
    success_only = in_set_success & ~in_set_failure & ~in_set_unknown & non_invalid
    failure_only = ~in_set_success & in_set_failure & ~in_set_unknown & non_invalid
    pred[success_only] = 1
    pred[failure_only] = 0
    # Everything else (including in_set_unknown, both in set, etc.) stays uncertain (-1)

    set_sizes = in_set_success.astype(int) + in_set_failure.astype(int) + in_set_unknown.astype(int)

    true_in_set = np.zeros(n, dtype=bool)
    true_in_set[y_true == 1] = in_set_success[y_true == 1]
    true_in_set[y_true == -1] = in_set_failure[y_true == -1]
    true_in_set[y_true == 0] = in_set_unknown[y_true == 0]

    confident_mask = (pred == 1) | (pred == 0)

    extras = {
        "q_hat": float(q_hat),
        "invalid_threshold": effective_invalid_threshold,
        "coverage": float(np.mean(true_in_set)),
        "coverage_confident": float(np.mean(true_in_set[confident_mask])) if np.any(confident_mask) else 0.0,
        "avg_set_size": float(np.mean(set_sizes[non_invalid])) if np.any(non_invalid) else 0.0,
        "n_pred_success": int(np.sum(pred == 1)),
        "n_pred_failure": int(np.sum(pred == 0)),
    }
    return pred, extras


def _predict_qhat_multi_class(
    p_success: np.ndarray,
    p_failure: np.ndarray,
    p_invalid: np.ndarray,
    y_true: np.ndarray,
    lambda_star: float,
    delta: float,
    q_hat_success: float,
    q_hat_failure: float,
    decision_rule: str,
    invalid_threshold: float | None,
    unknown_mode: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Prediction using per-class q_hat values with configurable unknown handling.

    Args:
        unknown_mode: One of "min", "max", "skip".
            - "min": unknown in set if score_unknown <= min(q_hat_success, q_hat_failure)
            - "max": unknown in set if score_unknown <= max(q_hat_success, q_hat_failure)
            - "skip": unknown never in set
    """
    n = len(p_success)
    effective_invalid_threshold = float((lambda_star - delta) if invalid_threshold is None else invalid_threshold)

    if decision_rule == "two_sided":
        score_success = nonconformity_scores_batch_two_sided(
            p_success, p_failure, np.ones(n, dtype=int), lambda_star, delta
        )
        score_failure = nonconformity_scores_batch_two_sided(
            p_success, p_failure, -np.ones(n, dtype=int), lambda_star, delta
        )
        score_unknown = nonconformity_scores_batch_two_sided(
            p_success, p_failure, np.zeros(n, dtype=int), lambda_star, delta
        )
    else:
        score_success = nonconformity_scores_batch_one_sided(
            p_success, np.ones(n, dtype=int), lambda_star, delta
        )
        score_failure = nonconformity_scores_batch_one_sided(
            p_success, -np.ones(n, dtype=int), lambda_star, delta
        )
        score_unknown = nonconformity_scores_batch_one_sided(
            p_success, np.zeros(n, dtype=int), lambda_star, delta
        )

    in_set_success = score_success <= q_hat_success
    in_set_failure = score_failure <= q_hat_failure

    if unknown_mode == "min":
        q_hat_unknown = min(q_hat_success, q_hat_failure)
        in_set_unknown = score_unknown <= q_hat_unknown
    elif unknown_mode == "max":
        q_hat_unknown = max(q_hat_success, q_hat_failure)
        in_set_unknown = score_unknown <= q_hat_unknown
    else:  # "skip"
        q_hat_unknown = None
        in_set_unknown = np.zeros(n, dtype=bool)

    pred = np.full(n, -1, dtype=np.int8)  # default: uncertain
    invalid_mask = p_invalid >= effective_invalid_threshold
    pred[invalid_mask] = -2

    non_invalid = ~invalid_mask
    success_only = in_set_success & ~in_set_failure & ~in_set_unknown & non_invalid
    failure_only = ~in_set_success & in_set_failure & ~in_set_unknown & non_invalid
    pred[success_only] = 1
    pred[failure_only] = 0

    set_sizes = in_set_success.astype(int) + in_set_failure.astype(int) + in_set_unknown.astype(int)

    true_in_set = np.zeros(n, dtype=bool)
    true_in_set[y_true == 1] = in_set_success[y_true == 1]
    true_in_set[y_true == -1] = in_set_failure[y_true == -1]
    true_in_set[y_true == 0] = in_set_unknown[y_true == 0]

    confident_mask = (pred == 1) | (pred == 0)

    extras = {
        "q_hat_success": float(q_hat_success),
        "q_hat_failure": float(q_hat_failure),
        "q_hat_unknown": float(q_hat_unknown) if q_hat_unknown is not None else None,
        "unknown_mode": unknown_mode,
        "invalid_threshold": effective_invalid_threshold,
        "coverage": float(np.mean(true_in_set)),
        "coverage_confident": float(np.mean(true_in_set[confident_mask])) if np.any(confident_mask) else 0.0,
        "avg_set_size": float(np.mean(set_sizes[non_invalid])) if np.any(non_invalid) else 0.0,
        "median_set_size": float(np.median(set_sizes[non_invalid])) if np.any(non_invalid) else 0.0,
        "n_pred_success": int(np.sum(pred == 1)),
        "n_pred_failure": int(np.sum(pred == 0)),
    }
    return pred, extras


def _compute_geodesic_error_stats(errors: np.ndarray, mask: np.ndarray | None = None) -> dict[str, Any]:
    if mask is not None:
        errors = errors[mask]
    if len(errors) == 0:
        return {"n_points": 0}

    norms = np.linalg.norm(errors, axis=1)
    return {
        "n_points": int(len(errors)),
        "mean": float(np.mean(norms)),
        "median": float(np.median(norms)),
        "variance": float(np.var(norms)),
        "mean_per_dim": [float(x) for x in errors.mean(axis=0)],
        "median_per_dim": [float(x) for x in np.median(errors, axis=0)],
        "variance_per_dim": [float(x) for x in np.var(errors, axis=0)],
    }


def _compute_mc_sample_error_stats(mc_errors: np.ndarray, mask: np.ndarray | None = None) -> dict[str, Any]:
    if mask is not None:
        mc_errors = mc_errors[mask]
    if len(mc_errors) == 0:
        return {"n_points": 0}

    stats = {
        "mean": np.mean(mc_errors, axis=1),
        "median": np.median(mc_errors, axis=1),
        "p90": np.percentile(mc_errors, 90, axis=1),
        "p99": np.percentile(mc_errors, 99, axis=1),
    }

    result: dict[str, Any] = {"n_points": int(len(mc_errors))}
    for name, values in stats.items():
        result[f"mean_of_{name}s"] = float(np.mean(values))
        result[f"median_of_{name}s"] = float(np.median(values))
        result[f"p90_of_{name}s"] = float(np.percentile(values, 90))
        result[f"p99_of_{name}s"] = float(np.percentile(values, 99))

    return result


def evaluate_full_roa_fast(
    flow_matcher,
    system,
    eval_states_file: str,
    num_mc_samples: int = 20,
    batch_size: int = 2048,
    lambda_star: float | None = None,
    delta: float = 0.05,
    q_hat: float | None = None,
    q_hat_success: float | None = None,
    q_hat_failure: float | None = None,
    attractor_radius: float = 0.2,
    device: str = "cuda",
    output_dir: str | None = None,
    verbose: bool = True,
    invalid_threshold: float | None = None,
    decision_rule: str | None = None,
    refine_invalids: bool = False,
    refine_t_range: tuple[float, float] = (0.7, 0.9),
    refine_num_steps: int = 100,
    refine_max_attempts: int = 5,
    mc_cache: "MCCache | None" = None,
) -> dict[str, Any]:
    """Fast batched full-ROA evaluation with v2-compatible outputs.

    If ``mc_cache`` is provided, the expensive GPU MC sampling loop is skipped
    entirely and predictions are derived from cached endpoints/labels.  When
    ``attractor_radius`` differs from the cache's radius, labels are
    recomputed from cached endpoints on the CPU.
    """
    from tqdm import tqdm

    hook = resolve_system_hook(system)
    effective_rule = decision_rule or hook.decision_rule

    X_all, end_states_all, y_all = load_eval_states(eval_states_file)
    n_total = len(y_all)
    n_success_true = int(np.sum(y_all == 1))
    n_failure_true = int(np.sum(y_all == -1))

    if lambda_star is None:
        lambda_star = 0.5

    if verbose:
        q_hat_str = f"{q_hat:.4f}" if q_hat is not None else "None"
        if mc_cache is not None:
            print(f"  [Full ROA] N={n_total} eval states ({n_success_true} success, "
                  f"{n_failure_true} failure), using MC cache (K={mc_cache.num_mc_samples})")
        else:
            refine_str = (
                f", refine t~U[{refine_t_range[0]}, {refine_t_range[1]}] max_attempts={refine_max_attempts}"
                if refine_invalids else ""
            )
            print(f"  [Full ROA] N={n_total} eval states ({n_success_true} success, {n_failure_true} failure), "
                  f"K={num_mc_samples} MC samples, batch={batch_size}{refine_str}")
        print(f"  [Full ROA] λ*={lambda_star:.4f}, δ={delta:.4f}, "
              f"q_hat={q_hat_str}, rule={effective_rule}")

    # ── Use cache if available ───────────────────────────────────────────
    if mc_cache is not None:
        assert mc_cache.n_states == n_total, (
            f"Cache size mismatch: cache has {mc_cache.n_states} states, "
            f"eval file has {n_total}"
        )
        # Reclassify if attractor radius changed
        if abs(mc_cache.attractor_radius - attractor_radius) > 1e-9:
            if verbose:
                print(f"  [Full ROA] Reclassifying cache: radius {mc_cache.attractor_radius} → {attractor_radius}")
            mc_cache = mc_cache.reclassify(system, attractor_radius)

        num_mc_samples = mc_cache.num_mc_samples
        mc_labels = mc_cache.mc_labels
        # Derive pred_sum and mc_errors from cached endpoints
        pred_sum = mc_cache.mc_endpoints.sum(axis=1).astype(np.float64)  # [N, state_dim]
        mc_errors = np.linalg.norm(
            mc_cache.mc_endpoints - end_states_all[:, np.newaxis, :],
            axis=2,
        ).astype(np.float32)  # [N, K]
    else:
        # ── GPU MC sampling loop (original path) ─────────────────────────
        X_tensor = torch.from_numpy(X_all).float().to(device)
        end_tensor = torch.from_numpy(end_states_all).float().to(device)

        mc_labels = np.zeros((n_total, num_mc_samples), dtype=np.int8)
        pred_sum = np.zeros((n_total, end_states_all.shape[1]), dtype=np.float64)
        mc_errors = np.zeros((n_total, num_mc_samples), dtype=np.float32)

        cumulative_rstats = RefinementStats(
            per_attempt_resolved=[0] * refine_max_attempts
        ) if refine_invalids else None

        n_batches = (n_total + batch_size - 1) // batch_size
        total_steps = n_batches * num_mc_samples

        flow_matcher.eval()
        with torch.no_grad():
          with tqdm(total=total_steps, desc="Full ROA eval", disable=not verbose) as pbar:
            for batch_start in range(0, n_total, batch_size):
                batch_end = min(batch_start + batch_size, n_total)
                batch_inputs = X_tensor[batch_start:batch_end]
                batch_actual = end_tensor[batch_start:batch_end]

                for sample_idx in range(num_mc_samples):
                    pred = flow_matcher.predict_endpoint(batch_inputs)
                    labels_tensor = system.classify_attractor(pred, attractor_radius)

                    if refine_invalids:
                        rstats = refine_invalid_endpoints(
                            flow_matcher, system,
                            pred, labels_tensor, batch_inputs,
                            attractor_radius=attractor_radius,
                            t_range=refine_t_range,
                            num_steps=refine_num_steps,
                            max_attempts=refine_max_attempts,
                        )
                        cumulative_rstats.accumulate(rstats)

                    labels = labels_tensor.cpu().numpy()
                    mc_labels[batch_start:batch_end, sample_idx] = labels
                    pred_np = pred.cpu().numpy()
                    pred_sum[batch_start:batch_end] += pred_np

                    if hasattr(flow_matcher, "distance_manifold"):
                        geodesic = flow_matcher.distance_manifold.dist(pred, batch_actual).cpu().numpy()
                    else:
                        geodesic = pred_np - batch_actual.cpu().numpy()
                    mc_errors[batch_start:batch_end, sample_idx] = np.linalg.norm(geodesic, axis=1)

                    pbar.update(1)

        if refine_invalids and verbose and cumulative_rstats.n_initially_invalid > 0:
            total_original = cumulative_rstats.n_initially_invalid
            total_resolved = cumulative_rstats.n_resolved
            resolve_rate = total_resolved / total_original * 100
            print(f"  [Refine] {total_resolved}/{total_original} invalid MC samples resolved ({resolve_rate:.1f}%) [max_attempts={refine_max_attempts}]")
        print(f"           → success: {cumulative_rstats.n_resolved_success}, → failure: {cumulative_rstats.n_resolved_failure}")
        attempt_strs = [f"a{i+1}={c}" for i, c in enumerate(cumulative_rstats.per_attempt_resolved) if c > 0]
        if attempt_strs:
            print(f"           per-attempt: {', '.join(attempt_strs)}")

    pred_mean = pred_sum / float(num_mc_samples)

    if flow_matcher is not None and hasattr(flow_matcher, "distance_manifold"):
        pred_tensor = torch.from_numpy(pred_mean).float().to(device)
        end_tensor_local = torch.from_numpy(end_states_all).float().to(device) if mc_cache is not None else end_tensor
        geodesic_errors = flow_matcher.distance_manifold.dist(pred_tensor, end_tensor_local).cpu().numpy()
        component_names = list(flow_matcher.get_manifold_component_names())
    else:
        geodesic_errors = (pred_mean - end_states_all).astype(np.float32)
        component_names = [f"dim_{i}" for i in range(geodesic_errors.shape[1])]

    p_success = (mc_labels == 1).sum(axis=1) / num_mc_samples
    p_failure = (mc_labels == -1).sum(axis=1) / num_mc_samples
    p_invalid = (mc_labels == 0).sum(axis=1) / num_mc_samples

    pred_conformal, conformal_extras = _predict_lambda_delta(
        p_success,
        p_failure,
        p_invalid,
        lambda_star=float(lambda_star),
        delta=float(delta),
        decision_rule=effective_rule,
        invalid_threshold=invalid_threshold,
    )
    metrics_conformal = _classification_metrics_from_predictions(pred_conformal, y_all)
    metrics_conformal.update(conformal_extras)

    pred_fixed, fixed_extras = _predict_fixed_threshold(p_success, p_failure, p_invalid)
    metrics_fixed = _classification_metrics_from_predictions(pred_fixed, y_all)
    metrics_fixed.update(fixed_extras)
    metrics_fixed["n_pred_success"] = int(np.sum(pred_fixed == 1))
    metrics_fixed["n_pred_failure"] = int(np.sum(pred_fixed == 0))
    metrics_fixed["n_pred_invalid"] = int(np.sum(pred_fixed == -2))
    metrics_fixed["n_pred_uncertain"] = int(np.sum(pred_fixed == -1))

    pred_lambda_only, lambda_only_extras = _predict_lambda_only(
        p_success,
        p_failure,
        p_invalid,
        lambda_star=float(lambda_star),
        decision_rule=effective_rule,
        invalid_threshold=invalid_threshold,
    )
    metrics_lambda_only = _classification_metrics_from_predictions(pred_lambda_only, y_all)
    metrics_lambda_only.update(lambda_only_extras)
    metrics_lambda_only["n_pred_success"] = int(np.sum(pred_lambda_only == 1))
    metrics_lambda_only["n_pred_failure"] = int(np.sum(pred_lambda_only == 0))
    metrics_lambda_only["n_pred_invalid"] = int(np.sum(pred_lambda_only == -2))
    metrics_lambda_only["n_pred_uncertain"] = int(np.sum(pred_lambda_only == -1))

    if q_hat is not None:
        pred_qhat, qhat_extras = _predict_qhat_prediction_sets(
            p_success,
            p_failure,
            p_invalid,
            y_all,
            lambda_star=float(lambda_star),
            delta=float(delta),
            q_hat=float(q_hat),
            decision_rule=effective_rule,
            invalid_threshold=invalid_threshold,
        )
        metrics_qhat = _classification_metrics_from_predictions(pred_qhat, y_all)
        metrics_qhat.update(qhat_extras)
    else:
        pred_qhat = None
        metrics_qhat = None

    # Multi-class conformal prediction (per-class q_hat)
    metrics_mc = {}
    pred_mc_all = {}
    if q_hat_success is not None and q_hat_failure is not None:
        for unknown_mode in ("min", "max", "skip"):
            pred_mc, mc_extras = _predict_qhat_multi_class(
                p_success, p_failure, p_invalid, y_all,
                lambda_star=float(lambda_star),
                delta=float(delta),
                q_hat_success=float(q_hat_success),
                q_hat_failure=float(q_hat_failure),
                decision_rule=effective_rule,
                invalid_threshold=invalid_threshold,
                unknown_mode=unknown_mode,
            )
            mc_metrics = _classification_metrics_from_predictions(pred_mc, y_all)
            mc_metrics.update(mc_extras)
            key = f"qhat_multi_class_{unknown_mode}"
            metrics_mc[key] = mc_metrics
            pred_mc_all[key] = pred_mc

    # Conservative metrics: classify separatrix as failure (full-coverage, safety-aware)
    metrics_conservative_ld = _conservative_metrics(pred_conformal, y_all)
    if pred_qhat is not None:
        metrics_conservative_qhat = _conservative_metrics(pred_qhat, y_all)
    else:
        metrics_conservative_qhat = None

    if verbose:
        mc_p_inv_mean = float(np.mean(p_invalid))
        mc_p_inv_median = float(np.median(p_invalid))
        print(f"  [Full ROA] MC probs: p_success mean={np.mean(p_success):.4f}, "
              f"p_failure mean={np.mean(p_failure):.4f}, p_invalid mean={mc_p_inv_mean:.4f} median={mc_p_inv_median:.4f}")
        print(f"  [Full ROA] λ±δ results:  F1={metrics_conformal['f1']:.4f}  "
              f"acc={metrics_conformal['accuracy']:.4f}  "
              f"prec={metrics_conformal['precision']:.4f}  "
              f"recall={metrics_conformal['recall']:.4f}  "
              f"separatrix={metrics_conformal['separatrix_pct']:.1%}")
        print(f"  [Full ROA] conservative: F1={metrics_conservative_ld['f1']:.4f}  "
              f"prec={metrics_conservative_ld['precision']:.4f}  "
              f"recall={metrics_conservative_ld['recall']:.4f}")
        if metrics_qhat is not None:
            print(f"  [Full ROA] q_hat sets:   F1={metrics_qhat['f1']:.4f}  "
                  f"acc={metrics_qhat['accuracy']:.4f}  "
                  f"coverage={metrics_qhat.get('coverage', 0):.4f}  "
                  f"avg_set={metrics_qhat.get('avg_set_size', 0):.2f}")
        for mc_key, mc_m in metrics_mc.items():
            mode_label = mc_key.replace("qhat_multi_class_", "mc_")
            print(f"  [Full ROA] {mode_label:10s}:  F1={mc_m['f1']:.4f}  "
                  f"acc={mc_m['accuracy']:.4f}  "
                  f"coverage={mc_m.get('coverage', 0):.4f}  "
                  f"avg_set={mc_m.get('avg_set_size', 0):.2f}")

    mask_invalid = pred_conformal == -2
    mask_uncertain = pred_conformal == -1
    mask_certain_success = pred_conformal == 1
    mask_certain_failure = pred_conformal == 0
    mask_certain = mask_certain_success | mask_certain_failure

    error_stats_full = _compute_geodesic_error_stats(geodesic_errors)
    error_stats_certain = _compute_geodesic_error_stats(geodesic_errors, mask_certain)
    error_stats_certain_success = _compute_geodesic_error_stats(geodesic_errors, mask_certain_success)
    error_stats_certain_failure = _compute_geodesic_error_stats(geodesic_errors, mask_certain_failure)
    error_stats_uncertain = _compute_geodesic_error_stats(geodesic_errors, mask_uncertain)
    error_stats_invalid = _compute_geodesic_error_stats(geodesic_errors, mask_invalid)

    mc_sample_full = _compute_mc_sample_error_stats(mc_errors)
    mc_sample_certain = _compute_mc_sample_error_stats(mc_errors, mask_certain)
    mc_sample_certain_success = _compute_mc_sample_error_stats(mc_errors, mask_certain_success)
    mc_sample_certain_failure = _compute_mc_sample_error_stats(mc_errors, mask_certain_failure)
    mc_sample_uncertain = _compute_mc_sample_error_stats(mc_errors, mask_uncertain)
    mc_sample_invalid = _compute_mc_sample_error_stats(mc_errors, mask_invalid)

    if pred_qhat is not None:
        q_mask_invalid = pred_qhat == -2
        q_mask_uncertain = pred_qhat == -1
        q_mask_certain_success = pred_qhat == 1
        q_mask_certain_failure = pred_qhat == 0
        q_mask_certain = q_mask_certain_success | q_mask_certain_failure

        error_stats_qhat = {
            "component_names": component_names,
            "full": error_stats_full,
            "certain": _compute_geodesic_error_stats(geodesic_errors, q_mask_certain),
            "certain_success": _compute_geodesic_error_stats(geodesic_errors, q_mask_certain_success),
            "certain_failure": _compute_geodesic_error_stats(geodesic_errors, q_mask_certain_failure),
            "uncertain": _compute_geodesic_error_stats(geodesic_errors, q_mask_uncertain),
            "invalid": _compute_geodesic_error_stats(geodesic_errors, q_mask_invalid),
        }
        mc_sample_qhat = {
            "full": mc_sample_full,
            "certain": _compute_mc_sample_error_stats(mc_errors, q_mask_certain),
            "certain_success": _compute_mc_sample_error_stats(mc_errors, q_mask_certain_success),
            "certain_failure": _compute_mc_sample_error_stats(mc_errors, q_mask_certain_failure),
            "uncertain": _compute_mc_sample_error_stats(mc_errors, q_mask_uncertain),
            "invalid": _compute_mc_sample_error_stats(mc_errors, q_mask_invalid),
        }
    else:
        error_stats_qhat = None
        mc_sample_qhat = None

    # Compute error stats for multi-class regions
    mc_error_stats = {}
    mc_mc_sample_stats = {}
    for mc_key, pred_mc in pred_mc_all.items():
        mc_mask_invalid = pred_mc == -2
        mc_mask_uncertain = pred_mc == -1
        mc_mask_certain_success = pred_mc == 1
        mc_mask_certain_failure = pred_mc == 0
        mc_mask_certain = mc_mask_certain_success | mc_mask_certain_failure

        mc_error_stats[mc_key] = {
            "component_names": component_names,
            "full": error_stats_full,
            "certain": _compute_geodesic_error_stats(geodesic_errors, mc_mask_certain),
            "certain_success": _compute_geodesic_error_stats(geodesic_errors, mc_mask_certain_success),
            "certain_failure": _compute_geodesic_error_stats(geodesic_errors, mc_mask_certain_failure),
            "uncertain": _compute_geodesic_error_stats(geodesic_errors, mc_mask_uncertain),
            "invalid": _compute_geodesic_error_stats(geodesic_errors, mc_mask_invalid),
        }
        mc_mc_sample_stats[mc_key] = {
            "full": mc_sample_full,
            "certain": _compute_mc_sample_error_stats(mc_errors, mc_mask_certain),
            "certain_success": _compute_mc_sample_error_stats(mc_errors, mc_mask_certain_success),
            "certain_failure": _compute_mc_sample_error_stats(mc_errors, mc_mask_certain_failure),
            "uncertain": _compute_mc_sample_error_stats(mc_errors, mc_mask_uncertain),
            "invalid": _compute_mc_sample_error_stats(mc_errors, mc_mask_invalid),
        }

    metrics = {
        "_doc": "Full ROA evaluation on held-out test set using MC sampling",
        "n_total": int(n_total),
        "num_mc_samples": int(num_mc_samples),
        "lambda_star": float(lambda_star),
        "delta": float(delta),
        "q_hat": float(q_hat) if q_hat is not None else None,
        "q_hat_success": float(q_hat_success) if q_hat_success is not None else None,
        "q_hat_failure": float(q_hat_failure) if q_hat_failure is not None else None,
        "lambda_delta": {
            "_doc": "Classification using conformal lambda +/- delta band: success if p>lambda+delta, failure if p<lambda-delta, uncertain otherwise",
            **metrics_conformal,
        },
        "fixed_threshold": {
            "_doc": "Fixed threshold baseline: success if p_success>0.6, invalid if p_invalid>0.5",
            **metrics_fixed,
        },
        "lambda_only": {
            "_doc": "Lambda-only classification (delta=0): success if p>lambda, failure if p<lambda",
            **metrics_lambda_only,
        },
        "qhat_prediction_sets": {
            "_doc": "Conformal prediction sets using calibrated q_hat for set-valued predictions with coverage guarantee",
            **metrics_qhat,
        } if metrics_qhat is not None else None,
        "conservative_lambda_delta": {
            "_doc": "Full-coverage safety-aware metrics: separatrix points classified as failure (not in ROA). "
                    "Precision unchanged, recall penalized for true ROA points in separatrix.",
            **metrics_conservative_ld,
        },
        "conservative_qhat": {
            "_doc": "Full-coverage safety-aware metrics (q_hat variant): separatrix points classified as failure.",
            **metrics_conservative_qhat,
        } if metrics_conservative_qhat is not None else None,
        "endpoint_errors": {
            "_doc": "Mean geodesic distance between predicted and true endpoints, partitioned by lambda_delta regions",
            "component_names": component_names,
            "full": error_stats_full,
            "certain": error_stats_certain,
            "certain_success": error_stats_certain_success,
            "certain_failure": error_stats_certain_failure,
            "uncertain": error_stats_uncertain,
            "invalid": error_stats_invalid,
        },
        "mc_sample_errors": {
            "_doc": "Per-point error variability across MC samples (mean/median/p90/p99 of per-sample norms), partitioned by lambda_delta regions",
            "full": mc_sample_full,
            "certain": mc_sample_certain,
            "certain_success": mc_sample_certain_success,
            "certain_failure": mc_sample_certain_failure,
            "uncertain": mc_sample_uncertain,
            "invalid": mc_sample_invalid,
        },
        "endpoint_errors_qhat_regions": error_stats_qhat,
        "mc_sample_errors_qhat_regions": mc_sample_qhat,
    }

    # Add multi-class metrics and error stats
    for mc_key, mc_m in metrics_mc.items():
        metrics[mc_key] = {
            "_doc": f"Per-class conformal prediction sets ({mc_key.split('_')[-1]} unknown mode)",
            **mc_m,
        }
    for mc_key in mc_error_stats:
        metrics[f"endpoint_errors_{mc_key}_regions"] = mc_error_stats[mc_key]
        metrics[f"mc_sample_errors_{mc_key}_regions"] = mc_mc_sample_stats[mc_key]

    if error_stats_qhat is not None:
        error_stats_qhat["_doc"] = "Same geodesic errors as endpoint_errors, but partitioned by qhat_prediction_sets regions instead"
    if mc_sample_qhat is not None:
        mc_sample_qhat["_doc"] = "Same MC sample errors, but partitioned by qhat_prediction_sets regions instead"

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        np.savez(
            str(out / "full_roa_per_point.npz"),
            start_states=X_all,
            p_success=p_success,
            p_failure=p_failure,
            p_invalid=p_invalid,
            true_labels=y_all,
            lambda_star=lambda_star,
            delta=delta,
            attractor_radius=attractor_radius,
        )

        hook.maybe_plot(
            start_states=X_all,
            p_success=p_success,
            true_labels=y_all,
            lambda_star=float(lambda_star),
            delta=float(delta),
            output_file=str(out / "full_roa_projections.png"),
        )

        if X_all.shape[1] >= 2:
            i, j = hook.projection_dims
            fig, ax = plt.subplots(figsize=(6, 5))
            sc = ax.scatter(X_all[:, i], X_all[:, j], c=p_success, s=2, cmap="RdYlBu", vmin=0.0, vmax=1.0)
            ax.set_xlabel(hook.projection_labels[0])
            ax.set_ylabel(hook.projection_labels[1])
            ax.set_title("p(success)")
            ax.grid(True, alpha=0.3)
            cbar = plt.colorbar(sc, ax=ax)
            cbar.set_label("p_success")
            fig.tight_layout()
            fig.savefig(str(out / "full_roa_heatmap.png"), dpi=150, bbox_inches="tight")
            plt.close(fig)

    return metrics


@torch.no_grad()
def evaluate_full_roa_classifier(
    classifier,
    system,
    eval_states_file: str,
    lambda_star: float | None = None,
    delta: float = 0.05,
    attractor_radius: float = 0.2,
    device: str = "cuda",
    batch_size: int = 8192,
    output_dir: str | None = None,
    verbose: bool = True,
    invalid_threshold: float | None = None,
    decision_rule: str | None = None,
) -> dict[str, Any]:
    """Full-ROA evaluation for a discriminative classifier (single forward pass).

    Reuses the model-agnostic threshold/metric helpers; produces the same
    top-level metric keys as ``evaluate_full_roa_fast`` so downstream artifact
    handling is unchanged. Flow-matching-only fields (geodesic / MC-sample
    errors, q_hat prediction sets) are ``None`` (a classifier has no generated
    endpoints and ``p_invalid`` is identically 0).
    """
    effective_rule = decision_rule
    hook = None
    if effective_rule is None or output_dir:
        hook = resolve_system_hook(system)
        if effective_rule is None:
            effective_rule = hook.decision_rule

    X_all, _end_states_all, y_all = load_eval_states(eval_states_file)
    n_total = len(y_all)
    n_success_true = int(np.sum(y_all == 1))
    n_failure_true = int(np.sum(y_all == -1))

    if lambda_star is None:
        lambda_star = 0.5

    classifier.eval()
    if hasattr(classifier, "to"):
        classifier.to(device)  # ensure model is on the eval device (standalone/reeval calls)
    X_tensor = torch.from_numpy(X_all).float().to(device)
    probs = []
    for start in range(0, n_total, batch_size):
        logits = classifier(X_tensor[start:start + batch_size])
        if logits.dim() > 1:
            logits = logits.squeeze(-1)
        probs.append(torch.sigmoid(logits))
    p_success = torch.cat(probs).to(torch.float64).cpu().numpy()
    p_failure = 1.0 - p_success
    p_invalid = np.zeros_like(p_success)

    pred_conformal, conformal_extras = _predict_lambda_delta(
        p_success, p_failure, p_invalid, float(lambda_star), float(delta),
        decision_rule=effective_rule, invalid_threshold=invalid_threshold,
    )
    metrics_conformal = _classification_metrics_from_predictions(pred_conformal, y_all)
    metrics_conformal.update(conformal_extras)

    pred_fixed, fixed_extras = _predict_fixed_threshold(p_success, p_failure, p_invalid)
    metrics_fixed = _classification_metrics_from_predictions(pred_fixed, y_all)
    metrics_fixed.update(fixed_extras)

    pred_lambda_only, lambda_only_extras = _predict_lambda_only(
        p_success, p_failure, p_invalid, float(lambda_star),
        decision_rule=effective_rule, invalid_threshold=invalid_threshold,
    )
    metrics_lambda_only = _classification_metrics_from_predictions(pred_lambda_only, y_all)
    metrics_lambda_only.update(lambda_only_extras)

    metrics_conservative_ld = _conservative_metrics(pred_conformal, y_all)

    if verbose:
        print(f"  [Full ROA / classifier] N={n_total} ({n_success_true} success, {n_failure_true} failure), "
              f"λ*={lambda_star:.4f}, δ={delta:.4f}, rule={effective_rule}")
        print(f"  [Full ROA / classifier] p_success mean={np.mean(p_success):.4f}")
        print(f"  [Full ROA / classifier] λ±δ:  F1={metrics_conformal['f1']:.4f}  "
              f"acc={metrics_conformal['accuracy']:.4f}  prec={metrics_conformal['precision']:.4f}  "
              f"recall={metrics_conformal['recall']:.4f}  uncertain={metrics_conformal['uncertain_pct']:.1%}")
        print(f"  [Full ROA / classifier] conservative: F1={metrics_conservative_ld['f1']:.4f}  "
              f"recall={metrics_conservative_ld['recall']:.4f}")

    metrics = {
        "_doc": "Full ROA evaluation on held-out test set using a discriminative classifier (single forward pass)",
        "predictor": "classifier",
        "n_total": int(n_total),
        "num_mc_samples": 1,
        "lambda_star": float(lambda_star),
        "delta": float(delta),
        "q_hat": None,
        "q_hat_success": None,
        "q_hat_failure": None,
        "lambda_delta": {
            "_doc": "Classification using lambda +/- delta band: success if p>lambda+delta, failure if p<lambda-delta, uncertain otherwise",
            **metrics_conformal,
        },
        "fixed_threshold": {
            "_doc": "Fixed threshold baseline: success if p_success>0.6",
            **metrics_fixed,
        },
        "lambda_only": {
            "_doc": "Lambda-only classification (delta=0): success if p>lambda, failure if p<lambda",
            **metrics_lambda_only,
        },
        "qhat_prediction_sets": None,
        "conservative_lambda_delta": {
            "_doc": "Full-coverage safety-aware metrics: uncertain points classified as failure (not in ROA).",
            **metrics_conservative_ld,
        },
        "conservative_qhat": None,
        "endpoint_errors": None,
        "mc_sample_errors": None,
        "endpoint_errors_qhat_regions": None,
        "mc_sample_errors_qhat_regions": None,
    }

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        np.savez(
            str(out / "full_roa_per_point.npz"),
            start_states=X_all, p_success=p_success, p_failure=p_failure,
            p_invalid=p_invalid, true_labels=y_all, lambda_star=lambda_star,
            delta=delta, attractor_radius=attractor_radius,
        )
        if hook is not None:
            hook.maybe_plot(
                start_states=X_all, p_success=p_success, true_labels=y_all,
                lambda_star=float(lambda_star), delta=float(delta),
                output_file=str(out / "full_roa_projections.png"),
            )
            if X_all.shape[1] >= 2:
                i, j = hook.projection_dims
                fig, ax = plt.subplots(figsize=(6, 5))
                sc = ax.scatter(X_all[:, i], X_all[:, j], c=p_success, s=2, cmap="RdYlBu", vmin=0.0, vmax=1.0)
                ax.set_xlabel(hook.projection_labels[0])
                ax.set_ylabel(hook.projection_labels[1])
                ax.set_title("p(success) — classifier")
                ax.grid(True, alpha=0.3)
                cbar = plt.colorbar(sc, ax=ax)
                cbar.set_label("p_success")
                fig.tight_layout()
                fig.savefig(str(out / "full_roa_heatmap.png"), dpi=150, bbox_inches="tight")
                plt.close(fig)

    return metrics


@dataclass
class FullROAEvaluator:
    """Evaluator adapter used by the v2 engine."""

    system: Any
    cfg: Any
    device: str

    def evaluate_epoch(
        self,
        model_handle: Any,
        threshold_state: ThresholdState,
        epoch_context: dict[str, Any],
    ) -> dict[str, Any]:
        predictor_type = str(self.cfg.get("predictor", "generative"))
        if predictor_type == "classifier":
            return evaluate_full_roa_classifier(
                classifier=model_handle,
                system=self.system,
                eval_states_file=epoch_context["eval_states_file"],
                lambda_star=threshold_state.lambda_star,
                delta=threshold_state.delta_star,
                attractor_radius=epoch_context.get(
                    "attractor_radius",
                    self.cfg.conformal.get("attractor_radius", resolve_system_hook(self.system).attractor_radius_default),
                ),
                device=self.device,
                batch_size=epoch_context.get("batch_size", self.cfg.get("val_batch_size", 8192)),
                output_dir=epoch_context.get("output_dir"),
                verbose=epoch_context.get("verbose", True),
                invalid_threshold=epoch_context.get("invalid_threshold", None),
                decision_rule=epoch_context.get("decision_rule", self.cfg.conformal.get("decision_rule", None)),
            )
        return evaluate_full_roa_fast(
            flow_matcher=model_handle,
            system=self.system,
            eval_states_file=epoch_context["eval_states_file"],
            num_mc_samples=epoch_context.get(
                "num_mc_samples",
                self.cfg.conformal.get("num_mc_samples_eval", 20),
            ),
            batch_size=epoch_context.get("batch_size", self.cfg.get("val_batch_size", 2048)),
            lambda_star=threshold_state.lambda_star,
            delta=threshold_state.delta_star,
            q_hat=threshold_state.q_hat_eval,
            q_hat_success=threshold_state.q_hat_success_eval,
            q_hat_failure=threshold_state.q_hat_failure_eval,
            attractor_radius=epoch_context.get(
                "attractor_radius",
                self.cfg.conformal.get("attractor_radius", resolve_system_hook(self.system).attractor_radius_default),
            ),
            device=self.device,
            output_dir=epoch_context.get("output_dir"),
            verbose=epoch_context.get("verbose", True),
            invalid_threshold=epoch_context.get("invalid_threshold", None),
            decision_rule=epoch_context.get("decision_rule", self.cfg.conformal.get("decision_rule", None)),
            refine_invalids=self.cfg.conformal.get("refine_invalids", False),
            refine_t_range=(
                self.cfg.conformal.get("refine_t_min", 0.7),
                self.cfg.conformal.get("refine_t_max", 0.9),
            ),
            refine_num_steps=self.cfg.conformal.get("refine_num_steps", 100),
            refine_max_attempts=self.cfg.conformal.get("refine_max_attempts", 5),
        )
