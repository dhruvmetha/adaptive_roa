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

    pred = np.full(n, -1, dtype=np.int8)
    invalid_mask = (p_invalid >= effective_invalid_threshold) | in_set_unknown
    pred[invalid_mask] = -2

    non_invalid = ~invalid_mask
    success_only = in_set_success & ~in_set_failure & non_invalid
    failure_only = ~in_set_success & in_set_failure & non_invalid
    pred[success_only] = 1
    pred[failure_only] = 0

    set_sizes = in_set_success.astype(int) + in_set_failure.astype(int) + in_set_unknown.astype(int)

    true_in_set = np.zeros(n, dtype=bool)
    true_in_set[y_true == 1] = in_set_success[y_true == 1]
    true_in_set[y_true == -1] = in_set_failure[y_true == -1]
    true_in_set[y_true == 0] = in_set_unknown[y_true == 0]

    extras = {
        "q_hat": float(q_hat),
        "invalid_threshold": effective_invalid_threshold,
        "coverage": float(np.mean(true_in_set)),
        "avg_set_size": float(np.mean(set_sizes[non_invalid])) if np.any(non_invalid) else 0.0,
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
) -> dict[str, Any]:
    """Fast batched full-ROA evaluation with v2-compatible outputs."""
    from tqdm import tqdm

    hook = resolve_system_hook(system)
    effective_rule = decision_rule or hook.decision_rule

    X_all, end_states_all, y_all = load_eval_states(eval_states_file)
    n_total = len(y_all)

    if lambda_star is None:
        lambda_star = 0.5

    X_tensor = torch.from_numpy(X_all).float().to(device)
    end_tensor = torch.from_numpy(end_states_all).float().to(device)

    mc_labels = np.zeros((n_total, num_mc_samples), dtype=np.int8)
    pred_sum = np.zeros((n_total, end_states_all.shape[1]), dtype=np.float64)
    mc_errors = np.zeros((n_total, num_mc_samples), dtype=np.float32)

    # Track refinement statistics
    total_invalids_before_refine = 0
    total_refined_to_success = 0
    total_refined_to_failure = 0

    flow_matcher.eval()
    with torch.no_grad():
        for batch_start in tqdm(range(0, n_total, batch_size), desc="Evaluating", disable=not verbose):
            batch_end = min(batch_start + batch_size, n_total)
            batch_inputs = X_tensor[batch_start:batch_end]
            batch_actual = end_tensor[batch_start:batch_end]

            for sample_idx in range(num_mc_samples):
                pred = flow_matcher.predict_endpoint(batch_inputs)
                labels_tensor = system.classify_attractor(pred, attractor_radius)

                # Iteratively refine invalid endpoints
                if refine_invalids:
                    still_invalid = (labels_tensor == 0)
                    n_initial_invalid = still_invalid.sum().item()
                    if n_initial_invalid > 0:
                        total_invalids_before_refine += n_initial_invalid

                        for _attempt in range(refine_max_attempts):
                            if not still_invalid.any():
                                break

                            refined = flow_matcher.refine_endpoints(
                                invalid_endpoints=pred[still_invalid],
                                start_states=batch_inputs[still_invalid],
                                t_range=refine_t_range,
                                num_steps=refine_num_steps,
                            )
                            refined_labels = system.classify_attractor(refined, attractor_radius)

                            resolved_success = (refined_labels == 1)
                            resolved_failure = (refined_labels == -1)

                            total_refined_to_success += resolved_success.sum().item()
                            total_refined_to_failure += resolved_failure.sum().item()

                            # Update pred and labels for resolved endpoints
                            still_invalid_indices = still_invalid.nonzero(as_tuple=True)[0]
                            labels_tensor[still_invalid_indices[resolved_success]] = 1
                            labels_tensor[still_invalid_indices[resolved_failure]] = -1
                            pred[still_invalid] = refined

                            # Narrow to still-invalid
                            still_invalid_remaining = (refined_labels == 0)
                            new_still_invalid = torch.zeros_like(still_invalid)
                            new_still_invalid[still_invalid_indices[still_invalid_remaining]] = True
                            still_invalid = new_still_invalid

                labels = labels_tensor.cpu().numpy()
                mc_labels[batch_start:batch_end, sample_idx] = labels
                pred_np = pred.cpu().numpy()
                pred_sum[batch_start:batch_end] += pred_np

                if hasattr(flow_matcher, "distance_manifold"):
                    geodesic = flow_matcher.distance_manifold.dist(pred, batch_actual).cpu().numpy()
                else:
                    geodesic = pred_np - batch_actual.cpu().numpy()
                mc_errors[batch_start:batch_end, sample_idx] = np.linalg.norm(geodesic, axis=1)

    if refine_invalids and verbose:
        total_refined = total_refined_to_success + total_refined_to_failure
        if total_invalids_before_refine > 0:
            resolve_rate = total_refined / total_invalids_before_refine * 100
            print(f"  [Refine] {total_refined}/{total_invalids_before_refine} invalid MC samples resolved ({resolve_rate:.1f}%) [max_attempts={refine_max_attempts}]")
            print(f"           → success: {total_refined_to_success}, → failure: {total_refined_to_failure}")

    pred_mean = pred_sum / float(num_mc_samples)

    pred_tensor = torch.from_numpy(pred_mean).float().to(device)
    if hasattr(flow_matcher, "distance_manifold"):
        geodesic_errors = flow_matcher.distance_manifold.dist(pred_tensor, end_tensor).cpu().numpy()
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

    metrics = {
        "_doc": "Full ROA evaluation on held-out test set using MC sampling",
        "n_total": int(n_total),
        "num_mc_samples": int(num_mc_samples),
        "lambda_star": float(lambda_star),
        "delta": float(delta),
        "q_hat": float(q_hat) if q_hat is not None else None,
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
