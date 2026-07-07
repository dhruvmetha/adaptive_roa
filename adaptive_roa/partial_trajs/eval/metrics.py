"""Dynamics-accuracy metrics for the partial-trajectory model.

Metric #1 (per-horizon T-step error): manifold-aware distance between the
predicted and true state one horizon ahead. Circular dimensions are wrapped;
all other dimensions (incl. quaternion components, per the raw-13D convention)
use plain differences. Reported overall and stratified by freeze/non-freeze and
by motion magnitude.
"""
from __future__ import annotations

from typing import Dict, Sequence

import torch


def manifold_state_distance(
    pred: torch.Tensor,
    target: torch.Tensor,
    circular_indices: Sequence[int] = (),
) -> torch.Tensor:
    """Per-sample L2 distance with circular dims wrapped to [-pi, pi]. Returns [B]."""
    diff = pred - target
    if len(circular_indices) > 0:
        idx = torch.as_tensor(list(circular_indices), device=diff.device, dtype=torch.long)
        wrapped = torch.atan2(torch.sin(diff[:, idx]), torch.cos(diff[:, idx]))
        diff = diff.clone()
        diff[:, idx] = wrapped
    return torch.linalg.norm(diff, dim=1)


def horizon_error_report(
    pred: torch.Tensor,
    target: torch.Tensor,
    is_freeze: torch.Tensor,
    motion: torch.Tensor,
    circular_indices: Sequence[int] = (),
    motion_quantile: float = 0.5,
) -> Dict[str, float]:
    """Aggregate metric #1 overall and stratified by freeze and motion."""
    dist = manifold_state_distance(pred, target, circular_indices)
    report: Dict[str, float] = {"n": int(dist.numel()), "mean": float(dist.mean())}

    freeze = is_freeze.bool()
    if (~freeze).any():
        report["mean_nonfreeze"] = float(dist[~freeze].mean())
    if freeze.any():
        report["mean_freeze"] = float(dist[freeze].mean())

    threshold = torch.quantile(motion.float(), motion_quantile)
    high = motion > threshold
    if high.any():
        report["mean_high_motion"] = float(dist[high].mean())
    if (~high).any():
        report["mean_low_motion"] = float(dist[~high].mean())

    return report


def roa_scores(
    pred_labels: torch.Tensor,
    true_labels: torch.Tensor,
) -> Dict[str, float]:
    """ROA success-detection scores.

    ``pred_labels`` are verifier outputs in {+1 success, -1 failure, 0 unresolved};
    ``true_labels`` are ground truth with +1 = success (anything else = non-success).
    F1/precision/recall are for the success class; ``accuracy`` is success-vs-not;
    ``unresolved_frac`` is the fraction the verifier left unresolved.
    """
    pred_success = pred_labels == 1
    true_success = true_labels == 1

    tp = float((pred_success & true_success).sum())
    fp = float((pred_success & ~true_success).sum())
    fn = float((~pred_success & true_success).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = float((pred_success == true_success).float().mean())
    unresolved_frac = float((pred_labels == 0).float().mean())

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "unresolved_frac": unresolved_frac,
    }
