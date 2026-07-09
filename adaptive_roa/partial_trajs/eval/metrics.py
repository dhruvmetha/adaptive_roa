"""Dynamics-accuracy metrics for the partial-trajectory model.

Metric #1 (per-horizon T-step error): manifold-aware distance between the
predicted and true state one horizon ahead. Circular dimensions are wrapped;
all other dimensions (incl. quaternion components, per the raw-13D convention)
use plain differences. Reported overall and stratified by freeze/non-freeze and
by motion magnitude.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

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


def _prf(tp: float, fp: float, fn: float) -> Dict[str, float]:
    """Precision / recall / F1 for the success class from confusion counts."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def roa_scores(
    pred_labels: torch.Tensor,
    true_labels: torch.Tensor,
) -> Dict[str, float]:
    """ROA success-detection scores.

    ``pred_labels`` are verifier outputs in {+1 success, -1 failure, 0 unresolved};
    ``true_labels`` are ground truth with +1 = success (anything else = non-success).
    F1/precision/recall are for the success class; ``accuracy`` is success-vs-not;
    ``unresolved_frac`` is the fraction the verifier left unresolved.

    The plain ``f1`` is the *full-coverage* (safety-aware) flavor: an unresolved
    query counts as not-success. The ``*_resolved_only`` variants restrict scoring
    to the queries the verifier committed on (``pred != 0``) — the analog of the
    adaptive "confident-set" metrics — and can differ sharply when many queries are
    left unresolved.
    """
    pred_success = pred_labels == 1
    true_success = true_labels == 1

    tp = float((pred_success & true_success).sum())
    fp = float((pred_success & ~true_success).sum())
    fn = float((~pred_success & true_success).sum())
    tn = float((~pred_success & ~true_success).sum())

    prf = _prf(tp, fp, fn)
    accuracy = float((pred_success == true_success).float().mean())
    unresolved_frac = float((pred_labels == 0).float().mean())

    # Resolved-only: score over the committed subset (pred != 0).
    resolved = pred_labels != 0
    ps_r, ts_r = pred_success[resolved], true_success[resolved]
    tp_r = float((ps_r & ts_r).sum())
    fp_r = float((ps_r & ~ts_r).sum())
    fn_r = float((~ps_r & ts_r).sum())
    prf_r = _prf(tp_r, fp_r, fn_r)

    return {
        **prf,
        "accuracy": accuracy,
        "unresolved_frac": unresolved_frac,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "n_total": int(pred_labels.numel()),
        "n_pred_success": int((pred_labels == 1).sum()),
        "n_pred_failure": int((pred_labels == -1).sum()),
        "n_pred_unresolved": int((pred_labels == 0).sum()),
        "precision_resolved_only": prf_r["precision"],
        "recall_resolved_only": prf_r["recall"],
        "f1_resolved_only": prf_r["f1"],
    }


def _error_stats(err: torch.Tensor) -> Optional[Dict[str, float]]:
    """``{n, mean, median, p90, max}`` for a 1-D error tensor; None if empty."""
    if err.numel() == 0:
        return None
    e = err.float()
    return {
        "n": int(e.numel()),
        "mean": float(e.mean()),
        "median": float(torch.quantile(e, 0.5)),
        "p90": float(torch.quantile(e, 0.9)),
        "max": float(e.max()),
    }


def _grouped_stats(err: torch.Tensor, masks: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, float]]:
    """Apply ``_error_stats`` per named mask, dropping empty (None) buckets."""
    out = {name: _error_stats(err[mask]) for name, mask in masks.items()}
    return {name: stats for name, stats in out.items() if stats is not None}


def stratified_error_report(
    err: torch.Tensor,
    gt_label: torch.Tensor,
    pred_label: torch.Tensor,
    terminal_class: Optional[torch.Tensor] = None,
    motion: Optional[torch.Tensor] = None,
) -> Dict[str, object]:
    """Metric #2 (rollout final-state error) stratified across GT/pred classes.

    Buckets (empty ones omitted), each a ``{n, mean, median, p90, max}`` leaf:
      - ``overall``
      - ``by_gt``: gt_success (label==1) / gt_nonsuccess
      - ``by_gt_terminal`` (if ``terminal_class`` given): success / failure /
        unresolved, from ``classify_attractor`` on the GT terminal state
      - ``by_pred``: pred_success (+1) / pred_failure (-1) / pred_unresolved (0)
      - ``by_gt_x_pred``: the ``{gt}__{pred}`` cross-tab cells
      - ``by_motion`` (if ``motion`` given): low / high, split at the motion median
    """
    err = err.float()
    gt_success = gt_label == 1
    gt_masks = {"gt_success": gt_success, "gt_nonsuccess": ~gt_success}
    pred_masks = {
        "pred_success": pred_label == 1,
        "pred_failure": pred_label == -1,
        "pred_unresolved": pred_label == 0,
    }

    report: Dict[str, object] = {"overall": _error_stats(err)}
    report["by_gt"] = _grouped_stats(err, gt_masks)

    if terminal_class is not None:
        report["by_gt_terminal"] = _grouped_stats(
            err,
            {
                "success": terminal_class == 1,
                "failure": terminal_class == -1,
                "unresolved": terminal_class == 0,
            },
        )

    report["by_pred"] = _grouped_stats(err, pred_masks)

    cross = {}
    for gk, gm in gt_masks.items():
        for pk, pm in pred_masks.items():
            stats = _error_stats(err[gm & pm])
            if stats is not None:
                cross[f"{gk}__{pk}"] = stats
    report["by_gt_x_pred"] = cross

    if motion is not None:
        high = motion.float() > torch.quantile(motion.float(), 0.5)
        report["by_motion"] = _grouped_stats(err, {"low": ~high, "high": high})

    return report
