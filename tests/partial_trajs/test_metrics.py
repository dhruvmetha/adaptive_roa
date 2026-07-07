"""Tests for partial-trajectory dynamics-accuracy metrics (#1)."""
import math

import torch

from adaptive_roa.partial_trajs.eval.metrics import (
    manifold_state_distance,
    horizon_error_report,
    roa_scores,
)


def test_distance_zero_for_identical_states():
    x = torch.randn(5, 3)
    d = manifold_state_distance(x, x, circular_indices=[0])
    assert torch.allclose(d, torch.zeros(5), atol=1e-6)


def test_circular_dimension_wraps():
    # angle at +0.05 vs (2π - 0.05): true circular gap is 0.1, not ~2π
    pred = torch.tensor([[0.05, 0.0]])
    target = torch.tensor([[2 * math.pi - 0.05, 0.0]])
    d_circ = manifold_state_distance(pred, target, circular_indices=[0])
    d_plain = manifold_state_distance(pred, target, circular_indices=[])
    assert d_circ.item() < 0.11
    assert d_plain.item() > 6.0  # ~2π when not wrapped


def test_report_stratifies_by_freeze_and_motion():
    pred = torch.zeros(4, 2)
    target = torch.tensor([[0.1, 0.0], [0.2, 0.0], [0.3, 0.0], [0.4, 0.0]])
    is_freeze = torch.tensor([1, 1, 0, 0])
    motion = torch.tensor([0.0, 0.1, 1.0, 2.0])
    rep = horizon_error_report(
        pred, target, is_freeze, motion, circular_indices=[0], motion_quantile=0.5
    )
    assert rep["n"] == 4
    assert "mean" in rep
    assert "mean_freeze" in rep and "mean_nonfreeze" in rep
    assert "mean_low_motion" in rep and "mean_high_motion" in rep
    # freeze rows have smaller targets -> smaller error than nonfreeze rows
    assert rep["mean_freeze"] < rep["mean_nonfreeze"]


def test_roa_scores_success_detection():
    # true: 2 successes (idx 0,1), 2 non-success (idx 2,3)
    true_labels = torch.tensor([1, 1, 0, 0])
    # verifier preds: idx0 success (TP), idx1 unresolved (FN), idx2 unresolved (TN), idx3 failure (TN)
    pred_labels = torch.tensor([1, 0, 0, -1])
    s = roa_scores(pred_labels, true_labels)
    assert s["precision"] == 1.0            # only positive is correct
    assert s["recall"] == 0.5               # 1 of 2 successes found
    assert abs(s["f1"] - (2 / 3)) < 1e-6
    assert s["accuracy"] == 0.75            # 3/4 success-vs-not correct
    assert s["unresolved_frac"] == 0.5      # 2/4 predicted unresolved
