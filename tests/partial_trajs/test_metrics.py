"""Tests for partial-trajectory dynamics-accuracy metrics (#1)."""
import math

import torch

from adaptive_roa.partial_trajs.eval.metrics import (
    manifold_state_distance,
    horizon_error_report,
    roa_scores,
    stratified_error_report,
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


def test_roa_scores_reports_confusion_and_counts():
    # true: successes idx0,1; preds: [success, unresolved, unresolved, failure]
    true_labels = torch.tensor([1, 1, 0, 0])
    pred_labels = torch.tensor([1, 0, 0, -1])
    s = roa_scores(pred_labels, true_labels)
    assert s["tp"] == 1 and s["fp"] == 0 and s["fn"] == 1 and s["tn"] == 2
    assert s["n_total"] == 4
    assert s["n_pred_success"] == 1
    assert s["n_pred_failure"] == 1
    assert s["n_pred_unresolved"] == 2


def test_roa_scores_resolved_only_excludes_unresolved():
    # idx1 is a true success left unresolved: excluding it lifts resolved-only recall.
    true_labels = torch.tensor([1, 1, 0, 0])
    pred_labels = torch.tensor([1, 0, 0, -1])
    s = roa_scores(pred_labels, true_labels)
    # full-coverage recall counts the unresolved success as a miss...
    assert s["recall"] == 0.5
    # ...resolved-only ignores unresolved queries entirely -> perfect on the committed set.
    assert s["precision_resolved_only"] == 1.0
    assert s["recall_resolved_only"] == 1.0
    assert s["f1_resolved_only"] == 1.0


def test_stratified_error_report_buckets():
    err = torch.tensor([1.0, 2.0, 3.0, 4.0])
    gt_label = torch.tensor([1, 1, 0, 0])          # success, success, nonsuccess, nonsuccess
    pred_label = torch.tensor([1, 0, 1, -1])        # succ, unresolved, succ(false alarm), fail
    terminal_class = torch.tensor([1, 1, -1, 0])    # GT terminal: succ, succ, failure, unresolved
    motion = torch.tensor([0.1, 0.2, 3.0, 4.0])

    rep = stratified_error_report(err, gt_label, pred_label, terminal_class, motion)

    # overall
    assert rep["overall"]["n"] == 4
    assert abs(rep["overall"]["mean"] - 2.5) < 1e-6
    assert abs(rep["overall"]["max"] - 4.0) < 1e-6
    assert {"mean", "median", "p90", "max", "n"} <= set(rep["overall"])

    # by ground-truth class
    assert abs(rep["by_gt"]["gt_success"]["mean"] - 1.5) < 1e-6
    assert abs(rep["by_gt"]["gt_nonsuccess"]["mean"] - 3.5) < 1e-6

    # by ground-truth terminal class (failure/unresolved split)
    assert rep["by_gt_terminal"]["failure"]["n"] == 1
    assert rep["by_gt_terminal"]["unresolved"]["n"] == 1
    assert rep["by_gt_terminal"]["success"]["n"] == 2

    # by predicted class
    assert abs(rep["by_pred"]["pred_success"]["mean"] - 2.0) < 1e-6  # err [1,3]
    assert rep["by_pred"]["pred_failure"]["n"] == 1
    assert rep["by_pred"]["pred_unresolved"]["n"] == 1

    # cross-tab cells partition the samples; false-alarm cell present
    cells = rep["by_gt_x_pred"]
    assert sum(c["n"] for c in cells.values()) == 4
    assert "gt_nonsuccess__pred_success" in cells         # false alarm
    assert "gt_success__pred_failure" not in cells        # empty bucket omitted

    # motion split at the median
    assert rep["by_motion"]["low"]["n"] == 2
    assert rep["by_motion"]["high"]["n"] == 2


def test_stratified_error_report_omits_optional_groups():
    err = torch.tensor([1.0, 2.0])
    gt_label = torch.tensor([1, 0])
    pred_label = torch.tensor([1, 0])
    rep = stratified_error_report(err, gt_label, pred_label)
    assert "by_gt_terminal" not in rep   # terminal_class not provided
    assert "by_motion" not in rep        # motion not provided
    assert rep["overall"]["n"] == 2
