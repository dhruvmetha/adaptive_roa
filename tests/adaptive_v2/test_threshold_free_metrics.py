"""Threshold-free metrics: AUC, AUPRC, Brier, smoothed log score.

These metrics summarise p(success|x) against the binary label without
committing to a lambda/delta operating point.
"""
import numpy as np
import pytest

from adaptive_roa.adaptive_v2.eval.full_roa import _threshold_free_metrics


def _logit(p):
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return np.log(p / (1 - p))


def _temperature(p, t):
    return 1.0 / (1.0 + np.exp(-_logit(p) / t))


def test_perfect_ranking_gives_auc_and_auprc_of_one():
    p = np.array([0.1, 0.2, 0.8, 0.9])
    y = np.array([-1, -1, 1, 1])

    m = _threshold_free_metrics(p, y)

    assert m["auc"] == pytest.approx(1.0)
    assert m["auprc"] == pytest.approx(1.0)


def test_inverted_ranking_gives_auc_of_zero():
    p = np.array([0.9, 0.8, 0.2, 0.1])
    y = np.array([-1, -1, 1, 1])

    m = _threshold_free_metrics(p, y)

    assert m["auc"] == pytest.approx(0.0)


def test_auc_is_invariant_to_monotone_recalibration_but_log_score_is_not():
    """The lambda/delta rule thresholds p, so any monotone rescaling is free.

    AUC/AUPRC must therefore be identical across temperature transforms while
    the log score moves -- this is precisely why log score alone cannot answer
    "performance without threshold optimization".
    """
    rng = np.random.default_rng(0)
    y = np.where(rng.random(500) < 0.4, 1, -1)
    p = np.clip(rng.normal(np.where(y == 1, 0.75, 0.25), 0.12), 0.01, 0.99)

    base = _threshold_free_metrics(p, y)
    hot = _threshold_free_metrics(_temperature(p, 3.0), y)
    cold = _threshold_free_metrics(_temperature(p, 1 / 3), y)

    assert hot["auc"] == pytest.approx(base["auc"], abs=1e-12)
    assert cold["auc"] == pytest.approx(base["auc"], abs=1e-12)
    assert hot["auprc"] == pytest.approx(base["auprc"], abs=1e-12)

    assert hot["log_score"] != pytest.approx(base["log_score"], abs=1e-6)
    assert cold["log_score"] != pytest.approx(base["log_score"], abs=1e-6)


def test_log_score_is_finite_when_mc_counts_are_confidently_wrong():
    """K-sample MC arms emit exactly 0.0/1.0; unsmoothed this is -inf."""
    p = np.array([0.0, 1.0, 0.0, 1.0])
    y = np.array([1, -1, 1, -1])  # every point saturated AND wrong

    m = _threshold_free_metrics(p, y, num_mc_samples=100)

    assert np.isfinite(m["log_score"])
    assert m["n_saturated"] == 4


def test_log_score_is_finite_when_continuous_probabilities_saturate():
    """Sigmoid/GP arms saturate to exactly 1.0 in float32 too."""
    p = np.array([0.0, 1.0, 0.3, 0.7])
    y = np.array([1, -1, -1, 1])

    m = _threshold_free_metrics(p, y)

    assert np.isfinite(m["log_score"])
    assert m["n_saturated"] == 2


def test_count_smoothing_uses_krichevsky_trofimov():
    """p=k/K is smoothed to (k+0.5)/(K+1), not clipped at an arbitrary eps."""
    p = np.array([0.0, 1.0])
    y = np.array([1, -1])
    K = 100

    m = _threshold_free_metrics(p, y, num_mc_samples=K)

    expected_p = 0.5 / (K + 1)  # k=0 -> both points cost -log(0.5/101)
    assert m["log_score"] == pytest.approx(-np.log(expected_p))
    assert m["log_score_smoothing"] == "kt_count"


def test_count_smoothing_shrinks_as_more_mc_samples_are_drawn():
    """Larger K -> less smoothing -> a confidently-wrong point costs more."""
    p = np.array([0.0])
    y = np.array([1])

    small_k = _threshold_free_metrics(p, y, num_mc_samples=10)
    large_k = _threshold_free_metrics(p, y, num_mc_samples=1000)

    assert large_k["log_score"] > small_k["log_score"]


def test_continuous_path_reports_clip_smoothing():
    m = _threshold_free_metrics(np.array([0.3, 0.7]), np.array([-1, 1]))

    assert m["log_score_smoothing"] == "clip"


def test_brier_is_computed_on_raw_probabilities():
    """Brier needs no smoothing, so saturated values pass through untouched."""
    p = np.array([0.0, 1.0])
    y = np.array([1, -1])

    m = _threshold_free_metrics(p, y, num_mc_samples=100)

    assert m["brier"] == pytest.approx(1.0)  # both maximally wrong


def test_brier_matches_closed_form():
    p = np.array([0.25, 0.5])
    y = np.array([1, -1])

    m = _threshold_free_metrics(p, y)

    # (0.25-1)^2 = 0.5625 ; (0.5-0)^2 = 0.25 ; mean = 0.40625
    assert m["brier"] == pytest.approx(0.40625)


def test_log_score_matches_closed_form_for_unsaturated_probabilities():
    p = np.array([0.8, 0.4])
    y = np.array([1, -1])

    m = _threshold_free_metrics(p, y)

    expected = -(np.log(0.8) + np.log(0.6)) / 2
    assert m["log_score"] == pytest.approx(expected)


def test_ranking_metrics_are_none_when_only_one_class_present():
    p = np.array([0.2, 0.8])
    y = np.array([1, 1])

    m = _threshold_free_metrics(p, y)

    assert m["auc"] is None
    assert m["auprc"] is None
    assert m["brier"] is not None  # still well defined


def test_empty_input_returns_none_metrics_without_raising():
    m = _threshold_free_metrics(np.array([]), np.array([]))

    assert m["auc"] is None
    assert m["brier"] is None
    assert m["n_scored"] == 0


def test_reports_base_rate_and_count():
    p = np.array([0.1, 0.2, 0.8, 0.9])
    y = np.array([-1, -1, -1, 1])

    m = _threshold_free_metrics(p, y)

    assert m["n_scored"] == 4
    assert m["base_rate"] == pytest.approx(0.25)


def test_invalid_mass_is_reported_as_a_diagnostic():
    """p_success+p_failure<1 means the three-outcome caveat is active."""
    p = np.array([0.2, 0.6])
    y = np.array([-1, 1])
    p_invalid = np.array([0.5, 0.0])

    m = _threshold_free_metrics(p, y, p_invalid=p_invalid)

    assert m["mean_p_invalid"] == pytest.approx(0.25)


def test_metrics_are_json_serialisable_floats():
    """artifacts_v2.json is written with the stdlib json encoder."""
    import json

    p = np.array([0.1, 0.9])
    y = np.array([-1, 1])

    m = _threshold_free_metrics(p, y, num_mc_samples=100)

    json.dumps(m)  # must not raise on np.float64/np.int64
