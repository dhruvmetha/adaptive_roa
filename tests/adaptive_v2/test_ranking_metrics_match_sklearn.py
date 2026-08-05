"""The in-tree AUC/AUPRC must match sklearn exactly, including on ties.

full_roa.py is on the core eval path and scikit-learn is NOT in install_requires,
so these are implemented in numpy. sklearn is used here only as the oracle.
"""
import numpy as np
import pytest

from adaptive_roa.adaptive_v2.eval.full_roa import _auprc, _auc

sklearn_metrics = pytest.importorskip("sklearn.metrics")


@pytest.mark.parametrize("seed", range(8))
def test_auc_matches_sklearn_on_continuous_scores(seed):
    rng = np.random.default_rng(seed)
    y = (rng.random(300) < 0.4).astype(np.float64)
    p = rng.random(300)

    assert _auc(p, y) == pytest.approx(sklearn_metrics.roc_auc_score(y, p))


@pytest.mark.parametrize("seed", range(8))
def test_auprc_matches_sklearn_on_continuous_scores(seed):
    rng = np.random.default_rng(seed)
    y = (rng.random(300) < 0.4).astype(np.float64)
    p = rng.random(300)

    assert _auprc(p, y) == pytest.approx(sklearn_metrics.average_precision_score(y, p))


@pytest.mark.parametrize("k", [2, 5, 20, 100])
def test_auc_matches_sklearn_under_heavy_ties(k):
    """MC arms emit p=k/K, so ties are the norm, not an edge case."""
    rng = np.random.default_rng(k)
    y = (rng.random(500) < 0.35).astype(np.float64)
    p = rng.integers(0, k + 1, size=500) / k  # quantised -> many ties

    assert _auc(p, y) == pytest.approx(sklearn_metrics.roc_auc_score(y, p))


@pytest.mark.parametrize("k", [2, 5, 20, 100])
def test_auprc_matches_sklearn_under_heavy_ties(k):
    rng = np.random.default_rng(k)
    y = (rng.random(500) < 0.35).astype(np.float64)
    p = rng.integers(0, k + 1, size=500) / k

    assert _auprc(p, y) == pytest.approx(sklearn_metrics.average_precision_score(y, p))


def test_auc_matches_sklearn_when_all_scores_identical():
    y = np.array([1.0, 0.0, 1.0, 0.0])
    p = np.full(4, 0.5)

    assert _auc(p, y) == pytest.approx(sklearn_metrics.roc_auc_score(y, p))


def test_auprc_matches_sklearn_on_saturated_probabilities():
    """Exactly 0.0/1.0 is the common case for unanimous MC arms."""
    y = np.array([1.0, 1.0, 0.0, 0.0, 1.0])
    p = np.array([1.0, 0.0, 0.0, 1.0, 1.0])

    assert _auprc(p, y) == pytest.approx(sklearn_metrics.average_precision_score(y, p))


def test_auc_matches_sklearn_on_real_run_probabilities():
    """Regression guard against the actual GP/FM probability distributions."""
    rng = np.random.default_rng(0)
    # bimodal, saturating at both ends -- mirrors a trained arm
    y = (rng.random(2000) < 0.38).astype(np.float64)
    p = np.clip(rng.normal(np.where(y == 1, 0.9, 0.1), 0.25), 0.0, 1.0)

    assert _auc(p, y) == pytest.approx(sklearn_metrics.roc_auc_score(y, p))
    assert _auprc(p, y) == pytest.approx(sklearn_metrics.average_precision_score(y, p))
