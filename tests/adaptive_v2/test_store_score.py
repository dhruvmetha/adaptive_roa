"""Store scoring (tracker H01): denominators, threshold-free scores, fixed-precision point."""
import numpy as np
import pytest

from adaptive_roa.adaptive_v2.eval import store_score as ss

sklearn = pytest.importorskip("sklearn.metrics")


def test_auroc_and_auprc_match_sklearn_with_ties():
    rng = np.random.default_rng(0)
    y = (rng.random(500) < 0.3).astype(int)
    p = np.round(np.clip(y * 0.3 + rng.random(500) * 0.7, 0, 1), 1)   # many ties
    assert ss._auroc(p, y) == pytest.approx(sklearn.roc_auc_score(y, p))
    assert ss._auprc(p, y) == pytest.approx(sklearn.average_precision_score(y, p))


def test_recovery_and_fpr_use_all_states_as_denominators():
    # 4 successes, 4 failures; one success abstains (uncertain), one failure admitted
    p = np.array([.9, .9, .9, .5, .1, .1, .1, .9])
    y = np.array([1, 1, 1, 1, -1, -1, -1, -1])
    r = ss.score_epoch(p, 1 - p, np.zeros(8), y, lam=0.5, delta=0.2, rule="one_sided", inv_thr=None)
    assert r["success_precision"] == pytest.approx(3 / 4)
    assert r["success_recovery"] == pytest.approx(3 / 4)      # abstained success still counts
    assert r["fpr_all_failures"] == pytest.approx(1 / 4)
    assert r["abstain_uncertain"] == pytest.approx(1 / 8)


def test_precision_threshold_is_chosen_on_val_only():
    p_val = np.array([.95, .9, .8, .7, .6])
    y_val = np.array([1, 1, 1, 0, 1])
    t = ss.threshold_for_precision(p_val, y_val, 0.99)
    assert t == pytest.approx(0.8)
    assert ss.threshold_for_precision(np.array([.9, .8]), np.array([0, 0]), 0.99) is None
