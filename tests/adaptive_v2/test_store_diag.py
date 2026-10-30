"""H07 diagnostics on a hand-built case."""
import numpy as np

from adaptive_roa.adaptive_v2.eval.store_diag import diagnose


def test_composition_and_valid_but_wrong():
    # 6 states: two confident right, one confident wrong, one uncertain success, two invalid
    p_s = np.array([.9, .1, .9, .5, .1, .0])
    p_i = np.array([0., 0., 0., 0., .8, .9])
    p_f = 1 - p_s - p_i
    y = np.array([1, -1, -1, 1, 1, -1])
    d = diagnose(p_s, p_f, p_i, y, lam=0.5, delta=0.2, rule="two_sided", inv_thr=0.3)
    assert d["invalid_share"] == 2 / 6 and d["uncertain_share"] == 1 / 6
    assert d["success_share_in_invalid"] == 0.5
    assert d["valid_but_wrong_share_of_committed"] == 1 / 3
    assert d["auroc_p_invalid_ranks_bad"] > 0.5
