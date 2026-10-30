"""Invalid threshold max(lambda, 1 - lambda) - delta (det paper, ruled 2026-10-30).

Under the two-sided rule an invalid flag must only ever fall on a state the rule
would otherwise call outcome-uncertain, for every (lambda, delta) the optimizer can
pick and every MC count split, ties included. The det pendulum must decide with the
two-sided rule, since its failure attractors define p_failure.
"""
import os
from pathlib import Path

import numpy as np
import pytest

from adaptive_roa.adaptive_v2.eval.full_roa import _predict_lambda_delta
from adaptive_roa.conformal.decision import invalid_threshold
from adaptive_roa.conformal.lambda_optimizer import apply_one_sided_rule, apply_two_sided_rule

CONFIG_DIR = str(Path(__file__).resolve().parents[2] / "configs" / "adaptive_v2")
LAMS = np.round(np.linspace(0.05, 0.95, 91), 4)
DELTAS = np.round(np.linspace(0.05, 0.45, 41), 4)


def _splits(K):
    s, f = np.meshgrid(np.arange(K + 1), np.arange(K + 1), indexing="ij")
    keep = s + f <= K
    s, f = s[keep], f[keep]
    return s / K, f / K, (K - s - f) / K


def _grid():
    for lam in LAMS:
        for d in DELTAS:
            if lam - d > 0 and lam + d < 1:      # the optimizer's admissible grid
                yield float(lam), float(d)


def test_threshold_values():
    assert invalid_threshold(0.7, 0.1) == pytest.approx(0.6)
    assert invalid_threshold(0.3, 0.1) == pytest.approx(0.6)        # max(0.3, 0.7) - 0.1
    assert invalid_threshold(0.405, 0.397) == pytest.approx(0.198)  # was 0.008 under lambda - delta
    assert invalid_threshold(0.24, 0.21) == 0.55                     # rounded: no 0.5499999...


@pytest.mark.parametrize("K", [20, 100])
def test_two_sided_invalid_only_flags_states_that_would_be_uncertain(K):
    ps, pf, pi = _splits(K)
    for lam, d in _grid():
        pred, info = _predict_lambda_delta(ps, pf, pi, lam, d, "two_sided", invalid_threshold=None)
        unchecked, _ = _predict_lambda_delta(ps, pf, pi, lam, d, "two_sided", invalid_threshold=2.0)
        flagged = pred == -2
        assert not np.any(flagged & (unchecked != -1)), (lam, d)
        assert info["invalid_threshold"] == invalid_threshold(lam, d)


def test_the_old_threshold_overrode_confident_successes_below_one_half():
    # The case the change fixes: lambda < 0.5, a would-be success flagged invalid.
    ps, pf, pi = np.array([0.6]), np.array([0.0]), np.array([0.4])
    unchecked, _ = _predict_lambda_delta(ps, pf, pi, 0.3, 0.1, "two_sided", invalid_threshold=2.0)
    old, _ = _predict_lambda_delta(ps, pf, pi, 0.3, 0.1, "two_sided", invalid_threshold=0.3 - 0.1)
    new, _ = _predict_lambda_delta(ps, pf, pi, 0.3, 0.1, "two_sided", invalid_threshold=None)
    assert unchecked[0] == 1 and old[0] == -2 and new[0] == 1


def test_optimizer_and_eval_flag_the_same_states():
    ps, pf, pi = _splits(50)
    for lam, d in list(_grid())[::37]:
        pred, _ = _predict_lambda_delta(ps, pf, pi, lam, d, "two_sided", invalid_threshold=None)
        opt = apply_two_sided_rule(ps, pf, lam, d, pi)
        # optimizer labels: 1 success, -1 failure, 0 unknown (invalid or uncertain)
        assert np.array_equal(pred == 1, opt == 1) and np.array_equal(pred == 0, opt == -1), (lam, d)


def test_classifiers_are_never_flagged_invalid():
    # p_invalid is 0 for a classifier; the threshold is at least 0.5 - 0.45 > 0.
    p = np.linspace(0, 1, 101)
    for lam, d in _grid():
        pred, _ = _predict_lambda_delta(p, 1 - p, np.zeros_like(p), lam, d, "one_sided", invalid_threshold=None)
        assert not np.any(pred == -2), (lam, d)


def _compose(*overrides):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from adaptive_roa.utils.env_config import get_data_dir, get_exp_dir, get_net_id, get_shared_data_base
    for name, fn in [("net_id", get_net_id), ("exp_dir", get_exp_dir), ("data_dir", get_data_dir),
                     ("shared_data_base", get_shared_data_base)]:
        if not OmegaConf.has_resolver(name):
            OmegaConf.register_new_resolver(name, (lambda f: (lambda d="": f() or d))(fn))
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        return compose(config_name="default", overrides=list(overrides))


@pytest.mark.parametrize("predictor", ["fm_det", "det_mlp_reg", "det_bnn_reg"])
def test_det_pendulum_decides_two_sided(predictor):
    cfg = _compose("system=pendulum", f"predictor={predictor}", "+experiment=det_rerun", "output_dir=/tmp/unused")
    assert cfg.decision_rule == "two_sided"
    assert cfg.threshold.decision_rule == "two_sided"
    assert cfg.system.failure_radius == 0.15 and cfg.system.failure_theta == pytest.approx(2.0946)


def test_det_pendulum_failure_attractors_label_failure():
    import torch
    from adaptive_roa.systems.pendulum import PendulumSystem
    data = "/common/users/shared/pracsys/genMoPlan/data_trajectories/deterministic/pendulum/lqr_failset"
    if not os.path.isdir(data):
        pytest.skip("pendulum failset dataset not mounted")
    sysm = PendulumSystem(dataset_dir=data, failure_radius=0.15, failure_theta=2.0946)
    ends = torch.tensor([[0.0, 0.0], [2.0946, 0.0], [-2.0946, 0.0], [1.0, 0.0]])
    assert sysm.classify_attractor(ends, radius=0.1).tolist() == [1, -1, -1, 0]
