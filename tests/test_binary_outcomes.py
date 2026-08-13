"""Stochastic systems have BINARY outcomes -- there is no 'invalid' class.

Ground truth for every stochastic dataset is ``p_success = successes / trials``
with no third outcome recorded: a rollout either reached the goal or it did not.
The model side, however, classifies each sampled endpoint with
``system.classify_attractor``, which returns 1 / -1 / 0 (success / failure /
unresolved) and therefore invents a third class with no ground-truth counterpart.

Consequences observed before this was fixed:

* cartpole ``sigma_020.0`` scored 67.9% of eval points "invalid" at epoch 0 and
  produced F1 = 0.035 with TP = 1 -- the invalid bucket swallowed nearly every
  true success.
* pendulum FM runs carried 20-57% invalid mass at every stochastic level.

Note this never touched ``p_success`` itself, so AUC / Brier / REL / RES and the
``recal`` verdicts are unaffected. The damage is confined to the lambda/delta
decision rule, which can emit prediction label -2 for a point whose true
p_success is 0.8.

These tests pin: (a) invalid draws fold into failure so the model's outcome space
matches the ground truth's, (b) no decision rule emits -2 under binary outcomes,
(c) the zero-threshold edge case cannot silently mark everything invalid, and
(d) ternary behaviour is untouched by default so runs in flight are unaffected.
"""

import numpy as np
import pytest

from adaptive_roa.adaptive_v2.eval.full_roa import (
    _predict_fixed_threshold,
    _predict_lambda_delta,
    _predict_lambda_only,
)
from adaptive_roa.adaptive_v2.types import OutcomeProbabilities


# ---------------------------------------------------------------- fold at source

def test_fold_invalid_into_failure_preserves_p_success():
    """p_success must NOT change -- every threshold-free metric is built on it."""
    from adaptive_roa.probabilistic_classifier.endpoint_mc import fold_invalid_into_failure

    probs = OutcomeProbabilities(
        p_success=np.array([0.30, 0.00, 1.00]),
        p_failure=np.array([0.20, 0.40, 0.00]),
        p_invalid=np.array([0.50, 0.60, 0.00]),
    )
    out = fold_invalid_into_failure(probs)
    np.testing.assert_allclose(out.p_success, [0.30, 0.00, 1.00])
    np.testing.assert_allclose(out.p_failure, [0.70, 1.00, 0.00])
    np.testing.assert_allclose(out.p_invalid, [0.0, 0.0, 0.0])


def test_folded_probabilities_sum_to_one():
    """Matching the ground truth means non-success mass is all failure."""
    from adaptive_roa.probabilistic_classifier.endpoint_mc import fold_invalid_into_failure

    rng = np.random.default_rng(0)
    raw = rng.dirichlet([1, 1, 1], size=200)
    probs = OutcomeProbabilities(raw[:, 0], raw[:, 1], raw[:, 2])
    out = fold_invalid_into_failure(probs)
    np.testing.assert_allclose(out.p_success + out.p_failure, 1.0, atol=1e-12)
    np.testing.assert_allclose(out.p_success, raw[:, 0])


# ------------------------------------------------------- decision rules go binary

@pytest.mark.parametrize("fn,kwargs", [
    (_predict_lambda_delta, dict(lambda_star=0.5, delta=0.1, decision_rule="one_sided",
                                 invalid_threshold=None)),
    (_predict_lambda_only, dict(lambda_star=0.5, decision_rule="one_sided",
                                invalid_threshold=None)),
    (_predict_fixed_threshold, dict(threshold=0.6, invalid_threshold=0.5)),
])
def test_binary_outcomes_never_emit_invalid(fn, kwargs):
    """-2 has no ground-truth counterpart on a stochastic system."""
    p_success = np.array([0.9, 0.5, 0.1, 0.0])
    p_failure = 1.0 - p_success
    p_invalid = np.zeros_like(p_success)
    pred, _ = fn(p_success, p_failure, p_invalid, binary_outcomes=True, **kwargs)
    assert -2 not in set(pred.tolist())


def test_zero_invalid_threshold_does_not_mark_everything_invalid():
    """The edge case that makes folding-alone unsafe.

    `invalid_mask = p_invalid >= threshold`. Folding sets p_invalid to 0, and
    lambda=0.5/delta=0.5 makes the effective threshold 0.0, so `0 >= 0` would
    mark EVERY point invalid -- the exact inversion of the intended fix.
    """
    p_success = np.array([0.9, 0.5, 0.1])
    p_failure = 1.0 - p_success
    p_invalid = np.zeros_like(p_success)

    pred_ternary, _ = _predict_lambda_delta(
        p_success, p_failure, p_invalid, lambda_star=0.5, delta=0.5,
        decision_rule="one_sided", invalid_threshold=None)
    assert (pred_ternary == -2).all(), "guard premise changed: expected the 0>=0 trap"

    pred_binary, _ = _predict_lambda_delta(
        p_success, p_failure, p_invalid, lambda_star=0.5, delta=0.5,
        decision_rule="one_sided", invalid_threshold=None, binary_outcomes=True)
    assert -2 not in set(pred_binary.tolist())


def test_true_successes_are_not_hidden_in_the_invalid_bucket():
    """The cartpole failure: a point with high p_success scored as 'invalid'."""
    p_success = np.array([0.85])
    p_failure = np.array([0.15])
    p_invalid_ternary = np.array([0.70])          # what classify_attractor produced

    pred_t, _ = _predict_lambda_delta(
        p_success, p_failure, p_invalid_ternary, lambda_star=0.5, delta=0.1,
        decision_rule="one_sided", invalid_threshold=None)
    assert pred_t[0] == -2, "premise: the ternary rule buries this success as invalid"

    pred_b, _ = _predict_lambda_delta(
        p_success, np.array([0.15]), np.zeros(1), lambda_star=0.5, delta=0.1,
        decision_rule="one_sided", invalid_threshold=None, binary_outcomes=True)
    assert pred_b[0] == 1, "binary rule must commit to success"


# -------------------------------------------------------- backward compatibility

@pytest.mark.parametrize("fn,kwargs", [
    (_predict_lambda_delta, dict(lambda_star=0.5, delta=0.1, decision_rule="one_sided",
                                 invalid_threshold=None)),
    (_predict_lambda_only, dict(lambda_star=0.5, decision_rule="one_sided",
                                invalid_threshold=None)),
    (_predict_fixed_threshold, dict(threshold=0.6, invalid_threshold=0.5)),
])
def test_default_is_unchanged_ternary(fn, kwargs):
    """Runs in flight re-import this module; the default must not move.

    Deterministic systems genuinely have three outcomes, and jobs already running
    resolved their config before this key existed.
    """
    p_success = np.array([0.9, 0.5, 0.1])
    p_failure = np.array([0.1, 0.5, 0.9])
    p_invalid = np.array([0.0, 0.6, 0.0])

    explicit, _ = fn(p_success, p_failure, p_invalid, binary_outcomes=False, **kwargs)
    implicit, _ = fn(p_success, p_failure, p_invalid, **kwargs)
    np.testing.assert_array_equal(explicit, implicit)
    assert -2 in set(implicit.tolist()), "ternary path must still flag invalid"
