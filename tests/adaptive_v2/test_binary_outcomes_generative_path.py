"""binary_outcomes must work on the GENERATIVE eval path, not just the classifier one.

Regression test for 17bf202, which added the invalid->failure fold to
``evaluate_full_roa_fast`` but placed it ~160 lines ABOVE where p_success /
p_failure / p_invalid are assigned. The block therefore raised
``UnboundLocalError: local variable 'p_failure' referenced before assignment``
every single time it was reached -- it could never have succeeded.

It went unnoticed because the gate is narrow and the tests missed it entirely:

  * ``binary_outcomes`` is inferred from ``eval_success_prob.npz`` existing beside
    the dataset, so only stochastic datasets trip it; deterministic runs skip the
    block and are unaffected.
  * ``evaluate_epoch`` dispatches classifier-type predictors to
    ``evaluate_full_roa_classifier``, which has no fold. Only generative-type
    arms (fm, mlp_det, gp_reg, bnn_*_reg) reach the broken line.
  * every test shipped in that commit exercised the helper predicates or the
    classifier path. None called ``evaluate_full_roa_fast`` with the flag set,
    so all 10 passed against code that crashed on contact in production.

The fold is not cosmetic: on stochastic pendulum an fm run puts ~70% of its mean
probability mass in p_invalid, so whether that mass counts as failure or as a
third class changes every metric downstream.
"""
import numpy as np
import pytest

from adaptive_roa.adaptive_v2.eval.full_roa import evaluate_full_roa_fast
from adaptive_roa.adaptive_v2.eval.mc_cache import MCCache


class _BinarySystem:
    """Stands in for a system whose dataset carries eval_success_prob.npz."""
    binary_outcomes = True


def _eval_file(tmp_path, n=120):
    rng = np.random.default_rng(7)
    rows = []
    for _ in range(n):
        s0, s1 = rng.uniform(-1, 1), rng.uniform(-1, 1)
        rows.append([s0, s1, 0.0, 0.0, 1 if s0 > 0 else 0])
    f = tmp_path / "eval.txt"
    np.savetxt(str(f), np.array(rows), delimiter=",", fmt="%.6f")
    return f, np.array(rows)


def _cache_with_invalid_mass(rows, K=20):
    """MC labels carrying a real invalid (0) population, as fm does in practice."""
    n = len(rows)
    true_success = rows[:, 4] == 1
    labels = np.where(true_success[:, None], 1, -1).astype(np.int8)
    labels = np.repeat(labels, K, axis=1)
    # Half of every point's samples land near no attractor -> label 0 (invalid).
    labels[:, : K // 2] = 0
    return MCCache(
        mc_endpoints=np.zeros((n, K, 2), dtype=np.float32),
        mc_labels=labels,
        start_states=rows[:, :2].astype(np.float32),
        attractor_radius=0.2,
        num_mc_samples=K,
    )


def test_generative_path_runs_when_system_is_binary_outcome(tmp_path):
    """The exact production call that raised UnboundLocalError."""
    f, rows = _eval_file(tmp_path)
    cache = _cache_with_invalid_mass(rows)

    metrics = evaluate_full_roa_fast(
        flow_matcher=None, system=_BinarySystem(), eval_states_file=str(f),
        num_mc_samples=20, lambda_star=0.5, delta=0.1,
        decision_rule="one_sided", device="cpu", output_dir=None,
        verbose=False, mc_cache=cache,
    )
    assert metrics["threshold_free"]["n_scored"] == len(rows)


def test_binary_outcomes_folds_invalid_mass_into_failure(tmp_path):
    """Half the MC mass is invalid; under binary scoring none may remain invalid.

    Asserts the fold's EFFECT, not merely that the call returns: a fix that
    deleted the block would keep the test above green while leaving the invalid
    class alive, which is the bug the commit set out to remove.
    """
    f, rows = _eval_file(tmp_path)
    cache = _cache_with_invalid_mass(rows)
    out = tmp_path / "out"

    evaluate_full_roa_fast(
        flow_matcher=None, system=_BinarySystem(), eval_states_file=str(f),
        num_mc_samples=20, lambda_star=0.5, delta=0.1,
        decision_rule="one_sided", device="cpu", output_dir=str(out),
        verbose=False, mc_cache=cache,
    )

    d = np.load(str(out / "full_roa_per_point.npz"))
    assert np.allclose(d["p_invalid"], 0.0), "invalid mass survived binary folding"
    # mass is conserved, not discarded: it moved into p_failure.
    assert np.allclose(d["p_success"] + d["p_failure"], 1.0, atol=1e-6)


def test_non_binary_system_keeps_the_invalid_class(tmp_path):
    """The deterministic path must be untouched -- invalid stays a real class."""
    f, rows = _eval_file(tmp_path)
    cache = _cache_with_invalid_mass(rows)
    out = tmp_path / "out_ternary"

    evaluate_full_roa_fast(
        flow_matcher=None, system=None, eval_states_file=str(f),
        num_mc_samples=20, lambda_star=0.5, delta=0.1,
        decision_rule="one_sided", device="cpu", output_dir=str(out),
        verbose=False, mc_cache=cache, binary_outcomes=False,
    )

    d = np.load(str(out / "full_roa_per_point.npz"))
    assert d["p_invalid"].max() > 0.0, "ternary scoring lost its invalid class"
