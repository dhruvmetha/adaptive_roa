import numpy as np
import pytest

from adaptive_roa.adaptive_v2.strategy.uncertainty_scores import (
    SCORE_MODES, aleatoric_uncertainty, binary_entropy, epistemic_bald,
    epistemic_variance, score_by_mode, total_uncertainty,
)


def test_binary_entropy_endpoints_and_peak():
    e = binary_entropy(np.array([0.0, 1.0, 0.5]))
    assert e[0] == pytest.approx(0.0)
    assert e[1] == pytest.approx(0.0)
    assert e[2] == pytest.approx(np.log(2))


def test_decomposition_identity_is_exact_when_members_are_noise_free():
    # total = aleatoric + epistemic_bald must hold exactly (classifier case)
    rng = np.random.default_rng(0)
    p = rng.uniform(0.01, 0.99, size=(5, 200))
    lhs = total_uncertainty(p)
    rhs = aleatoric_uncertainty(p) + epistemic_bald(p)
    np.testing.assert_allclose(lhs, rhs, atol=1e-12)


def test_agreeing_members_give_zero_epistemic():
    p = np.tile(np.array([0.1, 0.5, 0.9]), (5, 1))       # all members identical
    np.testing.assert_allclose(epistemic_bald(p), 0.0, atol=1e-12)
    np.testing.assert_allclose(epistemic_variance(p, k=None), 0.0, atol=1e-12)


def test_disagreeing_members_give_large_epistemic():
    p = np.array([[0.2, 0.2], [0.2, 0.2], [0.8, 0.8], [0.8, 0.8], [0.5, 0.5]])
    assert epistemic_bald(p).min() > 0.1
    assert epistemic_variance(p, k=None).min() > 0.05


def test_epistemic_variance_debias_removes_finite_k_noise():
    # Members agree in TRUTH; each p_hat is a K-sample binomial estimate.
    # Debiased variance must be ~0; raw variance must not be.
    rng = np.random.default_rng(0)
    M, K, N = 5, 20, 20000
    for p_true in (0.5, 0.3, 0.1):
        ph = rng.binomial(K, p_true, size=(M, N)) / K
        assert abs(epistemic_variance(ph, k=K).mean()) < 5e-4
        assert epistemic_variance(ph, k=None).mean() > 3e-3   # undebiased is inflated


def test_naive_bald_bias_is_the_documented_magnitude():
    # Asserts the bias that motivates epistemic_var, rather than assuming it.
    rng = np.random.default_rng(0)
    M, K, N = 5, 20, 20000
    ph = rng.binomial(K, 0.5, size=(M, N)) / K
    expected = (1.0 / (2 * K)) * (1 - 1.0 / M)             # ~0.020 nats
    assert epistemic_bald(ph).mean() == pytest.approx(expected, rel=0.25)


def test_bald_bias_vanishes_at_a_deterministic_state():
    ph = np.zeros((5, 100))
    assert epistemic_bald(ph).mean() == pytest.approx(0.0, abs=1e-12)


def test_score_by_mode_dispatches_and_rejects_unknown():
    p = np.tile(np.array([0.3, 0.7]), (5, 1))
    for mode in SCORE_MODES:
        assert score_by_mode(mode, p, k=20).shape == (2,)
    with pytest.raises(ValueError, match="unknown score mode"):
        score_by_mode("nope", p, k=20)


def test_out_of_domain_input_raises_rather_than_returning_nan():
    with pytest.raises(ValueError, match="probabilities in .0, 1."):
        binary_entropy(np.array([-0.5, 0.5]))
    with pytest.raises(ValueError, match="probabilities in .0, 1."):
        binary_entropy(np.array([0.5, 1.5]))


def test_tiny_float_drift_is_absorbed_not_rejected():
    # values that can arise from averaging/rounding, not from a real bug
    e = binary_entropy(np.array([-1e-15, 1.0 + 1e-15]))
    np.testing.assert_allclose(e, 0.0, atol=1e-12)


def test_boundaries_remain_exact_after_the_domain_guard():
    e = binary_entropy(np.array([0.0, 1.0]))
    assert e[0] == 0.0 and e[1] == 0.0     # exactly, not approximately
