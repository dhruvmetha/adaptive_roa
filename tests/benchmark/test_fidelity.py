import numpy as np
import pytest
from adaptive_roa.benchmark.fidelity import fidelity_vs_reference


CONVERGED = {"rhat_max": 1.02, "converged": True}
DIVERGED = {"rhat_max": 91.4, "converged": False}


def test_returns_a_number_when_the_reference_converged():
    a = np.array([0.9, 0.1, 0.8]); r = np.array([0.85, 0.15, 0.75])
    res = fidelity_vs_reference(a, r, CONVERGED)
    assert res.available
    assert res.total_variation == pytest.approx(0.05, abs=1e-9)


def test_withholds_the_number_when_the_reference_did_not_converge():
    a = np.array([0.9, 0.1]); r = np.array([0.85, 0.15])
    res = fidelity_vs_reference(a, r, DIVERGED)
    assert not res.available
    assert res.agreement is None and res.total_variation is None
    assert "rhat" in res.reason.lower()


def test_the_withheld_case_still_names_the_measured_value():
    res = fidelity_vs_reference(np.array([0.9]), np.array([0.8]), DIVERGED)
    assert "91.4" in res.reason


def test_identical_posteriors_score_perfectly():
    p = np.array([0.3, 0.7, 0.5])
    res = fidelity_vs_reference(p, p, CONVERGED)
    assert res.total_variation == pytest.approx(0.0)
    assert res.agreement == pytest.approx(1.0)


def test_a_missing_diagnostic_is_refused_rather_than_assumed_converged():
    with pytest.raises(ValueError, match="diagnostic"):
        fidelity_vs_reference(np.array([0.9]), np.array([0.8]), {})


def test_threshold_is_honoured():
    marginal = {"rhat_max": 1.05, "converged": True}
    assert fidelity_vs_reference(np.array([0.9]), np.array([0.8]), marginal,
                                 rhat_threshold=1.01).available is False


def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape"):
        fidelity_vs_reference(np.array([0.9, 0.1]), np.array([0.8]), CONVERGED)
