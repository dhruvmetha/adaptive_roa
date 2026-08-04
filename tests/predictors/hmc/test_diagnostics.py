import pytest
import torch

from adaptive_roa.predictors.hmc.diagnostics import (
    agreement,
    function_space_rhat,
    hmc_vs_hmc_ceiling,
    total_variation,
)


def test_rhat_is_near_one_for_chains_from_one_distribution():
    torch.manual_seed(0)
    preds = torch.rand(4, 200, 30)  # [chains, draws, points]
    r = function_space_rhat(preds)
    assert r.shape == (30,)
    assert r.max().item() < 1.1


def test_rhat_flags_chains_that_disagree():
    torch.manual_seed(0)
    preds = torch.rand(4, 200, 30) * 0.1
    preds[0] += 5.0  # one chain sampling somewhere else entirely
    assert function_space_rhat(preds).max().item() > 1.2


def test_rhat_needs_at_least_two_chains():
    with pytest.raises(ValueError, match="chains"):
        function_space_rhat(torch.rand(1, 100, 10))


def test_agreement_is_one_for_identical_predictions():
    p = torch.tensor([0.9, 0.2, 0.6])
    assert agreement(p, p.clone()) == pytest.approx(1.0)


def test_agreement_counts_matching_hard_labels():
    p = torch.tensor([0.9, 0.2, 0.6, 0.4])
    q = torch.tensor([0.8, 0.3, 0.4, 0.6])  # last two cross 0.5
    assert agreement(p, q) == pytest.approx(0.5)


def test_total_variation_is_zero_for_identical_and_one_for_opposite():
    p = torch.tensor([0.7, 0.3])
    assert total_variation(p, p.clone()) == pytest.approx(0.0)
    assert total_variation(torch.tensor([1.0, 0.0]),
                           torch.tensor([0.0, 1.0])) == pytest.approx(1.0)


def test_ceiling_reports_below_one_for_finite_chains():
    """The ceiling is what an approximation is actually competing against; it is
    below 1.0 because HMC's own chains differ at finite sample size."""
    torch.manual_seed(0)
    preds = torch.sigmoid(torch.randn(3, 150, 40))
    out = hmc_vs_hmc_ceiling(preds)
    assert set(out) >= {"agreement", "total_variation", "n_chains"}
    assert 0.0 < out["agreement"] <= 1.0
    assert out["n_chains"] == 3


def test_ceiling_needs_at_least_two_chains():
    with pytest.raises(ValueError, match="chains"):
        hmc_vs_hmc_ceiling(torch.rand(1, 50, 10))
