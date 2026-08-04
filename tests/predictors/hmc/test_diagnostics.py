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


def test_ceiling_is_high_for_agreeing_chains_and_low_for_disagreeing_ones():
    """The ceiling must actually measure inter-chain agreement. Asserting only
    that it lies in [0, 1] is vacuous -- agreement() is a mean of booleans and
    cannot leave that range, so a leave-one-out bug comparing a chain against
    itself would pass unnoticed."""
    torch.manual_seed(0)
    # Chains sampling the same predictive: agreement should be near 1.
    tight = torch.sigmoid(torch.randn(1, 200, 60) * 0.3 + 2.0).repeat(3, 1, 1)
    tight = tight + torch.randn_like(tight) * 0.01
    out_tight = hmc_vs_hmc_ceiling(tight.clamp(0, 1))
    assert out_tight["agreement"] > 0.9

    # Chains on opposite sides of the 0.5 decision boundary: agreement must drop.
    #
    # Deliberately 0.9 vs. 0.2, not 0.9 vs. 0.1: 0.9 + 0.1 sums to exactly
    # 1.0, so a leave-one-out fold that averages one 0.9-chain with one
    # 0.1-chain lands exactly on the 0.5 decision boundary, and which side
    # of that boundary the reduction rounds to then depends on dtype and
    # reduction order (a natively-float64 mean() rounds up through 0.5 here;
    # a float32 mean() cast up to float64 carries its already-rounded-down
    # float32 result instead -- same nominal fixture, different outcome).
    # 0.9 vs. 0.2 keeps every rest-mean comfortably away from 0.5 (0.2, or
    # (0.9+0.2)/2 = 0.55), so every fold shows genuine disagreement
    # regardless of dtype or reduction order. Measured: agreement ==
    # agreement_min == 0.0 -- all three leave-one-out folds disagree, so
    # this fixture demonstrates "low ceiling", not the mean-dilutes-a-single
    # bad-fold case (that needs at least one fold that *does* agree, which
    # requires a 4th, non-adversarial chain to set up without reintroducing
    # a boundary-adjacent rest-mean).
    split = torch.cat([
        torch.full((1, 200, 60), 0.9),
        torch.full((2, 200, 60), 0.2),
    ], dim=0)
    out_split = hmc_vs_hmc_ceiling(split)
    assert set(out_split) >= {
        "agreement", "agreement_min", "total_variation",
        "total_variation_max", "n_chains",
    }
    assert out_split["agreement"] == pytest.approx(0.0)
    assert out_split["agreement_min"] == pytest.approx(0.0)
    assert out_split["agreement_min"] <= out_split["agreement"]
    assert out_split["n_chains"] == 3


def test_ceiling_needs_at_least_two_chains():
    with pytest.raises(ValueError, match="chains"):
        hmc_vs_hmc_ceiling(torch.rand(1, 50, 10))
