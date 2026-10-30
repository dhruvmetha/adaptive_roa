import numpy as np
import pytest
import torch

from adaptive_roa.partx_faithful.readout import PiecewiseHandle, PiecewiseModel
from adaptive_roa.partx_faithful.region import Region
from adaptive_roa.partx_faithful.volume import (classified_falsification_volume,
                                                gp_quantile_falsification_volume,
                                                remaining_plus_violating_volume)


def _leaf(rid, low, high, rtype):
    return Region(rid=rid, parent=-1, low=np.array(low, float), high=np.array(high, float),
                  branch_dir=0, depth=0, rtype=rtype)


class _G:
    def __init__(self, m, s=0.01, p=0.5):
        self.m, self.s, self.p = m, s, p

    def latent(self, X):
        return np.full(len(X), self.m), np.full(len(X), self.s)

    def p_success(self, X):
        return np.full(len(np.atleast_2d(X)), self.p)


def test_classified_and_px_volumes():
    leaves = [_leaf(1, [0, 0], [0.5, 1], "-"), _leaf(2, [0.5, 0], [0.75, 1], "r"),
              _leaf(3, [0.75, 0], [1, 1], "+")]
    assert classified_falsification_volume(leaves, 1.0) == pytest.approx(0.5)
    assert remaining_plus_violating_volume(leaves, 1.0) == pytest.approx(0.75)


def test_gp_quantile_volume_counts_confident_negatives():
    leaves = [_leaf(1, [0, 0], [0.5, 1], "-"), _leaf(2, [0.5, 0], [1, 1], "+"),
              _leaf(3, [0, 0], [0, 0], "u")]
    models = {1: _G(-1.0), 2: _G(1.0), 3: _G(-1.0)}
    fv = gp_quantile_falsification_volume(leaves, models, 2, 20, [0.5, 0.05], np.random.default_rng(0), 1.0)
    assert fv[0.5] == pytest.approx(0.5) and fv[0.05] == pytest.approx(0.5)


def test_readout_uses_each_leafs_model_and_clips_outside_points():
    leaves = [_leaf(1, [0, 0], [0.5, 1], "-"), _leaf(2, [0.5, 0], [1, 1], "+")]
    pw = PiecewiseModel(leaves, {1: _G(0, p=0.2), 2: _G(0, p=0.9)}, np.zeros(2), np.ones(2))
    np.testing.assert_allclose(pw.p_success(np.array([[0.25, 0.5], [0.75, 0.5], [5.0, 0.5]])),
                               [0.2, 0.9, 0.9])


def test_handle_returns_logits_that_sigmoid_back_to_p():
    leaves = [_leaf(1, [0, 0], [1, 1], "r")]
    handle = PiecewiseHandle(PiecewiseModel(leaves, {1: _G(0, p=0.3)}, np.zeros(2), np.ones(2)))
    out = handle.eval().to("cpu")(torch.tensor([[0.5, 0.5]], dtype=torch.float32))
    assert torch.sigmoid(out).item() == pytest.approx(0.3, abs=1e-6)
