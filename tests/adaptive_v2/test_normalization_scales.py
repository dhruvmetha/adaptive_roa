import math

import pytest
import torch

from adaptive_roa.systems.base import DynamicalSystem, ManifoldComponent


class _FakeSystem(DynamicalSystem):
    """SO2 angle + Real velocity bounded to [-8, 8]."""

    def define_manifold_structure(self):
        return [
            ManifoldComponent("SO2", 1, "angle"),
            ManifoldComponent("Real", 1, "angular_velocity"),
        ]

    def define_state_bounds(self):
        return {"angle": (-math.pi, math.pi), "angular_velocity": (-8.0, 8.0)}


class _MultiDimFakeSystem(DynamicalSystem):
    """3-dim Real component plus an SO2 component with no bounds entry."""

    def define_manifold_structure(self):
        return [
            ManifoldComponent("Real", 3, "position"),
            ManifoldComponent("SO2", 1, "angle"),
        ]

    def define_state_bounds(self):
        return {"position": (-2.0, 6.0)}


class _DegenerateFakeSystem(DynamicalSystem):
    """Real component whose declared bounds have zero width."""

    def define_manifold_structure(self):
        return [ManifoldComponent("Real", 1, "frozen")]

    def define_state_bounds(self):
        return {"frozen": (0.0, 0.0)}


def test_scales_shape_matches_state_dim():
    system = _FakeSystem()
    assert system.get_normalization_scales().shape == (system.state_dim,)


def test_circular_dim_scale_is_pi():
    scales = _FakeSystem().get_normalization_scales()
    assert scales[0].item() == pytest.approx(math.pi)


def test_real_dim_scale_is_half_range():
    scales = _FakeSystem().get_normalization_scales()
    assert scales[1].item() == pytest.approx(8.0)


def test_multi_dim_component_expands_and_missing_bounds_still_scale():
    scales = _MultiDimFakeSystem().get_normalization_scales()
    assert scales.shape == (4,)
    # position half-range = (6.0 - (-2.0)) / 2 = 4.0, repeated across all 3 dims
    assert scales[:3].tolist() == pytest.approx([4.0, 4.0, 4.0])
    # circular dims use pi regardless of any bounds entry
    assert scales[3].item() == pytest.approx(math.pi)


def test_zero_width_bounds_fall_back_to_one():
    scales = _DegenerateFakeSystem().get_normalization_scales()
    assert scales[0].item() == pytest.approx(1.0)


def test_all_scales_finite_and_positive():
    for system in (_FakeSystem(), _MultiDimFakeSystem(), _DegenerateFakeSystem()):
        scales = system.get_normalization_scales()
        assert torch.isfinite(scales).all()
        assert (scales > 0).all()
