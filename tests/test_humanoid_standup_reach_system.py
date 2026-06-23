# tests/test_humanoid_standup_reach_system.py
import numpy as np
import torch
import pytest

from adaptive_roa.systems.humanoid_standup_reach import HumanoidStandUpReachSystem

DATASET_DIR = "/common/users/shared/pracsys/genMoPlan/data_trajectories/humanoid_get_up_medium"


@pytest.fixture(scope="module")
def system():
    return HumanoidStandUpReachSystem(dataset_dir=DATASET_DIR)


def test_state_dim_and_manifold_structure(system):
    assert system.state_dim == 67
    comps = system.define_manifold_structure()
    assert sum(c.dim for c in comps) == 67
    sphere = [c for c in comps if c.manifold_type == "Sphere"]
    assert len(sphere) == 1 and sphere[0].dim == 3


def test_bounds_loaded_and_sphere_identity(system):
    assert system._norm_center.shape == (67,)
    assert system._norm_half.shape == (67,)
    # Sphere dims are identity (center 0, half 1) so normalization leaves them unit-norm
    assert torch.allclose(system._norm_center[34:37], torch.zeros(3))
    assert torch.allclose(system._norm_half[34:37], torch.ones(3))


def test_normalize_denormalize_roundtrip(system):
    torch.manual_seed(0)
    raw = torch.randn(16, 67) * 3.0
    raw = system.project_to_manifold(raw)  # make sphere block unit-norm
    back = system.denormalize_state(system.normalize_state(raw))
    assert torch.allclose(back, raw, atol=1e-4)


def test_normalize_keeps_sphere_block_unchanged(system):
    raw = system.project_to_manifold(torch.randn(8, 67))
    norm = system.normalize_state(raw)
    assert torch.allclose(norm[:, 34:37], raw[:, 34:37], atol=1e-6)


def test_classify_attractor_thresholds(system):
    state = torch.zeros(4, 67)
    # row0: success (head=1.3 exactly, com speed 0)
    state[0, 21] = 1.3
    # row1: head too low
    state[1, 21] = 1.29
    # row2: head ok but com speed too high (0.21 along one axis)
    state[2, 21] = 1.5; state[2, 37] = 0.21
    # row3: head ok, com speed exactly 0.2 -> success
    state[3, 21] = 1.5; state[3, 37] = 0.2
    labels = system.classify_attractor(state)
    assert labels.tolist() == [1, -1, -1, 1]
    assert labels.dtype == torch.long


def test_classify_attractor_accepts_numpy_and_1d(system):
    s = np.zeros(67, dtype=np.float32); s[21] = 1.4
    out = system.classify_attractor(s)
    assert int(out.item() if out.dim() == 0 else out[0]) == 1


def test_get_loss_weights_shape(system):
    w = system.get_loss_weights()
    assert w.shape == (67,)
    assert torch.allclose(w[34:37], torch.ones(3))
