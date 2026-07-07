"""Tests for the partial-trajectory dataset-description parser."""
from pathlib import Path

import pytest

from adaptive_roa.partial_trajs.data.description import load_dataset_description

BASE = Path(
    "/common/users/shared/pracsys/genMoPlan/data_trajectories/partial_deterministic"
)
PENDULUM_T25 = BASE / "pendulum_lqr" / "pendulum_lqr_50k_T25"
QUAD3D_T25 = BASE / "quadrotor3D_lqr" / "quadrotor3D_lqr_T25"

pytestmark = pytest.mark.skipif(
    not BASE.exists(), reason="shared partial-trajectory dataset not available"
)


def test_loads_pendulum_horizon_and_system_metadata():
    desc = load_dataset_description(PENDULUM_T25)
    # horizon model
    assert desc.horizon_T == 25
    assert desc.autoregressive_K == 20
    assert desc.resolution_l == 494
    # source system
    assert desc.state_dim == 2
    assert desc.state_order == ["theta", "theta_dot"]
    assert desc.goal_state == [0, 0]
    assert desc.angle_dims == [0]
    assert desc.quaternion_dims is None
    # eval protocol
    assert desc.in_sample is True


def test_loads_quaternion_dims_for_quad3d():
    desc = load_dataset_description(QUAD3D_T25)
    assert desc.state_dim == 13
    assert desc.horizon_T == 25
    assert desc.quaternion_dims == [3, 4, 5, 6]
    assert desc.angle_dims == []


def test_accepts_json_path_directly():
    desc = load_dataset_description(PENDULUM_T25 / "dataset_description.json")
    assert desc.horizon_T == 25
