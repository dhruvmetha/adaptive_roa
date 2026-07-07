"""Tests for the deterministic T-step dynamics regressor."""
from pathlib import Path

import pytest
import torch

from adaptive_roa.systems.pendulum import PendulumSystem
from adaptive_roa.partial_trajs.model.regressor import DynamicsRegressor

DATA = Path(
    "/common/users/shared/pracsys/genMoPlan/data_trajectories/deterministic/pendulum_lqr_50k"
)

pytestmark = pytest.mark.skipif(
    not DATA.exists(), reason="shared pendulum dataset (for system bounds) not available"
)


@pytest.fixture(scope="module")
def system():
    return PendulumSystem()


def test_predict_returns_raw_state_shape(system):
    model = DynamicsRegressor(system, hidden_dims=[32, 32])
    x = torch.zeros(5, system.state_dim)
    out = model.predict(x)
    assert out.shape == (5, system.state_dim)


def test_training_step_returns_scalar_loss(system):
    model = DynamicsRegressor(system, hidden_dims=[32, 32])
    batch = {
        "x_start": torch.randn(4, system.state_dim),
        "x_end": torch.randn(4, system.state_dim),
    }
    loss = model.training_step(batch, 0)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_overfits_small_batch(system):
    torch.manual_seed(0)
    model = DynamicsRegressor(system, hidden_dims=[64, 64], lr=1e-2)
    x_start = torch.randn(8, system.state_dim)
    x_end = torch.randn(8, system.state_dim)
    batch = {"x_start": x_start, "x_end": x_end}
    opt = model.configure_optimizers()

    init = model.training_step(batch, 0).item()
    for _ in range(300):
        opt.zero_grad()
        loss = model.training_step(batch, 0)
        loss.backward()
        opt.step()
    final = model.training_step(batch, 0).item()
    assert final < init * 0.1  # learns the mapping
