"""Tests for the generative (conditional flow-matching) dynamics backend."""
from pathlib import Path

import pytest
import torch

from adaptive_roa.systems.pendulum import PendulumSystem
from adaptive_roa.partial_trajs.model.generative import GenerativeDynamics

DATA = Path(
    "/common/users/shared/pracsys/genMoPlan/data_trajectories/deterministic/pendulum_lqr_50k"
)

pytestmark = pytest.mark.skipif(
    not DATA.exists(), reason="shared pendulum dataset (for system bounds) not available"
)


@pytest.fixture(scope="module")
def system():
    return PendulumSystem()


def test_sample_and_predict_shapes(system):
    model = GenerativeDynamics(system, hidden_dims=[32, 32], num_integration_steps=5)
    x = torch.zeros(3, system.state_dim)
    samples = model.sample(x, num_samples=4)
    assert samples.shape == (4, 3, system.state_dim)
    assert model.predict(x).shape == (3, system.state_dim)


def test_training_step_returns_scalar_loss(system):
    model = GenerativeDynamics(system, hidden_dims=[32, 32])
    batch = {
        "x_start": torch.randn(8, system.state_dim),
        "x_end": torch.randn(8, system.state_dim),
    }
    loss = model.training_step(batch, 0)
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_learns_to_transport_samples_toward_targets(system):
    torch.manual_seed(0)
    model = GenerativeDynamics(
        system, hidden_dims=[64, 64], num_integration_steps=20, lr=1e-2
    )
    x_start = torch.randn(4, system.state_dim)
    x_end = torch.randn(4, system.state_dim)
    batch = {"x_start": x_start, "x_end": x_end}
    opt = model.configure_optimizers()

    def sample_mse():
        with torch.no_grad():
            mean = model.sample(x_start, num_samples=16).mean(dim=0)
        return ((mean - x_end) ** 2).mean().item()

    init = sample_mse()
    for _ in range(600):
        opt.zero_grad()
        loss = model.training_step(batch, 0)
        loss.backward()
        opt.step()
    final = sample_mse()
    assert final < init * 0.5  # samples move toward the targets
