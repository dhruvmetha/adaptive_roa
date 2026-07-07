"""Tests for backend selection in train.build_model."""
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from adaptive_roa.partial_trajs.systems import make_verifier_system
from adaptive_roa.partial_trajs.train import build_model
from adaptive_roa.partial_trajs.model.regressor import DynamicsRegressor
from adaptive_roa.partial_trajs.model.generative import GenerativeDynamics

DATA = Path(
    "/common/users/shared/pracsys/genMoPlan/data_trajectories/deterministic/pendulum_lqr_50k"
)

pytestmark = pytest.mark.skipif(not DATA.exists(), reason="pendulum bounds not available")


@pytest.fixture(scope="module")
def system():
    return make_verifier_system("pendulum")


def test_deterministic_backend(system):
    cfg = OmegaConf.create({"backend": "deterministic", "hidden_dims": [16], "lr": 1e-3})
    assert isinstance(build_model(cfg, system), DynamicsRegressor)


def test_generative_backend(system):
    cfg = OmegaConf.create(
        {"backend": "generative", "hidden_dims": [16], "lr": 1e-3, "num_integration_steps": 5}
    )
    assert isinstance(build_model(cfg, system), GenerativeDynamics)


def test_unknown_backend_raises(system):
    cfg = OmegaConf.create({"backend": "nope", "hidden_dims": [16], "lr": 1e-3})
    with pytest.raises(ValueError):
        build_model(cfg, system)
