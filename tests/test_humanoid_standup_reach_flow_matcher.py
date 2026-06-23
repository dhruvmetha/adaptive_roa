# tests/test_humanoid_standup_reach_flow_matcher.py
import torch
import pytest
from pathlib import Path

from adaptive_roa.systems.humanoid_standup_reach import HumanoidStandUpReachSystem
from adaptive_roa.model.quadrotor3d_unet import Quadrotor3DUNet
from adaptive_roa.flow_matching.humanoid_standup_reach.latent_conditional.flow_matcher import (
    HumanoidStandUpReachLatentConditionalFlowMatcher,
)

DATASET_DIR = "/common/users/shared/pracsys/genMoPlan/data_trajectories/humanoid_get_up_medium"

pytestmark = pytest.mark.skipif(not Path(DATASET_DIR).exists(), reason="shared humanoid dataset not available")


def _build(use_manifold=True):
    system = HumanoidStandUpReachSystem(dataset_dir=DATASET_DIR)
    model = Quadrotor3DUNet(embedded_dim=67, latent_dim=8, condition_dim=67,
                            time_emb_dim=64, hidden_dims=[128, 128], output_dim=67)
    fm = HumanoidStandUpReachLatentConditionalFlowMatcher(
        system=system, model=model, optimizer=None, scheduler=None,
        latent_dim=8, use_manifold=use_manifold)
    return system, fm


def test_manifold_dist_has_65_components():
    _, fm = _build(use_manifold=True)
    a = torch.randn(4, 67); a = fm.system.project_to_manifold(a)
    b = torch.randn(4, 67); b = fm.system.project_to_manifold(b)
    assert fm.manifold.dist(a, b).shape == (4, 65)


def test_distance_manifold_is_spherical_even_when_euclidean_training():
    _, fm = _build(use_manifold=False)
    a = fm.system.project_to_manifold(torch.randn(4, 67))
    b = fm.system.project_to_manifold(torch.randn(4, 67))
    # distance manifold must still be the S²-aware product (65 components)
    assert fm.distance_manifold.dist(a, b).shape == (4, 65)


def test_component_names_count():
    _, fm = _build()
    assert len(fm.get_manifold_component_names()) == 65


def test_sample_noisy_input_on_manifold():
    _, fm = _build()
    noise = fm.sample_noisy_input(8, torch.device("cpu"))
    assert noise.shape == (8, 67)
    assert torch.allclose(noise[:, 34:37].norm(dim=1), torch.ones(8), atol=1e-5)


def test_predict_endpoint_shape_and_unit_sphere():
    _, fm = _build()
    start = fm.system.project_to_manifold(torch.randn(2, 67))
    pred = fm.predict_endpoint(start, num_steps=5)
    assert pred.shape == (2, 67)
    assert torch.allclose(pred[:, 34:37].norm(dim=1), torch.ones(2), atol=1e-4)


def test_predict_endpoint_euclidean_mode():
    _, fm = _build(use_manifold=False)
    start = fm.system.project_to_manifold(torch.randn(2, 67))
    pred = fm.predict_endpoint(start, num_steps=5)
    assert pred.shape == (2, 67)
    assert torch.allclose(pred[:, 34:37].norm(dim=1), torch.ones(2), atol=1e-4)


def test_manifold_dist_dim_matches_metric_count():
    _, fm = _build(use_manifold=True)
    a = fm.system.project_to_manifold(torch.randn(4, 67))
    b = fm.system.project_to_manifold(torch.randn(4, 67))
    assert fm._manifold_dist_dim == 65
    assert len(fm.val_endpoint_mae_per_dim) == 65
    assert fm.manifold.dist(a, b).shape[1] == 65


def test_compute_flow_loss_runs():
    _, fm = _build()
    batch = {
        "start_state": fm.system.project_to_manifold(torch.randn(4, 67)),
        "end_state": fm.system.project_to_manifold(torch.randn(4, 67)),
    }
    loss = fm.compute_flow_loss(batch)
    assert loss.ndim == 0 and torch.isfinite(loss)
