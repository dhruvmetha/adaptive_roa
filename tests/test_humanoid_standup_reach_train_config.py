# tests/test_humanoid_standup_reach_train_config.py
import torch
import pytest
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf
import hydra
from pathlib import Path

from adaptive_roa.utils.env_config import get_net_id, get_exp_dir, get_shared_data_base

CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "configs")

_DATASET_DIR = "/common/users/shared/pracsys/genMoPlan/data_trajectories/humanoid_get_up_medium"
pytestmark = pytest.mark.skipif(not Path(_DATASET_DIR).exists(), reason="shared humanoid dataset not available")


def _register_resolvers():
    for name, fn in [("net_id", get_net_id), ("exp_dir", get_exp_dir), ("shared_data_base", get_shared_data_base)]:
        if not OmegaConf.has_resolver(name):
            OmegaConf.register_new_resolver(name, (lambda f: (lambda default="": f() or default))(fn))


def test_train_config_composes_and_instantiates():
    _register_resolvers()
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(config_name="train_humanoid_standup_reach")
    assert cfg.model.output_dim == 67
    assert cfg.model.embedded_dim == 67 and cfg.model.condition_dim == 67
    system = hydra.utils.instantiate(cfg.system)
    assert system.state_dim == 67
    model = hydra.utils.instantiate(cfg.model)
    fm = hydra.utils.instantiate(
        cfg.flow_matcher, system=system, model=model,
        optimizer=cfg.optimizer, scheduler=cfg.scheduler,
        model_config=OmegaConf.to_container(cfg.model, resolve=True),
        latent_dim=cfg.flow_matching.latent_dim,
        mae_val_frequency=cfg.flow_matching.mae_val_frequency,
        use_loss_weights=cfg.flow_matching.get("use_loss_weights", False),
        use_manifold=cfg.flow_matching.get("use_manifold", True),
        clamp_noise=cfg.flow_matching.get("clamp_noise", True),
        zero_latent=cfg.flow_matching.get("zero_latent", False),
        noise_scale=cfg.flow_matching.get("noise_scale", 1.0),
        _recursive_=False,
    )
    batch = {
        "start_state": system.project_to_manifold(torch.randn(4, 67)),
        "end_state": system.project_to_manifold(torch.randn(4, 67)),
    }
    loss = fm.compute_flow_loss(batch)
    assert torch.isfinite(loss)
