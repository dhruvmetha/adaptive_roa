"""Faithful Part-X runner (spec docs/superpowers/specs/2026-09-11-partx-faithful-stochastic-design.md).

Takes the same overrides as run_adaptive.py, e.g.

    python scripts/run_partx_faithful.py system=pendulum_stoch noise_level=low \
        initial_train_size=100 samples_per_epoch=100 n_epochs=20 seed=42 \
        output_dir=/common/users/shared/pracsys/adaptive_roa_experiments/gaussian_torque/pen_low_partx_faithful

Re-running the same command resumes from output_dir/partx_state.pkl.
"""
from __future__ import annotations

import os

import hydra
from omegaconf import DictConfig, OmegaConf

from adaptive_roa.utils.env_config import (
    get_data_dir,
    get_env_config,
    get_exp_dir,
    get_net_id,
    get_shared_data_base,
)

# Same resolvers as scripts/run_adaptive.py, so every shared config resolves.
if not OmegaConf.has_resolver("net_id"):
    OmegaConf.register_new_resolver("net_id", lambda: get_net_id())
if not OmegaConf.has_resolver("exp_dir"):
    OmegaConf.register_new_resolver("exp_dir", lambda: get_exp_dir())
if not OmegaConf.has_resolver("data_dir"):
    OmegaConf.register_new_resolver("data_dir", lambda: get_data_dir())
if not OmegaConf.has_resolver("shared_data_base"):
    OmegaConf.register_new_resolver("shared_data_base", lambda default="": get_shared_data_base() or default)
if not OmegaConf.has_resolver("env"):
    OmegaConf.register_new_resolver(
        "env", lambda key, default="": os.environ.get(key, get_env_config().get(key, default)))


@hydra.main(config_path="../configs/adaptive_v2", config_name="partx_faithful", version_base=None)
def main(cfg: DictConfig):
    from adaptive_roa.partx_faithful.runner import run_from_cfg

    print(OmegaConf.to_yaml(cfg.partx_faithful))
    run_from_cfg(cfg)


if __name__ == "__main__":
    main()
