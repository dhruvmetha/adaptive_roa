"""Unified adaptive training entrypoint for v2.

Examples:
    # Use system defaults (default family is unet per system config)
    python scripts/run_adaptive.py system=pendulum

    # Switch model family while keeping system-specific dimensions
    python scripts/run_adaptive.py system=pendulum model/family=adaln
    python scripts/run_adaptive.py system=cartpole_pybullet model/family=dit

    # Compose and inspect final resolved config
    python scripts/run_adaptive.py --cfg job --resolve system=quadrotor3d model/family=simple_mlp
"""

from __future__ import annotations

import os

import hydra
from omegaconf import DictConfig, OmegaConf

from adaptive_roa.adaptive_v2.engine import AdaptiveEngine
from adaptive_roa.utils.env_config import (
    get_data_dir,
    get_env_config,
    get_exp_dir,
    get_net_id,
    get_shared_data_base,
)


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
        "env",
        lambda key, default="": os.environ.get(key, get_env_config().get(key, default)),
    )


@hydra.main(config_path="../configs/adaptive_v2", config_name="default", version_base=None)
def main(cfg: DictConfig):
    print("=" * 70)
    print(f"UNIFIED ADAPTIVE PIPELINE (v2) - {cfg.adaptive_v2.system_name}")
    print("=" * 70)
    print(OmegaConf.to_yaml(cfg))

    engine = AdaptiveEngine(cfg)
    engine.run()


if __name__ == "__main__":
    main()
