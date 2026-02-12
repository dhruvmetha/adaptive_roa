"""
Shared utilities for loading checkpoints and Hydra configs in flow matcher models.
"""

from pathlib import Path
from typing import Optional


def find_training_dir(checkpoint_path: Path) -> Path:
    """
    Determine the training root directory from a resolved checkpoint file path.

    Handles two directory layouts:
      - v2 adaptive: epoch_XXX/checkpoints/best.ckpt  -> training_dir = epoch_XXX
      - legacy standalone: .../version_0/checkpoints/best.ckpt -> training_dir = parent of version_0
    """
    if checkpoint_path.parent.name == "checkpoints":
        potential_version_dir = checkpoint_path.parent.parent
        if potential_version_dir.name.startswith("version_"):
            return potential_version_dir.parent
        return potential_version_dir
    return checkpoint_path.parent


def load_hydra_config(training_dir: Path) -> Optional[dict]:
    """
    Search for and load Hydra config from training_dir or its ancestors.

    The adaptive loop stores .hydra in the top-level output dir, but
    engine.py also copies it into each epoch directory. This function
    checks the given directory and up to 2 parent levels.

    Returns:
        Resolved config dict, or None if not found / failed to load.
    """
    from omegaconf import OmegaConf

    # Register resolvers if needed (config may contain ${data_dir}, ${net_id}, etc.)
    try:
        from adaptive_roa.utils.env_config import get_net_id, get_exp_dir, get_data_dir
        if not OmegaConf.has_resolver("net_id"):
            OmegaConf.register_new_resolver("net_id", lambda: get_net_id())
        if not OmegaConf.has_resolver("exp_dir"):
            OmegaConf.register_new_resolver("exp_dir", lambda: get_exp_dir())
        if not OmegaConf.has_resolver("data_dir"):
            OmegaConf.register_new_resolver("data_dir", lambda: get_data_dir())
    except ImportError:
        pass
    if not OmegaConf.has_resolver("now"):
        OmegaConf.register_new_resolver("now", lambda fmt="": "")

    # Search current dir and up to 2 parent levels
    hydra_config_path = None
    search_dir = training_dir
    for _ in range(3):
        candidate = search_dir / ".hydra" / "config.yaml"
        if candidate.exists():
            hydra_config_path = candidate
            break
        search_dir = search_dir.parent

    if hydra_config_path is None:
        print(f"   Hydra config not found in {training_dir} or parent directories")
        return None

    try:
        print(f"   Loading Hydra config: {hydra_config_path}")
        omega_cfg = OmegaConf.load(hydra_config_path)
        hydra_config = OmegaConf.to_container(omega_cfg, resolve=True)
        print(f"   Hydra config loaded successfully")
        return hydra_config
    except Exception as e:
        print(f"   Warning: Could not load Hydra config: {e}")
        return None
