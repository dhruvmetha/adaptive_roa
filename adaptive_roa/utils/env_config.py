"""
Environment configuration utilities for adaptive_cartpole.

Loads user-specific paths from .env file so that the codebase can work
across different users without hardcoding netIDs.

Usage:
    from adaptive_roa.utils.env_config import get_env_config, get_path

    # Get the full config
    config = get_env_config()

    # Get interpolated paths
    exp_dir = get_path("EXP_DIR")
    data_dir = get_path("DATA_DIR")
"""

import os
from pathlib import Path
from typing import Optional
from functools import lru_cache


def _find_project_root() -> Path:
    """Find the project root by looking for .env or .git."""
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if (parent / ".env").exists() or (parent / ".git").exists():
            return parent
    # Fallback to src's parent
    return Path(__file__).resolve().parent.parent.parent


def _load_dotenv(env_path: Path) -> dict:
    """
    Simple .env file loader (no external dependencies).
    Handles basic variable interpolation like ${NET_ID}.
    """
    env_vars = {}
    if not env_path.exists():
        return env_vars

    with open(env_path, "r") as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if not line or line.startswith("#"):
                continue
            # Parse key=value
            if "=" in line:
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()
                # Remove quotes if present
                if (value.startswith('"') and value.endswith('"')) or \
                   (value.startswith("'") and value.endswith("'")):
                    value = value[1:-1]
                env_vars[key] = value

    # Now interpolate variables
    def interpolate(s: str, vars_dict: dict) -> str:
        """Replace ${VAR} with actual values."""
        import re
        pattern = r'\$\{([^}]+)\}'

        def replacer(match):
            var_name = match.group(1)
            # First check our loaded vars, then OS env
            return vars_dict.get(var_name, os.environ.get(var_name, match.group(0)))

        # Keep interpolating until no more changes (handles nested refs)
        prev = None
        while prev != s:
            prev = s
            s = re.sub(pattern, replacer, s)
        return s

    # Interpolate all values
    interpolated = {}
    for key, value in env_vars.items():
        interpolated[key] = interpolate(value, env_vars)

    return interpolated


@lru_cache(maxsize=1)
def get_env_config() -> dict:
    """
    Load and return the environment configuration.

    Loads from .env file in project root. Results are cached.

    Returns:
        Dictionary with environment variables and their interpolated values.
    """
    project_root = _find_project_root()
    env_path = project_root / ".env"

    config = _load_dotenv(env_path)

    # Also add some computed defaults if not present
    if "NET_ID" not in config:
        # Try to get from OS environment or fall back to current user
        config["NET_ID"] = os.environ.get("NET_ID", os.environ.get("USER", "unknown"))

    if "USER_BASE" not in config:
        config["USER_BASE"] = "/common/users/shared/pracsys/genMoPlan/global_dynamics_experiments"

    if "EXP_DIR" not in config:
        config["EXP_DIR"] = str(project_root)

    return config


def get_path(key: str, default: Optional[str] = None) -> str:
    """
    Get a path from the environment configuration.

    Args:
        key: The environment variable name (e.g., "EXP_DIR", "DATA_DIR")
        default: Default value if key not found

    Returns:
        The interpolated path string
    """
    config = get_env_config()
    return config.get(key, default or "")


def get_net_id() -> str:
    """Get the current user's netID from environment config."""
    return get_env_config().get("NET_ID", os.environ.get("USER", "unknown"))


def get_exp_dir() -> str:
    """
    Get the experiment output directory from environment config.

    This is where experiment outputs are saved.
    Used by Hydra configs via ${exp_dir:} resolver.
    """
    default = str(_find_project_root())
    return get_env_config().get("EXP_DIR", default)




def get_data_dir() -> str:
    """
    Get the data directory from environment config.

    This is where trajectory data lives (read-only).
    Used by Hydra configs via ${data_dir:} resolver.
    """
    default = "/common/users/shared/pracsys/genMoPlan/data_trajectories"
    return get_env_config().get("DATA_DIR", default)


def get_shared_data_base() -> str:
    """
    Get the shared data base directory.

    This is the base directory for shared trajectory data.
    Used by Hydra configs via ${shared_data_base:} resolver.

    Returns:
        Path to shared data base (default: /common/users/shared/pracsys/genMoPlan/data_trajectories)
    """
    default = "/common/users/shared/pracsys/genMoPlan/data_trajectories"
    return get_env_config().get("SHARED_DATA_BASE", default)


def get_noise_regime() -> str:
    """
    Get the dataset noise regime ("deterministic" or "noisy").

    Trajectory data is organized under a regime subdirectory:
        {data_dir}/{noise_regime}/{dataset_name}/...

    Direct (non-Hydra) instantiation reads this from the NOISE_REGIME env var
    (or the .env file); Hydra pipelines use the ``noise_regime`` config value
    instead. Both default to "deterministic".
    """
    return os.environ.get("NOISE_REGIME") or get_env_config().get(
        "NOISE_REGIME", "deterministic"
    )




def get_user_path(*parts: str) -> str:
    """
    Construct a path under the user's directory.

    Args:
        *parts: Path components to join after /common/users/{NET_ID}/

    Returns:
        Full path string

    Example:
        get_user_path("arcmg_datasets", "cartpole")
        # Returns: /common/users/{net_id}/arcmg_datasets/cartpole
    """
    config = get_env_config()
    user_base = config.get("USER_BASE", "/common/users")
    net_id = get_net_id()
    return str(Path(user_base) / net_id / Path(*parts))


def get_arcmg_path(*parts: str) -> str:
    """
    Construct a path under the user's arcmg_datasets directory.

    Args:
        *parts: Path components to join after /common/users/{NET_ID}/arcmg_datasets/

    Returns:
        Full path string

    Example:
        get_arcmg_path("cartpole", "data_bounds.pkl")
        # Returns: /common/users/{net_id}/arcmg_datasets/cartpole/data_bounds.pkl
    """
    return get_user_path("arcmg_datasets", *parts)


def get_exp_path(*parts: str) -> str:
    """
    Construct a path relative to the experiment output directory.

    Args:
        *parts: Path components to join after experiment directory

    Returns:
        Full path string
    """
    config = get_env_config()
    exp_dir = config.get("EXP_DIR", str(_find_project_root()))
    return str(Path(exp_dir) / Path(*parts)) if parts else exp_dir




def get_data_path(*parts: str) -> str:
    """
    Construct a path under the data directory.

    Args:
        *parts: Path components to join after data directory

    Returns:
        Full path string
    """
    config = get_env_config()
    data_dir = config.get("DATA_DIR", "/common/users/shared/pracsys/genMoPlan/data_trajectories")
    return str(Path(data_dir) / Path(*parts)) if parts else data_dir




# Convenience function for Hydra config resolvers
def resolve_user_path(relative_path: str) -> str:
    """
    Resolve a path that contains ${NET_ID} placeholder.

    This can be registered as a Hydra resolver:
        OmegaConf.register_new_resolver("user_path", resolve_user_path)

    Args:
        relative_path: Path with ${NET_ID} placeholders

    Returns:
        Path with NET_ID replaced
    """
    net_id = get_net_id()
    return relative_path.replace("${NET_ID}", net_id)
