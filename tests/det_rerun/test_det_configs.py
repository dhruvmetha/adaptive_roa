"""The four det system configs resolve to the disjoint split files for every paper seed.

Reads the shared dataset tree, so it skips where that tree is not mounted.
"""
import os
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

DATA = "/common/users/shared/pracsys/genMoPlan/data_trajectories"
CONFIG_DIR = str(Path(__file__).resolve().parents[2] / "configs/adaptive_v2")
SEEDS = (99512, 33407, 24475)
SYSTEMS = ("pendulum", "cartpole_pybullet", "quadrotor2d", "quadrotor3d")

pytestmark = pytest.mark.skipif(not os.path.isdir(f"{DATA}/deterministic"), reason="shared data not mounted")


def _register_resolvers():
    from adaptive_roa.utils.env_config import get_data_dir, get_exp_dir, get_net_id, get_shared_data_base
    for name, fn in [("net_id", get_net_id), ("exp_dir", get_exp_dir), ("shared_data_base", get_shared_data_base)]:
        if not OmegaConf.has_resolver(name):
            OmegaConf.register_new_resolver(name, (lambda f: (lambda d="": f() or d))(fn))
    if not OmegaConf.has_resolver("data_dir"):
        # Registered once per process and seen by later test modules, so fall back to
        # the .env value once monkeypatch has removed DATA_DIR again.
        OmegaConf.register_new_resolver("data_dir", lambda d="": os.environ.get("DATA_DIR") or get_data_dir() or d)


def _compose(system, seed):
    _register_resolvers()
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        return compose(config_name="default", overrides=[f"system={system}", f"seed={seed}", "output_dir=/tmp/unused"])


@pytest.mark.parametrize("system", SYSTEMS)
def test_paths_exist_and_differ_by_seed(system, monkeypatch):
    monkeypatch.setenv("DATA_DIR", DATA)
    pools = []
    for seed in SEEDS:
        ds = _compose(system, seed).data_source
        for key in ("shuffled_indices_file", "shuffled_labels_file", "cal_set_file", "val_set_file", "test_set_file"):
            assert Path(ds[key]).is_file(), f"{system} seed {seed}: {key} -> {ds[key]}"
        assert "splits_det_rerun" in ds.test_set_file and ds.timeout_intermediates == "drop"
        pools.append(Path(ds.shuffled_indices_file).read_text().split()[:50])
    assert pools[0] != pools[1] != pools[2], "the seed must change the pool order"


@pytest.mark.parametrize("system", SYSTEMS)
def test_npz_pool_matches_text_trajectories(system, monkeypatch):
    import numpy as np
    from adaptive_roa.adaptive.data_source import TrajectoryDataSourceConfig
    from adaptive_roa.adaptive.npz_data_source import NpzTrajectoryDataSource
    monkeypatch.setenv("DATA_DIR", DATA)
    ds = _compose(system, SEEDS[0]).data_source
    assert ds.pool_format == "npz"
    src = NpzTrajectoryDataSource(TrajectoryDataSourceConfig(
        trajectories_dir=ds.trajectories_dir, shuffled_indices_file=ds.shuffled_indices_file,
        shuffled_labels_file=ds.shuffled_labels_file, timeout_intermediates=ds.timeout_intermediates,
        horizon_steps=ds.get("horizon_steps", None)))
    assert set(src.labels.tolist()) <= {-1, 1}
    split_dir = Path(ds.split_dir)
    row_names = (split_dir / "npz_row_names.txt").read_text().split()
    text_dir = split_dir.parent / "trajectories"
    for k in (0, 1, 7, len(src.rollout_ids) // 2, len(src.rollout_ids) - 1):
        lines = (text_dir / row_names[src.rollout_ids[k]]).read_text().splitlines()
        expect = np.array([[float(v) for v in ln.split(",")] for ln in lines if ln.strip()], dtype=np.float32)
        assert np.array_equal(src.load_trajectory(k), expect), f"{system} pool index {k}"
