"""End-to-end smoke test: AdaptiveEngine in partx acquisition mode, cartpole.

Mirrors test_e2e_pendulum.py but exercises the 4D CartPole system (x, theta,
x_dot, theta_dot). CartPole's upright attractor is at [0, 0, 0, 0] -- theta=0
is interior to [-pi, pi], so (unlike the design-spec draft's original worry)
there is no circular seam problem for the partition tree here.

Composition mirrors scripts/run_adaptive.py: config_name="default" plus Hydra
group overrides (system=cartpole_pybullet, predictor=gp, acquisition=partx,
eval=partx). There is no configs/adaptive_v2/experiment/ group in this repo.

Note: CartPole is 4D, so PartXEvaluator's region-tree PNG is skipped (it is
guarded by system.state_dim == 2); only pendulum/2D systems render that plot.
"""
import json
import os
from pathlib import Path

import pytest
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf

DATASET_DIR = (
    "/common/users/shared/pracsys/genMoPlan/data_trajectories/deterministic/cartpole_pybullet"
)
pytestmark = pytest.mark.skipif(
    not Path(DATASET_DIR).exists(),
    reason="shared cartpole dataset not available",
)
CONFIG_DIR = str(Path(__file__).resolve().parents[2] / "configs/adaptive_v2")


def _register_resolvers() -> None:
    """Register Hydra/OmegaConf resolvers used by the adaptive_v2 configs."""
    from adaptive_roa.utils.env_config import get_net_id, get_exp_dir, get_shared_data_base

    for name, fn in [
        ("net_id", get_net_id),
        ("exp_dir", get_exp_dir),
        ("shared_data_base", get_shared_data_base),
    ]:
        if not OmegaConf.has_resolver(name):
            OmegaConf.register_new_resolver(name, (lambda f: (lambda d="": f() or d))(fn))

    if not OmegaConf.has_resolver("data_dir"):
        OmegaConf.register_new_resolver(
            "data_dir",
            lambda d="": os.environ.get("DATA_DIR", d) or d,
        )


@pytest.mark.slow
def test_partx_cartpole_end_to_end(tmp_path, monkeypatch):
    """Two-epoch CPU run of AdaptiveEngine with acquisition=partx, predictor=gp, system=cartpole_pybullet."""
    monkeypatch.setenv(
        "DATA_DIR",
        "/common/users/shared/pracsys/genMoPlan/data_trajectories",
    )
    monkeypatch.setenv("MPLBACKEND", "Agg")

    _register_resolvers()

    output_dir = str(tmp_path / "engine_out")
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(
            config_name="default",
            overrides=[
                "system=cartpole_pybullet",
                "predictor=gp",
                "acquisition=partx",
                "eval=partx",
                f"output_dir={output_dir}",
                "n_epochs=2",
                "samples_per_epoch=20",
                "initial_train_size=60",
                "device=cpu",
                "predictor.gp.n_iters=50",
                "acquisition.n_candidates=500",
                "acquisition.bounds.R=20",
                # Note: eval.max_eval_rows only caps the generative (MC) full-ROA
                # path; predictor=gp routes through predictor_type="classifier",
                # whose evaluate_full_roa_classifier loads the full cartpole
                # test_set unconditionally. That still runs quickly on CPU at
                # this scale, so no cap is needed here.
            ],
        )

    from adaptive_roa.adaptive_v2.engine import AdaptiveEngine

    result = AdaptiveEngine(cfg).run()

    assert len(result["epoch_results"]) == 2

    for epoch_idx in range(2):
        art = json.loads(
            (Path(output_dir) / f"epoch_{epoch_idx:03d}" / "artifacts_v2.json").read_text()
        )
        diag = art["acquisition"]["diagnostics"]
        assert "roa_volume" in diag and 0.0 <= diag["roa_volume"] <= 1.0
