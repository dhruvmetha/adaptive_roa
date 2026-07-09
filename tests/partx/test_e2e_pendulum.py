# tests/partx/test_e2e_pendulum.py
"""End-to-end smoke test: AdaptiveEngine in partx acquisition mode, pendulum.

Exercises the full partx stack (GP predictor, partition tree, straddle
acquisition, Bayesian RoA-volume bound, q_hat calibration, PartXEvaluator)
against the real pendulum trajectory pool on CPU with a tiny budget.

Composition mirrors scripts/run_adaptive.py: config_name="default" plus Hydra
group overrides (system=pendulum, predictor=gp, acquisition=partx, eval=partx).
There is no configs/adaptive_v2/experiment/ group in this repo (see Task 13).
"""
import json
import os
from pathlib import Path

import pytest
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf

DATASET_DIR = (
    "/common/users/shared/pracsys/genMoPlan/data_trajectories/deterministic/pendulum_lqr_50k"
)
pytestmark = pytest.mark.skipif(
    not Path(DATASET_DIR).exists(),
    reason="shared pendulum dataset not available",
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
def test_partx_pendulum_end_to_end(tmp_path, monkeypatch):
    """Two-epoch CPU run of AdaptiveEngine with acquisition=partx, predictor=gp."""
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
                "system=pendulum",
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
                # whose evaluate_full_roa_classifier loads the full ~48.7k-row
                # pendulum test_set unconditionally. That still runs in a few
                # seconds on CPU, so no cap is needed here.
            ],
        )

    from adaptive_roa.adaptive_v2.engine import AdaptiveEngine

    result = AdaptiveEngine(cfg).run()

    assert len(result["epoch_results"]) == 2

    art = json.loads((Path(output_dir) / "epoch_000" / "artifacts_v2.json").read_text())
    diag = art["acquisition"]["diagnostics"]
    assert "roa_volume" in diag and 0.0 <= diag["roa_volume"] <= 1.0
