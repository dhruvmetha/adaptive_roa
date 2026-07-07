"""Smoke test: the deterministic training pipeline runs end-to-end and reports metric #1."""
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from adaptive_roa.partial_trajs.train import run

PENDULUM_T25 = Path(
    "/common/users/shared/pracsys/genMoPlan/data_trajectories/partial_deterministic"
    "/pendulum_lqr/pendulum_lqr_50k_T25"
)

pytestmark = pytest.mark.skipif(
    not PENDULUM_T25.exists(), reason="shared partial-trajectory dataset not available"
)


def test_train_smoke_runs_and_reports_metric1():
    cfg = OmegaConf.create(
        {
            "system": "pendulum",
            "dataset_dir": str(PENDULUM_T25),
            "hidden_dims": [16, 16],
            "batch_size": 64,
            "val_fraction": 0.2,
            "seed": 0,
            "lr": 1.0e-3,
            "dropout": 0.0,
            "max_epochs": 1,
            "accelerator": "cpu",
            "devices": 1,
            "enable_checkpointing": False,
            "enable_progress_bar": False,
            "limit_train_batches": 2,
            "limit_val_batches": 2,
            "eval_max_batches": 2,
        }
    )
    model, report = run(cfg)
    assert report["n"] > 0
    assert report["mean"] >= 0.0
    # metric #1 is stratified
    assert "mean_freeze" in report or "mean_nonfreeze" in report
