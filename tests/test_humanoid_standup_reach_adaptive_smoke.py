# tests/test_humanoid_standup_reach_adaptive_smoke.py
"""End-to-end smoke test for AdaptiveEngine with humanoid_standup_reach.

Exercises the full pipeline for one tiny epoch on CPU:
  intermediate-candidate acquisition → tail-pair training files →
  FM train (val_loss non-empty) → threshold optimization on intermediate val →
  q_hat calibration on tiny cal file → ROA eval on tiny test file.

The real cal/test FPS files are 320 MB / 1.3 GB; we slice the first 60 lines
into tmp_path to keep the test fast.
"""
from pathlib import Path
import pytest
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf

DATASET_DIR = "/common/users/shared/pracsys/genMoPlan/data_trajectories/humanoid_get_up_medium"
pytestmark = pytest.mark.skipif(
    not Path(DATASET_DIR).exists(),
    reason="shared humanoid dataset not available",
)
CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "configs/adaptive_v2")

# Real FPS files to slice
_CAL_SRC = str(Path(DATASET_DIR) / "train_test_splits/cal_set_fps.txt")
_TEST_SRC = str(Path(DATASET_DIR) / "train_test_splits/test_set_fps.txt")


def _head_n_lines(src: str, n: int, dst: Path) -> None:
    """Write the first n non-empty lines of src to dst."""
    with open(src) as f_in, open(dst, "w") as f_out:
        count = 0
        for line in f_in:
            if line.strip():
                f_out.write(line)
                count += 1
                if count >= n:
                    break


def _register_resolvers() -> None:
    """Register Hydra/OmegaConf resolvers used by the adaptive_v2 configs."""
    import os
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


def test_adaptive_engine_one_epoch_cpu(tmp_path, monkeypatch):
    """One-epoch CPU run of AdaptiveEngine with candidate_mode=intermediate + FPS cal/test."""
    # Point DATA_DIR so data_dir resolver resolves correctly
    monkeypatch.setenv(
        "DATA_DIR",
        "/common/users/shared/pracsys/genMoPlan/data_trajectories",
    )
    # Headless matplotlib (engine writes plots)
    monkeypatch.setenv("MPLBACKEND", "Agg")

    _register_resolvers()

    # ── Build tiny cal / test slices ────────────────────────────────────────
    cal_small = tmp_path / "cal_small.txt"
    test_small = tmp_path / "test_small.txt"
    _head_n_lines(_CAL_SRC, 60, cal_small)
    _head_n_lines(_TEST_SRC, 60, test_small)

    # ── Build tiny pool index / label files ─────────────────────────────────
    # The real shuffled_indices/labels files have 800k rows; np.loadtxt on them
    # takes many minutes even for a smoke test.  We slice the first 50 entries
    # (which contain both label classes) into tmp_path so TrajectoryDataSource
    # initialises quickly.
    pool_dir = Path(DATASET_DIR)
    idx_src = pool_dir / "train_test_splits/all_shuffled_indices.txt"
    lab_src = pool_dir / "train_test_splits/all_shuffled_labels.txt"
    idx_small = tmp_path / "idx_small.txt"
    lab_small = tmp_path / "lab_small.txt"
    _head_n_lines(str(idx_src), 50, idx_small)
    _head_n_lines(str(lab_src), 50, lab_small)

    # ── Compose config ───────────────────────────────────────────────────────
    output_dir = str(tmp_path / "engine_out")
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(
            config_name="default",
            overrides=[
                "system=humanoid_standup_reach",
                # Computation device
                "device=cpu",
                "predictor.lightning_trainer.accelerator=cpu",
                "predictor.lightning_trainer.devices=1",
                "predictor.lightning_trainer.max_epochs=1",
                # Tiny training sizes
                "initial_train_size=8",
                "samples_per_epoch=4",
                "predictor.batch_size=4",
                "predictor.val_batch_size=8",
                "+predictor.num_workers=0",
                "n_epochs=1",
                # Evaluate every epoch (required to exercise cal/test eval paths)
                "eval_every=1",
                # Tiny MC samples
                "probability.num_mc_samples=2",
                "eval.num_mc_samples_eval=2",
                # Override cal / test to tiny slices
                f"data_source.cal_set_file={cal_small}",
                f"data_source.test_set_file={test_small}",
                # Override pool index/label to tiny slices (avoid loading 800k rows)
                f"data_source.shuffled_indices_file={idx_small}",
                f"data_source.shuffled_labels_file={lab_small}",
                # Point output to tmp_path (avoids complex ${now:...} resolver)
                f"output_dir={output_dir}",
            ],
        )

    from adaptive_roa.adaptive_v2.engine import AdaptiveEngine

    engine = AdaptiveEngine(cfg)
    result = engine.run()

    # ── Verify the integration exercised what the task requires ─────────────
    assert len(result["epoch_results"]) == 1, "Expected exactly one epoch result"

    epoch0 = result["epoch_results"][0]
    full_roa = epoch0["full_roa"]

    # ROA eval on tiny test file must have run (not been skipped)
    assert "skipped" not in full_roa, (
        f"ROA eval was skipped — eval_every likely 0 or run_eval==False. "
        f"full_roa keys: {list(full_roa.keys())}"
    )
    assert "n_total" in full_roa, (
        f"ROA eval result missing 'n_total' key. Keys: {list(full_roa.keys())}"
    )

    # q_hat calibration on tiny cal file must have run
    assert epoch0["q_hat_eval"] is not None, (
        "q_hat_eval is None — cal-set calibration path was not exercised. "
        "Check that cal_set_file override took effect and run_eval was True."
    )
