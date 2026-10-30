"""The det-paper rerun protocol composes as ruled (papers/det_paper/EXPERIMENT_TRACKER.md).

One acquisition K for every signal (D06), cal scores at the test K (so eval-time
coverage compares like with like), no confidence-pair filter (D16), and timeout
intermediates dropped (C15).
"""
from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "adaptive_v2"
DET_SYSTEMS = ["pendulum", "cartpole_pybullet", "quadrotor2d", "quadrotor3d"]


def _compose(*overrides):
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        return compose(config_name="default", overrides=list(overrides))


@pytest.mark.parametrize("acq", ["direct", "ranked", "entropy"])
def test_every_signal_draws_the_same_acquisition_k(acq):
    cfg = _compose("system=quadrotor2d", f"acquisition={acq}", "+experiment=det_rerun")
    assert cfg.probability.num_mc_samples == 20
    assert cfg.acquisition.num_mc_samples == 20


def test_one_override_moves_every_signal():
    cfg = _compose("system=quadrotor2d", "acquisition=entropy", "+experiment=det_rerun",
                   "acquisition_mc_samples=50")
    assert cfg.probability.num_mc_samples == 50
    assert cfg.acquisition.num_mc_samples == 50


def test_cal_scores_use_the_eval_k():
    cfg = _compose("system=quadrotor2d", "+experiment=det_rerun")
    assert cfg.eval.num_mc_samples_eval == 100
    assert cfg.calibration.num_mc_samples == cfg.eval.num_mc_samples_eval


def test_filter_off_and_timeouts_dropped():
    cfg = _compose("system=quadrotor2d", "+experiment=det_rerun")
    assert cfg.adaptive_v2.filter_confident_pairs is False
    assert cfg.data_source.timeout_intermediates == "drop"


def test_acquisition_runs_over_suffixes():
    cfg = _compose("system=quadrotor2d", "+experiment=det_rerun")
    assert cfg.candidate_mode == "suffix"
    assert cfg.test_ratio == 0


def test_default_d2_and_uniform_override():
    cfg = _compose("system=quadrotor2d", "acquisition=ranked", "+experiment=det_rerun")
    assert cfg.acquisition.d2_ratio == 0.5
    cfg = _compose("system=quadrotor2d", "acquisition=ranked", "+experiment=det_rerun",
                   "acquisition.d2_ratio=0")
    assert cfg.acquisition.d2_ratio == 0


def test_pending_rulings_default_to_the_runs_as_launched():
    # D25 and D27 stay off until ruled, so a launch from this branch matches the running cells.
    cfg = _compose("system=quadrotor2d", "acquisition=direct", "+experiment=det_rerun")
    assert cfg.acquisition.fill_shortfall is False
    assert cfg.threshold_fit_set == "pairs" and cfg.threshold.num_mc_samples == 10
    cfg = _compose("system=quadrotor2d", "acquisition=direct", "+experiment=det_rerun",
                   "threshold_fit_set=val_split", "threshold.num_mc_samples=${eval.num_mc_samples_eval}")
    assert cfg.threshold_fit_set == "val_split" and cfg.threshold.num_mc_samples == 100


@pytest.mark.parametrize("system", DET_SYSTEMS)
def test_every_det_system_names_its_horizon(system):
    cfg = _compose(f"system={system}", "+experiment=det_rerun")
    assert int(cfg.data_source.horizon_steps) > 0


def test_without_the_experiment_nothing_moves():
    # GRAPE and older runs never load det_rerun, so their K stays where it was.
    cfg = _compose("system=quadrotor2d", "acquisition=direct")
    assert cfg.probability.num_mc_samples == 10
    assert cfg.calibration.num_mc_samples == 10
    assert cfg.adaptive_v2.filter_confident_pairs is True


SCHEDULES = {"pendulum": (100, 50, 600), "cartpole_pybullet": (100, 100, 1100),
             "quadrotor2d": (2000, 1000, 12000), "quadrotor3d": (10000, 2000, 30000)}


@pytest.mark.parametrize("system", DET_SYSTEMS)
def test_schedules_are_the_ruled_ten_iterations(system):
    cfg = _compose(f"system={system}", "+experiment=det_rerun")
    init, step, final = SCHEDULES[system]
    assert (cfg.initial_train_size, cfg.samples_per_epoch, cfg.n_epochs) == (init, step, 11)
    # The epoch-10 model trains on initial + 10 * step trajectories.
    assert cfg.initial_train_size + (cfg.n_epochs - 1) * cfg.samples_per_epoch == final


def test_output_dir_names_every_knob_a_cell_varies():
    from omegaconf import OmegaConf
    cfg = _compose("system=quadrotor2d", "acquisition=entropy", "+experiment=det_rerun")
    raw = OmegaConf.to_container(cfg, resolve=False)["output_dir"]
    for knob in ("${predictor.name}", "${sampling_mode}", "${acquisition.d2_ratio}",
                 "${acquisition_mc_samples}", "${threshold.w}", "${seed}", "${samples_per_epoch}"):
        assert knob in raw


def test_fm_det_is_generative_with_grapes_schedule():
    a = _compose("system=quadrotor3d", "predictor=fm_det", "+experiment=det_rerun")
    b = _compose("system=quadrotor3d", "predictor=generative", "+experiment=det_rerun")
    assert a.predictor.lightning_trainer.max_epochs == 200
    es = a.predictor.lightning_trainer.callbacks[1]
    assert (es.patience, es.min_delta) == (30, 0.0001)
    assert a.predictor.lightning_trainer.callbacks[0].save_last is False
    assert a.predictor.name == b.predictor.name == "fm"
    assert a.predictor.flow_matching == b.predictor.flow_matching
    assert a.probability == b.probability
    assert "unet" in str(a.model._target_).lower()
