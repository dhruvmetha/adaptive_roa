import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from adaptive_roa.adaptive_v2.trainers.gp_regressor_trainer import GPRegressorTrainer
from adaptive_roa.systems.cartpole import CartPoleSystem


def _write_endpoints(path, n=200, seed=0):
    rng = np.random.default_rng(seed)
    start = rng.uniform(-0.5, 0.5, size=(n, 4))
    np.savetxt(path, np.column_stack([start, start * 0.5]))
    return str(path)


def _cfg(**overrides):
    gp = {"n_inducing": 16, "kernel": "matern52", "n_iters": 30, "lr": 0.05, "batch_size": 128}
    gp.update(overrides)
    return OmegaConf.create({
        "device": "cpu",
        "predictor": {"type": "generative", "name": "gp_reg",
                      "batch_size": 64, "val_batch_size": 64, "gp": gp},
    })


@pytest.fixture
def files(tmp_path):
    return {"train": _write_endpoints(tmp_path / "train.txt"),
            "val": _write_endpoints(tmp_path / "val.txt", n=100, seed=1)}


def test_trainer_returns_a_working_handle(files, tmp_path):
    handle = GPRegressorTrainer(_cfg(), CartPoleSystem(), "cartpole_pybullet").fit(
        files, str(tmp_path / "out")
    )
    x = torch.randn(16, 4) * 0.3
    out = handle.predict_endpoint(x)
    assert out.shape == (16, 4)
    assert torch.isfinite(out).all()
    assert not torch.allclose(handle.predict_endpoint(x), handle.predict_endpoint(x))


def test_trainer_writes_a_warm_start_checkpoint(files, tmp_path):
    """The engine globs checkpoints/best*.ckpt (engine.py:140). partx/trainer.py
    writes gp.pt instead, which is why the GP arm has never warm-started."""
    out = tmp_path / "out"
    GPRegressorTrainer(_cfg(), CartPoleSystem(), "cartpole_pybullet").fit(files, str(out))
    assert list((out / "checkpoints").glob("best*.ckpt"))


def test_warm_start_reloads_a_previous_checkpoint(files, tmp_path):
    trainer = GPRegressorTrainer(_cfg(), CartPoleSystem(), "cartpole_pybullet")
    first_out = tmp_path / "e0"
    trainer.fit(files, str(first_out))
    ckpt = list((first_out / "checkpoints").glob("best*.ckpt"))[0]

    resumed = GPRegressorTrainer(_cfg(n_iters=1), CartPoleSystem(), "cartpole_pybullet").fit(
        files, str(tmp_path / "e1"), resume_checkpoint=str(ckpt)
    )
    x = torch.randn(8, 4) * 0.3
    assert torch.isfinite(resumed.predict_endpoint(x)).all()


def test_the_arm_actually_learns_the_contraction(files, tmp_path):
    """Guards against shipping an untrained GP."""
    handle = GPRegressorTrainer(_cfg(n_iters=300), CartPoleSystem(), "cartpole_pybullet").fit(
        files, str(tmp_path / "out")
    )
    raw = np.loadtxt(files["val"])
    x = torch.as_tensor(raw[:, :4], dtype=torch.float32)
    y = torch.as_tensor(raw[:, 4:], dtype=torch.float32)
    feats = handle.system.embed_state_for_model(handle.system.normalize_state(x))
    pred = handle.decoder.decode(handle.gp.mean(feats))
    assert (pred - y).pow(2).mean().item() < 0.5 * (x - y).pow(2).mean().item()


def test_config_composes_with_the_endpoint_mc_backend():
    import os
    from hydra import compose, initialize_config_dir

    d = os.path.abspath("configs/adaptive_v2")
    with initialize_config_dir(config_dir=d, version_base=None):
        cfg = compose(config_name="default", overrides=["predictor=gp_reg"])
    assert cfg.predictor.name == "gp_reg"
    assert cfg.predictor.type == "generative"
    assert "EndpointMCProbabilityBackend" in cfg.probability._target_
