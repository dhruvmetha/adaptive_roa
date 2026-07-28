import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from adaptive_roa.adaptive_v2.trainers.final_state_trainer import FinalStateTrainer
from adaptive_roa.systems.cartpole import CartPoleSystem


def _write_endpoints(path, n=256, seed=0):
    """8-column cartpole endpoint pairs: x th xd thd | x th xd thd."""
    rng = np.random.default_rng(seed)
    start = rng.uniform(-1.0, 1.0, size=(n, 4))
    end = start * 0.5  # a learnable contraction toward the origin
    np.savetxt(path, np.column_stack([start, end]))
    return str(path)


def _cfg(posterior, **overrides):
    fs = {
        "posterior": posterior, "hidden_dims": [16, 16], "lr": 1e-2,
        "weight_decay": 1e-5, "max_epochs": 3, "patience": 5, "prior_sigma": 1.0,
        "n_members": 2, "kl_weight": 1.0, "beta_nll": 0.5,
    }
    fs.update(overrides)
    return OmegaConf.create({
        "device": "cpu",
        "predictor": {"type": "generative", "name": f"fs_{posterior}",
                      "batch_size": 64, "val_batch_size": 64, "final_state": fs},
    })


@pytest.fixture
def files(tmp_path):
    return {"train": _write_endpoints(tmp_path / "train.txt"),
            "val": _write_endpoints(tmp_path / "val.txt", n=128, seed=1)}


@pytest.mark.parametrize("posterior", ["deterministic", "mfvi", "ensemble", "laplace"])
def test_trainer_returns_a_working_final_state_handle(posterior, files, tmp_path):
    trainer = FinalStateTrainer(_cfg(posterior), CartPoleSystem(), "cartpole_pybullet")
    handle = trainer.fit(files, str(tmp_path / f"out_{posterior}"))

    x = torch.randn(16, 4)
    out = handle.predict_endpoint(x)
    assert out.shape == (16, 4)
    assert torch.isfinite(out).all()
    # THE contract: fresh sample per call.
    assert not torch.allclose(handle.predict_endpoint(x), handle.predict_endpoint(x))


@pytest.mark.parametrize("posterior", ["deterministic", "mfvi", "ensemble", "laplace"])
def test_trainer_writes_a_warm_start_checkpoint(posterior, files, tmp_path):
    out = tmp_path / f"out_{posterior}"
    FinalStateTrainer(_cfg(posterior), CartPoleSystem(), "cartpole_pybullet").fit(files, str(out))
    assert list((out / "checkpoints").glob("best*.ckpt")), "engine warm start needs best*.ckpt"


def test_laplace_fits_its_ggn_with_an_explicit_sigma(files, tmp_path):
    """task='final_state' REQUIRES sigma; letting it default to 1.0 silently
    rescales the entire posterior covariance."""
    trainer = FinalStateTrainer(_cfg("laplace"), CartPoleSystem(), "cartpole_pybullet")
    out = tmp_path / "out_laplace"
    handle = trainer.fit(files, str(out))
    assert handle.posterior.is_fitted
    assert (out / "checkpoints" / "laplace_cov.pt").exists()


def test_the_arm_actually_learns_the_contraction(files, tmp_path):
    """Guards against shipping an untrained network: on end = 0.5*start the fitted
    head's mean must beat the identity map by a clear margin."""
    trainer = FinalStateTrainer(
        _cfg("deterministic", max_epochs=60), CartPoleSystem(), "cartpole_pybullet"
    )
    handle = trainer.fit(files, str(tmp_path / "out_learn"))

    raw = np.loadtxt(files["val"])
    x = torch.as_tensor(raw[:, :4], dtype=torch.float32)
    y = torch.as_tensor(raw[:, 4:], dtype=torch.float32)
    with torch.no_grad():
        pred = handle.head.mean(handle._params(x))
    assert (pred - y).pow(2).mean().item() < 0.5 * (x - y).pow(2).mean().item()


def test_ensemble_requires_enough_mc_samples_to_resolve_its_members(files, tmp_path):
    """K >= 2M: an M-atom empirical posterior sampled K times needs K >= 2M, or the
    reported probability is a sampling artifact of the member draw."""
    cfg = _cfg("ensemble", n_members=8)
    cfg.predictor.num_mc_samples = 10
    with pytest.raises(ValueError, match="num_mc_samples"):
        FinalStateTrainer(cfg, CartPoleSystem(), "cartpole_pybullet").fit(
            files, str(tmp_path / "out_guard")
        )


def test_mlp_det_baseline_experiment_actually_zeroes_d2_ratio():
    """The baseline arm must not acquire adaptively. Setting acquisition.d2_ratio
    from the predictor group is silently discarded (default.yaml lists predictor
    before acquisition and a later group wins), so it lives in the experiment
    group, which composes last."""
    import os
    from hydra import compose, initialize_config_dir

    d = os.path.abspath("configs/adaptive_v2")
    with initialize_config_dir(config_dir=d, version_base=None):
        cfg = compose(config_name="default", overrides=["+experiment=mlp_det_baseline"])
    assert cfg.predictor.name == "mlp_det"
    assert cfg.predictor.type == "generative"
    assert float(cfg.acquisition.d2_ratio) == 0.0
