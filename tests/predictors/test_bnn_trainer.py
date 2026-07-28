import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from adaptive_roa.adaptive_v2.trainers.bayesian_mlp_trainer import BayesianMLPTrainer
from adaptive_roa.systems.cartpole import CartPoleSystem


def _write_dataset(path, n=256, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, size=(n, 4))
    y = (np.abs(X[:, 1]) < 0.5).astype(int)
    np.savetxt(path, np.column_stack([X, y]))
    return str(path)


def _cfg(posterior, **overrides):
    bnn = {
        "hidden_dims": [16, 16], "lr": 1e-2, "weight_decay": 1e-5,
        "max_epochs": 3, "patience": 5, "prior_sigma": 1.0,
        "n_members": 2, "kl_weight": 1.0, "n_marginal_samples": 8,
        "posterior": posterior,
    }
    bnn.update(overrides)
    return OmegaConf.create({
        "device": "cpu",
        "predictor": {"type": "classifier", "name": f"bnn_{posterior}",
                      "batch_size": 64, "bnn": bnn},
    })


@pytest.mark.parametrize("posterior", ["mfvi", "ensemble", "laplace"])
def test_trainer_returns_a_working_outcome_handle(posterior, tmp_path):
    files = {"train": _write_dataset(tmp_path / "train.txt"),
             "val": _write_dataset(tmp_path / "val.txt", n=128, seed=1)}
    trainer = BayesianMLPTrainer(_cfg(posterior), CartPoleSystem(), "cartpole")
    handle = trainer.fit(files, str(tmp_path / "out"))

    x = torch.randn(16, 4)
    logits = handle(x)
    assert logits.shape in {(16,), (16, 1)}
    assert torch.isfinite(logits).all()
    # Contract: repeated calls must agree.
    assert torch.allclose(handle(x), handle(x))


@pytest.mark.parametrize("posterior", ["mfvi", "ensemble", "laplace"])
def test_trainer_writes_a_checkpoint_for_warm_start(posterior, tmp_path):
    files = {"train": _write_dataset(tmp_path / "train.txt"),
             "val": _write_dataset(tmp_path / "val.txt", n=128, seed=1)}
    out = tmp_path / "out"
    BayesianMLPTrainer(_cfg(posterior), CartPoleSystem(), "cartpole").fit(files, str(out))
    assert list((out / "checkpoints").glob("best*.ckpt")), "engine warm start needs best*.ckpt"


def test_laplace_arm_fits_its_ggn_during_training(tmp_path):
    files = {"train": _write_dataset(tmp_path / "train.txt"),
             "val": _write_dataset(tmp_path / "val.txt", n=128, seed=1)}
    trainer = BayesianMLPTrainer(_cfg("laplace"), CartPoleSystem(), "cartpole")
    handle = trainer.fit(files, str(tmp_path / "out"))
    assert handle.posterior.is_fitted, "Laplace posterior left at the MAP point estimate"


def test_mfvi_reports_its_kl_weight(tmp_path):
    """beta is a reported protocol parameter, not a silent default."""
    files = {"train": _write_dataset(tmp_path / "train.txt"),
             "val": _write_dataset(tmp_path / "val.txt", n=128, seed=1)}
    trainer = BayesianMLPTrainer(_cfg("mfvi", kl_weight=1.0), CartPoleSystem(), "cartpole")
    assert trainer.kl_weight == 1.0
