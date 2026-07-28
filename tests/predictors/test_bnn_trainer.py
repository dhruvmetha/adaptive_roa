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


def _write_separable(path, n, seed):
    """A cleanly separable task: label = (cart position > 0), with a margin.

    The margin band is dropped so the Bayes error is exactly zero -- any
    accuracy below ~0.95 means the arm failed to fit, not that the task is hard.
    The range is +-4 because CartPoleSystem normalizes x by a cart limit of
    ~6.05, so a +-1 box would compress the only informative coordinate into
    +-0.17 and make the test needlessly marginal.
    """
    rng = np.random.default_rng(seed)
    X = rng.uniform(-4.0, 4.0, size=(int(n * 1.4), 4))
    X = X[np.abs(X[:, 0]) > 0.4][:n]
    y = (X[:, 0] > 0).astype(int)
    np.savetxt(path, np.column_stack([X, y]))
    return str(path), X, y


@pytest.mark.parametrize("posterior", ["mfvi", "ensemble", "laplace"])
def test_trained_arm_actually_learns_the_task(posterior, tmp_path):
    """Every other trainer test would pass on a network that never learned.

    Shapes, determinism and file existence are all satisfied by random weights,
    so nothing pinned the one property the arms exist for. Two defects hid in
    that gap: the trainer returned last-epoch weights while the export loaded
    the best checkpoint, and MFVI's checkpoint/early-stopping monitored the
    ELBO -- whose KL term is data-independent and monotone -- rather than the
    fit. Held-out accuracy is the assertion that notices.

    The bar is 0.8 on a task where the plain-MLP baseline reaches ~1.0: loose
    enough not to be flaky across arms and seeds, far enough above the 0.5
    chance level that an untrained or mis-loaded network cannot clear it.
    """
    train_file, _, _ = _write_separable(tmp_path / "train.txt", 2000, 0)
    val_file, _, _ = _write_separable(tmp_path / "val.txt", 600, 1)
    _, X_test, y_test = _write_separable(tmp_path / "test.txt", 600, 2)

    cfg = _cfg(posterior, hidden_dims=[32, 32], lr=1e-2, max_epochs=40,
               patience=40, n_members=2, n_marginal_samples=16)
    cfg.predictor.batch_size = 256

    torch.manual_seed(0)
    handle = BayesianMLPTrainer(cfg, CartPoleSystem(), "cartpole").fit(
        {"train": train_file, "val": val_file}, str(tmp_path / "out")
    )

    logits = handle(torch.as_tensor(X_test, dtype=torch.float32)).view(-1)
    assert torch.isfinite(logits).all()
    accuracy = ((logits > 0).long().numpy() == y_test).mean()
    assert accuracy >= 0.8, f"{posterior} arm did not learn: held-out accuracy {accuracy:.3f}"


def test_mfvi_reports_its_kl_weight(tmp_path):
    """beta is a reported protocol parameter, not a silent default."""
    files = {"train": _write_dataset(tmp_path / "train.txt"),
             "val": _write_dataset(tmp_path / "val.txt", n=128, seed=1)}
    trainer = BayesianMLPTrainer(_cfg("mfvi", kl_weight=1.0), CartPoleSystem(), "cartpole")
    assert trainer.kl_weight == 1.0
