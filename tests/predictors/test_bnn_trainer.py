import glob
import re
from pathlib import Path

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


# --- best-vs-last-epoch weights -------------------------------------------
#
# Both tests below need a run whose kept checkpoint is NOT the last epoch,
# otherwise "best weights" and "last weights" are the same tensors and the
# assertions are vacuous. That rules out T9's separable task, on which every arm
# reaches 1.000 and validation loss never turns up. Noisy labels near the
# decision boundary make the net overfit early instead, and _assert_best_is_not_last
# fails loudly if that ever stops happening.

# 40 rather than the ~10 the arms need: MFVI is the slowest to overfit (its KL
# term regularizes) and settles on epoch 21, so a shorter budget would leave the
# best-vs-last margin uncomfortably thin. Measured kept epochs: mfvi 21,
# ensemble 7-8, laplace 8.
_MAX_EPOCHS = 40
_EPOCH_IN_FILENAME = re.compile(r"epoch=(\d+)")


def _write_noisy(path, n, seed):
    """Labels flipped near the boundary, so validation loss turns up early."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-4.0, 4.0, size=(n, 4))
    y = (X[:, 0] + 3.0 * rng.normal(size=n) > 0).astype(int)
    np.savetxt(path, np.column_stack([X, y]))
    return str(path)


def _fit_noisy(posterior, tmp_path):
    cfg = _cfg(posterior, hidden_dims=[64, 64], lr=1e-2, max_epochs=_MAX_EPOCHS,
               # No early stopping: it would end the run AT the best epoch and
               # make "kept < last" true for the wrong reason.
               patience=_MAX_EPOCHS + 1, n_members=2, n_marginal_samples=8)
    cfg.predictor.batch_size = 128
    files = {"train": _write_noisy(tmp_path / "train.txt", 400, 0),
             "val": _write_noisy(tmp_path / "val.txt", 300, 1)}
    torch.manual_seed(0)
    out = tmp_path / "out"
    handle = BayesianMLPTrainer(cfg, CartPoleSystem(), "cartpole").fit(files, str(out))
    return handle, out / "checkpoints"


def _assert_best_is_not_last(ckpt_path):
    """Guard: the run must have kept an epoch other than its last one."""
    match = _EPOCH_IN_FILENAME.search(Path(ckpt_path).name)
    assert match, f"cannot read the kept epoch out of {ckpt_path}"
    kept = int(match.group(1))
    assert kept < _MAX_EPOCHS - 1, (
        f"degenerate fixture: kept epoch {kept} of max_epochs={_MAX_EPOCHS}, so the "
        f"best and last weights coincide and this test cannot tell them apart. "
        f"Make the labels noisier or train longer."
    )


def _posterior_state(ckpt_path):
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
    return {k[len("posterior."):]: v for k, v in sd.items() if k.startswith("posterior.")}


@pytest.mark.parametrize("posterior", ["mfvi", "ensemble", "laplace"])
def test_fit_returns_the_best_checkpoint_weights_not_the_last_epochs(posterior, tmp_path):
    """The returned handle must carry the weights the export will reload.

    ``load_from_run`` reads ``best*.ckpt``. If ``fit`` hands back last-epoch
    weights instead, the pipeline's own numbers and the exported ones come from
    different networks, and the BNN arms are not best-val-selected while the MLP
    baseline is. Nothing about shapes, determinism or file existence notices, so
    this compares the tensors directly.
    """
    handle, ckpt_dir = _fit_noisy(posterior, tmp_path)

    if posterior == "ensemble":
        # Each member has its own checkpoint; the top-level best-ensemble.ckpt is
        # written FROM the assembled posterior, so comparing against it would be
        # vacuous. Compare member-wise instead.
        pairs = []
        for m, member in enumerate(handle.posterior.members):
            found = sorted(glob.glob(str(ckpt_dir / f"member_{m}" / "best_member*.ckpt")))
            assert len(found) == 1, found
            _assert_best_is_not_last(found[0])
            pairs.append((member.state_dict(), _posterior_state(found[0])))
    else:
        found = sorted(glob.glob(str(ckpt_dir / "best*.ckpt")))
        assert len(found) == 1, found
        _assert_best_is_not_last(found[0])
        pairs = [(handle.posterior.state_dict(), _posterior_state(found[0]))]

    for live, saved in pairs:
        assert set(live) == set(saved), (set(live) ^ set(saved))
        for key in live:
            torch.testing.assert_close(live[key], saved[key], rtol=0, atol=0,
                                       msg=f"{key} differs from the kept checkpoint")


def test_laplace_covariance_is_the_ggn_at_the_exported_weights(tmp_path):
    """H^-1 must be evaluated at the parameters that ship, not some other ones.

    The covariance is curvature AT a point. Fitting the GGN before the best
    checkpoint is reloaded centres the Gaussian at the best-epoch weights while
    taking its curvature from the last epoch -- a silent math error that changes
    no shape, no value range, and no other test's status. Recompute the GGN here
    at the RETURNED handle's weights and require the shipped laplace_cov.pt to
    match it.
    """
    handle, ckpt_dir = _fit_noisy("laplace", tmp_path)
    _assert_best_is_not_last(sorted(glob.glob(str(ckpt_dir / "best*.ckpt")))[0])

    shipped = torch.load(ckpt_dir / "laplace_cov.pt", map_location="cpu", weights_only=True)

    system = CartPoleSystem()
    rows = np.loadtxt(tmp_path / "train.txt")
    X = torch.as_tensor(rows[:, :4], dtype=torch.float32)
    y = torch.as_tensor(rows[:, 4], dtype=torch.float32)
    posterior = handle.posterior
    posterior.eval()
    with torch.no_grad():
        embedded = system.embed_state_for_model(system.normalize_state(X))
        posterior.fit(posterior.body(embedded), y, task="outcome")

    torch.testing.assert_close(shipped, posterior.posterior_covariance,
                               rtol=1e-6, atol=1e-8)


def test_mfvi_reports_its_kl_weight(tmp_path):
    """beta is a reported protocol parameter, not a silent default."""
    files = {"train": _write_dataset(tmp_path / "train.txt"),
             "val": _write_dataset(tmp_path / "val.txt", n=128, seed=1)}
    trainer = BayesianMLPTrainer(_cfg("mfvi", kl_weight=1.0), CartPoleSystem(), "cartpole")
    assert trainer.kl_weight == 1.0
