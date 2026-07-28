import glob
import re
from pathlib import Path

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


# --- best-vs-last-epoch weights -------------------------------------------
#
# Both tests below need a run whose kept checkpoint is NOT the last epoch,
# otherwise "best weights" and "last weights" are the same tensors and the
# assertions are vacuous. Fitting the GGN before the best-checkpoint reload
# (or skipping the reload) changes no shape, no value range, and no other
# test's status -- the symptom is a Gaussian centred at one parameter point
# with curvature computed at a different one -- so this has to compare
# tensors directly against the actual kept checkpoint file.

_MAX_EPOCHS = 40
_EPOCH_IN_FILENAME = re.compile(r"epoch=(\d+)")


def _write_noisy(path, n, seed):
    """end = 0.5*start plus noise big enough that the net overfits it, so
    val_nll turns up early instead of monotonically improving to the last
    epoch."""
    rng = np.random.default_rng(seed)
    start = rng.uniform(-1.0, 1.0, size=(n, 4))
    end = 0.5 * start + 0.8 * rng.normal(size=(n, 4))
    np.savetxt(path, np.column_stack([start, end]))
    return str(path)


def _fit_noisy(posterior, tmp_path):
    cfg = _cfg(posterior, hidden_dims=[64, 64], lr=1e-2, max_epochs=_MAX_EPOCHS,
               # No early stopping: it would end the run AT the best epoch and
               # make "kept < last" true for the wrong reason.
               patience=_MAX_EPOCHS + 1, n_members=2)
    cfg.predictor.batch_size = 128
    files = {"train": _write_noisy(tmp_path / "train.txt", 400, 0),
             "val": _write_noisy(tmp_path / "val.txt", 300, 1)}
    torch.manual_seed(0)
    out = tmp_path / "out"
    handle = FinalStateTrainer(cfg, CartPoleSystem(), "cartpole_pybullet").fit(files, str(out))
    return handle, out / "checkpoints"


def _assert_best_is_not_last(ckpt_path):
    """Guard: the run must have kept an epoch other than its last one."""
    match = _EPOCH_IN_FILENAME.search(Path(ckpt_path).name)
    assert match, f"cannot read the kept epoch out of {ckpt_path}"
    kept = int(match.group(1))
    assert kept < _MAX_EPOCHS - 1, (
        f"degenerate fixture: kept epoch {kept} of max_epochs={_MAX_EPOCHS}, so the "
        f"best and last weights coincide and this test cannot tell them apart. "
        f"Make the targets noisier or train longer."
    )


def _posterior_state(ckpt_path):
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
    return {k[len("posterior."):]: v for k, v in sd.items() if k.startswith("posterior.")}


@pytest.mark.parametrize("posterior", ["deterministic", "mfvi", "ensemble", "laplace"])
def test_fit_returns_the_best_checkpoint_weights_not_the_last_epochs(posterior, tmp_path):
    """The returned handle must carry the weights the export will reload.

    If ``fit`` handed back last-epoch weights instead of the best-``val_nll``
    checkpoint's, the pipeline's own numbers and the exported ones would come
    from different networks. Nothing about shapes, determinism, or file
    existence notices that -- this compares the tensors directly.
    """
    handle, ckpt_dir = _fit_noisy(posterior, tmp_path)

    if posterior == "ensemble":
        # Each member has its own checkpoint; the top-level best-ensemble.ckpt
        # is written FROM the assembled posterior, so comparing against it
        # would be vacuous. Compare member-wise instead.
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
    checkpoint is reloaded centres the Gaussian at the best-epoch weights
    while taking its curvature from the last epoch -- a silent math error
    that changes no shape, no value range, and no other test's status.
    Recompute the GGN here at the RETURNED handle's weights (same features,
    same derived sigma) and require the shipped laplace_cov.pt to match it.
    """
    handle, ckpt_dir = _fit_noisy("laplace", tmp_path)
    _assert_best_is_not_last(sorted(glob.glob(str(ckpt_dir / "best*.ckpt")))[0])

    shipped = torch.load(ckpt_dir / "laplace_cov.pt", map_location="cpu", weights_only=True)

    system = CartPoleSystem()
    posterior = handle.posterior
    head = handle.head
    raw = np.loadtxt(tmp_path / "train.txt")
    starts = torch.as_tensor(raw[:, :4], dtype=torch.float32)
    ends = torch.as_tensor(raw[:, 4:], dtype=torch.float32)

    posterior.eval()
    with torch.no_grad():
        embedded = system.embed_state_for_model(system.normalize_state(starts))
        features = posterior.body(embedded)
        # The MAP prediction, NOT posterior.forward_sample: the posterior is
        # already fitted at this point (from the trainer's own run), so
        # forward_sample would draw a random posterior sample instead of
        # reproducing the deterministic prediction the trainer derived sigma
        # from (forward_sample only falls back to the MAP head_layer output
        # before the first fit -- see LastLayerLaplacePosterior.forward_sample).
        params = posterior.head_layer(features)
        resid = head.distance_per_component(head.mean(params), ends)
        sigma = float(resid.pow(2).mean().sqrt().clamp_min(1e-3))
        posterior.fit(features, ends, task="final_state", sigma=sigma)

    torch.testing.assert_close(shipped, posterior.posterior_covariance,
                               rtol=1e-6, atol=1e-8)
