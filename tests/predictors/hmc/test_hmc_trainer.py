import json

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from adaptive_roa.adaptive_v2.trainers.bayesian_mlp_trainer import BayesianMLPTrainer
from adaptive_roa.adaptive_v2.trainers.hmc_trainer import HMCTrainer
from adaptive_roa.systems.cartpole import CartPoleSystem


def _write_classification(path, n=160, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, size=(n, 4))
    np.savetxt(path, np.column_stack([X, (np.abs(X[:, 1]) < 0.5).astype(int)]))
    return str(path)


def _write_endpoints(path, n=160, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-0.5, 0.5, size=(n, 4))
    np.savetxt(path, np.column_stack([X, X * 0.5]))
    return str(path)


def _cfg(head, **overrides):
    hmc = {"hidden_dims": [8, 8], "activation": "tanh", "prior_sigma": 1.0,
           "n_chains": 2, "n_samples": 20, "n_warmup": 20, "n_leapfrog": 8, "seed": 0}
    hmc.update(overrides)
    return OmegaConf.create({
        "device": "cpu",
        "predictor": {"type": "classifier" if head == "outcome" else "generative",
                      "name": "hmc" if head == "outcome" else "hmc_reg",
                      "head": head, "batch_size": 64, "hmc": hmc},
    })


@pytest.fixture
def outcome_files(tmp_path):
    return {"train": _write_classification(tmp_path / "tr.txt"),
            "val": _write_classification(tmp_path / "va.txt", n=80, seed=1)}


@pytest.fixture
def endpoint_files(tmp_path):
    return {"train": _write_endpoints(tmp_path / "tr.txt"),
            "val": _write_endpoints(tmp_path / "va.txt", n=80, seed=1)}


def test_outcome_arm_returns_a_working_handle(outcome_files, tmp_path):
    handle = HMCTrainer(_cfg("outcome"), CartPoleSystem(), "cartpole_pybullet").fit(
        outcome_files, str(tmp_path / "out")
    )
    logits = handle(torch.randn(12, 4))
    assert logits.shape in {(12,), (12, 1)}
    assert torch.isfinite(logits).all()
    # The outcome handle marginalizes internally and must be deterministic.
    torch.testing.assert_close(handle(torch.zeros(4, 4)), handle(torch.zeros(4, 4)))


def test_final_state_arm_returns_a_working_handle(endpoint_files, tmp_path):
    handle = HMCTrainer(_cfg("final_state"), CartPoleSystem(), "cartpole_pybullet").fit(
        endpoint_files, str(tmp_path / "out")
    )
    x = torch.randn(12, 4) * 0.3
    out = handle.predict_endpoint(x)
    assert out.shape == (12, 4)
    assert torch.isfinite(out).all()
    # The final-state handle must draw fresh every call.
    assert not torch.allclose(handle.predict_endpoint(x), handle.predict_endpoint(x))


@pytest.mark.parametrize("head", ["outcome", "final_state"])
def test_writes_a_checkpoint_matching_the_engine_glob(head, tmp_path, outcome_files,
                                                      endpoint_files):
    files = outcome_files if head == "outcome" else endpoint_files
    out = tmp_path / f"out_{head}"
    HMCTrainer(_cfg(head), CartPoleSystem(), "cartpole_pybullet").fit(files, str(out))
    assert list((out / "checkpoints").glob("best*.ckpt"))


def test_writes_an_auditable_diagnostics_artifact(outcome_files, tmp_path):
    """A reference arm whose own convergence cannot be audited is not a reference."""
    out = tmp_path / "out"
    HMCTrainer(_cfg("outcome"), CartPoleSystem(), "cartpole_pybullet").fit(
        outcome_files, str(out)
    )
    d = json.loads((out / "checkpoints" / "hmc_diagnostics.json").read_text())
    assert len(d["chains"]) == 2
    for c in d["chains"]:
        assert 0.0 <= c["accept_rate"] <= 1.0
        assert c["step_size"] > 0.0
        assert "divergences" in c
    assert d["rhat_max"] >= 1.0
    assert 0.0 < d["ceiling"]["agreement"] <= 1.0


def test_reference_tier_pins_pos_weight_and_beta(outcome_files, endpoint_files, tmp_path):
    """beta-NLL is not a likelihood and pos_weight tempers one, so neither may
    reach the reference target -- otherwise HMC references a posterior no arm
    is approximating."""
    t = HMCTrainer(_cfg("outcome"), CartPoleSystem(), "cartpole_pybullet")
    t.fit(outcome_files, str(tmp_path / "a"))
    assert t.pos_weight == 1.0

    t2 = HMCTrainer(_cfg("final_state"), CartPoleSystem(), "cartpole_pybullet")
    t2.fit(endpoint_files, str(tmp_path / "b"))
    assert t2.head.beta == 0.0


def test_resume_checkpoint_raises_rather_than_being_ignored(outcome_files, tmp_path):
    """Silently accepting and discarding a resume path is the exact trap that
    made two sibling arms appear to warm-start when they did not."""
    with pytest.raises(ValueError, match="does not resume"):
        HMCTrainer(_cfg("outcome"), CartPoleSystem(), "cartpole_pybullet").fit(
            outcome_files, str(tmp_path / "out"), resume_checkpoint="/some/path.ckpt"
        )


def _write_imbalanced_classification(path, n=200, pos_frac=0.1, seed=0):
    """A classification fixture whose data-derived pos_weight is clearly != 1.0."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, size=(n, 4))
    n_pos = max(1, int(n * pos_frac))
    y = np.zeros(n, dtype=int)
    y[:n_pos] = 1
    rng.shuffle(y)
    np.savetxt(path, np.column_stack([X, y]))
    return str(path)


def _expected_pos_weight(train_path):
    """Reproduce AdaptiveClassificationDataModule's n_neg/n_pos computation."""
    data = np.loadtxt(train_path)
    y = data[:, -1]
    n_pos = float((y == 1.0).sum())
    n_neg = float((y == 0.0).sum())
    return (n_neg / n_pos) if n_pos > 0 else 1.0


def _bnn_cfg(**overrides):
    bnn = {"hidden_dims": [8, 8], "lr": 1e-2, "weight_decay": 1e-5,
           "max_epochs": 2, "patience": 5, "prior_sigma": 1.0,
           "kl_weight": 1.0, "n_marginal_samples": 8, "posterior": "mfvi"}
    bnn.update(overrides)
    return OmegaConf.create({
        "device": "cpu",
        "predictor": {"type": "classifier", "name": "bnn_mfvi",
                      "batch_size": 64, "bnn": bnn},
    })


def test_reference_tier_pos_weight_override_reaches_the_bnn_module(tmp_path):
    """The reference tier composes predictor.bnn.pos_weight = 1.0 so the BNN
    outcome arms target the SAME unweighted posterior HMC does (HMC itself never
    reads pos_weight from config -- it is hardcoded to 1.0). If
    BayesianMLPTrainer ignored the key and always used the data-derived weight,
    the BNN arms would stay on a class-reweighted target while HMC referenced the
    unweighted one, and the mismatch would be invisible in the diagnostics."""
    train_file = _write_imbalanced_classification(tmp_path / "tr.txt", pos_frac=0.1, seed=0)
    val_file = _write_imbalanced_classification(tmp_path / "va.txt", n=80, pos_frac=0.1, seed=1)
    files = {"train": train_file, "val": val_file}

    data_derived = _expected_pos_weight(train_file)
    assert data_derived != pytest.approx(1.0), (
        "fixture must be imbalanced or this test cannot tell the override worked"
    )

    override_dir = tmp_path / "override"
    BayesianMLPTrainer(_bnn_cfg(pos_weight=1.0), CartPoleSystem(), "cartpole_pybullet").fit(
        files, str(override_dir)
    )
    override_ckpt = sorted((override_dir / "checkpoints").glob("best*.ckpt"))[0]
    override_pw = torch.load(
        override_ckpt, map_location="cpu", weights_only=False
    )["state_dict"]["pos_weight"].item()

    default_dir = tmp_path / "default"
    BayesianMLPTrainer(_bnn_cfg(), CartPoleSystem(), "cartpole_pybullet").fit(
        files, str(default_dir)
    )
    default_ckpt = sorted((default_dir / "checkpoints").glob("best*.ckpt"))[0]
    default_pw = torch.load(
        default_ckpt, map_location="cpu", weights_only=False
    )["state_dict"]["pos_weight"].item()

    assert override_pw == pytest.approx(1.0)
    assert default_pw == pytest.approx(data_derived)
    assert override_pw != pytest.approx(default_pw)
