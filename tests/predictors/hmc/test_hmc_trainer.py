import json
import warnings

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from adaptive_roa.adaptive_v2.trainers.bayesian_mlp_trainer import BayesianMLPTrainer
from adaptive_roa.adaptive_v2.trainers.hmc_trainer import (
    HMCConvergenceError,
    HMCTrainer,
    _convergence_warning,
)
from adaptive_roa.predictors.heads import FinalStateHead
from adaptive_roa.predictors.hmc.sampler import _ess_and_acf1
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


def _cfg(head, run_seed=0, **overrides):
    """Mirrors the shipped arm configs: NO `predictor.hmc.seed`, so the chains
    follow the run seed exactly as they do in production."""
    hmc = {"hidden_dims": [8, 8], "activation": "tanh", "prior_sigma": 1.0,
           "n_chains": 2, "n_samples": 20, "n_warmup": 20, "n_leapfrog": 8}
    hmc.update(overrides)
    return OmegaConf.create({
        "device": "cpu",
        "seed": run_seed,
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


@pytest.mark.parametrize("head", ["outcome", "final_state"])
def test_writes_an_auditable_diagnostics_artifact(head, tmp_path, outcome_files,
                                                  endpoint_files):
    """A reference arm whose own convergence cannot be audited is not a reference.

    Parameterized over BOTH heads. It used to hardcode `outcome`, which is the
    head that happens to pass: on `final_state` it would have failed on
    `agreement == 0.0`, and that failure was the visible end of two real defects
    -- chains that do not mix, and a total-variation "distance" of 3.89 from
    feeding an unbounded scalar to a function defined on probabilities.
    """
    files = outcome_files if head == "outcome" else endpoint_files
    out = tmp_path / f"out_{head}"
    n_samples = 20   # _cfg's budget
    with warnings.catch_warnings():   # a non-converged fixture is expected here
        warnings.simplefilter("ignore", RuntimeWarning)
        HMCTrainer(_cfg(head), CartPoleSystem(), "cartpole_pybullet").fit(files, str(out))
    d = json.loads((out / "checkpoints" / "hmc_diagnostics.json").read_text())
    samples = torch.load(out / "checkpoints" / "best-hmc.ckpt", map_location="cpu",
                         weights_only=False)["samples"]

    assert len(d["chains"]) == 2
    for i, c in enumerate(d["chains"]):
        assert 0.0 <= c["accept_rate"] <= 1.0
        assert c["step_size"] > 0.0
        assert c["divergences"] >= 0 and c["warmup_divergences"] >= 0

        # ESS and lag-1 autocorrelation are the ONLY recorded quantities that
        # can see a trajectory-length resonance -- it reads as full acceptance
        # and zero divergences while destroying the chain's second moments. So
        # they are checked against a recomputation from the chain's own stored
        # draws, not merely for being in range: a plausible-looking constant
        # (`ess = n_samples`, `acf1 = 0`) is exactly what a dead field looks
        # like, and it satisfies any range assertion.
        chain = samples[i * n_samples:(i + 1) * n_samples]
        per_dim = [_ess_and_acf1(chain[:, dim] ** 2) for dim in range(chain.shape[1])]
        assert c["ess"] == pytest.approx(min(e for e, _ in per_dim), rel=1e-9)
        assert c["acf1"] == pytest.approx(max(a for _, a in per_dim), rel=1e-9)
        assert 0.0 < c["ess"] <= n_samples
        assert -1.0 <= c["acf1"] <= 1.0

    assert d["rhat_max"] >= 1.0
    assert d["converged"] == (d["rhat_max"] <= d["rhat_threshold"])
    # The ceiling is defined on PROBABILITIES for both heads: agreement
    # thresholds at 0.5 and total_variation is a distance in [0, 1].
    ceiling = d["ceiling"]
    assert 0.0 <= ceiling["agreement"] <= 1.0
    assert 0.0 <= ceiling["agreement_min"] <= ceiling["agreement"]
    assert 0.0 <= ceiling["total_variation"] <= 1.0
    assert ceiling["total_variation"] <= ceiling["total_variation_max"] <= 1.0
    # A ceiling computed where every point shares a label is trivially 1.0 and
    # says nothing about fidelity; positive_rate is what makes that visible.
    assert 0.0 <= ceiling["positive_rate"] <= 1.0


@pytest.mark.parametrize("key,value", [("pos_weight", 2.0), ("beta_nll", 0.5),
                                       ("beta", 0.5)])
def test_a_tempering_key_is_refused_rather_than_ignored(key, value, outcome_files,
                                                        tmp_path):
    """These keys are never read, so accepting one would let a caller believe the
    reference had been tempered to match an arm when it had not.

    This replaces an assert that read back a literal assigned two lines above
    (`self.pos_weight = 1.0; assert self.pos_weight == 1.0`) against an
    attribute nothing else used. That assert could not fail; this can.
    """
    with pytest.raises(ValueError, match="REFERENCE arm"):
        HMCTrainer(_cfg("outcome", **{key: value}), CartPoleSystem(),
                   "cartpole_pybullet").fit(outcome_files, str(tmp_path / "out"))


def test_the_head_the_handle_serves_is_the_untempered_one(endpoint_files, tmp_path):
    """beta-NLL multiplies each dim by a detached sigma^(2*beta) and is not a
    likelihood, so the reference must run at beta=0. Read off the RETURNED
    HANDLE -- the object that actually scores at serve time -- not off a
    trainer attribute that exists only to be read back.
    """
    handle = HMCTrainer(_cfg("final_state"), CartPoleSystem(), "cartpole_pybullet").fit(
        endpoint_files, str(tmp_path / "b")
    )
    assert handle.head.beta == 0.0

    # Behavioural, not just the attribute: at beta=0 the head's NLL is the plain
    # Gaussian one, which differs from any beta != 0 reweighting on a fixture
    # whose sigmas are not all 1.
    params = torch.randn(16, handle.head.n_params) * 0.5
    target = torch.randn(16, 4) * 0.3
    tempered = FinalStateHead(CartPoleSystem(), beta=0.5)
    assert not torch.allclose(handle.head.nll(params, target),
                              tempered.nll(params, target))


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


#
# Run-seed plumbing (the reference arm's own run-to-run floor)
# -----------------------------------------------------------
# `base_seed = int(hmc_cfg.get("seed", 0))` read `predictor.hmc.seed`, which both
# shipped arm configs pinned to 0. Every source of randomness in the arm derives
# from it, so run seeds 42/43/44 produced BIT-IDENTICAL draws and
# `pl.seed_everything(cfg.seed)` had no effect. For a reference arm that is the
# worst version of this bug: the reference's own sampling variability is exactly
# the scale an approximation's gap must be judged against, and it was 0 by
# construction.
#
def test_chain_seed_base_follows_the_run_seed():
    for seed in (42, 43, 44):
        t = HMCTrainer(_cfg("outcome", run_seed=seed), CartPoleSystem(),
                       "cartpole_pybullet")
        assert t._chain_seed_base() == seed


def test_an_explicit_hmc_seed_still_wins():
    t = HMCTrainer(_cfg("outcome", run_seed=43, seed=7), CartPoleSystem(),
                   "cartpole_pybullet")
    assert t._chain_seed_base() == 7


def test_different_run_seeds_draw_different_chains(outcome_files, tmp_path):
    """The behavioural end of it: two run seeds must not produce identical draws."""
    def samples(seed):
        out = tmp_path / f"s{seed}"
        HMCTrainer(_cfg("outcome", run_seed=seed), CartPoleSystem(),
                   "cartpole_pybullet").fit(outcome_files, str(out))
        return torch.load(out / "checkpoints" / "best-hmc.ckpt",
                          map_location="cpu", weights_only=False)["samples"]

    a, b = samples(42), samples(43)
    assert a.shape == b.shape
    assert not torch.allclose(a, b), "seed replicates are bit-identical"
    # Same seed still reproduces exactly.
    assert torch.equal(a, samples(42))


def test_the_shipped_arm_configs_do_not_pin_the_chain_seed():
    """Pinning `seed` in the arm config is what made the run seed unreachable, so
    an explicit-wins fallback alone is not enough -- the key must be absent."""
    for name in ("hmc", "hmc_reg"):
        cfg = OmegaConf.load(f"configs/adaptive_v2/predictor/{name}.yaml")
        assert cfg.predictor.hmc.get("seed") is None, (
            f"{name}.yaml pins predictor.hmc.seed; run seeds cannot reach the chains"
        )


#
# Convergence gate
# ----------------
# `fit()` wrote `rhat_max: 91.4` to hmc_diagnostics.json, returned a
# normal-looking handle, and flowed into the benchmark. Nothing anywhere read
# that file (`grep -rn hmc_diagnostics` found only the writer), so a fully
# diverged reference was indistinguishable from a healthy one.
#
def test_the_gate_passes_a_converged_run():
    assert _convergence_warning(1.02, [{"accept_rate": 0.8, "step_size": 0.03,
                                        "divergences": 0}], 1.1) is None


@pytest.mark.parametrize("rhat_max", [1.19, 26.0, 91.4])
def test_the_gate_fires_on_the_observed_non_converged_values(rhat_max):
    """1.19 is the outcome arm at the production budget; 26.0 and 91.4 are the
    final-state arm at the truncated and production budgets. All three exceed
    the 1.1 Gelman-Rubin convention and must be flagged."""
    msg = _convergence_warning(
        rhat_max,
        [{"accept_rate": 0.295, "step_size": 4.2e-4, "divergences": 46},
         {"accept_rate": 0.72, "step_size": 6.9e-4, "divergences": 0}],
        1.1,
    )
    assert msg is not None
    assert "DID NOT CONVERGE" in msg
    assert f"{rhat_max:.4g}" in msg
    assert "46" in msg, "the worst chain's divergence count must be reported"


def test_a_non_converged_fit_warns_and_records_it(endpoint_files, tmp_path):
    """The final-state fixture is the real non-mixing case (rhat_max ~ 26 at this
    budget). fit() must say so out loud AND leave a machine-readable flag."""
    out = tmp_path / "out"
    with pytest.warns(RuntimeWarning, match="DID NOT CONVERGE"):
        HMCTrainer(_cfg("final_state"), CartPoleSystem(), "cartpole_pybullet").fit(
            endpoint_files, str(out)
        )
    d = json.loads((out / "checkpoints" / "hmc_diagnostics.json").read_text())
    assert d["converged"] is False
    assert d["rhat_max"] > d["rhat_threshold"]


def test_strict_convergence_refuses_to_return_the_handle(endpoint_files, tmp_path):
    with pytest.raises(HMCConvergenceError, match="DID NOT CONVERGE"):
        HMCTrainer(_cfg("final_state", strict_convergence=True), CartPoleSystem(),
                   "cartpole_pybullet").fit(endpoint_files, str(tmp_path / "out"))
