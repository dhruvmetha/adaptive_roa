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


# --------------------------------------------------------------------------
# Black-box alpha-divergence (BB-alpha) training objective.
#
# Depeweg et al. (ICML 2018) fit q with black-box alpha-divergence minimization
# at alpha = 1.0, and their supplementary shows the decomposition degrading as
# alpha falls toward the variational-Bayes limit. Our MFVI arm IS that limit, so
# reproducing the paper's method needs the alpha-indexed objective.
# --------------------------------------------------------------------------

class _FixedLogitPosterior(torch.nn.Module):
    """Returns pre-set logit draws, so a loss can be checked against hand arithmetic.

    ``forward_samples`` is the only sampling entry point the BB-alpha loss uses;
    a stub keeps the objective's arithmetic separate from MFVI's randomness,
    which is what makes the alpha -> 0 comparison exact rather than statistical.
    """

    def __init__(self, logits):
        super().__init__()
        self.register_buffer("logits", logits)  # [K, B]

    def forward_samples(self, x, S, generator=None):
        assert int(S) == self.logits.shape[0], "stub is pinned to its own K"
        return self.logits.unsqueeze(-1)

    def forward_sample(self, x, generator=None):
        return self.logits[0].unsqueeze(-1)

    def kl_divergence(self):
        return torch.zeros((), dtype=self.logits.dtype)


class _IdentitySystem:
    def normalize_state(self, x):
        return x

    def embed_state_for_model(self, x):
        return x


def _alpha_module(logits, alpha, n_samples):
    from adaptive_roa.adaptive_v2.trainers.bayesian_mlp_trainer import _OutcomeModule

    return _OutcomeModule(
        posterior=_FixedLogitPosterior(logits), system=_IdentitySystem(),
        pos_weight=1.0, lr=1e-3, weight_decay=0.0, kl_weight=1.0, n_train=100,
        alpha=alpha, alpha_n_samples=n_samples,
    )


def _fixture(k=6, b=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(k, b, generator=g, dtype=torch.float64) * 2.0
    y = (torch.rand(b, generator=g, dtype=torch.float64) < 0.5).double()
    return logits, y


def test_bb_alpha_recovers_the_elbo_in_the_alpha_to_zero_limit():
    """The defining property: alpha -> 0 IS the ELBO's expected log-likelihood.

    (1/a)[logsumexp_k(a*l_k) - log K] -> mean_k l_k as a -> 0, so the objective we
    have been training all along is a special case of the new one. Compared
    against the SAME K draws rather than a fresh single draw, so any discrepancy
    is the alpha arithmetic and not sampling noise. float64 throughout: at
    a = 1e-6 the bracket is a difference of two nearly equal logs.
    """
    logits, y = _fixture()
    elbo_nll = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, y.expand_as(logits), reduction="none").mean(dim=0).mean()

    got = _alpha_module(logits, alpha=1e-6, n_samples=logits.shape[0])._bb_alpha_nll(
        torch.zeros(y.shape[0], 1, dtype=torch.float64), y)

    torch.testing.assert_close(got, elbo_nll, rtol=1e-6, atol=1e-9)


def test_bb_alpha_rejects_a_single_weight_draw():
    """K=1 makes the objective alpha-independent, so it must not be accepted.

    logsumexp over one term is that term and log K is 0, leaving (1/a)(a*l) = l:
    the plain likelihood at EVERY alpha. An arm configured that way would train
    identically to the ELBO while being labelled alpha=1, which is the one
    failure mode that would silently invalidate the whole reproduction.
    """
    with pytest.raises(ValueError, match="at least 2 weight draws"):
        _alpha_module(_fixture(k=1)[0], alpha=1.0, n_samples=1)


def test_bb_alpha_moves_with_alpha_and_in_the_direction_jensen_requires():
    """alpha must actually change the objective, and change it the right way.

    log mean_k p >= mean_k log p (Jensen), so the alpha=1 loss -- the log of the
    averaged likelihood -- is never above the alpha -> 0 loss on the same draws.
    Strict inequality whenever the draws disagree, which is what makes alpha=1
    tolerate a q that spreads mass across regions the ELBO would penalise.
    """
    logits, y = _fixture()
    x = torch.zeros(y.shape[0], 1, dtype=torch.float64)
    k = logits.shape[0]

    near_zero = _alpha_module(logits, alpha=1e-6, n_samples=k)._bb_alpha_nll(x, y)
    at_one = _alpha_module(logits, alpha=1.0, n_samples=k)._bb_alpha_nll(x, y)

    assert at_one < near_zero, (at_one.item(), near_zero.item())


def test_bb_alpha_at_one_is_the_log_of_the_averaged_likelihood():
    """At alpha=1 the objective is -log(mean_k p(y|x,w_k)): the log of the
    MC-averaged likelihood, not the average of the logs. That swap is the whole
    point -- it is what makes the fit mass-covering rather than mode-seeking.
    Rebuilt here from sigmoids so it never calls the implementation."""
    logits, y = _fixture()
    p = torch.sigmoid(logits)
    lik = torch.where(y.expand_as(p).bool(), p, 1.0 - p)      # p(y|w_k) per draw
    expected = -torch.log(lik.mean(dim=0)).mean()

    got = _alpha_module(logits, alpha=1.0, n_samples=logits.shape[0])._bb_alpha_nll(
        torch.zeros(y.shape[0], 1, dtype=torch.float64), y)

    torch.testing.assert_close(got, expected, rtol=1e-10, atol=1e-12)


def test_bb_alpha_is_finite_on_saturated_logits():
    """A naive exp(alpha * loglik) overflows or underflows to zero long before
    logits this size, and a saturated net is exactly where BB-alpha gets used."""
    logits = torch.tensor([[-400.0, 400.0, -400.0], [400.0, -400.0, 400.0]], dtype=torch.float64)
    y = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float64)

    got = _alpha_module(logits, alpha=1.0, n_samples=2)._bb_alpha_nll(
        torch.zeros(3, 1, dtype=torch.float64), y)

    assert torch.isfinite(got), got


def test_alpha_unset_leaves_the_existing_objective_untouched():
    """Every already-scored BNN arm was trained without alpha. The new path must
    be opt-in, or those runs stop being reproducible from this code."""
    from adaptive_roa.adaptive_v2.trainers.bayesian_mlp_trainer import _OutcomeModule

    logits, y = _fixture(k=1)
    m = _OutcomeModule(
        posterior=_FixedLogitPosterior(logits), system=_IdentitySystem(),
        pos_weight=1.0, lr=1e-3, weight_decay=0.0, kl_weight=1.0, n_train=100)
    assert m.alpha is None

    batch = {"inputs": torch.zeros(y.shape[0], 1, dtype=torch.float64), "label": y}
    expected = torch.nn.functional.binary_cross_entropy_with_logits(logits[0], y)
    # float32 tolerance: pos_weight is a registered buffer and therefore float32,
    # which caps the precision of both paths equally.
    torch.testing.assert_close(m._step(batch, "train"), expected, rtol=1e-6, atol=1e-7)


def test_alpha_arm_is_selected_on_the_predictive_nll_not_the_gibbs_nll():
    """Model selection must not fight the objective.

    Gibbs NLL (mean_k -log p) exceeds predictive NLL (-log mean_k p) by exactly
    the Jensen gap -- the posterior spread alpha=1 exists to preserve. Since q
    starts near-deterministic (rho_init=-5) and widens while training, a
    Gibbs-monitored run can stop when the arm starts working and keep the
    narrowest q. val_nll therefore carries the predictive value; the Gibbs value
    is still logged so the alpha arm stays auditable against legacy arms.
    """
    logits, y = _fixture()
    k = logits.shape[0]
    m = _alpha_module(logits, alpha=1.0, n_samples=k)
    x = torch.zeros(y.shape[0], 1, dtype=torch.float64)

    objective, gibbs, predictive = m._alpha_terms(x, y)

    p = torch.sigmoid(logits)
    lik = torch.where(y.expand_as(p).bool(), p, 1.0 - p)
    torch.testing.assert_close(predictive, -torch.log(lik.mean(dim=0)).mean(),
                               rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(gibbs, -torch.log(lik).mean(), rtol=1e-10, atol=1e-12)
    # Jensen, and at alpha=1 the objective IS the predictive NLL.
    assert predictive < gibbs
    torch.testing.assert_close(objective, predictive, rtol=1e-10, atol=1e-12)

    logged = {}
    m.log = lambda name, value, **kw: logged.__setitem__(name, value)
    m._step({"inputs": x, "label": y}, "val")

    # check_dtype=False: _step casts labels to float32 as production does.
    torch.testing.assert_close(logged["val_nll"], predictive, rtol=1e-5, atol=1e-6,
                               check_dtype=False)
    torch.testing.assert_close(logged["val_gibbs_nll"], gibbs, rtol=1e-5, atol=1e-6,
                               check_dtype=False)


def test_alpha_is_refused_on_posteriors_without_a_weight_distribution():
    """Laplace returns its MAP head until `fit` runs and ensemble members are
    deterministic, so K draws would be IDENTICAL and the objective would silently
    be plain BCE at every alpha -- an arm reported as alpha-fitted that is not."""
    from adaptive_roa.adaptive_v2.trainers.bayesian_mlp_trainer import BayesianMLPTrainer
    import numpy as np, tempfile, os

    for kind in ("laplace", "ensemble"):
        with tempfile.TemporaryDirectory() as d:
            f = _write_dataset(Path(d) / "train.txt")
            cfg = _cfg(kind, alpha=1.0, pos_weight=1.0)
            t = BayesianMLPTrainer(cfg, CartPoleSystem(), "cartpole_pybullet")
            with pytest.raises(ValueError, match="only 'mfvi'"):
                t.fit({"train": f, "val": f}, Path(d) / "out")


def test_alpha_is_refused_alongside_a_tempered_likelihood():
    """BB-alpha on a pos_weight-tempered likelihood fits a posterior neither the
    paper's arms nor the HMC reference target. hmc_trainer.py refuses pos_weight
    for the same reason."""
    from adaptive_roa.adaptive_v2.trainers.bayesian_mlp_trainer import BayesianMLPTrainer
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        f = _write_dataset(Path(d) / "train.txt")
        cfg = _cfg("mfvi", alpha=1.0, pos_weight=3.0)
        t = BayesianMLPTrainer(cfg, CartPoleSystem(), "cartpole_pybullet")
        with pytest.raises(ValueError, match="TEMPERED"):
            t.fit({"train": f, "val": f}, Path(d) / "out")
