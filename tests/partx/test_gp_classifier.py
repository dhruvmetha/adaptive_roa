import numpy as np
import pytest
from adaptive_roa.systems.pendulum import PendulumSystem
from adaptive_roa.partx.gp_classifier import GPClassifier


def _disk_data(n=400, seed=0):
    """Success (1) inside a disk in (θ, θ̇) space, failure (0) outside."""
    rng = np.random.default_rng(seed)
    X = np.column_stack([rng.uniform(-3.0, 3.0, n), rng.uniform(-8.0, 8.0, n)])
    r2 = (X[:, 0] / 1.5) ** 2 + (X[:, 1] / 4.0) ** 2
    y = (r2 < 1.0).astype(np.int64)
    return X, y


def test_gp_recovers_disk_boundary():
    system = PendulumSystem()
    X, y = _disk_data()
    gp = GPClassifier(system, n_inducing=64, n_iters=250).fit(X, y)

    center = np.array([[0.0, 0.0]])          # deep inside disk -> success
    outside = np.array([[2.8, 7.0]])         # far outside -> failure
    assert gp.p_success(center)[0] > 0.8
    assert gp.p_success(outside)[0] < 0.2

    # Latent variance is finite and larger far from data density at the edges.
    m, s2 = gp.latent_posterior(np.array([[1.5, 0.0]]))  # near boundary
    assert np.isfinite(m).all() and (s2 > 0).all()


def test_gp_state_dict_round_trip():
    system = PendulumSystem()
    X, y = _disk_data()
    gp = GPClassifier(system, n_inducing=64, n_iters=250).fit(X, y)

    probe = np.array([[0.0, 0.0], [2.8, 7.0], [1.5, 0.0]])
    original_p = gp.p_success(probe)

    sd = gp.state_dict()

    fresh = GPClassifier(system, n_inducing=64)
    fresh.load_state_dict(sd)
    restored_p = fresh.p_success(probe)

    np.testing.assert_allclose(restored_p, original_p, atol=1e-4)


def test_load_state_dict_syncs_the_kernel_attribute():
    """M5. ``load_state_dict`` built the module from ``sd["kernel"]`` but left
    ``self.kernel`` at its constructor value, so the next ``state_dict()`` wrote
    the WRONG kernel name. Because matern52 and rbf expose identical state-dict
    keys, the following epoch's load then succeeded silently against a genuinely
    different kernel."""
    import gpytorch

    system = PendulumSystem()
    X, y = _disk_data(n=120)
    trained = GPClassifier(system, n_inducing=16, kernel="matern52", n_iters=5).fit(X, y)

    mismatched = GPClassifier(system, n_inducing=16, kernel="rbf")
    mismatched.load_state_dict(trained.state_dict())

    assert mismatched.kernel == "matern52"
    assert mismatched.state_dict()["kernel"] == "matern52"
    # ... and the attribute now agrees with the module that was actually built.
    assert isinstance(mismatched.model.covar_module.base_kernel, gpytorch.kernels.MaternKernel)


def test_load_state_dict_returns_self():
    """Matches GPRegressor.load_state_dict so both are chainable."""
    system = PendulumSystem()
    X, y = _disk_data(n=120)
    trained = GPClassifier(system, n_inducing=16, n_iters=5).fit(X, y)
    fresh = GPClassifier(system, n_inducing=16)
    assert fresh.load_state_dict(trained.state_dict()) is fresh


def test_warm_start_grows_the_inducing_set_as_the_pool_grows():
    """The inducing-point capacity cap. ``min(n_inducing, n)`` ran on the COLD
    path only, so with ``warm_start: true`` epoch 0's dataset size pinned model
    capacity for the whole run: configs/adaptive_v2/experiment/partx_pendulum.yaml
    starts at initial_train_size=50 and grows to 500, leaving a GP configured for
    128 inducing points stuck at 50 while its pool grew tenfold."""
    import torch

    system = PendulumSystem()
    X0, y0 = _disk_data(n=50, seed=0)
    gp = GPClassifier(system, n_inducing=128, n_iters=5).fit(X0, y0)
    assert gp.inducing_count() == 50           # capped by epoch 0's pool

    old_z = gp.model.variational_strategy.inducing_points.detach().clone()
    p_before = gp.p_success(X0)

    X1, y1 = _disk_data(n=500, seed=1)
    gp.n_iters = 0                              # inducing LOCATIONS are learnable
    gp.fit(X1, y1)

    assert gp.inducing_count() == 128, "inducing set did not grow with the pool"
    new_z = gp.model.variational_strategy.inducing_points.detach()
    assert torch.equal(new_z[:50], old_z), "existing inducing locations were not preserved"
    # Growing is not a disguised cold start: with no further optimization the
    # predictive probabilities are unchanged, because the whitened block
    # embedding is exact (see predictors/gp_inducing.py).
    np.testing.assert_allclose(gp.p_success(X0), p_before, atol=2e-2)
