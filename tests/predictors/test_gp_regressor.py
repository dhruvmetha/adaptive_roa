import numpy as np
import pytest
import torch

from adaptive_roa.predictors.gp_regressor import GPRegressor


def _toy(n=256, d=3, t=2, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, size=(n, d)).astype(np.float32)
    Y = np.stack([X[:, 0] * 0.8, -X[:, 1] * 0.5], axis=-1).astype(np.float32)
    return torch.from_numpy(X), torch.from_numpy(Y)


def _fitted(n_iters=60, **kw):
    X, Y = _toy()
    return GPRegressor(num_tasks=2, input_dim=3, n_inducing=16, n_iters=n_iters, **kw).fit(X, Y), X, Y


def test_sample_shape_and_mean_shape():
    gp, X, _Y = _fitted()
    assert gp.sample(X[:7], num_samples=5).shape == (5, 7, 2)
    assert gp.mean(X[:7]).shape == (7, 2)


def test_sampling_goes_through_the_likelihood_so_it_carries_observation_noise():
    """The spec requires the GP's spread to include the aleatoric noise floor.
    Latent-function draws (model(x).sample()) omit it and would hand the GP an
    unfair advantage over the BNN arms. Predictive draws must be strictly wider."""
    gp, X, _Y = _fitted()
    latent_var = gp.latent_variance(X[:32])
    pred_var = gp.predictive_variance(X[:32])
    assert (pred_var > latent_var).all()
    # The gap is exactly the likelihood's learned noise.
    assert torch.isfinite(pred_var).all()


def test_empirical_spread_matches_the_predictive_variance():
    gp, X, _Y = _fitted()
    draws = gp.sample(X[:4], num_samples=4000)
    torch.testing.assert_close(
        draws.var(dim=0), gp.predictive_variance(X[:4]), rtol=0.15, atol=1e-3
    )


def test_fit_actually_learns_the_toy_mapping():
    """Guards against shipping an untrained GP: the fitted mean must beat
    predicting zeros by a clear margin."""
    gp, X, Y = _fitted(n_iters=300)
    fitted_mse = (gp.mean(X) - Y).pow(2).mean().item()
    zero_mse = Y.pow(2).mean().item()
    assert fitted_mse < 0.5 * zero_mse


def test_state_dict_round_trips_through_a_fresh_instance():
    """A gp_reg checkpoint is NOT load-compatible with the single-output
    GPClassifier -- its keys nest a level deeper and it needs num_tasks at
    construction. The schema must carry enough to rebuild without a config."""
    gp, X, _Y = _fitted()
    sd = gp.state_dict()

    restored = GPRegressor(num_tasks=2, input_dim=3, n_inducing=16)
    restored.load_state_dict(sd)
    torch.testing.assert_close(restored.mean(X[:8]), gp.mean(X[:8]), atol=1e-5, rtol=1e-5)


def test_state_dict_carries_num_tasks():
    gp, _X, _Y = _fitted()
    assert gp.state_dict()["num_tasks"] == 2


def test_minibatching_is_used_so_large_training_sets_are_viable():
    """quadrotor2d caps at 12k rows and quadrotor3d at 17k; a full-batch loop over
    that many points across every output task is not viable."""
    X, Y = _toy(n=2000)
    gp = GPRegressor(num_tasks=2, input_dim=3, n_inducing=16, n_iters=5, batch_size=256)
    gp.fit(X, Y)
    assert gp.last_batches_per_iter == pytest.approx(2000 / 256, abs=1.0)
