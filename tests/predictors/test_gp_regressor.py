import numpy as np
import pytest
import torch

from adaptive_roa.predictors.gp_regressor import GPRegressor, MultitaskSVGP


def _toy(n=256, d=3, t=2, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, size=(n, d)).astype(np.float32)
    Y = np.stack([X[:, 0] * 0.8, -X[:, 1] * 0.5], axis=-1).astype(np.float32)
    return torch.from_numpy(X), torch.from_numpy(Y)


def _toy_t(num_tasks, n=256, d=3, seed=0):
    """Toy problem with an arbitrary number of output tasks."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, size=(n, d)).astype(np.float32)
    W = (rng.normal(size=(d, num_tasks)) * 0.5).astype(np.float32)
    return torch.from_numpy(X), torch.from_numpy((X @ W).astype(np.float32))


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
    unfair advantage over the BNN arms. Predictive draws must be strictly wider,
    and the gap must equal exactly the likelihood's learned noise -- global noise
    plus each task's own noise -- not merely be some positive amount."""
    gp, X, _Y = _fitted()
    latent_var = gp.latent_variance(X[:32])
    pred_var = gp.predictive_variance(X[:32])
    assert (pred_var > latent_var).all()
    assert torch.isfinite(pred_var).all()

    gap = pred_var - latent_var
    expected_gap = (gp.likelihood.noise + gp.likelihood.task_noises).detach()
    torch.testing.assert_close(gap, expected_gap.expand_as(gap), rtol=1e-4, atol=1e-4)


def test_empirical_spread_matches_the_predictive_variance():
    gp, X, _Y = _fitted()
    draws = gp.sample(X[:4], num_samples=4000)
    torch.testing.assert_close(
        draws.var(dim=0), gp.predictive_variance(X[:4]), rtol=0.15, atol=1e-3
    )


@pytest.mark.parametrize("num_tasks,n_query", [(3, 1024), (5, 512), (13, 2048)])
def test_draws_keep_their_spread_at_production_batch_sizes(num_tasks, n_query):
    """C1 regression. ``sample`` draws from the JOINT MVN over the query batch,
    and gpytorch swaps the exact Cholesky root for a rank-100 truncated Lanczos
    root once the joint dimension exceeds ``max_cholesky_size`` (default 800).
    The joint dimension is ``N * num_tasks``, so the threshold is N > 800/T:
    pendulum (T=3) trips at 267 rows, quadrotor3d (T=13) at 62.

    Every production call site is above that line -- probability_estimator.py:143
    (1024), full_roa.py:600 (2048), mc_cache.py:143 (2048),
    endpoint_evaluation.py:109 (512), gaussian_process.py (whole split). Past the
    threshold the draws collapsed to 4-18% of the analytic predictive variance,
    which silently removed the arm's aleatoric noise floor, made p_success a
    function of how the caller batched, and biased endpoint_error low against the
    sibling arms.

    The parametrization deliberately samples at N*T of 3072, 2560 and 26624 --
    production-like batches, all far above 800. A version of this test at N=4
    (N*T=8) is what let the defect through.
    """
    X, Y = _toy_t(num_tasks, n=400, d=4)
    gp = GPRegressor(num_tasks=num_tasks, input_dim=4, n_inducing=32,
                     n_iters=40, batch_size=256).fit(X, Y)

    rng = np.random.default_rng(1)
    Xq = torch.from_numpy(rng.uniform(-1, 1, size=(n_query, 4)).astype(np.float32))
    assert n_query * num_tasks > 800, "test must sample ABOVE the threshold"

    ratio = (gp.sample(Xq, num_samples=512).var(dim=0)
             / gp.predictive_variance(Xq)).mean().item()
    assert 0.9 < ratio < 1.1, (
        f"empirical/predictive variance ratio {ratio:.4f} at N={n_query}, "
        f"T={num_tasks} (joint {n_query * num_tasks}); draws are not carrying "
        f"the analytic predictive spread"
    )


def test_sampling_is_independent_of_how_the_caller_batches():
    """p_success must be a function of the query state, not of the batch the
    caller happened to put it in. Measured before the fix: the spread of a fixed
    query set changed by an order of magnitude between a 64-row and a 2048-row
    call."""
    X, Y = _toy_t(3, n=400, d=4)
    gp = GPRegressor(num_tasks=3, input_dim=4, n_inducing=32,
                     n_iters=40, batch_size=256).fit(X, Y)
    rng = np.random.default_rng(2)
    Xq = torch.from_numpy(rng.uniform(-1, 1, size=(2048, 4)).astype(np.float32))

    small = gp.sample(Xq[:64], num_samples=400).var(dim=0).mean().item()
    large = gp.sample(Xq, num_samples=400).var(dim=0)[:64].mean().item()
    assert abs(large - small) / small < 0.15, (
        f"per-point spread depends on batch size: {small:.5g} in a 64-row call "
        f"vs {large:.5g} in a 2048-row one"
    )


def test_the_chunk_budget_is_counted_in_rows_times_tasks():
    """Dividing the 800-element budget by rows alone would leave quadrotor3d
    (13 tasks) tripping the Lanczos threshold at 62 rows."""
    import gpytorch

    budget = gpytorch.settings.max_cholesky_size.value()
    for num_tasks in (1, 3, 13):
        gp = GPRegressor(num_tasks=num_tasks, input_dim=2)
        rows = gp._chunk_rows()
        assert rows >= 1
        assert rows * num_tasks <= budget


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


def test_input_dim_is_validated_rather_than_silently_overwritten():
    """`input_dim` used to be accepted and never checked; `_build` derived the
    real width from the data, so a wrong constructor value was unfalsifiable and
    the ARD lengthscales would be learned against whatever arrived."""
    X, Y = _toy()  # d = 3
    with pytest.raises(ValueError, match="input_dim"):
        GPRegressor(num_tasks=2, input_dim=7, n_inducing=8, n_iters=1).fit(X, Y)
    with pytest.raises(ValueError, match="num_tasks"):
        GPRegressor(num_tasks=5, input_dim=3, n_inducing=8, n_iters=1).fit(X, Y)


def test_load_state_dict_refreshes_input_dim():
    """The export wrapper constructs with a placeholder input_dim=1 and loads;
    leaving the attribute stale would misreport the loaded architecture."""
    gp, _X, _Y = _fitted()
    restored = GPRegressor(num_tasks=1, input_dim=1)
    restored.load_state_dict(gp.state_dict())
    assert restored.input_dim == 3
    assert restored.inducing_count() == gp.inducing_count() == 16


def test_validation_selection_keeps_the_best_state_not_the_last():
    """gp_reg must be selected the same way its BNN siblings are (best val_nll),
    not by taking whatever the fixed iteration budget ends on."""
    X, Y = _toy(n=200)
    X_val, Y_val = _toy(n=120, seed=7)
    gp = GPRegressor(num_tasks=2, input_dim=3, n_inducing=16, n_iters=40,
                     eval_every=5, lr=0.05).fit(X, Y, X_val, Y_val)

    assert gp.best_val_nll is not None
    # The restored state must actually BE the best one seen: re-scoring it now
    # has to reproduce the recorded score, and no worse.
    assert gp.val_nll(X_val, Y_val) == pytest.approx(gp.best_val_nll, rel=1e-5)

    # ... and it must beat the terminal state of the same budget without
    # selection, or the mechanism is inert.
    torch.manual_seed(0)
    unselected = GPRegressor(num_tasks=2, input_dim=3, n_inducing=16, n_iters=40,
                             eval_every=5, lr=0.05).fit(X, Y)
    assert gp.best_val_nll <= unselected.val_nll(X_val, Y_val) + 1e-6


def test_no_validation_data_keeps_the_terminal_state():
    """Backwards-compatible: fit(X, Y) alone must still work and not select."""
    gp, _X, _Y = _fitted()
    assert gp.best_val_nll is None


def test_warm_start_grows_the_inducing_set_as_the_pool_grows():
    """The inducing-point capacity cap. `min(n_inducing, n)` ran on the COLD path
    only, so under warm_start epoch 0's dataset size pinned capacity forever --
    partx_pendulum starts at 50 rows and grows to 500 with n_inducing=128, so the
    GP would have stayed at 50 inducing points all run."""
    X0, Y0 = _toy(n=40)
    gp = GPRegressor(num_tasks=2, input_dim=3, n_inducing=128, n_iters=2).fit(X0, Y0)
    assert gp.inducing_count() == 40  # capped by epoch 0's pool, as intended

    old_z = gp._base_strategy.inducing_points.detach().clone()
    X1, Y1 = _toy(n=400, seed=3)
    # n_iters=0 so the assertion below sees the state right after the top-up;
    # inducing LOCATIONS are learnable, so any further optimization moves them.
    gp.n_iters = 0
    gp.fit(X1, Y1)

    assert gp.inducing_count() == 128, "inducing set did not grow with the pool"
    # Existing locations preserved, new ones appended.
    new_z = gp._base_strategy.inducing_points.detach()
    assert torch.equal(new_z[..., :40, :], old_z)


def test_growing_the_inducing_set_preserves_the_learned_posterior():
    """Growing must not be a disguised cold start: with zero further optimization
    the predictive mean over the old inducing block is unchanged, because the
    whitened parameterization makes the block embedding exact."""
    X0, Y0 = _toy(n=40)
    gp = GPRegressor(num_tasks=2, input_dim=3, n_inducing=128, n_iters=120,
                     lr=0.05).fit(X0, Y0)
    before = gp.mean(X0).clone()
    lengthscale_before = gp.model.covar_module.base_kernel.raw_lengthscale.detach().clone()

    gp.n_iters = 0
    X1, Y1 = _toy(n=400, seed=3)
    gp.fit(X1, Y1)

    assert gp.inducing_count() == 128
    torch.testing.assert_close(gp.mean(X0), before, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(
        gp.model.covar_module.base_kernel.raw_lengthscale, lengthscale_before
    )


def test_minibatching_is_used_so_large_training_sets_are_viable():
    """quadrotor2d caps at 12k rows and quadrotor3d at 17k; a full-batch loop over
    that many points across every output task is not viable.

    This observes the ACTUAL tensor sizes reaching the GP's forward pass during
    fit, rather than trusting last_batches_per_iter's self-reported arithmetic
    (a `fit` that trains on the full batch every step while still computing
    last_batches_per_iter with the right formula would pass a test that only
    checks that arithmetic -- it must not pass this one).

    gpytorch's VariationalStrategy concatenates the inducing points with the
    data slice before calling model.forward, so the size of the data slice
    that actually reached this training step is `x.size(-2) - n_inducing`.
    """
    X, Y = _toy(n=2000)
    n_inducing = 16
    batch_size = 256
    n_iters = 5

    seen_data_sizes = []
    orig_forward = MultitaskSVGP.forward

    def traced_forward(self, x):
        seen_data_sizes.append(x.size(-2) - n_inducing)
        return orig_forward(self, x)

    MultitaskSVGP.forward = traced_forward
    try:
        gp = GPRegressor(num_tasks=2, input_dim=3, n_inducing=n_inducing,
                          n_iters=n_iters, batch_size=batch_size)
        gp.fit(X, Y)
    finally:
        MultitaskSVGP.forward = orig_forward

    assert len(seen_data_sizes) > 0
    assert all(size <= batch_size for size in seen_data_sizes)
    batches_per_epoch = len(seen_data_sizes) / n_iters
    assert batches_per_epoch > 1
