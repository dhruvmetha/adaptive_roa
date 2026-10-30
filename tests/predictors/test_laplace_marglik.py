"""Evidence-tuned prior precision for last-layer Laplace (tracker C12)."""
import math

import pytest
import torch

from adaptive_roa.predictors.laplace_marglik import log_evidence, optimize_prior_precision
from adaptive_roa.predictors.posteriors import LastLayerLaplacePosterior, accumulate_ggn


def _random_problem(seed=0, d=12, o=3, n=200):
    g = torch.Generator().manual_seed(seed)
    phi = torch.randn(n, d, generator=g, dtype=torch.float64)
    G = phi.T @ phi
    W = torch.randn(o, d, generator=g, dtype=torch.float64) * 0.7
    return G, W


def test_optimum_matches_a_brute_force_grid():
    G, W = _random_problem()
    fit = optimize_prior_precision(G, W)
    g = torch.linalg.eigvalsh(G)
    grid = torch.logspace(-6, 6, 4001, dtype=torch.float64)
    vals = torch.tensor([log_evidence(g, float(W.pow(2).sum()), W.shape[0], float(d))
                         for d in grid])
    best = float(grid[vals.argmax()])
    assert not fit.clipped
    assert math.isclose(fit.prior_precision, best, rel_tol=0.01)
    assert fit.log_evidence >= float(vals.max()) - 1e-9


def test_optimum_satisfies_mackays_fixed_point():
    """delta ||W||^2 = O sum_i g_i / (g_i + delta) at the maximizer."""
    G, W = _random_problem(seed=1)
    fit = optimize_prior_precision(G, W)
    lhs = fit.prior_precision * fit.weight_sqnorm
    assert math.isclose(lhs, fit.effective_params, rel_tol=1e-6)


def test_recovers_the_true_prior_on_bayesian_linear_regression():
    """Weights drawn from N(0, 1/delta0) with plenty of data: the evidence
    should land near delta0. The MAP is fixed (least squares), as in the
    post-hoc use."""
    g = torch.Generator().manual_seed(3)
    d, o, n, sigma, delta0 = 15, 400, 4000, 0.3, 4.0
    phi = torch.randn(n, d, generator=g, dtype=torch.float64)
    W_true = torch.randn(o, d, generator=g, dtype=torch.float64) / math.sqrt(delta0)
    Y = phi @ W_true.T + sigma * torch.randn(n, o, generator=g, dtype=torch.float64)
    W_map = torch.linalg.lstsq(phi, Y).solution.T
    fit = optimize_prior_precision(phi.T @ phi / sigma ** 2, W_map)
    assert 0.8 * delta0 < fit.prior_precision < 1.25 * delta0


def test_flat_data_term_is_reported_as_clipped():
    """With G = 0 the evidence rises without bound as delta falls."""
    fit = optimize_prior_precision(torch.zeros(4, 4, dtype=torch.float64),
                                   torch.ones(2, 4, dtype=torch.float64))
    assert fit.clipped


def test_zero_weights_are_refused():
    with pytest.raises(ValueError, match="identically zero"):
        optimize_prior_precision(torch.eye(3, dtype=torch.float64), torch.zeros(1, 3))


def test_shape_mismatch_is_refused():
    with pytest.raises(ValueError):
        optimize_prior_precision(torch.eye(3), torch.ones(1, 4))


# --- LastLayerLaplacePosterior -------------------------------------------


def _posterior(d_in=5, d_hidden=7, d_out=1, prior=1.0, tune=False, seed=0):
    torch.manual_seed(seed)
    body = torch.nn.Sequential(torch.nn.Linear(d_in, d_hidden), torch.nn.ReLU())
    head = torch.nn.Linear(d_hidden, d_out)
    return LastLayerLaplacePosterior(body, head, prior_precision=prior,
                                     tune_prior_precision=tune)


def _reference_fit(post, features, task, sigma=None):
    """The fit formula as it stood before C12, to pin the default path."""
    phi = torch.cat([features, torch.ones(features.shape[0], 1)], dim=1)
    if task == "outcome":
        p = torch.sigmoid(post.head_layer(features).view(-1))
        lam = (p * (1.0 - p)).clamp_min(1e-6)
    else:
        lam = torch.full((phi.shape[0],), 1.0 / float(sigma) ** 2)
    phi64, lam64 = phi.double(), lam.double()
    H = torch.einsum("n,ni,nj->ij", lam64, phi64, phi64)
    H = H + post.prior_precision * torch.eye(phi.shape[1], dtype=torch.float64)
    return torch.linalg.cholesky(0.5 * (H + H.T))


@pytest.mark.parametrize("task,d_out,sigma", [("outcome", 1, None), ("final_state", 4, 0.3)])
def test_default_fit_is_bit_identical_to_the_pre_c12_formula(task, d_out, sigma):
    post = _posterior(d_out=d_out, prior=2.5)
    x = torch.randn(64, 5)
    with torch.no_grad():
        feats = post.body(x)
        ref = _reference_fit(post, feats, task, sigma)
        post.fit(feats, None, task=task, sigma=sigma)
    assert torch.equal(post.precision_cholesky, ref)
    assert post.prior_precision == 2.5


def test_tuned_fit_uses_the_evidence_maximizer():
    post = _posterior(d_out=3, tune=True)
    x = torch.randn(128, 5)
    with torch.no_grad():
        feats = post.body(x)
        post.fit(feats, None, task="final_state", sigma=0.5)
        G = accumulate_ggn(None, feats, torch.full((128,), 1.0 / 0.25))
    w = torch.cat([post.head_layer.weight, post.head_layer.bias.unsqueeze(1)], 1).double()
    expected = optimize_prior_precision(G, w).prior_precision
    assert math.isclose(post.prior_precision, expected, rel_tol=1e-9)
    assert post.fit_info and post.fit_info[0]["tuned"] is True


def test_chunked_accumulation_matches_one_shot():
    feats = torch.randn(1000, 6)
    lam = torch.rand(1000)
    one = accumulate_ggn(None, feats, lam)
    acc = None
    for i in range(0, 1000, 137):
        acc = accumulate_ggn(acc, feats[i:i + 137], lam[i:i + 137])
    torch.testing.assert_close(acc, one, rtol=1e-12, atol=1e-10)


def test_fit_from_ggn_matches_fit():
    post_a, post_b = _posterior(d_out=2, prior=1.7), _posterior(d_out=2, prior=1.7)
    x = torch.randn(80, 5)
    with torch.no_grad():
        feats = post_a.body(x)
        post_a.fit(feats, None, task="final_state", sigma=0.4)
        post_b.fit_from_ggn(accumulate_ggn(None, feats, torch.full((80,), 1 / 0.16)))
    torch.testing.assert_close(post_a.precision_cholesky, post_b.precision_cholesky,
                               rtol=1e-10, atol=1e-10)


def test_block_mode_keeps_unblocked_rows_at_map_and_matches_block_covariance():
    torch.manual_seed(0)
    post = LastLayerLaplacePosterior(torch.nn.Identity(), torch.nn.Linear(3, 4),
                                     prior_precision=2.0)
    with torch.no_grad():
        feats = torch.randn(50, 3)
        G = accumulate_ggn(None, feats, torch.ones(50))
        post.fit_blocks([("a", [0, 2], G), ("b", [1], 3.0 * G)])
        gen = torch.Generator().manual_seed(0)
        z = torch.zeros(1, 3)
        draws = torch.stack([post.forward_sample(z, generator=gen)[0] for _ in range(4000)])
    # Row 3 is in no block: it never moves.
    assert torch.all(draws[:, 3] == draws[0, 3])
    # At a zero feature vector the output is the bias draw; its variance is the
    # bias entry of the block covariance.
    for row, scale in [(0, 1.0), (1, 3.0), (2, 1.0)]:
        H = scale * G + 2.0 * torch.eye(4, dtype=torch.float64)
        var = float(torch.linalg.inv(H)[-1, -1])
        assert math.isclose(float(draws[:, row].double().var()), var, rel_tol=0.12)


def test_block_state_round_trips_draw_for_draw():
    post = _posterior(d_out=3)
    with torch.no_grad():
        feats = post.body(torch.randn(40, 5))
        G = accumulate_ggn(None, feats, torch.ones(40))
        post.fit_blocks([("a", [0, 1], G)])
    clone = _posterior(d_out=3)
    clone.load_state_dict(post.state_dict())
    clone.load_block_state(post.block_state())
    x = torch.randn(6, 5)
    a = post.forward_sample(x, generator=torch.Generator().manual_seed(5))
    b = clone.forward_sample(x, generator=torch.Generator().manual_seed(5))
    assert torch.equal(a, b)


def test_overlapping_blocks_are_refused():
    post = _posterior(d_out=3)
    G = torch.eye(8, dtype=torch.float64)
    with pytest.raises(ValueError, match="reuses"):
        post.fit_blocks([("a", [0, 1], G), ("b", [1, 2], G)])
