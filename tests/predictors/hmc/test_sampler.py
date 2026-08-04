import torch

from adaptive_roa.predictors.hmc.sampler import hmc_chain, leapfrog


def _gaussian_target(sigma=1.0):
    """Standard target with a known answer: N(0, sigma^2 I)."""
    def log_prob(theta):
        return (-0.5 * (theta / sigma) ** 2).sum()

    def grad_log_prob(theta):
        return -theta / (sigma ** 2)

    return log_prob, grad_log_prob


def test_leapfrog_is_reversible():
    """Flip the momentum, run the same number of steps, and you return to the
    start. Reversibility is what makes the MH correction valid; without it the
    chain silently samples the wrong distribution."""
    _lp, glp = _gaussian_target()
    torch.manual_seed(0)
    theta0, mom0 = torch.randn(5), torch.randn(5)

    theta1, mom1 = leapfrog(theta0, mom0, glp, step_size=0.1, n_steps=12)
    theta2, mom2 = leapfrog(theta1, -mom1, glp, step_size=0.1, n_steps=12)

    torch.testing.assert_close(theta2, theta0, atol=1e-5, rtol=0)
    torch.testing.assert_close(-mom2, mom0, atol=1e-5, rtol=0)


def test_leapfrog_conserves_energy_to_second_order():
    """Energy error must shrink like step_size^2 for a smooth target. A first-order
    scheme (a wrong half-step) would show a linear trend instead."""
    lp, glp = _gaussian_target()
    torch.manual_seed(0)
    theta0, mom0 = torch.randn(8), torch.randn(8)

    def energy(th, mo):
        return (-lp(th) + 0.5 * (mo ** 2).sum()).item()

    e0 = energy(theta0, mom0)
    errs = []
    for eps in (0.2, 0.1, 0.05):
        th, mo = leapfrog(theta0, mom0, glp, step_size=eps, n_steps=int(1.0 / eps))
        errs.append(abs(energy(th, mo) - e0))
    # Halving the step size should cut the error by roughly 4x.
    assert errs[1] < errs[0] / 3.0
    assert errs[2] < errs[1] / 3.0


def test_chain_recovers_a_gaussian_target():
    lp, glp = _gaussian_target(sigma=2.0)
    res = hmc_chain(lp, glp, torch.zeros(3), n_samples=800, n_warmup=400,
                    n_leapfrog=20, seed=0)
    assert res.samples.shape == (800, 3)
    assert res.samples.mean(0).abs().max().item() < 0.35
    assert abs(res.samples.std(0).mean().item() - 2.0) < 0.35


def test_step_size_adapts_toward_the_target_acceptance():
    """Dual averaging should track the requested acceptance level itself, not
    just its direction: a stricter target must yield a higher achieved
    acceptance and a smaller step size, AND the achieved rate must land near
    the requested target for each of two distinct target_accept values --
    ordering alone would also be satisfied by a mechanism that moves the
    right way but overshoots or undershoots by a large, uncontrolled amount.

    Targets are 0.60/0.85 rather than the ceiling-adjacent 0.95: since
    accept_rate <= 1.0 always, a band around 0.95 is one-sided in practice
    (a chain saturating at 1.000 would pass regardless of how well adaptation
    is actually tracking). n_warmup=1000 (not the sampler's usual few hundred)
    because jitter adds noise to the dual-averaging signal; even so, the low
    target carries a measured positive bias -- see the module docstring.
    """
    lp, glp = _gaussian_target()
    kw = dict(theta_init=torch.zeros(4), n_samples=400, n_warmup=1000,
              n_leapfrog=15, seed=0)
    low = hmc_chain(lp, glp, target_accept=0.60, **kw)
    high = hmc_chain(lp, glp, target_accept=0.85, **kw)

    assert high.accept_rate >= low.accept_rate
    assert high.step_size < low.step_size
    assert low.divergences == 0 and high.divergences == 0
    assert abs(low.accept_rate - 0.60) < 0.15
    assert abs(high.accept_rate - 0.85) < 0.15


def test_different_seeds_give_different_chains_and_same_seed_reproduces():
    lp, glp = _gaussian_target()
    kw = dict(theta_init=torch.zeros(3), n_samples=100, n_warmup=100, n_leapfrog=10)
    a = hmc_chain(lp, glp, seed=0, **kw).samples
    b = hmc_chain(lp, glp, seed=0, **kw).samples
    c = hmc_chain(lp, glp, seed=1, **kw).samples
    torch.testing.assert_close(a, b)
    assert not torch.allclose(a, c)


def test_divergences_are_counted_not_silently_accepted():
    """A cliff target must register divergences somewhere rather than quietly
    producing garbage. They land in warmup here, which is exactly why warmup
    divergences are recorded separately instead of being discarded."""
    def log_prob(theta):
        return torch.where(theta.abs().max() < 1.0,
                           -0.5 * (theta ** 2).sum(), torch.tensor(-1e6))

    def grad_log_prob(theta):
        return -theta * 1e3

    res = hmc_chain(log_prob, grad_log_prob, torch.zeros(2), n_samples=50,
                    n_warmup=50, n_leapfrog=20, seed=0)
    assert res.warmup_divergences + res.divergences > 0
    assert torch.isfinite(res.samples).all()


def test_persistent_divergences_on_a_pathological_target_are_counted_post_warmup():
    """Nothing else in this suite asserts divergences > 0 on the post-warmup
    counter -- it could be hardcoded to 0 and every other test would stay
    green. Neal's funnel has curvature that varies by orders of magnitude
    across its neck, so no single global step size (which is all fixed-length
    HMC ever adapts) is stable everywhere; even a well-adapted chain
    occasionally diverges when it revisits the neck."""
    def log_prob(theta):
        v = theta[0]
        x = theta[1:]
        return -0.5 * (v / 3.0) ** 2 - 0.5 * v - 0.5 * (x ** 2 * torch.exp(-v)).sum()

    def grad_log_prob(theta):
        v = theta[0]
        x = theta[1:]
        dv = -v / 9.0 - 0.5 + 0.5 * (x ** 2 * torch.exp(-v)).sum()
        dx = -x * torch.exp(-v)
        return torch.cat([dv.reshape(1), dx])

    res = hmc_chain(log_prob, grad_log_prob, torch.zeros(3), n_samples=800,
                    n_warmup=200, n_leapfrog=20, seed=0)
    assert res.divergences > 0


def test_a_well_behaved_target_produces_no_divergences_after_warmup():
    """Post-warmup divergences are the diagnostic that matters: once the step
    size has settled, a well-conditioned target must not diverge at all.

    Warmup divergences are NOT asserted to be zero. A quadratic target's
    stability boundary is a hard cliff, and dual averaging deliberately probes
    across it while adapting -- which is precisely why the two counters are
    reported separately rather than summed.
    """
    lp, glp = _gaussian_target()
    res = hmc_chain(lp, glp, torch.zeros(3), n_samples=200, n_warmup=200,
                    n_leapfrog=15, seed=0)
    assert res.divergences == 0
    # Adaptation must still converge: the settled chain recovers the target.
    assert res.samples.mean(0).abs().max().item() < 0.4
    assert abs(res.samples.std(0).mean().item() - 1.0) < 0.4
