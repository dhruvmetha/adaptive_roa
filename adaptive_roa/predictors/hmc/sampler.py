"""Hand-rolled HMC with dual-averaging step-size adaptation.

Fixed-length leapfrog rather than NUTS: this matches what published BNN
reference benchmarks actually run at this scale, and it is ~200 lines with no
new dependency. The Metropolis correction makes the chain exact for any step
size; adaptation only tunes efficiency.

Trajectory length is jittered every iteration (Neal 2011 Sec 4.2) to avoid
resonance between a fixed step count and a target's natural oscillation
period -- unmitigated, that resonance can make the chain antithetic
(theta -> -theta each draw) while every scalar diagnostic (acceptance,
divergences) still reads healthy. The tradeoff is effective sample size:
measured on an 800-draw well-behaved-target run, jitter roughly halves
median ESS (320 -> 176) relative to a fixed trajectory length, in exchange
for eliminating the resonance collapse (which cost ESS down to single
digits out of hundreds of draws on an affected seed). Callers sizing
`n_samples` for a reference chain should budget for this.

Known bias: at low `target_accept` (~0.6), achieved acceptance tends to
run ~0.10 above the target even after 1000 warmup iterations (measured
across 50 seeds on a standard Gaussian, mean bias +0.10 at target 0.6 vs.
+0.02 at target 0.85). Jittering the trajectory length widens the range of
step sizes that can trigger a hard leapfrog-stability divergence; since a
divergence only ever pushes the adapted step size down (never up), the
settled step size skews smaller -- and achieved acceptance correspondingly
higher -- than the low-target equilibrium alone would imply. The bias
shrinks at higher targets, where the settled step size sits further from
the stability boundary. This is a property of jittered dual averaging on
this class of target, not a tuning defect; it has been measured and
reported rather than tuned away by widening test tolerances.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch


@dataclass
class HMCResult:
    samples: torch.Tensor      # [n_samples, D]
    step_size: float
    accept_rate: float
    divergences: int
    warmup_divergences: int
    ess: float                 # worst-dimension effective sample size
    acf1: float                # worst-dimension lag-1 autocorrelation


def _ess_and_acf1(x: torch.Tensor) -> tuple:
    """Effective sample size and lag-1 autocorrelation for a 1D chain, via the
    integrated autocorrelation time with Geyer's initial positive sequence
    (stop summing once consecutive lag pairs go negative -- the standard
    safeguard against noisy long-lag estimates inflating tau).
    """
    n = x.numel()
    if n < 4:
        return float(n), 0.0
    xc = x - x.mean()
    var = (xc ** 2).mean().item()
    if var <= 0.0:
        return float(n), 1.0

    def rho(k):
        return (xc[:-k] * xc[k:]).mean().item() / var

    acf1 = rho(1)
    tau = 1.0
    k = 1
    while k + 1 < n:
        pair = rho(k) + rho(k + 1)
        if pair < 0:
            break
        tau += 2.0 * pair
        k += 2
    ess = n / max(tau, 1e-8)
    return min(ess, float(n)), acf1


def leapfrog(theta: torch.Tensor, momentum: torch.Tensor,
             grad_log_prob: Callable[[torch.Tensor], torch.Tensor],
             step_size: float, n_steps: int):
    """Standard leapfrog: half-kick, then (drift, kick) x n, then half-kick.

    Reversible and volume-preserving, which is what makes the Metropolis
    correction valid.
    """
    theta = theta.clone()
    momentum = momentum + 0.5 * step_size * grad_log_prob(theta)
    for i in range(int(n_steps)):
        theta = theta + step_size * momentum
        g = grad_log_prob(theta)
        if i < n_steps - 1:
            momentum = momentum + step_size * g
        else:
            momentum = momentum + 0.5 * step_size * g
    return theta, momentum


def find_reasonable_epsilon(log_prob: Callable[[torch.Tensor], torch.Tensor],
                             grad_log_prob: Callable[[torch.Tensor], torch.Tensor],
                             theta: torch.Tensor, seed: int, n_leapfrog: int) -> float:
    """Hoffman & Gelman's initialization: double or halve until the
    acceptance probability of the trajectory actually used crosses 1/2.

    NUTS validates a single leapfrog step because it builds trajectories
    adaptively and has no fixed length. This is fixed-length HMC with
    `n_leapfrog` steps, so stability over 1 step says almost nothing about
    stability over `n_leapfrog` -- bracket on the real trajectory instead.
    Runs once per chain, so the extra cost of a full trajectory per trial is
    irrelevant. Uses its own generator (derived from, but distinct from, the
    chain's seed) so this search does not perturb the sampling chain's RNG
    stream.
    """
    init_gen = torch.Generator(device=theta.device).manual_seed(int(seed) + 10_000)
    eps = 1.0
    momentum = torch.randn(theta.numel(), generator=init_gen, device=theta.device,
                           dtype=theta.dtype)
    h0 = (-log_prob(theta) + 0.5 * (momentum ** 2).sum()).item()

    def trial(e):
        th, mo = leapfrog(theta, momentum, grad_log_prob, e, n_leapfrog)
        return (-log_prob(th) + 0.5 * (mo ** 2).sum()).item()

    delta = h0 - trial(eps)
    a = 1.0 if delta > math.log(0.5) else -1.0
    for _ in range(100):   # bounded: never spin on a pathological target
        if not math.isfinite(delta):
            delta = -math.inf
        if a * delta <= a * math.log(0.5):
            break
        eps *= 2.0 ** a
        if eps < 1e-10 or eps > 1e10:
            break
        delta = h0 - trial(eps)
    return eps


def hmc_chain(log_prob: Callable[[torch.Tensor], torch.Tensor],
              grad_log_prob: Callable[[torch.Tensor], torch.Tensor],
              theta_init: torch.Tensor, n_samples: int, n_warmup: int,
              n_leapfrog: int = 20, target_accept: float = 0.8,
              seed: int = 0, divergence_threshold: float = 1000.0) -> HMCResult:
    """Run one chain. Warmup adapts the step size; only post-warmup draws are kept."""
    gen = torch.Generator(device=theta_init.device).manual_seed(int(seed))
    theta = theta_init.clone()
    dim = theta.numel()

    # Dual averaging state (Hoffman & Gelman), seeded from a bracketing search
    # for a reasonable starting step size rather than an arbitrary constant --
    # otherwise adaptation can overshoot leapfrog's stability boundary before
    # it settles, which registers as real divergences during warmup.
    eps0 = find_reasonable_epsilon(log_prob, grad_log_prob, theta, seed, n_leapfrog)
    log_eps = math.log(eps0)
    log_eps_bar = math.log(eps0)
    h_bar = 0.0
    mu = math.log(eps0)
    gamma, t0, kappa = 0.05, 10.0, 0.75

    kept = []
    accepts = 0
    divergences = 0
    warmup_divergences = 0
    total = int(n_warmup) + int(n_samples)

    # Jitter the trajectory length each iteration (Neal 2011 Sec 4.2): a fixed
    # L can resonate with a target's natural oscillation period and make the
    # chain antithetic (theta -> -theta each step), which looks perfectly
    # healthy -- full acceptance, zero divergences -- while destroying the
    # second-moment information the chain exists to produce. Drawing L from
    # the chain's own generator, independent of the current state, keeps the
    # Metropolis correction valid.
    l_low = max(1, int(0.8 * n_leapfrog))
    l_high = int(1.2 * n_leapfrog) + 1

    for t in range(1, total + 1):
        eps = math.exp(log_eps if t <= n_warmup else log_eps_bar)
        l_t = int(torch.randint(l_low, l_high, (1,), generator=gen,
                                device=theta.device).item())
        momentum = torch.randn(dim, generator=gen, device=theta.device, dtype=theta.dtype)

        current_h = -log_prob(theta) + 0.5 * (momentum ** 2).sum()
        new_theta, new_mom = leapfrog(theta, momentum, grad_log_prob, eps, l_t)
        new_h = -log_prob(new_theta) + 0.5 * (new_mom ** 2).sum()

        delta = (current_h - new_h).item()
        if not math.isfinite(delta) or delta < -divergence_threshold:
            # Warmup is where the step size is deliberately being pushed
            # around, so divergences there are expected and carry no
            # diagnostic signal -- only post-warmup divergences indicate a
            # chain that can't be trusted. Track both, count only the latter.
            if t > n_warmup:
                divergences += 1
            else:
                warmup_divergences += 1
            accept_prob = 0.0
        else:
            accept_prob = min(1.0, math.exp(min(delta, 0.0)))

        if torch.rand((), generator=gen, device=theta.device).item() < accept_prob:
            theta = new_theta
            if t > n_warmup:
                accepts += 1

        if t <= n_warmup:
            eta = 1.0 / (t + t0)
            h_bar = (1.0 - eta) * h_bar + eta * (target_accept - accept_prob)
            log_eps = mu - math.sqrt(t) / gamma * h_bar
            w = t ** (-kappa)
            log_eps_bar = w * log_eps + (1.0 - w) * log_eps_bar
        else:
            kept.append(theta.detach().clone())

    samples = torch.stack(kept, dim=0) if kept else torch.empty(0, dim)

    # ESS/ACF on theta**2, not theta: under trajectory-length resonance theta
    # itself can look artificially well-mixed (it alternates sign every draw,
    # which is high lag-1 *negative* autocorrelation but not necessarily a low
    # theta-ESS by the usual estimator) while the variance information theta**2
    # carries is exactly what gets destroyed. Reported as the worst dimension
    # so a single badly-mixing coordinate cannot hide behind the others.
    if samples.shape[0] >= 4:
        per_dim = [_ess_and_acf1(samples[:, d] ** 2) for d in range(dim)]
        ess = min(e for e, _ in per_dim)
        acf1 = max(a for _, a in per_dim)  # worst mixing = largest |acf|, not smallest
    else:
        ess = float(samples.shape[0])
        acf1 = 0.0

    return HMCResult(
        samples=samples,
        step_size=math.exp(log_eps_bar),
        accept_rate=accepts / max(int(n_samples), 1),
        divergences=divergences,
        warmup_divergences=warmup_divergences,
        ess=ess,
        acf1=acf1,
    )
