"""Hand-rolled HMC with dual-averaging step-size adaptation.

Fixed-length leapfrog rather than NUTS: this matches what published BNN
reference benchmarks actually run at this scale, and it is ~200 lines with no
new dependency. The Metropolis correction makes the chain exact for any step
size; adaptation only tunes efficiency.
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
        delta = h0 - trial(eps)
        if not math.isfinite(delta):
            delta = -math.inf
        if a * delta > a * math.log(0.5):
            break
        eps *= 2.0 ** a
        if eps < 1e-10 or eps > 1e10:
            break
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

    for t in range(1, total + 1):
        eps = math.exp(log_eps if t <= n_warmup else log_eps_bar)
        momentum = torch.randn(dim, generator=gen, device=theta.device, dtype=theta.dtype)

        current_h = -log_prob(theta) + 0.5 * (momentum ** 2).sum()
        new_theta, new_mom = leapfrog(theta, momentum, grad_log_prob, eps, n_leapfrog)
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

    return HMCResult(
        samples=torch.stack(kept, dim=0) if kept else torch.empty(0, dim),
        step_size=math.exp(log_eps_bar),
        accept_rate=accepts / max(int(n_samples), 1),
        divergences=divergences,
        warmup_divergences=warmup_divergences,
    )
