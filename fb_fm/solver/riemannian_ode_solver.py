# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

import math
from contextlib import nullcontext
from typing import Callable, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor

from fb_fm.solver.solver import Solver
from fb_fm.utils import ModelWrapper
from fb_fm.utils.manifolds import Manifold, geodesic

try:
    from tqdm import tqdm
    _TQDM = True
except Exception:
    _TQDM = False


# -------------------- Low-level helpers --------------------

def _apply_velocity(
    manifold: Manifold,
    velocity_model: Callable[[Tensor, Tensor], Tensor],
    x: Tensor,
    t: Tensor,
    proju: bool,
) -> Tensor:
    u = velocity_model(x, t)
    return manifold.proju(x, u) if proju else u


def _project_x(manifold: Manifold, x: Tensor, projx: bool) -> Tensor:
    return manifold.projx(x) if projx else x


def _euler_step(
    manifold: Manifold,
    velocity_model: Callable[[Tensor, Tensor], Tensor],
    x: Tensor,
    t0: Tensor,
    dt: Tensor,
    *,
    projx: bool,
    proju: bool,
) -> Tensor:
    v = _apply_velocity(manifold, velocity_model, x, t0, proju)
    x = x + dt * v
    return _project_x(manifold, x, projx)


def _midpoint_step(
    manifold: Manifold,
    velocity_model: Callable[[Tensor, Tensor], Tensor],
    x: Tensor,
    t0: Tensor,
    dt: Tensor,
    *,
    projx: bool,
    proju: bool,
) -> Tensor:
    v0 = _apply_velocity(manifold, velocity_model, x, t0, proju)
    x_mid = _project_x(manifold, x + 0.5 * dt * v0, projx)
    v_mid = _apply_velocity(manifold, velocity_model, x_mid, t0 + 0.5 * dt, proju)
    x = x + dt * v_mid
    return _project_x(manifold, x, projx)


def _rk4_step(
    manifold: Manifold,
    velocity_model: Callable[[Tensor, Tensor], Tensor],
    x: Tensor,
    t0: Tensor,
    dt: Tensor,
    *,
    projx: bool,
    proju: bool,
) -> Tensor:
    k1 = _apply_velocity(manifold, velocity_model, x, t0, proju)
    k2 = _apply_velocity(manifold, velocity_model, _project_x(manifold, x + dt * k1 / 3, projx), t0 + dt / 3, proju)
    k3 = _apply_velocity(manifold, velocity_model, _project_x(manifold, x + dt * (k2 - k1 / 3), projx), t0 + 2 * dt / 3, proju)
    k4 = _apply_velocity(manifold, velocity_model, _project_x(manifold, x + dt * (k1 - k2 + k3), projx), t0 + dt, proju)
    x = x + (k1 + 3 * (k2 + k3) + k4) * dt * 0.125  # 1/8
    return _project_x(manifold, x, projx)


_STEP_FNS = {"euler": _euler_step, "midpoint": _midpoint_step, "rk4": _rk4_step}


def _geodesic_interp(
    manifold: Manifold,
    x0: Tensor,
    x1: Tensor,
    t0: Tensor,
    t1: Tensor,
    t_req: Tensor,
) -> Tensor:
    """Interpolate along the manifold geodesic between x0@t0 and x1@t1 at scalar time t_req."""
    s = (t_req - t0) / (t1 - t0)
    gamma = geodesic(manifold, x0, x1)
    return gamma(s).reshape_as(x0)


# -------------------- Public integrator --------------------

def riemannian_odeint(
    *,
    manifold: Manifold,
    velocity_func: Callable[[Tensor, Tensor], Tensor],
    x_init: Tensor,
    time_grid: Tensor,
    step_size: Optional[float],
    method: str = "euler",
    projx: bool = True,
    proju: bool = True,
    return_intermediates: bool = False,
    verbose: bool = False,
    enable_grad: bool = False,
    accumulator_func: Optional[Callable[[Tensor, Tensor], Tensor]] = None,
) -> Tuple[Union[Tensor, Sequence[Tensor]], Optional[Tensor]]:
    """
    Integrate x' = u(x,t) on a manifold. Respects `time_grid` order (ascending or descending).
    If `step_size` is None: step across consecutive `time_grid` knots.
    Otherwise: use a uniform discretization from time_grid[0] → time_grid[-1] (inclusive).

    If `accumulator_func(x,t)` is provided, returns the left-Riemann time integral ∑ f(x_t,t)*Δt.
    """
    assert method in _STEP_FNS, f"Unknown method '{method}'."
    step_fn = _STEP_FNS[method]

    device = x_init.device
    time_grid = time_grid.to(device=device, dtype=torch.get_default_dtype())
    assert time_grid.ndim == 1 and time_grid.numel() >= 2, "time_grid must be 1D with ≥2 points."

    t0 = time_grid[0]
    tT = time_grid[-1]

    # Build discretization
    if step_size is None:
        t_disc = time_grid
    else:
        span = torch.abs(tT - t0).item()
        assert span > 0.0, "time_grid endpoints must differ when using step_size."
        n_steps = max(1, math.ceil(span / float(step_size)))
        t_disc = torch.linspace(t0.item(), tT.item(), n_steps + 1, device=device, dtype=time_grid.dtype)

    # Accumulator
    acc = None
    if accumulator_func is not None:
        acc = torch.zeros(x_init.shape[0], device=device, dtype=x_init.dtype)

    want_inter = return_intermediates
    if want_inter:
        outs: list[Tensor] = []

    # Progress context
    if verbose:
        if not _TQDM:
            raise ImportError("tqdm is not installed but verbose=True was requested.")
        total = abs((tT - t0).item())
        pbar = tqdm(total=total if total > 0 else None, desc="NFE: 0")
        ctx = pbar
    else:
        ctx = nullcontext()

    x = x_init
    nfe = 0

    # Pointer for requested times when using uniform steps
    i_ret = 0 if (want_inter and step_size is not None) else None

    # If using manual grid and wanting intermediates, push the initial
    if want_inter and step_size is None:
        outs.append(x.clone())

    # Trajectory grads only if requested; divergence handles its own grad context
    with ctx, torch.set_grad_enabled(enable_grad):
        for a, b in zip(t_disc[:-1], t_disc[1:]):
            dt = b - a  # signed

            if accumulator_func is not None:
                acc = acc + accumulator_func(x, a) * dt

            x_next = step_fn(manifold, velocity_func, x, a, dt, projx=projx, proju=proju)
            nfe += 1

            if want_inter:
                if step_size is None:
                    # manual grid: exact next knot
                    outs.append(x_next.clone())
                else:
                    # uniform steps: append exactly once per requested time (handles both directions)
                    low, high = (a, b) if (a <= b) else (b, a)
                    while i_ret is not None and i_ret < len(time_grid) and (low <= time_grid[i_ret] <= high):
                        t_req = time_grid[i_ret]
                        if torch.isclose(t_req, a):
                            outs.append(x.clone())
                        elif torch.isclose(t_req, b):
                            outs.append(x_next.clone())
                        else:
                            outs.append(_geodesic_interp(manifold, x, x_next, a, b, t_req))
                        i_ret += 1

            x = x_next

            if verbose and _TQDM:
                ctx.n = abs((a - t0).item())
                ctx.set_description(f"NFE: {nfe}")
                ctx.refresh()

    sol = torch.stack(outs, dim=0) if want_inter else x
    return sol, acc


# -------------------- Solver class --------------------

class RiemannianODESolver(Solver):
    """Riemannian ODE solver wrapping a velocity model and a manifold."""

    def __init__(self, manifold: Manifold, velocity_model: ModelWrapper):
        super().__init__()
        self.manifold = manifold
        self.velocity_model = velocity_model

    def sample(
        self,
        x_init: Tensor,
        step_size: Optional[float],
        projx: bool = True,
        proju: bool = True,
        method: str = "euler",
        time_grid: Optional[Tensor] = None,
        return_intermediates: bool = False,
        verbose: bool = False,
        enable_grad: bool = False,
        **model_extras,
    ) -> Union[Tensor, Sequence[Tensor]]:
        """
        Solve forward from min(time_grid) → max(time_grid). If not provided, uses [0, 1].
        Returns final x unless `return_intermediates=True` (then returns states at `time_grid`).
        """
        if time_grid is None:
            time_grid = torch.tensor([0.0, 1.0], device=x_init.device, dtype=torch.get_default_dtype())
        else:
            time_grid = time_grid.to(device=x_init.device, dtype=torch.get_default_dtype())
        # forward semantics: increasing grid
        time_grid = torch.sort(time_grid).values

        def velocity_func(x: Tensor, t: Tensor) -> Tensor:
            return self.velocity_model(x=x, t=t, **model_extras)

        sol, _ = riemannian_odeint(
            manifold=self.manifold,
            velocity_func=velocity_func,
            x_init=x_init,
            time_grid=time_grid,
            step_size=step_size,
            method=method,
            projx=projx,
            proju=proju,
            return_intermediates=return_intermediates,
            verbose=verbose,
            enable_grad=enable_grad,
            accumulator_func=None,
        )
        return sol

    def compute_likelihood(
        self,
        x_1: Tensor,
        log_p0: Callable[[Tensor], Tensor],
        step_size: Optional[float],
        method: str = "euler",
        atol: float = 1e-5,            # accepted for API parity; unused in fixed-step integrator
        rtol: float = 1e-5,            # accepted for API parity; unused in fixed-step integrator
        time_grid: Optional[Tensor] = None,
        return_intermediates: bool = False,
        *,
        exact_divergence: bool = False,
        projx: bool = True,
        proju: bool = True,
        verbose: bool = False,
        enable_grad: bool = False,     # keep ∂x/∂θ (path) gradients when True; eval-friendly when False
        **model_extras,
    ) -> Union[Tuple[Tensor, Tensor], Tuple[Sequence[Tensor], Tensor]]:
        """
        Compute log p(x_1) by integrating from t=1 → t=0:
            log p(x_1) = log p0(x_0) + ∫_{1}^{0} div u(x_t, t) dt
        Uses exact trace (costly) or Hutchinson estimator (default).
        """
        device = x_1.device
        if time_grid is None:
            time_grid = torch.tensor([1.0, 0.0], device=device, dtype=torch.get_default_dtype())
        else:
            time_grid = time_grid.to(device=device, dtype=torch.get_default_dtype())
        assert time_grid[0] > time_grid[-1], "Likelihood expects a reverse-time grid starting at 1.0 and ending at 0.0."

        # One Hutchinson probe fixed across time (variance reduction)
        if not exact_divergence:
            z = torch.empty_like(x_1).bernoulli_(0.5).mul_(2.0).sub_(1.0)  # Rademacher ±1
        else:
            z = None

        def velocity_func(x: Tensor, t: Tensor) -> Tensor:
            return self.velocity_model(x=x, t=t, **model_extras)

        def divergence_func(x: Tensor, t: Tensor) -> Tensor:
            """
            div u(x,t) w.r.t. ambient coordinates.
            - Path-gradient policy is governed by `enable_grad` at the integrator level.
            - We *always* enable grads locally here to compute the inner derivative,
              then optionally allow those higher-order grads to flow depending on `enable_grad`.
            """
            # Cut or keep path dependence:
            x_div = x if enable_grad else x.detach()

            with torch.set_grad_enabled(True):
                x_div = x_div.requires_grad_(True)
                u = velocity_func(x_div, t)
                if proju:
                    u = self.manifold.proju(x_div, u)

                u_flat = u.reshape(u.shape[0], -1)

                if exact_divergence:
                    # Exact: sum_j ∂u_j/∂x_j (O(D) VJPs)
                    div = torch.zeros(x_div.shape[0], device=x_div.device, dtype=x_div.dtype)
                    for j in range(u_flat.shape[1]):
                        g = torch.autograd.grad(
                            u_flat[:, j].sum(), x_div,
                            create_graph=enable_grad,  # allow higher-order grads only if we keep path grads
                            retain_graph=enable_grad,
                            allow_unused=False,
                        )[0]
                        div = div + g.reshape(g.shape[0], -1)[:, j]
                else:
                    # Hutchinson: z^T J_u(x) z
                    z_flat = z.reshape(z.shape[0], -1)
                    u_dot_z = (u_flat * z_flat).sum(dim=1)  # (B,)
                    grad_u_dot_z = torch.autograd.grad(
                        u_dot_z.sum(), x_div,
                        create_graph=enable_grad,
                        retain_graph=enable_grad,
                        allow_unused=False,
                    )[0]
                    div = (grad_u_dot_z.reshape(grad_u_dot_z.shape[0], -1) * z_flat).sum(dim=1)

            return div  # (B,)

        # Integrate 1 → 0; dt carries the sign
        sol, log_det = riemannian_odeint(
            manifold=self.manifold,
            velocity_func=velocity_func,
            x_init=x_1,
            time_grid=time_grid,
            step_size=step_size,
            method=method,
            projx=projx,
            proju=proju,
            return_intermediates=return_intermediates,
            verbose=verbose,
            enable_grad=enable_grad,      # trajectory/path grads toggle
            accumulator_func=divergence_func,
        )

        x0 = sol[-1] if return_intermediates else sol
        log_p0_x0 = log_p0(x0)
        log_p_x1 = log_p0_x0 + log_det

        if return_intermediates:
            return sol, log_p_x1
        else:
            return x0, log_p_x1
