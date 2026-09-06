"""Profile p_success query latency for every predictor family, on every system.

Scope: the six distinct inference methods behind the arms documented in
``docs/stochastic`` (see its ``METHODS.md``). Acquisition variants share a
predictor and therefore share an inference cost, so ``epi_var``, ``yield_a1``,
``clf_epi_bald`` and friends collapse onto the predictor they run:

    fm_ensemble    M flow-matching members x K MC endpoints x ODE steps
    clf_ensemble   M MLP forward passes, per-member values exposed
    bnn_ensemble   the same M MLPs, marginal only
    bnn_mfvi       S weight draws through one MLP
    bnn_laplace    S weight draws against the GGN precision factor
    gp             sparse variational GP, 128 inducing points

Two numbers are reported per cell, because they answer different questions and
differ by orders of magnitude:

  * batch-1 latency, over ``--n-queries`` separately timed calls. What a caller
    that asks about one state at a time pays. On GPU this carries kernel-launch
    overhead, which compresses cheap methods together.
  * batched cost, amortized per state over a single call on ``--batch-size``
    states. What the eval grid and acquisition scoring actually pay.

Every timed call goes through the run's OWN probability backend, instantiated
from its recorded hydra config and bound to weights loaded off disk. That is
the same ``estimate()`` the pipeline calls, not a reimplementation of it, so a
number here cannot drift from what the campaigns ran. Nothing is trained.

The FM caching path is deliberately bypassed: ``estimate()`` recomputes, which
is the cost a fresh query pays.

Usage:
    python scripts/profile_p_success.py                    # full run
    python scripts/profile_p_success.py --smoke             # 5 queries, pendulum
    python scripts/profile_p_success.py --systems pendulum cartpole
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import platform
import re
import sys
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np
import torch

from adaptive_roa.probabilistic_classifier.export import load_cfg, resolve_system, resolve_predictor_name
from adaptive_roa.probabilistic_classifier import get_probabilistic_classifier_class
from adaptive_roa.adaptive_v2.probability.ensemble_prob import (
    EnsembleEndpointMCProbabilityBackend,
)

EXP_BASE = Path("/common/users/shared/pracsys/adaptive_roa_experiments")
DEFAULT_OUT = Path("/common/users/shared/pracsys/genMoPlan/docs/stochastic/profiling")

# system label -> run-dir prefix. One representative noise level per system,
# matching the level each system's README reports on.
SYSTEM_PREFIX = {
    "pendulum": EXP_BASE / "gaussian_torque" / "pen_high",
    "cartpole": EXP_BASE / "gaussian_torque" / "cp_high_v2",
    "quadrotor2d": EXP_BASE / "quadrotor_stoch" / "q2d_cs",
    "quadrotor3d": EXP_BASE / "quadrotor_stoch" / "q3d_cs030",
}

# run-dir suffix -> the arm whose predictor it carries. The predictor NAME is
# read from each run's own config rather than assumed here, so a mislabeled
# suffix cannot silently profile the wrong family.
ARM_SUFFIXES = ["dir00_s42", "clf_dir00", "bnn_ens", "bnn_mfvi", "bnn_lap", "partx_fix"]

# `predictor.name` -> the registry wrapper that knows its checkpoint layout.
# clf_ensemble is trained by BayesianMLPTrainer with posterior=ensemble, so it
# writes bnn_ensemble's layout; it is absent from the registry, and the family
# alias ("classifier") would resolve to the SINGLE-MLP wrapper and under-report
# it by a factor of M with no error raised. fm_ensemble is handled separately.
WRAPPER_ALIAS = {"clf_ensemble": "bnn_ensemble"}


# ── model loading ────────────────────────────────────────────────────────────

def epoch_dirs(run_dir: Path) -> list[Path]:
    dirs = [Path(d) for d in glob.glob(str(run_dir / "epoch_*")) if Path(d).is_dir()]
    return sorted(dirs, key=lambda p: int(re.search(r"epoch_(\d+)", p.name).group(1)))


def last_epoch(run_dir: Path) -> int:
    dirs = epoch_dirs(run_dir)
    if not dirs:
        raise FileNotFoundError(f"no epoch_* directories under {run_dir}")
    return int(re.search(r"epoch_(\d+)", dirs[-1].name).group(1))


def load_fm_ensemble_handle(run_dir: Path, epoch: int, cfg, system, system_name: str):
    """Assemble the M-member flow-matching handle the pipeline uses.

    Reuses ``EnsembleFlowMatchingTrainer._load_member`` rather than globbing
    checkpoints here, so the reconstruction cannot drift from the one the
    trainer performs at the end of every real epoch.
    """
    from omegaconf import OmegaConf

    from adaptive_roa.adaptive_v2.trainers.ensemble_flow_matching_trainer import (
        EnsembleFlowMatchingTrainer,
        EnsembleFlowMatcherHandle,
    )

    # The trainer navigates the config by attribute (cfg.predictor.ensemble),
    # which a plain dict from load_cfg does not support.
    trainer = EnsembleFlowMatchingTrainer(OmegaConf.create(dict(cfg)), system, system_name)
    epoch_dir = run_dir / f"epoch_{epoch:03d}"
    members = [trainer._load_member(str(epoch_dir), m) for m in range(trainer.n_members)]
    if len(members) != trainer.n_members:
        raise RuntimeError(
            f"loaded {len(members)} FM members, config says {trainer.n_members}"
        )
    return EnsembleFlowMatcherHandle(members)


def load_handle(run_dir: Path, epoch: int, cfg, system, system_name: str, device: str):
    """Return a bound-ready model handle for whatever predictor this run used."""
    name = resolve_predictor_name(cfg, str(run_dir))
    if name == "fm_ensemble":
        return name, load_fm_ensemble_handle(run_dir, epoch, cfg, system, system_name)

    wrapper_cls = get_probabilistic_classifier_class(WRAPPER_ALIAS.get(name, name))
    wrapper = wrapper_cls.load_from_run(str(run_dir), epoch, cfg, system, device=device)
    handle = getattr(wrapper, "handle", None)
    if handle is None:
        raise RuntimeError(f"{wrapper_cls.__name__} exposes no .handle for {name}")
    return name, handle


def build_backend(cfg, system, device: str, handle, mc_batched: bool = False):
    """Instantiate and bind the run's own probability backend.

    ``mc_batched`` swaps ONLY the endpoint-MC backend for the variant that
    issues its K draws in one call. Every other backend is untouched, and the
    swap is opt-in so the default profile measures the shipped code path.
    """
    import hydra
    from omegaconf import OmegaConf

    prob_cfg = cfg["probability"]
    if not isinstance(prob_cfg, (dict, list)):
        prob_cfg = OmegaConf.to_container(prob_cfg, resolve=True)
    node = OmegaConf.create(dict(prob_cfg))
    target = node.pop("_target_")
    cls = hydra.utils.get_class(target)
    if mc_batched and issubclass(cls, EnsembleEndpointMCProbabilityBackend):
        cls = MCBatchedEnsembleEndpointMC
    backend = cls(node, system, device)
    backend.bind_model(handle)
    return backend


class StepOverrideHandle:
    """Forces ``num_steps`` on every member endpoint call.

    The MC backends call ``predict_endpoint_member(m, x)`` with no kwargs, so
    the step count comes from the ``predict_endpoint`` signature default (100).
    Wrapping the handle injects the override without touching either backend,
    and works for the sequential and batched paths alike.

    Unlike MC batching, this CHANGES THE ANSWER: fewer Euler steps is a coarser
    discretization of the same ODE, so the endpoints move and p_success moves
    with them. Any timing taken here must be reported next to its accuracy cost.
    """

    def __init__(self, inner, num_steps: int):
        self._inner = inner
        self.num_steps_override = int(num_steps)

    def predict_endpoint_member(self, m: int, x, **kw):
        kw.setdefault("num_steps", self.num_steps_override)
        return self._inner.predict_endpoint_member(m, x, **kw)

    def predict_endpoint(self, x, **kw):
        kw.setdefault("num_steps", self.num_steps_override)
        return self._inner.predict_endpoint(x, **kw)

    def __getattr__(self, name):
        # Only reached for attributes the wrapper does not define, so the two
        # methods above always win. n_members, members, eval, to all delegate.
        return getattr(self._inner, name)


# ── MC-batched FM backend ────────────────────────────────────────────────────

class MCBatchedEnsembleEndpointMC(EnsembleEndpointMCProbabilityBackend):
    """Identical estimator to the shipped backend, with the K draws batched.

    The shipped ``estimate_members`` (ensemble_prob.py:84) loops K times per
    member, issuing K separate ``predict_endpoint_member`` calls:

        for m in range(M):
            for _ in range(K):
                pred = handle.predict_endpoint_member(m, x)

    so one query costs M*K = 100 SEQUENTIAL ODE solves of 100 steps each,
    i.e. 10,000 sequential network evaluations on a batch of one. The ODE steps
    are inherently sequential and stay that way. The K draws are not: they are
    independent, differing only in the noise and latent, which
    ``_prepare_model_inputs`` samples PER ROW (flow_matcher.py:664). Repeating
    the states K times and issuing one call therefore draws exactly the same
    distribution while collapsing K launches into one.

    This is a strict launch-count change, not an estimator change: same K, same
    radius, same per-member mean of Bernoulli hits.
    """

    #: rows per call. K*N stays in one launch below this; above it the call is
    #: split, which restores some sequentiality at very wide batches.
    max_rows_per_call: int = 65536

    @torch.no_grad()
    def estimate_members(self, start_states: np.ndarray, verbose: bool = False) -> np.ndarray:
        handle = self.model_handle
        if handle is None:
            raise RuntimeError("ensemble backend used before bind_model")
        x = torch.as_tensor(np.asarray(start_states), dtype=torch.float32,
                            device=self.device)
        n, k = x.shape[0], self.num_mc_samples
        rows_per_state = max(1, self.max_rows_per_call // max(k, 1))

        out = np.empty((self.n_members, n), dtype=np.float64)
        for m in range(self.n_members):
            hits = torch.zeros(n, dtype=torch.float64)
            for start in range(0, n, rows_per_state):
                chunk = x[start:start + rows_per_state]
                # repeat_interleave, NOT repeat: rows must group by state so the
                # view below reshapes to [chunk, K] with each row one state's K
                # draws. `repeat` would tile states and silently transpose the
                # grouping, mixing draws across states.
                rep = chunk.repeat_interleave(k, dim=0)
                pred = handle.predict_endpoint_member(m, rep)
                lab = self.system.classify_attractor(pred, self.attractor_radius)
                got = (lab == 1).view(chunk.shape[0], k).double().sum(dim=1).cpu()
                hits[start:start + chunk.shape[0]] = got
            out[m] = (hits / float(k)).numpy()
        return out


# ── cost drivers ─────────────────────────────────────────────────────────────

def _member_predict_endpoint(handle):
    """The bound ``predict_endpoint``, or None for a non-FM handle."""
    members = getattr(handle, "members", None)
    target = members[0] if members else handle
    return getattr(target, "predict_endpoint", None)


def _live_ode_steps(handle, fm_cfg) -> int | None:
    """Integration steps actually spent, read from the bound callable.

    NOT from ``predictor.flow_matching.num_integration_steps``. The MC backend
    calls ``predict_endpoint_member(m, x)`` with no kwargs
    (ensemble_prob.py:93), so what runs is the ``predict_endpoint`` signature
    default. The two happen to agree at 100 in every campaign config, which is
    exactly why reading the config would look correct right up until someone
    changed one of them.
    """
    import inspect

    override = getattr(handle, "num_steps_override", None)
    if override is not None:
        return int(override)
    fn = _member_predict_endpoint(handle)
    if fn is None:
        return None
    param = inspect.signature(fn).parameters.get("num_steps")
    if param is not None and param.default is not inspect.Parameter.empty:
        return int(param.default)
    return int(fm_cfg.get("num_integration_steps", 0)) or None


def _live_ode_method(handle) -> str | None:
    """Solver actually used; quadrotor3d differs (euler_riemannian)."""
    import inspect

    fn = _member_predict_endpoint(handle)
    if fn is None:
        return None
    param = inspect.signature(fn).parameters.get("method")
    if param is not None and param.default is not inspect.Parameter.empty:
        return str(param.default)
    return None


def cost_drivers(cfg, backend, handle) -> dict:
    """The knobs that set this cell's cost, read off the LIVE objects.

    A latency without them is unreadable: 5 members at K=20 and 1 member at
    K=100 are different methods that could post the same number.

    M and S are read from the bound posterior, not from the config, because
    the config lists both for every BNN arm while only one is spent. An
    EnsemblePosterior overrides ``predictive_logit_samples`` to enumerate its M
    members and IGNORES ``n_marginal_samples`` (posteriors.py:166), so
    reporting the config's S=64 for ``bnn_ensemble`` would name a cost it never
    pays and hide the M=5 it does.
    """
    pred = cfg.get("predictor", {})
    bnn = pred.get("bnn", {}) if not isinstance(pred, str) else {}
    fm = pred.get("flow_matching", {}) if not isinstance(pred, str) else {}

    posterior = getattr(handle, "posterior", None)
    members = getattr(posterior, "n_members", None) if posterior is not None else None
    if members is None:
        members = getattr(backend, "n_members", None) or None
    # Only continuous posteriors actually draw S samples; enumerating ones do not.
    enumerates = posterior is not None and hasattr(posterior, "n_members")
    n_marginal = None if enumerates else (int(getattr(handle, "n_marginal_samples", 0)) or None)

    # inducing_count is a method on GPClassifier, not a property. The GP grows
    # its inducing set with the training data, so read the live count rather
    # than the config's n_inducing.
    inducing = None
    gp = getattr(handle, "gp", None)
    counter = getattr(gp, "inducing_count", None)
    if callable(counter):
        inducing = int(counter())

    # Which branch of LastLayerLaplacePosterior._weight_noise is live. Runs
    # written before laplace_prec_chol.pt shipped (2026-08-28) have no
    # precision factor and fall back to a float64 eigendecomposition of Sigma
    # on EVERY draw, S times per query. That is a legacy path costing ~50x the
    # triangular solve, so a cell on it is not comparable to one that is not.
    code_path = None
    if posterior is not None and hasattr(posterior, "_prec_chol"):
        code_path = ("prec_chol" if posterior._prec_chol.numel()
                     else "cov_eigh_fallback")

    return {
        "backend": type(backend).__name__,
        "code_path": code_path,
        "mc_mode": ("batched" if isinstance(backend, MCBatchedEnsembleEndpointMC)
                    else "sequential" if isinstance(backend, EnsembleEndpointMCProbabilityBackend)
                    else None),
        "n_members": int(members) if members else None,
        "num_mc_samples": int(getattr(backend, "num_mc_samples", 0)) or None,
        "n_marginal_samples": n_marginal,
        "n_inducing": inducing,
        "num_integration_steps": _live_ode_steps(handle, fm),
        "ode_method": _live_ode_method(handle),
        "hidden_dims": str(list(bnn.get("hidden_dims", []))) or None,
        "attractor_radius": float(getattr(backend, "attractor_radius", float("nan"))),
    }


def check_member_count(name: str, cfg, backend, handle=None) -> None:
    """A 1-member 'ensemble' would post an M-times-fast number and no error."""
    pred = cfg.get("predictor", {})
    if isinstance(pred, str):
        return
    expected = None
    if name == "fm_ensemble":
        expected = int(pred.get("ensemble", {}).get("n_members", 0))
    elif name in ("clf_ensemble", "bnn_ensemble"):
        expected = int(pred.get("bnn", {}).get("n_members", 0))
    if not expected:
        return
    # ClassifierProbabilityBackend marginalizes inside the handle and exposes no
    # n_members of its own, so fall through to the bound posterior. Without this
    # the bnn_ensemble cell would go unchecked entirely.
    got = int(getattr(backend, "n_members", 0))
    if not got and handle is not None:
        got = int(getattr(getattr(handle, "posterior", None), "n_members", 0) or 0)
    if got and got != expected:
        raise RuntimeError(
            f"{name}: backend bound {got} members, config declares {expected}. "
            f"Timing a short ensemble would report a spuriously fast method."
        )


# ── timing ───────────────────────────────────────────────────────────────────

def sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


@dataclass
class Timing:
    n_calls: int = 0
    mean_ms: float = float("nan")
    std_ms: float = float("nan")
    median_ms: float = float("nan")
    p95_ms: float = float("nan")
    min_ms: float = float("nan")


def summarize(samples_s: np.ndarray, n_calls: int) -> Timing:
    ms = samples_s * 1e3
    return Timing(
        n_calls=n_calls,
        mean_ms=float(ms.mean()),
        std_ms=float(ms.std(ddof=1)) if ms.size > 1 else 0.0,
        median_ms=float(np.median(ms)),
        p95_ms=float(np.percentile(ms, 95)),
        min_ms=float(ms.min()),
    )


def time_single(backend, states: np.ndarray, n_calls: int, warmup: int, device: str) -> Timing:
    """Time ``n_calls`` separate batch-1 estimate() calls."""
    n = len(states)
    for i in range(warmup):
        backend.estimate(states[i % n][None, :])
    sync(device)

    out = np.empty(n_calls, dtype=np.float64)
    for i in range(n_calls):
        x = states[i % n][None, :]
        sync(device)
        t0 = time.perf_counter()
        backend.estimate(x)
        sync(device)
        out[i] = time.perf_counter() - t0
    return summarize(out, n_calls)


def time_batched(backend, states: np.ndarray, batch_size: int, repeats: int,
                 warmup: int, device: str) -> tuple[Timing, float, float]:
    """Time whole-batch calls; return (per-call timing, per-state us mean/std)."""
    batch = states[:batch_size]
    for _ in range(max(1, warmup // 10)):
        backend.estimate(batch)
    sync(device)

    out = np.empty(repeats, dtype=np.float64)
    for i in range(repeats):
        sync(device)
        t0 = time.perf_counter()
        backend.estimate(batch)
        sync(device)
        out[i] = time.perf_counter() - t0

    per_state_us = out / batch_size * 1e6
    std = float(per_state_us.std(ddof=1)) if per_state_us.size > 1 else 0.0
    return summarize(out, repeats), float(per_state_us.mean()), std


# ── query states ─────────────────────────────────────────────────────────────

def load_query_states(cfg, system, n_needed: int, seed: int = 0) -> np.ndarray:
    """Real in-distribution states from the run's own test set.

    Synthetic or degenerate states could take a different code path (invalid
    handling, early exits), so the profile uses states the model was actually
    evaluated on.

    Read here rather than through ``adaptive.data_source.load_eval_states``,
    which parses the ENDPOINT format (start, end, label -> 2*state_dim + 1
    columns). These stochastic test sets carry the continuous ground-truth
    field instead (state, p_success -> state_dim + 1), which that reader
    rejects outright on the quadrotors and mis-parses on pendulum without
    complaining. Both layouts put the query state in the leading columns, so
    both are accepted and the column count is checked rather than assumed.
    """
    path = str(cfg["data_source"]["test_set_file"])
    data = np.loadtxt(path, delimiter=",", ndmin=2)
    sd = int(system.state_dim)
    if data.shape[1] not in (sd + 1, 2 * sd + 1):
        raise ValueError(
            f"{path} has {data.shape[1]} columns; expected {sd + 1} "
            f"(state, p_success) or {2 * sd + 1} (start, end, label) for a "
            f"{sd}-D state. Slicing the leading {sd} columns anyway would "
            f"profile on states that are not states."
        )
    states = np.ascontiguousarray(data[:, :sd], dtype=np.float32)
    if len(states) < n_needed:
        rng = np.random.default_rng(seed)
        return np.ascontiguousarray(states[rng.integers(0, len(states), size=n_needed)])
    return states[:n_needed]


# ── one cell ─────────────────────────────────────────────────────────────────

@dataclass
class Row:
    system: str
    state_dim: int
    method: str
    run_dir: str
    epoch: int
    device_name: str
    single: Timing = field(default_factory=Timing)
    batch_size: int = 0
    batch_repeats: int = 0
    batch_call_ms_mean: float = float("nan")
    batch_call_ms_std: float = float("nan")
    per_state_us_mean: float = float("nan")
    per_state_us_std: float = float("nan")
    drivers: dict = field(default_factory=dict)
    error: str = ""


def profile_cell(system_label: str, run_dir: Path, args, device: str,
                 device_name: str) -> Row:
    cfg = load_cfg(str(run_dir))
    system = resolve_system(cfg)
    system_name = str(cfg.get("dataset_name", system_label))
    epoch = args.epoch if args.epoch is not None else last_epoch(run_dir)

    name, handle = load_handle(run_dir, epoch, cfg, system, system_name, device)
    if args.fm_ode_steps and name == "fm_ensemble":
        handle = StepOverrideHandle(handle, args.fm_ode_steps)
    if hasattr(handle, "eval"):
        handle.eval()
    if hasattr(handle, "to"):
        handle.to(device)

    backend = build_backend(cfg, system, device, handle, args.mc_batched)
    check_member_count(name, cfg, backend, handle)

    states = load_query_states(cfg, system, max(args.n_queries, args.batch_size))

    row = Row(
        system=system_label,
        state_dim=int(system.state_dim),
        method=name,
        run_dir=str(run_dir),
        epoch=epoch,
        device_name=device_name,
        batch_size=args.batch_size,
        batch_repeats=args.batch_repeats,
        drivers=cost_drivers(cfg, backend, handle),
    )

    row.single = time_single(backend, states, args.n_queries, args.warmup, device)
    call, us_mean, us_std = time_batched(
        backend, states, args.batch_size, args.batch_repeats, args.warmup, device
    )
    row.batch_call_ms_mean = call.mean_ms
    row.batch_call_ms_std = call.std_ms
    row.per_state_us_mean = us_mean
    row.per_state_us_std = us_std
    return row


# ── output ───────────────────────────────────────────────────────────────────

CSV_FIELDS = [
    "system", "state_dim", "method", "device_name", "epoch",
    "n_queries", "single_mean_ms", "single_std_ms", "single_median_ms",
    "single_p95_ms", "single_min_ms",
    "batch_size", "batch_repeats", "batch_call_ms_mean", "batch_call_ms_std",
    "per_state_us_mean", "per_state_us_std",
    "backend", "code_path", "mc_mode", "n_members", "num_mc_samples",
    "n_marginal_samples",
    "n_inducing",
    "num_integration_steps", "ode_method", "hidden_dims", "attractor_radius",
    "run_dir", "error",
]


def row_to_dict(r: Row) -> dict:
    d = {
        "system": r.system, "state_dim": r.state_dim, "method": r.method,
        "device_name": r.device_name, "epoch": r.epoch,
        "n_queries": r.single.n_calls,
        "single_mean_ms": r.single.mean_ms, "single_std_ms": r.single.std_ms,
        "single_median_ms": r.single.median_ms, "single_p95_ms": r.single.p95_ms,
        "single_min_ms": r.single.min_ms,
        "batch_size": r.batch_size, "batch_repeats": r.batch_repeats,
        "batch_call_ms_mean": r.batch_call_ms_mean,
        "batch_call_ms_std": r.batch_call_ms_std,
        "per_state_us_mean": r.per_state_us_mean,
        "per_state_us_std": r.per_state_us_std,
        "run_dir": r.run_dir, "error": r.error,
    }
    d.update({k: r.drivers.get(k) for k in
              ("backend", "code_path", "mc_mode", "n_members", "num_mc_samples",
               "n_marginal_samples", "n_inducing", "num_integration_steps",
               "ode_method", "hidden_dims", "attractor_radius")})
    return d


def write_csv(rows: list[Row], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(row_to_dict(r))


def _fmt(v, spec=".3f"):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "-"
    return format(v, spec)


def markdown_table(rows: list[Row], meta: dict) -> str:
    ok = [r for r in rows if not r.error]
    bad = [r for r in rows if r.error]
    out = ["# p_success query latency", ""]
    out.append(f"Measured {meta['timestamp']} on {meta['device_name']}, "
               f"torch {meta['torch']}, {meta['host']}.")
    out.append("")
    out.append(f"Batch-1 latency is the mean over {meta['n_queries']} separately timed "
               f"`estimate()` calls, each wrapped in `cuda.synchronize()`. Batched cost "
               f"amortizes one call on {meta['batch_size']} states, "
               f"{meta['batch_repeats']} repeats.")
    out.append("")
    header = ("| system | dim | method | batch-1 mean ms | sd | median | p95 | "
              "batched per-state us | cost drivers |")
    out += [header, "|" + "---|" * 9]
    for r in sorted(ok, key=lambda r: (r.state_dim, r.single.mean_ms)):
        d = r.drivers
        bits = []
        if d.get("n_members"):
            bits.append(f"M={d['n_members']}")
        if d.get("num_mc_samples"):
            bits.append(f"K={d['num_mc_samples']}"
                        + (" batched" if d.get("mc_mode") == "batched" else " seq"))
        if d.get("n_marginal_samples"):
            bits.append(f"S={d['n_marginal_samples']}")
        if d.get("n_inducing"):
            bits.append(f"inducing={d['n_inducing']}")
        if d.get("code_path") == "cov_eigh_fallback":
            bits.append("**legacy eigh path**")
        if d.get("num_integration_steps"):
            bits.append(f"ODE={d['num_integration_steps']}"
                        + (f" {d['ode_method']}" if d.get("ode_method") else ""))
        out.append(
            f"| {r.system} | {r.state_dim} | `{r.method}` | "
            f"{_fmt(r.single.mean_ms)} | {_fmt(r.single.std_ms)} | "
            f"{_fmt(r.single.median_ms)} | {_fmt(r.single.p95_ms)} | "
            f"{_fmt(r.per_state_us_mean, '.2f')} ± {_fmt(r.per_state_us_std, '.2f')} | "
            f"{', '.join(bits) or '-'} |"
        )
    legacy = [r for r in ok if r.drivers.get("code_path") == "cov_eigh_fallback"]
    if legacy:
        cells = ", ".join(f"{r.system}/`{r.method}`" for r in legacy)
        out += ["", "## Not comparable: legacy Laplace path", "",
                f"{cells} carry no `laplace_prec_chol.pt`. Those runs predate the "
                "2026-08-28 precision-factor fix, so sampling falls back to a "
                "float64 eigendecomposition of Sigma on every draw "
                "(`posteriors.py:_weight_noise`), S times per query, instead of one "
                "triangular solve. The cost is roughly 50x and belongs to the old "
                "code path, not to Laplace as it runs today. Re-fitting those "
                "posteriors is the only way to get a comparable number; do not read "
                "these against the cells that show `prec_chol`."]
    if bad:
        out += ["", "## Cells that did not run", ""]
        for r in bad:
            out.append(f"- `{r.system}` / `{r.method}`: {r.error}")
    return "\n".join(out) + "\n"


# ── main ─────────────────────────────────────────────────────────────────────

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--systems", nargs="+", default=list(SYSTEM_PREFIX),
                   choices=list(SYSTEM_PREFIX))
    p.add_argument("--arms", nargs="+", default=ARM_SUFFIXES)
    p.add_argument("--n-queries", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=1000)
    p.add_argument("--batch-repeats", type=int, default=5)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--epoch", type=int, default=None,
                   help="epoch to load; default is the last one on disk")
    p.add_argument("--mc-batched", action="store_true",
                   help="issue the FM ensemble's K MC draws in one call per "
                        "member instead of K sequential calls; same estimator, "
                        "M launches instead of M*K")
    p.add_argument("--fm-ode-steps", type=int, default=None,
                   help="override the FM ensemble's Euler step count (default "
                        "100 from predict_endpoint). CHANGES THE ANSWER: pair "
                        "any timing with an accuracy check against 100 steps.")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--tag", default="", help="suffix for the output filenames")
    p.add_argument("--smoke", action="store_true",
                   help="5 queries on pendulum only, into a scratch dir")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.smoke:
        args.systems = ["pendulum"]
        args.n_queries, args.batch_size, args.batch_repeats, args.warmup = 5, 20, 2, 2

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA unavailable; falling back to CPU", file=sys.stderr)
        device = "cpu"
    device_name = (torch.cuda.get_device_name(0) if device.startswith("cuda")
                   else platform.processor() or "cpu")

    rows: list[Row] = []
    for s in args.systems:
        prefix = SYSTEM_PREFIX[s]
        for suffix in args.arms:
            run_dir = Path(f"{prefix}_{suffix}")
            label = f"{s}/{suffix}"
            if not run_dir.is_dir():
                print(f"[skip] {label}: no run dir at {run_dir}", file=sys.stderr)
                rows.append(Row(system=s, state_dim=0, method=suffix,
                                run_dir=str(run_dir), epoch=-1,
                                device_name=device_name,
                                error=f"missing run dir {run_dir}"))
                continue
            print(f"[run ] {label} ...", flush=True)
            t0 = time.perf_counter()
            try:
                row = profile_cell(s, run_dir, args, device, device_name)
                rows.append(row)
                print(f"[done] {label}: {row.method} "
                      f"batch-1 {row.single.mean_ms:.3f} +/- {row.single.std_ms:.3f} ms, "
                      f"batched {row.per_state_us_mean:.2f} us/state "
                      f"({time.perf_counter() - t0:.1f}s)", flush=True)
            except Exception as exc:  # noqa: BLE001 - report and keep going
                print(f"[FAIL] {label}: {type(exc).__name__}: {exc}", file=sys.stderr,
                      flush=True)
                rows.append(Row(system=s, state_dim=0, method=suffix,
                                run_dir=str(run_dir), epoch=-1,
                                device_name=device_name,
                                error=f"{type(exc).__name__}: {exc}"))
            finally:
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()

    meta = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M"),
        "device_name": device_name,
        "torch": torch.__version__,
        "host": platform.node(),
        "n_queries": args.n_queries,
        "batch_size": args.batch_size,
        "batch_repeats": args.batch_repeats,
    }

    out_dir = Path("/common/users/st1122/tmp/p_success_profile") if args.smoke else args.out_dir
    tag = f"_{args.tag}" if args.tag else ""
    csv_path = out_dir / f"p_success_latency{tag}.csv"
    md_path = out_dir / f"p_success_latency{tag}.md"
    meta_path = out_dir / f"p_success_latency{tag}_meta.json"

    write_csv(rows, csv_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path.write_text(markdown_table(rows, meta))
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")

    print("\n" + markdown_table(rows, meta))
    print(f"wrote {csv_path}\nwrote {md_path}")
    return 1 if all(r.error for r in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
