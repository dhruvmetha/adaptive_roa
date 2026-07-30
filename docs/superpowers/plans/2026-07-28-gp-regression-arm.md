# GP Regression Arm Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `gp_reg`, a multi-output sparse variational GP final-state arm, and close the two GP debts left by earlier plans — the missing GP export wrapper and the warm-start bug that has meant the GP arm never resumes.

**Architecture:** The GP regresses to the **embedded, normalized** endpoint (sin/cos for angles, the raw quaternion for SO3), not the raw state — that is what keeps the SO2 seam and the quaternion double-cover out of the regression target. Predictive samples are drawn through the likelihood so they carry observation noise, then decoded back to raw coordinates. The resulting handle satisfies the same endpoint-MC contract the Bayesian final-state arms already meet, so nothing downstream changes.

**Tech Stack:** PyTorch 2.5.1, gpytorch 1.15.2, Lightning, Hydra, pytest. No new pip dependencies.

**Design spec:** `docs/superpowers/specs/2026-07-27-probabilistic-predictors-design.md`
**Prior plans:** `docs/superpowers/plans/2026-07-28-probabilistic-predictors-foundation.md` (outcome arms), `docs/superpowers/plans/2026-07-28-final-state-predictors.md` (final-state family). Both are merged; this plan builds on `adaptive_roa/predictors/`.

## Global Constraints

- Branch: create `predictors-gp-reg` off `main`. Repo root: `/common/home/st1122/Projects/adaptive_roa`.
- Python interpreter is the repo-local env: **`./env/bin/python`**. Never plain `python`.
- Run tests as `./env/bin/python -m pytest <path> -v` from the repo root.
- Import root is `adaptive_roa` (editable install).
- **No new pip dependencies.** gpytorch 1.15.2 is already installed and all required APIs are verified present (see below).
- **Commit messages must contain zero AI/tool attribution** — no `Co-Authored-By`, no `Claude-Session`, no tool or model names. Hard user preference.
- **Another session commits to this repo concurrently.** Stage explicit paths only; never `git add -A` or `git commit -a`. **Destructive git operations are forbidden**: no `git stash`, `git reset --hard`, `git checkout`, `git clean`. Restore any experiment by editing the file back.
- `predictor.type` must be `generative` for `gp_reg` (it routes the endpoint dataset kind, the endpoint-MC backend, and the `evaluate_full_roa_fast` eval branch). `predictor.name` is the unique arm name.
- The 152 tests under `tests/predictors/` and the 33 under `tests/partx/` must keep passing. Whole suite is currently 442 collected, 0 failed.

## Verified facts this plan depends on

Established by direct measurement against the installed environment — do not re-derive:

- **gpytorch 1.15.2 exposes everything needed:** `IndependentMultitaskVariationalStrategy(base, num_tasks, task_dim=-1)`, `LMCVariationalStrategy`, `MultitaskGaussianLikelihood(num_tasks, ...)`, `VariationalELBO(likelihood, model, num_data, beta=1.0)`, and `batch_shape` works on `ConstantMean`, `ScaleKernel`/`MaternKernel`, and `CholeskyVariationalDistribution`.
- **`likelihood(model(x)).sample([K])` adds observation noise; `model(x).sample([K])` does not.** Measured on a 2-task SVGP: `y.var - f.var` equalled `task_noise_t + global_noise` exactly, and 20 000 draws matched the analytic moments. **The handle must sample through the likelihood** — the spec requires the GP's spread to include the noise floor, or the GP gets an aleatoric-free advantage over the BNN arms in the very comparison this benchmark exists to run.
- **A multi-output GP's `state_dict` keys nest one level deeper** than the existing single-output `GPClassifier` (`variational_strategy.base_variational_strategy.*`), so it needs its own schema persisting `num_tasks` alongside the inducing shape and kernel.
- **The existing GP training loop is full-batch** (`adaptive_roa/partx/gp_classifier.py:63-68`) with no minibatching. Training-set sizes are capped implicitly by `initial_train_size + n_epochs * samples_per_epoch`: pendulum 2 000, cartpole ~1 800, quadrotor2d 12 000, quadrotor3d 17 000. At 12k+ points across many output tasks the full-batch loop is not viable — this plan uses a minibatched ELBO.
- **Embedded widths** (the GP's task count): pendulum 3, cartpole 5, quadrotor2d 7, quadrotor3d 13. `embed_state_for_model` is the identity for quadrotor3d.
- **`normalize_state` is exactly the identity on every SO2 slice and leaves SO3 quaternions at unit norm**, for all four systems; `denormalize_state` re-normalizes the quaternion by a positive scale, so `qw >= 0` survives it. (Verified during the Plan 2 fix wave.)
- **`adaptive_roa/flow_matching/utils/state_transformations.py:28` `extract_circular_state` is pendulum-specific** (hardcoded 3→2) and has zero call sites. It is not reusable here; Task 1 writes a generic decoder.

## File Structure

**Create:**
- `adaptive_roa/predictors/embedding.py` — `EmbeddedStateDecoder`, mapping embedded+normalized vectors back to raw states per manifold component
- `adaptive_roa/predictors/gp_regressor.py` — `MultitaskSVGP` (the gpytorch module) and `GPRegressor` (fit/sample/state_dict wrapper)
- `adaptive_roa/predictors/gp_final_state_handle.py` — `GPFinalStateHandle`
- `adaptive_roa/adaptive_v2/trainers/gp_regressor_trainer.py` — `GPRegressorTrainer`
- `adaptive_roa/probabilistic_classifier/gaussian_process.py` — export wrappers for `gp`, `gp_optdelta` and `gp_reg`
- `configs/adaptive_v2/predictor/gp_reg.yaml`
- `tests/predictors/{test_embedding.py,test_gp_regressor.py,test_gp_final_state_handle.py,test_gp_trainer.py}`
- `tests/partx/test_gp_export.py`

**Modify:**
- `adaptive_roa/partx/trainer.py:57` — checkpoint filename, to fix warm start
- `adaptive_roa/probabilistic_classifier/__init__.py` — side-effect import
- `configs/adaptive_v2/predictor/gp.yaml`, `gp_optdelta.yaml` — restore `predictor.name`

---

### Task 1: `EmbeddedStateDecoder`

The GP regresses in embedded space; this inverts that mapping. Correctness here is what keeps the SO2 seam and the quaternion double-cover out of the regression.

**Files:**
- Create: `adaptive_roa/predictors/embedding.py`
- Test: `tests/predictors/test_embedding.py`

**Interfaces:**
- Consumes: `wrap_angle`, `canonicalize_quaternion` from `adaptive_roa.predictors.manifold_likelihood`
- Produces: `EmbeddedStateDecoder(system)` with `.embed_dim -> int`, `.state_dim -> int`, `.decode(embedded) -> Tensor[B, state_dim]` (RAW coordinates)

- [ ] **Step 1: Write the failing test**

Create `tests/predictors/test_embedding.py`:

```python
import math

import pytest
import torch

from adaptive_roa.predictors.embedding import EmbeddedStateDecoder
from adaptive_roa.systems.cartpole import CartPoleSystem
from adaptive_roa.systems.quadrotor2d import Quadrotor2DSystem
from adaptive_roa.systems.quadrotor3d import Quadrotor3DSystem

SYSTEMS = [(CartPoleSystem, 4, 5), (Quadrotor2DSystem, 6, 7), (Quadrotor3DSystem, 13, 13)]


@pytest.mark.parametrize("cls,state_dim,embed_dim", SYSTEMS)
def test_dims_match_the_system(cls, state_dim, embed_dim):
    dec = EmbeddedStateDecoder(cls())
    assert dec.state_dim == state_dim
    assert dec.embed_dim == embed_dim


@pytest.mark.parametrize("cls,state_dim,embed_dim", SYSTEMS)
def test_decode_inverts_embed_for_in_range_states(cls, state_dim, embed_dim):
    """decode(embed(normalize(x))) == x for states inside the system's bounds.
    This is the round trip the GP relies on: it learns in embedded space and the
    handle must recover the raw state exactly."""
    system = cls()
    dec = EmbeddedStateDecoder(system)

    torch.manual_seed(0)
    raw = system.denormalize_state(torch.rand(64, state_dim) * 2 - 1)
    if cls is Quadrotor3DSystem:  # keep the quaternion on the sphere, qw >= 0
        q = torch.nn.functional.normalize(raw[:, 3:7], dim=-1)
        raw[:, 3:7] = torch.where(q[:, 0:1] < 0, -q, q)

    embedded = system.embed_state_for_model(system.normalize_state(raw))
    assert embedded.shape == (64, embed_dim)
    torch.testing.assert_close(dec.decode(embedded), raw, atol=1e-4, rtol=1e-4)


def test_decode_recovers_angles_across_the_seam():
    """The whole reason the GP works in embedded space: an angle near +/-pi has
    no discontinuity in (sin, cos), so decode must return it correctly."""
    system = CartPoleSystem()
    dec = EmbeddedStateDecoder(system)
    for theta in (math.pi - 1e-3, -math.pi + 1e-3, 3.0, -3.0, 0.0):
        raw = torch.tensor([[0.0, theta, 0.0, 0.0]])
        out = dec.decode(system.embed_state_for_model(system.normalize_state(raw)))
        assert out[0, 1].item() == pytest.approx(theta, abs=1e-4)


def test_decode_normalizes_and_canonicalizes_quaternions():
    """A GP sample is an arbitrary 4-vector, not a unit quaternion. Quadrotor3D's
    classify_attractor compares raw 13-vectors by L2 against an identity-quaternion
    goal, so an un-canonicalized -q sits 2.0 away and is misclassified."""
    system = Quadrotor3DSystem()
    dec = EmbeddedStateDecoder(system)
    embedded = torch.randn(32, dec.embed_dim)
    embedded[:16, 3] = -abs(embedded[:16, 3])  # force qw < 0 on half the rows
    out = dec.decode(embedded)
    q = out[:, 3:7]
    torch.testing.assert_close(torch.linalg.norm(q, dim=-1), torch.ones(32), atol=1e-5, rtol=0)
    assert (q[:, 0] >= 0).all()


def test_decode_output_is_consumable_by_classify_attractor():
    for cls, state_dim, _e in SYSTEMS:
        system = cls()
        dec = EmbeddedStateDecoder(system)
        out = dec.decode(torch.randn(16, dec.embed_dim))
        assert out.shape == (16, state_dim)
        assert system.classify_attractor(out, radius=0.3).shape == (16,)


def test_unknown_component_type_is_rejected():
    from adaptive_roa.systems.base import ManifoldComponent

    class StubSystem:
        manifold_components = [ManifoldComponent("Hyperbolic", 2, "weird")]

    with pytest.raises(ValueError, match="Hyperbolic"):
        EmbeddedStateDecoder(StubSystem())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./env/bin/python -m pytest tests/predictors/test_embedding.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'adaptive_roa.predictors.embedding'`

- [ ] **Step 3: Write the implementation**

Create `adaptive_roa/predictors/embedding.py`:

```python
"""Inverse of ``system.embed_state_for_model``.

A Gaussian-process regressor cannot learn an angle directly: theta = +/-pi is a
seam, and the pendulum's failure attractors sit exactly on it. Nor can it learn a
quaternion directly, since -q is the same rotation as q. So the GP regresses to
the EMBEDDED representation -- (sin, cos) per angle, the 4-vector per rotation --
and this decoder maps a sample back to a raw state.

``extract_circular_state`` in flow_matching/utils is pendulum-specific (a
hardcoded 3 -> 2 mapping) and has no call sites; this is the generic form.
"""
from __future__ import annotations

from typing import List, Tuple

import torch

from adaptive_roa.predictors.manifold_likelihood import (
    canonicalize_quaternion,
    wrap_angle,
)

# Embedded width consumed per component, keyed by manifold type.
_EMBED_WIDTH = {"Real": lambda dim: int(dim), "SO2": lambda dim: 2, "SO3": lambda dim: 4}


class EmbeddedStateDecoder:
    """Maps ``embed_state_for_model(normalize_state(x))`` back to raw ``x``."""

    def __init__(self, system):
        self.system = system
        self._parts: List[Tuple[str, int, int, int, int]] = []

        embed_offset = 0
        state_offset = 0
        for comp in system.manifold_components:
            width_fn = _EMBED_WIDTH.get(comp.manifold_type)
            if width_fn is None:
                raise ValueError(
                    f"no embedding decoder for manifold component "
                    f"{comp.manifold_type!r}; expected one of {sorted(_EMBED_WIDTH)}"
                )
            width = width_fn(comp.dim)
            self._parts.append((comp.manifold_type, embed_offset, width, state_offset, comp.dim))
            embed_offset += width
            state_offset += comp.dim

        self.embed_dim = embed_offset
        self.state_dim = state_offset

    def decode(self, embedded: torch.Tensor) -> torch.Tensor:
        """[B, embed_dim] -> [B, state_dim] in RAW coordinates."""
        out = []
        for kind, e0, width, _s0, dim in self._parts:
            chunk = embedded[..., e0:e0 + width]
            if kind == "Real":
                out.append(chunk)
            elif kind == "SO2":
                # atan2 over an unnormalized direction: magnitude is irrelevant and
                # there is no seam, which is the point of regressing in this space.
                out.append(wrap_angle(torch.atan2(chunk[..., 0:1], chunk[..., 1:2])))
            else:  # SO3
                out.append(canonicalize_quaternion(chunk))
        normalized = torch.cat(out, dim=-1)
        return self.system.denormalize_state(normalized)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./env/bin/python -m pytest tests/predictors/test_embedding.py -v`
Expected: PASS (13 tests — 2 parametrized ×3, plus 4 singles and the reject test)

- [ ] **Step 5: Commit**

```bash
git add adaptive_roa/predictors/embedding.py tests/predictors/test_embedding.py
git commit -m "feat(predictors): generic decoder from embedded to raw state"
```

---

### Task 2: `MultitaskSVGP` and `GPRegressor`

**Files:**
- Create: `adaptive_roa/predictors/gp_regressor.py`
- Test: `tests/predictors/test_gp_regressor.py`

**Interfaces:**
- Consumes: nothing from earlier tasks
- Produces: `GPRegressor(num_tasks, input_dim, n_inducing=128, kernel="matern52", n_iters=300, lr=0.01, batch_size=1024, device="cpu")` with `.fit(X, Y) -> self`, `.sample(X, num_samples) -> Tensor[num_samples, N, num_tasks]`, `.mean(X) -> Tensor[N, num_tasks]`, `.eval()`, `.to(device)`, `.state_dict()`, `.load_state_dict(sd)`

- [ ] **Step 1: Write the failing test**

Create `tests/predictors/test_gp_regressor.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./env/bin/python -m pytest tests/predictors/test_gp_regressor.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'adaptive_roa.predictors.gp_regressor'`

- [ ] **Step 3: Write the implementation**

Create `adaptive_roa/predictors/gp_regressor.py`:

```python
"""Multi-output sparse variational GP over embedded endpoints.

One independent SVGP per output task, sharing an inducing-point set, wrapped in
gpytorch's IndependentMultitaskVariationalStrategy. Trained with a minibatched
ELBO because the adaptive pool reaches 12k-17k rows on the quadrotors and the
existing single-output GP's full-batch loop does not scale across tasks.

Sampling goes through the LIKELIHOOD, not the latent function: the spec requires
the arm's predictive spread to include the observation-noise floor, or the GP
gets an aleatoric-free advantage over the Bayesian NN arms.
"""
from __future__ import annotations

import gpytorch
import torch


class MultitaskSVGP(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points: torch.Tensor, num_tasks: int, kernel: str = "matern52"):
        # inducing_points: [num_tasks, n_inducing, input_dim]
        batch_shape = torch.Size([num_tasks])
        vd = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(-2), batch_shape=batch_shape
        )
        base = gpytorch.variational.VariationalStrategy(
            self, inducing_points, vd, learn_inducing_locations=True
        )
        super().__init__(
            gpytorch.variational.IndependentMultitaskVariationalStrategy(base, num_tasks=num_tasks)
        )
        d = inducing_points.size(-1)
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=batch_shape)
        base_kernel = (
            gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=d, batch_shape=batch_shape)
            if kernel == "matern52"
            else gpytorch.kernels.RBFKernel(ard_num_dims=d, batch_shape=batch_shape)
        )
        self.covar_module = gpytorch.kernels.ScaleKernel(base_kernel, batch_shape=batch_shape)

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )


class GPRegressor:
    """Fit/sample wrapper with its own state_dict schema."""

    def __init__(self, num_tasks: int, input_dim: int, n_inducing: int = 128,
                 kernel: str = "matern52", n_iters: int = 300, lr: float = 0.01,
                 batch_size: int = 1024, device: str = "cpu"):
        self.num_tasks = int(num_tasks)
        self.input_dim = int(input_dim)
        self.n_inducing = int(n_inducing)
        self.kernel = kernel
        self.n_iters = int(n_iters)
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self.device = device
        self.model = None
        self.likelihood = None
        self.last_batches_per_iter = 0

    def _build(self, inducing: torch.Tensor):
        self.model = MultitaskSVGP(inducing, self.num_tasks, self.kernel).to(self.device)
        self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
            num_tasks=self.num_tasks
        ).to(self.device)

    def fit(self, X: torch.Tensor, Y: torch.Tensor) -> "GPRegressor":
        X = torch.as_tensor(X, dtype=torch.float32).to(self.device)
        Y = torch.as_tensor(Y, dtype=torch.float32).to(self.device)
        n = X.size(0)
        n_ind = min(self.n_inducing, n)
        perm = torch.randperm(n)[:n_ind]
        inducing = X[perm].clone().unsqueeze(0).repeat(self.num_tasks, 1, 1)
        self._build(inducing)

        self.model.train()
        self.likelihood.train()
        opt = torch.optim.Adam(
            list(self.model.parameters()) + list(self.likelihood.parameters()), lr=self.lr
        )
        mll = gpytorch.mlls.VariationalELBO(self.likelihood, self.model, num_data=n)

        bs = min(self.batch_size, n)
        self.last_batches_per_iter = (n + bs - 1) // bs
        for _ in range(self.n_iters):
            order = torch.randperm(n, device=X.device)
            for start in range(0, n, bs):
                idx = order[start:start + bs]
                opt.zero_grad()
                loss = -mll(self.model(X[idx]), Y[idx])
                loss.backward()
                opt.step()
        self.eval()
        return self

    def _predictive(self, X: torch.Tensor):
        self.model.eval()
        self.likelihood.eval()
        X = torch.as_tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            return self.likelihood(self.model(X))

    def sample(self, X: torch.Tensor, num_samples: int) -> torch.Tensor:
        """Predictive draws WITH observation noise: [num_samples, N, num_tasks]."""
        with torch.no_grad():
            return self._predictive(X).sample(torch.Size([int(num_samples)]))

    def mean(self, X: torch.Tensor) -> torch.Tensor:
        return self._predictive(X).mean

    def predictive_variance(self, X: torch.Tensor) -> torch.Tensor:
        return self._predictive(X).variance

    def latent_variance(self, X: torch.Tensor) -> torch.Tensor:
        """Latent-function variance, EXCLUDING observation noise. Diagnostic only —
        never sample from this path; see the module docstring."""
        self.model.eval()
        X = torch.as_tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            return self.model(X).variance

    def eval(self):
        if self.model is not None:
            self.model.eval()
            self.likelihood.eval()
        return self

    def to(self, device):
        self.device = device
        if self.model is not None:
            self.model.to(device)
            self.likelihood.to(device)
        return self

    def state_dict(self):
        inducing = self.model.variational_strategy.base_variational_strategy.inducing_points
        return {
            "model": self.model.state_dict(),
            "likelihood": self.likelihood.state_dict(),
            "inducing_shape": tuple(inducing.shape),
            "num_tasks": self.num_tasks,
            "kernel": self.kernel,
        }

    def load_state_dict(self, sd):
        if self.model is None:
            self.num_tasks = int(sd["num_tasks"])
            self.kernel = sd["kernel"]
            self._build(torch.zeros(sd["inducing_shape"]))
        self.model.load_state_dict(sd["model"])
        self.likelihood.load_state_dict(sd["likelihood"])
        self.eval()
        return self
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./env/bin/python -m pytest tests/predictors/test_gp_regressor.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add adaptive_roa/predictors/gp_regressor.py tests/predictors/test_gp_regressor.py
git commit -m "feat(predictors): multi-output sparse variational GP regressor"
```

---

### Task 3: `GPFinalStateHandle`

**Files:**
- Create: `adaptive_roa/predictors/gp_final_state_handle.py`
- Test: `tests/predictors/test_gp_final_state_handle.py`

**Interfaces:**
- Consumes: `GPRegressor` (Task 2), `EmbeddedStateDecoder` (Task 1), `FinalStateHead` from `adaptive_roa.predictors.heads`
- Produces: `GPFinalStateHandle(gp, system, device="cpu")` with `.eval()`, `.to(device)`, `.training`, `.predict_endpoint(states)`, `.get_manifold_component_names()`, `.compute_manifold_distance_per_component(pred, true)`, `.distance_manifold`

**The contract this must satisfy** — all called unguarded by the pipeline:
`predict_endpoint(states)` takes ONE positional arg, returns `[B, state_dim]` in RAW coordinates, and must draw a **fresh** sample every call (the estimator calls it K times on one batch and the spread IS the probability). `get_manifold_component_names()` is called at `adaptive_roa/adaptive/endpoint_evaluation.py:100` and `compute_manifold_distance_per_component` at `:116`; their shapes must agree. `full_roa.py` guards the names call behind `hasattr(model, "distance_manifold")`, so supply both.

**Reuse note:** `FinalStateHead.distance_per_component` and `.component_names` depend only on the system, not on any distribution parameters, so this handle instantiates a `FinalStateHead` purely for those two members rather than duplicating the per-component distance logic.

**Units note — this is the defect Plan 2's final review caught.** `compute_manifold_distance_per_component` must **normalize both arguments** before measuring, matching `adaptive_roa/flow_matching/base/flow_matcher.py:1156-1161`. Both land in the same `artifacts_v2.json` field, which `scripts/compile_adaptive_metrics.py` reads positionally. `distance_manifold.dist` stays RAW, because `full_roa.py:621,649` passes raw for the flow-matching family too. Copy the arrangement in `adaptive_roa/predictors/final_state_handle.py` exactly.

- [ ] **Step 1: Write the failing test**

Create `tests/predictors/test_gp_final_state_handle.py`:

```python
import numpy as np
import pytest
import torch

from adaptive_roa.predictors.embedding import EmbeddedStateDecoder
from adaptive_roa.predictors.gp_final_state_handle import GPFinalStateHandle
from adaptive_roa.predictors.gp_regressor import GPRegressor
from adaptive_roa.systems.cartpole import CartPoleSystem
from adaptive_roa.systems.quadrotor3d import Quadrotor3DSystem


def _handle(cls=CartPoleSystem, n_iters=40):
    system = cls()
    dec = EmbeddedStateDecoder(system)
    rng = np.random.default_rng(0)
    raw = torch.as_tensor(rng.uniform(-0.5, 0.5, size=(200, dec.state_dim)), dtype=torch.float32)
    feats = system.embed_state_for_model(system.normalize_state(raw))
    targets = system.embed_state_for_model(system.normalize_state(raw * 0.5))
    gp = GPRegressor(
        num_tasks=dec.embed_dim, input_dim=dec.embed_dim, n_inducing=16, n_iters=n_iters
    ).fit(feats, targets)
    return GPFinalStateHandle(gp, system), system


def test_predict_endpoint_draws_a_fresh_sample_every_call():
    """The estimator calls this K times on the SAME batch and the spread across
    calls IS the outcome probability. A deterministic handle collapses the arm
    to p in {0, 1}."""
    handle, system = _handle()
    x = torch.randn(16, int(system.state_dim)) * 0.3
    assert not torch.allclose(handle.predict_endpoint(x), handle.predict_endpoint(x))


def test_predict_endpoint_returns_finite_raw_states():
    handle, system = _handle()
    out = handle.predict_endpoint(torch.randn(16, int(system.state_dim)) * 0.3)
    assert out.shape == (16, int(system.state_dim))
    assert torch.isfinite(out).all()


def test_predict_endpoint_takes_a_single_positional_arg():
    import inspect

    sig = inspect.signature(GPFinalStateHandle.predict_endpoint)
    required = [p for n, p in list(sig.parameters.items())[1:]
                if p.default is inspect.Parameter.empty]
    assert len(required) == 1


def test_component_names_and_distance_shapes_agree():
    handle, system = _handle()
    a = handle.predict_endpoint(torch.randn(8, int(system.state_dim)) * 0.3)
    b = handle.predict_endpoint(torch.randn(8, int(system.state_dim)) * 0.3)
    assert handle.compute_manifold_distance_per_component(a, b).shape == (
        8, len(handle.get_manifold_component_names())
    )


def test_endpoint_error_uses_the_normalized_convention():
    """Must match flow_matcher.py:1156-1161, which normalizes both arguments.
    Both land in artifacts_v2.json's endpoint_error, read positionally across
    predictor families."""
    handle, system = _handle()
    a = torch.zeros(1, int(system.state_dim))
    b = torch.zeros(1, int(system.state_dim))
    b[0, 0] = 1.0  # one raw unit on a Euclidean component
    got = handle.compute_manifold_distance_per_component(a, b)
    expected = handle.head.distance_per_component(
        system.normalize_state(a), system.normalize_state(b)
    )
    torch.testing.assert_close(got, expected)


def test_distance_manifold_stays_raw():
    """full_roa.py passes RAW endpoints for the flow-matching family too, so the
    shim must not normalize or endpoint_errors would diverge instead."""
    handle, system = _handle()
    a = torch.zeros(1, int(system.state_dim))
    b = torch.zeros(1, int(system.state_dim))
    b[0, 0] = 1.0
    torch.testing.assert_close(
        handle.distance_manifold.dist(a, b), handle.head.distance_per_component(a, b)
    )


def test_quadrotor3d_endpoints_are_canonicalized():
    handle, system = _handle(cls=Quadrotor3DSystem, n_iters=20)
    out = handle.predict_endpoint(torch.randn(32, int(system.state_dim)) * 0.2)
    q = out[:, 3:7]
    torch.testing.assert_close(torch.linalg.norm(q, dim=-1), torch.ones(32), atol=1e-5, rtol=0)
    assert (q[:, 0] >= 0).all()
    assert system.classify_attractor(out, radius=0.3).shape == (32,)


def test_handle_is_module_shaped_for_the_engine():
    handle, _ = _handle()
    assert handle.eval() is not None
    assert handle.to("cpu") is not None
    assert isinstance(handle.training, bool)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./env/bin/python -m pytest tests/predictors/test_gp_final_state_handle.py -v`
Expected: FAIL — `ModuleNotFoundError` for `gp_final_state_handle`

- [ ] **Step 3: Write the implementation**

Create `adaptive_roa/predictors/gp_final_state_handle.py`:

```python
"""Model handle binding the GP regressor to the endpoint-MC backend.

Mirrors ``adaptive_roa/predictors/final_state_handle.py``: fresh sample per call,
raw-coordinate output, and the same normalized/raw split between
``compute_manifold_distance_per_component`` (normalized, to match the flow
matcher) and ``distance_manifold.dist`` (raw, because full_roa passes raw for
both families).
"""
from __future__ import annotations

from typing import Any, List

import numpy as np
import torch

from adaptive_roa.predictors.embedding import EmbeddedStateDecoder
from adaptive_roa.predictors.heads import FinalStateHead


class _ManifoldDistanceShim:
    def __init__(self, head):
        self._head = head

    def dist(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self._head.distance_per_component(x, y)


class GPFinalStateHandle:
    def __init__(self, gp, system: Any, device: str = "cpu"):
        self.gp = gp
        self.system = system
        self.device = device
        self.training = False
        self.decoder = EmbeddedStateDecoder(system)
        # FinalStateHead is used ONLY for its system-derived distance and naming;
        # neither depends on distribution parameters, so this avoids duplicating
        # the per-component geodesic logic.
        self.head = FinalStateHead(system)
        self.distance_manifold = _ManifoldDistanceShim(self.head)

    def eval(self):
        self.gp.eval()
        self.training = False
        return self

    def to(self, device):
        self.device = device
        self.gp.to(device)
        return self

    def predict_endpoint(self, states) -> torch.Tensor:
        """[B, state_dim] raw -> [B, state_dim] raw. Fresh predictive draw per call."""
        if torch.is_tensor(states):
            x = states.detach().to(dtype=torch.float32)
            out_device = states.device
        else:
            x = torch.as_tensor(np.asarray(states), dtype=torch.float32)
            out_device = self.device
        feats = self.system.embed_state_for_model(self.system.normalize_state(x))
        # One predictive draw, through the likelihood so it carries observation noise.
        sample = self.gp.sample(feats, num_samples=1)[0]
        return self.decoder.decode(sample).to(out_device)

    def get_manifold_component_names(self) -> List[str]:
        return list(self.head.component_names)

    def compute_manifold_distance_per_component(self, predicted, true) -> torch.Tensor:
        # Normalized, matching flow_matcher.py:1156-1161 -- both feed the same
        # artifacts_v2.json field, compared positionally across families.
        return self.head.distance_per_component(
            self.system.normalize_state(predicted), self.system.normalize_state(true)
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./env/bin/python -m pytest tests/predictors/test_gp_final_state_handle.py -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add adaptive_roa/predictors/gp_final_state_handle.py tests/predictors/test_gp_final_state_handle.py
git commit -m "feat(predictors): GP final-state handle for the endpoint-MC contract"
```

---

### Task 4: `GPRegressorTrainer` and the `gp_reg` config

**Files:**
- Create: `adaptive_roa/adaptive_v2/trainers/gp_regressor_trainer.py`
- Create: `configs/adaptive_v2/predictor/gp_reg.yaml`
- Test: `tests/predictors/test_gp_trainer.py`

**Interfaces:**
- Consumes: `GPRegressor` (Task 2), `EmbeddedStateDecoder` (Task 1), `GPFinalStateHandle` (Task 3)
- Produces: `GPRegressorTrainer(cfg, system, system_name)` with `fit(dataset_files, output_dir, resume_checkpoint=None) -> GPFinalStateHandle`

**Read first:** `adaptive_roa/adaptive_v2/trainers/final_state_trainer.py` for the endpoint-datamodule selection and the `_full_training_tensors` helper, and `adaptive_roa/partx/trainer.py` for the GP trainer conventions. This trainer has no Lightning loop — `GPRegressor.fit` owns its own optimization — so it is much shorter than either.

**Checkpoint naming is load-bearing:** write `checkpoints/best-gp.ckpt`. The engine globs `checkpoints/best*.ckpt` (`adaptive_roa/adaptive_v2/engine.py:140`); the existing partx trainer writes `gp.pt`, which is why the GP arm has never warm-started. Task 5 fixes that one; this one must not repeat it.

- [ ] **Step 1: Write the failing test**

Create `tests/predictors/test_gp_trainer.py`:

```python
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from adaptive_roa.adaptive_v2.trainers.gp_regressor_trainer import GPRegressorTrainer
from adaptive_roa.systems.cartpole import CartPoleSystem


def _write_endpoints(path, n=200, seed=0):
    rng = np.random.default_rng(seed)
    start = rng.uniform(-0.5, 0.5, size=(n, 4))
    np.savetxt(path, np.column_stack([start, start * 0.5]))
    return str(path)


def _cfg(**overrides):
    gp = {"n_inducing": 16, "kernel": "matern52", "n_iters": 30, "lr": 0.05, "batch_size": 128}
    gp.update(overrides)
    return OmegaConf.create({
        "device": "cpu",
        "predictor": {"type": "generative", "name": "gp_reg",
                      "batch_size": 64, "val_batch_size": 64, "gp": gp},
    })


@pytest.fixture
def files(tmp_path):
    return {"train": _write_endpoints(tmp_path / "train.txt"),
            "val": _write_endpoints(tmp_path / "val.txt", n=100, seed=1)}


def test_trainer_returns_a_working_handle(files, tmp_path):
    handle = GPRegressorTrainer(_cfg(), CartPoleSystem(), "cartpole_pybullet").fit(
        files, str(tmp_path / "out")
    )
    x = torch.randn(16, 4) * 0.3
    out = handle.predict_endpoint(x)
    assert out.shape == (16, 4)
    assert torch.isfinite(out).all()
    assert not torch.allclose(handle.predict_endpoint(x), handle.predict_endpoint(x))


def test_trainer_writes_a_warm_start_checkpoint(files, tmp_path):
    """The engine globs checkpoints/best*.ckpt (engine.py:140). partx/trainer.py
    writes gp.pt instead, which is why the GP arm has never warm-started."""
    out = tmp_path / "out"
    GPRegressorTrainer(_cfg(), CartPoleSystem(), "cartpole_pybullet").fit(files, str(out))
    assert list((out / "checkpoints").glob("best*.ckpt"))


def test_warm_start_reloads_a_previous_checkpoint(files, tmp_path):
    trainer = GPRegressorTrainer(_cfg(), CartPoleSystem(), "cartpole_pybullet")
    first_out = tmp_path / "e0"
    trainer.fit(files, str(first_out))
    ckpt = list((first_out / "checkpoints").glob("best*.ckpt"))[0]

    resumed = GPRegressorTrainer(_cfg(n_iters=1), CartPoleSystem(), "cartpole_pybullet").fit(
        files, str(tmp_path / "e1"), resume_checkpoint=str(ckpt)
    )
    x = torch.randn(8, 4) * 0.3
    assert torch.isfinite(resumed.predict_endpoint(x)).all()


def test_the_arm_actually_learns_the_contraction(files, tmp_path):
    """Guards against shipping an untrained GP."""
    handle = GPRegressorTrainer(_cfg(n_iters=300), CartPoleSystem(), "cartpole_pybullet").fit(
        files, str(tmp_path / "out")
    )
    raw = np.loadtxt(files["val"])
    x = torch.as_tensor(raw[:, :4], dtype=torch.float32)
    y = torch.as_tensor(raw[:, 4:], dtype=torch.float32)
    feats = handle.system.embed_state_for_model(handle.system.normalize_state(x))
    pred = handle.decoder.decode(handle.gp.mean(feats))
    assert (pred - y).pow(2).mean().item() < 0.5 * (x - y).pow(2).mean().item()


def test_config_composes_with_the_endpoint_mc_backend():
    import os
    from hydra import compose, initialize_config_dir

    d = os.path.abspath("configs/adaptive_v2")
    with initialize_config_dir(config_dir=d, version_base=None):
        cfg = compose(config_name="default", overrides=["predictor=gp_reg"])
    assert cfg.predictor.name == "gp_reg"
    assert cfg.predictor.type == "generative"
    assert "EndpointMCProbabilityBackend" in cfg.probability._target_
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./env/bin/python -m pytest tests/predictors/test_gp_trainer.py -v`
Expected: FAIL — `ModuleNotFoundError` for `gp_regressor_trainer`

- [ ] **Step 3: Write the trainer**

Create `adaptive_roa/adaptive_v2/trainers/gp_regressor_trainer.py`:

```python
"""GP regressor trainer for the gp_reg final-state arm.

Contract matches every sibling: __init__(cfg, system, system_name) and
fit(dataset_files, output_dir, resume_checkpoint=None) -> model handle. There is
no Lightning loop -- GPRegressor.fit owns its own minibatched ELBO optimization.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from adaptive_roa.adaptive_v2.trainers.final_state_trainer import (
    _DATAMODULES,
    _full_training_tensors,
)
from adaptive_roa.predictors.embedding import EmbeddedStateDecoder
from adaptive_roa.predictors.gp_final_state_handle import GPFinalStateHandle
from adaptive_roa.predictors.gp_regressor import GPRegressor


class GPRegressorTrainer:
    def __init__(self, cfg: Any, system: Any, system_name: str):
        self.cfg = cfg
        self.system = system
        self.system_name = system_name

    @property
    def _predictor_cfg(self):
        pred = self.cfg.get("predictor")
        return pred if pred is not None else self.cfg

    def _create_datamodule(self, dataset_files):
        dm_cls = _DATAMODULES.get(self.system_name)
        if dm_cls is None:
            raise ValueError(
                f"no endpoint datamodule for system {self.system_name!r}; "
                f"expected one of {sorted(_DATAMODULES)}"
            )
        dm = dm_cls(
            data_file=dataset_files["train"],
            validation_file=dataset_files["val"],
            test_file=dataset_files["val"],
            batch_size=self._predictor_cfg.get("batch_size", 1024),
            val_batch_size=self._predictor_cfg.get("val_batch_size", 2048),
            num_workers=0,
        )
        dm.setup()
        return dm

    def fit(self, dataset_files: dict, output_dir: str, resume_checkpoint: str | None = None):
        gp_cfg = self._predictor_cfg.get("gp", {})
        device = str(self._predictor_cfg.get("device", self.cfg.get("device", "cuda:0")))
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"

        decoder = EmbeddedStateDecoder(self.system)
        data_module = self._create_datamodule(dataset_files)
        starts, ends = _full_training_tensors(data_module)

        # Both features and targets live in EMBEDDED, NORMALIZED space: an angle
        # regressed directly would carry the +/-pi seam, and a quaternion would
        # carry the double cover.
        feats = self.system.embed_state_for_model(self.system.normalize_state(starts))
        targets = self.system.embed_state_for_model(self.system.normalize_state(ends))

        gp = GPRegressor(
            num_tasks=decoder.embed_dim,
            input_dim=int(feats.shape[-1]),
            n_inducing=int(gp_cfg.get("n_inducing", 128)),
            kernel=str(gp_cfg.get("kernel", "matern52")),
            n_iters=int(gp_cfg.get("n_iters", 300)),
            lr=float(gp_cfg.get("lr", 0.01)),
            batch_size=int(gp_cfg.get("batch_size", 1024)),
            device=device,
        )

        if resume_checkpoint and Path(resume_checkpoint).exists():
            print(f"Warm start: loading GP regressor state from {resume_checkpoint}")
            gp.load_state_dict(torch.load(resume_checkpoint, map_location="cpu",
                                          weights_only=False))
            gp.to(device)

        gp.fit(feats, targets)

        ckpt_dir = Path(output_dir) / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        # MUST match the engine's glob, checkpoints/best*.ckpt (engine.py:140).
        torch.save(gp.state_dict(), ckpt_dir / "best-gp.ckpt")

        return GPFinalStateHandle(gp, self.system, device=device).eval().to(device)
```

Both imported names are verified present at module level in
`final_state_trainer.py` — `_DATAMODULES` at line 36 and `_full_training_tensors`
at line 416. Import them; do not duplicate either. A second copy of the
datamodule map is exactly how the two trainers would silently drift apart.

- [ ] **Step 4: Write the config**

Create `configs/adaptive_v2/predictor/gp_reg.yaml`:

```yaml
# @package _global_
defaults:
  - /probability: endpoint_mc
predictor:
  type: generative          # family tag: endpoint data, endpoint-MC backend, fast eval branch
  name: gp_reg              # arm name: keys the export registry
  trainer_target: adaptive_roa.adaptive_v2.trainers.gp_regressor_trainer.GPRegressorTrainer
  batch_size: 1024
  val_batch_size: 2048
  gp:
    n_inducing: 128
    kernel: matern52
    n_iters: 300
    lr: 0.01
    batch_size: 1024        # minibatch for the ELBO; the pool reaches 12k-17k rows
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `./env/bin/python -m pytest tests/predictors/test_gp_trainer.py -v`
Expected: PASS (5 tests)

- [ ] **Step 6: Commit**

```bash
git add adaptive_roa/adaptive_v2/trainers/gp_regressor_trainer.py configs/adaptive_v2/predictor/gp_reg.yaml tests/predictors/test_gp_trainer.py
git commit -m "feat(predictors): GP regressor trainer and gp_reg arm config"
```

---

### Task 5: Fix the GP warm-start bug and add the GP export wrappers

Closes two debts from earlier plans in one place, because both are about GP checkpoints being loadable.

**Files:**
- Modify: `adaptive_roa/partx/trainer.py:57`
- Create: `adaptive_roa/probabilistic_classifier/gaussian_process.py`
- Modify: `adaptive_roa/probabilistic_classifier/__init__.py`
- Modify: `configs/adaptive_v2/predictor/gp.yaml`, `configs/adaptive_v2/predictor/gp_optdelta.yaml`
- Test: `tests/partx/test_gp_export.py`

**Interfaces:**
- Consumes: `GPClassifier` and `GPModelHandle` from `adaptive_roa.partx`, `GPRegressor` (Task 2), `GPFinalStateHandle` (Task 3)
- Produces: `GPProbabilisticClassifier` (name `gp`), `GPOptDeltaProbabilisticClassifier` (name `gp_optdelta`), `GPRegProbabilisticClassifier` (name `gp_reg`)

**Background:** `adaptive_roa/partx/trainer.py:57` writes `checkpoints/gp.pt`, but the engine globs `checkpoints/best*.ckpt` (`engine.py:140`), so the GP outcome arm has silently never warm-started even with `warm_start: true`. Renaming the file fixes it. Runs already on disk carry `gp.pt`, so the export loader must accept **either** name.

`gp.yaml` and `gp_optdelta.yaml` deliberately carry no `predictor.name` today, because claiming a name with no registered wrapper makes `export_run` raise `KeyError` instead of falling back to the family alias. This task registers the wrappers, so the names can be restored.

- [ ] **Step 1: Write the failing test**

Create `tests/partx/test_gp_export.py`:

```python
import numpy as np
import pytest
import torch


@pytest.mark.parametrize("arm", ["gp", "gp_optdelta", "gp_reg"])
def test_each_gp_arm_is_registered_for_export(arm):
    from adaptive_roa.probabilistic_classifier import get_probabilistic_classifier_class

    cls = get_probabilistic_classifier_class(arm)
    assert cls.predictor_name == arm


def test_outcome_gp_arms_declare_the_classifier_family():
    from adaptive_roa.probabilistic_classifier import get_probabilistic_classifier_class

    for arm in ("gp", "gp_optdelta"):
        cls = get_probabilistic_classifier_class(arm)
        assert cls.predictor_type == "classifier"
        assert cls.native_probs == ("p_success",)


def test_gp_reg_declares_the_generative_family():
    from adaptive_roa.probabilistic_classifier import get_probabilistic_classifier_class

    cls = get_probabilistic_classifier_class("gp_reg")
    assert cls.predictor_type == "generative"
    assert cls.native_probs == ("p_success", "p_failure", "p_invalid")


def test_gp_arms_do_not_steal_the_legacy_family_aliases():
    """Runs written before arm names existed must still resolve."""
    from adaptive_roa.probabilistic_classifier import get_probabilistic_classifier_class
    from adaptive_roa.probabilistic_classifier.classifier import (
        ClassifierProbabilisticClassifier,
    )
    from adaptive_roa.probabilistic_classifier.flow_matching import FMProbabilisticClassifier

    assert get_probabilistic_classifier_class("classifier") is ClassifierProbabilisticClassifier
    assert get_probabilistic_classifier_class("generative") is FMProbabilisticClassifier


def test_partx_trainer_writes_a_warm_start_checkpoint(tmp_path):
    """engine.py:140 globs checkpoints/best*.ckpt. Writing gp.pt meant the GP arm
    silently never warm-started."""
    from omegaconf import OmegaConf
    from adaptive_roa.partx.trainer import GPPredictorTrainer
    from adaptive_roa.systems.cartpole import CartPoleSystem

    system = CartPoleSystem()
    rng = np.random.default_rng(0)
    X = rng.uniform(-1.0, 1.0, size=(120, 4))
    y = (np.abs(X[:, 1]) < 0.5).astype(int)
    path = tmp_path / "train.txt"
    np.savetxt(path, np.column_stack([X, y]))

    cfg = OmegaConf.create({
        "device": "cpu",
        "predictor": {"type": "classifier",
                      "gp": {"n_inducing": 8, "kernel": "matern52", "n_iters": 3, "lr": 0.1}},
    })
    out = tmp_path / "out"
    GPPredictorTrainer(cfg, system, "cartpole_pybullet").fit(
        {"train": str(path)}, str(out)
    )
    assert list((out / "checkpoints").glob("best*.ckpt")), "engine warm start needs best*.ckpt"


def test_gp_export_loader_accepts_the_legacy_checkpoint_name(tmp_path):
    """Runs already on disk carry checkpoints/gp.pt; they must still export."""
    from adaptive_roa.probabilistic_classifier.gaussian_process import _find_gp_checkpoint

    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    legacy = ckpt_dir / "gp.pt"
    legacy.write_bytes(b"x")
    assert _find_gp_checkpoint(ckpt_dir) == legacy

    modern = ckpt_dir / "best-gp.ckpt"
    modern.write_bytes(b"x")
    assert _find_gp_checkpoint(ckpt_dir) == modern  # prefer the new name


def test_missing_gp_checkpoint_raises(tmp_path):
    from adaptive_roa.probabilistic_classifier.gaussian_process import _find_gp_checkpoint

    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        _find_gp_checkpoint(ckpt_dir)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./env/bin/python -m pytest tests/partx/test_gp_export.py -v`
Expected: FAIL — `KeyError: 'gp'` and `ModuleNotFoundError` for `gaussian_process`

- [ ] **Step 3: Fix the partx checkpoint name**

In `adaptive_roa/partx/trainer.py`, replace the tail of `fit` (currently
`gp.fit(X, y)` through the `torch.save(..., ckpt_dir / "gp.pt")` line) with:

```python
        if resume_checkpoint and Path(resume_checkpoint).exists():
            print(f"Warm start: loading GP classifier state from {resume_checkpoint}")
            gp.load_state_dict(
                torch.load(resume_checkpoint, map_location="cpu", weights_only=False)
            )
            gp.to(device)

        gp.fit(X, y)
        ckpt_dir = Path(output_dir) / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        # MUST match the engine's glob, checkpoints/best*.ckpt (engine.py:140).
        # This previously wrote "gp.pt", which the glob never matched, so the GP
        # arm silently never warm-started even with warm_start: true.
        torch.save(gp.state_dict(), ckpt_dir / "best-gp.ckpt")
        return GPModelHandle(gp, self.system).eval().to(device)
```

**That load is a no-op unless `GPClassifier.fit` is also changed.** `fit`
currently rebuilds unconditionally at `adaptive_roa/partx/gp_classifier.py:56-57`:

```python
        self.model = _VarGP(inducing, self.kernel).to(self.device)
        self.likelihood = gpytorch.likelihoods.BernoulliLikelihood().to(self.device)
```

so anything loaded beforehand is discarded. Renaming the checkpoint without this
would be worse than the current bug: the engine would start passing a
`resume_checkpoint` that the trainer silently throws away, which *looks* like
working warm start.

In `gp_classifier.py`, guard the rebuild so an already-loaded model continues:

```python
        if self.model is None or self.likelihood is None:
            self.model = _VarGP(inducing, self.kernel).to(self.device)
            self.likelihood = gpytorch.likelihoods.BernoulliLikelihood().to(self.device)
        # else: warm start -- keep the loaded variational distribution, kernel
        # hyperparameters and inducing locations, and continue optimizing them
        # against the (now larger) training set. num_data below is recomputed
        # from the current y, so the ELBO scaling stays correct.
```

Add a test in `tests/partx/test_gp_export.py` proving the warm start is real:
fit once, save, construct a fresh `GPClassifier`, `load_state_dict`, then fit
again with `n_iters=1` and assert the resulting `p_success` on held-out points is
closer to the first model's than a from-scratch `n_iters=1` model is. If that
assertion proves too noisy to be stable, assert instead that the inducing points
after the warm-started fit equal the loaded ones while a cold fit's differ —
that is a direct, deterministic check of the same property.

- [ ] **Step 4: Write the export wrappers**

Create `adaptive_roa/probabilistic_classifier/gaussian_process.py`:

```python
"""Export wrappers for the three GP arms.

`gp` and `gp_optdelta` are outcome-probability arms (a GP classifier's predictive
probability); `gp_reg` is a final-state arm whose probabilities come from
MC-classified endpoints. All three register under their own names so a GP run is
never loaded as some other arm's checkpoint.
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import torch

from adaptive_roa.adaptive_v2.types import OutcomeProbabilities
from adaptive_roa.partx.gp_classifier import GPClassifier
from adaptive_roa.partx.model_handle import GPModelHandle
from adaptive_roa.predictors.gp_final_state_handle import GPFinalStateHandle
from adaptive_roa.predictors.gp_regressor import GPRegressor
from .base import ProbabilisticClassifier
from .registry import register_probabilistic_classifier


def _find_gp_checkpoint(ckpt_dir) -> Path:
    """Prefer the warm-startable name; accept the legacy one for older runs.

    The partx trainer used to write ``gp.pt``, which the engine's
    ``checkpoints/best*.ckpt`` glob never matched. Runs already on disk carry it.
    """
    ckpt_dir = Path(ckpt_dir)
    modern = sorted(glob.glob(str(ckpt_dir / "best*.ckpt")))
    if modern:
        return Path(modern[0])
    legacy = ckpt_dir / "gp.pt"
    if legacy.exists():
        return legacy
    raise FileNotFoundError(f"no GP checkpoint (best*.ckpt or gp.pt) in {ckpt_dir}")


@register_probabilistic_classifier
class GPProbabilisticClassifier(ProbabilisticClassifier):
    predictor_type = "classifier"
    predictor_name = "gp"
    native_probs = ("p_success",)

    def __init__(self, gp, system, device):
        self.gp = gp
        self.handle = GPModelHandle(gp, system)
        self.system = system
        self.device = device

    def predict(self, states: np.ndarray) -> OutcomeProbabilities:
        p_success = np.asarray(self.gp.p_success(np.asarray(states)), dtype=np.float64)
        return OutcomeProbabilities(
            p_success=p_success,
            p_failure=1.0 - p_success,
            p_invalid=np.zeros_like(p_success),
        )

    @classmethod
    def load_from_run(cls, run_dir, epoch, cfg, system, device="cuda"):
        ckpt = _find_gp_checkpoint(Path(run_dir) / f"epoch_{epoch:03d}" / "checkpoints")
        # GPClassifier.load_state_dict rebuilds from inducing_shape + kernel
        # (gp_classifier.py:107-114), so no architecture config is needed here.
        gp = GPClassifier(system, device=device)
        gp.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False))
        gp.to(device)
        return cls(gp, system, device)


@register_probabilistic_classifier
class GPOptDeltaProbabilisticClassifier(GPProbabilisticClassifier):
    predictor_name = "gp_optdelta"


@register_probabilistic_classifier
class GPRegProbabilisticClassifier(ProbabilisticClassifier):
    predictor_type = "generative"
    predictor_name = "gp_reg"
    native_probs = ("p_success", "p_failure", "p_invalid")

    def __init__(self, handle, system, device, attractor_radius, num_mc_samples):
        self.handle = handle
        self.gp = handle.gp
        self.system = system
        self.device = device
        self.attractor_radius = attractor_radius
        self.num_mc_samples = num_mc_samples

    def predict(self, states: np.ndarray) -> OutcomeProbabilities:
        x = torch.as_tensor(np.asarray(states), dtype=torch.float32)
        counts = torch.zeros(3, x.shape[0], dtype=torch.long)
        with torch.no_grad():
            for _ in range(self.num_mc_samples):
                labels = self.system.classify_attractor(
                    self.handle.predict_endpoint(x), radius=self.attractor_radius
                ).cpu()
                counts[0] += (labels == 1).long()
                counts[1] += (labels == -1).long()
                counts[2] += (labels == 0).long()
        p = counts.double().numpy() / float(self.num_mc_samples)
        return OutcomeProbabilities(p_success=p[0], p_failure=p[1], p_invalid=p[2])

    @classmethod
    def load_from_run(cls, run_dir, epoch, cfg, system, device="cuda"):
        prob = cfg.get("probability", {})
        # No hardcoded radius fallback: a wrong radius silently relabels every
        # endpoint, which looks like a method result rather than a bug.
        if "attractor_radius" not in prob:
            raise KeyError(
                f"run config at {run_dir} has no probability.attractor_radius; "
                "refusing to guess a radius, which would relabel every endpoint."
            )
        ckpt = _find_gp_checkpoint(Path(run_dir) / f"epoch_{epoch:03d}" / "checkpoints")
        # GPRegressor's state_dict carries num_tasks, inducing_shape and kernel,
        # so the architecture is reconstructed without consulting the config.
        gp = GPRegressor(num_tasks=1, input_dim=1, device=device)
        gp.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False))
        gp.to(device)
        handle = GPFinalStateHandle(gp, system, device=device).eval().to(device)
        return cls(
            handle, system, device,
            attractor_radius=float(prob["attractor_radius"]),
            num_mc_samples=int(prob.get("num_mc_samples", 10)),
        )
```

Add `from . import gaussian_process as _gaussian_process  # noqa: F401,E402` to
`adaptive_roa/probabilistic_classifier/__init__.py` **after** the existing
`_classifier` and `_flow_matching` imports, so those keep owning the
`"classifier"` and `"generative"` family aliases (registration is first-wins).

- [ ] **Step 5: Restore the arm names to the GP configs**

In `configs/adaptive_v2/predictor/gp.yaml` add `name: gp` and in `gp_optdelta.yaml`
add `name: gp_optdelta`, directly under `type:`. Replace the existing comment
explaining why the name was absent with one noting the wrapper now exists in
`adaptive_roa/probabilistic_classifier/gaussian_process.py`.

- [ ] **Step 6: Run tests to verify they pass**

Run: `./env/bin/python -m pytest tests/partx/ tests/probabilistic_classifier/ -v`
Expected: PASS — the 7 new tests plus all pre-existing partx and export tests.

- [ ] **Step 7: Commit**

```bash
git add adaptive_roa/partx/trainer.py adaptive_roa/probabilistic_classifier/ configs/adaptive_v2/predictor/gp.yaml configs/adaptive_v2/predictor/gp_optdelta.yaml tests/partx/test_gp_export.py
git commit -m "fix(partx): warm-startable GP checkpoints and export wrappers for all three GP arms"
```

---

### Task 6: End-to-end smoke and regression sweep

**Files:**
- Test: `tests/predictors/test_gp_trainer.py` (append)

**Interfaces:**
- Consumes: everything from Tasks 1-5
- Produces: nothing new

- [ ] **Step 1: Write the failing test**

Append to `tests/predictors/test_gp_trainer.py`:

```python
def test_gp_reg_export_round_trips_a_trained_arm(tmp_path):
    """The export path is what produces the per-point probabilities used in
    analysis; nothing else exercises it."""
    from omegaconf import OmegaConf
    from adaptive_roa.probabilistic_classifier.gaussian_process import (
        GPRegProbabilisticClassifier,
    )

    files = {"train": _write_endpoints(tmp_path / "train.txt"),
             "val": _write_endpoints(tmp_path / "val.txt", n=100, seed=1)}
    run_dir = tmp_path / "run"
    GPRegressorTrainer(_cfg(), CartPoleSystem(), "cartpole_pybullet").fit(
        files, str(run_dir / "epoch_000")
    )

    cfg = OmegaConf.create({
        "predictor": {"type": "generative", "name": "gp_reg",
                      "gp": {"n_inducing": 16, "kernel": "matern52"}},
        "probability": {"attractor_radius": 0.37, "num_mc_samples": 7},
    })
    clf = GPRegProbabilisticClassifier.load_from_run(
        str(run_dir), 0, cfg, CartPoleSystem(), device="cpu"
    )
    assert clf.attractor_radius == 0.37
    assert clf.num_mc_samples == 7

    probs = clf.predict(np.random.default_rng(0).uniform(-0.3, 0.3, size=(24, 4)).astype("float32"))
    total = probs.p_success + probs.p_failure + probs.p_invalid
    np.testing.assert_allclose(total, np.ones(24), atol=1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./env/bin/python -m pytest tests/predictors/test_gp_trainer.py -k export -v`
Expected: FAIL until Task 5's wrapper exposes `attractor_radius` / `num_mc_samples`
as attributes; if it already does, this passes immediately and is a regression guard.

- [ ] **Step 3: Make it pass**

If the test fails, expose `self.attractor_radius` and `self.num_mc_samples` on
`GPRegProbabilisticClassifier` in `adaptive_roa/probabilistic_classifier/gaussian_process.py`,
read from the run config's `probability` block with no hardcoded fallback.

- [ ] **Step 4: Run the pipeline smoke test**

Run:
```bash
./env/bin/python scripts/run_adaptive.py \
  --config-name=default system=pendulum predictor=gp_reg \
  sampling_mode=ranked +adaptive_v2.smoke_mode=true n_epochs=2 \
  predictor.gp.n_iters=20
```
Expected: two epochs complete; `epoch_000/artifacts_v2.json` and
`epoch_001/artifacts_v2.json` exist. `n_epochs` is the correct top-level key.
If it fails on dataset paths for environmental reasons rather than code, record the
exact command and output and proceed.

Also verify the GP outcome arm now warm-starts, which it never has:
```bash
./env/bin/python scripts/run_adaptive.py \
  --config-name=default system=pendulum predictor=gp acquisition=partx eval=partx \
  +adaptive_v2.smoke_mode=true n_epochs=2 warm_start=true predictor.gp.n_iters=20
```
Expected: epoch 1 prints a warm-start line naming `epoch_000/checkpoints/best-gp.ckpt`.

- [ ] **Step 5: Run the full suite for regressions**

Run: `./env/bin/python -m pytest tests -q`
Expected: no failures. All pre-existing `tests/predictors/` and `tests/partx/` tests
must still pass.

- [ ] **Step 6: Commit**

```bash
git add tests/predictors/test_gp_trainer.py adaptive_roa/probabilistic_classifier/gaussian_process.py
git commit -m "test(predictors): end-to-end export round trip for the gp_reg arm"
```

---

## Self-Review

**Spec coverage for this plan's scope:**

| Requirement | Task |
|---|---|
| `gp_reg` multi-output SVGP | 2 |
| Minibatched ELBO for the 12k-17k pool | 2 |
| Own state-dict schema carrying `num_tasks` | 2 |
| Predictive sampling includes observation noise (fairness vs BNN arms) | 2, 3 |
| Manifold-safe regression target (SO2 seam, SO3 double cover) | 1, 3 |
| Endpoint-MC contract, fresh draw per call | 3 |
| Normalized/raw distance split matching the flow matcher | 3 |
| `gp_reg` trainer and config | 4 |
| GP export wrapper deferred from Plan 1 | 5 |
| `predictor.name` restored to `gp.yaml` / `gp_optdelta.yaml` | 5 |
| `gp.pt` vs `best*.ckpt` warm-start bug | 5 |
| GP outcome arm honours `resume_checkpoint` | 5 |
| End-to-end export round trip | 6 |

**Deferred deliberately:** the MDN likelihood (spec marks it optional; build it once the Gaussian head demonstrably fails at separatrices), the HMC reference tier, and benchmark orchestration.

**Type consistency check:** `EmbeddedStateDecoder(system)` exposing `.embed_dim`, `.state_dim`, `.decode` is constructed identically in Tasks 1, 3 and 4. `GPRegressor(num_tasks, input_dim, ...)` with `.fit`, `.sample`, `.mean`, `.predictive_variance`, `.latent_variance`, `.state_dict`, `.load_state_dict` is used identically in Tasks 2-6. `GPFinalStateHandle(gp, system, device)` is constructed identically in Tasks 3-6. `_find_gp_checkpoint(ckpt_dir) -> Path` is defined and consumed in Task 5.

**Known risk to watch during execution:** Task 4 imports `_DATAMODULES` and `_full_training_tensors` from `final_state_trainer.py`. Both are private-by-convention names in a module this plan does not own. If either has a different name or signature, the implementer must read that file and adapt rather than duplicate — the instruction is in the task, and a duplicated datamodule map is exactly how the two trainers would drift apart.
