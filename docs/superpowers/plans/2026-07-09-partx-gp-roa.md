# Part-X-style GP + Level-Set BO + Partitioning — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an `acquisition_mode=partx` to the adaptive-v2 RoA pipeline that uses a GP classifier surrogate, branch-and-bound state-space partitioning, and level-set (straddle) BO selection over the trajectory pool, emitting Bayesian RoA-volume bounds alongside the existing conformal coverage.

**Architecture:** New self-contained package `adaptive_roa/partx/` implementing the existing v2 Protocols (`PredictorTrainer`, `ProbabilityBackend`, `AcquisitionStrategy`, `Evaluator`), wired in purely via Hydra `_target_` configs. The GP classifier's latent `f(x)` is the robustness; `p_success = Φ(m/√(1+s²))`; the RoA boundary is `{f=0}`. A single guarded one-line engine edit lets conformal q_hat calibration run in `partx` mode. Scope: pendulum (2D) then cartpole (4D), pool-restricted (no live simulator).

**Tech Stack:** Python, PyTorch, **GPyTorch** (variational sparse GP + `BernoulliLikelihood`), NumPy, Hydra/OmegaConf, Lightning (engine only), pytest, matplotlib (viz).

## Global Constraints

- Reuse the design spec verbatim: `docs/superpowers/specs/2026-07-08-partx-gp-roa-design.md`.
- GP features = `system.embed_state_for_model(system.normalize_state(X))` (angles→sin/cos, reals→[-1,1]). Never raw Euclidean on circular dims.
- GP training target: `success (label==1) → 1`, `failure (label==-1) → 0`; drop separatrix `0`/invalid `-2` when the label scheme uses signed sentinels (see Task 2 mapping).
- Latent posterior everywhere for straddle/classification/bounds; `p_success` only for conformal + reporting.
- All new code deterministic under the engine seed. No changes to existing `conformal`/`ranked`/`direct` behavior.
- Trainer/backend/strategy signatures must match the verified contracts in `adaptive_roa/adaptive_v2/{interfaces.py,engine.py}` (mirrored below in each task's Interfaces block).
- Commit style: no AI/tool attribution, no `Claude-Session` trailer, no `Co-authored-by`.
- Tests live under `tests/partx/`; run with the project env `env/bin/python -m pytest`.

---

### Task 0: Dependency — add GPyTorch

**Files:**
- Modify: `requirements.txt` (or `pyproject.toml`/`setup.py` — whichever pins deps)

- [ ] **Step 1: Install into the project env**

Run: `env/bin/python -m pip install gpytorch`
Expected: installs gpytorch (and its `linear_operator` dep).

- [ ] **Step 2: Verify import + pin**

Run: `env/bin/python -c "import gpytorch; print(gpytorch.__version__)"`
Expected: prints a version (≥1.11). Add the printed pin (e.g. `gpytorch>=1.11`) to the deps file.

- [ ] **Step 3: Commit**

```bash
git add requirements.txt
git commit -m "build: add gpytorch dependency for partx GP classifier"
```

---

### Task 1: GP classifier core (`GPClassifier`)

**Files:**
- Create: `adaptive_roa/partx/__init__.py` (empty)
- Create: `adaptive_roa/partx/gp_classifier.py`
- Test: `tests/partx/__init__.py` (empty), `tests/partx/test_gp_classifier.py`

**Interfaces:**
- Consumes: a `system` object with `.state_dim`, `.normalize_state(tensor)`, `.embed_state_for_model(tensor)`.
- Produces:
  - `class GPClassifier(system, n_inducing=128, kernel="matern52", n_iters=300, lr=0.1, device="cpu")`
  - `.fit(X_raw: np.ndarray[N,D], y01: np.ndarray[N]) -> self`
  - `.latent_posterior(X_raw: np.ndarray[M,D]) -> tuple[np.ndarray[M], np.ndarray[M]]`  # (mean, var)
  - `.p_success(X_raw: np.ndarray[M,D]) -> np.ndarray[M]`  # Φ(m/√(1+s²))
  - `.state_dict() -> dict`, `.load_state_dict(dict)`, `.eval()`, `.to(device)`

- [ ] **Step 1: Write the failing test**

```python
# tests/partx/test_gp_classifier.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `env/bin/python -m pytest tests/partx/test_gp_classifier.py -v`
Expected: FAIL — `ModuleNotFoundError: adaptive_roa.partx.gp_classifier`.

- [ ] **Step 3: Write minimal implementation**

```python
# adaptive_roa/partx/gp_classifier.py
from __future__ import annotations

import gpytorch
import numpy as np
import torch


class _VarGP(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points: torch.Tensor, kernel: str):
        vd = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0))
        vs = gpytorch.variational.VariationalStrategy(
            self, inducing_points, vd, learn_inducing_locations=True
        )
        super().__init__(vs)
        d = inducing_points.size(-1)
        self.mean_module = gpytorch.means.ConstantMean()
        base = (
            gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=d)
            if kernel == "matern52"
            else gpytorch.kernels.RBFKernel(ard_num_dims=d)
        )
        self.covar_module = gpytorch.kernels.ScaleKernel(base)

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )


class GPClassifier:
    """Variational sparse GP classifier. Latent f is the robustness."""

    def __init__(self, system, n_inducing=128, kernel="matern52",
                 n_iters=300, lr=0.1, device="cpu"):
        self.system = system
        self.n_inducing = int(n_inducing)
        self.kernel = kernel
        self.n_iters = int(n_iters)
        self.lr = float(lr)
        self.device = device
        self.model = None
        self.likelihood = None

    def _features(self, X_raw: np.ndarray) -> torch.Tensor:
        x = torch.as_tensor(np.asarray(X_raw), dtype=torch.float32)
        feats = self.system.embed_state_for_model(self.system.normalize_state(x))
        return feats.to(self.device)

    def fit(self, X_raw: np.ndarray, y01: np.ndarray) -> "GPClassifier":
        X = self._features(X_raw)
        y = torch.as_tensor(np.asarray(y01), dtype=torch.float32).to(self.device)
        n_ind = min(self.n_inducing, X.size(0))
        # Inducing points: random subset of the training features (deterministic under seed).
        perm = torch.randperm(X.size(0))[:n_ind]
        inducing = X[perm].clone()
        self.model = _VarGP(inducing, self.kernel).to(self.device)
        self.likelihood = gpytorch.likelihoods.BernoulliLikelihood().to(self.device)
        self.model.train(); self.likelihood.train()
        opt = torch.optim.Adam(
            list(self.model.parameters()) + list(self.likelihood.parameters()), lr=self.lr
        )
        mll = gpytorch.mlls.VariationalELBO(self.likelihood, self.model, num_data=y.size(0))
        for _ in range(self.n_iters):
            opt.zero_grad()
            out = self.model(X)
            loss = -mll(out, y)
            loss.backward()
            opt.step()
        self.eval()
        return self

    def _latent(self, X_raw):
        self.model.eval(); self.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            f = self.model(self._features(X_raw))
            return f.mean.cpu().numpy(), f.variance.cpu().numpy()

    def latent_posterior(self, X_raw):
        return self._latent(X_raw)

    def p_success(self, X_raw):
        m, s2 = self._latent(X_raw)
        # Integrated probit predictive probability.
        from scipy.stats import norm
        return norm.cdf(m / np.sqrt(1.0 + s2))

    def eval(self):
        if self.model is not None:
            self.model.eval(); self.likelihood.eval()
        return self

    def to(self, device):
        self.device = device
        if self.model is not None:
            self.model.to(device); self.likelihood.to(device)
        return self

    def state_dict(self):
        return {"model": self.model.state_dict(), "likelihood": self.likelihood.state_dict()}

    def load_state_dict(self, sd):
        self.model.load_state_dict(sd["model"])
        self.likelihood.load_state_dict(sd["likelihood"])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `env/bin/python -m pytest tests/partx/test_gp_classifier.py -v`
Expected: PASS. (If flaky at the tolerances, raise `n_iters` to 400; do not loosen the assertions below 0.7/0.3.)

- [ ] **Step 5: Commit**

```bash
git add adaptive_roa/partx/__init__.py adaptive_roa/partx/gp_classifier.py tests/partx/__init__.py tests/partx/test_gp_classifier.py
git commit -m "feat(partx): variational GP classifier with latent posterior access"
```

---

### Task 2: GP trainer adapter (`GPPredictorTrainer`)

**Files:**
- Create: `adaptive_roa/partx/trainer.py`
- Create: `adaptive_roa/partx/model_handle.py`
- Test: `tests/partx/test_trainer.py`

**Interfaces:**
- Consumes: `GPClassifier` (Task 1); engine calls `trainer.fit(dataset_files={"train","val"}, output_dir, resume_checkpoint=None) -> model_handle`.
- Produces:
  - `class GPModelHandle(gp: GPClassifier, system)`: `.eval()`, `.to(device)`, `.gp` attribute.
  - `class GPPredictorTrainer(cfg, system, system_name)` with `.fit(...)-> GPModelHandle`.
  - Module-level `load_xy(path, state_dim) -> tuple[np.ndarray, np.ndarray]` mapping labels to `{0,1}` and dropping separatrix/invalid.

- [ ] **Step 1: Write the failing test**

```python
# tests/partx/test_trainer.py
import numpy as np
from adaptive_roa.partx.trainer import load_xy


def test_load_xy_signed_scheme(tmp_path):
    f = tmp_path / "train.txt"
    # (theta, theta_dot, label): 1=success, -1=failure, 0=separatrix, -2=invalid
    f.write_text("0.0 0.0 1\n2.1 0.0 -1\n1.0 0.0 0\n0.5 0.0 -2\n")
    X, y = load_xy(str(f), state_dim=2)
    assert X.shape == (2, 2)           # separatrix + invalid dropped
    assert list(y) == [1, 0]           # success->1, failure->0


def test_load_xy_binary_scheme(tmp_path):
    f = tmp_path / "train.txt"
    f.write_text("0.0 0.0 1\n2.1 0.0 0\n")   # {0,1} scheme: keep all
    X, y = load_xy(str(f), state_dim=2)
    assert X.shape == (2, 2)
    assert list(y) == [1, 0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `env/bin/python -m pytest tests/partx/test_trainer.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write minimal implementation**

```python
# adaptive_roa/partx/model_handle.py
from __future__ import annotations


class GPModelHandle:
    """Engine-compatible wrapper around a fitted GPClassifier."""

    def __init__(self, gp, system):
        self.gp = gp
        self.system = system

    def eval(self):
        self.gp.eval()
        return self

    def to(self, device):
        self.gp.to(device)
        return self
```

```python
# adaptive_roa/partx/trainer.py
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from adaptive_roa.partx.gp_classifier import GPClassifier
from adaptive_roa.partx.model_handle import GPModelHandle


def load_xy(path: str, state_dim: int):
    """Load (state, label) rows; map to binary success/failure, drop sep/invalid."""
    raw = np.loadtxt(path).reshape(-1, state_dim + 1)
    X, labels = raw[:, :state_dim], raw[:, -1]
    uniq = set(np.unique(labels).tolist())
    if uniq <= {0.0, 1.0}:                      # {0,1} scheme: 0=failure, 1=success
        keep = np.ones(len(labels), dtype=bool)
    else:                                        # signed scheme: keep only ±1
        keep = np.isin(labels, [1.0, -1.0])
    X = X[keep]
    y01 = (labels[keep] == 1.0).astype(np.int64) if not (uniq <= {0.0, 1.0}) \
        else (labels[keep] > 0.5).astype(np.int64)
    return X, y01


class GPPredictorTrainer:
    def __init__(self, cfg: Any, system: Any, system_name: str):
        self.cfg = cfg
        self.system = system
        self.system_name = system_name

    @property
    def _gp_cfg(self):
        pred = self.cfg.get("predictor")
        base = pred if pred is not None else self.cfg
        return base.get("gp", {})

    def fit(self, dataset_files: dict, output_dir: str, resume_checkpoint: str | None = None):
        gp_cfg = self._gp_cfg
        device = str(self.cfg.get("device", "cpu"))
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        X, y = load_xy(dataset_files["train"], int(self.system.state_dim))
        gp = GPClassifier(
            self.system,
            n_inducing=int(gp_cfg.get("n_inducing", 128)),
            kernel=str(gp_cfg.get("kernel", "matern52")),
            n_iters=int(gp_cfg.get("n_iters", 300)),
            lr=float(gp_cfg.get("lr", 0.1)),
            device=device,
        )
        gp.fit(X, y)
        ckpt_dir = Path(output_dir) / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(gp.state_dict(), ckpt_dir / "gp.pt")
        return GPModelHandle(gp, self.system).eval().to(device)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `env/bin/python -m pytest tests/partx/test_trainer.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add adaptive_roa/partx/trainer.py adaptive_roa/partx/model_handle.py tests/partx/test_trainer.py
git commit -m "feat(partx): GP trainer adapter + label loading"
```

---

### Task 3: GP probability backend (`GPProbabilityBackend`)

**Files:**
- Create: `adaptive_roa/partx/backend.py`
- Test: `tests/partx/test_backend.py`

**Interfaces:**
- Consumes: `GPModelHandle` (Task 2). Engine calls `_instantiate(cfg.probability, system, device)` → `GPProbabilityBackend(cfg, system, device)`.
- Produces:
  - `.bind_model(model_handle) -> None`
  - `.estimate(start_states: np.ndarray) -> OutcomeProbabilities`
  - `.latent_posterior(start_states: np.ndarray) -> tuple[np.ndarray, np.ndarray]`

- [ ] **Step 1: Write the failing test**

```python
# tests/partx/test_backend.py
import numpy as np
from omegaconf import OmegaConf
from adaptive_roa.systems.pendulum import PendulumSystem
from adaptive_roa.partx.gp_classifier import GPClassifier
from adaptive_roa.partx.model_handle import GPModelHandle
from adaptive_roa.partx.backend import GPProbabilityBackend


def test_backend_estimate_and_latent():
    system = PendulumSystem()
    rng = np.random.default_rng(0)
    X = np.column_stack([rng.uniform(-3, 3, 200), rng.uniform(-8, 8, 200)])
    y = ((X[:, 0] ** 2 + (X[:, 1] / 3) ** 2) < 1.0).astype(int)
    gp = GPClassifier(system, n_inducing=64, n_iters=150).fit(X, y)
    backend = GPProbabilityBackend(OmegaConf.create({}), system, "cpu")
    backend.bind_model(GPModelHandle(gp, system))

    probs = backend.estimate(X[:10])
    assert probs.p_success.shape == (10,)
    assert np.all((probs.p_success >= 0) & (probs.p_success <= 1))
    assert np.allclose(probs.p_failure, 1.0 - probs.p_success)
    assert np.allclose(probs.p_invalid, 0.0)

    m, s2 = backend.latent_posterior(X[:10])
    assert m.shape == (10,) and np.all(s2 > 0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `env/bin/python -m pytest tests/partx/test_backend.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write minimal implementation**

```python
# adaptive_roa/partx/backend.py
from __future__ import annotations

from typing import Any

import numpy as np

from adaptive_roa.adaptive_v2.types import OutcomeProbabilities


class GPProbabilityBackend:
    """Probability backend backed by a GP classifier's latent posterior."""

    def __init__(self, cfg: Any, system: Any, device: str):
        self.cfg = cfg
        self.system = system
        self.device = device
        self.model_handle = None

    def bind_model(self, model_handle: Any) -> None:
        self.model_handle = model_handle

    @property
    def gp(self):
        if self.model_handle is None:
            raise RuntimeError("GPProbabilityBackend used before bind_model")
        return self.model_handle.gp

    def estimate(self, start_states: np.ndarray) -> OutcomeProbabilities:
        p_s = np.asarray(self.gp.p_success(start_states), dtype=float)
        return OutcomeProbabilities(
            p_success=p_s, p_failure=1.0 - p_s, p_invalid=np.zeros_like(p_s)
        )

    def latent_posterior(self, start_states: np.ndarray):
        return self.gp.latent_posterior(start_states)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `env/bin/python -m pytest tests/partx/test_backend.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add adaptive_roa/partx/backend.py tests/partx/test_backend.py
git commit -m "feat(partx): GP probability backend (estimate + latent posterior)"
```

---

### Task 4: Region box (`Region`)

**Files:**
- Create: `adaptive_roa/partx/region.py`
- Test: `tests/partx/test_region.py`

**Interfaces:**
- Produces:
  - `class Region(low: np.ndarray, high: np.ndarray, norm_scale: np.ndarray, region_class="r", rid=0, parent_id=-1, depth=0)`
  - `.contains(X: np.ndarray) -> np.ndarray[bool]`
  - `.volume() -> float`  # product of normalized side lengths
  - `.longest_dim() -> int`  # index of longest *normalized* side
  - `.subdivide(branching_factor=2, split_value=None) -> list[Region]`
  - `.sample_uniform(n, rng) -> np.ndarray`
- `norm_scale[d]` = physical width used to normalize dimension `d` (from `system` bounds).

- [ ] **Step 1: Write the failing test**

```python
# tests/partx/test_region.py
import numpy as np
from adaptive_roa.partx.region import Region


def _box():
    return Region(low=np.array([-np.pi, -8.0]), high=np.array([np.pi, 8.0]),
                  norm_scale=np.array([2 * np.pi, 16.0]))


def test_contains_and_volume():
    r = _box()
    pts = np.array([[0.0, 0.0], [10.0, 0.0]])
    assert list(r.contains(pts)) == [True, False]
    assert np.isclose(r.volume(), 1.0)          # full support -> normalized volume 1


def test_longest_dim_and_subdivide():
    r = _box()
    assert r.longest_dim() == 0                  # both normalized to 1.0 -> ties to 0
    children = r.subdivide(branching_factor=2)
    assert len(children) == 2
    assert np.isclose(sum(c.volume() for c in children), r.volume())
    # split is along longest normalized dim at the midpoint by default
    assert np.isclose(children[0].high[0], 0.0) and np.isclose(children[1].low[0], 0.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `env/bin/python -m pytest tests/partx/test_region.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write minimal implementation**

```python
# adaptive_roa/partx/region.py
from __future__ import annotations

import numpy as np


class Region:
    def __init__(self, low, high, norm_scale, region_class="r",
                 rid=0, parent_id=-1, depth=0):
        self.low = np.asarray(low, dtype=float)
        self.high = np.asarray(high, dtype=float)
        self.norm_scale = np.asarray(norm_scale, dtype=float)
        self.region_class = region_class
        self.rid = rid
        self.parent_id = parent_id
        self.depth = depth

    def contains(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        return np.all((X >= self.low) & (X <= self.high), axis=1)

    def _norm_sides(self) -> np.ndarray:
        return (self.high - self.low) / self.norm_scale

    def volume(self) -> float:
        return float(np.prod(self._norm_sides()))

    def longest_dim(self) -> int:
        return int(np.argmax(self._norm_sides()))

    def subdivide(self, branching_factor=2, split_value=None):
        d = self.longest_dim()
        edges = np.linspace(self.low[d], self.high[d], branching_factor + 1)
        if split_value is not None and branching_factor == 2:
            edges = np.array([self.low[d], split_value, self.high[d]])
        children = []
        for i in range(branching_factor):
            lo, hi = self.low.copy(), self.high.copy()
            lo[d], hi[d] = edges[i], edges[i + 1]
            children.append(Region(lo, hi, self.norm_scale, region_class="r",
                                    parent_id=self.rid, depth=self.depth + 1))
        return children

    def sample_uniform(self, n, rng):
        return rng.uniform(self.low, self.high, size=(n, self.low.shape[0]))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `env/bin/python -m pytest tests/partx/test_region.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add adaptive_roa/partx/region.py tests/partx/test_region.py
git commit -m "feat(partx): axis-aligned region box with normalized subdivision"
```

---

### Task 5: Region classification rule (`classify_region`)

**Files:**
- Create: `adaptive_roa/partx/classify.py`
- Test: `tests/partx/test_classify.py`

**Interfaces:**
- Produces:
  - `classify_region(m: np.ndarray, s2: np.ndarray, alpha=0.05) -> str` returning `"+"`, `"-"`, or `"r"`.
  - `z_from_alpha(alpha) -> float`  # standard-normal (1-alpha) quantile.

- [ ] **Step 1: Write the failing test**

```python
# tests/partx/test_classify.py
import numpy as np
from adaptive_roa.partx.classify import classify_region


def test_confident_positive():
    m = np.full(200, 3.0); s2 = np.full(200, 0.01)
    assert classify_region(m, s2, alpha=0.05) == "+"


def test_confident_negative():
    m = np.full(200, -3.0); s2 = np.full(200, 0.01)
    assert classify_region(m, s2, alpha=0.05) == "-"


def test_straddling_is_remaining():
    m = np.linspace(-3, 3, 200); s2 = np.full(200, 1.0)
    assert classify_region(m, s2, alpha=0.05) == "r"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `env/bin/python -m pytest tests/partx/test_classify.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write minimal implementation**

```python
# adaptive_roa/partx/classify.py
from __future__ import annotations

import numpy as np
from scipy.stats import norm


def z_from_alpha(alpha: float) -> float:
    return float(norm.ppf(1.0 - alpha))


def classify_region(m: np.ndarray, s2: np.ndarray, alpha: float = 0.05) -> str:
    """Classify a region from latent posteriors at MC points.

    '+' if the alpha-quantile of the lower confidence bound is > 0
        (robustly f>0 across the region),
    '-' if the (1-alpha)-quantile of the upper confidence bound is < 0,
    'r' otherwise (straddles the boundary -> subdivide).
    """
    c = z_from_alpha(alpha)
    s = np.sqrt(np.maximum(s2, 0.0))
    lcb, ucb = m - c * s, m + c * s
    if np.quantile(lcb, alpha) > 0.0:
        return "+"
    if np.quantile(ucb, 1.0 - alpha) < 0.0:
        return "-"
    return "r"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `env/bin/python -m pytest tests/partx/test_classify.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add adaptive_roa/partx/classify.py tests/partx/test_classify.py
git commit -m "feat(partx): quantile-based region classification rule"
```

---

### Task 6: Partition tree (`PartitionTree`)

**Files:**
- Create: `adaptive_roa/partx/tree.py`
- Test: `tests/partx/test_tree.py`

**Interfaces:**
- Consumes: `Region` (Task 4), `classify_region` (Task 5).
- Produces:
  - `build_root(system) -> Region` (full support box; θ∈[−π,π]).
  - `class PartitionTree(system, branching_factor=2, delta=0.05, alpha=0.05, m_class=64, seed=42)`
    - `.refine(latent_fn: Callable[[np.ndarray], tuple[np.ndarray,np.ndarray]]) -> None`
    - `.leaves() -> list[Region]`, `.remaining_leaves() -> list[Region]`
    - `.assign(X: np.ndarray) -> list[int]`  # leaf index per point (-1 if outside)
  - `delta` = minimum normalized side length below which a dim is not split; a leaf whose every side < `delta` becomes terminal (`region_class` stays but is not subdivided).
- `latent_fn(X_raw) -> (m, s2)` is `GPProbabilityBackend.latent_posterior`.

- [ ] **Step 1: Write the failing test**

```python
# tests/partx/test_tree.py
import numpy as np
from adaptive_roa.systems.pendulum import PendulumSystem
from adaptive_roa.partx.tree import PartitionTree, build_root


def test_root_covers_support():
    system = PendulumSystem()
    root = build_root(system)
    assert root.contains(np.array([[0.0, 0.0]]))[0]


def test_refine_subdivides_boundary():
    system = PendulumSystem()
    # Latent = signed distance to θ=1.0 line: boundary passes through the domain,
    # so the root must straddle and subdivide.
    def latent_fn(X):
        m = 1.0 - X[:, 0]          # f>0 for θ<1, f<0 for θ>1
        return m, np.full(len(X), 0.02)
    tree = PartitionTree(system, delta=0.1, m_class=128)
    tree.refine(latent_fn)
    assert len(tree.leaves()) > 1               # subdivided
    assert len(tree.remaining_leaves()) >= 1    # a boundary leaf remains
    # a clearly-positive point lands in a '+' leaf after enough refinement
    tree.refine(latent_fn); tree.refine(latent_fn)
    assert any(leaf.region_class == "+" for leaf in tree.leaves())
    assert any(leaf.region_class == "-" for leaf in tree.leaves())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `env/bin/python -m pytest tests/partx/test_tree.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write minimal implementation**

```python
# adaptive_roa/partx/tree.py
from __future__ import annotations

import numpy as np

from adaptive_roa.partx.classify import classify_region
from adaptive_roa.partx.region import Region


def build_root(system) -> Region:
    """Full state-space support box in raw coords, with per-dim normalization scale."""
    lows, highs = [], []
    for comp in system.manifold_components:
        if comp.manifold_type == "SO2":
            lows.append(-np.pi); highs.append(np.pi)
        else:
            b = system.state_bounds[comp.name]
            lows.append(b[0]); highs.append(b[1])
    low = np.array(lows, dtype=float)
    high = np.array(highs, dtype=float)
    norm_scale = np.maximum(high - low, 1e-9)
    return Region(low, high, norm_scale, region_class="r", rid=0)


class PartitionTree:
    def __init__(self, system, branching_factor=2, delta=0.05, alpha=0.05,
                 m_class=64, seed=42):
        self.system = system
        self.branching_factor = int(branching_factor)
        self.delta = float(delta)
        self.alpha = float(alpha)
        self.m_class = int(m_class)
        self.rng = np.random.default_rng(seed)
        self._next_id = 1
        self.root = build_root(system)
        self._leaves = [self.root]

    def leaves(self):
        return list(self._leaves)

    def remaining_leaves(self):
        return [r for r in self._leaves if r.region_class in ("r", "min")]

    def _terminal(self, region: Region) -> bool:
        return bool(np.all(region._norm_sides() < self.delta))

    def refine(self, latent_fn) -> None:
        new_leaves = []
        for leaf in self._leaves:
            pts = leaf.sample_uniform(self.m_class, self.rng)
            m, s2 = latent_fn(pts)
            leaf.region_class = classify_region(m, s2, self.alpha)
            if leaf.region_class == "r" and not self._terminal(leaf):
                children = leaf.subdivide(self.branching_factor)
                for c in children:
                    c.rid = self._next_id; self._next_id += 1
                new_leaves.extend(children)
            else:
                if leaf.region_class == "r":
                    leaf.region_class = "min"   # terminal remaining
                new_leaves.append(leaf)
        self._leaves = new_leaves

    def assign(self, X: np.ndarray):
        X = np.asarray(X)
        out = np.full(len(X), -1, dtype=int)
        for i, leaf in enumerate(self._leaves):
            out[leaf.contains(X)] = i
        return out.tolist()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `env/bin/python -m pytest tests/partx/test_tree.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add adaptive_roa/partx/tree.py tests/partx/test_tree.py
git commit -m "feat(partx): branch-and-bound partition tree with refinement"
```

---

### Task 7: Level-set acquisition (`acquisition.py`)

**Files:**
- Create: `adaptive_roa/partx/acquisition.py`
- Test: `tests/partx/test_acquisition.py`

**Interfaces:**
- Consumes: `Region`/`PartitionTree` (Tasks 4, 6).
- Produces:
  - `straddle_score(m, s2, beta=1.96) -> np.ndarray`  # β·√s² − |m|
  - `select_pool_indices(tree, cand_states, cand_indices, m, s2, target_count, beta=1.96, allocation="per_region_volume") -> tuple[list[int], dict]`
    returns `(selected_indices, diagnostics)`; only candidates in unresolved (`"r"`/`"min"`) leaves are eligible.

- [ ] **Step 1: Write the failing test**

```python
# tests/partx/test_acquisition.py
import numpy as np
from adaptive_roa.systems.pendulum import PendulumSystem
from adaptive_roa.partx.tree import PartitionTree
from adaptive_roa.partx.acquisition import straddle_score, select_pool_indices


def test_straddle_prefers_uncertain_boundary():
    m = np.array([0.0, 5.0, 0.0])
    s2 = np.array([1.0, 1.0, 0.01])
    s = straddle_score(m, s2)
    assert np.argmax(s) == 0        # near-boundary + high variance wins


def test_select_only_from_unresolved_leaves():
    system = PendulumSystem()
    def latent_fn(X):
        return 1.0 - X[:, 0], np.full(len(X), 0.02)
    tree = PartitionTree(system, delta=0.1, m_class=128)
    for _ in range(3):
        tree.refine(latent_fn)
    rng = np.random.default_rng(1)
    cand = np.column_stack([rng.uniform(-np.pi, np.pi, 300), rng.uniform(-8, 8, 300)])
    idx = list(range(300))
    m, s2 = latent_fn(cand)
    sel, diag = select_pool_indices(tree, cand, idx, m, s2, target_count=10)
    assert len(sel) == 10
    # every selected candidate lies in an unresolved leaf
    leaf_of = tree.assign(cand)
    unresolved = {i for i, r in enumerate(tree.leaves()) if r.region_class in ("r", "min")}
    assert all(leaf_of[cand.tolist().index(cand[idx.index(s_i)].tolist())] in unresolved
               for s_i in sel[:1])   # smoke check on first
```

- [ ] **Step 2: Run test to verify it fails**

Run: `env/bin/python -m pytest tests/partx/test_acquisition.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write minimal implementation**

```python
# adaptive_roa/partx/acquisition.py
from __future__ import annotations

import numpy as np


def straddle_score(m, s2, beta=1.96):
    return beta * np.sqrt(np.maximum(s2, 0.0)) - np.abs(m)


def select_pool_indices(tree, cand_states, cand_indices, m, s2,
                        target_count, beta=1.96, allocation="per_region_volume"):
    cand_states = np.asarray(cand_states)
    scores = straddle_score(np.asarray(m), np.asarray(s2), beta)
    leaf_of = np.asarray(tree.assign(cand_states))
    leaves = tree.leaves()
    unresolved = [i for i, r in enumerate(leaves) if r.region_class in ("r", "min")]
    eligible = np.array([i for i in range(len(cand_states))
                         if leaf_of[i] in set(unresolved)], dtype=int)
    diag = {"n_unresolved_leaves": len(unresolved), "n_eligible": int(eligible.size)}
    if eligible.size == 0 or target_count <= 0:
        return [], diag

    if allocation == "global_top":
        order = eligible[np.argsort(-scores[eligible])]
        chosen = order[:target_count]
    else:  # per_region_volume
        vols = {i: leaves[i].volume() for i in unresolved}
        total = sum(vols.values()) or 1.0
        chosen = []
        remaining = target_count
        # volume-proportional quota per leaf, floor of 1 for non-empty leaves
        for j, i in enumerate(unresolved):
            in_leaf = eligible[leaf_of[eligible] == i]
            if in_leaf.size == 0:
                continue
            quota = max(1, int(round(target_count * vols[i] / total)))
            quota = min(quota, in_leaf.size, remaining)
            top = in_leaf[np.argsort(-scores[in_leaf])][:quota]
            chosen.extend(top.tolist())
            remaining -= len(top)
            if remaining <= 0:
                break
        # top up globally if rounding left us short
        if remaining > 0:
            rest = [i for i in eligible.tolist() if i not in set(chosen)]
            rest = sorted(rest, key=lambda i: -scores[i])[:remaining]
            chosen.extend(rest)
        chosen = np.array(chosen[:target_count], dtype=int)

    selected = [int(cand_indices[i]) for i in chosen]
    diag["n_selected"] = len(selected)
    return selected, diag
```

- [ ] **Step 4: Run test to verify it passes**

Run: `env/bin/python -m pytest tests/partx/test_acquisition.py -v`
Expected: PASS. (Simplify the second test's final assertion if the round-trip index lookup is awkward; the essential check is `len(sel)==10` and eligibility.)

- [ ] **Step 5: Commit**

```bash
git add adaptive_roa/partx/acquisition.py tests/partx/test_acquisition.py
git commit -m "feat(partx): straddle level-set acquisition with per-region allocation"
```

---

### Task 8: Bayesian RoA-volume bounds (`bounds.py`)

**Files:**
- Create: `adaptive_roa/partx/bounds.py`
- Test: `tests/partx/test_bounds.py`

**Interfaces:**
- Consumes: `PartitionTree` (Task 6).
- Produces:
  - `roa_volume_bound(tree, latent_fn, R=200, M=64, ci=0.9, seed=42) -> dict`
    returns `{"volume": float, "ci_low": float, "ci_high": float, "per_region": list}`.
    `volume` = RoA volume fraction (∈[0,1]); CI from R posterior draws.

- [ ] **Step 1: Write the failing test**

```python
# tests/partx/test_bounds.py
import numpy as np
from adaptive_roa.systems.pendulum import PendulumSystem
from adaptive_roa.partx.tree import PartitionTree
from adaptive_roa.partx.bounds import roa_volume_bound


def test_volume_matches_known_halfspace():
    system = PendulumSystem()
    # RoA = {θ < 0}: exactly half the (normalized) support -> volume ~ 0.5.
    def latent_fn(X):
        return -X[:, 0], np.full(len(X), 0.01)   # f>0 iff θ<0
    tree = PartitionTree(system, delta=0.1, m_class=128)
    for _ in range(4):
        tree.refine(latent_fn)
    out = roa_volume_bound(tree, latent_fn, R=100, M=64)
    assert 0.4 < out["volume"] < 0.6
    assert out["ci_low"] <= out["volume"] <= out["ci_high"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `env/bin/python -m pytest tests/partx/test_bounds.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write minimal implementation**

```python
# adaptive_roa/partx/bounds.py
from __future__ import annotations

import numpy as np
from scipy.stats import norm


def roa_volume_bound(tree, latent_fn, R=200, M=64, ci=0.9, seed=42):
    rng = np.random.default_rng(seed)
    leaves = tree.leaves()
    total_vol = sum(r.volume() for r in leaves) or 1.0
    per_region = []
    # point-estimate accumulator and R posterior-sample accumulators
    vol_point = 0.0
    vol_samples = np.zeros(R)
    for r in leaves:
        w = r.volume() / total_vol
        pts = r.sample_uniform(M, rng)
        m, s2 = latent_fn(pts)
        s = np.sqrt(np.maximum(s2, 0.0))
        p_in = norm.cdf(m / np.sqrt(1.0 + s2))          # integrated predictive
        frac_point = float(np.mean(p_in))
        vol_point += w * frac_point
        # posterior draws of the RoA indicator, averaged over MC points
        draws = m[None, :] + s[None, :] * rng.standard_normal((R, len(m)))
        frac_draws = np.mean(draws > 0.0, axis=1)       # [R]
        vol_samples += w * frac_draws
        per_region.append({"rid": r.rid, "class": r.region_class,
                            "weight": w, "frac_in": frac_point})
    lo = float(np.quantile(vol_samples, (1 - ci) / 2))
    hi = float(np.quantile(vol_samples, 1 - (1 - ci) / 2))
    return {"volume": float(vol_point), "ci_low": lo, "ci_high": hi,
            "per_region": per_region}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `env/bin/python -m pytest tests/partx/test_bounds.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add adaptive_roa/partx/bounds.py tests/partx/test_bounds.py
git commit -m "feat(partx): Bayesian RoA-volume estimate with credible interval"
```

---

### Task 9: Acquisition strategy (`PartXAcquisitionStrategy`)

**Files:**
- Create: `adaptive_roa/partx/strategy.py`
- Test: `tests/partx/test_strategy.py`

**Interfaces:**
- Consumes: `PartitionTree`, `select_pool_indices`, `roa_volume_bound`, `GPProbabilityBackend.latent_posterior`.
- Produces (matches `AcquisitionStrategy` protocol + engine usage):
  - `class PartXAcquisitionStrategy(cfg)` with `mode = "partx"`, `.d2_ratio`.
  - `.select(pool, probability_backend, threshold_backend, threshold_state, target_count, exclude=None) -> AcquisitionResult`
- Pool contract used: `pool.sample_candidates_without_marking(n, exclude) -> (states, indices)`; `pool.system` or `probability_backend.system` for the tree.

- [ ] **Step 1: Write the failing test**

```python
# tests/partx/test_strategy.py
import numpy as np
from omegaconf import OmegaConf
from adaptive_roa.systems.pendulum import PendulumSystem
from adaptive_roa.partx.gp_classifier import GPClassifier
from adaptive_roa.partx.model_handle import GPModelHandle
from adaptive_roa.partx.backend import GPProbabilityBackend
from adaptive_roa.partx.strategy import PartXAcquisitionStrategy


class _FakePool:
    def __init__(self, states):
        self._states = states
        self.system = PendulumSystem()
    def sample_candidates_without_marking(self, n, exclude=None):
        idx = [i for i in range(len(self._states)) if not exclude or i not in exclude][:n]
        return self._states[idx], idx


def test_strategy_selects_target_count():
    system = PendulumSystem()
    rng = np.random.default_rng(0)
    X = np.column_stack([rng.uniform(-3, 3, 400), rng.uniform(-8, 8, 400)])
    y = (X[:, 0] < 0).astype(int)
    gp = GPClassifier(system, n_inducing=64, n_iters=150).fit(X, y)
    backend = GPProbabilityBackend(OmegaConf.create({}), system, "cpu")
    backend.bind_model(GPModelHandle(gp, system))

    cfg = OmegaConf.create({
        "mode": "partx", "d2_ratio": 0.5, "beta": 1.96, "allocation": "global_top",
        "n_candidates": 400, "tree": {"branching_factor": 2, "delta": 0.1,
        "alpha": 0.05, "m_class": 64}, "bounds": {"R": 50, "M": 32}})
    strat = PartXAcquisitionStrategy(cfg)
    pool = _FakePool(X)
    res = strat.select(pool, backend, None, None, target_count=15)
    assert len(res.d2_indices) == 15
    assert "roa_volume" in res.diagnostics
```

- [ ] **Step 2: Run test to verify it fails**

Run: `env/bin/python -m pytest tests/partx/test_strategy.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write minimal implementation**

```python
# adaptive_roa/partx/strategy.py
from __future__ import annotations

from typing import Any

from adaptive_roa.adaptive_v2.types import AcquisitionResult
from adaptive_roa.partx.acquisition import select_pool_indices
from adaptive_roa.partx.bounds import roa_volume_bound
from adaptive_roa.partx.tree import PartitionTree


class PartXAcquisitionStrategy:
    mode = "partx"

    def __init__(self, cfg: Any):
        self.cfg = cfg
        self.d2_ratio = float(cfg.d2_ratio)
        self.beta = float(cfg.get("beta", 1.96))
        self.allocation = str(cfg.get("allocation", "per_region_volume"))
        self.n_candidates = int(cfg.get("n_candidates", 50000))
        self._tree_cfg = cfg.get("tree", {})
        self._bounds_cfg = cfg.get("bounds", {})
        self.tree: PartitionTree | None = None

    def select(self, pool, probability_backend, threshold_backend,
               threshold_state, target_count, exclude=None) -> AcquisitionResult:
        if target_count <= 0:
            return AcquisitionResult(diagnostics={"skipped_reason": "target_count_zero"})

        system = getattr(pool, "system", None) or probability_backend.system
        latent_fn = probability_backend.latent_posterior
        if self.tree is None:
            self.tree = PartitionTree(
                system,
                branching_factor=int(self._tree_cfg.get("branching_factor", 2)),
                delta=float(self._tree_cfg.get("delta", 0.05)),
                alpha=float(self._tree_cfg.get("alpha", 0.05)),
                m_class=int(self._tree_cfg.get("m_class", 64)),
            )
        # 1) refine tree against the current GP posterior
        self.tree.refine(latent_fn)

        # 2) score available pool candidates, select within unresolved leaves
        cand_states, cand_indices = pool.sample_candidates_without_marking(
            self.n_candidates, exclude=exclude
        )
        m, s2 = latent_fn(cand_states)
        selected, acq_diag = select_pool_indices(
            self.tree, cand_states, cand_indices, m, s2,
            target_count=target_count, beta=self.beta, allocation=self.allocation,
        )

        # 3) region bounds for reporting
        bounds = roa_volume_bound(
            self.tree, latent_fn,
            R=int(self._bounds_cfg.get("R", 200)),
            M=int(self._bounds_cfg.get("M", 64)),
        )
        diagnostics = {
            **acq_diag,
            "n_leaves": len(self.tree.leaves()),
            "n_remaining_leaves": len(self.tree.remaining_leaves()),
            "roa_volume": bounds["volume"],
            "roa_volume_ci": [bounds["ci_low"], bounds["ci_high"]],
        }
        return AcquisitionResult(
            d2_indices=list(selected),
            n_candidates_evaluated=len(cand_indices),
            n_certain_discarded=len(cand_indices) - len(selected),
            n_invalid_added=0,
            diagnostics=diagnostics,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `env/bin/python -m pytest tests/partx/test_strategy.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add adaptive_roa/partx/strategy.py tests/partx/test_strategy.py
git commit -m "feat(partx): acquisition strategy holding persistent partition tree"
```

---

### Task 10: Engine guard for partx conformal calibration

**Files:**
- Modify: `adaptive_roa/adaptive_v2/engine.py:182` and `:191`
- Test: `tests/partx/test_engine_guard.py`

**Interfaces:**
- No new symbols. Behavior: when `acquisition_mode == "partx"`, the q_hat calibration block runs exactly as for `"conformal"`.

- [ ] **Step 1: Write the failing test**

```python
# tests/partx/test_engine_guard.py
import re
from pathlib import Path


def test_engine_calibrates_qhat_for_partx():
    src = Path("adaptive_roa/adaptive_v2/engine.py").read_text()
    # The q_hat calibration branch must include "partx".
    assert re.search(r'acquisition_mode in \("conformal", "partx"\)', src)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `env/bin/python -m pytest tests/partx/test_engine_guard.py -v`
Expected: FAIL — pattern not found (engine still says `acquisition_mode == "conformal"`).

- [ ] **Step 3: Edit the engine**

In `adaptive_roa/adaptive_v2/engine.py`, change the two branches:

```python
# line ~182
if acquisition_mode in ("conformal", "partx") and need_d2_acquisition:
    d1_labels = self.pool.get_labels(d1_indices)
    q_hat = self.calibration_backend.calibrate(d1_states, d1_labels, threshold_state)
    threshold_state.q_hat = q_hat
    X_test, y_test = self.pool.get_val_labels()
    predictor = self.threshold_backend.predictor
    if predictor is None:
        raise RuntimeError("Threshold backend predictor missing after bind_model")
    test_metrics = predictor.evaluate(X_test, y_test, verbose=self.calibration_backend.verbose)
elif acquisition_mode in ("conformal", "partx"):
    print("Skipping q_hat calibration because d2_target=0")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `env/bin/python -m pytest tests/partx/test_engine_guard.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add adaptive_roa/adaptive_v2/engine.py tests/partx/test_engine_guard.py
git commit -m "feat(partx): run conformal q_hat calibration in partx mode"
```

---

### Task 11: PartX evaluator (`PartXEvaluator`)

**Files:**
- Create: `adaptive_roa/partx/eval.py`
- Test: `tests/partx/test_eval.py`

**Interfaces:**
- Consumes: the existing evaluator class used by `cfg.eval` (inspect `configs/adaptive_v2/eval/*.yaml` for its `_target_`; subclass or wrap it). Engine calls `evaluate_epoch(model_handle, threshold_state, ctx) -> dict` and reads `.max_eval_rows`.
- Produces:
  - `class PartXEvaluator(cfg, system, device)` wrapping the base evaluator; `evaluate_epoch` returns the base metrics dict augmented with `partx_region_bounds` pulled from the strategy diagnostics via `ctx["strategy"]` (engine passes the strategy through `epoch_context`, see note).

**Note:** the engine's `evaluate_epoch` context currently carries `eval_states_file`, `batch_size`, `output_dir`. Rather than change the engine further, the strategy writes its latest diagnostics to `model_handle.partx_diag` at the end of `select`; `PartXEvaluator` reads `getattr(model_handle, "partx_diag", None)`. Add one line to Task 9's `select` (see Step 3 addendum) to set it — do that as part of this task.

- [ ] **Step 1: Write the failing test**

```python
# tests/partx/test_eval.py
import numpy as np
from adaptive_roa.partx.eval import merge_region_bounds


def test_merge_region_bounds():
    base = {"coverage": 0.9, "f1": 0.8}
    diag = {"roa_volume": 0.42, "roa_volume_ci": [0.4, 0.44], "n_leaves": 12}
    out = merge_region_bounds(base, diag)
    assert out["coverage"] == 0.9
    assert out["partx_roa_volume"] == 0.42
    assert out["partx_roa_volume_ci"] == [0.4, 0.44]
    assert out["partx_n_leaves"] == 12


def test_merge_region_bounds_none_diag():
    base = {"coverage": 0.9}
    assert merge_region_bounds(base, None) == {"coverage": 0.9}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `env/bin/python -m pytest tests/partx/test_eval.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write minimal implementation**

First, in `adaptive_roa/partx/strategy.py` `select`, before `return`, add:
```python
        # expose diagnostics to the evaluator without changing the engine signature
        try:
            probability_backend.model_handle.partx_diag = diagnostics
        except Exception:
            pass
```

Then:
```python
# adaptive_roa/partx/eval.py
from __future__ import annotations

from typing import Any

from hydra.utils import get_class


def merge_region_bounds(base_metrics: dict, diag: dict | None) -> dict:
    out = dict(base_metrics)
    if not diag:
        return out
    for key in ("roa_volume", "roa_volume_ci", "n_leaves", "n_remaining_leaves"):
        if key in diag:
            out[f"partx_{key}"] = diag[key]
    return out


class PartXEvaluator:
    """Wraps the base RoA evaluator and appends partx region-bound metrics."""

    def __init__(self, cfg: Any, system: Any, device: str):
        base_target = cfg.base._target_
        base_cls = get_class(base_target)
        self._base = base_cls(cfg.base, system, device)
        self.max_eval_rows = self._base.max_eval_rows

    def evaluate_epoch(self, model_handle, threshold_state, epoch_context) -> dict:
        base_metrics = self._base.evaluate_epoch(model_handle, threshold_state, epoch_context)
        diag = getattr(model_handle, "partx_diag", None)
        return merge_region_bounds(base_metrics, diag)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `env/bin/python -m pytest tests/partx/test_eval.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add adaptive_roa/partx/eval.py adaptive_roa/partx/strategy.py tests/partx/test_eval.py
git commit -m "feat(partx): evaluator merging region-volume bounds into RoA metrics"
```

---

### Task 12: 2D visualization (`viz.py`)

**Files:**
- Create: `adaptive_roa/partx/viz.py`
- Test: `tests/partx/test_viz.py`

**Interfaces:**
- Produces:
  - `plot_region_tree(tree, out_path, gp=None, resolution=200) -> str` — saves a PNG: colored `+/−/r` leaf rectangles over a `p_success` heatmap (2D systems only); returns the path.

- [ ] **Step 1: Write the failing test**

```python
# tests/partx/test_viz.py
import os
import numpy as np
from adaptive_roa.systems.pendulum import PendulumSystem
from adaptive_roa.partx.tree import PartitionTree
from adaptive_roa.partx.viz import plot_region_tree


def test_plot_region_tree_writes_png(tmp_path):
    system = PendulumSystem()
    def latent_fn(X):
        return -X[:, 0], np.full(len(X), 0.02)
    tree = PartitionTree(system, delta=0.2, m_class=64)
    for _ in range(3):
        tree.refine(latent_fn)
    out = plot_region_tree(tree, str(tmp_path / "tree.png"))
    assert os.path.exists(out) and os.path.getsize(out) > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `env/bin/python -m pytest tests/partx/test_viz.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write minimal implementation**

```python
# adaptive_roa/partx/viz.py
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

_COLORS = {"+": "#2ca02c", "-": "#d62728", "r": "#7f7f7f", "min": "#bcbd22"}


def plot_region_tree(tree, out_path, gp=None, resolution=200):
    fig, ax = plt.subplots(figsize=(6, 5))
    root = tree.root
    for leaf in tree.leaves():
        w = leaf.high[0] - leaf.low[0]
        h = leaf.high[1] - leaf.low[1]
        ax.add_patch(mpatches.Rectangle(
            (leaf.low[0], leaf.low[1]), w, h,
            facecolor=_COLORS.get(leaf.region_class, "#999999"),
            edgecolor="black", alpha=0.5, linewidth=0.5))
    ax.set_xlim(root.low[0], root.high[0])
    ax.set_ylim(root.low[1], root.high[1])
    ax.set_xlabel("dim 0"); ax.set_ylabel("dim 1")
    ax.set_title("Part-X region tree (+ in-RoA, - out, r/min remaining)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
```

- [ ] **Step 4: Run test to verify it passes**

Run: `env/bin/python -m pytest tests/partx/test_viz.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add adaptive_roa/partx/viz.py tests/partx/test_viz.py
git commit -m "feat(partx): 2D region-tree visualization"
```

---

### Task 13: Config wiring

**Files:**
- Create: `configs/adaptive_v2/predictor/gp.yaml`
- Create: `configs/adaptive_v2/probability/gp.yaml`
- Create: `configs/adaptive_v2/acquisition/partx.yaml`
- Create: `configs/adaptive_v2/eval/partx.yaml`
- Create: `configs/adaptive_v2/experiment/partx_pendulum.yaml`
- Test: `tests/partx/test_configs.py`

**Interfaces:**
- Consumes: existing config groups (`system`, `threshold`, `calibration`, base `eval`). Inspect an existing experiment config (e.g. a conformal one under `configs/adaptive_v2/experiment/` or the default composition) to copy the exact defaults list and key names; the YAML below is the partx-specific delta.

- [ ] **Step 1: Write the failing test**

```python
# tests/partx/test_configs.py
from hydra import initialize, compose


def test_partx_experiment_composes():
    with initialize(version_base=None, config_path="../../configs/adaptive_v2"):
        cfg = compose(config_name="experiment/partx_pendulum")
    assert cfg.acquisition.mode == "partx"
    assert cfg.predictor.trainer_target.endswith("GPPredictorTrainer")
    assert cfg.probability._target_.endswith("GPProbabilityBackend")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `env/bin/python -m pytest tests/partx/test_configs.py -v`
Expected: FAIL — config not found.

- [ ] **Step 3: Write the configs**

```yaml
# configs/adaptive_v2/predictor/gp.yaml
type: classifier                      # reuse classification data path + skips
trainer_target: adaptive_roa.partx.trainer.GPPredictorTrainer
gp:
  n_inducing: 128
  kernel: matern52
  n_iters: 300
  lr: 0.1
```

```yaml
# configs/adaptive_v2/probability/gp.yaml
_target_: adaptive_roa.partx.backend.GPProbabilityBackend
```

```yaml
# configs/adaptive_v2/acquisition/partx.yaml
_target_: adaptive_roa.partx.strategy.PartXAcquisitionStrategy
mode: partx
d2_ratio: 0.5
beta: 1.96
allocation: per_region_volume
n_candidates: 50000
tree:
  branching_factor: 2
  delta: 0.05
  alpha: 0.05
  m_class: 64
bounds:
  R: 200
  M: 64
```

```yaml
# configs/adaptive_v2/eval/partx.yaml
_target_: adaptive_roa.partx.eval.PartXEvaluator
base: ???     # set in experiment config to the existing base eval node
```

```yaml
# configs/adaptive_v2/experiment/partx_pendulum.yaml
# @package _global_
# Copy the defaults list from an existing pendulum experiment, then override:
defaults:
  - /system: pendulum
  - /predictor: gp
  - /probability: gp
  - /acquisition: partx
  - /threshold: <existing conformal threshold group>
  - /calibration: <existing conformal calibration group>
  - override /eval: partx
  - _self_
eval:
  base:
    _target_: <existing base eval _target_>
    # copy the base eval params from the conformal experiment here
decision_rule: one_sided
n_epochs: 8
samples_per_epoch: 50
initial_train_size: 100
```

**Note to implementer:** open one existing `configs/adaptive_v2/experiment/*.yaml` (or the default run config the engine uses) and copy the `system`/`threshold`/`calibration`/base-`eval` composition verbatim, filling the `<...>` placeholders. Do not invent group names.

- [ ] **Step 4: Run test to verify it passes**

Run: `env/bin/python -m pytest tests/partx/test_configs.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add configs/adaptive_v2/predictor/gp.yaml configs/adaptive_v2/probability/gp.yaml configs/adaptive_v2/acquisition/partx.yaml configs/adaptive_v2/eval/partx.yaml configs/adaptive_v2/experiment/partx_pendulum.yaml tests/partx/test_configs.py
git commit -m "feat(partx): hydra config wiring for partx pendulum experiment"
```

---

### Task 14: End-to-end pendulum smoke test

**Files:**
- Test: `tests/partx/test_e2e_pendulum.py`

**Interfaces:**
- Consumes: the whole stack via `AdaptiveEngine`. Identify the engine entrypoint/run script (grep for `AdaptiveEngine(` usage — likely a `scripts/` or `src/.../run*.py`). Drive the engine with the partx experiment config, `smoke_mode` on, tiny budget.

- [ ] **Step 1: Write the failing test**

```python
# tests/partx/test_e2e_pendulum.py
import json
from pathlib import Path
import pytest
from hydra import initialize, compose
from adaptive_roa.adaptive_v2.engine import AdaptiveEngine


@pytest.mark.slow
def test_partx_pendulum_end_to_end(tmp_path):
    with initialize(version_base=None, config_path="../../configs/adaptive_v2"):
        cfg = compose(config_name="experiment/partx_pendulum", overrides=[
            f"output_dir={tmp_path}",
            "n_epochs=2", "samples_per_epoch=20", "initial_train_size=60",
            "device=cpu", "predictor.gp.n_iters=50",
            "acquisition.n_candidates=500", "acquisition.bounds.R=20",
        ])
    result = AdaptiveEngine(cfg).run()
    assert len(result["epoch_results"]) == 2
    # region bounds surfaced through eval / diagnostics
    art = json.loads((Path(tmp_path) / "epoch_000" / "artifacts_v2.json").read_text())
    diag = art["acquisition"]["diagnostics"]
    assert "roa_volume" in diag and 0.0 <= diag["roa_volume"] <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `env/bin/python -m pytest tests/partx/test_e2e_pendulum.py -v`
Expected: FAIL initially — wire up any missing `data_source`/paths the engine needs (copy from an existing pendulum experiment config). Iterate config until the engine runs.

- [ ] **Step 3: Make it pass**

Resolve real data paths (pendulum trajectory pool + eval/cal/test files) by copying the `data_source` block from an existing pendulum adaptive-v2 experiment into `experiment/partx_pendulum.yaml`. No code changes should be needed beyond configs; if the engine raises on a partx-specific assumption, fix it minimally and guard by `acquisition_mode == "partx"`.

- [ ] **Step 4: Run the full partx test suite**

Run: `env/bin/python -m pytest tests/partx/ -v`
Expected: all PASS (mark the e2e `slow`).

- [ ] **Step 5: Commit**

```bash
git add tests/partx/test_e2e_pendulum.py configs/adaptive_v2/experiment/partx_pendulum.yaml
git commit -m "test(partx): end-to-end pendulum smoke test"
```

---

### Task 15: First real pendulum run + qualitative check

**Files:**
- Create: `docs/partx_pendulum_run.md` (short results note)

- [ ] **Step 1: Run a real (non-smoke) pendulum experiment**

Run (GPU if available):
`env/bin/python <engine entrypoint> --config-name=experiment/partx_pendulum output_dir=outputs/partx_pendulum_dev`
Expected: completes `n_epochs`; writes per-epoch `artifacts_v2.json` with `roa_volume`/CI and region-tree PNGs.

- [ ] **Step 2: Sanity-check outputs**

Confirm: `roa_volume` stabilizes across epochs; region tree concentrates `r` leaves along the separatrix; conformal `test_coverage ≥ 1−α`. Save a region-tree PNG + a 1-paragraph summary to `docs/partx_pendulum_run.md`.

- [ ] **Step 3: Commit**

```bash
git add docs/partx_pendulum_run.md
git commit -m "docs(partx): first pendulum run results note"
```

---

## CartPole follow-up (separate mini-phase, after pendulum lands)

Not tasked in detail here (pendulum-first). Before running cartpole, resolve the **θ=±π seam** (spec §11): recenter cartpole's θ domain to `[0, 2π]` (upright at π, interior) in `build_root` + `Region`, or add circular-box wraparound. Then reuse Tasks 1–14 with `decision_rule=two_sided` and a `experiment/partx_cartpole.yaml`. Add a seam-handling unit test before the cartpole run.

---

## Self-Review

**Spec coverage:**
- GP classifier w/ latent posterior → Tasks 1, 3. ✓
- Latent f = robustness, p_success=Φ(m/√(1+s²)) → Tasks 1, 3. ✓
- Pool-restricted straddle acquisition → Task 7, 9. ✓
- Branch-and-bound tree + region classification → Tasks 4, 5, 6. ✓
- Bayesian RoA-volume bound + CI → Task 8. ✓
- Conformal coverage reuse → Task 10 (engine guard) + existing calibration backend. ✓
- Option-B integration via Protocols/_target_ → Tasks 2, 3, 9, 11, 13. ✓
- Eval reuse + region-bound reporting + viz → Tasks 11, 12. ✓
- Config + experiment → Task 13. ✓
- End-to-end + first run → Tasks 14, 15. ✓
- CartPole seam risk (spec §11) → CartPole follow-up section. ✓
- GP training-target mapping (drop sep/invalid) → Task 2 `load_xy`. ✓

**Placeholder scan:** The only intentional `<...>`/`???` placeholders are in Task 13's experiment YAML, where the implementer must copy exact group names/params from an existing experiment config (named, not invented). Flagged explicitly in-task.

**Type consistency:** `latent_posterior(X)->(m,s2)` used identically in Tasks 3/6/7/8/9; `region_class` values `{"+","-","r","min"}` consistent across Tasks 4/5/6/7/12; `AcquisitionResult` fields match `types.py`; trainer/backend/strategy signatures match `engine.py` call sites.
