# Probabilistic Classifier + Generalized Per-Point Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a thin `ProbabilisticClassifier` wrapper over the existing probability backends and rewrite the per-point export script to be run-dir-driven, model-agnostic, and to emit four split files (train/val/cal/test) per epoch using each method's natively-supported probabilities.

**Architecture:** A new `adaptive_roa/probabilistic_classifier/` package wraps the existing `ClassifierProbabilityBackend` / `EndpointMCProbabilityBackend` math without touching `OutcomeProbabilities` or any model-agnostic plumbing. A `native_probs` class attribute declares which probability arrays each method produces, and a registry maps the `predictor` config value to a wrapper class. The export script resolves everything from a training run directory and writes only the declared arrays, so the writer has no per-method branches.

**Tech Stack:** Python, NumPy, PyTorch, pandas (fast text reads), pytest. Reuses existing `adaptive_roa` modules: `adaptive_v2.types.OutcomeProbabilities`, `adaptive_v2.eval.mc_cache` (`load_mc_cache`, `compute_mc_predictions`, `save_mc_cache`, `MCCache`), `adaptive.data_source.load_eval_states`, `model.classifier_mlp` (`ClassifierMLP`, `ClassifierModule`).

## Global Constraints

- GT label convention (internal): `1 = success`, `-1 = failure`, `0 = invalid` (FM MC only). Verbatim from spec.
- Native probability arrays per method: classifier → `("p_success",)`; FM → `("p_success", "p_failure", "p_invalid")`. Verbatim from spec.
- Output format: `.npz`, one file per split named `{split}.npz` (`train`, `val`, `cal`, `test`).
- `OutcomeProbabilities` and the `ProbabilityBackend` implementations must NOT be modified.
- Full split, no row cap. Missing checkpoint/split for an (epoch, split) → log and skip that pair, never abort the run.
- The set of arrays written is driven solely by the wrapper's `native_probs`; the writer contains no `if predictor == ...` branches.
- `predictor` config values in this codebase are `"classifier"` and `"generative"` (FM). There is no `model_type` field.

---

### Task 1: Wrapper base class + registry

**Files:**
- Create: `adaptive_roa/probabilistic_classifier/__init__.py`
- Create: `adaptive_roa/probabilistic_classifier/base.py`
- Create: `adaptive_roa/probabilistic_classifier/registry.py`
- Test: `tests/probabilistic_classifier/test_registry.py`

**Interfaces:**
- Consumes: `adaptive_roa.adaptive_v2.types.OutcomeProbabilities` (dataclass with float-array fields `p_success`, `p_failure`, `p_invalid`).
- Produces:
  - `ProbabilisticClassifier` (ABC) with class attrs `predictor_type: str`, `native_probs: tuple[str, ...]`; instance method `predict(states: np.ndarray) -> OutcomeProbabilities`; instance method `predict_cached(run_dir: str, epoch: int, split: str, states: np.ndarray) -> Optional[OutcomeProbabilities]` (base returns `None`); classmethod `load_from_run(run_dir: str, epoch: int, cfg, system, device: str = "cuda") -> ProbabilisticClassifier`.
  - `register_probabilistic_classifier(cls)` (decorator using `cls.predictor_type`) and `get_probabilistic_classifier_class(predictor_type: str) -> type[ProbabilisticClassifier]`.

- [ ] **Step 1: Write the failing test**

`tests/probabilistic_classifier/test_registry.py`:
```python
import numpy as np
import pytest
from adaptive_roa.adaptive_v2.types import OutcomeProbabilities
from adaptive_roa.probabilistic_classifier.base import ProbabilisticClassifier
from adaptive_roa.probabilistic_classifier.registry import (
    register_probabilistic_classifier,
    get_probabilistic_classifier_class,
)


@register_probabilistic_classifier
class _DummyPC(ProbabilisticClassifier):
    predictor_type = "dummy"
    native_probs = ("p_success",)

    def predict(self, states):
        n = len(states)
        return OutcomeProbabilities(
            p_success=np.full(n, 0.5),
            p_failure=np.full(n, 0.5),
            p_invalid=np.zeros(n),
        )

    @classmethod
    def load_from_run(cls, run_dir, epoch, cfg, system, device="cuda"):
        return cls()


def test_registry_resolves_registered_type():
    assert get_probabilistic_classifier_class("dummy") is _DummyPC


def test_registry_raises_on_unknown_type():
    with pytest.raises(KeyError):
        get_probabilistic_classifier_class("does-not-exist")


def test_base_predict_cached_defaults_to_none():
    pc = _DummyPC()
    assert pc.predict_cached("run", 0, "test", np.zeros((3, 2))) is None


def test_native_probs_declared():
    assert _DummyPC.native_probs == ("p_success",)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/probabilistic_classifier/test_registry.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'adaptive_roa.probabilistic_classifier'`

- [ ] **Step 3: Write minimal implementation**

`adaptive_roa/probabilistic_classifier/__init__.py`:
```python
from .base import ProbabilisticClassifier
from .registry import (
    register_probabilistic_classifier,
    get_probabilistic_classifier_class,
)

__all__ = [
    "ProbabilisticClassifier",
    "register_probabilistic_classifier",
    "get_probabilistic_classifier_class",
]
```

`adaptive_roa/probabilistic_classifier/base.py`:
```python
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from adaptive_roa.adaptive_v2.types import OutcomeProbabilities


class ProbabilisticClassifier(ABC):
    """Uniform wrapper over a probability-producing model.

    Subclasses declare ``predictor_type`` (matches the Hydra ``predictor``
    config value) and ``native_probs`` (the OutcomeProbabilities field names
    the method natively produces; these are the arrays the export writes).
    """

    predictor_type: str = ""
    native_probs: tuple[str, ...] = ()

    @abstractmethod
    def predict(self, states: np.ndarray) -> OutcomeProbabilities:
        """Compute probabilities for ``states`` via live model inference."""
        raise NotImplementedError

    def predict_cached(
        self, run_dir: str, epoch: int, split: str, states: np.ndarray
    ) -> Optional[OutcomeProbabilities]:
        """Return cached probabilities for (epoch, split) if available, else None.

        Base implementation has no cache. Subclasses override when a fast path
        (e.g. a precomputed MC cache) exists.
        """
        return None

    @classmethod
    @abstractmethod
    def load_from_run(
        cls, run_dir: str, epoch: int, cfg, system, device: str = "cuda"
    ) -> "ProbabilisticClassifier":
        """Load the model for ``epoch`` from a training run directory."""
        raise NotImplementedError
```

`adaptive_roa/probabilistic_classifier/registry.py`:
```python
from __future__ import annotations

from typing import Type

from .base import ProbabilisticClassifier

_REGISTRY: dict[str, Type[ProbabilisticClassifier]] = {}


def register_probabilistic_classifier(cls: Type[ProbabilisticClassifier]):
    """Class decorator: register ``cls`` under its ``predictor_type``."""
    if not cls.predictor_type:
        raise ValueError(f"{cls.__name__} must set a non-empty predictor_type")
    _REGISTRY[cls.predictor_type] = cls
    return cls


def get_probabilistic_classifier_class(
    predictor_type: str,
) -> Type[ProbabilisticClassifier]:
    if predictor_type not in _REGISTRY:
        raise KeyError(
            f"No probabilistic classifier registered for predictor="
            f"{predictor_type!r}. Registered: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[predictor_type]
```

Also create empty `tests/probabilistic_classifier/__init__.py` if the test suite uses package-style test dirs (check an existing `tests/` subdir; if other test dirs have no `__init__.py`, skip it).

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/probabilistic_classifier/test_registry.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add adaptive_roa/probabilistic_classifier/__init__.py adaptive_roa/probabilistic_classifier/base.py adaptive_roa/probabilistic_classifier/registry.py tests/probabilistic_classifier/
git commit -m "feat: ProbabilisticClassifier base + registry"
```

---

### Task 2: Classifier wrapper

**Files:**
- Create: `adaptive_roa/probabilistic_classifier/classifier.py`
- Test: `tests/probabilistic_classifier/test_classifier_wrapper.py`

**Interfaces:**
- Consumes: `ProbabilisticClassifier`, `register_probabilistic_classifier` (Task 1); `adaptive_roa.model.classifier_mlp.ClassifierMLP`, `ClassifierModule`; `OutcomeProbabilities`.
- Produces:
  - `ClassifierProbabilisticClassifier(ProbabilisticClassifier)` with `predictor_type = "classifier"`, `native_probs = ("p_success",)`, `__init__(self, module, system, device)`.
  - Module-level `clf_forward(module, system, states, device, bs=8192) -> np.ndarray` and `load_clf_module(epoch_dir, system, cfg, device) -> ClassifierModule` (moved verbatim from `scripts/export_probabilities.py`).

- [ ] **Step 1: Write the failing test**

`tests/probabilistic_classifier/test_classifier_wrapper.py`:
```python
import numpy as np
import torch

from adaptive_roa.probabilistic_classifier.classifier import (
    ClassifierProbabilisticClassifier,
)
from adaptive_roa.probabilistic_classifier.registry import (
    get_probabilistic_classifier_class,
)
from adaptive_roa.model.classifier_mlp import ClassifierMLP, ClassifierModule
from adaptive_roa.systems.pendulum import PendulumSystem


def _tiny_module(system):
    dummy = torch.zeros(1, int(system.state_dim))
    input_dim = int(
        system.embed_state_for_model(system.normalize_state(dummy)).shape[-1]
    )
    mlp = ClassifierMLP(input_dim=input_dim, hidden_dims=[8], output_dim=1, dropout=0.0)
    return ClassifierModule(
        mlp=mlp, system=system, pos_weight=torch.tensor(1.0), lr=1e-3, weight_decay=1e-5
    )


def test_registered_under_classifier():
    assert get_probabilistic_classifier_class("classifier") is ClassifierProbabilisticClassifier


def test_native_probs_is_p_success_only():
    assert ClassifierProbabilisticClassifier.native_probs == ("p_success",)


def test_predict_shapes_and_ranges():
    system = PendulumSystem()
    module = _tiny_module(system).eval()
    pc = ClassifierProbabilisticClassifier(module, system, device="cpu")
    states = np.random.randn(5, int(system.state_dim)).astype(np.float32)
    out = pc.predict(states)
    assert out.p_success.shape == (5,)
    assert np.all((out.p_success >= 0.0) & (out.p_success <= 1.0))
    assert np.allclose(out.p_failure, 1.0 - out.p_success)
    assert np.allclose(out.p_invalid, 0.0)


def test_predict_cached_returns_none():
    system = PendulumSystem()
    pc = ClassifierProbabilisticClassifier(_tiny_module(system).eval(), system, device="cpu")
    assert pc.predict_cached("run", 0, "test", np.zeros((2, 2))) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/probabilistic_classifier/test_classifier_wrapper.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'adaptive_roa.probabilistic_classifier.classifier'`

- [ ] **Step 3: Write minimal implementation**

`adaptive_roa/probabilistic_classifier/classifier.py`:
```python
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import torch

from adaptive_roa.adaptive_v2.types import OutcomeProbabilities
from adaptive_roa.model.classifier_mlp import ClassifierMLP, ClassifierModule
from .base import ProbabilisticClassifier
from .registry import register_probabilistic_classifier


def clf_forward(module, system, states, device, bs=8192):
    out = []
    module.eval()
    with torch.no_grad():
        for i in range(0, len(states), bs):
            x = torch.as_tensor(states[i:i + bs], dtype=torch.float32, device=device)
            logits = module(x).squeeze(-1)
            out.append(torch.sigmoid(logits).double().cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def load_clf_module(epoch_dir, system, cfg, device):
    cls = cfg.get("classifier", {})
    dummy = torch.zeros(1, int(system.state_dim))
    input_dim = int(system.embed_state_for_model(system.normalize_state(dummy)).shape[-1])
    mlp = ClassifierMLP(
        input_dim=input_dim,
        hidden_dims=list(cls.get("hidden_dims", [256, 512, 256])),
        output_dim=1,
        dropout=float(cls.get("dropout", 0.0)),
    )
    module = ClassifierModule(
        mlp=mlp, system=system, pos_weight=torch.tensor(1.0), lr=1e-3, weight_decay=1e-5
    )
    best = glob.glob(str(Path(epoch_dir) / "checkpoints" / "best*.ckpt"))
    if not best:
        raise FileNotFoundError(f"no checkpoint in {epoch_dir}")
    ck = torch.load(best[0], map_location="cpu", weights_only=False)
    missing, _ = module.load_state_dict(ck["state_dict"], strict=False)
    bad = [k for k in missing if k.startswith("mlp.")]
    if bad:
        raise RuntimeError(f"checkpoint missing classifier weights {bad[:3]} in {best[0]}")
    module.eval().to(device)
    return module


@register_probabilistic_classifier
class ClassifierProbabilisticClassifier(ProbabilisticClassifier):
    predictor_type = "classifier"
    native_probs = ("p_success",)

    def __init__(self, module, system, device):
        self.module = module
        self.system = system
        self.device = device

    def predict(self, states: np.ndarray) -> OutcomeProbabilities:
        p_success = clf_forward(self.module, self.system, states, self.device)
        return OutcomeProbabilities(
            p_success=p_success,
            p_failure=1.0 - p_success,
            p_invalid=np.zeros_like(p_success),
        )

    @classmethod
    def load_from_run(cls, run_dir, epoch, cfg, system, device="cuda"):
        epoch_dir = str(Path(run_dir) / f"epoch_{epoch:03d}")
        module = load_clf_module(epoch_dir, system, cfg, device)
        return cls(module, system, device)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/probabilistic_classifier/test_classifier_wrapper.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add adaptive_roa/probabilistic_classifier/classifier.py tests/probabilistic_classifier/test_classifier_wrapper.py
git commit -m "feat: classifier ProbabilisticClassifier wrapper"
```

---

### Task 3: FM wrapper (with MC-cache fast path)

**Files:**
- Create: `adaptive_roa/probabilistic_classifier/flow_matching.py`
- Test: `tests/probabilistic_classifier/test_fm_wrapper.py`

**Interfaces:**
- Consumes: `ProbabilisticClassifier`, `register_probabilistic_classifier`; `adaptive_roa.adaptive_v2.eval.mc_cache` (`MCCache`, `load_mc_cache`, `compute_mc_predictions`); `OutcomeProbabilities`.
- Produces:
  - `FMProbabilisticClassifier(ProbabilisticClassifier)` with `predictor_type = "generative"`, `native_probs = ("p_success", "p_failure", "p_invalid")`, `__init__(self, flow_matcher, system, device, attractor_radius, num_mc_samples)`.
  - Overrides `predict_cached` to read `mc_cache/epoch_{epoch:03d}_{split}.npz`.
  - Module-level `resolve_fm_class(cfg) -> type` and `resolve_radius_mc(run_dir, cfg) -> tuple[float, int]`.

- [ ] **Step 1: Write the failing test**

`tests/probabilistic_classifier/test_fm_wrapper.py`:
```python
import numpy as np

from adaptive_roa.probabilistic_classifier.flow_matching import (
    FMProbabilisticClassifier,
)
from adaptive_roa.probabilistic_classifier.registry import (
    get_probabilistic_classifier_class,
)
from adaptive_roa.adaptive_v2.eval.mc_cache import MCCache, save_mc_cache


def test_registered_under_generative():
    assert get_probabilistic_classifier_class("generative") is FMProbabilisticClassifier


def test_native_probs_includes_invalid():
    assert FMProbabilisticClassifier.native_probs == ("p_success", "p_failure", "p_invalid")


def test_predict_cached_reads_mc_cache(tmp_path):
    # N=2, K=4 MC labels: row0 -> 3 success / 1 invalid; row1 -> 2 fail / 2 success
    start = np.zeros((2, 2), dtype=np.float32)
    endpoints = np.zeros((2, 4, 2), dtype=np.float32)
    labels = np.array([[1, 1, 1, 0], [-1, -1, 1, 1]], dtype=np.int64)
    cache = MCCache(
        start_states=start,
        mc_endpoints=endpoints,
        mc_labels=labels,
        attractor_radius=0.2,
        num_mc_samples=4,
    )
    cache_dir = tmp_path / "mc_cache"
    cache_dir.mkdir()
    save_mc_cache(cache, str(cache_dir / "epoch_000_test.npz"))

    pc = FMProbabilisticClassifier(
        flow_matcher=None, system=None, device="cpu",
        attractor_radius=0.2, num_mc_samples=4,
    )
    out = pc.predict_cached(str(tmp_path), 0, "test", start)
    assert out is not None
    assert np.allclose(out.p_success, [0.75, 0.5])
    assert np.allclose(out.p_failure, [0.0, 0.5])
    assert np.allclose(out.p_invalid, [0.25, 0.0])


def test_predict_cached_missing_returns_none(tmp_path):
    pc = FMProbabilisticClassifier(
        flow_matcher=None, system=None, device="cpu",
        attractor_radius=0.2, num_mc_samples=4,
    )
    assert pc.predict_cached(str(tmp_path), 7, "val", np.zeros((1, 2))) is None
```

NOTE before writing: open `adaptive_roa/adaptive_v2/eval/mc_cache.py` and confirm the `MCCache` constructor field names (`start_states`, `mc_endpoints`, `mc_labels`, `attractor_radius`, `num_mc_samples`) and that `save_mc_cache(cache, path)` / `load_mc_cache(path)` and `cache.probabilities()` exist with these signatures. If a field name differs, match it exactly in both the test and implementation.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/probabilistic_classifier/test_fm_wrapper.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'adaptive_roa.probabilistic_classifier.flow_matching'`

- [ ] **Step 3: Write minimal implementation**

`adaptive_roa/probabilistic_classifier/flow_matching.py`:
```python
from __future__ import annotations

import glob
import importlib
from pathlib import Path

import numpy as np

from adaptive_roa.adaptive_v2.types import OutcomeProbabilities
from adaptive_roa.adaptive_v2.eval.mc_cache import (
    load_mc_cache,
    compute_mc_predictions,
)
from .base import ProbabilisticClassifier
from .registry import register_probabilistic_classifier

# system._target_ class name -> (FMModule, FMClass)
_FM_BY_SYSTEM = {
    "PendulumSystem": (
        "adaptive_roa.flow_matching.pendulum.latent_conditional.flow_matcher",
        "PendulumLatentConditionalFlowMatcher",
    ),
    "CartPoleSystem": (
        "adaptive_roa.flow_matching.cartpole.latent_conditional.flow_matcher",
        "CartPoleLatentConditionalFlowMatcher",
    ),
    "Quadrotor2DSystem": (
        "adaptive_roa.flow_matching.quadrotor_2d.latent_conditional.flow_matcher",
        "Quadrotor2DLatentConditionalFlowMatcher",
    ),
    "Quadrotor3DSystem": (
        "adaptive_roa.flow_matching.quadrotor_3d.latent_conditional.flow_matcher",
        "Quadrotor3DLatentConditionalFlowMatcher",
    ),
}


def resolve_fm_class(cfg):
    target = cfg.get("system", {}).get("_target_", "")
    cn = target.rsplit(".", 1)[-1]
    if cn not in _FM_BY_SYSTEM:
        raise KeyError(f"No flow-matcher class registered for system {cn!r}")
    mod, name = _FM_BY_SYSTEM[cn]
    return getattr(importlib.import_module(mod), name)


def resolve_radius_mc(run_dir, cfg):
    """Radius + MC-sample count: prefer a test mc_cache, else fall back to cfg."""
    caches = sorted(glob.glob(str(Path(run_dir) / "mc_cache" / "*_test.npz")))
    if caches:
        c = load_mc_cache(caches[0])
        return float(c.attractor_radius), int(c.num_mc_samples)
    conf = cfg.get("conformal", {})
    ev = cfg.get("evaluation", {})
    radius = float(conf.get("attractor_radius", ev.get("attractor_radius", 0.2)))
    nmc = int(conf.get("num_mc_samples", ev.get("num_samples", 20)))
    return radius, nmc


@register_probabilistic_classifier
class FMProbabilisticClassifier(ProbabilisticClassifier):
    predictor_type = "generative"
    native_probs = ("p_success", "p_failure", "p_invalid")

    def __init__(self, flow_matcher, system, device, attractor_radius, num_mc_samples):
        self.flow_matcher = flow_matcher
        self.system = system
        self.device = device
        self.attractor_radius = attractor_radius
        self.num_mc_samples = num_mc_samples

    def predict(self, states: np.ndarray) -> OutcomeProbabilities:
        cache = compute_mc_predictions(
            self.flow_matcher, self.system, states,
            num_mc_samples=self.num_mc_samples,
            attractor_radius=self.attractor_radius,
            batch_size=2048, device=self.device,
            verbose=False, refine_invalids=False,
        )
        ps, pf, pinv = cache.probabilities()
        return OutcomeProbabilities(p_success=ps, p_failure=pf, p_invalid=pinv)

    def predict_cached(self, run_dir, epoch, split, states):
        path = Path(run_dir) / "mc_cache" / f"epoch_{epoch:03d}_{split}.npz"
        if not path.exists():
            return None
        cache = load_mc_cache(str(path))
        ps, pf, pinv = cache.probabilities()
        return OutcomeProbabilities(p_success=ps, p_failure=pf, p_invalid=pinv)

    @classmethod
    def load_from_run(cls, run_dir, epoch, cfg, system, device="cuda"):
        fm_class = resolve_fm_class(cfg)
        ckpts = glob.glob(
            str(Path(run_dir) / f"epoch_{epoch:03d}" / "checkpoints" / "best*.ckpt")
        )
        if not ckpts:
            raise FileNotFoundError(
                f"no FM checkpoint in {run_dir}/epoch_{epoch:03d}"
            )
        fm = fm_class.load_from_checkpoint(ckpts[0], device=device)
        radius, nmc = resolve_radius_mc(run_dir, cfg)
        return cls(fm, system, device, radius, nmc)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/probabilistic_classifier/test_fm_wrapper.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add adaptive_roa/probabilistic_classifier/flow_matching.py tests/probabilistic_classifier/test_fm_wrapper.py
git commit -m "feat: FM ProbabilisticClassifier wrapper with mc-cache fast path"
```

---

### Task 4: Run-dir-driven export script (4 splits, native-prob writer)

**Files:**
- Create: `adaptive_roa/probabilistic_classifier/export.py` (library functions)
- Modify: `scripts/export_probabilities.py` (replace hard-coded run maps with a `--run-dir` CLI over the library)
- Test: `tests/probabilistic_classifier/test_export_integration.py`

**Interfaces:**
- Consumes: `get_probabilistic_classifier_class` (Task 1); the registered wrappers (import for side-effect registration); `load_eval_states`; pandas for whitespace reads.
- Produces:
  - `write_split(out_dir, split, query_state, gt_label, probs, native_probs) -> int`
  - `load_split_states(run_dir, split, predictor_type, system, cfg) -> tuple[np.ndarray, Optional[np.ndarray]]` returning `(query_state, gt_label_or_None)` where `gt_label` is `None` for FM train/val (labels derived later from endpoints + radius).
  - `export_run(run_dir, out_dir, device="cuda", epochs=None) -> dict` writing per-epoch `{split}.npz` + a run-level `metadata.json`, returning per-epoch row counts.
  - `resolve_system(cfg)` returning an instantiated system.

- [ ] **Step 1: Write the failing test**

`tests/probabilistic_classifier/test_export_integration.py` (classifier path — fast, no FM model):
```python
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from adaptive_roa.model.classifier_mlp import ClassifierMLP, ClassifierModule
from adaptive_roa.systems.pendulum import PendulumSystem
from adaptive_roa.probabilistic_classifier.export import export_run


def _save_clf_ckpt(epoch_dir, system):
    dummy = torch.zeros(1, int(system.state_dim))
    input_dim = int(system.embed_state_for_model(system.normalize_state(dummy)).shape[-1])
    mlp = ClassifierMLP(input_dim=input_dim, hidden_dims=[8], output_dim=1, dropout=0.0)
    module = ClassifierModule(mlp=mlp, system=system, pos_weight=torch.tensor(1.0), lr=1e-3, weight_decay=1e-5)
    ckpt_dir = Path(epoch_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    torch.save({"state_dict": module.state_dict()}, ckpt_dir / "best.ckpt")


def _write_clf_rows(path, n, sd):
    rows = np.random.randn(n, sd)
    labels = np.random.randint(0, 2, size=(n, 1))
    np.savetxt(path, np.hstack([rows, labels]), delimiter=" ")


def _write_eval_states(path, n, sd):
    start = np.random.randn(n, sd)
    end = np.random.randn(n, sd)
    labels = np.random.randint(0, 2, size=(n, 1))
    np.savetxt(path, np.hstack([start, end, labels]), delimiter=",")


def test_export_run_classifier_writes_four_splits(tmp_path):
    system = PendulumSystem()
    sd = int(system.state_dim)
    run_dir = tmp_path / "run"
    (run_dir / "datasets").mkdir(parents=True)
    _save_clf_ckpt(run_dir / "epoch_000", system)
    _write_clf_rows(run_dir / "datasets" / "train_classification_dataset.txt", 6, sd)
    _write_clf_rows(run_dir / "datasets" / "val_classification_dataset.txt", 4, sd)
    cal_file = tmp_path / "cal_set.txt"
    test_file = tmp_path / "test_set.txt"
    _write_eval_states(cal_file, 5, sd)
    _write_eval_states(test_file, 7, sd)

    cfg = {
        "predictor": "classifier",
        "system": {"_target_": "adaptive_roa.systems.pendulum.PendulumSystem"},
        "classifier": {"hidden_dims": [8], "dropout": 0.0},
        "data_source": {"cal_set_file": str(cal_file), "test_set_file": str(test_file)},
    }
    hydra_dir = run_dir / ".hydra"
    hydra_dir.mkdir()
    (hydra_dir / "config.yaml").write_text(yaml.safe_dump(cfg))

    out_dir = tmp_path / "out"
    export_run(str(run_dir), str(out_dir), device="cpu")

    ep = out_dir / "epoch_000"
    for split, n in [("train", 6), ("val", 4), ("cal", 5), ("test", 7)]:
        d = np.load(ep / f"{split}.npz")
        assert d["query_state"].shape == (n, sd)
        assert d["gt_label"].shape == (n,)
        assert set(np.unique(d["gt_label"])).issubset({-1, 1})
        assert "p_success" in d.files
        assert "p_failure" not in d.files  # classifier native_probs = (p_success,)
        assert np.all((d["p_success"] >= 0.0) & (d["p_success"] <= 1.0))
    meta = json.loads((out_dir / "metadata.json").read_text())
    assert meta["predictor"] == "classifier"
    assert meta["native_probs"] == ["p_success"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/probabilistic_classifier/test_export_integration.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'adaptive_roa.probabilistic_classifier.export'`

- [ ] **Step 3: Write minimal implementation**

`adaptive_roa/probabilistic_classifier/export.py`:
```python
from __future__ import annotations

import glob
import importlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from adaptive_roa.adaptive.data_source import load_eval_states
from adaptive_roa.flow_matching.base.checkpoint_utils import load_hydra_config
from .registry import get_probabilistic_classifier_class
# Import wrappers for registration side-effects.
from . import classifier as _clf  # noqa: F401
from . import flow_matching as _fm  # noqa: F401

_DATASET_KIND = {"classifier": "classification", "generative": "endpoint"}


def _read_ws(path):
    return pd.read_csv(path, header=None, sep=r"\s+").to_numpy()


def resolve_system(cfg):
    target = cfg.get("system", {}).get("_target_", "")
    mod, name = target.rsplit(".", 1)
    return getattr(importlib.import_module(mod), name)()


def load_cfg(run_dir):
    for cand in [run_dir, str(Path(run_dir) / "epoch_000")]:
        cfg = load_hydra_config(Path(cand))
        if cfg is not None:
            return cfg
    raise FileNotFoundError(f"No hydra config for {run_dir}")


def epoch_dirs(run_dir):
    dirs = [d for d in glob.glob(f"{run_dir}/epoch_*") if Path(d).is_dir()]
    return sorted(dirs, key=lambda p: int(re.search(r"epoch_(\d+)", p).group(1)))


def epoch_num(d):
    return int(re.search(r"epoch_(\d+)", d).group(1))


def load_split_states(run_dir, split, predictor_type, system, cfg):
    """Return (query_state[N,D], gt_label[N] or None).

    gt_label is None for FM train/val (derived from endpoints + radius later).
    """
    sd = int(system.state_dim)
    if split in ("cal", "test"):
        key = "cal_set_file" if split == "cal" else "test_set_file"
        path = str(cfg["data_source"][key])
        states, _end, labels = load_eval_states(path)
        return states, labels
    # train / val come from the run-level dataset files
    kind = _DATASET_KIND[predictor_type]
    path = str(Path(run_dir) / "datasets" / f"{split}_{kind}_dataset.txt")
    data = _read_ws(path)
    states = data[:, :sd].astype(np.float32)
    if predictor_type == "classifier":
        labels = np.where(data[:, -1] > 0.5, 1, -1).astype(np.int64)
        return states, labels
    # FM: derive labels from endpoint columns once radius is known (caller fills)
    return states, None


def _fm_labels_from_endpoints(run_dir, split, system, radius):
    sd = int(system.state_dim)
    kind = _DATASET_KIND["generative"]
    path = str(Path(run_dir) / "datasets" / f"{split}_{kind}_dataset.txt")
    end = _read_ws(path)[:, sd:2 * sd].astype(np.float32)
    return system.classify_attractor(
        torch.as_tensor(end, dtype=torch.float32), radius=radius
    ).cpu().numpy().astype(np.int64)


def write_split(out_dir, split, query_state, gt_label, probs, native_probs):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    d = {
        "query_state": np.asarray(query_state, dtype=np.float32),
        "gt_label": np.asarray(gt_label, dtype=np.int64),
    }
    for name in native_probs:
        d[name] = np.asarray(getattr(probs, name), dtype=np.float64)
    np.savez_compressed(Path(out_dir) / f"{split}.npz", **d)
    return len(gt_label)


def export_run(run_dir, out_dir, device="cuda", epochs=None):
    device = device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
    cfg = load_cfg(run_dir)
    predictor_type = str(cfg.get("predictor", "generative"))
    system = resolve_system(cfg)
    pc_class = get_probabilistic_classifier_class(predictor_type)
    native = pc_class.native_probs

    splits = ["train", "val", "cal", "test"]
    split_states = {}
    split_labels = {}
    for split in splits:
        try:
            states, labels = load_split_states(run_dir, split, predictor_type, system, cfg)
            split_states[split] = states
            split_labels[split] = labels
        except (FileNotFoundError, KeyError, OSError) as e:
            print(f"[skip split {split}] {type(e).__name__}: {e}", flush=True)

    eds = epoch_dirs(run_dir)
    if epochs is not None:
        eds = [d for d in eds if epoch_num(d) in set(epochs)]

    counts = {}
    for ed in eds:
        ep = epoch_num(ed)
        odir = Path(out_dir) / f"epoch_{ep:03d}"
        try:
            pc = pc_class.load_from_run(run_dir, ep, cfg, system, device)
        except (FileNotFoundError, RuntimeError, IndexError) as e:
            print(f"[skip epoch {ep:03d}] load failed: {type(e).__name__}: {e}", flush=True)
            counts[ep] = {"error": str(e)}
            continue
        c = {}
        for split in split_states:
            try:
                states = split_states[split]
                labels = split_labels[split]
                if labels is None:  # FM train/val
                    labels = _fm_labels_from_endpoints(
                        run_dir, split, system, pc.attractor_radius
                    )
                probs = pc.predict_cached(run_dir, ep, split, states)
                if probs is None:
                    probs = pc.predict(states)
                c[split] = write_split(odir, split, states, labels, probs, native)
            except Exception as e:  # never abort the whole run on one split
                print(f"[epoch {ep:03d} split {split}] ERROR {type(e).__name__}: {e}", flush=True)
                c[split] = {"error": str(e)}
        counts[ep] = c
        print(f"epoch {ep:03d}: {c}", flush=True)
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()

    meta = {
        "run_dir": str(run_dir),
        "predictor": predictor_type,
        "native_probs": list(native),
        "gt_label_convention": {"1": "success", "-1": "failure", "0": "invalid (FM MC only)"},
        "prob_definitions": (
            "classifier: p_success = sigmoid(logit)"
            if predictor_type == "classifier"
            else "FM: p_{success,failure,invalid} = fraction of MC endpoints with mc_label 1 / -1 / 0"
        ),
        "epoch_counts": counts,
    }
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "metadata.json").write_text(json.dumps(meta, indent=2))
    return counts
```

`scripts/export_probabilities.py` — replace the whole file with a thin CLI over the library (delete the `_SYS`/`FM_SPEC`/`CLF_DIR`/`build_runs` maps and the old per-method loop):
```python
#!/usr/bin/env python3
"""Export per-point (query_state, gt_label, native probs) for train/val/cal/test,
per adaptive step (epoch), driven by a training run directory.

The set of probability arrays written is whatever the run's predictor declares
native: classifier -> p_success ; FM (generative) -> p_success, p_failure, p_invalid.

Usage:
  python scripts/export_probabilities.py --run-dir <run_dir> [--out-dir DIR] \
         [--device cuda] [--epochs 0 1 2]
"""
import argparse

from adaptive_roa.probabilistic_classifier.export import export_run

OUT_ROOT = "/common/users/shared/pracsys/adaptive_roa_experiments/exp_probabilities"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out-dir", default=None,
                    help=f"output dir (default: {OUT_ROOT}/<run basename>)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, nargs="*", default=None)
    args = ap.parse_args()

    out_dir = args.out_dir
    if out_dir is None:
        from pathlib import Path
        out_dir = str(Path(OUT_ROOT) / Path(args.run_dir.rstrip("/")).name)

    counts = export_run(args.run_dir, out_dir, device=args.device, epochs=args.epochs)
    print(f"done. wrote {len(counts)} epochs to {out_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/probabilistic_classifier/test_export_integration.py -v`
Expected: PASS (1 passed)

Then run the full package test suite:
Run: `python -m pytest tests/probabilistic_classifier/ -v`
Expected: PASS (all)

- [ ] **Step 5: Commit**

```bash
git add adaptive_roa/probabilistic_classifier/export.py scripts/export_probabilities.py tests/probabilistic_classifier/test_export_integration.py
git commit -m "feat: run-dir-driven per-point export over ProbabilisticClassifier wrappers"
```

---

## Notes for the implementer

- Confirm `OutcomeProbabilities` field names at `adaptive_roa/adaptive_v2/types.py:11-17` before Task 1 (expected: `p_success`, `p_failure`, `p_invalid`). If they differ, update `native_probs` and the writer's attribute access to match exactly.
- The classifier dataset files are whitespace-separated `[state… binary_label∈{0,1}]`; the cal/test eval-states files are comma-separated `[start… end… label]`. This asymmetry is handled in `load_split_states` — do not unify them.
- FM train/val labels are derived via `system.classify_attractor(endpoints, radius)` because the endpoint dataset's stored label may predate the eval radius; this matches the existing script's behavior.
- The old script exported only test/cal/val and emitted `p_success`+`p_failure` for FM. The new behavior (train added, FM emits all three native arrays) is intentional per the spec; downstream consumers of the old `exp_probabilities` layout must be updated separately if any exist.
```
