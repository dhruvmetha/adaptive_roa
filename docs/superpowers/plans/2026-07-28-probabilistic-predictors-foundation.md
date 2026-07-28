# Probabilistic Predictors: Foundation + Outcome Arms Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add three Bayesian-NN outcome-probability arms (MFVI, deep ensemble, last-layer Laplace) to the adaptive ROA pipeline, and make the existing GP arm usable outside partx acquisition.

**Architecture:** Every arm is a `(backbone, posterior, head)` triple. This plan builds the posterior layer and the outcome head, both bound to the *existing* `ClassifierProbabilityBackend` through a duck-typed model handle that satisfies `handle(raw_states) -> logits`. No new probability backends, acquisition strategies, or evaluators.

**Tech Stack:** PyTorch 2.5.1, Lightning, Hydra, gpytorch 1.15.2, pytest. No new pip dependencies.

**Design spec:** `docs/superpowers/specs/2026-07-27-probabilistic-predictors-design.md`

## Global Constraints

- Branch: `probabilistic-predictors`. Repo root: `/common/home/st1122/Projects/adaptive_roa`.
- Python interpreter is the repo-local env: **`./env/bin/python`**. Never plain `python`.
- Run tests as `./env/bin/python -m pytest <path> -v`. `pytest.ini` sets `testpaths = tests`.
- Import root is `adaptive_roa` (editable install).
- **No new pip dependencies in this plan.**
- **Commit messages must contain zero AI/tool attribution** — no `Co-Authored-By`, no `Claude-Session`, no tool names. This is a hard user preference.
- Backbone default `hidden_dims: [256, 512, 256]`, matching `configs/adaptive_v2/predictor/classifier.yaml`.
- Prior for all Bayesian arms: isotropic Gaussian, `prior_sigma` identical across arms, reported in config.
- Existing behavior must not regress: `predictor=classifier` and `predictor=generative` runs already on disk must still train and still export.

## File Structure

**Create:**
- `adaptive_roa/predictors/__init__.py` — package exports
- `adaptive_roa/predictors/posteriors.py` — `VILinear`, `DeterministicPosterior`, `MFVIPosterior`, `EnsemblePosterior`, `LastLayerLaplacePosterior`
- `adaptive_roa/predictors/bayesian_mlp.py` — `BayesianMLP` (body + head split, so Laplace can reach penultimate features)
- `adaptive_roa/predictors/handles.py` — `OutcomeModelHandle`
- `adaptive_roa/adaptive_v2/trainers/bayesian_mlp_trainer.py` — `BayesianMLPTrainer`
- `adaptive_roa/probabilistic_classifier/bayesian.py` — export wrappers for the three arms
- `configs/adaptive_v2/predictor/{bnn_mfvi,bnn_ensemble,bnn_laplace}.yaml`
- `tests/predictors/{__init__.py,test_posteriors.py,test_bayesian_mlp.py,test_outcome_handle.py,test_bnn_trainer.py}`

**Modify:**
- `adaptive_roa/adaptive_v2/interfaces.py` — fix stale `PredictorTrainer`, add handle Protocols
- `adaptive_roa/partx/backend.py` — add `.estimator`
- `adaptive_roa/probabilistic_classifier/registry.py` — key on arm name
- `adaptive_roa/probabilistic_classifier/export.py:20,110-128` — resolve `predictor.name`, extend `_DATASET_KIND`
- `adaptive_roa/probabilistic_classifier/{classifier,flow_matching}.py` — declare `predictor_name`
- `configs/adaptive_v2/predictor/{classifier,generative,gp,gp_optdelta}.yaml` — add `name`

---

### Task 1: Correct the stale trainer Protocol and document the handle contracts

`adaptive_roa/adaptive_v2/interfaces.py:12-20` declares `fit(train_file, val_file, output_dir, ...)`. No implementation has that signature — all three take `dataset_files: dict`. The handle contracts are currently unwritten convention.

**Files:**
- Modify: `adaptive_roa/adaptive_v2/interfaces.py:12-20`
- Test: `tests/predictors/test_interfaces.py`

**Interfaces:**
- Consumes: nothing
- Produces: `PredictorTrainer`, `OutcomeModelHandle` (Protocol), `FinalStateModelHandle` (Protocol) — later tasks type against these.

- [ ] **Step 1: Create the test package and write the failing test**

Create `tests/predictors/__init__.py` (empty file), then `tests/predictors/test_interfaces.py`:

```python
"""The Protocols in interfaces.py must match what implementations actually do."""
import inspect

from adaptive_roa.adaptive_v2 import interfaces
from adaptive_roa.adaptive_v2.trainers.classifier_trainer import ClassifierTrainer
from adaptive_roa.adaptive_v2.trainers.flow_matching_trainer import FlowMatchingTrainer


def test_predictor_trainer_protocol_matches_implementations():
    """Both shipped trainers take dataset_files: dict, not train_file/val_file."""
    proto_params = list(
        inspect.signature(interfaces.PredictorTrainer.fit).parameters
    )
    assert "dataset_files" in proto_params
    assert "train_file" not in proto_params
    for trainer_cls in (ClassifierTrainer, FlowMatchingTrainer):
        impl_params = list(inspect.signature(trainer_cls.fit).parameters)
        assert impl_params[1] == "dataset_files", trainer_cls.__name__


def test_handle_protocols_exist():
    """Both handle contracts are declared, including the determinism asymmetry."""
    assert hasattr(interfaces, "OutcomeModelHandle")
    assert hasattr(interfaces, "FinalStateModelHandle")
    assert "identical" in interfaces.OutcomeModelHandle.__doc__
    assert "fresh" in interfaces.FinalStateModelHandle.__doc__
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./env/bin/python -m pytest tests/predictors/test_interfaces.py -v`
Expected: FAIL — `assert "dataset_files" in proto_params` fails, and `AttributeError` on `OutcomeModelHandle`.

- [ ] **Step 3: Fix the Protocol and add the handle contracts**

In `adaptive_roa/adaptive_v2/interfaces.py`, replace the `PredictorTrainer` class body (lines 12-20) with:

```python
class PredictorTrainer(Protocol):
    """Trains one epoch's predictor and returns a ready-to-use model handle.

    ``dataset_files`` keys depend on the predictor family and prediction mode:
    ``{"train", "val"}`` for classification and global endpoint data,
    ``{"train_trajectories", "val_trajectories"}`` for ``prediction_mode: local``.
    """

    def fit(
        self,
        dataset_files: dict,
        output_dir: str,
        resume_checkpoint: str | None = None,
    ) -> Any:
        ...
```

Then append these two Protocols to the end of the file:

```python
class OutcomeModelHandle(Protocol):
    """Predictor that emits p(success | x) directly.

    Bound to ``ClassifierProbabilityBackend``, which calls the handle ONCE per
    query. Threshold optimization, calibration, and evaluation each call it
    separately and must agree, so a Bayesian handle marginalizes its weight
    posterior INTERNALLY and returns identical logits for repeated calls on the
    same states (use a ``torch.Generator`` seeded at construction).
    """

    def eval(self) -> Any: ...

    def to(self, device: Any) -> Any: ...

    def __call__(self, raw_states: Any) -> Any:
        """Raw (un-normalized) states [B, state_dim] -> logits [B] or [B, 1]."""
        ...


class FinalStateModelHandle(Protocol):
    """Predictor that emits a distribution over the final state.

    Bound to ``EndpointMCProbabilityBackend``, which calls ``predict_endpoint``
    K times on the SAME batch and counts ``system.classify_attractor`` labels.
    The spread across those calls IS the outcome probability, so each call must
    draw a fresh posterior sample. Seeding this handle collapses every arm to
    p in {0, 1}.
    """

    def eval(self) -> Any: ...

    def to(self, device: Any) -> Any: ...

    def predict_endpoint(self, states: Any) -> Any:
        """One posterior draw: [B, state_dim] -> [B, state_dim], raw space."""
        ...

    def get_manifold_component_names(self) -> list:
        """Called UNGUARDED by adaptive/endpoint_evaluation.py:100."""
        ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./env/bin/python -m pytest tests/predictors/test_interfaces.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Verify nothing else regressed**

Run: `./env/bin/python -m pytest tests/adaptive_v2 -q`
Expected: PASS, same count as before this task.

- [ ] **Step 6: Commit**

```bash
git add adaptive_roa/adaptive_v2/interfaces.py tests/predictors/
git commit -m "fix(interfaces): correct PredictorTrainer signature, declare handle contracts"
```

---

### Task 2: Unblock the GP arm for standard acquisition

`GPProbabilityBackend` has no `.estimator`, but `ranked`, `conformal`, and `direct` all read `probability_backend.estimator` (`adaptive_v2/strategy/ranked.py:44`). GP therefore works only with partx acquisition. `GPModelHandle` already satisfies the classifier contract, so a `ClassifierProbabilityEstimator` over it is all that is missing.

**Files:**
- Modify: `adaptive_roa/partx/backend.py`
- Test: `tests/partx/test_backend.py`

**Interfaces:**
- Consumes: `GPModelHandle` from `adaptive_roa/partx/model_handle.py`
- Produces: `GPProbabilityBackend.estimator` — a `ClassifierProbabilityEstimator`

- [ ] **Step 1: Write the failing test**

Append to `tests/partx/test_backend.py`:

```python
def test_backend_exposes_estimator_for_standard_strategies(tmp_path):
    """ranked/conformal/direct read probability_backend.estimator; GP must have it."""
    import numpy as np
    from adaptive_roa.partx.backend import GPProbabilityBackend
    from adaptive_roa.partx.gp_classifier import GPClassifier
    from adaptive_roa.partx.model_handle import GPModelHandle
    from adaptive_roa.systems.pendulum import PendulumSystem
    from omegaconf import OmegaConf

    system = PendulumSystem()
    rng = np.random.default_rng(0)
    X = rng.uniform(-3.0, 3.0, size=(64, 2)).astype(np.float32)
    y = (np.abs(X[:, 0]) < 1.0).astype(np.float32)

    gp = GPClassifier(system, n_inducing=16, n_iters=5, device="cpu").fit(X, y)
    backend = GPProbabilityBackend(
        OmegaConf.create({"attractor_radius": 0.2}), system, "cpu"
    )
    backend.bind_model(GPModelHandle(gp, system))

    assert backend.estimator is not None
    p_success, p_failure, p_invalid = backend.estimator.estimate(X)
    assert p_success.shape == (64,)
    # Must agree with the backend's own estimate() path.
    np.testing.assert_allclose(p_success, backend.estimate(X).p_success, atol=1e-5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./env/bin/python -m pytest tests/partx/test_backend.py::test_backend_exposes_estimator_for_standard_strategies -v`
Expected: FAIL — `AttributeError: 'GPProbabilityBackend' object has no attribute 'estimator'`

- [ ] **Step 3: Add the estimator**

In `adaptive_roa/partx/backend.py`, add imports at the top:

```python
from adaptive_roa.conformal import ConformalConfig
from adaptive_roa.conformal.classifier_probability_estimator import (
    ClassifierProbabilityEstimator,
)
```

Set `self.estimator = None` in `__init__`, and replace `bind_model` with:

```python
    def bind_model(self, model_handle: Any) -> None:
        self.model_handle = model_handle
        # ranked/conformal/direct acquisition read `.estimator` directly
        # (adaptive_v2/strategy/ranked.py:44). GPModelHandle already satisfies
        # the classifier contract (model(raw_states) -> logits), so the shared
        # classifier estimator works unchanged and keeps GP usable outside partx.
        conf = ConformalConfig(
            attractor_radius=float(self.cfg.attractor_radius),
            delta=0.05, w=0.9, alpha=0.1,
        )
        self.estimator = ClassifierProbabilityEstimator(
            model_handle, self.system, conf, self.device
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./env/bin/python -m pytest tests/partx/test_backend.py -v`
Expected: PASS (all tests in the file)

- [ ] **Step 5: Verify partx e2e still works**

Run: `./env/bin/python -m pytest tests/partx -q`
Expected: PASS, same count as before.

- [ ] **Step 6: Commit**

```bash
git add adaptive_roa/partx/backend.py tests/partx/test_backend.py
git commit -m "feat(partx): expose estimator so GP works with ranked/conformal/direct acquisition"
```

---

### Task 3: Key the export registry on arm name

`_REGISTRY` is keyed on `predictor.type`, which is a *family* tag. All five outcome arms declare `type: classifier`, so without an arm-level key the export layer would load a BNN checkpoint as a `ClassifierMLP` — silently, producing plausible wrong numbers.

**Files:**
- Modify: `adaptive_roa/probabilistic_classifier/registry.py`
- Modify: `adaptive_roa/probabilistic_classifier/export.py:20,110-128`
- Modify: `adaptive_roa/probabilistic_classifier/classifier.py:52-54`
- Modify: `adaptive_roa/probabilistic_classifier/flow_matching.py:61-63`
- Modify: `configs/adaptive_v2/predictor/{classifier,generative,gp,gp_optdelta}.yaml`
- Test: `tests/probabilistic_classifier/test_registry.py`

**Interfaces:**
- Consumes: nothing
- Produces: `register_probabilistic_classifier` keying on `predictor_name`; `resolve_predictor_name(cfg) -> str`; `ProbabilisticClassifier.predictor_name` class attribute.

- [ ] **Step 1: Write the failing test**

Append to `tests/probabilistic_classifier/test_registry.py`:

```python
def test_registry_keys_on_arm_name_not_family():
    from adaptive_roa.probabilistic_classifier import get_probabilistic_classifier_class
    from adaptive_roa.probabilistic_classifier.classifier import (
        ClassifierProbabilisticClassifier,
    )
    assert get_probabilistic_classifier_class("mlp") is ClassifierProbabilisticClassifier


def test_legacy_family_keys_still_resolve():
    """Runs already on disk carry only predictor.type; they must still export."""
    from adaptive_roa.probabilistic_classifier import get_probabilistic_classifier_class
    from adaptive_roa.probabilistic_classifier.classifier import (
        ClassifierProbabilisticClassifier,
    )
    from adaptive_roa.probabilistic_classifier.flow_matching import (
        FMProbabilisticClassifier,
    )
    assert get_probabilistic_classifier_class("classifier") is ClassifierProbabilisticClassifier
    assert get_probabilistic_classifier_class("generative") is FMProbabilisticClassifier


def test_resolve_predictor_name_prefers_name_over_type():
    from omegaconf import OmegaConf
    from adaptive_roa.probabilistic_classifier.export import resolve_predictor_name

    named = OmegaConf.create({"predictor": {"type": "classifier", "name": "bnn_mfvi"}})
    legacy = OmegaConf.create({"predictor": {"type": "classifier"}})
    assert resolve_predictor_name(named) == "bnn_mfvi"
    assert resolve_predictor_name(legacy) == "classifier"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./env/bin/python -m pytest tests/probabilistic_classifier/test_registry.py -v`
Expected: FAIL — `KeyError` for `"mlp"`, `ImportError` for `resolve_predictor_name`.

- [ ] **Step 3: Register under both keys**

In `adaptive_roa/probabilistic_classifier/base.py`, add a class attribute next to `predictor_type`:

```python
    predictor_name: str = ""
```

In `adaptive_roa/probabilistic_classifier/registry.py`, replace `register_probabilistic_classifier`:

```python
def register_probabilistic_classifier(cls: Type[ProbabilisticClassifier]):
    """Register ``cls`` under its arm name, and under its family tag as an alias.

    ``predictor_type`` is a family tag ("classifier"/"generative") shared by
    several arms, so it cannot be the primary key -- five outcome arms would
    collide and the export would load the wrong checkpoint class. The family
    alias is kept so runs written before ``predictor.name`` existed still
    resolve; first registration wins so a later arm cannot steal the alias.
    """
    if not cls.predictor_type:
        raise ValueError(f"{cls.__name__} must set a non-empty predictor_type")
    if not cls.predictor_name:
        raise ValueError(f"{cls.__name__} must set a non-empty predictor_name")
    _REGISTRY[cls.predictor_name] = cls
    _REGISTRY.setdefault(cls.predictor_type, cls)
    return cls
```

In `classifier.py:52-54` add `predictor_name = "mlp"`. In `flow_matching.py:61-63` add `predictor_name = "fm"`.

- [ ] **Step 4: Resolve the name in export**

In `adaptive_roa/probabilistic_classifier/export.py`, change `_DATASET_KIND` (line 20) to map by family, and add a resolver:

```python
_DATASET_KIND = {"classifier": "classification", "generative": "endpoint"}


def resolve_predictor_family(cfg) -> str:
    """Family tag: drives which dataset files a run wrote."""
    predictor = cfg.get("predictor", None)
    if predictor is None:
        raise ValueError(
            "run config has no 'predictor' entry; refusing to guess "
            "the export type (classifier vs generative)."
        )
    if isinstance(predictor, str):
        return predictor
    family = predictor.get("type", None)
    if family is None:
        raise ValueError(
            "run config predictor block has no 'type'; cannot determine export type."
        )
    return str(family)


def resolve_predictor_name(cfg) -> str:
    """Arm name, falling back to the family tag for pre-``name`` runs."""
    predictor = cfg.get("predictor", None)
    if predictor is None or isinstance(predictor, str):
        return resolve_predictor_family(cfg)
    return str(predictor.get("name", None) or resolve_predictor_family(cfg))
```

In `export_run` (lines 110-128), replace the inline predictor-type block with:

```python
    predictor_family = resolve_predictor_family(cfg)
    predictor_name = resolve_predictor_name(cfg)
    system = resolve_system(cfg)
    pc_class = get_probabilistic_classifier_class(predictor_name)
```

Then update every downstream use in `export_run`: pass `predictor_family` to `load_split_states` (it selects dataset kind and label basis), and write `"predictor": predictor_name` in the metadata at line 185. The comparisons at lines 190 and 197 (`if predictor_type == "classifier"`) become `if predictor_family == "classifier"`.

- [ ] **Step 5: Add `name` to the four shipped configs**

In each of `configs/adaptive_v2/predictor/{classifier,generative,gp,gp_optdelta}.yaml`, add a `name` line directly under `type:` — `mlp`, `fm`, `gp`, `gp_optdelta` respectively.

- [ ] **Step 6: Run tests to verify they pass**

Run: `./env/bin/python -m pytest tests/probabilistic_classifier -v`
Expected: PASS (all tests, including the pre-existing export integration tests)

- [ ] **Step 7: Commit**

```bash
git add adaptive_roa/probabilistic_classifier/ configs/adaptive_v2/predictor/ tests/probabilistic_classifier/
git commit -m "feat(export): key probabilistic-classifier registry on arm name with family alias"
```

---

### Task 4: `VILinear` and the MFVI posterior

**Files:**
- Create: `adaptive_roa/predictors/__init__.py`, `adaptive_roa/predictors/posteriors.py`
- Test: `tests/predictors/test_posteriors.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `VILinear(in_features, out_features, prior_sigma=1.0, rho_init=-5.0)` with `.forward(x, generator=None)` and `.kl_divergence() -> Tensor` (scalar)
  - `Posterior` ABC with `.forward_sample(x, generator=None) -> Tensor`, `.forward_samples(x, S, generator=None) -> Tensor [S,B,out]`, `.kl_divergence() -> Tensor`
  - `DeterministicPosterior(net)`, `MFVIPosterior(net)`

- [ ] **Step 1: Write the failing test**

Create `tests/predictors/test_posteriors.py`:

```python
import torch

from adaptive_roa.predictors.posteriors import (
    DeterministicPosterior,
    MFVIPosterior,
    VILinear,
)


def test_vilinear_kl_is_finite_and_positive():
    layer = VILinear(4, 3, prior_sigma=1.0)
    kl = layer.kl_divergence()
    assert kl.ndim == 0
    assert torch.isfinite(kl)
    assert kl.item() > 0.0


def test_vilinear_kl_is_zero_when_q_equals_prior():
    """KL(q||p) == 0 exactly when q has the prior's mean and scale."""
    layer = VILinear(4, 3, prior_sigma=1.0)
    with torch.no_grad():
        layer.weight_mu.zero_()
        layer.bias_mu.zero_()
        # softplus(rho) == 1.0  =>  rho == log(e - 1)
        rho = torch.log(torch.expm1(torch.tensor(1.0)))
        layer.weight_rho.fill_(rho)
        layer.bias_rho.fill_(rho)
    assert layer.kl_divergence().abs().item() < 1e-5


def test_vilinear_forward_is_stochastic():
    layer = VILinear(4, 3)
    x = torch.randn(8, 4)
    assert not torch.allclose(layer(x), layer(x))


def test_vilinear_forward_is_reproducible_under_a_seeded_generator():
    layer = VILinear(4, 3)
    x = torch.randn(8, 4)
    g1 = torch.Generator().manual_seed(0)
    g2 = torch.Generator().manual_seed(0)
    assert torch.allclose(layer(x, generator=g1), layer(x, generator=g2))


def test_deterministic_posterior_has_zero_kl_and_no_spread():
    net = torch.nn.Linear(4, 2)
    post = DeterministicPosterior(net)
    x = torch.randn(8, 4)
    assert post.kl_divergence().item() == 0.0
    assert torch.allclose(post.forward_sample(x), post.forward_sample(x))


def test_mfvi_posterior_spreads_and_accumulates_kl():
    net = torch.nn.Sequential(VILinear(4, 6), torch.nn.ReLU(), VILinear(6, 2))
    post = MFVIPosterior(net)
    x = torch.randn(8, 4)
    assert not torch.allclose(post.forward_sample(x), post.forward_sample(x))
    assert post.kl_divergence().item() > 0.0
    samples = post.forward_samples(x, S=5)
    assert samples.shape == (5, 8, 2)
    assert samples.std(dim=0).mean().item() > 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./env/bin/python -m pytest tests/predictors/test_posteriors.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'adaptive_roa.predictors'`

- [ ] **Step 3: Write the implementation**

Create `adaptive_roa/predictors/__init__.py` (empty for now), then `adaptive_roa/predictors/posteriors.py`:

```python
"""Weight posteriors for Bayesian MLP arms.

A BNN predicts by marginalizing p(y|x,D) = int p(y|x,w) p(w|D) dw. These classes
supply the p(w|D) half: each exposes ``forward_sample`` (one draw from the
approximate posterior) and ``kl_divergence`` (zero for non-variational members).
The likelihood half lives in the heads.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F


class VILinear(nn.Module):
    """Linear layer with a mean-field Gaussian posterior over weights.

    Bayes-by-Backprop (Blundell et al., 2015): w = mu + softplus(rho) * eps.
    ``rho_init = -5.0`` gives an initial scale of softplus(-5) ~ 0.0067, small
    enough that early training behaves like a deterministic net and the KL term
    does not dominate before the likelihood has any signal.
    """

    def __init__(self, in_features: int, out_features: int,
                 prior_sigma: float = 1.0, rho_init: float = -5.0):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.prior_sigma = float(prior_sigma)

        bound = 1.0 / math.sqrt(self.in_features)
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features).uniform_(-bound, bound))
        self.weight_rho = nn.Parameter(torch.full((out_features, in_features), float(rho_init)))
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_rho = nn.Parameter(torch.full((out_features,), float(rho_init)))

    def forward(self, x: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        w_sigma = F.softplus(self.weight_rho)
        b_sigma = F.softplus(self.bias_rho)
        w_eps = torch.randn(self.weight_mu.shape, generator=generator,
                            device=self.weight_mu.device, dtype=self.weight_mu.dtype)
        b_eps = torch.randn(self.bias_mu.shape, generator=generator,
                            device=self.bias_mu.device, dtype=self.bias_mu.dtype)
        return F.linear(x, self.weight_mu + w_sigma * w_eps, self.bias_mu + b_sigma * b_eps)

    def kl_divergence(self) -> torch.Tensor:
        """Closed-form KL(N(mu, sigma^2) || N(0, prior_sigma^2)), summed."""
        total = torch.zeros((), device=self.weight_mu.device, dtype=self.weight_mu.dtype)
        for mu, rho in ((self.weight_mu, self.weight_rho), (self.bias_mu, self.bias_rho)):
            sigma = F.softplus(rho)
            total = total + (
                math.log(self.prior_sigma) - torch.log(sigma)
                + (sigma.pow(2) + mu.pow(2)) / (2.0 * self.prior_sigma ** 2)
                - 0.5
            ).sum()
        return total


class Posterior(nn.Module, ABC):
    """Uniform interface over approximate weight posteriors."""

    @abstractmethod
    def forward_sample(self, x: torch.Tensor,
                       generator: torch.Generator | None = None) -> torch.Tensor:
        """One draw from q(w): [B, in] -> [B, out]."""

    def forward(self, x: torch.Tensor,
                generator: torch.Generator | None = None) -> torch.Tensor:
        """Delegate ``__call__`` to ``forward_sample``.

        Required so a Posterior can be nested inside another Posterior --
        EnsemblePosterior's members are DeterministicPosterior instances and are
        invoked as ``member(x)``.
        """
        return self.forward_sample(x, generator=generator)

    def forward_samples(self, x: torch.Tensor, S: int,
                        generator: torch.Generator | None = None) -> torch.Tensor:
        """S independent draws: [B, in] -> [S, B, out]."""
        return torch.stack([self.forward_sample(x, generator=generator) for _ in range(int(S))], dim=0)

    def kl_divergence(self) -> torch.Tensor:
        """KL(q||p); zero for non-variational posteriors."""
        return torch.zeros((), device=next(self.parameters()).device)


class DeterministicPosterior(Posterior):
    """Point estimate. Included so the deterministic arm shares one code path."""

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward_sample(self, x, generator=None):
        return self.net(x)


class MFVIPosterior(Posterior):
    """Mean-field variational inference over a net built from VILinear layers."""

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net
        if not any(isinstance(m, VILinear) for m in self.net.modules()):
            raise ValueError("MFVIPosterior requires a net containing VILinear layers")

    def forward_sample(self, x, generator=None):
        h = x
        for module in self.net:
            h = module(h, generator=generator) if isinstance(module, VILinear) else module(h)
        return h

    def kl_divergence(self):
        return sum(m.kl_divergence() for m in self.net.modules() if isinstance(m, VILinear))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./env/bin/python -m pytest tests/predictors/test_posteriors.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add adaptive_roa/predictors/ tests/predictors/test_posteriors.py
git commit -m "feat(predictors): VILinear layer and mean-field VI posterior"
```

---

### Task 5: Ensemble posterior

Deep ensembles are MAP inference rather than posterior inference (D'Angelo & Fortuin, 2021) and are included as a strong, contested baseline. The final-state head will draw one member per call, so `K >= 2M` is required to resolve an M-atom posterior; that guard lands in Task 9 where K is known.

**Files:**
- Modify: `adaptive_roa/predictors/posteriors.py`
- Test: `tests/predictors/test_posteriors.py`

**Interfaces:**
- Consumes: `Posterior` from Task 4
- Produces: `EnsemblePosterior(nets: list[nn.Module])` with `.members -> nn.ModuleList` and `.n_members -> int`

- [ ] **Step 1: Write the failing test**

Append to `tests/predictors/test_posteriors.py`:

```python
def test_ensemble_posterior_draws_a_member_per_call():
    from adaptive_roa.predictors.posteriors import EnsemblePosterior

    torch.manual_seed(0)
    nets = [torch.nn.Linear(4, 2) for _ in range(5)]
    # Force members apart so "which member" is observable in the output.
    with torch.no_grad():
        for i, net in enumerate(nets):
            net.bias.fill_(float(i))
    post = EnsemblePosterior(nets)
    x = torch.zeros(1, 4)

    assert post.n_members == 5
    assert post.kl_divergence().item() == 0.0
    draws = {round(post.forward_sample(x)[0, 0].item()) for _ in range(200)}
    assert len(draws) == 5, f"expected all 5 members to be drawn, saw {draws}"


def test_ensemble_forward_samples_has_spread():
    from adaptive_roa.predictors.posteriors import EnsemblePosterior

    torch.manual_seed(0)
    nets = [torch.nn.Linear(4, 2) for _ in range(5)]
    with torch.no_grad():
        for i, net in enumerate(nets):
            net.bias.fill_(float(i))
    post = EnsemblePosterior(nets)
    samples = post.forward_samples(torch.zeros(3, 4), S=40)
    assert samples.shape == (40, 3, 2)
    assert samples.std(dim=0).mean().item() > 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./env/bin/python -m pytest tests/predictors/test_posteriors.py -k ensemble -v`
Expected: FAIL — `ImportError: cannot import name 'EnsemblePosterior'`

- [ ] **Step 3: Write the implementation**

Append to `adaptive_roa/predictors/posteriors.py`:

```python
class EnsemblePosterior(Posterior):
    """Deep ensemble: M independently trained members, drawn uniformly.

    Formally MAP inference rather than Bayesian inference (D'Angelo & Fortuin,
    2021), but empirically a closer match to the HMC predictive than mean-field
    VI (Izmailov et al., 2021), so it is carried as a baseline.

    ``forward_sample`` returns ONE member, giving an M-atom empirical posterior.
    Callers drawing K samples need K >= 2M to resolve it.
    """

    def __init__(self, nets):
        super().__init__()
        nets = list(nets)
        if len(nets) < 2:
            raise ValueError(f"EnsemblePosterior needs >= 2 members, got {len(nets)}")
        self.members = nn.ModuleList(nets)

    @property
    def n_members(self) -> int:
        return len(self.members)

    def forward_sample(self, x, generator=None):
        idx = int(torch.randint(self.n_members, (1,), generator=generator,
                                device="cpu").item())
        return self.members[idx](x)

    def forward_all_members(self, x: torch.Tensor) -> torch.Tensor:
        """Every member, deterministically: [B, in] -> [M, B, out]."""
        return torch.stack([m(x) for m in self.members], dim=0)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./env/bin/python -m pytest tests/predictors/test_posteriors.py -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add adaptive_roa/predictors/posteriors.py tests/predictors/test_posteriors.py
git commit -m "feat(predictors): deep ensemble posterior"
```

---

### Task 6: Last-layer Laplace posterior

Post-hoc Gaussian over last-layer weights via the GGN. Restricting to the last layer keeps the Hessian small and PSD without a new dependency.

**Files:**
- Modify: `adaptive_roa/predictors/posteriors.py`
- Test: `tests/predictors/test_posteriors.py`

**Interfaces:**
- Consumes: `Posterior` from Task 4
- Produces: `LastLayerLaplacePosterior(body, head_layer, prior_precision=1.0)` with `.fit(features, targets, task)` where `task in {"outcome", "final_state"}`, and `.posterior_covariance -> Tensor`

- [ ] **Step 1: Write the failing test**

Append to `tests/predictors/test_posteriors.py`:

```python
def test_laplace_covariance_is_psd_and_shrinks_with_data():
    from adaptive_roa.predictors.posteriors import LastLayerLaplacePosterior

    torch.manual_seed(0)
    body = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU())
    head = torch.nn.Linear(8, 1)
    post = LastLayerLaplacePosterior(body, head, prior_precision=1.0)

    x_small, x_large = torch.randn(16, 4), torch.randn(512, 4)
    post.fit(body(x_small), torch.randint(0, 2, (16,)).float(), task="outcome")
    cov_small = post.posterior_covariance.clone()
    post.fit(body(x_large), torch.randint(0, 2, (512,)).float(), task="outcome")
    cov_large = post.posterior_covariance.clone()

    # Symmetric positive definite.
    assert torch.allclose(cov_small, cov_small.T, atol=1e-6)
    assert torch.linalg.eigvalsh(cov_small).min().item() > 0.0
    # More data => tighter posterior.
    assert cov_large.diagonal().mean().item() < cov_small.diagonal().mean().item()


def test_laplace_forward_is_stochastic_only_after_fit():
    from adaptive_roa.predictors.posteriors import LastLayerLaplacePosterior

    torch.manual_seed(0)
    body = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU())
    head = torch.nn.Linear(8, 1)
    post = LastLayerLaplacePosterior(body, head, prior_precision=1.0)
    x = torch.randn(8, 4)

    # Before fit: falls back to the MAP point estimate.
    assert torch.allclose(post.forward_sample(x), post.forward_sample(x))
    post.fit(body(x), torch.randint(0, 2, (8,)).float(), task="outcome")
    assert not torch.allclose(post.forward_sample(x), post.forward_sample(x))


def test_laplace_regression_requires_an_explicit_sigma():
    """sigma is the observation noise; silently defaulting it to 1.0 is the
    documented laplace-torch trap and would scale the whole covariance."""
    from adaptive_roa.predictors.posteriors import LastLayerLaplacePosterior

    body = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU())
    head = torch.nn.Linear(8, 3)
    post = LastLayerLaplacePosterior(body, head, prior_precision=1.0)
    with pytest.raises(ValueError, match="sigma"):
        post.fit(body(torch.randn(8, 4)), torch.randn(8, 3), task="final_state")
```

Add `import pytest` to the top of the test file.

- [ ] **Step 2: Run test to verify it fails**

Run: `./env/bin/python -m pytest tests/predictors/test_posteriors.py -k laplace -v`
Expected: FAIL — `ImportError: cannot import name 'LastLayerLaplacePosterior'`

- [ ] **Step 3: Write the implementation**

Append to `adaptive_roa/predictors/posteriors.py`:

```python
class LastLayerLaplacePosterior(Posterior):
    """Post-hoc Gaussian over last-layer weights via the GGN.

    H = sum_n Lambda_n * phi_n phi_n^T + prior_precision * I, with
    Lambda = p(1-p) for a Bernoulli head and Lambda = 1/sigma^2 for a Gaussian
    head. The last-layer restriction keeps H small and PSD.

    ``sigma`` is the observation noise and is REQUIRED for regression. Letting
    it default to 1.0 (the laplace-torch default) silently scales the entire
    posterior covariance by an arbitrary constant unrelated to the data.
    """

    def __init__(self, body: nn.Module, head_layer: nn.Linear, prior_precision: float = 1.0):
        super().__init__()
        self.body = body
        self.head_layer = head_layer
        self.prior_precision = float(prior_precision)
        self.register_buffer("_cov", torch.empty(0), persistent=False)

    @property
    def posterior_covariance(self) -> torch.Tensor:
        if self._cov.numel() == 0:
            raise RuntimeError("LastLayerLaplacePosterior.fit has not been called")
        return self._cov

    @property
    def is_fitted(self) -> bool:
        return self._cov.numel() > 0

    def fit(self, features: torch.Tensor, targets: torch.Tensor,
            task: str, sigma: float | None = None) -> "LastLayerLaplacePosterior":
        """Fit the GGN over last-layer weights. ``features`` are body outputs."""
        phi = features.detach()
        phi = torch.cat([phi, torch.ones(phi.shape[0], 1, dtype=phi.dtype, device=phi.device)], dim=1)

        if task == "outcome":
            with torch.no_grad():
                p = torch.sigmoid(self.head_layer(features.detach()).view(-1))
            lam = (p * (1.0 - p)).clamp_min(1e-6)
        elif task == "final_state":
            if sigma is None:
                raise ValueError(
                    "final_state Laplace requires an explicit observation noise "
                    "sigma; defaulting it to 1.0 would scale the posterior "
                    "covariance by an arbitrary constant."
                )
            lam = torch.full((phi.shape[0],), 1.0 / float(sigma) ** 2,
                             dtype=phi.dtype, device=phi.device)
        else:
            raise ValueError(f"unknown task {task!r}; expected 'outcome' or 'final_state'")

        H = torch.einsum("n,ni,nj->ij", lam, phi, phi)
        H = H + self.prior_precision * torch.eye(phi.shape[1], dtype=phi.dtype, device=phi.device)
        cov = torch.linalg.inv(H)
        self._cov = 0.5 * (cov + cov.T)  # symmetrize away round-off
        return self

    def forward_sample(self, x, generator=None):
        features = self.body(x)
        if not self.is_fitted:
            return self.head_layer(features)  # MAP fallback before fit
        phi = torch.cat(
            [features, torch.ones(features.shape[0], 1, dtype=features.dtype, device=features.device)],
            dim=1,
        )
        map_w = torch.cat([self.head_layer.weight, self.head_layer.bias.unsqueeze(1)], dim=1)
        L = torch.linalg.cholesky(self._cov.to(features.dtype))
        eps = torch.randn(map_w.shape[0], L.shape[0], generator=generator,
                          device="cpu", dtype=features.dtype).to(features.device)
        w = map_w + eps @ L.T
        return phi @ w.T
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./env/bin/python -m pytest tests/predictors/test_posteriors.py -v`
Expected: PASS (11 tests)

- [ ] **Step 5: Commit**

```bash
git add adaptive_roa/predictors/posteriors.py tests/predictors/test_posteriors.py
git commit -m "feat(predictors): last-layer Laplace posterior with required observation noise"
```

---

### Task 7: `BayesianMLP` with a body/head split

The body/head split exists so Laplace can reach penultimate features and so the final-state head (Plan 2) can attach a different output layer to the same body.

**Files:**
- Create: `adaptive_roa/predictors/bayesian_mlp.py`
- Test: `tests/predictors/test_bayesian_mlp.py`

**Interfaces:**
- Consumes: everything from `posteriors.py`
- Produces: `build_bayesian_mlp(input_dim, hidden_dims, output_dim, posterior, *, prior_sigma=1.0, n_members=5, dropout=0.0, activation="relu") -> Posterior`, where `posterior in {"deterministic", "mfvi", "ensemble", "laplace"}`

- [ ] **Step 1: Write the failing test**

Create `tests/predictors/test_bayesian_mlp.py`:

```python
import pytest
import torch

from adaptive_roa.predictors.bayesian_mlp import build_bayesian_mlp
from adaptive_roa.predictors.posteriors import (
    DeterministicPosterior,
    EnsemblePosterior,
    LastLayerLaplacePosterior,
    MFVIPosterior,
)


@pytest.mark.parametrize(
    "kind,expected",
    [
        ("deterministic", DeterministicPosterior),
        ("mfvi", MFVIPosterior),
        ("ensemble", EnsemblePosterior),
        ("laplace", LastLayerLaplacePosterior),
    ],
)
def test_build_returns_the_right_posterior_and_output_shape(kind, expected):
    model = build_bayesian_mlp(
        input_dim=3, hidden_dims=[16, 16], output_dim=1, posterior=kind, n_members=3
    )
    assert isinstance(model, expected)
    assert model.forward_sample(torch.randn(8, 3)).shape == (8, 1)


def test_unknown_posterior_is_rejected():
    with pytest.raises(ValueError, match="unknown posterior"):
        build_bayesian_mlp(input_dim=3, hidden_dims=[8], output_dim=1, posterior="mcdropout")


def test_mfvi_net_is_built_from_vilinear_layers():
    from adaptive_roa.predictors.posteriors import VILinear

    model = build_bayesian_mlp(input_dim=3, hidden_dims=[8, 8], output_dim=1, posterior="mfvi")
    assert sum(isinstance(m, VILinear) for m in model.modules()) == 3  # 2 hidden + 1 output
    assert model.kl_divergence().item() > 0.0


def test_ensemble_members_are_independently_initialized():
    model = build_bayesian_mlp(
        input_dim=3, hidden_dims=[8], output_dim=1, posterior="ensemble", n_members=4
    )
    assert model.n_members == 4
    first = model.members[0].net[0].weight
    assert not any(torch.allclose(first, m.net[0].weight) for m in list(model.members)[1:])


def test_laplace_exposes_a_body_for_feature_extraction():
    model = build_bayesian_mlp(input_dim=3, hidden_dims=[8, 8], output_dim=1, posterior="laplace")
    assert model.body(torch.randn(5, 3)).shape == (5, 8)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./env/bin/python -m pytest tests/predictors/test_bayesian_mlp.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'adaptive_roa.predictors.bayesian_mlp'`

- [ ] **Step 3: Write the implementation**

Create `adaptive_roa/predictors/bayesian_mlp.py`:

```python
"""Backbone construction for Bayesian MLP arms.

The MLP is split into a ``body`` (everything up to the last hidden activation)
and a single output ``Linear``. Two things need that split: last-layer Laplace
fits its GGN over the body's penultimate features, and the final-state head
attaches a different output layer to the same body.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from adaptive_roa.predictors.posteriors import (
    DeterministicPosterior,
    EnsemblePosterior,
    LastLayerLaplacePosterior,
    MFVIPosterior,
    Posterior,
    VILinear,
)

_ACTIVATIONS = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU}


class _Body(nn.Module):
    """Hidden stack, exposed as a module so Laplace can call it directly."""

    def __init__(self, layers: List[nn.Module]):
        super().__init__()
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def _hidden_layers(input_dim, hidden_dims, activation, dropout, linear_cls, **linear_kwargs):
    act_cls = _ACTIVATIONS.get(activation)
    if act_cls is None:
        raise ValueError(f"unknown activation {activation!r}; expected one of {sorted(_ACTIVATIONS)}")
    dims = [int(input_dim)] + [int(h) for h in hidden_dims]
    layers: List[nn.Module] = []
    for a, b in zip(dims[:-1], dims[1:]):
        layers.append(linear_cls(a, b, **linear_kwargs))
        layers.append(act_cls())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
    return layers, dims[-1]


def build_bayesian_mlp(
    input_dim: int,
    hidden_dims: List[int],
    output_dim: int,
    posterior: str,
    *,
    prior_sigma: float = 1.0,
    n_members: int = 5,
    dropout: float = 0.0,
    activation: str = "relu",
) -> Posterior:
    """Build an MLP wrapped in the requested approximate weight posterior."""
    kind = str(posterior)

    if kind == "mfvi":
        layers, last_hidden = _hidden_layers(
            input_dim, hidden_dims, activation, dropout, VILinear, prior_sigma=prior_sigma
        )
        layers.append(VILinear(last_hidden, int(output_dim), prior_sigma=prior_sigma))
        return MFVIPosterior(nn.Sequential(*layers))

    if kind == "deterministic":
        layers, last_hidden = _hidden_layers(
            input_dim, hidden_dims, activation, dropout, nn.Linear
        )
        layers.append(nn.Linear(last_hidden, int(output_dim)))
        return DeterministicPosterior(nn.Sequential(*layers))

    if kind == "ensemble":
        members = [
            build_bayesian_mlp(
                input_dim, hidden_dims, output_dim, "deterministic",
                dropout=dropout, activation=activation,
            )
            for _ in range(int(n_members))
        ]
        return EnsemblePosterior(members)

    if kind == "laplace":
        layers, last_hidden = _hidden_layers(
            input_dim, hidden_dims, activation, dropout, nn.Linear
        )
        return LastLayerLaplacePosterior(
            body=_Body(layers),
            head_layer=nn.Linear(last_hidden, int(output_dim)),
            prior_precision=1.0 / float(prior_sigma) ** 2,
        )

    raise ValueError(
        f"unknown posterior {posterior!r}; expected one of "
        "'deterministic', 'mfvi', 'ensemble', 'laplace'"
    )
```

Note the ensemble branch nests `DeterministicPosterior` members, so `model.members[i].net` is the `nn.Sequential` — matching the test's `m.net[0].weight` access.

- [ ] **Step 4: Run test to verify it passes**

Run: `./env/bin/python -m pytest tests/predictors/test_bayesian_mlp.py -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add adaptive_roa/predictors/bayesian_mlp.py tests/predictors/test_bayesian_mlp.py
git commit -m "feat(predictors): BayesianMLP builder with body/head split"
```

---

### Task 8: `OutcomeModelHandle` with internal marginalization

This is the contract-critical task. The handle is called once per query by `ClassifierProbabilityEstimator` but separately by threshold optimization, calibration, and evaluation — all of which must agree. It therefore marginalizes internally and is seeded.

**Files:**
- Create: `adaptive_roa/predictors/handles.py`
- Test: `tests/predictors/test_outcome_handle.py`

**Interfaces:**
- Consumes: `Posterior` from Task 4, `build_bayesian_mlp` from Task 7
- Produces: `OutcomeModelHandle(posterior, system, n_marginal_samples=64, seed=0)` with `.eval()`, `.to(device)`, `__call__(raw_states) -> Tensor [B]`

- [ ] **Step 1: Write the failing test**

Create `tests/predictors/test_outcome_handle.py`:

```python
import numpy as np
import torch

from adaptive_roa.predictors.bayesian_mlp import build_bayesian_mlp
from adaptive_roa.predictors.handles import OutcomeModelHandle
from adaptive_roa.systems.cartpole import CartPoleSystem


def _handle(kind="mfvi", **kw):
    system = CartPoleSystem()
    dummy = torch.zeros(1, int(system.state_dim))
    input_dim = int(system.embed_state_for_model(system.normalize_state(dummy)).shape[-1])
    posterior = build_bayesian_mlp(
        input_dim=input_dim, hidden_dims=[16, 16], output_dim=1, posterior=kind
    )
    return OutcomeModelHandle(posterior, system, **kw), system


def test_handle_is_deterministic_across_repeated_calls():
    """Threshold optimization, calibration, and eval each call this separately
    and must agree. A stochastic handle silently desynchronizes them."""
    handle, system = _handle()
    x = torch.randn(16, int(system.state_dim))
    assert torch.allclose(handle(x), handle(x))
    assert torch.allclose(handle(x), handle(x))  # and again


def test_handle_returns_logits_the_classifier_estimator_can_consume():
    handle, system = _handle()
    out = handle(torch.randn(16, int(system.state_dim)))
    assert out.shape in {(16,), (16, 1)}
    assert torch.isfinite(out).all()


def test_handle_marginalizes_rather_than_taking_one_draw():
    """More marginal samples => a tighter estimate of the same quantity.

    The seed must be set BEFORE _handle() builds the network, so all eight
    handles share identical weights and the only thing varying is the
    marginalization draw. Seeding afterwards would leave the weights different
    and the measured spread would be weight noise, not MC error.
    """
    x = torch.randn(32, 4)
    spreads = []
    for n in (2, 128):
        outs = []
        for seed in range(8):
            torch.manual_seed(0)
            handle, _ = _handle(n_marginal_samples=n, seed=seed)
            outs.append(handle(x))
        spreads.append(torch.stack(outs).std(dim=0).mean().item())
    assert spreads[1] < spreads[0]


def test_handle_accepts_numpy_and_moves_to_device():
    handle, system = _handle()
    handle.eval().to("cpu")
    out = handle(np.random.randn(8, int(system.state_dim)).astype(np.float32))
    assert torch.is_tensor(out)


def test_deterministic_posterior_handle_matches_a_plain_forward():
    handle, system = _handle(kind="deterministic")
    x = torch.randn(8, int(system.state_dim))
    normalized = system.embed_state_for_model(system.normalize_state(x))
    expected = handle.posterior.forward_sample(normalized).view(-1)
    assert torch.allclose(handle(x), expected, atol=1e-5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./env/bin/python -m pytest tests/predictors/test_outcome_handle.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'adaptive_roa.predictors.handles'`

- [ ] **Step 3: Write the implementation**

Create `adaptive_roa/predictors/handles.py`:

```python
"""Duck-typed model handles binding predictors to the existing v2 backends.

``OutcomeModelHandle`` reproduces the contract that
``ClassifierProbabilityEstimator`` and ``full_roa.evaluate_full_roa_classifier``
already rely on -- ``model(raw_states) -> logits``, with ``sigmoid(logits)`` read
as p(success) -- exactly as ``adaptive_roa/partx/model_handle.py`` does for the
GP. Satisfying it means the Bayesian arms route through the shared conformal,
threshold, and evaluation machinery with no changes to that code.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

_EPS = 1e-12


class OutcomeModelHandle:
    """Marginalizes a weight posterior into a single deterministic logit vector.

    The marginalization is INTERNAL and seeded on purpose. The estimator calls
    this once per query, but threshold optimization, calibration, and evaluation
    each call it separately on overlapping states and must agree; a fresh draw
    per call would desynchronize them in a way that looks like a calibration bug
    rather than a sampling bug.
    """

    def __init__(self, posterior, system: Any, n_marginal_samples: int = 64,
                 seed: int = 0, device: str = "cpu"):
        self.posterior = posterior
        self.system = system
        self.n_marginal_samples = int(n_marginal_samples)
        self.seed = int(seed)
        self.device = device

    def eval(self):
        self.posterior.eval()
        return self

    def to(self, device):
        self.device = device
        self.posterior.to(device)
        return self

    def __call__(self, states) -> torch.Tensor:
        if torch.is_tensor(states):
            x = states.detach().to(dtype=torch.float32)
            out_device = states.device
        else:
            x = torch.as_tensor(np.asarray(states), dtype=torch.float32)
            out_device = self.device
        x = x.to(next(self.posterior.parameters()).device)

        embedded = self.system.embed_state_for_model(self.system.normalize_state(x))

        generator = torch.Generator().manual_seed(self.seed)
        with torch.no_grad():
            samples = self.posterior.forward_samples(
                embedded, S=self.n_marginal_samples, generator=generator
            )  # [S, B, 1]
            # Marginalize in PROBABILITY space, not logit space: the Bayesian
            # model average is E_q[p(y|x,w)], and averaging logits instead would
            # be a different (and systematically overconfident) estimator.
            p = torch.sigmoid(samples.view(samples.shape[0], samples.shape[1])).mean(dim=0)

        p = p.clamp(_EPS, 1.0 - _EPS)
        return torch.log(p / (1.0 - p)).to(out_device)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./env/bin/python -m pytest tests/predictors/test_outcome_handle.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add adaptive_roa/predictors/handles.py tests/predictors/test_outcome_handle.py
git commit -m "feat(predictors): outcome model handle with seeded internal marginalization"
```

---

### Task 9: `BayesianMLPTrainer` and the three arm configs

**Files:**
- Create: `adaptive_roa/adaptive_v2/trainers/bayesian_mlp_trainer.py`
- Create: `configs/adaptive_v2/predictor/{bnn_mfvi,bnn_ensemble,bnn_laplace}.yaml`
- Test: `tests/predictors/test_bnn_trainer.py`

**Interfaces:**
- Consumes: `build_bayesian_mlp` (Task 7), `OutcomeModelHandle` (Task 8)
- Produces: `BayesianMLPTrainer(cfg, system, system_name)` with `fit(dataset_files, output_dir, resume_checkpoint=None) -> OutcomeModelHandle`

- [ ] **Step 1: Write the failing test**

Create `tests/predictors/test_bnn_trainer.py`:

```python
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from adaptive_roa.adaptive_v2.trainers.bayesian_mlp_trainer import BayesianMLPTrainer
from adaptive_roa.systems.cartpole import CartPoleSystem


def _write_dataset(path, n=256, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, size=(n, 4))
    y = (np.abs(X[:, 1]) < 0.5).astype(int)
    np.savetxt(path, np.column_stack([X, y]))
    return str(path)


def _cfg(posterior, **overrides):
    bnn = {
        "hidden_dims": [16, 16], "lr": 1e-2, "weight_decay": 1e-5,
        "max_epochs": 3, "patience": 5, "prior_sigma": 1.0,
        "n_members": 2, "kl_weight": 1.0, "n_marginal_samples": 8,
        "posterior": posterior,
    }
    bnn.update(overrides)
    return OmegaConf.create({
        "device": "cpu",
        "predictor": {"type": "classifier", "name": f"bnn_{posterior}",
                      "batch_size": 64, "bnn": bnn},
    })


@pytest.mark.parametrize("posterior", ["mfvi", "ensemble", "laplace"])
def test_trainer_returns_a_working_outcome_handle(posterior, tmp_path):
    files = {"train": _write_dataset(tmp_path / "train.txt"),
             "val": _write_dataset(tmp_path / "val.txt", n=128, seed=1)}
    trainer = BayesianMLPTrainer(_cfg(posterior), CartPoleSystem(), "cartpole")
    handle = trainer.fit(files, str(tmp_path / "out"))

    x = torch.randn(16, 4)
    logits = handle(x)
    assert logits.shape in {(16,), (16, 1)}
    assert torch.isfinite(logits).all()
    # Contract: repeated calls must agree.
    assert torch.allclose(handle(x), handle(x))


@pytest.mark.parametrize("posterior", ["mfvi", "ensemble", "laplace"])
def test_trainer_writes_a_checkpoint_for_warm_start(posterior, tmp_path):
    files = {"train": _write_dataset(tmp_path / "train.txt"),
             "val": _write_dataset(tmp_path / "val.txt", n=128, seed=1)}
    out = tmp_path / "out"
    BayesianMLPTrainer(_cfg(posterior), CartPoleSystem(), "cartpole").fit(files, str(out))
    assert list((out / "checkpoints").glob("best*.ckpt")), "engine warm start needs best*.ckpt"


def test_laplace_arm_fits_its_ggn_during_training(tmp_path):
    files = {"train": _write_dataset(tmp_path / "train.txt"),
             "val": _write_dataset(tmp_path / "val.txt", n=128, seed=1)}
    trainer = BayesianMLPTrainer(_cfg("laplace"), CartPoleSystem(), "cartpole")
    handle = trainer.fit(files, str(tmp_path / "out"))
    assert handle.posterior.is_fitted, "Laplace posterior left at the MAP point estimate"


def test_mfvi_reports_its_kl_weight(tmp_path):
    """beta is a reported protocol parameter, not a silent default."""
    files = {"train": _write_dataset(tmp_path / "train.txt"),
             "val": _write_dataset(tmp_path / "val.txt", n=128, seed=1)}
    trainer = BayesianMLPTrainer(_cfg("mfvi", kl_weight=1.0), CartPoleSystem(), "cartpole")
    assert trainer.kl_weight == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./env/bin/python -m pytest tests/predictors/test_bnn_trainer.py -v`
Expected: FAIL — `ModuleNotFoundError` for `bayesian_mlp_trainer`

- [ ] **Step 3: Write the trainer**

Create `adaptive_roa/adaptive_v2/trainers/bayesian_mlp_trainer.py`:

```python
"""Trainer for Bayesian MLP arms with an outcome head.

Mirrors ``ClassifierTrainer``'s contract:
    __init__(cfg, system, system_name)
    fit(dataset_files, output_dir, resume_checkpoint=None) -> OutcomeModelHandle
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from adaptive_roa.data.adaptive_classification_data import AdaptiveClassificationDataModule
from adaptive_roa.predictors.bayesian_mlp import build_bayesian_mlp
from adaptive_roa.predictors.handles import OutcomeModelHandle
from adaptive_roa.predictors.posteriors import EnsemblePosterior


class _OutcomeModule(pl.LightningModule):
    """One posterior's ELBO: Bernoulli likelihood + KL(q||p)/N."""

    def __init__(self, posterior, system, pos_weight, lr, weight_decay, kl_weight, n_train):
        super().__init__()
        self.posterior = posterior
        self.system = system  # plain attr; methods are device-agnostic
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.kl_weight = float(kl_weight)
        self.n_train = max(int(n_train), 1)
        self.register_buffer("pos_weight", torch.as_tensor(float(pos_weight)))

    def forward(self, raw_states):
        embedded = self.system.embed_state_for_model(self.system.normalize_state(raw_states))
        return self.posterior.forward_sample(embedded)

    def _step(self, batch, stage: str):
        y = batch["label"].float().view(-1)
        logits = self(batch["inputs"]).view(-1)
        nll = F.binary_cross_entropy_with_logits(logits, y, pos_weight=self.pos_weight)
        # The ELBO's KL is a per-DATASET term while nll is a per-batch mean, so
        # scale by 1/N to put them on the same footing.
        kl = self.posterior.kl_divergence() / self.n_train
        loss = nll + self.kl_weight * kl
        self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log(f"{stage}_nll", nll, on_epoch=True, on_step=False)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=10, min_lr=1e-6
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "monitor": "val_loss"}}


class BayesianMLPTrainer:
    def __init__(self, cfg: Any, system: Any, system_name: str):
        self.cfg = cfg
        self.system = system
        self.system_name = system_name
        bnn = self._predictor_cfg.get("bnn", {})
        self.posterior_kind = str(bnn.get("posterior", "mfvi"))
        # beta: 1.0 is untempered. Any other value makes the run a TEMPERED
        # result and must be labelled as such in the writeup.
        self.kl_weight = float(bnn.get("kl_weight", 1.0))

    @property
    def _predictor_cfg(self):
        pred = self.cfg.get("predictor")
        return pred if pred is not None else self.cfg

    def _embedded_dim(self) -> int:
        dummy = torch.zeros(1, int(self.system.state_dim))
        return int(self.system.embed_state_for_model(
            self.system.normalize_state(dummy)).shape[-1])

    def _build(self, bnn, posterior_kind):
        return build_bayesian_mlp(
            input_dim=self._embedded_dim(),
            hidden_dims=list(bnn.get("hidden_dims", [256, 512, 256])),
            output_dim=1,
            posterior=posterior_kind,
            prior_sigma=float(bnn.get("prior_sigma", 1.0)),
            n_members=int(bnn.get("n_members", 5)),
            dropout=float(bnn.get("dropout", 0.0)),
            activation=str(bnn.get("activation", "relu")),
        )

    def _run_lightning(self, module, data_module, bnn, ckpt_dir, tag):
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        trainer_cfg = self._predictor_cfg.get("lightning_trainer", {})
        device = str(self._predictor_cfg.get("device", self.cfg.get("device", "cuda:0")))
        use_gpu = device.startswith("cuda") and torch.cuda.is_available()
        trainer = pl.Trainer(
            max_epochs=int(bnn.get("max_epochs", 200)),
            accelerator="gpu" if use_gpu else "cpu",
            devices=1,
            gradient_clip_val=trainer_cfg.get("gradient_clip_val", 1.0),
            log_every_n_steps=trainer_cfg.get("log_every_n_steps", 10),
            check_val_every_n_epoch=1,
            enable_progress_bar=False,
            enable_model_summary=False,
            callbacks=[
                ModelCheckpoint(
                    dirpath=str(ckpt_dir), monitor="val_loss", mode="min",
                    save_top_k=1, filename=f"{tag}-{{epoch:02d}}-{{val_loss:.4f}}",
                ),
                EarlyStopping(monitor="val_loss", mode="min",
                              patience=int(bnn.get("patience", 20))),
            ],
            logger=CSVLogger(save_dir=str(ckpt_dir.parent), name=f"bnn_logs_{tag}"),
        )
        trainer.fit(module, data_module)
        return use_gpu, device

    def fit(self, dataset_files: dict, output_dir: str, resume_checkpoint: str | None = None):
        bnn = self._predictor_cfg.get("bnn", {})
        ckpt_dir = Path(output_dir) / "checkpoints"

        data_module = AdaptiveClassificationDataModule(
            train_file=dataset_files["train"],
            val_file=dataset_files["val"],
            batch_size=self._predictor_cfg.get("batch_size", 1024),
            num_workers=0,  # in-memory tensors; workers break on NFS
        )
        data_module.setup()
        # The datamodule keeps its datasets on the private `_train` / `_val`
        # attrs (adaptive_classification_data.py:58-59); each dataset exposes
        # `.states` and `.labels` as full in-memory tensors.
        n_train = len(data_module._train)

        common = dict(
            system=self.system, pos_weight=data_module.pos_weight,
            lr=float(bnn.get("lr", 1e-3)),
            weight_decay=float(bnn.get("weight_decay", 1e-5)),
            n_train=n_train,
        )

        if self.posterior_kind == "ensemble":
            # Members must be independent: own seed, own optimizer, own shuffling.
            members, use_gpu, device = [], False, "cpu"
            for m in range(int(bnn.get("n_members", 5))):
                torch.manual_seed(int(bnn.get("seed", 0)) + m)
                member = self._build(bnn, "deterministic")
                module = _OutcomeModule(posterior=member, kl_weight=0.0, **common)
                use_gpu, device = self._run_lightning(
                    module, data_module, bnn, ckpt_dir / f"member_{m}", f"best_member{m}"
                )
                members.append(member)
            posterior = EnsemblePosterior(members)
            # The engine's warm start globs checkpoints/best*.ckpt (engine.py:140-142),
            # so the assembled ensemble needs one at the top level.
            torch.save({"state_dict": posterior.state_dict()}, ckpt_dir / "best-ensemble.ckpt")
        else:
            # "laplace" builds a body+head split; build_bayesian_mlp handles it.
            posterior = self._build(bnn, self.posterior_kind)
            kl_weight = self.kl_weight if self.posterior_kind == "mfvi" else 0.0
            module = _OutcomeModule(posterior=posterior, kl_weight=kl_weight, **common)
            if resume_checkpoint and Path(resume_checkpoint).exists():
                ckpt = torch.load(resume_checkpoint, map_location="cpu", weights_only=False)
                module.load_state_dict(ckpt["state_dict"], strict=False)
            use_gpu, device = self._run_lightning(module, data_module, bnn, ckpt_dir, "best")

            if self.posterior_kind == "laplace":
                # Post-hoc GGN over the full training set. Without this the arm
                # stays at the MAP point estimate and reports zero uncertainty.
                posterior.eval()
                with torch.no_grad():
                    x = data_module._train.states.float()
                    y = data_module._train.labels.float().view(-1)
                    embedded = self.system.embed_state_for_model(self.system.normalize_state(x))
                    posterior.fit(posterior.body(embedded), y, task="outcome")
                # `_cov` is a non-persistent buffer (its shape is unknown until
                # fit), so it will NOT round-trip through the Lightning
                # checkpoint. Save it explicitly or the exported arm silently
                # falls back to its MAP point estimate and reports zero spread.
                torch.save(posterior.posterior_covariance, ckpt_dir / "laplace_cov.pt")

        handle = OutcomeModelHandle(
            posterior, self.system,
            n_marginal_samples=int(bnn.get("n_marginal_samples", 64)),
            seed=int(bnn.get("seed", 0)),
        ).eval()
        return handle.to(device) if use_gpu else handle
```

Note the `pos_weight` handling matches `ClassifierTrainer` exactly: the datamodule
computes it during `setup()` (`adaptive_classification_data.py:63`), which is why
`setup()` is called eagerly before the module is built.

- [ ] **Step 4: Write the three configs**

Create `configs/adaptive_v2/predictor/bnn_mfvi.yaml`:

```yaml
# @package _global_
defaults:
  - /probability: classifier_prob
threshold:
  decision_rule: one_sided
calibration:
  decision_rule: one_sided
predictor:
  type: classifier          # family tag: drives dataset kind and eval branch
  name: bnn_mfvi            # arm name: keys the export registry
  trainer_target: adaptive_roa.adaptive_v2.trainers.bayesian_mlp_trainer.BayesianMLPTrainer
  batch_size: 1024
  val_batch_size: 2048
  bnn:
    posterior: mfvi
    hidden_dims: [256, 512, 256]
    lr: 1.0e-3
    weight_decay: 1.0e-5
    max_epochs: 200
    patience: 20
    prior_sigma: 1.0        # isotropic Gaussian prior, identical across arms
    kl_weight: 1.0          # beta = 1: untempered. Any other value is a tempered result.
    n_marginal_samples: 64
  lightning_trainer:
    gradient_clip_val: 1.0
    log_every_n_steps: 10
```

`bnn_ensemble.yaml` is identical with `name: bnn_ensemble`, `posterior: ensemble`, `n_members: 5`, and no `kl_weight`. `bnn_laplace.yaml` is identical with `name: bnn_laplace`, `posterior: laplace`, and no `kl_weight`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `./env/bin/python -m pytest tests/predictors/test_bnn_trainer.py -v`
Expected: PASS (9 tests)

- [ ] **Step 6: Verify the configs compose under Hydra**

Run:
```bash
./env/bin/python -c "
from hydra import compose, initialize_config_dir
import os
d = os.path.abspath('configs/adaptive_v2')
for arm in ['bnn_mfvi', 'bnn_ensemble', 'bnn_laplace']:
    with initialize_config_dir(config_dir=d, version_base=None):
        cfg = compose(config_name='default', overrides=[f'predictor={arm}'])
    assert cfg.predictor.name == arm, cfg.predictor.name
    assert cfg.predictor.type == 'classifier'
    print(arm, 'OK', cfg.probability._target_)
"
```
Expected: three `OK` lines naming `ClassifierProbabilityBackend`.

- [ ] **Step 7: Commit**

```bash
git add adaptive_roa/adaptive_v2/trainers/bayesian_mlp_trainer.py configs/adaptive_v2/predictor/ tests/predictors/test_bnn_trainer.py
git commit -m "feat(predictors): Bayesian MLP trainer and MFVI/ensemble/Laplace arm configs"
```

---

### Task 10: Export wrappers and end-to-end smoke

**Files:**
- Create: `adaptive_roa/probabilistic_classifier/bayesian.py`
- Modify: `adaptive_roa/probabilistic_classifier/__init__.py`
- Test: `tests/predictors/test_bnn_e2e.py`

**Interfaces:**
- Consumes: `BayesianMLPTrainer` (Task 9), registry from Task 3
- Produces: `BNNProbabilisticClassifier` subclasses registered as `bnn_mfvi`, `bnn_ensemble`, `bnn_laplace`

- [ ] **Step 1: Write the failing test**

Create `tests/predictors/test_bnn_e2e.py`:

```python
import pytest


@pytest.mark.parametrize("arm", ["bnn_mfvi", "bnn_ensemble", "bnn_laplace"])
def test_each_arm_is_registered_for_export(arm):
    from adaptive_roa.probabilistic_classifier import get_probabilistic_classifier_class

    cls = get_probabilistic_classifier_class(arm)
    assert cls.predictor_name == arm
    assert cls.predictor_type == "classifier"
    assert cls.native_probs == ("p_success",)


def test_arms_do_not_steal_the_legacy_classifier_alias():
    """A BNN run must never be loaded as a plain ClassifierMLP."""
    from adaptive_roa.probabilistic_classifier import get_probabilistic_classifier_class
    from adaptive_roa.probabilistic_classifier.classifier import (
        ClassifierProbabilisticClassifier,
    )

    assert get_probabilistic_classifier_class("classifier") is ClassifierProbabilisticClassifier
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./env/bin/python -m pytest tests/predictors/test_bnn_e2e.py -v`
Expected: FAIL — `KeyError: 'bnn_mfvi'`

- [ ] **Step 3: Write the export wrappers**

Create `adaptive_roa/probabilistic_classifier/bayesian.py`:

```python
"""Export wrappers for the Bayesian MLP outcome arms.

Each arm registers under its own name so a BNN run is never loaded as a plain
``ClassifierMLP`` -- the checkpoints are structurally different and the mis-load
would be silent.
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import torch

from adaptive_roa.adaptive_v2.types import OutcomeProbabilities
from adaptive_roa.predictors.bayesian_mlp import build_bayesian_mlp
from adaptive_roa.predictors.handles import OutcomeModelHandle
from .base import ProbabilisticClassifier
from .registry import register_probabilistic_classifier


class BNNProbabilisticClassifier(ProbabilisticClassifier):
    """Shared loader; subclasses only set ``predictor_name`` and the posterior."""

    predictor_type = "classifier"
    native_probs = ("p_success",)
    posterior_kind = ""

    def __init__(self, handle, system, device):
        self.handle = handle
        self.system = system
        self.device = device

    def predict(self, states: np.ndarray) -> OutcomeProbabilities:
        out = []
        with torch.no_grad():
            for i in range(0, len(states), 8192):
                x = torch.as_tensor(states[i:i + 8192], dtype=torch.float32, device=self.device)
                out.append(torch.sigmoid(self.handle(x).view(-1)).double().cpu().numpy())
        p_success = np.concatenate(out) if out else np.zeros(0)
        return OutcomeProbabilities(
            p_success=p_success,
            p_failure=1.0 - p_success,
            p_invalid=np.zeros_like(p_success),
        )

    @classmethod
    def load_from_run(cls, run_dir, epoch, cfg, system, device="cuda"):
        bnn = cfg.get("predictor", {}).get("bnn", {})
        dummy = torch.zeros(1, int(system.state_dim))
        input_dim = int(system.embed_state_for_model(system.normalize_state(dummy)).shape[-1])
        posterior = build_bayesian_mlp(
            input_dim=input_dim,
            hidden_dims=list(bnn.get("hidden_dims", [256, 512, 256])),
            output_dim=1,
            posterior=cls.posterior_kind,
            prior_sigma=float(bnn.get("prior_sigma", 1.0)),
            n_members=int(bnn.get("n_members", 5)),
            dropout=float(bnn.get("dropout", 0.0)),
            activation=str(bnn.get("activation", "relu")),
        )
        ckpt_dir = Path(run_dir) / f"epoch_{epoch:03d}" / "checkpoints"
        best = sorted(glob.glob(str(ckpt_dir / "best*.ckpt")))
        if not best:
            raise FileNotFoundError(f"no BNN checkpoint in {ckpt_dir}")
        ckpt = torch.load(best[0], map_location="cpu", weights_only=False)
        state = ckpt.get("state_dict", ckpt)
        # Lightning prefixes module attrs; strip "posterior." when present.
        state = {k[len("posterior."):] if k.startswith("posterior.") else k: v
                 for k, v in state.items()}
        posterior.load_state_dict(state, strict=False)

        # The Laplace GGN covariance is a non-persistent buffer, so the trainer
        # writes it alongside the checkpoint. Without it the arm would load as a
        # MAP point estimate and silently report zero epistemic uncertainty.
        cov_path = ckpt_dir / "laplace_cov.pt"
        if cls.posterior_kind == "laplace":
            if not cov_path.exists():
                raise FileNotFoundError(
                    f"{cov_path} missing; a Laplace arm without its covariance is "
                    "a MAP model and must not be reported as Bayesian."
                )
            posterior._cov = torch.load(cov_path, map_location="cpu")

        handle = OutcomeModelHandle(
            posterior, system,
            n_marginal_samples=int(bnn.get("n_marginal_samples", 64)),
            seed=int(bnn.get("seed", 0)),
        ).eval().to(device)
        return cls(handle, system, device)


@register_probabilistic_classifier
class MFVIProbabilisticClassifier(BNNProbabilisticClassifier):
    predictor_name = "bnn_mfvi"
    posterior_kind = "mfvi"


@register_probabilistic_classifier
class EnsembleProbabilisticClassifier(BNNProbabilisticClassifier):
    predictor_name = "bnn_ensemble"
    posterior_kind = "ensemble"


@register_probabilistic_classifier
class LaplaceProbabilisticClassifier(BNNProbabilisticClassifier):
    predictor_name = "bnn_laplace"
    posterior_kind = "laplace"
```

Add `from . import bayesian as _bayesian  # noqa: F401,E402` to
`adaptive_roa/probabilistic_classifier/__init__.py` alongside the existing
side-effect imports.

Add this round-trip test to `tests/predictors/test_bnn_e2e.py` so the covariance
persistence is actually verified rather than assumed:

```python
def test_laplace_export_round_trips_its_covariance(tmp_path):
    """A Laplace arm that loses its GGN covariance is a MAP model wearing a
    Bayesian label -- the failure is silent, so it needs an explicit test."""
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from adaptive_roa.adaptive_v2.trainers.bayesian_mlp_trainer import BayesianMLPTrainer
    from adaptive_roa.probabilistic_classifier.bayesian import LaplaceProbabilisticClassifier
    from adaptive_roa.systems.cartpole import CartPoleSystem

    rng = np.random.default_rng(0)
    for split in ("train", "val"):
        X = rng.uniform(-1.0, 1.0, size=(256, 4))
        np.savetxt(tmp_path / f"{split}.txt",
                   np.column_stack([X, (np.abs(X[:, 1]) < 0.5).astype(int)]))

    cfg = OmegaConf.create({
        "device": "cpu",
        "predictor": {"type": "classifier", "name": "bnn_laplace", "batch_size": 64,
                      "bnn": {"posterior": "laplace", "hidden_dims": [16, 16],
                              "max_epochs": 2, "n_marginal_samples": 8}},
    })
    run_dir = tmp_path / "run"
    epoch_dir = run_dir / "epoch_000"
    trainer = BayesianMLPTrainer(cfg, CartPoleSystem(), "cartpole")
    fitted = trainer.fit(
        {"train": str(tmp_path / "train.txt"), "val": str(tmp_path / "val.txt")},
        str(epoch_dir),
    )
    loaded = LaplaceProbabilisticClassifier.load_from_run(
        str(run_dir), 0, cfg, CartPoleSystem(), device="cpu"
    )
    assert loaded.handle.posterior.is_fitted
    torch.testing.assert_close(
        loaded.handle.posterior.posterior_covariance,
        fitted.posterior.posterior_covariance,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./env/bin/python -m pytest tests/predictors/test_bnn_e2e.py -v`
Expected: PASS (5 tests — 3 parametrized registration + alias + covariance round-trip)

- [ ] **Step 5: Run a pipeline smoke test for one arm**

Run:
```bash
./env/bin/python scripts/run_adaptive.py \
  --config-name=default system=pendulum predictor=bnn_mfvi \
  sampling_mode=ranked +adaptive_v2.smoke_mode=true \
  +trainer.limit_train_batches=1 +trainer.limit_val_batches=1 \
  adaptive_v2.max_epochs=2
```
Expected: two epochs complete; `epoch_000/artifacts_v2.json` and `epoch_001/artifacts_v2.json` exist. If the run fails on dataset paths for pendulum, substitute `system=cartpole_pybullet`.

- [ ] **Step 6: Run the full suite for regressions**

Run: `./env/bin/python -m pytest tests -q`
Expected: PASS. Compare the count against `git stash && ./env/bin/python -m pytest tests -q` if anything looks off.

- [ ] **Step 7: Commit**

```bash
git add adaptive_roa/probabilistic_classifier/ tests/predictors/test_bnn_e2e.py
git commit -m "feat(export): probabilistic-classifier wrappers for the three BNN arms"
```

---

## Self-Review

**Spec coverage for this plan's scope:**

| Spec requirement | Task |
|---|---|
| `PredictorTrainer` corrected; handle Protocols added | 1 |
| `GPProbabilityBackend` gains `.estimator` | 2 |
| `predictor.name` keys the export registry | 3 |
| MFVI posterior | 4 |
| Ensemble posterior, `n_members = 5` | 5 |
| Last-layer Laplace, `sigma` required | 6 |
| `BayesianMLP` backbone | 7 |
| Outcome head; seeded internal marginalization | 8 |
| Three arm configs; `beta = 1` headline; isotropic prior | 9 |
| Export wrappers; no silent mis-load | 10 |

**Deferred to later plans (deliberately, not gaps):** final-state heads and the manifold likelihood, `gp_reg`, `mlp_det`, β-NLL, MDN, HMC and the two-tier split, `get_manifold_component_names()` on final-state handles, the random-acquisition control, and separatrix-conditioned reporting. Fix #4 from the spec (`get_manifold_component_names`) belongs to Plan 2 because nothing in this plan produces a final-state handle.

**Type consistency check:** `forward_sample(x, generator=None)`, `forward_samples(x, S, generator=None)`, and `kl_divergence()` are used identically in Tasks 4-8. `build_bayesian_mlp(...)` returns a `Posterior` in Tasks 7-9. `OutcomeModelHandle(posterior, system, n_marginal_samples, seed)` is constructed identically in Tasks 8-10. `predictor_name` is introduced in Task 3 and consumed in Task 10.

## Remaining plans in this sequence

- **Plan 2 — Final-state family:** manifold-aware `FinalStateHead` (`Real`/`SO2`/`SO3` likelihoods), β-NLL, optional MDN, `FinalStateModelHandle` (fresh draw per call, plus `get_manifold_component_names`), `gp_reg` multi-output SVGP, `mlp_det` baseline, the `K >= 2M` guard, and the degeneracy guard test.
- **Plan 3 — HMC reference tier:** hand-rolled leapfrog with dual-averaging, tanh/GELU backbone, the `[50,50]` reference configs across all four systems, function-space R-hat and the HMC-vs-HMC ceiling artifact.
- **Plan 4 — Benchmark orchestration:** paired random-acquisition controls, matched training budgets, prior-scale sensitivity sweep, separatrix-conditioned metric reporting.
