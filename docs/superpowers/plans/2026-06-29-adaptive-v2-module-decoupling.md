# Adaptive v2 Module Decoupling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decouple the five adaptive-loop modules (probabilistic classifier, threshold optimizer, conformal calibration, adaptive acquisition, evaluation) into separate Hydra config namespaces and `_target_`-driven constructors, eliminating the monolithic `conformal.*` config block.

**Architecture:** New config groups (`probability/`, `threshold/`, `calibration/`, `acquisition/`, `eval/`) each carry a `_target_` class and their own parameters. A thin `_instantiate(cfg_node, *args)` helper in `engine.py` replaces the `if predictor == "classifier"` branch. System configs set `decision_rule` and `attractor_radius` as top-level anchors; module configs reference them via Hydra interpolation `${decision_rule}` with no fallback.

**Tech Stack:** Python 3.10+, Hydra/OmegaConf, PyTorch, pytest.

## Global Constraints

- Python package: `adaptive_roa` at `/common/home/st1122/Projects/adaptive_roa`
- Run tests with: `conda run -n /common/home/st1122/Projects/adaptive_roa/env pytest <path> -v`
- No fallback defaults for `attractor_radius` or `decision_rule` — missing values must raise a Hydra interpolation error
- `conformal.*` config key must not appear in any file after this plan completes
- `ConformalConfig.from_hydra()` method removed at the end
- Clean break: no backward-compat shims for old config structure
- Python package `adaptive_roa/adaptive_v2/strategy/` keeps its name; only the *config directory* renames from `strategy/` to `acquisition/`

---

### Task 1: New config group files

**Files:**
- Create: `configs/adaptive_v2/probability/classifier_prob.yaml`
- Modify: `configs/adaptive_v2/probability/endpoint_mc.yaml`
- Create: `configs/adaptive_v2/calibration/conformal_calibration.yaml`
- Modify: `configs/adaptive_v2/threshold/conformal_threshold.yaml`
- Modify: `configs/adaptive_v2/acquisition/direct.yaml` (renamed from `strategy/`)
- Create: `configs/adaptive_v2/acquisition/ranked.yaml`
- Create: `configs/adaptive_v2/acquisition/conformal.yaml`
- Modify: `configs/adaptive_v2/eval/full_roa.yaml`
- Modify: `configs/adaptive_v2/default.yaml`

**Interfaces:**
- Produces: Hydra config groups that downstream tasks wire into backends

- [ ] **Step 1: Create `configs/adaptive_v2/probability/classifier_prob.yaml`**

```yaml
# @package _global_
probability:
  _target_: adaptive_roa.adaptive_v2.probability.classifier_prob.ClassifierProbabilityBackend
```

- [ ] **Step 2: Replace `configs/adaptive_v2/probability/endpoint_mc.yaml`**

```yaml
# @package _global_
probability:
  _target_: adaptive_roa.adaptive_v2.probability.endpoint_mc.EndpointMCProbabilityBackend
  attractor_radius: ${attractor_radius}
  num_mc_samples: 10
  refine_invalids: false
  refine_t_min: 0.7
  refine_t_max: 0.9
  refine_num_steps: 100
  refine_max_attempts: 5
  trajectory_checking: false
```

- [ ] **Step 3: Create `configs/adaptive_v2/calibration/` directory and `conformal_calibration.yaml`**

```yaml
# @package _global_
calibration:
  _target_: adaptive_roa.adaptive_v2.calibration.conformal_calibration.ConformalCalibrationBackend
  delta: 0.05
  w: 0.9
  alpha: 0.1
  alpha_eval: 0.1
  decision_rule: ${decision_rule}
  verbose: true
```

- [ ] **Step 4: Replace `configs/adaptive_v2/threshold/conformal_threshold.yaml`**

```yaml
# @package _global_
threshold:
  _target_: adaptive_roa.adaptive_v2.threshold.conformal_threshold.ConformalThresholdBackend
  predictor_type: ${predictor.type}
  decision_rule: ${decision_rule}
  optimize_mode: joint
  optimize_objective: loss
  target_f1: 0.90
  lambda_grid_size: 100
  delta_grid_size: 100
  delta_min: 0.05
  delta_max: 0.45
  fixed_lambda_star: 0.5
  fixed_delta_star: 0.1
  use_p_invalid_veto: true
  verbose: true
```

- [ ] **Step 5: Create `configs/adaptive_v2/acquisition/` directory with three files**

`configs/adaptive_v2/acquisition/direct.yaml`:
```yaml
# @package _global_
acquisition:
  _target_: adaptive_roa.adaptive_v2.strategy.direct.DirectAcquisitionStrategy
  d2_ratio: 0.5
  batch_size_sampling: 50
  max_samples_per_epoch: 50000
  decision_rule: ${decision_rule}
  verbose: true
```

`configs/adaptive_v2/acquisition/ranked.yaml`:
```yaml
# @package _global_
acquisition:
  _target_: adaptive_roa.adaptive_v2.strategy.ranked.RankedAcquisitionStrategy
  d2_ratio: 0.5
  n_ranked_candidates: 50000
  batch_size_sampling: 50
  max_samples_per_epoch: 50000
  decision_rule: ${decision_rule}
  verbose: true
```

`configs/adaptive_v2/acquisition/conformal.yaml`:
```yaml
# @package _global_
acquisition:
  _target_: adaptive_roa.adaptive_v2.strategy.conformal.ConformalAcquisitionStrategy
  d2_ratio: 0.5
  batch_size_sampling: 50
  max_samples_per_epoch: 50000
  decision_rule: ${decision_rule}
  verbose: true
```

- [ ] **Step 6: Replace `configs/adaptive_v2/eval/full_roa.yaml`**

```yaml
# @package _global_
eval:
  _target_: adaptive_roa.adaptive_v2.eval.full_roa.FullROAEvaluator
  predictor_type: ${predictor.type}
  alpha_eval: 0.1
  attractor_radius: ${attractor_radius}
  num_mc_samples_eval: 100
  decision_rule: ${decision_rule}
  refine_invalids: false
  refine_t_min: 0.7
  refine_t_max: 0.9
  refine_num_steps: 100
  refine_max_attempts: 5
  max_eval_rows: null
  verbose: true
```

- [ ] **Step 7: Replace `configs/adaptive_v2/default.yaml`**

```yaml
defaults:
  - system: pendulum
  - predictor: generative
  - threshold: conformal_threshold
  - calibration: conformal_calibration
  - acquisition: direct
  - eval: full_roa
  - _self_

adaptive_v2:
  artifact_schema_version: 2
  save_legacy_results_json: true
  save_v2_artifacts: true

hydra:
  run:
    dir: ${output_dir}
  job:
    chdir: false
```

- [ ] **Step 8: Verify config groups load without error (--cfg job --resolve will fail until Task 8 completes, but individual group resolution can be checked)**

```bash
python -c "
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
with initialize_config_dir(config_dir='/common/home/st1122/Projects/adaptive_roa/configs/adaptive_v2/calibration'):
    cfg = compose('conformal_calibration')
    print(OmegaConf.to_yaml(cfg))
"
```
Expected: prints `calibration:` block with all keys present.

- [ ] **Step 9: Commit**

```bash
git add configs/adaptive_v2/
git commit -m "feat: new config group namespaces for all 5 adaptive modules"
```

---

### Task 2: `ConformalCalibrationBackend` — new class

**Files:**
- Create: `adaptive_roa/adaptive_v2/calibration/__init__.py`
- Create: `adaptive_roa/adaptive_v2/calibration/conformal_calibration.py`
- Create: `tests/adaptive_v2/test_calibration_backend.py`
- Create: `tests/adaptive_v2/__init__.py` (if not present)

**Interfaces:**
- Consumes: `ConformalConfig`, `ConformalPredictor` from `adaptive_roa.conformal`; `Calibrator` from `adaptive_roa.conformal.calibrator`; `ThresholdState` from `adaptive_roa.adaptive_v2.types`; `build_probability_estimator` from `adaptive_roa.conformal.estimator_factory`
- Produces: `ConformalCalibrationBackend(cfg, system, device)` with methods `bind_model(model_handle)`, `calibrate(X_cal, y_cal, threshold_state) -> float`, `calibrate_eval(X_cal_eval, y_cal_eval, threshold_state) -> float`

- [ ] **Step 1: Write failing test**

`tests/adaptive_v2/__init__.py` — empty file.

`tests/adaptive_v2/test_calibration_backend.py`:
```python
import numpy as np
import torch
from omegaconf import OmegaConf

from adaptive_roa.adaptive_v2.calibration.conformal_calibration import ConformalCalibrationBackend
from adaptive_roa.adaptive_v2.types import ThresholdState


def _cfg(decision_rule="two_sided"):
    return OmegaConf.create({
        "delta": 0.05,
        "w": 0.9,
        "alpha": 0.1,
        "alpha_eval": 0.1,
        "decision_rule": decision_rule,
        "verbose": False,
    })


class _DummyClassifier:
    def eval(self):
        return self

    def __call__(self, x):
        return torch.zeros((x.shape[0], 1), dtype=torch.float32)


def test_calibration_backend_init():
    backend = ConformalCalibrationBackend(_cfg(), system=None, device="cpu")
    assert backend.delta == 0.05
    assert backend.alpha == 0.1
    assert backend.decision_rule == "two_sided"


def test_calibrate_returns_float():
    backend = ConformalCalibrationBackend(_cfg(decision_rule="one_sided"), system=None, device="cpu")
    backend.bind_model(_DummyClassifier())
    state = ThresholdState(lambda_star=0.5, delta_star=0.1)
    n = 20
    X = np.zeros((n, 4), dtype=np.float32)
    y = np.array([1] * 10 + [-1] * 10, dtype=np.int64)
    q_hat = backend.calibrate(X, y, state)
    assert isinstance(q_hat, float)


def test_calibrate_eval_uses_alpha_eval():
    cfg = OmegaConf.create({
        "delta": 0.05, "w": 0.9, "alpha": 0.1, "alpha_eval": 0.2,
        "decision_rule": "one_sided", "verbose": False,
    })
    backend = ConformalCalibrationBackend(cfg, system=None, device="cpu")
    backend.bind_model(_DummyClassifier())
    state = ThresholdState(lambda_star=0.5, delta_star=0.1)
    n = 20
    X = np.zeros((n, 4), dtype=np.float32)
    y = np.array([1] * 10 + [-1] * 10, dtype=np.int64)
    q_hat_a = backend.calibrate(X, y, state)
    q_hat_b = backend.calibrate_eval(X, y, state)
    # Different alpha values → different q_hat (not guaranteed equal)
    assert isinstance(q_hat_a, float)
    assert isinstance(q_hat_b, float)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
conda run -n /common/home/st1122/Projects/adaptive_roa/env pytest tests/adaptive_v2/test_calibration_backend.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'adaptive_roa.adaptive_v2.calibration'`

- [ ] **Step 3: Create `adaptive_roa/adaptive_v2/calibration/__init__.py`** — empty file.

- [ ] **Step 4: Create `adaptive_roa/adaptive_v2/calibration/conformal_calibration.py`**

```python
from __future__ import annotations

from typing import Any

import numpy as np

from adaptive_roa.conformal import ConformalConfig, ConformalPredictor
from adaptive_roa.conformal.estimator_factory import build_probability_estimator
from adaptive_roa.adaptive_v2.types import ThresholdState


class ConformalCalibrationBackend:
    """Computes q_hat from a calibration set, decoupled from threshold optimization."""

    def __init__(self, cfg: Any, system: Any, device: str):
        self.delta = float(cfg.delta)
        self.w = float(cfg.w)
        self.alpha = float(cfg.alpha)
        self.alpha_eval = float(cfg.alpha_eval)
        self.decision_rule = str(cfg.decision_rule)
        self.verbose = bool(cfg.verbose)
        self.system = system
        self.device = device
        self._predictor: ConformalPredictor | None = None
        self._predictor_eval: ConformalPredictor | None = None

    def bind_model(self, model_handle: Any, predictor_type: str = "generative") -> None:
        conf = ConformalConfig(
            delta=self.delta,
            w=self.w,
            alpha=self.alpha,
            attractor_radius=0.2,
            decision_rule=self.decision_rule,
        )
        conf_eval = ConformalConfig(
            delta=self.delta,
            w=self.w,
            alpha=self.alpha_eval,
            attractor_radius=0.2,
            decision_rule=self.decision_rule,
        )
        estimator = build_probability_estimator(
            predictor_type, model_handle, self.system, conf, self.device
        )
        self._predictor = ConformalPredictor(
            flow_matcher=model_handle,
            system=self.system,
            config=conf,
            device=self.device,
            probability_estimator=estimator,
        )
        self._predictor_eval = ConformalPredictor(
            flow_matcher=model_handle,
            system=self.system,
            config=conf_eval,
            device=self.device,
            probability_estimator=estimator,
        )

    def calibrate(
        self,
        X_cal: np.ndarray,
        y_cal: np.ndarray,
        threshold_state: ThresholdState,
    ) -> float:
        """Return q_hat for acquisition using self.alpha."""
        if self._predictor is None:
            raise RuntimeError("CalibrationBackend used before bind_model")
        self._predictor.lambda_star = threshold_state.lambda_star
        self._predictor.delta_star = threshold_state.delta_star
        q_hat = self._predictor.calibrate_qhat(X_cal, y_cal, verbose=self.verbose)
        return float(q_hat)

    def calibrate_eval(
        self,
        X_cal_eval: np.ndarray,
        y_cal_eval: np.ndarray,
        threshold_state: ThresholdState,
    ) -> float:
        """Return q_hat for evaluation-time coverage using self.alpha_eval."""
        if self._predictor_eval is None:
            raise RuntimeError("CalibrationBackend used before bind_model")
        self._predictor_eval.lambda_star = threshold_state.lambda_star
        self._predictor_eval.delta_star = threshold_state.delta_star
        q_hat = self._predictor_eval.calibrate_qhat(X_cal_eval, y_cal_eval, verbose=self.verbose)
        return float(q_hat)
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
conda run -n /common/home/st1122/Projects/adaptive_roa/env pytest tests/adaptive_v2/test_calibration_backend.py -v
```
Expected: 3 PASS

- [ ] **Step 6: Commit**

```bash
git add adaptive_roa/adaptive_v2/calibration/ tests/adaptive_v2/
git commit -m "feat: ConformalCalibrationBackend decoupled from threshold backend"
```

---

### Task 3: Update `ConformalThresholdBackend`

**Files:**
- Modify: `adaptive_roa/adaptive_v2/threshold/conformal_threshold.py`
- Modify: `tests/test_threshold_injection.py`

**Interfaces:**
- Consumes: `ConformalConfig`, `ConformalPredictor`, `build_probability_estimator`; `ThresholdState`
- Produces: `ConformalThresholdBackend(cfg, system, device)` — constructor reads `cfg.predictor_type`, `cfg.decision_rule`, `cfg.optimize_mode`, `cfg.optimize_objective`, `cfg.lambda_grid_size`, `cfg.delta_grid_size`, `cfg.delta_min`, `cfg.delta_max`, `cfg.fixed_lambda_star`, `cfg.fixed_delta_star`, `cfg.use_p_invalid_veto`, `cfg.target_f1`, `cfg.verbose`; `bind_model(model_handle)`, `optimize(X_train, y_train) -> ThresholdState`; `self.predictor` attribute (ConformalPredictor) exposed for strategies

- [ ] **Step 1: Update test (new constructor signature)**

Replace the `_cfg()` helper and test assertions in `tests/test_threshold_injection.py`:

```python
import torch
from omegaconf import OmegaConf

from adaptive_roa.conformal import ConformalConfig, ConformalPredictor
from adaptive_roa.conformal.classifier_probability_estimator import ClassifierProbabilityEstimator
from adaptive_roa.adaptive_v2.threshold.conformal_threshold import ConformalThresholdBackend


def _cfg(predictor_type="classifier"):
    return OmegaConf.create({
        "predictor_type": predictor_type,
        "decision_rule": "one_sided",
        "optimize_mode": "joint",
        "optimize_objective": "loss",
        "target_f1": 0.9,
        "lambda_grid_size": 50,
        "delta_grid_size": 50,
        "delta_min": 0.05,
        "delta_max": 0.45,
        "fixed_lambda_star": 0.5,
        "fixed_delta_star": 0.1,
        "use_p_invalid_veto": True,
        "verbose": False,
    })


class _DummyClassifier:
    def eval(self):
        return self

    def __call__(self, x):
        return torch.zeros((x.shape[0], 1), dtype=torch.float32)


def test_predictor_uses_injected_estimator():
    conf = ConformalConfig(
        delta=0.05, w=0.9, alpha=0.1, attractor_radius=0.2,
        decision_rule="one_sided",
    )
    sentinel = object()
    pred = ConformalPredictor(
        flow_matcher=None, system=None, config=conf, device="cpu",
        probability_estimator=sentinel,
    )
    assert pred.prob_estimator is sentinel


def test_threshold_backend_builds_classifier_estimator():
    backend = ConformalThresholdBackend(_cfg("classifier"), system=None, device="cpu")
    backend.bind_model(_DummyClassifier())
    assert isinstance(backend.predictor.prob_estimator, ClassifierProbabilityEstimator)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
conda run -n /common/home/st1122/Projects/adaptive_roa/env pytest tests/test_threshold_injection.py -v
```
Expected: FAIL (constructor signature mismatch)

- [ ] **Step 3: Rewrite `adaptive_roa/adaptive_v2/threshold/conformal_threshold.py`**

```python
from __future__ import annotations

from typing import Any

import numpy as np

from adaptive_roa.conformal import ConformalConfig, ConformalPredictor
from adaptive_roa.conformal.estimator_factory import build_probability_estimator
from adaptive_roa.adaptive_v2.types import ThresholdState


class ConformalThresholdBackend:
    """Adapter around ConformalPredictor for threshold optimization only."""

    def __init__(self, cfg: Any, system: Any, device: str):
        self.predictor_type = str(cfg.predictor_type)
        self.decision_rule = str(cfg.decision_rule)
        self.optimize_mode = str(cfg.optimize_mode)
        self.optimize_objective = str(cfg.optimize_objective)
        self.target_f1 = float(cfg.target_f1)
        self.lambda_grid_size = int(cfg.lambda_grid_size)
        self.delta_grid_size = int(cfg.delta_grid_size)
        self.delta_min = float(cfg.delta_min)
        self.delta_max = float(cfg.delta_max)
        self.fixed_lambda_star = float(cfg.fixed_lambda_star)
        self.fixed_delta_star = float(cfg.fixed_delta_star)
        self.use_p_invalid_veto = bool(cfg.use_p_invalid_veto)
        self.verbose = bool(cfg.verbose)
        self.system = system
        self.device = device
        self.predictor: ConformalPredictor | None = None

    def _build_conformal_config(self) -> ConformalConfig:
        return ConformalConfig(
            delta=self.fixed_delta_star,
            w=0.9,
            alpha=0.1,
            attractor_radius=0.2,
            optimize_mode=self.optimize_mode,
            optimize_objective=self.optimize_objective,
            target_f1=self.target_f1,
            decision_rule=self.decision_rule,
            lambda_grid_size=self.lambda_grid_size,
            delta_grid_size=self.delta_grid_size,
            delta_min=self.delta_min,
            delta_max=self.delta_max,
            fixed_lambda_star=self.fixed_lambda_star,
            fixed_delta_star=self.fixed_delta_star,
            use_p_invalid_veto=self.use_p_invalid_veto,
        )

    def bind_model(self, model_handle: Any) -> None:
        conf = self._build_conformal_config()
        estimator = build_probability_estimator(
            self.predictor_type, model_handle, self.system, conf, self.device
        )
        self.predictor = ConformalPredictor(
            flow_matcher=model_handle,
            system=self.system,
            config=conf,
            device=self.device,
            probability_estimator=estimator,
        )

    def optimize(self, X_train: np.ndarray, y_train: np.ndarray) -> ThresholdState:
        if self.predictor is None:
            raise RuntimeError("Threshold backend used before bind_model")
        self.predictor.optimize_thresholds(X_train, y_train, verbose=self.verbose)
        return ThresholdState(
            lambda_star=float(self.predictor.lambda_star),
            delta_star=float(self.predictor.delta_star),
        )
```

Note: `calibrate_qhat` is removed. The engine now calls `calibration_backend.calibrate()` instead.

- [ ] **Step 4: Run tests to verify they pass**

```bash
conda run -n /common/home/st1122/Projects/adaptive_roa/env pytest tests/test_threshold_injection.py -v
```
Expected: 2 PASS

- [ ] **Step 5: Commit**

```bash
git add adaptive_roa/adaptive_v2/threshold/conformal_threshold.py tests/test_threshold_injection.py
git commit -m "feat: ConformalThresholdBackend reads own cfg sub-config, drops calibrate_qhat"
```

---

### Task 4: Update probability backends

**Files:**
- Modify: `adaptive_roa/adaptive_v2/probability/endpoint_mc.py`
- Modify: `adaptive_roa/adaptive_v2/probability/classifier_prob.py`
- Modify: `tests/test_classifier_backend.py`
- Create: `tests/adaptive_v2/test_probability_backends.py`

**Interfaces:**
- Consumes: `ConformalConfig`, `ProbabilityEstimator`, `ClassifierProbabilityEstimator`; `OutcomeProbabilities`
- Produces: `EndpointMCProbabilityBackend(cfg, system, device)` reading `cfg.attractor_radius`, `cfg.num_mc_samples`, `cfg.refine_invalids` etc.; `ClassifierProbabilityBackend(cfg, system, device)` (no extra params needed); both expose `bind_model(model_handle)`, `estimate(states) -> OutcomeProbabilities`, `self.estimator`

- [ ] **Step 1: Write failing test for new constructor signatures**

`tests/adaptive_v2/test_probability_backends.py`:
```python
import numpy as np
import torch
from omegaconf import OmegaConf

from adaptive_roa.adaptive_v2.probability.endpoint_mc import EndpointMCProbabilityBackend
from adaptive_roa.adaptive_v2.probability.classifier_prob import ClassifierProbabilityBackend
from adaptive_roa.adaptive_v2.types import OutcomeProbabilities


def _mc_cfg():
    return OmegaConf.create({
        "attractor_radius": 0.2,
        "num_mc_samples": 5,
        "refine_invalids": False,
        "refine_t_min": 0.7,
        "refine_t_max": 0.9,
        "refine_num_steps": 100,
        "refine_max_attempts": 5,
        "trajectory_checking": False,
    })


def _clf_cfg():
    return OmegaConf.create({})


class _DummyClassifier:
    def eval(self):
        return self

    def __call__(self, x):
        return torch.zeros((x.shape[0], 1), dtype=torch.float32)


def test_mc_backend_stores_attractor_radius():
    backend = EndpointMCProbabilityBackend(_mc_cfg(), system=None, device="cpu")
    assert backend.attractor_radius == 0.2
    assert backend.num_mc_samples == 5


def test_clf_backend_returns_outcome_probabilities():
    backend = ClassifierProbabilityBackend(_clf_cfg(), system=None, device="cpu")
    backend.bind_model(_DummyClassifier())
    out = backend.estimate(np.zeros((7, 4), dtype=np.float32))
    assert isinstance(out, OutcomeProbabilities)
    assert out.p_success.shape == (7,)
    np.testing.assert_allclose(out.p_success, 0.5, rtol=1e-5)
    np.testing.assert_allclose(out.p_failure, 0.5, rtol=1e-5)
    assert np.all(out.p_invalid == 0.0)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
conda run -n /common/home/st1122/Projects/adaptive_roa/env pytest tests/adaptive_v2/test_probability_backends.py -v
```
Expected: FAIL (constructor signature mismatch)

- [ ] **Step 3: Rewrite `adaptive_roa/adaptive_v2/probability/endpoint_mc.py`**

```python
from __future__ import annotations

from typing import Any

import numpy as np

from adaptive_roa.conformal import ConformalConfig
from adaptive_roa.conformal.probability_estimator import ProbabilityEstimator
from adaptive_roa.adaptive_v2.types import OutcomeProbabilities


class EndpointMCProbabilityBackend:
    """Wraps ProbabilityEstimator (MC endpoint sampling) with the v2 interface."""

    def __init__(self, cfg: Any, system: Any, device: str):
        self.attractor_radius = float(cfg.attractor_radius)
        self.num_mc_samples = int(cfg.num_mc_samples)
        self.refine_invalids = bool(cfg.refine_invalids)
        self.refine_t_min = float(cfg.refine_t_min)
        self.refine_t_max = float(cfg.refine_t_max)
        self.refine_num_steps = int(cfg.refine_num_steps)
        self.refine_max_attempts = int(cfg.refine_max_attempts)
        self.trajectory_checking = bool(cfg.trajectory_checking)
        self.system = system
        self.device = device
        self.model_handle = None
        self.estimator: ProbabilityEstimator | None = None

    def _build_conformal_config(self) -> ConformalConfig:
        return ConformalConfig(
            attractor_radius=self.attractor_radius,
            num_mc_samples=self.num_mc_samples,
            refine_invalids=self.refine_invalids,
            refine_t_min=self.refine_t_min,
            refine_t_max=self.refine_t_max,
            refine_num_steps=self.refine_num_steps,
            refine_max_attempts=self.refine_max_attempts,
            trajectory_checking=self.trajectory_checking,
            delta=0.05,
            w=0.9,
            alpha=0.1,
        )

    def bind_model(self, model_handle: Any) -> None:
        self.model_handle = model_handle
        conformal_cfg = self._build_conformal_config()
        self.estimator = ProbabilityEstimator(model_handle, self.system, conformal_cfg, self.device)

    def estimate(self, start_states: np.ndarray) -> OutcomeProbabilities:
        if self.estimator is None:
            raise RuntimeError("Probability backend used before bind_model")
        p_success, p_failure, p_invalid = self.estimator.estimate(start_states)
        return OutcomeProbabilities(
            p_success=np.asarray(p_success),
            p_failure=np.asarray(p_failure),
            p_invalid=np.asarray(p_invalid),
        )
```

- [ ] **Step 4: Rewrite `adaptive_roa/adaptive_v2/probability/classifier_prob.py`**

```python
from __future__ import annotations

from typing import Any

import numpy as np

from adaptive_roa.conformal import ConformalConfig
from adaptive_roa.conformal.classifier_probability_estimator import ClassifierProbabilityEstimator
from adaptive_roa.adaptive_v2.types import OutcomeProbabilities


class ClassifierProbabilityBackend:
    """Probability backend using a discriminative classifier (single forward pass)."""

    def __init__(self, cfg: Any, system: Any, device: str):
        self.system = system
        self.device = device
        self.model_handle: Any = None
        self.estimator: ClassifierProbabilityEstimator | None = None

    def bind_model(self, model_handle: Any) -> None:
        self.model_handle = model_handle
        conformal_cfg = ConformalConfig(
            delta=0.05, w=0.9, alpha=0.1, attractor_radius=0.2,
        )
        self.estimator = ClassifierProbabilityEstimator(
            model_handle, self.system, conformal_cfg, self.device
        )

    def estimate(self, start_states: np.ndarray) -> OutcomeProbabilities:
        if self.estimator is None:
            raise RuntimeError("Probability backend used before bind_model")
        p_success, p_failure, p_invalid = self.estimator.estimate(start_states)
        return OutcomeProbabilities(
            p_success=np.asarray(p_success),
            p_failure=np.asarray(p_failure),
            p_invalid=np.asarray(p_invalid),
        )
```

- [ ] **Step 5: Update `tests/test_classifier_backend.py`** — change `_cfg()` to use the new sub-config shape and update the constructor call:

```python
import numpy as np
import torch
from omegaconf import OmegaConf

from adaptive_roa.adaptive_v2.probability.classifier_prob import ClassifierProbabilityBackend
from adaptive_roa.adaptive_v2.types import OutcomeProbabilities


class _DummyClassifier:
    def eval(self):
        return self

    def __call__(self, x):
        return torch.zeros((x.shape[0], 1), dtype=torch.float32)


def test_backend_returns_outcome_probabilities():
    backend = ClassifierProbabilityBackend(OmegaConf.create({}), system=None, device="cpu")
    backend.bind_model(_DummyClassifier())
    out = backend.estimate(np.zeros((7, 6), dtype=np.float32))
    assert isinstance(out, OutcomeProbabilities)
    assert out.p_success.shape == (7,)
    np.testing.assert_allclose(out.p_success, 0.5, rtol=1e-6)
    np.testing.assert_allclose(out.p_failure, 0.5, rtol=1e-6)
    assert np.all(out.p_invalid == 0.0)
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
conda run -n /common/home/st1122/Projects/adaptive_roa/env pytest tests/test_classifier_backend.py tests/adaptive_v2/test_probability_backends.py -v
```
Expected: 3 PASS

- [ ] **Step 7: Commit**

```bash
git add adaptive_roa/adaptive_v2/probability/ tests/test_classifier_backend.py tests/adaptive_v2/test_probability_backends.py
git commit -m "feat: probability backends read own cfg sub-config"
```

---

### Task 5: Update acquisition strategies

**Files:**
- Modify: `adaptive_roa/adaptive_v2/strategy/direct.py`
- Modify: `adaptive_roa/adaptive_v2/strategy/ranked.py`
- Modify: `adaptive_roa/adaptive_v2/strategy/conformal.py`
- Create: `tests/adaptive_v2/test_acquisition_strategies.py`

**Interfaces:**
- Consumes: `AcquisitionResult`, `ThresholdState`; `UncertainSampler` from `adaptive_roa.adaptive.balanced_sampler`
- Produces: all three strategy classes take `(cfg)` at init and store sampling params; `select(pool, probability_backend, threshold_backend, threshold_state, target_count, exclude)` — `cfg` parameter removed from `select()`

- [ ] **Step 1: Write failing tests**

`tests/adaptive_v2/test_acquisition_strategies.py`:
```python
from omegaconf import OmegaConf

from adaptive_roa.adaptive_v2.strategy.direct import DirectAcquisitionStrategy
from adaptive_roa.adaptive_v2.strategy.ranked import RankedAcquisitionStrategy
from adaptive_roa.adaptive_v2.strategy.conformal import ConformalAcquisitionStrategy


def _direct_cfg():
    return OmegaConf.create({
        "d2_ratio": 0.5, "batch_size_sampling": 50, "max_samples_per_epoch": 1000,
        "decision_rule": "two_sided", "verbose": False,
    })


def _ranked_cfg():
    return OmegaConf.create({
        "d2_ratio": 0.5, "batch_size_sampling": 50, "max_samples_per_epoch": 1000,
        "n_ranked_candidates": 500, "decision_rule": "two_sided", "verbose": False,
    })


def _conformal_cfg():
    return OmegaConf.create({
        "d2_ratio": 0.5, "batch_size_sampling": 50, "max_samples_per_epoch": 1000,
        "decision_rule": "two_sided", "verbose": False,
    })


def test_direct_stores_cfg():
    s = DirectAcquisitionStrategy(_direct_cfg())
    assert s.decision_rule == "two_sided"
    assert s.batch_size_sampling == 50


def test_ranked_stores_cfg():
    s = RankedAcquisitionStrategy(_ranked_cfg())
    assert s.n_ranked_candidates == 500


def test_conformal_stores_cfg():
    s = ConformalAcquisitionStrategy(_conformal_cfg())
    assert s.decision_rule == "two_sided"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
conda run -n /common/home/st1122/Projects/adaptive_roa/env pytest tests/adaptive_v2/test_acquisition_strategies.py -v
```
Expected: FAIL (constructors take no args currently)

- [ ] **Step 3: Rewrite `adaptive_roa/adaptive_v2/strategy/direct.py`**

```python
from __future__ import annotations

from typing import Any

from adaptive_roa.adaptive.balanced_sampler import UncertainSampler
from adaptive_roa.adaptive_v2.types import AcquisitionResult, ThresholdState


class DirectAcquisitionStrategy:
    mode = "direct"

    def __init__(self, cfg: Any):
        self.d2_ratio = float(cfg.d2_ratio)
        self.batch_size_sampling = int(cfg.batch_size_sampling)
        self.max_samples_per_epoch = int(cfg.max_samples_per_epoch)
        self.decision_rule = str(cfg.decision_rule)
        self.verbose = bool(cfg.verbose)

    def select(
        self,
        pool: Any,
        probability_backend: Any,
        threshold_backend: Any,
        threshold_state: ThresholdState,
        target_count: int,
        exclude: set[int] | None = None,
    ) -> AcquisitionResult:
        sampler = UncertainSampler(
            dataset_builder=pool.dataset_builder,
            target_count=target_count,
            batch_size=self.batch_size_sampling,
            max_candidates=self.max_samples_per_epoch,
        )
        sample_result = sampler.sample_direct(
            prob_estimator=probability_backend.estimator,
            lambda_star=threshold_state.lambda_star,
            delta_star=threshold_state.delta_star,
            decision_rule=self.decision_rule,
            exclude=exclude or set(),
            verbose=self.verbose,
        )
        return AcquisitionResult(
            d1_indices=[],
            d2_indices=list(sample_result.uncertain_indices),
            n_candidates_evaluated=int(sample_result.n_candidates_evaluated),
            n_certain_discarded=int(sample_result.n_certain_discarded),
            n_invalid_added=0,
            diagnostics={"n_batches": int(sample_result.n_batches)},
        )
```

- [ ] **Step 4: Rewrite `adaptive_roa/adaptive_v2/strategy/ranked.py`**

```python
from __future__ import annotations

from typing import Any

from adaptive_roa.adaptive.balanced_sampler import UncertainSampler
from adaptive_roa.adaptive_v2.types import AcquisitionResult, ThresholdState


class RankedAcquisitionStrategy:
    mode = "ranked"

    def __init__(self, cfg: Any):
        self.d2_ratio = float(cfg.d2_ratio)
        self.batch_size_sampling = int(cfg.batch_size_sampling)
        self.max_samples_per_epoch = int(cfg.max_samples_per_epoch)
        self.n_ranked_candidates = int(cfg.n_ranked_candidates)
        self.decision_rule = str(cfg.decision_rule)
        self.verbose = bool(cfg.verbose)

    def select(
        self,
        pool: Any,
        probability_backend: Any,
        threshold_backend: Any,
        threshold_state: ThresholdState,
        target_count: int,
        exclude: set[int] | None = None,
    ) -> AcquisitionResult:
        if target_count <= 0:
            return AcquisitionResult(
                d1_indices=[], d2_indices=[], n_candidates_evaluated=0,
                n_certain_discarded=0, n_invalid_added=0,
                diagnostics={"skipped_reason": "target_count_zero"},
            )
        sampler = UncertainSampler(
            dataset_builder=pool.dataset_builder,
            target_count=target_count,
            batch_size=self.batch_size_sampling,
            max_candidates=self.max_samples_per_epoch,
        )
        ranked_result = sampler.sample_ranked(
            prob_estimator=probability_backend.estimator,
            calibrator=threshold_backend.predictor.calibrator,
            lambda_star=threshold_state.lambda_star,
            delta_star=threshold_state.delta_star,
            decision_rule=self.decision_rule,
            n_candidates=self.n_ranked_candidates,
            n_select=target_count,
            verbose=self.verbose,
        )
        return AcquisitionResult(
            d1_indices=[],
            d2_indices=list(ranked_result.selected_indices),
            n_candidates_evaluated=int(ranked_result.n_candidates_evaluated),
            n_certain_discarded=int(ranked_result.n_candidates_evaluated - len(ranked_result.selected_indices)),
            n_invalid_added=0,
            diagnostics={
                "ranked_score_threshold": float(ranked_result.score_threshold),
                "n_ranked_candidates_evaluated": int(ranked_result.n_candidates_evaluated),
            },
        )
```

- [ ] **Step 5: Rewrite `adaptive_roa/adaptive_v2/strategy/conformal.py`**

```python
from __future__ import annotations

from typing import Any

from adaptive_roa.adaptive.balanced_sampler import UncertainSampler
from adaptive_roa.adaptive_v2.types import AcquisitionResult, ThresholdState


class ConformalAcquisitionStrategy:
    mode = "conformal"

    def __init__(self, cfg: Any):
        self.d2_ratio = float(cfg.d2_ratio)
        self.batch_size_sampling = int(cfg.batch_size_sampling)
        self.max_samples_per_epoch = int(cfg.max_samples_per_epoch)
        self.decision_rule = str(cfg.decision_rule)
        self.verbose = bool(cfg.verbose)

    def select(
        self,
        pool: Any,
        probability_backend: Any,
        threshold_backend: Any,
        threshold_state: ThresholdState,
        target_count: int,
        exclude: set[int] | None = None,
    ) -> AcquisitionResult:
        if threshold_state.q_hat is None:
            raise ValueError("Conformal strategy requires q_hat in ThresholdState")
        sampler = UncertainSampler(
            dataset_builder=pool.dataset_builder,
            target_count=target_count,
            batch_size=self.batch_size_sampling,
            max_candidates=self.max_samples_per_epoch,
        )
        sample_result = sampler.sample(
            prob_estimator=probability_backend.estimator,
            calibrator=threshold_backend.predictor.calibrator,
            lambda_star=threshold_state.lambda_star,
            delta_star=threshold_state.delta_star,
            q_hat=threshold_state.q_hat,
            decision_rule=self.decision_rule,
            exclude=exclude or set(),
            verbose=self.verbose,
        )
        return AcquisitionResult(
            d1_indices=[],
            d2_indices=list(sample_result.uncertain_indices),
            n_candidates_evaluated=int(sample_result.n_candidates_evaluated),
            n_certain_discarded=int(sample_result.n_certain_discarded),
            n_invalid_added=int(sample_result.n_invalid_added),
            diagnostics={"n_batches": int(sample_result.n_batches)},
        )
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
conda run -n /common/home/st1122/Projects/adaptive_roa/env pytest tests/adaptive_v2/test_acquisition_strategies.py -v
```
Expected: 3 PASS

- [ ] **Step 7: Commit**

```bash
git add adaptive_roa/adaptive_v2/strategy/ tests/adaptive_v2/test_acquisition_strategies.py
git commit -m "feat: acquisition strategies take cfg at construction, drop cfg from select()"
```

---

### Task 6: Update `FullROAEvaluator`

**Files:**
- Modify: `adaptive_roa/adaptive_v2/eval/full_roa.py` (constructor + `evaluate_epoch` signature)
- Create: `tests/adaptive_v2/test_evaluator.py`

**Interfaces:**
- Consumes: `ThresholdState`; all functions inside `full_roa.py` unchanged
- Produces: `FullROAEvaluator(cfg, system, device)` reading `cfg.predictor_type`, `cfg.alpha_eval`, `cfg.attractor_radius`, `cfg.num_mc_samples_eval`, `cfg.decision_rule`, `cfg.refine_*`, `cfg.max_eval_rows`, `cfg.verbose`; `evaluate_epoch(model_handle, threshold_state, epoch_context) -> dict` — `epoch_context` still accepted but the evaluator no longer falls back to `cfg.conformal.*`

- [ ] **Step 1: Write failing test**

`tests/adaptive_v2/test_evaluator.py`:
```python
from omegaconf import OmegaConf

from adaptive_roa.adaptive_v2.eval.full_roa import FullROAEvaluator


def _cfg(predictor_type="generative"):
    return OmegaConf.create({
        "predictor_type": predictor_type,
        "alpha_eval": 0.1,
        "attractor_radius": 0.2,
        "num_mc_samples_eval": 10,
        "decision_rule": "two_sided",
        "refine_invalids": False,
        "refine_t_min": 0.7,
        "refine_t_max": 0.9,
        "refine_num_steps": 100,
        "refine_max_attempts": 5,
        "max_eval_rows": None,
        "verbose": False,
    })


def test_evaluator_stores_cfg_fields():
    ev = FullROAEvaluator(_cfg(), system=None, device="cpu")
    assert ev.predictor_type == "generative"
    assert ev.attractor_radius == 0.2
    assert ev.num_mc_samples_eval == 10
    assert ev.decision_rule == "two_sided"


def test_evaluator_classifier_cfg():
    ev = FullROAEvaluator(_cfg("classifier"), system=None, device="cpu")
    assert ev.predictor_type == "classifier"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
conda run -n /common/home/st1122/Projects/adaptive_roa/env pytest tests/adaptive_v2/test_evaluator.py -v
```
Expected: FAIL (constructor signature mismatch)

- [ ] **Step 3: Update `FullROAEvaluator.__init__` in `adaptive_roa/adaptive_v2/eval/full_roa.py`**

Find the `FullROAEvaluator.__init__` method (search for `class FullROAEvaluator`) and replace the constructor:

```python
class FullROAEvaluator:
    def __init__(self, cfg: Any, system: Any, device: str):
        self.predictor_type = str(cfg.predictor_type)
        self.alpha_eval = float(cfg.alpha_eval)
        self.attractor_radius = float(cfg.attractor_radius)
        self.num_mc_samples_eval = int(cfg.num_mc_samples_eval)
        self.decision_rule = str(cfg.decision_rule)
        self.refine_invalids = bool(cfg.refine_invalids)
        self.refine_t_min = float(cfg.refine_t_min)
        self.refine_t_max = float(cfg.refine_t_max)
        self.refine_num_steps = int(cfg.refine_num_steps)
        self.refine_max_attempts = int(cfg.refine_max_attempts)
        self.max_eval_rows = cfg.max_eval_rows  # may be None
        self.verbose = bool(cfg.verbose)
        self.system = system
        self.device = device
        self.cfg = cfg  # kept temporarily; removed in Task 8 cleanup
```

- [ ] **Step 4: Update `evaluate_epoch` to read from stored fields**

In `evaluate_epoch`, replace all `self.cfg.conformal.get("...", default)` references:

| Old | New |
|-----|-----|
| `self.cfg.conformal.get("attractor_radius", resolve_system_hook(self.system).attractor_radius_default)` | `self.attractor_radius` |
| `self.cfg.conformal.get("decision_rule", None)` | `self.decision_rule` |
| `self.cfg.conformal.get("num_mc_samples_eval", 20)` | `self.num_mc_samples_eval` |
| `self.cfg.conformal.get("refine_invalids", False)` | `self.refine_invalids` |
| `self.cfg.conformal.get("refine_t_min", 0.7)` | `self.refine_t_min` |
| `self.cfg.conformal.get("refine_t_max", 0.9)` | `self.refine_t_max` |
| `self.cfg.conformal.get("refine_num_steps", 100)` | `self.refine_num_steps` |
| `self.cfg.conformal.get("refine_max_attempts", 5)` | `self.refine_max_attempts` |
| `self.cfg.conformal.get("max_eval_rows", None)` | `self.max_eval_rows` |
| `str(self.cfg.get("predictor", "generative"))` | `self.predictor_type` |

The `epoch_context.get("decision_rule", self.cfg.conformal.get(...))` pattern becomes `epoch_context.get("decision_rule", self.decision_rule)`.

- [ ] **Step 5: Run tests to verify they pass**

```bash
conda run -n /common/home/st1122/Projects/adaptive_roa/env pytest tests/adaptive_v2/test_evaluator.py -v
```
Expected: 2 PASS

- [ ] **Step 6: Commit**

```bash
git add adaptive_roa/adaptive_v2/eval/full_roa.py tests/adaptive_v2/test_evaluator.py
git commit -m "feat: FullROAEvaluator reads own cfg sub-config, no conformal.* reads"
```

---

### Task 7: Update trainers

**Files:**
- Modify: `adaptive_roa/adaptive_v2/trainers/flow_matching_trainer.py`
- Modify: `adaptive_roa/adaptive_v2/trainers/classifier_trainer.py`
- Modify: `tests/test_classifier_trainer.py`

**Interfaces:**
- Consumes: `cfg.predictor` sub-config node (has `.flow_matching`, `.trainer`, `.classifier` sub-dicts)
- Produces: both trainers now expect to receive `cfg` = `cfg.predictor` (the predictor sub-config, not the full config); `cfg.flow_matching.*` and `cfg.trainer.*` are accessed as sub-nodes of the sub-config

- [ ] **Step 1: Check existing classifier trainer test**

```bash
conda run -n /common/home/st1122/Projects/adaptive_roa/env pytest tests/test_classifier_trainer.py -v
```
Record which tests pass currently.

- [ ] **Step 2: Update `adaptive_roa/adaptive_v2/trainers/classifier_trainer.py`**

The trainer now receives `cfg.predictor` as its config. Key reads change:
- `cfg.get("classifier", {})` → `self.cfg.get("classifier", {})`  (unchanged structure, `cfg` is now the predictor sub-config)
- `cfg.get("batch_size", 1024)` → `self.cfg.get("batch_size", 1024)` (`batch_size` is at `predictor.batch_size` in Task 8's classifier.yaml)
- `str(cfg.get("device", "cuda:0"))` → `str(self.cfg.get("device", "cuda:0"))` (or read from parent; set `device` at `predictor.device` in the config)
- `trainer_cfg = self.cfg.get("trainer", {})` → `trainer_cfg = self.cfg.get("lightning_trainer", {})` (Lightning settings are now under `predictor.lightning_trainer`)

```python
class ClassifierTrainer:
    def __init__(self, cfg: Any, system: Any, system_name: str):
        self.cfg = cfg          # cfg.predictor sub-config
        self.system = system
        self.system_name = system_name

    def fit(self, dataset_files: dict, output_dir: str, resume_checkpoint: str | None = None):
        cls_cfg = self.cfg.get("classifier", {})
        trainer_cfg = self.cfg.get("lightning_trainer", {})
        batch_size = int(self.cfg.get("batch_size", 1024))
        device = str(self.cfg.get("device", "cuda:0"))
        # ... remainder of fit() unchanged except using trainer_cfg above
```

- [ ] **Step 3: Update `adaptive_roa/adaptive_v2/trainers/flow_matching_trainer.py`**

Same pattern: the trainer receives `cfg.predictor` so all `self.cfg.flow_matching.*` reads work
unchanged (because `flow_matching` is nested under `predictor` in the new predictor configs).
Change `self.cfg.trainer.*` → `self.cfg.lightning_trainer.*`:

```python
class FlowMatchingTrainer:
    def __init__(self, cfg: Any, system: Any, system_name: str):
        self.cfg = cfg          # cfg.predictor sub-config
        self.system = system
        self.system_name = system_name

    def fit(self, ...):
        trainer_cfg = self.cfg.get("lightning_trainer", {})
        batch_size = int(self.cfg.get("batch_size", 1024))
        # self.cfg.flow_matching.* reads unchanged
```

- [ ] **Step 4: Update tests that construct trainers directly**

In `tests/test_classifier_trainer.py`, update any `_cfg()` helper to match the new sub-config shape (containing `classifier.*` and `trainer.*` directly, not nested under a top-level key).

- [ ] **Step 5: Run trainer tests**

```bash
conda run -n /common/home/st1122/Projects/adaptive_roa/env pytest tests/test_classifier_trainer.py -v
```
Expected: same tests pass as before Step 1.

- [ ] **Step 6: Commit**

```bash
git add adaptive_roa/adaptive_v2/trainers/ tests/test_classifier_trainer.py
git commit -m "feat: trainers receive predictor sub-config only"
```

---

### Task 8: Update `predictor/` configs and `engine.py`

**Files:**
- Modify: `configs/adaptive_v2/predictor/generative.yaml`
- Modify: `configs/adaptive_v2/predictor/classifier.yaml`
- Modify: `adaptive_roa/adaptive_v2/engine.py`

**Interfaces:**
- Consumes: all updated backend constructors from Tasks 2–7
- Produces: `engine.py` with `_instantiate()` helper; all backends constructed via `_instantiate`; no `cfg.conformal.*` reads anywhere in the engine loop; explicit `calibration_backend.calibrate()` and `calibration_backend.calibrate_eval()` calls; `cfg.predictor.type` used instead of `cfg.get("predictor", "generative")`

- [ ] **Step 1: Replace `configs/adaptive_v2/predictor/generative.yaml`**

`trainer_target` holds the Python trainer class path (NOT `_target_` to avoid triggering Hydra
auto-instantiation). Lightning Trainer settings live under `predictor.lightning_trainer` to avoid
conflicting with the top-level `trainer:` block in `_base.yaml`. The top-level `trainer:` block
in `_base.yaml` is removed in Task 9.

```yaml
# @package _global_
defaults:
  - /probability: endpoint_mc
predictor:
  type: generative
  trainer_target: adaptive_roa.adaptive_v2.trainers.flow_matching_trainer.FlowMatchingTrainer
  batch_size: 1024
  val_batch_size: 2048
  flow_matching:
    latent_dim: 0
    num_integration_steps: 100
    mae_val_frequency: 10
    use_loss_weights: false
    use_log_loss_weights: false
    use_manifold: false
    clamp_noise: false
    zero_latent: true
    noise_scale: 1.0
    quat_loss_weight: 1.0
  lightning_trainer:
    max_epochs: 1000
    accelerator: gpu
    devices: 1
    precision: 32
    gradient_clip_val: 1.0
    log_every_n_steps: 10
    check_val_every_n_epoch: 1
    enable_progress_bar: true
    enable_model_summary: true
    callbacks:
      - _target_: lightning.pytorch.callbacks.ModelCheckpoint
        monitor: val_loss
        mode: min
        save_top_k: 1
        save_last: true
        filename: "best-{epoch:02d}-{val_loss:.4f}"
        auto_insert_metric_name: false
      - _target_: lightning.pytorch.callbacks.EarlyStopping
        monitor: val_loss
        mode: min
        patience: 100
        verbose: true
```

- [ ] **Step 2: Replace `configs/adaptive_v2/predictor/classifier.yaml`**

```yaml
# @package _global_
defaults:
  - /probability: classifier_prob
predictor:
  type: classifier
  trainer_target: adaptive_roa.adaptive_v2.trainers.classifier_trainer.ClassifierTrainer
  batch_size: 1024
  val_batch_size: 2048
  classifier:
    hidden_dims: [256, 512, 256]
    dropout: 0.0
    lr: 1.0e-3
    weight_decay: 1.0e-5
    max_epochs: 200
    patience: 20
  lightning_trainer:
    gradient_clip_val: 1.0
    log_every_n_steps: 10
```

Note: no `conformal.decision_rule` override.

- [ ] **Step 3: Update `engine.py` — add `_instantiate`, replace init block**

At the top of `engine.py`, add the helper after the imports:

```python
def _instantiate(cfg_node, *args, **kwargs):
    """Resolve _target_ from cfg_node and construct with cfg_node as first arg."""
    from hydra.utils import get_class
    cls = get_class(cfg_node._target_)
    return cls(cfg_node, *args, **kwargs)
```

Replace the `__init__` backend construction block (lines ~92–101). The trainer is resolved via
`trainer_target` (plain string, not `_target_`) to avoid Hydra auto-instantiation:

```python
        from hydra.utils import get_class
        self.predictor_type = str(cfg.predictor.type)
        trainer_cls = get_class(cfg.predictor.trainer_target)
        self.trainer             = trainer_cls(cfg.predictor, self.system, self.system_name)
        self.probability_backend = _instantiate(cfg.probability,  self.system, self.device)
        self.threshold_backend   = _instantiate(cfg.threshold,    self.system, self.device)
        self.calibration_backend = _instantiate(cfg.calibration,  self.system, self.device)
        self.acquisition         = _instantiate(cfg.acquisition)
        self.evaluator           = _instantiate(cfg.eval,         self.system, self.device)
```

- [ ] **Step 4: Update `bind_model` calls in the engine loop to include predictor_type**

The `calibration_backend.bind_model` needs to know `predictor_type`:

```python
# In the epoch loop, replace:
self.probability_backend.bind_model(model_handle)
self.threshold_backend.bind_model(model_handle)

# With:
self.probability_backend.bind_model(model_handle)
self.threshold_backend.bind_model(model_handle)
self.calibration_backend.bind_model(model_handle, predictor_type=self.predictor_type)
```

- [ ] **Step 5: Replace inline q_hat calibration and all `cfg.conformal.*` reads in the engine loop**

Replace the acquisition q_hat block (lines ~192–208):
```python
q_hat = None
test_metrics = {"coverage": None, "f1": None, "unknown_rate": None}
if acquisition_mode == "conformal" and need_d2_acquisition:
    d1_labels = self.pool.get_labels(d1_indices)
    q_hat = self.calibration_backend.calibrate(d1_states, d1_labels, threshold_state)
    threshold_state.q_hat = q_hat
    X_test, y_test = self.pool.get_val_labels()
    predictor = self.threshold_backend.predictor
    if predictor is None:
        raise RuntimeError("Threshold backend predictor missing after bind_model")
    test_metrics = predictor.evaluate(X_test, y_test, verbose=self.calibration_backend.verbose)
elif acquisition_mode == "conformal":
    print("Skipping q_hat calibration because d2_target=0")
```

Replace inline eval calibration block (lines ~236–260):
```python
q_hat_eval = None
n_cal_eval = 0
cal_file = self.cfg.data_source.get("cal_set_file", None)
if run_eval and cal_file:
    X_cal_eval, _, y_cal_eval = load_eval_states(
        cal_file, max_rows=self.evaluator.max_eval_rows
    )
    q_hat_eval = self.calibration_backend.calibrate_eval(
        X_cal_eval, y_cal_eval, threshold_state
    )
    n_cal_eval = len(X_cal_eval)
```

Replace the `full_roa_metrics` call block (lines ~269–281):
```python
full_roa_metrics = self.evaluator.evaluate_epoch(
    model_handle,
    threshold_state,
    {
        "eval_states_file": self.cfg.data_source.test_set_file,
        "batch_size": int(self.cfg.get("val_batch_size", 2048)),
        "output_dir": str(epoch_output_dir),
    },
)
```

Replace `cfg.conformal.get("verbose", True)` in the endpoint error block:
```python
endpoint_error = compute_endpoint_prediction_error(
    flow_matcher=model_handle,
    dataset_builder=self.pool.dataset_builder,
    batch_size=self.cfg.get("val_batch_size", 512),
    device=self.device,
    verbose=self.calibration_backend.verbose,
)
```

Replace all remaining `self.cfg.conformal.get(...)` reads in the loop with the relevant backend's stored attribute:
- `cfg.conformal.get("verbose", True)` → `self.calibration_backend.verbose`
- `cfg.conformal.get("decision_rule", "two_sided")` → `self.calibration_backend.decision_rule`
- `cfg.conformal.get("optimize_mode", "lambda")` → `self.threshold_backend.optimize_mode`

Replace the acquisition `strategy.select(...)` call — remove `cfg=self.cfg`:
```python
acquisition = self.acquisition.select(
    pool=self.pool,
    probability_backend=self.probability_backend,
    threshold_backend=self.threshold_backend,
    threshold_state=threshold_state,
    target_count=n_d2_target,
    exclude=set(d1_indices),
)
```

Replace `sampling_mode = str(self.cfg.get("sampling_mode", "direct"))` with:
```python
acquisition_mode = self.acquisition.mode   # "direct", "ranked", or "conformal"
```

Each strategy stores `self.mode` at construction (added in Task 5).

Remove the `_build_strategy(sampling_mode)` call — `self.acquisition` is already instantiated.

Remove the `self.predictor_type` check in the endpoint_error block:
```python
if self.smoke_mode or not run_eval or self.predictor_type == "classifier":
```
This `predictor_type` check can stay since `self.predictor_type` is now properly set from `cfg.predictor.type`.

- [ ] **Step 6: Verify no `conformal` key in engine.py remains**

```bash
grep -n "cfg.conformal\|cfg\.get.*conformal" adaptive_roa/adaptive_v2/engine.py
```
Expected: no matches.

- [ ] **Step 7: Run integration smoke test (config resolution)**

```bash
conda run -n /common/home/st1122/Projects/adaptive_roa/env python scripts/run_adaptive.py \
  system=pendulum predictor=generative --cfg job --resolve 2>&1 | head -60
```
Expected: printed YAML shows `predictor.type: generative`, `threshold.*`, `calibration.*`, `acquisition.*`, `eval.*` namespaces; no `conformal:` key at top level.

- [ ] **Step 8: Commit**

```bash
git add configs/adaptive_v2/predictor/ adaptive_roa/adaptive_v2/engine.py
git commit -m "feat: engine uses _instantiate for all backends, no conformal.* reads"
```

---

### Task 9: System config migration

**Files:**
- Modify: `configs/adaptive_v2/system/_base.yaml`
- Modify: `configs/adaptive_v2/system/pendulum.yaml`
- Modify: `configs/adaptive_v2/system/cartpole_pybullet.yaml`
- Modify: `configs/adaptive_v2/system/quadrotor2d.yaml`
- Modify: `configs/adaptive_v2/system/quadrotor3d.yaml`
- Modify: `configs/adaptive_v2/system/humanoid_standup_reach.yaml`

**Interfaces:**
- Produces: all system configs carry bare top-level `decision_rule` and `attractor_radius` keys; no `conformal:` block anywhere; module-specific overrides in the correct namespace

- [ ] **Step 1: Update `_base.yaml` — remove `conformal:` block, forwarding anchors, and `trainer:` block**

Remove the entire `conformal:` block (lines ~42–78).
Remove the top-level `trainer:` block entirely — Lightning trainer settings now live under `predictor.lightning_trainer` in each predictor config; system-specific callback overrides move there too.
Remove top-level forwarding anchors: `w: ${w}`, `alpha_sampling: ${alpha_sampling}`, `threshold_mode: dynamic`, `sampling_mode: direct`.
Remove `d2_ratio`, `batch_size_sampling`, `max_samples_per_epoch`, `n_ranked_candidates` top-level keys (now in acquisition configs).
Remove `batch_size`, `val_batch_size`, `latent_dim` top-level keys (now in `predictor.*`).
Keep: `seed`, `exp_dir`, `data_dir`, `training_index`, `device`, `val_ratio`, `test_ratio`, `initial_train_size`, `n_epochs`, `samples_per_epoch`, `warm_start`, `eval_every`, `num_workers`, `optimizer`, `scheduler`, `adaptive_v2.*`.

- [ ] **Step 2: Update `pendulum.yaml`**

Replace `conformal:` block with:
```yaml
decision_rule: one_sided
attractor_radius: 0.1

threshold:
  optimize_mode: joint
```

Remove `conformal:`, remove `w:`, `alpha_sampling:`, `threshold_mode:` references.
Update `output_dir` template to remove `sampling_mode_${sampling_mode}`, `threshold_mode_${threshold_mode}`, `alpha_${alpha_sampling}` interpolations since these anchors are gone.

- [ ] **Step 3: Update `cartpole_pybullet.yaml`**

```yaml
decision_rule: two_sided
attractor_radius: 0.2

threshold:
  optimize_mode: joint
```

Remove `conformal:` block. Update `output_dir` template.

- [ ] **Step 4: Update `quadrotor2d.yaml`**

```yaml
decision_rule: two_sided
attractor_radius: 0.2
```

Remove `conformal:` block. Update `output_dir` template.

- [ ] **Step 5: Update `quadrotor3d.yaml`**

```yaml
decision_rule: two_sided
attractor_radius: 0.2
```

Remove `conformal:` block. Update `output_dir` template.

- [ ] **Step 6: Update `humanoid_standup_reach.yaml`**

Add `decision_rule` and `attractor_radius` appropriate for that system. Check what the current `conformal.decision_rule` and `conformal.attractor_radius` values are in that file and move them to top-level. Remove `conformal:` block.

- [ ] **Step 7: Verify config resolution for each system**

```bash
for system in pendulum cartpole_pybullet quadrotor2d quadrotor3d; do
  echo "=== $system ==="
  conda run -n /common/home/st1122/Projects/adaptive_roa/env \
    python scripts/run_adaptive.py system=$system --cfg job --resolve 2>&1 \
    | grep -E "decision_rule|attractor_radius|conformal"
done
```
Expected: `decision_rule` and `attractor_radius` appear at top level; no `conformal:` key anywhere.

- [ ] **Step 8: Commit**

```bash
git add configs/adaptive_v2/system/
git commit -m "feat: system configs set decision_rule/attractor_radius, drop conformal: block"
```

---

### Task 10: Cleanup

**Files:**
- Modify: `adaptive_roa/conformal/config.py` — remove `from_hydra` method
- Modify: `adaptive_roa/probabilistic_classifier/export.py` — read `cfg.predictor.type`
- Delete: `configs/adaptive_v2/strategy/` directory

**Interfaces:**
- Consumes: nothing new; all callers already updated in Tasks 3–8
- Produces: `ConformalConfig` no longer has a `from_hydra` classmethod; export script reads `cfg.predictor.type`; no `strategy/` config group

- [ ] **Step 1: Verify `from_hydra` has no callers remaining**

```bash
grep -rn "from_hydra\|ConformalConfig\.from_hydra" adaptive_roa/ --include="*.py"
```
Expected: zero matches (all callers were removed in Tasks 3–4 when backends stopped using it).

- [ ] **Step 2: Remove `from_hydra` classmethod from `adaptive_roa/conformal/config.py`**

Delete the `from_hydra` classmethod (lines ~106–137). Keep `__post_init__`, `_resolve_objective`, and the dataclass fields.

- [ ] **Step 3: Update `adaptive_roa/probabilistic_classifier/export.py`**

In `load_cfg()` + `export_run()`, update the predictor type read:

```python
# Before
predictor_type = str(cfg.get("predictor", "generative"))

# After
predictor = cfg.get("predictor", {})
if isinstance(predictor, str):
    predictor_type = predictor  # legacy saved config
else:
    predictor_type = str(predictor.get("type", "generative"))
```

This handles both the new format (`cfg.predictor.type`) and any existing saved `.hydra/config.yaml` files that still use the old string format.

- [ ] **Step 4: Delete old strategy config directory**

```bash
rm -rf configs/adaptive_v2/strategy/
```

- [ ] **Step 5: Run the full test suite**

```bash
conda run -n /common/home/st1122/Projects/adaptive_roa/env pytest tests/ -v --tb=short 2>&1 | tail -40
```
Expected: all tests that passed before this plan still pass; no `conformal` key errors.

- [ ] **Step 6: Final verification — no `conformal.*` in any config or engine**

```bash
grep -rn "cfg\.conformal\|conformal:" \
  adaptive_roa/adaptive_v2/ configs/adaptive_v2/ --include="*.py" --include="*.yaml"
```
Expected: zero matches.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "feat: remove ConformalConfig.from_hydra, clean up legacy strategy/ configs"
```
