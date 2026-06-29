# Adaptive v2 — Module Decoupling Design

**Date:** 2026-06-29
**Status:** Approved (design)
**Branch:** (to be created)

## Goal

Fully decouple the five adaptive-loop modules — Probabilistic Classifier, Threshold Optimizer,
Adaptive Acquisition, Conformal Calibration, and Evaluation — in both configuration and code.
Each module gets its own Hydra config namespace and a constructor that receives only its own
sub-config. The monolithic `conformal.*` block is deleted. Module implementations are
selected via Hydra `_target_` + a thin `_instantiate` helper; `engine.py` contains no
`if predictor ==` branches.

## Current State

### Config coupling

`configs/adaptive_v2/_base.yaml` has a single `conformal:` block (~25 keys) consumed by every
module:

- **Threshold opt:** `optimize_mode`, `optimize_objective`, `lambda_grid_size`,
  `delta_grid_size`, `delta_min/max`, `fixed_lambda_star/delta_star`, `use_p_invalid_veto`,
  `target_f1`, `decision_rule`, `verbose`
- **Calibration (q_hat):** `delta`, `w`, `alpha_sampling`, `decision_rule`, `verbose`
- **Probability backend:** `attractor_radius`, `num_mc_samples`, `refine_invalids`,
  `refine_t_*`, `trajectory_checking`
- **Evaluation:** `alpha_eval`, `num_mc_samples_eval`, `attractor_radius`, `decision_rule`,
  `refine_*`, `max_eval_rows`
- **Acquisition:** `decision_rule`, `alpha_sampling`, `verbose`

`predictor/classifier.yaml` also silently overrides `conformal.decision_rule: one_sided`,
coupling predictor selection to threshold/calibration behavior.

### Code coupling

`engine.py` hardcodes backend selection:
```python
if self.predictor_type == "classifier":
    self.trainer = ClassifierTrainer(cfg, system, system_name)
    self.probability_backend = ClassifierProbabilityBackend(system, cfg, device)
else:
    self.trainer = FlowMatchingTrainer(cfg, system, system_name)
    self.probability_backend = EndpointMCProbabilityBackend(system, cfg, device)
```

All backends accept the full `cfg` and call `cfg.conformal.get("...", default)` internally,
making them impossible to test or reuse without a full config tree.

Calibration (q_hat) is embedded inside `ConformalThresholdBackend.calibrate_qhat()` — there
is no standalone calibration class.

## Decisions

- **Separation:** Five independent Hydra config groups (`predictor`, `probability`, `threshold`,
  `calibration`, `acquisition`, `eval`). No shared `conformal.*` block.
- **System anchors:** `attractor_radius` and `decision_rule` are intrinsic system properties.
  They are set as bare top-level keys in each system config and referenced via Hydra
  interpolation (`${attractor_radius}`, `${decision_rule}`) in module configs. No fallback
  defaults — missing values raise a clear Hydra interpolation error.
- **Wiring:** Engine uses a thin `_instantiate(cfg_node, *args, **kwargs)` helper that resolves
  `cfg_node._target_` and calls `cls(cfg_node, *args, **kwargs)`. Each constructor receives
  its own sub-config only.
- **Probability ↔ predictor coupling:** `predictor/generative.yaml` and
  `predictor/classifier.yaml` include a Hydra `defaults` list that auto-selects the matching
  probability config (`endpoint_mc` / `classifier_prob`). Still overridable via CLI.
- **Calibration class:** A new `ConformalCalibrationBackend` is extracted from
  `ConformalThresholdBackend`. Threshold opt and q_hat calibration become two explicit engine
  steps. `ConformalConfig` dataclass is retained for internal conformal math, constructed
  inside each backend from its own sub-config fields.
- **Clean break:** All system configs drop `conformal:` and gain the new module-namespaced
  overrides. No backward-compat shims.

## Architecture

### Config structure

```
configs/adaptive_v2/
  default.yaml                         # defaults list only (no inline params)
  system/
    _base.yaml                         # shared training params (no conformal:)
    pendulum.yaml                      # + attractor_radius, decision_rule, module overrides
    cartpole_pybullet.yaml
    quadrotor2d.yaml
    quadrotor3d.yaml
    humanoid_standup_reach.yaml
  predictor/
    generative.yaml                    # predictor: generative + defaults: probability: endpoint_mc
    classifier.yaml                    # predictor: classifier + defaults: probability: classifier_prob
                                       # NO conformal.decision_rule side-effect
  probability/
    endpoint_mc.yaml                   # _target_, attractor_radius, MC params, refinement
    classifier_prob.yaml               # _target_ only (arch lives in predictor/)
  threshold/
    conformal_threshold.yaml           # _target_, optimize_mode/objective, grid sizes, verbose
  calibration/
    conformal_calibration.yaml         # _target_, delta, w, alpha, verbose
  acquisition/                         # renamed from strategy/ config group
    direct.yaml                        # _target_, d2_ratio, batch sizes
    ranked.yaml                        # _target_, n_ranked_candidates, d2_ratio, alpha
    conformal.yaml                     # _target_, alpha, verbose
  eval/
    full_roa.yaml                      # _target_, alpha_eval, MC eval params, max_eval_rows
```

### default.yaml

```yaml
defaults:
  - system: pendulum
  - predictor: generative        # auto-loads probability: endpoint_mc
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

### Per-module config contents

**`probability/endpoint_mc.yaml`**
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

**`probability/classifier_prob.yaml`**
```yaml
# @package _global_
probability:
  _target_: adaptive_roa.adaptive_v2.probability.classifier_prob.ClassifierProbabilityBackend
```

**`threshold/conformal_threshold.yaml`**
```yaml
# @package _global_
threshold:
  _target_: adaptive_roa.adaptive_v2.threshold.conformal_threshold.ConformalThresholdBackend
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

**`calibration/conformal_calibration.yaml`**
```yaml
# @package _global_
calibration:
  _target_: adaptive_roa.adaptive_v2.calibration.conformal_calibration.ConformalCalibrationBackend
  delta: 0.05
  w: 0.9
  alpha: 0.1
  decision_rule: ${decision_rule}
  verbose: true
```

**`acquisition/direct.yaml`**
```yaml
# @package _global_
acquisition:
  _target_: adaptive_roa.adaptive_v2.strategy.direct.DirectAcquisitionStrategy
  d2_ratio: 0.5
  batch_size_sampling: 50
  max_samples_per_epoch: 50000
```

**`acquisition/ranked.yaml`**
```yaml
# @package _global_
acquisition:
  _target_: adaptive_roa.adaptive_v2.strategy.ranked.RankedAcquisitionStrategy
  d2_ratio: 0.5
  n_ranked_candidates: 50000
  batch_size_sampling: 50
  max_samples_per_epoch: 50000
  alpha: 0.1
  decision_rule: ${decision_rule}
  verbose: true
```

**`acquisition/conformal.yaml`**
```yaml
# @package _global_
acquisition:
  _target_: adaptive_roa.adaptive_v2.strategy.conformal.ConformalAcquisitionStrategy
  d2_ratio: 0.5
  batch_size_sampling: 50
  max_samples_per_epoch: 50000
  alpha: 0.1
  decision_rule: ${decision_rule}
  verbose: true
```

**`eval/full_roa.yaml`**
```yaml
# @package _global_
eval:
  _target_: adaptive_roa.adaptive_v2.eval.full_roa.FullROAEvaluator
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

### System config migration

Each system config gains two required bare top-level keys and drops `conformal:`. System-specific
module overrides move to the appropriate namespace.

```yaml
# system/pendulum.yaml (after)
decision_rule: one_sided
attractor_radius: 0.1

threshold:
  optimize_mode: joint

# ... rest unchanged ...
```

```yaml
# system/cartpole_pybullet.yaml (after)
decision_rule: two_sided
attractor_radius: 0.2
```

`_base.yaml` loses the entire `conformal:` block. The `w`, `alpha_sampling`, `threshold_mode`
top-level interpolation anchors that were forwarded into `conformal:` are removed; each module
config declares its value directly.

The top-level `sampling_mode: direct | conformal | ranked` key is also removed from `_base.yaml`.
The acquisition implementation is now selected entirely by which `acquisition/` config group is
loaded (e.g. `acquisition=ranked`), making `sampling_mode` redundant.

### Engine changes

```python
# adaptive_roa/adaptive_v2/engine.py

def _instantiate(cfg_node, *args, **kwargs):
    """Resolve _target_ from cfg_node and construct with cfg_node + extra args."""
    from hydra.utils import get_class
    cls = get_class(cfg_node._target_)
    return cls(cfg_node, *args, **kwargs)

class AdaptiveEngine:
    def __init__(self, cfg):
        ...
        self.trainer             = _instantiate(cfg.predictor.trainer, self.system, self.system_name)
        self.probability_backend = _instantiate(cfg.probability,       self.system, self.device)
        self.threshold_backend   = _instantiate(cfg.threshold,         self.system, self.device)
        self.calibration_backend = _instantiate(cfg.calibration,       self.system, self.device)
        self.acquisition         = _instantiate(cfg.acquisition)
        self.evaluator           = _instantiate(cfg.eval,              self.system, self.device)
```

The `if self.predictor_type == "classifier": ... else: ...` block is deleted.

The engine loop explicitly sequences threshold opt and calibration as separate steps, making the
two-step process visible rather than hidden inside a backend method:

```python
# In the per-epoch training loop:
threshold_state = self.threshold_backend.optimize(X_train, y_train)
q_hat = self.calibration_backend.calibrate(X_cal, y_cal, threshold_state)
```

### New `ConformalCalibrationBackend`

A new thin class in `adaptive_roa/adaptive_v2/calibration/conformal_calibration.py`:

```python
class ConformalCalibrationBackend:
    def __init__(self, cfg, system, device):
        self.delta = cfg.delta
        self.w = cfg.w
        self.alpha = cfg.alpha
        self.decision_rule = cfg.decision_rule
        self.verbose = cfg.verbose
        ...

    def bind_model(self, model_handle): ...

    def calibrate(self, X_cal, y_cal, threshold_state) -> float:
        """Returns q_hat."""
        ...
```

`ConformalThresholdBackend.calibrate_qhat()` is removed. `ConformalConfig` is constructed
internally inside each backend that needs it, from its own sub-config fields only. No code
outside `adaptive_roa/conformal/` constructs `ConformalConfig` from `cfg.conformal.*`.

### Per-module constructor signatures

| Module | Old signature | New signature |
|--------|--------------|---------------|
| `EndpointMCProbabilityBackend` | `(system, cfg, device)` reads `cfg.conformal.*` | `(cfg, system, device)` reads `cfg.probability.*` |
| `ClassifierProbabilityBackend` | `(system, cfg, device)` reads `cfg.conformal.*` | `(cfg, system, device)` reads `cfg.probability.*` |
| `ConformalThresholdBackend` | `(system, cfg, device)` reads `cfg.conformal.*` | `(cfg, system, device)` reads `cfg.threshold.*` |
| `ConformalCalibrationBackend` | *(new class)* | `(cfg, system, device)` reads `cfg.calibration.*` |
| `DirectAcquisitionStrategy` | `()` reads cfg at call time | `(cfg)` stores alpha/decision_rule at construction |
| `RankedAcquisitionStrategy` | `()` reads cfg at call time | `(cfg)` |
| `ConformalAcquisitionStrategy` | `()` reads cfg at call time | `(cfg)` |
| `FullROAEvaluator` | `(system, cfg, device)` reads `cfg.conformal.*` | `(cfg, system, device)` reads `cfg.eval.*` |
| `ClassifierTrainer` | `(cfg, system, system_name)` reads `cfg.classifier.*` | `(cfg, system, system_name)` reads `cfg.predictor.*` (sub-config) |
| `FlowMatchingTrainer` | `(cfg, system, system_name)` reads `cfg.flow_matching.*` | `(cfg, system, system_name)` reads `cfg.predictor.*` (sub-config) |

### Predictor config structure

`predictor` becomes a dict namespace (`cfg.predictor.type`, `cfg.predictor.trainer`, etc.).
The bare string form `predictor: generative` is replaced everywhere with `cfg.predictor.type`.
This affects `engine.py`, `export.py`, and `probabilistic_classifier/export.py` — all read
`cfg.predictor.type` (or `cfg.get("predictor", {}).get("type", "generative")` for backward-compat
reads of saved configs).

`predictor/generative.yaml` nests trainer config and includes the probability group default:
```yaml
# @package _global_
defaults:
  - /probability: endpoint_mc
predictor:
  type: generative
  trainer:
    _target_: adaptive_roa.adaptive_v2.trainers.flow_matching_trainer.FlowMatchingTrainer
  flow_matching:
    latent_dim: 0
    num_integration_steps: 100
    mae_val_frequency: 10
    use_manifold: false
    use_loss_weights: false
    use_log_loss_weights: false
    clamp_noise: false
    zero_latent: true
    noise_scale: 1.0
    quat_loss_weight: 1.0
```

`predictor/classifier.yaml`:
```yaml
# @package _global_
defaults:
  - /probability: classifier_prob
predictor:
  type: classifier
  trainer:
    _target_: adaptive_roa.adaptive_v2.trainers.classifier_trainer.ClassifierTrainer
  classifier:
    hidden_dims: [256, 512, 256]
    dropout: 0.0
    lr: 1.0e-3
    weight_decay: 1.0e-5
    max_epochs: 200
    patience: 20
```

No `conformal.decision_rule` override in `predictor/classifier.yaml`.

The trainer constructors receive `cfg.predictor` as their sub-config node. They read
`cfg_node.flow_matching.*` (FM) or `cfg_node.classifier.*` (classifier) from within it.

## Testing

- Existing unit tests for each backend updated to pass a mock sub-config (`OmegaConf.create({...})`) instead of a full config tree.
- New unit test for `ConformalCalibrationBackend`.
- Integration: `python scripts/run_adaptive.py system=pendulum predictor=classifier --cfg job --resolve` must show no `conformal:` key in the resolved config.

## Migration Summary

Files changed:
- `configs/adaptive_v2/default.yaml` — add `calibration` group, rename `strategy` to `acquisition`, remove inline `sampling_mode`
- `configs/adaptive_v2/_base.yaml` — delete `conformal:` block; remove `sampling_mode`, `w`, `alpha_sampling`, `threshold_mode` forwarding anchors
- `configs/adaptive_v2/system/*.yaml` — add `decision_rule`, `attractor_radius`; drop `conformal:`; move overrides to module namespaces
- `configs/adaptive_v2/predictor/generative.yaml` — change to dict namespace, add probability default, nest flow_matching/trainer under `predictor:`
- `configs/adaptive_v2/predictor/classifier.yaml` — change to dict namespace, add probability default, remove `conformal.decision_rule`, nest classifier/trainer under `predictor:`
- `configs/adaptive_v2/probability/endpoint_mc.yaml` — expand with full params
- `configs/adaptive_v2/probability/classifier_prob.yaml` — new file
- `configs/adaptive_v2/threshold/conformal_threshold.yaml` — expand with full params
- `configs/adaptive_v2/calibration/conformal_calibration.yaml` — new file
- `configs/adaptive_v2/acquisition/direct.yaml` (+ ranked, conformal) — new directory (renamed from `strategy/`), expand with full params; old `configs/adaptive_v2/strategy/` removed
- `configs/adaptive_v2/eval/full_roa.yaml` — expand with full params
- `adaptive_roa/adaptive_v2/engine.py` — add `_instantiate`, remove if/else branches, explicit calibration step
- `adaptive_roa/adaptive_v2/probability/endpoint_mc.py` — new constructor signature
- `adaptive_roa/adaptive_v2/probability/classifier_prob.py` — new constructor signature
- `adaptive_roa/adaptive_v2/threshold/conformal_threshold.py` — new constructor, remove `calibrate_qhat`
- `adaptive_roa/adaptive_v2/calibration/conformal_calibration.py` — new file
- `adaptive_roa/adaptive_v2/strategy/direct.py` (+ ranked, conformal) — new constructors
- `adaptive_roa/adaptive_v2/eval/full_roa.py` — new constructor signature
- `adaptive_roa/adaptive_v2/trainers/flow_matching_trainer.py` — read from `cfg.flow_matching` (already a sub-node)
- `adaptive_roa/adaptive_v2/trainers/classifier_trainer.py` — read from `cfg.classifier`
- `adaptive_roa/conformal/config.py` — `from_hydra` method removed; `ConformalConfig` is now only constructed directly from explicit kwargs inside backends
- `adaptive_roa/probabilistic_classifier/export.py` — read `cfg.predictor.type` instead of `cfg.get("predictor", "generative")`
- `adaptive_roa/adaptive_v2/strategy/` (Python package) — constructors updated; config group directory renamed to `acquisition/` but Python package stays as `strategy/`
