# Classifier Predictor Pathway (+ quad2d run) — Implementation Plan

> **For agentic workers:** implement task-by-task. Steps use checkbox (`- [ ]`) syntax. Commit after each task. Run pragmatic TDD: real unit tests for pure logic (estimator, factory, dataset build), smoke tests for integration.

**Goal:** Add a discriminative binary CLASSIFIER as a config-swappable predictor alongside the generative flow-matching predictor in the `adaptive_v2` engine, then run it end-to-end on **quadrotor2d** with `d2_ratio=1.0` (full uncertainty sampling) and produce ROA results.

**Architecture:** Introduce ONE predictor-agnostic probability seam. A `ClassifierProbabilityEstimator(model, system, config, device)` exposes the exact `.estimate(states) -> (p_success, p_failure, p_invalid)` contract the acquisition/threshold code already calls. A small factory picks MC vs classifier by a new `predictor` config axis. The acquisition backend, the threshold backend's `ConformalPredictor`, and the evaluator all obtain probabilities through that seam. The classifier is trained on `(state, binary_label)` data built from the same trajectory pool.

**Tech Stack:** PyTorch Lightning, Hydra, numpy. conda env `/common/users/dm1487/envs/arcmg`.

## Global Constraints

- **NEVER write to or delete anything under `DATA_DIR=/common/users/shared/pracsys/genMoPlan/data_trajectories`** (shared, owned by `st1122`, read-only for us). All dataset builds write under `EXP_DIR=/common/users/shared/pracsys/adaptive_roa_experiments/dhruv/` (the experiment `output_dir/datasets/`). Add a runtime assert in the classification build that the target path is NOT under DATA_DIR.
- Binary labels only: `p_success=sigmoid`, `p_failure=1−p_success`, `p_invalid≡0`. No "unresolved" class.
- quad2d is ~8% success (imbalanced) → use `pos_weight = n_neg/n_pos` in BCE (computed per training file). No data dropped.
- Run the classifier with `decision_rule=one_sided` (binary classifier can't exercise two_sided).
- Device: `cuda:0` (A100). Python: `/common/users/dm1487/envs/arcmg/bin/python`.
- The generative pathway must remain byte-for-byte unchanged when `predictor=generative` (the default).
- Branch: `direct-classification`. Commit after each task.

## Key existing interfaces (verified)

- `ProbabilityEstimator(flow_matcher, system, config, device)`; `.estimate(states, verbose=True, refine_invalids=None, refine_t_range=None, refine_num_steps=None, refine_max_attempts=None) -> (p_success, p_failure, p_invalid)` each `[N]` np.float64. Construction sites: `adaptive_v2/probability/endpoint_mc.py:27`, `conformal/predictor.py:64`.
- Acquisition calls `p_success, p_failure, _ = prob_estimator.estimate(states_np)` (`adaptive/balanced_sampler.py` `_classify_batch_direct`, `sample_ranked`).
- `EndpointMCProbabilityBackend(system, cfg, device)`: `bind_model(model_handle)` builds `self.estimator = ProbabilityEstimator(...)`; `estimate(start_states)->OutcomeProbabilities`.
- `ConformalThresholdBackend(system, cfg, device)`: `bind_model` builds `ConformalPredictor(flow_matcher=model_handle, system, config, device)`; `optimize(X,y)->ThresholdState`.
- `ConformalPredictor(flow_matcher, system, config, device)` builds `self.prob_estimator` at `:64`, used in `optimize_thresholds`/`calibrate_qhat`/`predict`/`select_uncertain`/`evaluate`.
- `FullROAEvaluator.evaluate_epoch(model_handle, threshold_state, epoch_context)` (`eval/full_roa.py:944`) → `evaluate_full_roa_fast(flow_matcher=model_handle, system, ...)` (`:485`) which calls `flow_matcher.predict_endpoint` + `system.classify_attractor` + MC. Shared, model-agnostic helpers: `_predict_lambda_delta:185`, `_predict_lambda_only:248`, `_predict_fixed_threshold:224`, `_classification_metrics_from_predictions:23`, `_conservative_metrics:63`.
- `TrajectoryPool.initialize(n)->{train,val,train_trajectories,val_trajectories}`; `build_all_datasets()->same`; `get_labels(indices)->internal {-1,1}`; `add_to_training_balanced(indices)`; `sample_candidates_without_marking(n)->(states[n,D], indices)`; `.train_size`, `.dataset_builder`.
- `AdaptiveDatasetBuilder`/`data_source.save_endpoint_dataset(indices, filepath, mode)` writes endpoint pairs `[start... end...]` space-sep, NO label.
- System: `embed_state(state)->embedded` (SO2→sin/cos), `normalize_state`, `embed_state_for_model`, `state_dim`. quad2d: state_dim=6, embedded_dim=7.
- `ConformalConfig.from_hydra(cfg)`; engine `__init__` hardcodes trainer/probability/threshold/evaluator at `engine.py:89-92`; `_build_strategy(mode)` at `:53`.

## File Structure

**New files**
- `adaptive_roa/conformal/classifier_probability_estimator.py` — `ClassifierProbabilityEstimator` (forward-pass probabilities).
- `adaptive_roa/conformal/estimator_factory.py` — `build_probability_estimator(predictor_type, model, system, config, device)`.
- `adaptive_roa/model/classifier_mlp.py` — `ClassifierMLP(nn.Module)` + `ClassifierModule(pl.LightningModule)` (embed via system → 1 logit, BCE w/ pos_weight).
- `adaptive_roa/data/adaptive_classification_data.py` — `AdaptiveClassificationDataModule` reading `(state..., label)` files.
- `adaptive_roa/adaptive_v2/trainers/classifier_trainer.py` — `ClassifierTrainer(cfg, system, system_name)`, `.fit(...)->ClassifierModule`.
- `adaptive_roa/adaptive_v2/probability/classifier_prob.py` — `ClassifierProbabilityBackend`.
- `configs/adaptive_v2/predictor/generative.yaml`, `configs/adaptive_v2/predictor/classifier.yaml` — config group.
- `tests/` — `conftest.py` + per-task unit/smoke tests.

**Edited files**
- `adaptive_roa/conformal/predictor.py` — `ConformalPredictor.__init__` accepts optional `probability_estimator=None`; use it if provided.
- `adaptive_roa/adaptive_v2/threshold/conformal_threshold.py` — `bind_model` builds estimator via factory, injects into `ConformalPredictor`.
- `adaptive_roa/adaptive/data_source.py` + `adaptive/dataset_builder.py` — add classification build (`save_classification_dataset` / `build_*_classification_dataset`).
- `adaptive_roa/adaptive_v2/pool/trajectory_pool.py` — `initialize`/`build_all_datasets` accept `mode` ("endpoint"|"classification").
- `adaptive_roa/adaptive_v2/eval/full_roa.py` — add `evaluate_full_roa_classifier`; `evaluate_epoch` dispatches on predictor type.
- `adaptive_roa/adaptive_v2/engine.py` — `__init__` selects trainer + probability backend by `cfg.predictor`; `run()` builds classification datasets when classifier.
- `configs/adaptive_v2/default.yaml` — add `predictor: generative` to defaults.

---

## Tasks

### Task 0: pytest harness
- Create: `tests/conftest.py` (sys.path + a dummy 2-state system fixture), `pytest.ini` (`[pytest]\ntestpaths = tests`).
- Verify: `/common/users/dm1487/envs/arcmg/bin/python -m pytest tests/ -q` runs (0 tests OK).
- Commit.

### Task 1: ClassifierProbabilityEstimator + factory
- `ClassifierProbabilityEstimator(model, system, config, device)`: `.estimate(states, verbose=True, **refine_kwargs)`:
  - accept np or tensor `[N, state_dim]`; `x = torch.as_tensor(states, dtype=float32, device); xn = system.normalize_state(x); xe = system.embed_state_for_model(xn); logits = model(xe).squeeze(-1); p_s = sigmoid(logits).cpu().numpy().astype(float64)`; `p_f = 1 - p_s`; `p_inv = zeros_like(p_s)`; return `(p_s, p_f, p_inv)`. `@torch.no_grad()`, `model.eval()`.
- `build_probability_estimator(predictor_type, model, system, config, device)`: `"classifier"` → ClassifierProbabilityEstimator; else `ProbabilityEstimator`.
- Test: dummy model returning fixed logits → assert p_s=sigmoid, p_f=1-p_s, p_inv=0, shapes/dtype. Factory returns right class.
- Commit.

### Task 2: inject estimator into ConformalPredictor + threshold backend
- `predictor.py`: add `probability_estimator: Optional[Any] = None` to `__init__`; replace `:64` construction with `self.prob_estimator = probability_estimator if probability_estimator is not None else ProbabilityEstimator(...)`.
- `conformal_threshold.py` `bind_model`: `est = build_probability_estimator(self.cfg.get("predictor","generative"), model_handle, self.system, conf, self.device)`; pass `probability_estimator=est` to `ConformalPredictor(...)`.
- Test: build ConformalPredictor with an injected fake estimator → `optimize_thresholds` uses it (returns finite λ*,δ*). Generative path unchanged (None → builds ProbabilityEstimator).
- Commit.

### Task 3: ClassifierMLP + ClassifierModule
- `ClassifierMLP(input_dim, hidden_dims, output_dim=1)`: MLP with ReLU; returns logits `[B,1]`.
- `ClassifierModule(pl.LightningModule)`: holds the MLP + a reference to `system` (for normalize+embed) + `pos_weight`; `forward(embedded)->logits`; `training_step`/`validation_step` compute `BCEWithLogitsLoss(pos_weight=...)` on `{"inputs":[B,D_state_raw],"label":[B]}` (normalize+embed inside step); logs `val_loss`, val accuracy; `configure_optimizers` from cfg. Provide a `predict_proba(raw_states)` helper used by the estimator (or the estimator embeds — keep embed in ONE place: the estimator does normalize+embed and calls `model.forward(embedded)`; the module's steps also normalize+embed via system, so the saved module is self-contained).
- Test: forward shape; one train step decreases loss on a trivially separable batch.
- Commit.

### Task 4: classification dataset build (state,label) — writes under EXP_DIR only
- `data_source.save_classification_dataset(indices, filepath, balance=False)`: for each trajectory index, read its states (all points like endpoint build), write `[state_features..., label01]` space-sep where `label01 = 1 if internal_label==1 else 0`. **Assert `DATA_DIR` not in `os.path.abspath(filepath)`.**
- `dataset_builder`: `build_train_classification_dataset`/`build_val_classification_dataset` mirroring endpoint ones; `build_all_classification_datasets()->{train,val}`.
- Test: tmp dir, fake data_source with 2 trajectories (labels 1 and -1) → file has correct rows + label mapping; assert refuses a DATA_DIR path.
- Commit.

### Task 5: AdaptiveClassificationDataModule
- `AdaptiveClassificationDataModule(train_file, val_file, batch_size, num_workers)`: reads space-sep `[state..., label]`; yields `{"inputs": state[:, :state_dim] float32, "label": state[:, -1] float32}`. Computes `pos_weight` from train labels, exposes `.pos_weight`.
- Test: tmp files → batch shapes/dtypes, pos_weight = n_neg/n_pos.
- Commit.

### Task 6: ClassifierTrainer
- `ClassifierTrainer(cfg, system, system_name)`; `.fit(dataset_files, output_dir, resume_checkpoint=None)->ClassifierModule`:
  - dataset_files keys `{train, val}` (classification files). Build datamodule; build `ClassifierMLP` (input=system embedded_dim, hidden from cfg.classifier.hidden_dims); build `ClassifierModule(mlp, system, pos_weight=datamodule.pos_weight, cfg)`; Lightning `Trainer` from `cfg.trainer` (reuse); `fit`; load best ckpt; `.eval().to(device)`; return module.
- Test: smoke — tiny tmp files, 2 epochs CPU, returns a module whose `predict`/forward runs.
- Commit.

### Task 7: ClassifierProbabilityBackend
- `ClassifierProbabilityBackend(system, cfg, device)`: `bind_model(model_handle)` → `self.estimator = ClassifierProbabilityEstimator(model_handle, system, ConformalConfig.from_hydra(cfg), device)`; `estimate(start_states)->OutcomeProbabilities` (mirror EndpointMC).
- Test: bind dummy classifier → estimate returns OutcomeProbabilities with p_invalid all-zero.
- Commit.

### Task 8: classifier eval path
- `evaluate_full_roa_classifier(classifier_module, system, cfg, threshold_state, eval_context)`: load eval/test states (same loader as FM path); forward-pass p_success over grid (batched, GPU); apply threshold via existing `_predict_lambda_delta`/`_predict_lambda_only` (one_sided); compute metrics via `_classification_metrics_from_predictions` + `_conservative_metrics`; skip geodesic/MC-error stats. Return same metrics dict keys the engine logs.
- `evaluate_epoch`: if `cfg.get("predictor","generative")=="classifier"` → call classifier eval; else `evaluate_full_roa_fast`.
- Test: monkeypatch a classifier returning a separable p_success on a tiny labeled grid → metrics dict has accuracy/f1 keys, sane values.
- Commit.

### Task 9: engine factory wiring
- `engine.py __init__`: `pred = str(cfg.get("predictor","generative"))`; if `pred=="classifier"`: `self.trainer=ClassifierTrainer(...)`, `self.probability_backend=ClassifierProbabilityBackend(...)`; else current. threshold_backend/evaluator unchanged (they branch internally via cfg.predictor).
- `run()`: when classifier, build classification datasets (`pool.initialize(..., mode="classification")` and `pool.build_all_datasets(mode="classification")`) so trainer gets `{train,val}` (state,label) files.
- Pool: thread `mode` through `initialize`/`build_all_datasets` → endpoint (default) vs classification build.
- Test: construct engine with `predictor=classifier` (smoke_mode) → wires classifier classes; `predictor=generative` → unchanged classes.
- Commit.

### Task 10: predictor config group
- `configs/adaptive_v2/predictor/generative.yaml`: `# @package _global_\npredictor: generative`.
- `configs/adaptive_v2/predictor/classifier.yaml`: `# @package _global_\npredictor: classifier` + `classifier: {hidden_dims: [256,512,256], lr: 1e-3, max_epochs: 200, patience: 20}` + sensible trainer overrides for classification (monitor val_loss).
- `default.yaml`: add `- predictor: generative` to defaults.
- Test: `--cfg job` composes for `predictor=classifier` and shows `predictor: classifier` (and unchanged for generative).
- Commit.

### Task 11: smoke + full quad2d run
- Smoke: `... system=quadrotor2d predictor=classifier sampling_mode=direct d2_ratio=1.0 conformal.decision_rule=one_sided device=cuda:0 n_epochs=2 initial_train_size=300 samples_per_epoch=100 adaptive_v2.smoke_mode=false eval_every=1` → end-to-end runs, produces epoch metrics. Debug until green.
- Full (background, overnight): same with `n_epochs=<fits by morning>` (calibrate from smoke per-epoch time), `initial_train_size=2000 samples_per_epoch=500`. Output under EXP_DIR.
- (If time) generative baseline: same config `predictor=generative` for side-by-side.
- Collect: per-epoch ROA accuracy/F1/coverage learning curve; summarize.

## Run commands

```bash
PY=/common/users/dm1487/envs/arcmg/bin/python
# smoke
$PY scripts/run_adaptive.py system=quadrotor2d predictor=classifier \
  sampling_mode=direct d2_ratio=1.0 conformal.decision_rule=one_sided \
  device=cuda:0 n_epochs=2 initial_train_size=300 samples_per_epoch=100
# full (background)
$PY scripts/run_adaptive.py system=quadrotor2d predictor=classifier \
  sampling_mode=direct d2_ratio=1.0 conformal.decision_rule=one_sided \
  device=cuda:0 n_epochs=40 initial_train_size=2000 samples_per_epoch=500
```

## Self-review notes
- Coverage: acquisition (Task 7), threshold-opt (Task 2), eval (Task 8) all routed through the seam → no flow-matching call remains on the classifier path.
- Generative path untouched when `predictor=generative` (factory default + None-injection fallback).
- No DATA_DIR writes (Task 4 assert).
- Imbalance handled (pos_weight, Tasks 5/6).

## Status (2026-06-19)

- Tasks 0–10 **implemented + unit/smoke tested** (13 tests green via the arcmg python). Each task committed separately on `direct-classification`.
- Task 11: quad2d classifier pipeline **validated end-to-end** (smoke EXIT=0, F1≈0.59 at 1 epoch); all outputs under EXP_DIR, **nothing written to DATA_DIR** (verified).
- **NFS fix:** classifier DataLoader uses `num_workers=0` (workers crashed on `rmtree` of `.nfs*` temp dirs, Errno 16). Data is in-memory so workers add no value.
- **Full run launched** (background): `n_epochs=60 initial_train_size=2000 samples_per_epoch=500 classifier.max_epochs=80` → self-terminates ~epoch 36 when the 20k-trajectory pool exhausts. Log: `/tmp/clf_quad2d_full.log`. Output dir under `EXP_DIR/adaptive_quadrotor2d/outputs/...sampling_mode_direct/<ts>/`.
- Generative side-by-side (`predictor=generative`, same config) NOT yet run — much slower (MC + ODE); deferred.
