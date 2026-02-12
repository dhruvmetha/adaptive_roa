# Adaptive v2 Architecture

## Unified entrypoint
- Command: `python scripts/run_adaptive.py ...`
- System selection: `system=<pendulum|cartpole_pybullet|quadrotor2d|quadrotor3d>`

## Pipeline components
- Engine: `adaptive_roa/adaptive_v2/engine.py`
- Pool: `adaptive_roa/adaptive_v2/pool/trajectory_pool.py`
- Trainer: `adaptive_roa/adaptive_v2/trainers/flow_matching_trainer.py`
- Probability backend: `adaptive_roa/adaptive_v2/probability/endpoint_mc.py`
- Threshold backend: `adaptive_roa/adaptive_v2/threshold/conformal_threshold.py`
- Strategies:
  - `adaptive_roa/adaptive_v2/strategy/ranked.py`
  - `adaptive_roa/adaptive_v2/strategy/conformal.py`
  - `adaptive_roa/adaptive_v2/strategy/direct.py`
- Evaluation: `adaptive_roa/adaptive_v2/eval/full_roa.py`

## Interfaces and types
- Contracts: `adaptive_roa/adaptive_v2/interfaces.py`
- Core data types: `adaptive_roa/adaptive_v2/types.py`

## Behavior notes
- Uses one shared epoch loop across all 4 systems.
- Uses self-contained v2 system configs in `configs/adaptive_v2/system/*.yaml`.
- Keeps conformal, direct, and ranked sampling semantics.
- Keeps reevaluation compatibility through canonical `evaluate_full_roa_fast` in v2 eval.
- Uses `artifacts_v2.json` as the canonical per-epoch artifact format.
- `artifacts_v2.json` includes `legacy_epoch_metrics` and `conformal_state` so legacy statistics are retained in the canonical file.
- Legacy `results.json` and `conformal_state.json` are also emitted by default (`adaptive_v2.save_legacy_results_json=true`).
- If a config requests CUDA but CUDA is unavailable, v2 falls back to CPU automatically.
- Archived v1 configs for the four unified systems are stored in `configs/archive/legacy_adaptive_v1/`.

## Smoke test mode
- `+adaptive_v2.smoke_mode=true` skips expensive endpoint/full-ROA evaluation in the epoch loop.
- `+trainer.limit_train_batches=1` and `+trainer.limit_val_batches=1` are supported by the v2 trainer adapter for fast pipeline checks.
- For tiny smoke runs, use `sampling_mode=ranked` with small `n_ranked_candidates`, `batch_size_sampling`, and `max_samples_per_epoch`.
