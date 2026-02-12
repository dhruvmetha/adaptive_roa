# Adaptive v2 Migration

## Command mapping
- Old: `python scripts/run_adaptive_pendulum.py ...`
  New: `python scripts/run_adaptive.py system=pendulum ...`
- Old: `python scripts/run_adaptive_cartpole.py ...`
  New: `python scripts/run_adaptive.py system=cartpole_pybullet ...`
- Old: `python scripts/run_adaptive_quadrotor2d.py ...`
  New: `python scripts/run_adaptive.py system=quadrotor2d ...`
- Old: `python scripts/run_adaptive_quadrotor3d.py ...`
  New: `python scripts/run_adaptive.py system=quadrotor3d ...`

## Removed scripts
- `scripts/run_adaptive_pendulum.py`
- `scripts/run_adaptive_cartpole.py`
- `scripts/run_adaptive_quadrotor2d.py`
- `scripts/run_adaptive_quadrotor3d.py`

## Reevaluation coupling fix
- Reevaluation scripts now import from:
  `adaptive_roa.adaptive_v2.eval.full_roa`
- They no longer import evaluation functions from adaptive run scripts.
- They read per-epoch thresholds from `epoch_XXX/artifacts_v2.json`.

## Legacy exception
- `scripts/run_adaptive_mountain_car.py` remains legacy in this phase.
- Four-system v1 root configs are archived under `configs/archive/legacy_adaptive_v1/`.

## Smoke-test overrides
- Fast pipeline checks can use:
  - `+adaptive_v2.smoke_mode=true`
  - `+trainer.limit_train_batches=1`
  - `+trainer.limit_val_batches=1`
- For `quadrotor3d` smoke runs, prefer `initial_train_size>=20` to avoid degenerate one-sample validation splits.

## Canonical invocation
- Preferred: `python scripts/run_adaptive.py system=<pendulum|cartpole_pybullet|quadrotor2d|quadrotor3d> ...`
- The script's Hydra config root is `configs/adaptive_v2/`, so `system=quadrotor2d` directly selects the config group.
