# Legacy Adaptive v1 (Archived)

This document is for historical reference only.

## Archived four-system configs

The old root adaptive configs for unified systems were moved to:

- `configs/archive/legacy_adaptive_v1/adaptive_pendulum.yaml`
- `configs/archive/legacy_adaptive_v1/adaptive_cartpole_pybullet.yaml`
- `configs/archive/legacy_adaptive_v1/adaptive_quadrotor2d.yaml`
- `configs/archive/legacy_adaptive_v1/adaptive_quadrotor3d.yaml`

These are no longer used by active v2 run paths.

## Removed four-system run scripts

The following scripts were removed in the hard cutover:

- `scripts/run_adaptive_pendulum.py`
- `scripts/run_adaptive_cartpole.py`
- `scripts/run_adaptive_quadrotor2d.py`
- `scripts/run_adaptive_quadrotor3d.py`

## Active replacement

Use the unified entrypoint:

```bash
python scripts/run_adaptive.py system=<pendulum|cartpole_pybullet|quadrotor2d|quadrotor3d>
```

## Legacy exception

`mountain_car` remains legacy in this phase and still runs via:

```bash
python scripts/run_adaptive_mountain_car.py
```
