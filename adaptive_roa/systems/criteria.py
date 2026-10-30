"""One source for each system's success and failure criteria.

A system built from config holds every criterion itself: goal, success test, and
failure test (constructor arguments set in configs/adaptive_v2/system/<name>.yaml).
Callers still pass a radius to ``classify_attractor`` (the top-level
``attractor_radius`` points at the system's own value). These helpers make a
second source impossible to use silently: a passed radius that disagrees with the
configured one raises, and ``require_criteria`` refuses to fall back to a value
written in the class.
"""
from __future__ import annotations


def resolve_radius(configured: float | None, passed, system: str, legacy_default: float | None = None) -> float:
    """The success radius to test with: the configured one, which a passed radius must match.

    ``legacy_default`` keeps the old keyword default for callers outside the det
    configs that construct a system without criteria and pass no radius; a det
    config sets ``require_criteria`` and can never reach it.
    """
    if configured is None:
        if passed is None:
            if legacy_default is None:
                raise ValueError(f"{system}: no success radius configured and none passed")
            return float(legacy_default)
        return float(passed)
    if passed is not None and abs(float(passed) - configured) > 1e-12:
        raise ValueError(
            f"{system}: classify_attractor was passed radius {float(passed)}, but the system's "
            f"criteria say {configured}. The criteria have one source, the system config "
            f"(configs/adaptive_v2/system/*.yaml); attractor_radius must point at it.")
    return configured


def check_required(system: str, **criteria) -> None:
    """With require_criteria, every criterion must come from config, never a class default."""
    missing = sorted(k for k, v in criteria.items() if v is None)
    if missing:
        raise ValueError(f"{system}: require_criteria is set but {missing} are not configured; "
                         f"set them in the system config rather than relying on class defaults")


def optional_radius(value) -> float | None:
    """Read attractor_radius from config: None where the success test is not a ball (cartpole box)."""
    return None if value is None else float(value)
