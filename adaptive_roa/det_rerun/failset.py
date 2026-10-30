"""First-hitting outcomes for pendulum trajectories with an explicit failure set.

The `pendulum/lqr` collector stops a trajectory at its first entry into the success
ball and lets every other trajectory run to the horizon. Its failures settle at two
equilibria where saturated torque balances gravity, so the failure set is a ball
around each. A trajectory's outcome is whichever set it enters first.
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import numpy as np


def wrap_angle(a: np.ndarray) -> np.ndarray:
    """Map angles to [-pi, pi]."""
    return np.arctan2(np.sin(a), np.cos(a))


def pendulum_membership(
    states: np.ndarray,
    success_radius: float,
    failure_centers: Iterable[Sequence[float]],
    failure_radius: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-row membership of [T, 2] (theta, theta_dot) states in the success and failure sets.

    Success uses the collector's rule, sqrt(theta^2 + theta_dot^2) < radius on the stored
    theta, so it reproduces the stored labels. Failure wraps the angle difference, since
    the failure centres sit 1.05 rad from the +-pi seam.
    """
    th, thd = states[:, 0], states[:, 1]
    in_success = np.sqrt(th ** 2 + thd ** 2) < success_radius
    in_failure = np.zeros(len(states), dtype=bool)
    for c_th, c_thd in failure_centers:
        d = np.sqrt(wrap_angle(th - c_th) ** 2 + (thd - c_thd) ** 2)
        in_failure |= d < failure_radius
    return in_success, in_failure


def first_hit(in_success: np.ndarray, in_failure: np.ndarray) -> Tuple[Optional[int], int]:
    """Row and outcome of the first entry into either set.

    Returns (row, 1) for success, (row, 0) for failure, (None, -1) when neither set is
    entered (a timeout). Raises if a row lies in both sets, which would make the
    outcome ambiguous.
    """
    both = in_success & in_failure
    if both.any():
        raise ValueError(f"row {int(np.argmax(both))} lies in both the success and failure sets")
    hit = in_success | in_failure
    if not hit.any():
        return None, -1
    row = int(np.argmax(hit))
    return row, 1 if in_success[row] else 0
