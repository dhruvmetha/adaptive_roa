"""Executable spec for the label conventions.

Label values mean DIFFERENT things at different layers of this repo. That has bitten people.
Prose kept going stale, so the codebook lives here instead: if someone changes a convention,
this fails.

    layer                          values              meaning
    -------------------------------------------------------------------------------
    1. on disk (deterministic)     {0, 1}              0 = failure, 1 = success
    2. internal (data_source)      {-1, 0, +1}        -1 = failure, 0 = SEPARATRIX, 1 = success
    3. system.classify_attractor   {-1, 0, +1}        -1 = failure, 0 = SEPARATRIX, 1 = success
    4. evaluate_roa predictions    {1, 0, -1, -2}      1 = success, 0 = FAILURE,
                                                      -1 = uncertain, -2 = invalid

The traps:
  * 0 means SEPARATRIX in layers 2-3 but FAILURE in layer 4.
  * -1 means FAILURE in layers 2-3 but UNCERTAIN in layer 4.
  * layer-4 -1 and -2 are dropped from metrics, so F1 is computed on a retained subset and is
    NOT comparable across methods with different abstention rates.
"""
import torch
import pytest

from adaptive_roa.systems.pendulum import PendulumSystem


SUCCESS, FAILURE, SEPARATRIX = 1, -1, 0


@pytest.fixture(scope="module")
def pendulum():
    return PendulumSystem()


def test_classify_attractor_codebook(pendulum):
    """Layer 3: -1 = failure, 0 = separatrix, +1 = success."""
    states = torch.tensor([
        [0.0, 0.0],    # upright / goal          -> SUCCESS
        [2.1, 0.0],    # saturation equilibrium  -> FAILURE
        [-2.1, 0.0],   # saturation equilibrium  -> FAILURE
        [1.0, 0.0],    # in neither basin        -> SEPARATRIX
    ])
    got = pendulum.classify_attractor(states).tolist()
    assert got == [SUCCESS, FAILURE, FAILURE, SEPARATRIX]


def test_zero_means_separatrix_not_failure(pendulum):
    """The trap: 0 is SEPARATRIX here, but 0 means FAILURE in evaluate_roa's predictions.

    If this ever starts returning FAILURE for an unresolved state, the two conventions have been
    unified and the docs/tests describing the flip must be updated.
    """
    unresolved = torch.tensor([[1.0, 0.0]])
    assert pendulum.classify_attractor(unresolved).item() == SEPARATRIX
    assert pendulum.classify_attractor(unresolved).item() != FAILURE


def test_goal_is_upright_not_bottom(pendulum):
    """theta=0 is the UNSTABLE upright equilibrium and is the goal.

    The EOM is theta_ddot = (g/l) sin(theta) + u/I - (b/I) theta_dot, so
    d(theta_ddot)/d(theta) at 0 = +(g/l) > 0 -> unstable. Both dataset_description.json files
    agree that 0 = upright. Comments in pendulum.py and configs/system/pendulum.yaml once claimed
    0 was a stable "bottom" equilibrium; they were wrong and have been corrected.
    """
    assert pendulum.classify_attractor(torch.tensor([[0.0, 0.0]])).item() == SUCCESS


def test_failure_attractors_are_the_saturation_equilibria(pendulum):
    """+/-2.1 is not "the top" -- it is where saturated torque balances gravity.

        sin(theta*) = (u_sat / I) / (g / l) = 0.866025   ->   theta* = 2.0944 rad = 120 deg

    Guards the magic constant against being "corrected" to +/-pi by someone who reads it as the
    downward/top equilibrium.
    """
    import math
    g, l, I, u_sat = 9.81, 0.5, 0.0375, 0.6371781908344007
    theta_star = math.pi - math.asin((u_sat / I) / (g / l))
    assert theta_star == pytest.approx(2.0944, abs=1e-3)

    attractors = [list(map(float, a)) for a in pendulum.attractors()]
    assert [0.0, 0.0] in attractors
    for a in attractors:
        if a != [0.0, 0.0]:
            assert abs(abs(a[0]) - 2.1) < 0.05, f"unexpected attractor {a}"
