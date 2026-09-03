import json

import torch

from adaptive_roa.systems.cartpole import CartPoleSystem


def _stochastic_cartpole(tmp_path):
    bounds = {
        name: {"min": -10.0, "max": 10.0}
        for name in ("x", "theta", "x_dot", "theta_dot")
    }
    description = {
        "achieved_bounds": bounds,
        "state_space": {
            "state_order": ["x", "theta", "x_dot", "theta_dot"]
        },
        "collection": {
            "success_rule": {
                "kind": "per_channel_box_entry",
                "tol": [0.1, 0.1, 0.1, 0.1],
                "hold_steps": 1,
            }
        },
    }
    (tmp_path / "dataset_description.json").write_text(json.dumps(description))
    return CartPoleSystem(dataset_dir=str(tmp_path))


def test_stochastic_cartpole_uses_declared_box_not_l2_ball(tmp_path):
    system = _stochastic_cartpole(tmp_path)
    states = torch.tensor(
        [
            [0.09, 0.09, 0.09, 0.09],  # inside box, outside radius-0.1 L2 ball
            [0.00, 0.11, 0.00, 0.00],  # theta outside box
            [0.00, 2 * torch.pi, 0.00, 0.00],  # raw theta is not wrapped
        ]
    )

    assert system.is_in_attractor(states, radius=0.1).tolist() == [True, False, False]
    assert system.classify_attractor(states, radius=0.1).tolist() == [1, 0, 0]


def _rl_cartpole(tmp_path):
    """safe_explorer_ppo geometry: the cart parks at x = 0.7, |theta| ends at 0.48.

    No success_rule in the description, so this takes the L2 branch. That branch
    measures distance from self.goal rather than from the origin, and nothing
    else in the suite builds a CartPoleSystem with a goal away from the origin.
    """
    bounds = {
        name: {"min": -3.0, "max": 3.0}
        for name in ("x", "theta", "x_dot", "theta_dot")
    }
    description = {
        "achieved_bounds": bounds,
        "state_space": {
            "state_order": ["x", "theta", "x_dot", "theta_dot"]
        },
    }
    (tmp_path / "dataset_description.json").write_text(json.dumps(description))
    return CartPoleSystem(
        dataset_dir=str(tmp_path),
        goal=[0.7, 0.0, 0.0, 0.0],
        angle_limit=0.48,
    )


def test_success_is_measured_from_the_goal_not_the_origin(tmp_path):
    """Guards the failure that withdrew the 2026-08-31 fleet.

    Scoring an RL run against an origin-centred ball marks every real success a
    failure, p_hat collapses to ~0 and sAUROC pins at exactly 0.500. Row 1 is
    where the policy actually parks (||state|| ~ 0.72); row 2 is the origin,
    which only an origin-centred rule would call a success.
    """
    system = _rl_cartpole(tmp_path)
    states = torch.tensor(
        [
            [0.72, 0.00, 0.00, 0.00],  # where the policy parks: SUCCESS
            [0.00, 0.00, 0.00, 0.00],  # the origin: NOT the goal here
            [0.70, 0.40, 0.00, 0.00],  # right x, theta too far from goal theta
        ]
    )

    assert system.is_in_attractor(states, radius=0.1).tolist() == [True, False, False]
    assert system.classify_attractor(states, radius=0.1).tolist() == [1, 0, 0]
    assert system.attractors() == [[0.7, 0.0, 0.0, 0.0]]


def test_theta_distance_wraps_around_the_goal_angle(tmp_path):
    """The circular component subtracts the goal angle before wrapping.

    A goal at theta = pi - 0.05 must accept theta = -pi + 0.05, which is 0.1 away
    the short way round and 2*pi - 0.1 away if the code wraps theta alone.
    """
    bounds = {
        name: {"min": -3.0, "max": 3.0}
        for name in ("x", "theta", "x_dot", "theta_dot")
    }
    (tmp_path / "dataset_description.json").write_text(json.dumps({
        "achieved_bounds": bounds,
        "state_space": {"state_order": ["x", "theta", "x_dot", "theta_dot"]},
    }))
    goal_theta = torch.pi - 0.05
    system = CartPoleSystem(
        dataset_dir=str(tmp_path), goal=[0.0, goal_theta, 0.0, 0.0]
    )

    states = torch.tensor([[0.0, -torch.pi + 0.05, 0.0, 0.0]])
    assert bool(system.is_in_attractor(states, radius=0.2)) is True


def test_angle_limit_override_reaches_the_normalisation_bounds(tmp_path):
    """pi would squeeze this policy's usable range into +-0.153 of the axis."""
    system = _rl_cartpole(tmp_path)
    assert system.angle_limit == 0.48
    assert system.define_state_bounds()["pole_angle"] == (-0.48, 0.48)
