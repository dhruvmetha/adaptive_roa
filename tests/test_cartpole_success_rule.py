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
