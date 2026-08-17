"""CartPoleSystem must read either dataset_description schema, not assume one.

Two schemas ship. The OLD one (noisy_action, deterministic) carries
``achieved_bounds``. The NEW one (noisy_torque) does not -- it declares
``termination_thresholds``, and its own note records that the numbers moved:
"The previously shipped stochastic set relaxed x_dot/theta_dot to 20.0; this
restores the deterministic 5.0."

Loading the new data with the old assumption is not a crash-and-notice bug --
it raised KeyError('achieved_bounds') and killed two launched runs in 8s. The
subtler risk is the opposite: silently reusing the old family's wider limits
would inflate the normalisation scales ~1.5x on x_dot and ~1.7x on theta_dot,
rescaling the embedding the kNN/MLP length models measure distance in.
"""

import numpy as np
import pytest

from adaptive_roa.systems.cartpole import CartPoleSystem


OLD_SCHEMA = {
    "achieved_bounds": {
        "x":         {"min": -6.044, "max": 6.048},
        "theta":     {"min": -3.142, "max": 3.142},
        "x_dot":     {"min": -7.315, "max": 7.338},
        "theta_dot": {"min": -8.499, "max": 8.571},
    },
    "state_space": {"state_order": ["x", "theta", "x_dot", "theta_dot"]},
}

NEW_SCHEMA = {
    "termination_thresholds": {
        "x": 6.0, "x_dot": 5.0, "theta_dot": 5.0, "theta": "inf (periodic)",
    },
    "sampling": {"train": {"bounds": {"x": 6.0, "x_dot": 5.0,
                                      "theta": np.pi, "theta_dot": 5.0}}},
    "data_format": {"state_order": ["x", "theta", "x_dot", "theta_dot"]},
}


def test_old_schema_is_returned_untouched():
    out = CartPoleSystem._bounds_from_schema(OLD_SCHEMA)
    assert out is OLD_SCHEMA["achieved_bounds"]


def test_new_schema_maps_termination_thresholds_to_bounds():
    out = CartPoleSystem._bounds_from_schema(NEW_SCHEMA)
    assert out["x"] == {"min": -6.0, "max": 6.0}
    assert out["x_dot"] == {"min": -5.0, "max": 5.0}
    assert out["theta_dot"] == {"min": -5.0, "max": 5.0}
    assert out["theta"]["max"] == pytest.approx(np.pi)


def test_the_two_schemas_give_genuinely_different_limits():
    """Guards the silent-failure mode: if these ever coincide, a regression that
    reuses the old bounds on new data would pass unnoticed."""
    old = CartPoleSystem._bounds_from_schema(OLD_SCHEMA)
    new = CartPoleSystem._bounds_from_schema(NEW_SCHEMA)
    assert old["x_dot"]["max"] / new["x_dot"]["max"] > 1.4
    assert old["theta_dot"]["max"] / new["theta_dot"]["max"] > 1.6


def test_theta_string_sentinel_does_not_leak_through():
    """The new schema stores theta as the string 'inf (periodic)'. It must be
    replaced by +/-pi, never returned as a string into arithmetic."""
    out = CartPoleSystem._bounds_from_schema(NEW_SCHEMA)
    assert isinstance(out["theta"]["min"], float)
    assert isinstance(out["theta"]["max"], float)


def test_a_schema_with_neither_source_fails_loudly():
    with pytest.raises(KeyError, match="achieved_bounds"):
        CartPoleSystem._bounds_from_schema({"dataset_name": "nothing useful"})


def test_sampling_bounds_used_when_termination_thresholds_absent():
    schema = {"sampling": {"train": {"bounds": {"x": 4.0, "x_dot": 3.0,
                                                "theta": np.pi, "theta_dot": 2.0}}}}
    out = CartPoleSystem._bounds_from_schema(schema)
    assert out["x"]["max"] == 4.0 and out["theta_dot"]["max"] == 2.0
