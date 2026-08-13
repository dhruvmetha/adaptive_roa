import json

import pytest

from adaptive_roa.data.state_schema import (
    CANONICAL_CARTPOLE_STATE_ORDER,
    validate_dataset_state_order,
)


def _write_description(tmp_path, value):
    (tmp_path / "dataset_description.json").write_text(json.dumps(value))


def test_validate_dataset_state_order_accepts_consistent_nested_declarations(tmp_path):
    order = list(CANONICAL_CARTPOLE_STATE_ORDER)
    _write_description(
        tmp_path,
        {"state_space": {"state_order": order}, "collection": {"state_order": order}},
    )

    assert validate_dataset_state_order(tmp_path, order) == tuple(order)


def test_validate_dataset_state_order_rejects_mismatch(tmp_path):
    _write_description(
        tmp_path,
        {"state_space": {"state_order": ["x", "x_dot", "theta", "theta_dot"]}},
    )

    with pytest.raises(ValueError, match="State-order mismatch"):
        validate_dataset_state_order(tmp_path, CANONICAL_CARTPOLE_STATE_ORDER)


def test_validate_dataset_state_order_rejects_missing_declaration(tmp_path):
    _write_description(tmp_path, {"dataset_name": "ambiguous"})

    with pytest.raises(ValueError, match="does not declare state_order"):
        validate_dataset_state_order(tmp_path, CANONICAL_CARTPOLE_STATE_ORDER)
