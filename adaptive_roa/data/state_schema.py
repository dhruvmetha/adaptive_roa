"""State-schema declarations and validation for trajectory datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


CANONICAL_CARTPOLE_STATE_ORDER = ("x", "theta", "x_dot", "theta_dot")


def _declared_state_orders(value: Any, path: str = "") -> list[tuple[str, tuple[str, ...]]]:
    """Return every ``state_order`` declaration in a JSON-like object."""
    found: list[tuple[str, tuple[str, ...]]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else key
            if key == "state_order":
                if not isinstance(child, list) or not all(isinstance(v, str) for v in child):
                    raise ValueError(f"{child_path} must be a list of coordinate names")
                found.append((child_path, tuple(child)))
            else:
                found.extend(_declared_state_orders(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_declared_state_orders(child, f"{path}[{index}]"))
    return found


def validate_dataset_state_order(
    dataset_dir: str | Path,
    expected: Iterable[str],
    metadata_name: str = "dataset_description.json",
) -> tuple[str, ...]:
    """Require every metadata declaration to match ``expected`` exactly.

    Refusing missing metadata is intentional: silently guessing a state order can
    train a plausible-looking model on the wrong physical coordinates.
    """
    dataset_dir = Path(dataset_dir)
    metadata_path = dataset_dir / metadata_name
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Cannot validate state order: {metadata_path} does not exist"
        )

    with metadata_path.open() as stream:
        metadata = json.load(stream)

    declarations = _declared_state_orders(metadata)
    if not declarations:
        raise ValueError(f"{metadata_path} does not declare state_order")

    expected_tuple = tuple(expected)
    mismatches = [
        f"{path}={list(order)!r}"
        for path, order in declarations
        if order != expected_tuple
    ]
    if mismatches:
        details = ", ".join(mismatches)
        raise ValueError(
            f"State-order mismatch in {metadata_path}: expected "
            f"{list(expected_tuple)!r}; found {details}"
        )
    return expected_tuple
