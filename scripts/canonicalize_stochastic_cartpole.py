#!/usr/bin/env python3
"""Create a canonical-order copy of the stochastic CartPole datasets.

The Safe-Control-Gym artifacts use environment order
``(x, x_dot, theta, theta_dot)``. Adaptive ROA models use canonical order
``(x, theta, x_dot, theta_dot)``. This script swaps columns 1 and 2 in every
state-bearing artifact while preserving labels, probabilities, counts, split
membership, trajectory boundaries, seeds, shapes, and dtypes.

The source tree is read-only from this script's perspective. Each destination
noise-level directory is assembled under a temporary sibling and renamed into
place only after complete validation.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
from itertools import zip_longest
from pathlib import Path
from typing import Any

import numpy as np


SOURCE_ORDER = ("x", "x_dot", "theta", "theta_dot")
TARGET_ORDER = ("x", "theta", "x_dot", "theta_dot")
PERMUTATION = (0, 2, 1, 3)
DEFAULT_LEVELS = ("sigma_015.0", "sigma_020.0", "sigma_030.0", "sigma_040.0")
STATE_TEXT_FILES = {
    "eval_states.txt",
    "cal_set.txt",
    "test_set.txt",
    "success_probabilities.txt",
}
STATE_NPZ_KEYS = {
    "train.npz": {"states", "starts"},
    "eval_success_prob.npz": {"starts"},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _script_revision() -> dict[str, Any]:
    script = Path(__file__).resolve()
    repo = script.parent.parent
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    return {
        "script": str(script.relative_to(repo)),
        "script_sha256": sha256(script),
        "git_revision": revision,
    }


def _find_state_orders(value: Any, path: str = "") -> list[tuple[str, tuple[str, ...]]]:
    found: list[tuple[str, tuple[str, ...]]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else key
            if key == "state_order":
                if not isinstance(child, list):
                    raise ValueError(f"{child_path} is not a list")
                found.append((child_path, tuple(child)))
            else:
                found.extend(_find_state_orders(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_find_state_orders(child, f"{path}[{index}]"))
    return found


def _replace_state_orders(value: Any) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "state_order":
                value[key] = list(TARGET_ORDER)
            else:
                _replace_state_orders(child)
    elif isinstance(value, list):
        for child in value:
            _replace_state_orders(child)


def _conversion_record(source_level: Path) -> dict[str, Any]:
    return {
        "kind": "state_order_only",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_directory": str(source_level.resolve()),
        "source_state_order": list(SOURCE_ORDER),
        "target_state_order": list(TARGET_ORDER),
        "target_from_source_indices": list(PERMUTATION),
        "labels_changed": False,
        "probabilities_changed": False,
        "physics_rerun": False,
        "success_rule_changed": False,
        "converter": _script_revision(),
    }


def convert_json(source: Path, destination: Path, record: dict[str, Any]) -> None:
    with source.open() as stream:
        value = json.load(stream)
    declarations = _find_state_orders(value)
    if declarations:
        bad = [(path, order) for path, order in declarations if order != SOURCE_ORDER]
        if bad:
            raise ValueError(f"Unexpected state_order declaration in {source}: {bad}")
        _replace_state_orders(value)
    value["schema_version"] = 2
    value["canonicalization"] = record
    destination.write_text(json.dumps(value, indent=2) + "\n")


def convert_text(source: Path, destination: Path) -> None:
    """Permute the first four comma-separated fields without reformatting them."""
    with source.open() as src, destination.open("w") as dst:
        for line_number, line in enumerate(src, start=1):
            stripped = line.rstrip("\r\n")
            newline = line[len(stripped):]
            fields = stripped.split(",")
            if len(fields) < 4:
                raise ValueError(f"{source}:{line_number} has fewer than four columns")
            reordered = [fields[index] for index in PERMUTATION] + fields[4:]
            dst.write(",".join(reordered) + newline)


def convert_npz(source: Path, destination: Path) -> None:
    keys_to_permute = STATE_NPZ_KEYS[source.name]
    with np.load(source) as archive:
        arrays: dict[str, np.ndarray] = {}
        for key in archive.files:
            value = archive[key]
            if key in keys_to_permute:
                if value.ndim != 2 or value.shape[1] != len(SOURCE_ORDER):
                    raise ValueError(
                        f"{source}:{key} has shape {value.shape}, expected (*, 4)"
                    )
                value = value[:, PERMUTATION]
            arrays[key] = value
        with destination.open("wb") as stream:
            np.savez(stream, **arrays)


def _verify_npz(source: Path, destination: Path) -> None:
    keys_to_permute = STATE_NPZ_KEYS[source.name]
    with np.load(source) as before, np.load(destination) as after:
        if before.files != after.files:
            raise AssertionError(f"NPZ key order changed for {source.name}")
        for key in before.files:
            expected = before[key][:, PERMUTATION] if key in keys_to_permute else before[key]
            actual = after[key]
            if expected.shape != actual.shape or expected.dtype != actual.dtype:
                raise AssertionError(f"shape/dtype changed for {source.name}:{key}")
            if not np.array_equal(expected, actual, equal_nan=True):
                raise AssertionError(f"values changed unexpectedly for {source.name}:{key}")


def _verify_text(source: Path, destination: Path) -> None:
    with source.open() as before, destination.open() as after:
        for line_number, pair in enumerate(zip_longest(before, after), start=1):
            old_line, new_line = pair
            if old_line is None or new_line is None:
                raise AssertionError(f"line count changed for {source.name}")
            old = old_line.rstrip("\r\n").split(",")
            new = new_line.rstrip("\r\n").split(",")
            expected = [old[index] for index in PERMUTATION] + old[4:]
            if new != expected:
                raise AssertionError(f"text conversion mismatch at {source}:{line_number}")


def verify_level(source: Path, destination: Path) -> None:
    source_files = {
        path.relative_to(source) for path in source.rglob("*") if path.is_file()
    }
    destination_files = {
        path.relative_to(destination)
        for path in destination.rglob("*")
        if path.is_file() and path.name != "conversion_manifest.json"
    }
    if source_files != destination_files:
        raise AssertionError(
            f"file-set mismatch: missing={source_files - destination_files}, "
            f"extra={destination_files - source_files}"
        )

    for relative in sorted(source_files):
        src = source / relative
        dst = destination / relative
        if src.name in STATE_NPZ_KEYS:
            _verify_npz(src, dst)
        elif src.name in STATE_TEXT_FILES:
            _verify_text(src, dst)
        elif src.suffix == ".json":
            with dst.open() as stream:
                converted = json.load(stream)
            declarations = _find_state_orders(converted)
            if declarations and any(order != TARGET_ORDER for _, order in declarations):
                raise AssertionError(f"non-canonical metadata remains in {dst}")
            if converted.get("schema_version") != 2:
                raise AssertionError(f"schema_version missing in {dst}")
        elif sha256(src) != sha256(dst):
            raise AssertionError(f"non-state file changed: {relative}")


def convert_level(source: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing destination: {destination}")
    if source.resolve() == destination.resolve():
        raise ValueError("Source and destination must differ")
    if not source.is_dir():
        raise FileNotFoundError(source)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.tmp-{os.getpid()}"
    if temporary.exists():
        raise FileExistsError(f"Temporary path already exists: {temporary}")
    temporary.mkdir()
    record = _conversion_record(source)
    try:
        source_hashes: dict[str, str] = {}
        for src in sorted(path for path in source.rglob("*") if path.is_file()):
            relative = src.relative_to(source)
            dst = temporary / relative
            dst.parent.mkdir(parents=True, exist_ok=True)
            source_hashes[str(relative)] = sha256(src)
            if src.name in STATE_NPZ_KEYS:
                convert_npz(src, dst)
            elif src.name in STATE_TEXT_FILES:
                convert_text(src, dst)
            elif src.suffix == ".json":
                convert_json(src, dst, record)
            else:
                shutil.copy2(src, dst)

        manifest = {
            **record,
            "source_file_sha256": source_hashes,
        }
        (temporary / "conversion_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
        verify_level(source, temporary)
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--destination-root", type=Path, required=True)
    parser.add_argument("--levels", nargs="+", default=list(DEFAULT_LEVELS))
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="validate already-converted destination levels without writing",
    )
    args = parser.parse_args()

    for level in args.levels:
        source = args.source_root / level
        destination = args.destination_root / level
        if args.verify_only:
            verify_level(source, destination)
            action = "verified"
        else:
            convert_level(source, destination)
            action = "converted and verified"
        print(f"[{level}] {action}: {destination}", flush=True)


if __name__ == "__main__":
    main()
