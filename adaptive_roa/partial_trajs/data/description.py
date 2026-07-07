"""Parser for partial-trajectory horizon-dataset ``dataset_description.json``.

Exposes the horizon-model and source-system metadata the loaders and verifier
need, in a system-agnostic dataclass. The resolution *criteria* themselves live
in ``systems/`` (``classify_attractor``); this parser only reads structural
metadata (T, K, l, dims, goal, manifold dims, eval protocol).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union


@dataclass(frozen=True)
class DatasetDescription:
    """Structural metadata for one ``<system>_T<T>`` horizon dataset."""

    horizon_T: int
    autoregressive_K: int
    resolution_l: int
    state_dim: int
    state_order: List[str]
    goal_state: List[float]
    angle_dims: List[int]
    quaternion_dims: Optional[List[int]]
    in_sample: bool


def load_dataset_description(
    path: Union[str, Path],
) -> DatasetDescription:
    """Load a ``DatasetDescription`` from a dataset directory or JSON file."""
    path = Path(path)
    if path.is_dir():
        path = path / "dataset_description.json"
    with open(path) as f:
        info = json.load(f)

    source = info["source_system"]
    horizon = info["horizon_model"]
    eval_sets = info.get("eval_sets", {})

    quaternion_dims = source.get("quaternion_dims")
    return DatasetDescription(
        horizon_T=int(horizon["T_steps"]),
        autoregressive_K=int(horizon["autoregressive_calls_K"]),
        resolution_l=int(horizon["resolution_horizon_l_steps"]),
        state_dim=int(source["state_dim"]),
        state_order=list(source["state_order"]),
        goal_state=list(source["goal_state"]),
        angle_dims=list(source.get("angle_dims") or []),
        quaternion_dims=list(quaternion_dims) if quaternion_dims else None,
        in_sample=bool(eval_sets.get("in_sample", False)),
    )
