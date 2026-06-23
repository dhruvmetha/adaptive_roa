"""HumanoidStandUpReach system: ℝ³⁴ × S² × ℝ³⁰ (67-D get-up state)."""
import json
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import torch

from adaptive_roa.systems.base import DynamicalSystem, ManifoldComponent
from adaptive_roa.utils.env_config import get_shared_data_base

HEAD_HEIGHT_IDX = 21
COM_VEL_START = 37
COM_VEL_END = 40
SPHERE_START = 34
SPHERE_END = 37
SUCCESS_HEAD_HEIGHT = 1.3
SUCCESS_COM_SPEED = 0.2


class HumanoidStandUpReachSystem(DynamicalSystem):
    """67-D humanoid get-up system. Manifold ℝ³⁴ × S² × ℝ³⁰; sphere block = dims 34:37."""

    def __init__(self, dataset_dir: str = None):
        if dataset_dir is None:
            dataset_dir = f"{get_shared_data_base()}/humanoid_get_up_medium"
        dataset_dir = Path(dataset_dir)
        self.dataset_dir = str(dataset_dir)
        json_path = dataset_dir / "dataset_description.json"
        if not json_path.exists():
            raise FileNotFoundError(f"dataset_description.json not found at {json_path}")
        self._load_bounds_from_json(json_path)
        super().__init__()
        self.name = "humanoid_standup_reach"

    def _load_bounds_from_json(self, json_path: Path):
        with open(json_path) as f:
            info = json.load(f)
        achieved = info["achieved_bounds"]
        per_min = np.asarray(achieved["per_dimension_min"], dtype=np.float32)
        per_max = np.asarray(achieved["per_dimension_max"], dtype=np.float32)
        if per_min.shape != (67,) or per_max.shape != (67,):
            raise ValueError("expected 67-length bounds")
        center = (per_max + per_min) / 2.0
        half = (per_max - per_min) / 2.0
        half[half < 1e-6] = 1.0  # guard degenerate dims
        # Sphere block (34:37) is left identity so it stays unit-norm after normalization
        center[SPHERE_START:SPHERE_END] = 0.0
        half[SPHERE_START:SPHERE_END] = 1.0
        self._per_min = per_min
        self._per_max = per_max
        self._norm_center = torch.from_numpy(center)
        self._norm_half = torch.from_numpy(half)
        self.dataset_info = info
        self.achieved_bounds = achieved
        print(f"HumanoidStandUpReach bounds loaded from {json_path}")

    # ---- manifold / bounds -------------------------------------------------
    def define_manifold_structure(self) -> List[ManifoldComponent]:
        comps: List[ManifoldComponent] = [ManifoldComponent("Real", 1, f"e_{i}") for i in range(34)]
        comps.append(ManifoldComponent("Sphere", 3, "torso_vertical"))
        comps += [ManifoldComponent("Real", 1, f"e_{i}") for i in range(37, 67)]
        return comps

    def define_state_bounds(self) -> Dict[str, Tuple[float, float]]:
        bounds: Dict[str, Tuple[float, float]] = {}
        for i in range(34):
            bounds[f"e_{i}"] = (float(self._per_min[i]), float(self._per_max[i]))
        bounds["torso_vertical"] = (-1.0, 1.0)
        for i in range(37, 67):
            bounds[f"e_{i}"] = (float(self._per_min[i]), float(self._per_max[i]))
        return bounds

    # ---- normalization / embedding ----------------------------------------
    def normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        center = self._norm_center.to(state.device, state.dtype)
        half = self._norm_half.to(state.device, state.dtype)
        return (state - center) / half

    def denormalize_state(self, normalized_state: torch.Tensor) -> torch.Tensor:
        center = self._norm_center.to(normalized_state.device, normalized_state.dtype)
        half = self._norm_half.to(normalized_state.device, normalized_state.dtype)
        return normalized_state * half + center

    def embed_state_for_model(self, normalized_state: torch.Tensor) -> torch.Tensor:
        return normalized_state  # sphere already continuous in ℝ³

    def project_to_manifold(self, state: torch.Tensor) -> torch.Tensor:
        out = state.clone()
        sph = out[:, SPHERE_START:SPHERE_END]
        out[:, SPHERE_START:SPHERE_END] = sph / sph.norm(dim=1, keepdim=True).clamp(min=1e-8)
        return out

    # ---- attractor / classification ---------------------------------------
    def is_in_attractor(self, state, radius: float = None):
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()
        if state.dim() == 1:
            state = state.unsqueeze(0)
        head = state[:, HEAD_HEIGHT_IDX]
        com_speed = state[:, COM_VEL_START:COM_VEL_END].norm(dim=1)
        result = (head >= SUCCESS_HEAD_HEIGHT) & (com_speed <= SUCCESS_COM_SPEED)
        return result

    def classify_attractor(self, state: torch.Tensor, radius: float = None) -> torch.Tensor:
        in_attr = self.is_in_attractor(state, radius=radius)
        return torch.where(in_attr,
                           torch.ones_like(in_attr, dtype=torch.long),
                           -torch.ones_like(in_attr, dtype=torch.long))

    def attractors(self) -> List[List[float]]:
        a = [0.0] * 67
        a[HEAD_HEIGHT_IDX] = 1.4   # above success threshold (viz only)
        a[34], a[35], a[36] = 0.0, 0.0, 1.0  # torso vertical up
        return [a]

    def get_loss_weights(self) -> torch.Tensor:
        # Per-dim tangent weights: half-range for Euclidean dims, 1.0 for sphere block.
        return self._norm_half.clone()

    def __repr__(self) -> str:
        return "HumanoidStandUpReachSystem(ℝ³⁴ × S² × ℝ³⁰, 67-D)"
