# HumanoidStandUpReach Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fresh `humanoid_standup_reach` module set (system, data module, latent-conditional flow matcher, train/inference) that trains a manifold-aware flow-matching model on the `humanoid_get_up_medium` dataset.

**Architecture:** Mirror the mature `quadrotor_3d` latent-conditional code path for a 67-D state on the product manifold ℝ³⁴ × S² × ℝ³⁰. Reuse `BaseFlowMatcher` and the generic, latent-aware `Quadrotor3DUNet`. Bounds and per-dimension normalization come from `dataset_description.json`; success is a binary physical threshold. The old `humanoid` scaffold is left untouched.

**Tech Stack:** PyTorch Lightning, Hydra, the Facebook `flow_matching` library (`Product`/`Sphere`/`Euclidean`, `GeodesicProbPath`, `RiemannianODESolver`), pytest.

**Spec:** `docs/superpowers/specs/2026-06-23-humanoid-standup-reach-design.md`
**Scope note:** This is Plan 1 of 2. Plan 2 (ROA eval configs + `adaptive_v2` conformal/adaptive-sampling pipeline with the intermediate-state pool) follows after the core trains.

## Global Constraints

- **State dim 67**, layout: `joint_angles[0:21] + head_height[21] + extremities[22:34] + torso_vertical[34:37] + com_velocity[37:40] + velocity[40:67]`.
- **Manifold**: ℝ³⁴ × S² × ℝ³⁰. Sphere block = dims **34:37**. FB FM `Product(input_dim=67, manifolds=[(Euclidean(),34,34),(Sphere(),3,3),(Euclidean(),30,30)])`. `Product.dist` → **65** components. Model **`output_dim=67`** in both `use_manifold` modes.
- **Dataset dir**: `/common/users/shared/pracsys/genMoPlan/data_trajectories/humanoid_get_up_medium`.
- **Trajectory files**: `trajectories/sequence_{i}.txt`, comma-separated → `np.loadtxt(path, delimiter=',')`. **On load failure, RAISE — never zero-fill.**
- **Bounds**: `dataset_description.json` → `achieved_bounds["per_dimension_min"]` / `["per_dimension_max"]` (67-length arrays).
- **Success classify** (binary, on RAW states): label `1` iff `head_height (idx 21) >= 1.3` AND `‖com_velocity (idx 37:40)‖₂ <= 0.2`, else `-1`. No separatrix. `radius` arg accepted but ignored.
- **Naming**: module dir `humanoid_standup_reach`, class prefix `HumanoidStandUpReach`.
- **Tests**: live in `tests/`, run with `pytest` using the repo env `/common/home/st1122/Projects/adaptive_roa/env/bin/python -m pytest`. `conftest.py` already puts the repo root on `sys.path`.
- **Git**: working branch is `main`. Commit after each task. No `Co-Authored-By`. End commit messages with `Claude-Session: https://claude.ai/code/session_01NZMXZPhQUpzSKYDqNqypoK`.

---

### Task 1: Sphere manifold verification (independent)

De-risks the whole build by verifying the stock FB FM `Sphere`/`Product` behavior we depend on, beyond the library's bundled tests. Test-only; no production code.

**Files:**
- Test: `tests/test_sphere_manifold.py`

**Interfaces:**
- Consumes: `flow_matching.utils.manifolds.{Product, Euclidean, Sphere}`.
- Produces: nothing for later tasks (documentation/guard rail). Confirms `output_dim=67`, `dist→65`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sphere_manifold.py
"""Independent verification of the FB FM Sphere/Product used by HumanoidStandUpReach.

Confirms the ambient-3 tangent convention (singularity-free), so the flow matcher's
model output_dim is 67 and the distance manifold returns 65 components.
"""
import pytest
import torch
from flow_matching.utils.manifolds import Product, Euclidean, Sphere


def _unit_sphere_block(x):
    x = x.clone()
    n = x[:, 34:37].norm(dim=1, keepdim=True).clamp(min=1e-8)
    x[:, 34:37] = x[:, 34:37] / n
    return x


def _humanoid_product():
    return Product(
        input_dim=67,
        manifolds=[(Euclidean(), 34, 34), (Sphere(), 3, 3), (Euclidean(), 30, 30)],
    )


def test_product_rejects_reduced_sphere_tangent():
    with pytest.raises(ValueError, match="Sphere manifold must have state_dim == tangent_dim"):
        Product(input_dim=67, manifolds=[(Euclidean(), 34, 34), (Sphere(), 3, 2), (Euclidean(), 30, 30)])


def test_product_accepts_ambient_sphere_and_dist_has_65_components():
    m = _humanoid_product()
    a = _unit_sphere_block(torch.randn(8, 67))
    b = _unit_sphere_block(torch.randn(8, 67))
    d = m.dist(a, b)
    assert d.shape == (8, 65)


def test_bare_sphere_dist_is_one_value_per_pair():
    s = Sphere()
    x = torch.randn(8, 3); x = x / x.norm(dim=1, keepdim=True)
    y = torch.randn(8, 3); y = y / y.norm(dim=1, keepdim=True)
    assert s.dist(x, y).shape == (8, 1)


def test_proju_is_orthogonal_to_point():
    s = Sphere()
    x = torch.randn(8, 3); x = x / x.norm(dim=1, keepdim=True)
    v = torch.randn(8, 3)
    pv = s.proju(x, v)
    assert pv.shape == (8, 3)
    assert torch.allclose((x * pv).sum(dim=1), torch.zeros(8), atol=1e-5)


def test_expmap_preserves_unit_norm():
    s = Sphere()
    x = torch.randn(8, 3); x = x / x.norm(dim=1, keepdim=True)
    u = s.proju(x, torch.randn(8, 3) * 0.3)
    y = s.expmap(x, u)
    assert torch.allclose(y.norm(dim=1), torch.ones(8), atol=1e-5)


def test_log_exp_roundtrip_small_tangent():
    s = Sphere()
    x = torch.randn(8, 3); x = x / x.norm(dim=1, keepdim=True)
    u = s.proju(x, torch.randn(8, 3) * 0.1)  # small to avoid antipodal ambiguity
    u_back = s.logmap(x, s.expmap(x, u))
    assert torch.allclose(u, u_back, atol=1e-4)
```

- [ ] **Step 2: Run test to verify it passes (this is a verification task)**

Run: `/common/home/st1122/Projects/adaptive_roa/env/bin/python -m pytest tests/test_sphere_manifold.py -v`
Expected: All 6 tests PASS. If `test_log_exp_roundtrip_small_tangent` fails, reduce the tangent scale (0.1 → 0.05). If any other test fails, STOP — the manifold assumptions in the spec are wrong and must be revisited before continuing.

- [ ] **Step 3: Commit**

```bash
git add tests/test_sphere_manifold.py
git commit -m "test: verify FB FM Sphere/Product ambient-3 tangent for humanoid

Confirms Product requires (Sphere,3,3), dist returns 65 components, proju is
orthogonal, expmap preserves unit norm, and log/exp round-trips. Locks in
model output_dim=67.

Claude-Session: https://claude.ai/code/session_01NZMXZPhQUpzSKYDqNqypoK"
```

---

### Task 2: HumanoidStandUpReachSystem

**Files:**
- Create: `adaptive_roa/systems/humanoid_standup_reach.py`
- Test: `tests/test_humanoid_standup_reach_system.py`

**Interfaces:**
- Consumes: `adaptive_roa.systems.base.{DynamicalSystem, ManifoldComponent}`, `adaptive_roa.utils.env_config.get_shared_data_base`, `dataset_description.json`.
- Produces (used by Tasks 4 & 5):
  - `HumanoidStandUpReachSystem(dataset_dir: str = None)` with attribute `dataset_dir: str`.
  - `normalize_state(state: Tensor[B,67]) -> Tensor[B,67]`, `denormalize_state(...)` (exact inverse), `embed_state_for_model(state) -> state` (identity), `project_to_manifold(state) -> Tensor[B,67]` (unit-norm dims 34:37).
  - `classify_attractor(state, radius=None) -> LongTensor[B]` in `{1,-1}`; `is_in_attractor(state, radius=None) -> BoolTensor[B]`.
  - `get_loss_weights() -> Tensor[67]`, `attractors() -> List[List[float]]`, `state_dim == 67`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_humanoid_standup_reach_system.py
import numpy as np
import torch
import pytest

from adaptive_roa.systems.humanoid_standup_reach import HumanoidStandUpReachSystem

DATASET_DIR = "/common/users/shared/pracsys/genMoPlan/data_trajectories/humanoid_get_up_medium"


@pytest.fixture(scope="module")
def system():
    return HumanoidStandUpReachSystem(dataset_dir=DATASET_DIR)


def test_state_dim_and_manifold_structure(system):
    assert system.state_dim == 67
    comps = system.define_manifold_structure()
    assert sum(c.dim for c in comps) == 67
    sphere = [c for c in comps if c.manifold_type == "Sphere"]
    assert len(sphere) == 1 and sphere[0].dim == 3


def test_bounds_loaded_and_sphere_identity(system):
    assert system._norm_center.shape == (67,)
    assert system._norm_half.shape == (67,)
    # Sphere dims are identity (center 0, half 1) so normalization leaves them unit-norm
    assert torch.allclose(system._norm_center[34:37], torch.zeros(3))
    assert torch.allclose(system._norm_half[34:37], torch.ones(3))


def test_normalize_denormalize_roundtrip(system):
    torch.manual_seed(0)
    raw = torch.randn(16, 67) * 3.0
    raw = system.project_to_manifold(raw)  # make sphere block unit-norm
    back = system.denormalize_state(system.normalize_state(raw))
    assert torch.allclose(back, raw, atol=1e-4)


def test_normalize_keeps_sphere_block_unchanged(system):
    raw = system.project_to_manifold(torch.randn(8, 67))
    norm = system.normalize_state(raw)
    assert torch.allclose(norm[:, 34:37], raw[:, 34:37], atol=1e-6)


def test_classify_attractor_thresholds(system):
    state = torch.zeros(4, 67)
    # row0: success (head=1.3 exactly, com speed 0)
    state[0, 21] = 1.3
    # row1: head too low
    state[1, 21] = 1.29
    # row2: head ok but com speed too high (0.21 along one axis)
    state[2, 21] = 1.5; state[2, 37] = 0.21
    # row3: head ok, com speed exactly 0.2 -> success
    state[3, 21] = 1.5; state[3, 37] = 0.2
    labels = system.classify_attractor(state)
    assert labels.tolist() == [1, -1, -1, 1]
    assert labels.dtype == torch.long


def test_classify_attractor_accepts_numpy_and_1d(system):
    s = np.zeros(67, dtype=np.float32); s[21] = 1.4
    out = system.classify_attractor(s)
    assert int(out.item() if out.dim() == 0 else out[0]) == 1


def test_get_loss_weights_shape(system):
    w = system.get_loss_weights()
    assert w.shape == (67,)
    assert torch.allclose(w[34:37], torch.ones(3))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/common/home/st1122/Projects/adaptive_roa/env/bin/python -m pytest tests/test_humanoid_standup_reach_system.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'adaptive_roa.systems.humanoid_standup_reach'`.

- [ ] **Step 3: Write minimal implementation**

```python
# adaptive_roa/systems/humanoid_standup_reach.py
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
        assert per_min.shape == (67,) and per_max.shape == (67,), "expected 67-length bounds"
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/common/home/st1122/Projects/adaptive_roa/env/bin/python -m pytest tests/test_humanoid_standup_reach_system.py -v`
Expected: all 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add adaptive_roa/systems/humanoid_standup_reach.py tests/test_humanoid_standup_reach_system.py
git commit -m "feat: add HumanoidStandUpReachSystem (R34 x S2 x R30)

JSON per-dimension bounds, per-dim normalization (sphere block identity),
binary head-height/CoM-speed success classify, unit-norm manifold projection.

Claude-Session: https://claude.ai/code/session_01NZMXZPhQUpzSKYDqNqypoK"
```

---

### Task 3: HumanoidStandUpReach endpoint data module

**Files:**
- Create: `adaptive_roa/data/humanoid_standup_reach_endpoint_data.py`
- Test: `tests/test_humanoid_standup_reach_data.py`

**Interfaces:**
- Consumes: trajectory files `sequence_{i}.txt` (comma-delimited, 67 cols), a shuffled-indices file listing those filenames.
- Produces (used by Task 5):
  - `HumanoidStandUpReachEndpointDataset(shuffled_indices_file, trajectories_dir, query_mode="random_intermediate", max_samples=None)` returning `{"start_state": Tensor[67], "end_state": Tensor[67]}` with the sphere block unit-norm.
  - `HumanoidStandUpReachEndpointDataModule(train_indices_file, val_indices_file, test_indices_file, trajectories_dir, query_mode, batch_size, val_batch_size, num_workers)`.
  - `query_mode ∈ {"start","random_intermediate","all_intermediate"}`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_humanoid_standup_reach_data.py
import numpy as np
import torch
import pytest

from adaptive_roa.data.humanoid_standup_reach_endpoint_data import (
    HumanoidStandUpReachEndpointDataset,
)


def _make_traj(path, n_rows, seed):
    rng = np.random.default_rng(seed)
    rows = rng.normal(size=(n_rows, 67)).astype(np.float64)
    # make sphere block non-unit on purpose to test projection
    rows[:, 34:37] *= 5.0
    np.savetxt(path, rows, delimiter=", ", fmt="%.6f")  # comma+space like the dataset
    return rows


@pytest.fixture
def tiny_dataset(tmp_path):
    traj_dir = tmp_path / "trajectories"
    traj_dir.mkdir()
    raws = {}
    for i in range(3):
        raws[f"sequence_{i}.txt"] = _make_traj(traj_dir / f"sequence_{i}.txt", n_rows=5 + i, seed=i)
    idx = tmp_path / "shuffled_indices_0.txt"
    idx.write_text("\n".join(raws.keys()) + "\n")
    return str(idx), str(traj_dir), raws


def test_start_mode_returns_first_and_last_row(tiny_dataset):
    idx, traj_dir, raws = tiny_dataset
    ds = HumanoidStandUpReachEndpointDataset(shuffled_indices_file=idx, trajectories_dir=traj_dir, query_mode="start")
    item = ds[0]
    raw = raws["sequence_0.txt"]
    # end_state == final row (denormalized euclidean dims match raw exactly)
    assert torch.allclose(item["end_state"][:34], torch.tensor(raw[-1, :34], dtype=torch.float32), atol=1e-4)
    # start_state euclidean part == first row
    assert torch.allclose(item["start_state"][:34], torch.tensor(raw[0, :34], dtype=torch.float32), atol=1e-4)


def test_sphere_block_is_unit_norm(tiny_dataset):
    idx, traj_dir, _ = tiny_dataset
    ds = HumanoidStandUpReachEndpointDataset(shuffled_indices_file=idx, trajectories_dir=traj_dir, query_mode="start")
    item = ds[1]
    assert torch.allclose(item["start_state"][34:37].norm(), torch.tensor(1.0), atol=1e-5)
    assert torch.allclose(item["end_state"][34:37].norm(), torch.tensor(1.0), atol=1e-5)


def test_random_intermediate_query_is_non_terminal(tiny_dataset):
    idx, traj_dir, raws = tiny_dataset
    ds = HumanoidStandUpReachEndpointDataset(shuffled_indices_file=idx, trajectories_dir=traj_dir, query_mode="random_intermediate")
    raw = raws["sequence_2.txt"]  # 7 rows
    torch.manual_seed(0)
    queries = [ds[2]["start_state"][:34].numpy() for _ in range(20)]
    # every query must equal some non-terminal row (rows 0..n-2), never the final row
    final = raw[-1, :34]
    for q in queries:
        assert not np.allclose(q, final, atol=1e-4), "query must be non-terminal"
        assert any(np.allclose(q, raw[r, :34], atol=1e-4) for r in range(raw.shape[0] - 1))


def test_missing_file_raises_not_zero_fill(tmp_path):
    traj_dir = tmp_path / "trajectories"; traj_dir.mkdir()
    idx = tmp_path / "idx.txt"; idx.write_text("sequence_999.txt\n")
    ds = HumanoidStandUpReachEndpointDataset(shuffled_indices_file=str(idx), trajectories_dir=str(traj_dir), query_mode="start")
    with pytest.raises(Exception):
        _ = ds[0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/common/home/st1122/Projects/adaptive_roa/env/bin/python -m pytest tests/test_humanoid_standup_reach_data.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'adaptive_roa.data.humanoid_standup_reach_endpoint_data'`.

- [ ] **Step 3: Write minimal implementation**

```python
# adaptive_roa/data/humanoid_standup_reach_endpoint_data.py
"""Endpoint data for HumanoidStandUpReach: (query_state -> final_state) pairs.

query_mode controls which trajectory row is the query:
  - "start": row 0 (Quad3D parity)
  - "random_intermediate": a uniformly sampled non-terminal row (default; FPS-style coverage)
  - "all_intermediate": every non-terminal row expanded into its own pair
The target is always the trajectory's final row. Sphere block (dims 34:37) is unit-normalized.
"""
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as pl

SPHERE_START, SPHERE_END = 34, 37


def _load_trajectory(path: Path) -> np.ndarray:
    # Comma delimiter handles both "," and ", "; failures raise (never zero-fill).
    traj = np.loadtxt(path, delimiter=",")
    if traj.ndim == 1:
        traj = traj[None, :]
    if traj.shape[1] != 67:
        raise ValueError(f"{path}: expected 67 columns, got {traj.shape[1]}")
    return traj.astype(np.float32)


def _unit_sphere(vec: np.ndarray) -> np.ndarray:
    out = vec.copy()
    block = out[SPHERE_START:SPHERE_END]
    n = np.linalg.norm(block)
    out[SPHERE_START:SPHERE_END] = block / n if n > 1e-8 else block
    return out


class HumanoidStandUpReachEndpointDataset(Dataset):
    def __init__(self, shuffled_indices_file: str, trajectories_dir: str,
                 query_mode: str = "random_intermediate", max_samples: int = None):
        assert query_mode in ("start", "random_intermediate", "all_intermediate")
        self.query_mode = query_mode
        self.trajectories_dir = Path(trajectories_dir)
        with open(shuffled_indices_file) as f:
            filenames = [ln.strip() for ln in f if ln.strip()]
        if max_samples is not None:
            filenames = filenames[:max_samples]
        if query_mode == "all_intermediate":
            # Expand to (filename, row_idx) pairs. Reads each file once up front.
            self._index = []
            for fn in filenames:
                traj = _load_trajectory(self.trajectories_dir / fn)
                for r in range(traj.shape[0] - 1):  # non-terminal rows only
                    self._index.append((fn, r))
        else:
            self._index = [(fn, None) for fn in filenames]
        print(f"HumanoidStandUpReach dataset: {len(self._index)} samples (query_mode={query_mode})")

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        fname, fixed_row = self._index[idx]
        traj = _load_trajectory(self.trajectories_dir / fname)
        n = traj.shape[0]
        end = traj[-1]
        if self.query_mode == "start":
            q = 0
        elif self.query_mode == "all_intermediate":
            q = fixed_row
        else:  # random_intermediate; uses global torch RNG (varies per epoch, seedable)
            q = int(torch.randint(0, max(n - 1, 1), (1,)).item())
        start = traj[q]
        return {
            "start_state": torch.from_numpy(_unit_sphere(start)),
            "end_state": torch.from_numpy(_unit_sphere(end)),
        }


class HumanoidStandUpReachEndpointDataModule(pl.LightningDataModule):
    def __init__(self, train_indices_file: str, val_indices_file: str, test_indices_file: str,
                 trajectories_dir: str, query_mode: str = "random_intermediate",
                 batch_size: int = 256, val_batch_size: Optional[int] = None,
                 num_workers: int = 4, pin_memory: bool = True,
                 max_train_samples: int = None, max_val_samples: int = None):
        super().__init__()
        self.train_indices_file = train_indices_file
        self.val_indices_file = val_indices_file
        self.test_indices_file = test_indices_file
        self.trajectories_dir = trajectories_dir
        self.query_mode = query_mode
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size or batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.max_train_samples = max_train_samples
        self.max_val_samples = max_val_samples
        self.state_dim = 67
        self.embedded_dim = 67

    def setup(self, stage: Optional[str] = None):
        if stage in ("fit", None):
            self.train_dataset = HumanoidStandUpReachEndpointDataset(
                self.train_indices_file, self.trajectories_dir, self.query_mode, self.max_train_samples)
            self.val_dataset = HumanoidStandUpReachEndpointDataset(
                self.val_indices_file, self.trajectories_dir, self.query_mode, self.max_val_samples)
        if stage in ("test", None):
            self.test_dataset = HumanoidStandUpReachEndpointDataset(
                self.test_indices_file, self.trajectories_dir, self.query_mode)

    def _loader(self, ds, bs, shuffle):
        return DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=self.num_workers,
                          pin_memory=self.pin_memory, persistent_workers=self.num_workers > 0)

    def train_dataloader(self):
        return self._loader(self.train_dataset, self.batch_size, True)

    def val_dataloader(self):
        return self._loader(self.val_dataset, self.val_batch_size, False)

    def test_dataloader(self):
        return self._loader(self.test_dataset, self.val_batch_size, False)

    def predict_dataloader(self):
        return self.test_dataloader()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/common/home/st1122/Projects/adaptive_roa/env/bin/python -m pytest tests/test_humanoid_standup_reach_data.py -v`
Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add adaptive_roa/data/humanoid_standup_reach_endpoint_data.py tests/test_humanoid_standup_reach_data.py
git commit -m "feat: add HumanoidStandUpReach endpoint data module

Shuffled-indices loader with comma delimiter, query_mode (start/random_intermediate/
all_intermediate), unit-norm sphere block, raises on load failure (no zero-fill).

Claude-Session: https://claude.ai/code/session_01NZMXZPhQUpzSKYDqNqypoK"
```

---

### Task 4: HumanoidStandUpReach latent-conditional flow matcher

**Files:**
- Create: `adaptive_roa/flow_matching/humanoid_standup_reach/__init__.py` (empty)
- Create: `adaptive_roa/flow_matching/humanoid_standup_reach/latent_conditional/__init__.py` (empty)
- Create: `adaptive_roa/flow_matching/humanoid_standup_reach/latent_conditional/flow_matcher.py`
- Test: `tests/test_humanoid_standup_reach_flow_matcher.py`

**Interfaces:**
- Consumes: `adaptive_roa.flow_matching.base.flow_matcher.BaseFlowMatcher`, `HumanoidStandUpReachSystem` (Task 2), `adaptive_roa.model.quadrotor3d_unet.Quadrotor3DUNet`, `flow_matching.utils.manifolds.{Product,Euclidean,Sphere}`.
- Produces (used by Task 5):
  - `HumanoidStandUpReachLatentConditionalFlowMatcher(system, model, optimizer, scheduler, model_config=None, latent_dim=8, mae_val_frequency=10, use_loss_weights=False, use_manifold=True, clamp_noise=True, zero_latent=False, val_error_log_file=None, noise_scale=1.0)`.
  - Overrides: `_create_manifold`, `_create_distance_manifold`, `sample_noisy_input`, `_get_start_states`, `_get_end_states`, `_get_dimension_name`, `normalize_state`, `denormalize_state`, `embed_state_for_model`, `get_manifold_component_names`, `predict_endpoint`. Inherits `forward`, `compute_flow_loss`, `predict_endpoints_batch` (define passthrough), `training_step`, `validation_step`, `configure_optimizers` from base.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_humanoid_standup_reach_flow_matcher.py
import torch
import pytest

from adaptive_roa.systems.humanoid_standup_reach import HumanoidStandUpReachSystem
from adaptive_roa.model.quadrotor3d_unet import Quadrotor3DUNet
from adaptive_roa.flow_matching.humanoid_standup_reach.latent_conditional.flow_matcher import (
    HumanoidStandUpReachLatentConditionalFlowMatcher,
)

DATASET_DIR = "/common/users/shared/pracsys/genMoPlan/data_trajectories/humanoid_get_up_medium"


def _build(use_manifold=True):
    system = HumanoidStandUpReachSystem(dataset_dir=DATASET_DIR)
    model = Quadrotor3DUNet(embedded_dim=67, latent_dim=8, condition_dim=67,
                            time_emb_dim=64, hidden_dims=[128, 128], output_dim=67)
    fm = HumanoidStandUpReachLatentConditionalFlowMatcher(
        system=system, model=model, optimizer=None, scheduler=None,
        latent_dim=8, use_manifold=use_manifold)
    return system, fm


def test_manifold_dist_has_65_components():
    _, fm = _build(use_manifold=True)
    a = torch.randn(4, 67); a = fm.system.project_to_manifold(a)
    b = torch.randn(4, 67); b = fm.system.project_to_manifold(b)
    assert fm.manifold.dist(a, b).shape == (4, 65)


def test_distance_manifold_is_spherical_even_when_euclidean_training():
    _, fm = _build(use_manifold=False)
    a = fm.system.project_to_manifold(torch.randn(4, 67))
    b = fm.system.project_to_manifold(torch.randn(4, 67))
    # distance manifold must still be the S²-aware product (65 components)
    assert fm.distance_manifold.dist(a, b).shape == (4, 65)


def test_component_names_count():
    _, fm = _build()
    assert len(fm.get_manifold_component_names()) == 65


def test_sample_noisy_input_on_manifold():
    _, fm = _build()
    noise = fm.sample_noisy_input(8, torch.device("cpu"))
    assert noise.shape == (8, 67)
    assert torch.allclose(noise[:, 34:37].norm(dim=1), torch.ones(8), atol=1e-5)


def test_predict_endpoint_shape_and_unit_sphere():
    _, fm = _build()
    start = fm.system.project_to_manifold(torch.randn(2, 67))
    pred = fm.predict_endpoint(start, num_steps=5)
    assert pred.shape == (2, 67)
    assert torch.allclose(pred[:, 34:37].norm(dim=1), torch.ones(2), atol=1e-4)


def test_compute_flow_loss_runs():
    _, fm = _build()
    batch = {
        "start_state": fm.system.project_to_manifold(torch.randn(4, 67)),
        "end_state": fm.system.project_to_manifold(torch.randn(4, 67)),
    }
    loss = fm.compute_flow_loss(batch)
    assert loss.ndim == 0 and torch.isfinite(loss)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/common/home/st1122/Projects/adaptive_roa/env/bin/python -m pytest tests/test_humanoid_standup_reach_flow_matcher.py -v`
Expected: FAIL — `ModuleNotFoundError` for the `humanoid_standup_reach` flow matcher package.

- [ ] **Step 3: Write minimal implementation**

First create the two empty package files:

```python
# adaptive_roa/flow_matching/humanoid_standup_reach/__init__.py
```
```python
# adaptive_roa/flow_matching/humanoid_standup_reach/latent_conditional/__init__.py
```

Then the flow matcher:

```python
# adaptive_roa/flow_matching/humanoid_standup_reach/latent_conditional/flow_matcher.py
"""HumanoidStandUpReach Latent Conditional Flow Matching (Facebook FM).

Manifold: ℝ³⁴ × S² × ℝ³⁰ (67-D state). The sphere block (dims 34:37) uses the FB FM
ambient-3 tangent (singularity-free), so model output_dim = 67 in both manifold modes.
"""
from typing import Dict, Optional

import torch
import torch.nn as nn

from flow_matching.utils.manifolds import Product, Euclidean, Sphere

from adaptive_roa.flow_matching.base.flow_matcher import BaseFlowMatcher
from adaptive_roa.systems.base import DynamicalSystem

SPHERE_START, SPHERE_END = 34, 37
_DIM_NAMES = (
    [f"joint_{i}" for i in range(21)] + ["head_height"]
    + [f"extremity_{i}" for i in range(12)]
    + ["torso_vx", "torso_vy", "torso_vz"]
    + ["com_vx", "com_vy", "com_vz"]
    + [f"vel_{i}" for i in range(27)]
)


class HumanoidStandUpReachLatentConditionalFlowMatcher(BaseFlowMatcher):
    def __init__(self, system: DynamicalSystem, model: nn.Module, optimizer, scheduler,
                 model_config: Optional[dict] = None, latent_dim: int = 8,
                 mae_val_frequency: int = 10, use_loss_weights: bool = False,
                 use_manifold: bool = True, clamp_noise: bool = True,
                 zero_latent: bool = False, val_error_log_file: Optional[str] = None,
                 noise_scale: float = 1.0):
        # Must be set before super().__init__ (it calls _create_manifold()).
        self.use_manifold = use_manifold
        super().__init__(system, model, optimizer, scheduler, model_config, latent_dim,
                         mae_val_frequency, use_loss_weights, clamp_noise, zero_latent,
                         val_error_log_file, noise_scale)
        print(f"✅ HumanoidStandUpReach LCFM | manifold={use_manifold} | latent={latent_dim}")

    # ---- manifolds ---------------------------------------------------------
    def _create_manifold(self):
        if self.use_manifold:
            return Product(input_dim=67, manifolds=[
                (Euclidean(), 34, 34), (Sphere(), 3, 3), (Euclidean(), 30, 30)])
        return Euclidean()

    def _create_distance_manifold(self):
        # Always the S²-aware product (65 distance components), regardless of use_manifold.
        return Product(input_dim=67, manifolds=[
            (Euclidean(), 34, 34), (Sphere(), 3, 3), (Euclidean(), 30, 30)])

    def get_manifold_component_names(self) -> list:
        # Euclidean(34) -> 34 per-dim + Sphere -> 1 geodesic + Euclidean(30) -> 30 = 65
        names = [f"e1_{i}" for i in range(34)] + ["torso_vertical_geo"] + [f"e2_{i}" for i in range(30)]
        return names

    # ---- noise -------------------------------------------------------------
    def sample_noisy_input(self, batch_size: int, device: torch.device) -> torch.Tensor:
        euclid1 = torch.randn(batch_size, 34, device=device)
        sphere = torch.randn(batch_size, 3, device=device)
        euclid2 = torch.randn(batch_size, 30, device=device)
        if self.noise_scale != 1.0:
            euclid1 *= self.noise_scale; sphere *= self.noise_scale; euclid2 *= self.noise_scale
        if self.clamp_noise:
            euclid1 = torch.clamp(euclid1, -1.0, 1.0)
            euclid2 = torch.clamp(euclid2, -1.0, 1.0)
        sphere = sphere / sphere.norm(dim=1, keepdim=True).clamp(min=1e-8)
        noisy = torch.cat([euclid1, sphere, euclid2], dim=1)
        return self.manifold.projx(noisy)

    # ---- batch accessors / delegation -------------------------------------
    def _get_start_states(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        return batch["start_state"]

    def _get_end_states(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        return batch["end_state"]

    def _get_dimension_name(self, dim_idx: int) -> str:
        return _DIM_NAMES[dim_idx] if 0 <= dim_idx < len(_DIM_NAMES) else f"dim_{dim_idx}"

    def normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        return self.system.normalize_state(state)

    def denormalize_state(self, normalized_state: torch.Tensor) -> torch.Tensor:
        return self.system.denormalize_state(normalized_state)

    def embed_state_for_model(self, normalized_state: torch.Tensor) -> torch.Tensor:
        return self.system.embed_state_for_model(normalized_state)

    # ---- prediction --------------------------------------------------------
    def predict_endpoint(self, start_states: torch.Tensor, num_steps: int = 100,
                         latent: Optional[torch.Tensor] = None,
                         method: str = "euler_riemannian") -> torch.Tensor:
        endpoints = super().predict_endpoint(start_states, num_steps, latent, method)
        return self.system.project_to_manifold(endpoints)

    def predict_endpoints_batch(self, start_states: torch.Tensor, num_steps: int = 100,
                                num_samples: int = 1) -> torch.Tensor:
        if num_samples == 1:
            return self.predict_endpoint(start_states, num_steps)
        return torch.cat([self.predict_endpoint(start_states, num_steps, latent=None)
                          for _ in range(num_samples)], dim=0)
```

Note on `method`: the base default is `"euler"`; we default to `"euler_riemannian"` here to match Quad3D's manifold integration. If `test_predict_endpoint_shape_and_unit_sphere` errors on an unknown method when `use_manifold=False`, fall back to `method="euler"` for the Euclidean path (mirror Quad3D, which keeps `euler_riemannian`; the RiemannianODESolver supports both).

- [ ] **Step 4: Run test to verify it passes**

Run: `/common/home/st1122/Projects/adaptive_roa/env/bin/python -m pytest tests/test_humanoid_standup_reach_flow_matcher.py -v`
Expected: all 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add adaptive_roa/flow_matching/humanoid_standup_reach/ tests/test_humanoid_standup_reach_flow_matcher.py
git commit -m "feat: add HumanoidStandUpReach latent-conditional flow matcher

Product manifold R34 x S2(ambient-3) x R30, use_manifold toggle with always-on
S2 distance manifold (65 comps), sphere-aware noise, output_dim 67, unit-norm
endpoint projection. Reuses Quadrotor3DUNet.

Claude-Session: https://claude.ai/code/session_01NZMXZPhQUpzSKYDqNqypoK"
```

---

### Task 5: Train script, inference, and configs (integration smoke test)

**Files:**
- Create: `adaptive_roa/flow_matching/humanoid_standup_reach/latent_conditional/train.py`
- Create: `adaptive_roa/flow_matching/humanoid_standup_reach/latent_conditional/inference.py`
- Create: `configs/system/humanoid_standup_reach.yaml`
- Create: `configs/model/humanoid_standup_reach_unet.yaml`
- Create: `configs/train_humanoid_standup_reach.yaml`
- Test: `tests/test_humanoid_standup_reach_train_config.py`

**Interfaces:**
- Consumes: Tasks 2–4 plus `configs/system/quadrotor3d.yaml` (pattern), `configs/train_quadrotor3d.yaml` (pattern), `adaptive_roa.utils.env_config` resolvers.
- Produces: a runnable `train.py` (`@hydra.main(config_name="train_humanoid_standup_reach")`) and Hydra-composable configs that instantiate system+model+flow matcher with `output_dim=67`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_humanoid_standup_reach_train_config.py
import torch
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf
import hydra
from pathlib import Path

from adaptive_roa.utils.env_config import get_net_id, get_exp_dir, get_shared_data_base

CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "configs")


def _register_resolvers():
    for name, fn in [("net_id", get_net_id), ("exp_dir", get_exp_dir), ("shared_data_base", get_shared_data_base)]:
        if not OmegaConf.has_resolver(name):
            OmegaConf.register_new_resolver(name, (lambda f: (lambda default="": f() or default))(fn))


def test_train_config_composes_and_instantiates():
    _register_resolvers()
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(config_name="train_humanoid_standup_reach")
    assert cfg.model.output_dim == 67
    assert cfg.model.embedded_dim == 67 and cfg.model.condition_dim == 67
    system = hydra.utils.instantiate(cfg.system)
    assert system.state_dim == 67
    model = hydra.utils.instantiate(cfg.model)
    fm = hydra.utils.instantiate(
        cfg.flow_matcher, system=system, model=model,
        optimizer=cfg.optimizer, scheduler=cfg.scheduler,
        model_config=OmegaConf.to_container(cfg.model, resolve=True),
        latent_dim=cfg.flow_matching.latent_dim,
        mae_val_frequency=cfg.flow_matching.mae_val_frequency,
        use_loss_weights=cfg.flow_matching.get("use_loss_weights", False),
        use_manifold=cfg.flow_matching.get("use_manifold", True),
        clamp_noise=cfg.flow_matching.get("clamp_noise", True),
        zero_latent=cfg.flow_matching.get("zero_latent", False),
        noise_scale=cfg.flow_matching.get("noise_scale", 1.0),
        _recursive_=False,
    )
    batch = {
        "start_state": system.project_to_manifold(torch.randn(4, 67)),
        "end_state": system.project_to_manifold(torch.randn(4, 67)),
    }
    loss = fm.compute_flow_loss(batch)
    assert torch.isfinite(loss)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/common/home/st1122/Projects/adaptive_roa/env/bin/python -m pytest tests/test_humanoid_standup_reach_train_config.py -v`
Expected: FAIL — config `train_humanoid_standup_reach` not found.

- [ ] **Step 3: Write the configs and scripts**

```yaml
# configs/system/humanoid_standup_reach.yaml
# @package _global_.system
# Humanoid get-up: ℝ³⁴ × S² × ℝ³⁰ (67-D). Bounds from dataset_description.json.
_target_: adaptive_roa.systems.humanoid_standup_reach.HumanoidStandUpReachSystem
dataset_dir: ${shared_data_base:/common/users/shared/pracsys/genMoPlan/data_trajectories}/humanoid_get_up_medium
```

```yaml
# configs/model/humanoid_standup_reach_unet.yaml
# Generic latent-conditional UNet (reused from quadrotor3d) sized for 67-D humanoid.
_target_: adaptive_roa.model.quadrotor3d_unet.Quadrotor3DUNet
embedded_dim: 67
latent_dim: ${latent_dim}
condition_dim: 67
time_emb_dim: 64
hidden_dims: [512, 1024, 1024, 512]
output_dim: 67          # ambient tangent (34 Euclid + 3 Sphere-ambient + 30 Euclid)
use_input_embeddings: false
input_emb_dim: 128
```

```yaml
# configs/train_humanoid_standup_reach.yaml
# Humanoid StandUpReach Latent Conditional Flow Matching (Facebook FM)
# Manifold ℝ³⁴ × S² × ℝ³⁰, GeodesicProbPath, RiemannianODESolver. State 67-D, tangent 67-D.

defaults:
  - system: humanoid_standup_reach
  - _self_

net_id: ${net_id:}
user_base: ${exp_dir:}
shared_data: ${shared_data_base:/common/users/shared/pracsys/genMoPlan/data_trajectories}

name: humanoid_standup_reach_latent_conditional_fm
seed: 42
batch_size: 256
val_batch_size: 1024
base_lr: 1e-4
num_workers: 4
latent_dim: 8

flow_matcher:
  _target_: adaptive_roa.flow_matching.humanoid_standup_reach.latent_conditional.flow_matcher.HumanoidStandUpReachLatentConditionalFlowMatcher

model:
  _target_: adaptive_roa.model.quadrotor3d_unet.Quadrotor3DUNet
  embedded_dim: 67
  latent_dim: ${latent_dim}
  condition_dim: 67
  time_emb_dim: 64
  hidden_dims: [512, 1024, 1024, 512]
  output_dim: 67
  use_input_embeddings: false
  input_emb_dim: 128

data:
  _target_: adaptive_roa.data.humanoid_standup_reach_endpoint_data.HumanoidStandUpReachEndpointDataModule
  train_indices_file: ${shared_data}/humanoid_get_up_medium/train_test_splits/shuffled_indices_0.txt
  val_indices_file: ${shared_data}/humanoid_get_up_medium/train_test_splits/shuffled_indices_1.txt
  test_indices_file: ${shared_data}/humanoid_get_up_medium/train_test_splits/shuffled_indices_2.txt
  trajectories_dir: ${shared_data}/humanoid_get_up_medium/trajectories
  query_mode: random_intermediate
  batch_size: ${batch_size}
  val_batch_size: ${val_batch_size}
  num_workers: ${num_workers}

optimizer:
  _target_: torch.optim.AdamW
  lr: ${base_lr}
  weight_decay: 1e-5
  betas: [0.9, 0.999]

scheduler:
  _target_: torch.optim.lr_scheduler.ReduceLROnPlateau
  mode: min
  factor: 0.5
  patience: 10
  min_lr: 1e-6

trainer:
  _target_: lightning.pytorch.Trainer
  max_epochs: 500
  accelerator: gpu
  devices: [0]
  precision: 32
  gradient_clip_val: 1.0
  log_every_n_steps: 10
  check_val_every_n_epoch: 1
  enable_progress_bar: true
  enable_model_summary: true
  logger:
    _target_: lightning.pytorch.loggers.TensorBoardLogger
    save_dir: "${hydra:runtime.output_dir}"
    name: ""
    version: null
  callbacks:
    - _target_: lightning.pytorch.callbacks.ModelCheckpoint
      dirpath: "${hydra:runtime.output_dir}/version_0/checkpoints"
      monitor: val_loss
      mode: min
      save_top_k: 3
      save_last: true
      filename: "epoch{epoch:02d}-val_loss{val_loss:.4f}"
      auto_insert_metric_name: false
    - _target_: lightning.pytorch.callbacks.EarlyStopping
      monitor: val_loss
      mode: min
      patience: 50
      verbose: true

flow_matching:
  latent_dim: ${latent_dim}
  num_integration_steps: 100
  mae_val_frequency: 10
  use_loss_weights: false
  use_manifold: true
  clamp_noise: true
  zero_latent: true
  noise_scale: 1.0

hydra:
  run:
    dir: outputs/${name}/${now:%Y-%m-%d_%H-%M-%S}
```

```python
# adaptive_roa/flow_matching/humanoid_standup_reach/latent_conditional/train.py
#!/usr/bin/env python3
"""Train HumanoidStandUpReach Latent Conditional Flow Matching (Facebook FM).

Manifold: ℝ³⁴ × S² × ℝ³⁰ (67-D state, 67-D ambient tangent).
"""
import hydra
from omegaconf import DictConfig, OmegaConf
import lightning.pytorch as pl

from adaptive_roa.utils.env_config import get_net_id, get_exp_dir, get_shared_data_base

if not OmegaConf.has_resolver("net_id"):
    OmegaConf.register_new_resolver("net_id", lambda default="": get_net_id() or default)
if not OmegaConf.has_resolver("exp_dir"):
    OmegaConf.register_new_resolver("exp_dir", lambda default="": get_exp_dir() or default)
if not OmegaConf.has_resolver("shared_data_base"):
    OmegaConf.register_new_resolver("shared_data_base", lambda default="": get_shared_data_base() or default)


@hydra.main(version_base=None, config_path="../../../../configs", config_name="train_humanoid_standup_reach")
def main(cfg: DictConfig):
    print("=" * 80)
    print("🚀 HumanoidStandUpReach Latent Conditional Flow Matching (Facebook FM)")
    print(f"   Manifold: ℝ³⁴ × S² × ℝ³⁰ (67-D) | seed={cfg.seed}")
    print("=" * 80)
    pl.seed_everything(cfg.seed)

    system = hydra.utils.instantiate(cfg.system)
    data_module = hydra.utils.instantiate(cfg.data)
    model = hydra.utils.instantiate(cfg.model)

    flow_matcher = hydra.utils.instantiate(
        cfg.flow_matcher,
        system=system,
        model=model,
        optimizer=cfg.optimizer,
        scheduler=cfg.scheduler,
        model_config=OmegaConf.to_container(cfg.model, resolve=True),
        latent_dim=cfg.flow_matching.latent_dim,
        mae_val_frequency=cfg.flow_matching.mae_val_frequency,
        use_loss_weights=cfg.flow_matching.get("use_loss_weights", False),
        use_manifold=cfg.flow_matching.get("use_manifold", True),
        clamp_noise=cfg.flow_matching.get("clamp_noise", True),
        zero_latent=cfg.flow_matching.get("zero_latent", False),
        noise_scale=cfg.flow_matching.get("noise_scale", 1.0),
        _recursive_=False,
    )

    trainer = hydra.utils.instantiate(cfg.trainer)
    trainer.fit(flow_matcher, data_module)
    print("✅ Training complete")


if __name__ == "__main__":
    main()
```

```python
# adaptive_roa/flow_matching/humanoid_standup_reach/latent_conditional/inference.py
"""Inference helpers for HumanoidStandUpReach LCFM."""
import torch
from adaptive_roa.flow_matching.humanoid_standup_reach.latent_conditional.flow_matcher import (
    HumanoidStandUpReachLatentConditionalFlowMatcher,
)


def load_model(checkpoint_path: str, dataset_dir: str = None):
    """Load a trained flow matcher from a Lightning checkpoint."""
    return HumanoidStandUpReachLatentConditionalFlowMatcher.load_from_checkpoint(checkpoint_path)


@torch.no_grad()
def predict_endpoints(model, start_states: torch.Tensor, num_steps: int = 100) -> torch.Tensor:
    model.eval()
    return model.predict_endpoint(start_states, num_steps=num_steps)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/common/home/st1122/Projects/adaptive_roa/env/bin/python -m pytest tests/test_humanoid_standup_reach_train_config.py -v`
Expected: PASS. (If `Quadrotor3DUNet` rejects any kwarg in `humanoid_standup_reach_unet.yaml`, align the model block to the exact `Quadrotor3DUNet.__init__` signature — see `configs/train_quadrotor3d.yaml`'s `model:` block.)

- [ ] **Step 5: Run the full suite**

Run: `/common/home/st1122/Projects/adaptive_roa/env/bin/python -m pytest tests/test_humanoid_standup_reach_system.py tests/test_humanoid_standup_reach_data.py tests/test_humanoid_standup_reach_flow_matcher.py tests/test_humanoid_standup_reach_train_config.py tests/test_sphere_manifold.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add adaptive_roa/flow_matching/humanoid_standup_reach/latent_conditional/train.py \
        adaptive_roa/flow_matching/humanoid_standup_reach/latent_conditional/inference.py \
        configs/system/humanoid_standup_reach.yaml \
        configs/model/humanoid_standup_reach_unet.yaml \
        configs/train_humanoid_standup_reach.yaml \
        tests/test_humanoid_standup_reach_train_config.py
git commit -m "feat: add HumanoidStandUpReach train/inference scripts and configs

Hydra train config (output_dim 67, query_mode random_intermediate), system/model
configs, inference helpers, and an integration smoke test that composes the config
and runs compute_flow_loss end-to-end.

Claude-Session: https://claude.ai/code/session_01NZMXZPhQUpzSKYDqNqypoK"
```

---

## Optional follow-up (not a task): real-data training smoke

After Task 5, optionally confirm end-to-end on real data with a tiny run (1 epoch, capped samples):

```bash
/common/home/st1122/Projects/adaptive_roa/env/bin/python \
  adaptive_roa/flow_matching/humanoid_standup_reach/latent_conditional/train.py \
  trainer.max_epochs=1 trainer.accelerator=cpu trainer.devices=1 \
  data.num_workers=0 data.max_train_samples=64 data.max_val_samples=32 batch_size=16
```
Expected: trains one epoch without error, writes a checkpoint under `outputs/humanoid_standup_reach_latent_conditional_fm/`.

## Plan 2 preview (separate plan)
ROA eval config (`evaluate_roa.py` `system.module`/`system.class` routing) + `adaptive_v2` wiring: `configs/adaptive_v2/system/humanoid_standup_reach.yaml`, `configs/adaptive_v2/model/system_dims/humanoid_standup_reach.yaml` (`output_dim: 67`, `unet_target: …quadrotor3d_unet.Quadrotor3DUNet`), FPS-primary `cal_set_file`/`test_set_file`, and the intermediate-state pool extension to `TrajectoryDataSource`/`AdaptiveDatasetBuilder`.
