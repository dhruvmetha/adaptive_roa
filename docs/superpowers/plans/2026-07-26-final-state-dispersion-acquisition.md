# Final-State Dispersion Acquisition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `dispersion` acquisition strategy that scores candidate uncertainty from the geometry of the predicted final-state cloud instead of from label counts.

**Architecture:** A new strategy samples K raw endpoints per candidate, scores each candidate by the mean pairwise distance among its endpoints under a range-normalized circular-aware metric, and selects by one of three configurable rules. The success criteria (`classify_attractor`, `attractor_radius`, λ\*, δ\*, q_hat) are not consulted during selection — only for an optional logged diagnostic and for threshold/eval, which are unchanged. The existing `ranked` / `direct` / `conformal` / `partx` strategies are untouched.

**Tech Stack:** Python, PyTorch, NumPy, SciPy (`scipy.stats.spearmanr`, v1.15.2 present), Hydra/OmegaConf configs, pytest 9.1.1.

**Spec:** `docs/superpowers/specs/2026-07-26-final-state-dispersion-acquisition-design.md`

## Global Constraints

- Selection must never call `system.classify_attractor()`, read `attractor_radius`, or read any field of `ThresholdState`. Only the Task 6 diagnostic may call `classify_attractor`, and its result must not influence which candidates are selected.
- Acquisition strategies are constructed by `engine.py:28` as `cls(cfg_node)` — a single positional arg holding the whole config node. Constructors take `(self, cfg)` and read `cfg.<field>`.
- `engine.py:200` calls `select(pool=, probability_backend=, threshold_backend=, threshold_state=, target_count=, exclude=)` by keyword. The signature must accept all six.
- Circular wrapping uses `atan2(sin(Δ), cos(Δ))`, matching `adaptive_roa/systems/pendulum.py:130`.
- Real systems (`PendulumSystem`, `CartPoleSystem`, …) load bounds from a `dataset_description.json` at construction and raise `FileNotFoundError` without it. **Never construct a real system in a unit test.** Use a locally-defined fake subclass of `DynamicalSystem`; it needs only `define_manifold_structure()` and `define_state_bounds()`, the only two abstract methods.
- Prediction labels: `1`=success, `-1`=failure, `0`=invalid/separatrix.
- Commit messages must contain no AI/tool attribution of any kind.

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `adaptive_roa/systems/base.py` | Modify | Add `get_normalization_scales()` (Task 1). |
| `adaptive_roa/adaptive_v2/strategy/dispersion_score.py` | Create | Pure functions: the metric (Task 2) and the three selection rules (Task 3). No model/pipeline imports. |
| `adaptive_roa/conformal/probability_estimator.py` | Modify | Add `sample_endpoints()` (Task 4). |
| `adaptive_roa/adaptive_v2/probability/endpoint_mc.py` | Modify | Add `sample_endpoints()` delegate (Task 4). |
| `adaptive_roa/adaptive_v2/strategy/dispersion.py` | Create | `DispersionAcquisitionStrategy` — orchestration only (Tasks 5, 6). |
| `configs/adaptive_v2/acquisition/dispersion.yaml` | Create | `acquisition=dispersion` (Task 5). |
| `tests/adaptive_v2/test_normalization_scales.py` | Create | Task 1. |
| `tests/adaptive_v2/test_dispersion_score.py` | Create | Tasks 2, 3. |
| `tests/adaptive_v2/test_probability_backends.py` | Modify | Task 4 (extend the existing file). |
| `tests/adaptive_v2/test_dispersion_strategy.py` | Create | Tasks 5, 6, 7. |

**No engine change is needed.** `engine.py:334` serializes `EpochArtifacts.__dict__` through `_convert_numpy`, which converts dataclasses via `asdict` (`engine.py:36`), so the full `AcquisitionResult.diagnostics` dict lands in `artifacts_v2.json` automatically. Only the legacy `results.json` cherry-picks two ranked-specific keys (`engine.py:289-290`); those will be `None` for dispersion runs, which is correct.

---

### Task 1: Per-dimension normalization scales

**Files:**
- Modify: `adaptive_roa/systems/base.py` (add method after `get_loss_weights`, which ends at line 135)
- Test: `tests/adaptive_v2/test_normalization_scales.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `DynamicalSystem.get_normalization_scales() -> torch.Tensor` of shape `[state_dim]`, dtype float32, all entries finite and > 0. Tasks 2, 3, 5 use it.

**Context:** This is *not* `get_loss_weights()`. That method returns weights *proportional* to each dimension's range (pendulum: θ→1.0, θ̇→8.0) to amplify wide dimensions in a loss. For a distance metric we need the opposite — divide by the range so a full-width spread costs the same in every dimension.

**Deviation from the spec:** the spec's testing section calls for checking the scales "for every registered system." That is not achievable as a unit test — `PendulumSystem`, `CartPoleSystem`, and the rest raise `FileNotFoundError` at construction without a `dataset_description.json` (`systems/pendulum.py:38`). The method lives entirely on the base class and is inherited unmodified by every system, so exercising the base-class logic through fake subclasses gives equivalent coverage without a data dependency.

- [ ] **Step 1: Write the failing test**

Create `tests/adaptive_v2/test_normalization_scales.py`:

```python
import math

import pytest
import torch

from adaptive_roa.systems.base import DynamicalSystem, ManifoldComponent


class _FakeSystem(DynamicalSystem):
    """SO2 angle + Real velocity bounded to [-8, 8]."""

    def define_manifold_structure(self):
        return [
            ManifoldComponent("SO2", 1, "angle"),
            ManifoldComponent("Real", 1, "angular_velocity"),
        ]

    def define_state_bounds(self):
        return {"angle": (-math.pi, math.pi), "angular_velocity": (-8.0, 8.0)}


class _MultiDimFakeSystem(DynamicalSystem):
    """3-dim Real component plus an SO2 component with no bounds entry."""

    def define_manifold_structure(self):
        return [
            ManifoldComponent("Real", 3, "position"),
            ManifoldComponent("SO2", 1, "angle"),
        ]

    def define_state_bounds(self):
        return {"position": (-2.0, 6.0)}


class _DegenerateFakeSystem(DynamicalSystem):
    """Real component whose declared bounds have zero width."""

    def define_manifold_structure(self):
        return [ManifoldComponent("Real", 1, "frozen")]

    def define_state_bounds(self):
        return {"frozen": (0.0, 0.0)}


def test_scales_shape_matches_state_dim():
    system = _FakeSystem()
    assert system.get_normalization_scales().shape == (system.state_dim,)


def test_circular_dim_scale_is_pi():
    scales = _FakeSystem().get_normalization_scales()
    assert scales[0].item() == pytest.approx(math.pi)


def test_real_dim_scale_is_half_range():
    scales = _FakeSystem().get_normalization_scales()
    assert scales[1].item() == pytest.approx(8.0)


def test_multi_dim_component_expands_and_missing_bounds_still_scale():
    scales = _MultiDimFakeSystem().get_normalization_scales()
    assert scales.shape == (4,)
    # position half-range = (6.0 - (-2.0)) / 2 = 4.0, repeated across all 3 dims
    assert scales[:3].tolist() == pytest.approx([4.0, 4.0, 4.0])
    # circular dims use pi regardless of any bounds entry
    assert scales[3].item() == pytest.approx(math.pi)


def test_zero_width_bounds_fall_back_to_one():
    scales = _DegenerateFakeSystem().get_normalization_scales()
    assert scales[0].item() == pytest.approx(1.0)


def test_all_scales_finite_and_positive():
    for system in (_FakeSystem(), _MultiDimFakeSystem(), _DegenerateFakeSystem()):
        scales = system.get_normalization_scales()
        assert torch.isfinite(scales).all()
        assert (scales > 0).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/adaptive_v2/test_normalization_scales.py -v`
Expected: FAIL with `AttributeError: '_FakeSystem' object has no attribute 'get_normalization_scales'`

- [ ] **Step 3: Add the `math` import**

In `adaptive_roa/systems/base.py`, the imports currently start:

```python
import torch
from abc import ABC, abstractmethod
```

Add `import math` above `import torch`.

- [ ] **Step 4: Implement the method**

In `adaptive_roa/systems/base.py`, immediately after `get_loss_weights()` (which ends with its `return torch.tensor(...)` at line 135), add:

```python
    def get_normalization_scales(self) -> torch.Tensor:
        """
        Get per-dimension distance scales for range-normalized state metrics.

        Dividing a state difference by these scales puts every dimension on
        equal footing, so a full-width spread costs the same in each one.
        Without it, the widest-range dimension dominates any distance computed
        over the full state vector.

        This is NOT get_loss_weights(): those weights are proportional to each
        dimension's range (amplifying wide dimensions), which is the wrong sign
        for a distance metric.

        Circular (SO2) dimensions use π, since angle differences wrap into
        [-π, π]. SO3/Sphere components use 1.0 (their coordinates live in
        [-1, 1]). Real components use half their declared range, falling back
        to 1.0 when bounds are missing or zero-width.

        Returns:
            torch.Tensor: Per-dimension scales [state_dim], all finite and > 0
        """
        scales = []

        for comp in self._manifold_components:
            if comp.manifold_type == "SO2":
                scales.extend([math.pi] * comp.dim)
            elif comp.manifold_type in ("SO3", "Sphere"):
                scales.extend([1.0] * comp.dim)
            else:
                lo, hi = self._state_bounds.get(comp.name, (-1.0, 1.0))
                half_range = (hi - lo) / 2.0
                scales.extend([half_range if half_range > 0 else 1.0] * comp.dim)

        return torch.tensor(scales, dtype=torch.float32)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/adaptive_v2/test_normalization_scales.py -v`
Expected: 6 passed

- [ ] **Step 6: Commit**

```bash
git add adaptive_roa/systems/base.py tests/adaptive_v2/test_normalization_scales.py
git commit -m "feat(systems): per-dimension normalization scales for state distance metrics"
```

---

### Task 2: Mean pairwise dispersion metric

**Files:**
- Create: `adaptive_roa/adaptive_v2/strategy/dispersion_score.py`
- Test: `tests/adaptive_v2/test_dispersion_score.py`

**Interfaces:**
- Consumes: `get_normalization_scales()` output (as a numpy array) from Task 1.
- Produces:
  - `mean_pairwise_dispersion(endpoints: np.ndarray, scales: np.ndarray, circular_mask: np.ndarray, chunk_size: int = 2048, device: str = "cpu") -> np.ndarray` — takes `endpoints [M, K, D]`, returns `[M]` float64. Entries are `NaN` for candidates with any non-finite endpoint.
  - `normalized_distances(states: np.ndarray, reference: np.ndarray, scales: np.ndarray, circular_mask: np.ndarray) -> np.ndarray` — takes `states [P, D]` and a single `reference [D]`, returns `[P]` float64. Used by Task 3.

**Context:** The score is `2/(K(K−1)) · Σ_{i<j} d_ij`. Because the full pairwise matrix contains each pair twice plus a zero diagonal, `sum(all d_ij) = 2·Σ_{i<j}`, so the score simplifies to `sum(all d_ij) / (K(K−1))`. There is no algebraic shortcut that avoids materializing the pairs — unlike variance, mean pairwise distance genuinely needs all K². The `[M,K,K,D]` broadcast is 960 MB at M=50k, K=20, D=12, hence chunking over candidates.

- [ ] **Step 1: Write the failing test**

Create `tests/adaptive_v2/test_dispersion_score.py`:

```python
import math

import numpy as np
import pytest

from adaptive_roa.adaptive_v2.strategy.dispersion_score import (
    mean_pairwise_dispersion,
    normalized_distances,
)

# A pendulum-like 2-D space: theta circular, theta_dot bounded to [-8, 8].
SCALES = np.array([math.pi, 8.0], dtype=np.float64)
CIRCULAR = np.array([True, False])


def test_identical_endpoints_score_zero():
    endpoints = np.tile(np.array([[0.3, 1.5]]), (4, 1))[None, :, :]  # [1, 4, 2]
    scores = mean_pairwise_dispersion(endpoints, SCALES, CIRCULAR)
    assert scores.shape == (1,)
    assert scores[0] == pytest.approx(0.0, abs=1e-6)


def test_circular_dimension_wraps_across_pi():
    eps = 0.01
    endpoints = np.array([[[math.pi - eps, 0.0], [-math.pi + eps, 0.0]]])  # [1, 2, 2]
    scores = mean_pairwise_dispersion(endpoints, SCALES, CIRCULAR)
    # True separation is 2*eps, not 2*pi - 2*eps; normalized by pi.
    assert scores[0] == pytest.approx(2 * eps / math.pi, abs=1e-5)


def test_range_normalization_equalizes_dimensions():
    # Full-width spread in theta (2*pi wide, but wraps -> pi apart max)
    theta_spread = np.array([[[math.pi / 2, 0.0], [-math.pi / 2, 0.0]]])
    # Full-width spread in theta_dot (16 wide -> 8.0 apart after halving)
    vel_spread = np.array([[[0.0, 4.0], [0.0, -4.0]]])
    s_theta = mean_pairwise_dispersion(theta_spread, SCALES, CIRCULAR)
    s_vel = mean_pairwise_dispersion(vel_spread, SCALES, CIRCULAR)
    assert s_theta[0] == pytest.approx(s_vel[0], rel=1e-5)


def test_two_modes_outscore_one_tight_blob():
    tight = np.array([[[0.0, 0.0], [0.01, 0.0], [0.0, 0.01], [0.01, 0.01]]])
    bimodal = np.array([[[0.0, 0.0], [0.01, 0.0], [1.5, 0.0], [1.51, 0.0]]])
    scores = mean_pairwise_dispersion(
        np.concatenate([tight, bimodal], axis=0), SCALES, CIRCULAR
    )
    assert scores[1] > scores[0]


def test_known_two_point_value():
    # Two endpoints 4.0 apart in theta_dot only: d = 4.0 / 8.0 = 0.5.
    # K=2 -> mean over the single pair = 0.5.
    endpoints = np.array([[[0.0, 0.0], [0.0, 4.0]]])
    scores = mean_pairwise_dispersion(endpoints, SCALES, CIRCULAR)
    assert scores[0] == pytest.approx(0.5, abs=1e-6)


def test_chunking_matches_unchunked():
    rng = np.random.default_rng(0)
    endpoints = rng.normal(size=(37, 6, 2))
    full = mean_pairwise_dispersion(endpoints, SCALES, CIRCULAR, chunk_size=1024)
    chunked = mean_pairwise_dispersion(endpoints, SCALES, CIRCULAR, chunk_size=5)
    np.testing.assert_allclose(full, chunked, rtol=1e-5)


def test_non_finite_candidate_scores_nan_without_poisoning_neighbours():
    endpoints = np.array(
        [
            [[0.0, 0.0], [0.0, 4.0]],
            [[0.0, np.nan], [0.0, 4.0]],
            [[0.0, 0.0], [np.inf, 4.0]],
            [[0.0, 0.0], [0.0, 4.0]],
        ]
    )
    scores = mean_pairwise_dispersion(endpoints, SCALES, CIRCULAR, chunk_size=2)
    assert np.isnan(scores[1])
    assert np.isnan(scores[2])
    assert scores[0] == pytest.approx(0.5, abs=1e-6)
    assert scores[3] == pytest.approx(0.5, abs=1e-6)


def test_requires_at_least_two_samples():
    with pytest.raises(ValueError, match="at least 2"):
        mean_pairwise_dispersion(np.zeros((3, 1, 2)), SCALES, CIRCULAR)


def test_normalized_distances_wraps_and_scales():
    states = np.array([[math.pi - 0.01, 0.0], [0.0, 8.0]])
    reference = np.array([-math.pi + 0.01, 0.0])
    d = normalized_distances(states, reference, SCALES, CIRCULAR)
    assert d[0] == pytest.approx(0.02 / math.pi, abs=1e-5)
    # second: theta differs by pi - 0.01 (normalized ~0.997), vel by 8/8 = 1.0
    expected = math.hypot((math.pi - 0.01) / math.pi, 1.0)
    assert d[1] == pytest.approx(expected, rel=1e-5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/adaptive_v2/test_dispersion_score.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'adaptive_roa.adaptive_v2.strategy.dispersion_score'`

- [ ] **Step 3: Write the implementation**

Create `adaptive_roa/adaptive_v2/strategy/dispersion_score.py`:

```python
"""Pure scoring functions for dispersion-based acquisition.

Nothing here imports a model, a pipeline component, or the success criteria:
every function operates on plain arrays so it can be unit-tested without a
trained model and reasoned about in isolation.
"""

from __future__ import annotations

import numpy as np
import torch


def _wrap_circular_(diff: torch.Tensor, circular_mask: torch.Tensor) -> torch.Tensor:
    """Wrap circular columns of `diff` into [-pi, pi], in place.

    Uses atan2(sin, cos) to match the convention in
    adaptive_roa/systems/pendulum.py:130. Only the circular columns are
    materialized, which keeps the temporaries small for wide state vectors.
    """
    if not bool(circular_mask.any()):
        return diff
    circ = diff[..., circular_mask]
    diff[..., circular_mask] = torch.atan2(torch.sin(circ), torch.cos(circ))
    return diff


def mean_pairwise_dispersion(
    endpoints: np.ndarray,
    scales: np.ndarray,
    circular_mask: np.ndarray,
    chunk_size: int = 2048,
    device: str = "cpu",
) -> np.ndarray:
    """Score each candidate by the mean pairwise distance among its endpoints.

    score(x) = 2 / (K(K-1)) * sum_{i<j} || wrap(e_i - e_j) / scales ||_2

    The full pairwise matrix holds each pair twice plus a zero diagonal, so
    sum(all d_ij) == 2 * sum_{i<j} d_ij and the score reduces to
    sum(all d_ij) / (K(K-1)).

    Args:
        endpoints: Predicted endpoint clouds [M, K, D]
        scales: Per-dimension distance scales [D], all > 0
        circular_mask: Boolean [D], True where the dimension wraps
        chunk_size: Candidates processed per block (bounds the [m,K,K,D] temp)
        device: Torch device for the computation

    Returns:
        np.ndarray: Scores [M] as float64. NaN for any candidate with a
            non-finite endpoint; such candidates are excluded downstream.
    """
    endpoints = np.asarray(endpoints)
    if endpoints.ndim != 3:
        raise ValueError(f"endpoints must be [M, K, D], got shape {endpoints.shape}")

    M, K, D = endpoints.shape
    if K < 2:
        raise ValueError(f"dispersion needs at least 2 endpoint samples, got K={K}")
    if scales.shape != (D,):
        raise ValueError(f"scales must be [{D}], got {scales.shape}")
    if circular_mask.shape != (D,):
        raise ValueError(f"circular_mask must be [{D}], got {circular_mask.shape}")

    scales_t = torch.as_tensor(np.asarray(scales), dtype=torch.float32, device=device)
    circ_t = torch.as_tensor(np.asarray(circular_mask), dtype=torch.bool, device=device)
    out = np.empty(M, dtype=np.float64)

    for start in range(0, M, chunk_size):
        stop = min(start + chunk_size, M)
        chunk = torch.as_tensor(
            endpoints[start:stop], dtype=torch.float32, device=device
        )

        finite = torch.isfinite(chunk).all(dim=2).all(dim=1)          # [m]
        diff = chunk.unsqueeze(2) - chunk.unsqueeze(1)                # [m, K, K, D]
        diff = _wrap_circular_(diff, circ_t)
        dists = torch.linalg.vector_norm(diff / scales_t, dim=-1)     # [m, K, K]
        scores = dists.sum(dim=(1, 2)) / (K * (K - 1))                # [m]

        scores = torch.where(finite, scores, torch.full_like(scores, float("nan")))
        out[start:stop] = scores.double().cpu().numpy()

    return out


def normalized_distances(
    states: np.ndarray,
    reference: np.ndarray,
    scales: np.ndarray,
    circular_mask: np.ndarray,
) -> np.ndarray:
    """Distance from every state to one reference state, same metric as above.

    Args:
        states: States [P, D]
        reference: Single state [D]
        scales: Per-dimension distance scales [D]
        circular_mask: Boolean [D], True where the dimension wraps

    Returns:
        np.ndarray: Distances [P] as float64
    """
    diff = np.asarray(states, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    circular_mask = np.asarray(circular_mask)
    if circular_mask.any():
        circ = diff[:, circular_mask]
        diff[:, circular_mask] = np.arctan2(np.sin(circ), np.cos(circ))
    return np.linalg.norm(diff / np.asarray(scales, dtype=np.float64), axis=1)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/adaptive_v2/test_dispersion_score.py -v`
Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add adaptive_roa/adaptive_v2/strategy/dispersion_score.py tests/adaptive_v2/test_dispersion_score.py
git commit -m "feat(adaptive_v2): mean pairwise dispersion metric for endpoint clouds"
```

---

### Task 3: Selection rules

**Files:**
- Modify: `adaptive_roa/adaptive_v2/strategy/dispersion_score.py` (append)
- Test: `tests/adaptive_v2/test_dispersion_score.py` (append)

**Interfaces:**
- Consumes: `normalized_distances()` from Task 2.
- Produces, all returning `np.ndarray` of **positions into the candidate array** (not pool indices), dtype int:
  - `select_greedy(scores: np.ndarray, n_select: int) -> np.ndarray`
  - `select_greedy_diverse(scores: np.ndarray, states: np.ndarray, scales: np.ndarray, circular_mask: np.ndarray, n_select: int, pool_multiplier: int = 5) -> np.ndarray`
  - `select_proportional(scores: np.ndarray, n_select: int, temperature: float = 0.1, seed: int | None = None) -> np.ndarray`

**Context:** All three must skip `NaN` scores (Task 2 marks non-finite candidates that way). `select_greedy_diverse` measures distance in **initial-state** space, not endpoint space — the point is a batch that covers different regions of the state space. `select_proportional` uses Gumbel-top-k, which yields exact sampling-without-replacement from `softmax(logits)` in one vectorized pass.

- [ ] **Step 1: Write the failing test**

Append to `tests/adaptive_v2/test_dispersion_score.py`:

```python
from adaptive_roa.adaptive_v2.strategy.dispersion_score import (
    select_greedy,
    select_greedy_diverse,
    select_proportional,
)


def test_greedy_takes_highest_scores():
    scores = np.array([0.1, 0.9, 0.5, 0.7])
    assert select_greedy(scores, 2).tolist() == [1, 3]


def test_greedy_skips_nan_scores():
    scores = np.array([0.1, np.nan, 0.5, np.nan, 0.7])
    picked = select_greedy(scores, 3).tolist()
    assert picked == [4, 2, 0]


def test_greedy_returns_all_when_fewer_than_requested():
    scores = np.array([0.1, np.nan, 0.5])
    assert sorted(select_greedy(scores, 10).tolist()) == [0, 2]


def test_greedy_diverse_spreads_further_than_greedy():
    # Four high-scoring states clustered together, plus three lower-scoring
    # states that are far away. Greedy takes only the cluster; diverse spreads.
    states = np.array(
        [
            [0.00, 0.0], [0.01, 0.0], [0.02, 0.0], [0.03, 0.0],
            [2.00, 0.0], [-2.00, 0.0], [0.00, 6.0],
        ]
    )
    scores = np.array([0.99, 0.98, 0.97, 0.96, 0.90, 0.89, 0.88])

    greedy = select_greedy(scores, 3)
    diverse = select_greedy_diverse(scores, states, SCALES, CIRCULAR, 3, pool_multiplier=3)

    def min_separation(idx):
        pts = states[idx]
        return min(
            normalized_distances(pts, pts[i], SCALES, CIRCULAR)[j]
            for i in range(len(pts))
            for j in range(len(pts))
            if i != j
        )

    assert min_separation(diverse) > min_separation(greedy)


def test_greedy_diverse_seeds_at_highest_score():
    states = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    scores = np.array([0.1, 0.9, 0.5, 0.4])
    picked = select_greedy_diverse(scores, states, SCALES, CIRCULAR, 2, pool_multiplier=4)
    assert picked[0] == 1


def test_greedy_diverse_returns_unique_indices():
    rng = np.random.default_rng(3)
    states = rng.normal(size=(40, 2))
    scores = rng.random(40)
    picked = select_greedy_diverse(scores, states, SCALES, CIRCULAR, 10)
    assert len(set(picked.tolist())) == 10


def test_greedy_diverse_skips_nan_scores():
    states = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    scores = np.array([0.9, np.nan, 0.5, 0.4])
    picked = select_greedy_diverse(scores, states, SCALES, CIRCULAR, 3, pool_multiplier=5)
    assert 1 not in picked.tolist()


def test_proportional_is_reproducible_under_seed():
    rng = np.random.default_rng(1)
    scores = rng.random(50)
    a = select_proportional(scores, 10, temperature=0.1, seed=7)
    b = select_proportional(scores, 10, temperature=0.1, seed=7)
    np.testing.assert_array_equal(a, b)


def test_proportional_differs_across_seeds():
    rng = np.random.default_rng(1)
    scores = rng.random(200)
    a = select_proportional(scores, 20, temperature=0.5, seed=1)
    b = select_proportional(scores, 20, temperature=0.5, seed=2)
    assert set(a.tolist()) != set(b.tolist())


def test_proportional_favours_high_scores_at_low_temperature():
    scores = np.concatenate([np.full(10, 1.0), np.full(90, 0.0)])
    picked = select_proportional(scores, 10, temperature=0.01, seed=5)
    assert set(picked.tolist()) == set(range(10))


def test_proportional_selects_unique_and_skips_nan():
    scores = np.array([0.9, np.nan, 0.5, 0.4, 0.8, np.nan, 0.2])
    picked = select_proportional(scores, 3, temperature=0.2, seed=0)
    assert len(set(picked.tolist())) == 3
    assert 1 not in picked.tolist()
    assert 5 not in picked.tolist()


def test_proportional_handles_all_equal_scores():
    scores = np.full(20, 0.42)
    picked = select_proportional(scores, 5, temperature=0.1, seed=0)
    assert len(set(picked.tolist())) == 5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/adaptive_v2/test_dispersion_score.py -v`
Expected: FAIL with `ImportError: cannot import name 'select_greedy'`

- [ ] **Step 3: Write the implementation**

Append to `adaptive_roa/adaptive_v2/strategy/dispersion_score.py`:

```python
def _finite_order(scores: np.ndarray) -> np.ndarray:
    """Positions of finite scores, ordered by score descending (stable)."""
    valid = np.flatnonzero(np.isfinite(scores))
    return valid[np.argsort(-scores[valid], kind="stable")]


def select_greedy(scores: np.ndarray, n_select: int) -> np.ndarray:
    """Take the n_select highest-scoring candidates.

    Args:
        scores: Dispersion scores [M]; NaN entries are skipped
        n_select: Number of candidates to select

    Returns:
        np.ndarray: Positions into `scores`, highest score first
    """
    return _finite_order(scores)[:n_select]


def select_greedy_diverse(
    scores: np.ndarray,
    states: np.ndarray,
    scales: np.ndarray,
    circular_mask: np.ndarray,
    n_select: int,
    pool_multiplier: int = 5,
) -> np.ndarray:
    """Farthest-point-sample n_select from the top-scoring shortlist.

    Takes the top `pool_multiplier * n_select` candidates by score, then
    greedily picks points that are far apart in *initial-state* space, seeded
    at the highest-scoring candidate. This stops the batch collapsing onto a
    single high-uncertainty pocket.

    Args:
        scores: Dispersion scores [M]; NaN entries are skipped
        states: Candidate initial states [M, D]
        scales: Per-dimension distance scales [D]
        circular_mask: Boolean [D], True where the dimension wraps
        n_select: Number of candidates to select
        pool_multiplier: Shortlist size as a multiple of n_select

    Returns:
        np.ndarray: Positions into `scores`, seed candidate first
    """
    order = _finite_order(scores)
    shortlist = order[: max(n_select * pool_multiplier, n_select)]
    if len(shortlist) <= n_select:
        return shortlist

    X = np.asarray(states)[shortlist]
    selected = [0]  # shortlist is score-sorted, so 0 is the highest scorer
    dist = normalized_distances(X, X[0], scales, circular_mask)
    dist[0] = -np.inf

    while len(selected) < n_select:
        nxt = int(np.argmax(dist))
        selected.append(nxt)
        dist = np.minimum(dist, normalized_distances(X, X[nxt], scales, circular_mask))
        dist[selected] = -np.inf

    return shortlist[np.asarray(selected, dtype=int)]


def select_proportional(
    scores: np.ndarray,
    n_select: int,
    temperature: float = 0.1,
    seed: int | None = None,
) -> np.ndarray:
    """Sample n_select candidates with probability rising in the score.

    Scores are min-max normalized within the batch before the softmax, so
    `temperature` carries the same meaning across systems and epochs. Sampling
    without replacement uses the Gumbel-top-k trick, which is exact and needs
    no renormalization loop.

    Args:
        scores: Dispersion scores [M]; NaN entries are skipped
        n_select: Number of candidates to select
        temperature: Softmax temperature; lower concentrates on top scores
        seed: Seed for reproducibility; None draws fresh entropy

    Returns:
        np.ndarray: Positions into `scores`, unordered
    """
    valid = np.flatnonzero(np.isfinite(scores))
    if len(valid) <= n_select:
        return valid

    s = np.asarray(scores, dtype=np.float64)[valid]
    lo, hi = s.min(), s.max()
    s_norm = np.zeros_like(s) if hi <= lo else (s - lo) / (hi - lo)

    logits = s_norm / max(float(temperature), 1e-12)
    rng = np.random.default_rng(seed)
    keys = logits + rng.gumbel(size=len(s))

    top = np.argpartition(-keys, n_select - 1)[:n_select]
    return valid[top]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/adaptive_v2/test_dispersion_score.py -v`
Expected: 20 passed

- [ ] **Step 5: Commit**

```bash
git add adaptive_roa/adaptive_v2/strategy/dispersion_score.py tests/adaptive_v2/test_dispersion_score.py
git commit -m "feat(adaptive_v2): greedy, diverse, and proportional selection rules"
```

---

### Task 4: Raw endpoint cloud sampling

**Files:**
- Modify: `adaptive_roa/conformal/probability_estimator.py` (add method after `estimate()`, which ends at line 180)
- Modify: `adaptive_roa/adaptive_v2/probability/endpoint_mc.py` (add method after `estimate()`, line 57)
- Test: `tests/adaptive_v2/test_probability_backends.py` (append)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `ProbabilityEstimator.sample_endpoints(states, num_samples: int, verbose: bool = True) -> np.ndarray` of shape `[N, K, D]`, float32.
  - `EndpointMCProbabilityBackend.sample_endpoints(start_states: np.ndarray, num_samples: int, verbose: bool = True) -> np.ndarray`, same shape.

**Context:** This is deliberately a **second MC path**, not a refactor of `estimate()`. `estimate()` interleaves per-pass refinement of invalid endpoints (`probability_estimator.py:148`), which is label-driven and meaningless here; merging them would drag the success criteria back into the acquisition path. `sample_endpoints` classifies nothing, refines nothing, and never reads `attractor_radius`.

The output array is allocated lazily from the first prediction's width rather than from `states.shape[1]`, so it stays correct if an endpoint's dimensionality ever differs from the input state's.

- [ ] **Step 1: Write the failing test**

Append to `tests/adaptive_v2/test_probability_backends.py`:

```python
from adaptive_roa.conformal.config import ConformalConfig
from adaptive_roa.conformal.probability_estimator import ProbabilityEstimator


class _CountingFlowMatcher:
    """Returns a distinct constant endpoint per call, so passes are traceable."""

    def __init__(self):
        self.calls = 0

    def eval(self):
        return self

    def predict_endpoint(self, states):
        self.calls += 1
        return torch.full((states.shape[0], 2), float(self.calls))


def test_sample_endpoints_shape_and_contents():
    fm = _CountingFlowMatcher()
    cfg = ConformalConfig(num_mc_samples=99, attractor_radius=0.2)
    estimator = ProbabilityEstimator(fm, system=None, config=cfg, device="cpu")

    out = estimator.sample_endpoints(np.zeros((7, 2), dtype=np.float32), 3, verbose=False)

    assert out.shape == (7, 3, 2)
    # One batch (mc_batch_size=1024 > 7), so the three passes are calls 1, 2, 3.
    np.testing.assert_allclose(out[:, 0, :], 1.0)
    np.testing.assert_allclose(out[:, 1, :], 2.0)
    np.testing.assert_allclose(out[:, 2, :], 3.0)
    # num_samples overrides config.num_mc_samples entirely.
    assert fm.calls == 3


def test_sample_endpoints_respects_mc_batch_size():
    fm = _CountingFlowMatcher()
    cfg = ConformalConfig(num_mc_samples=1, attractor_radius=0.2, mc_batch_size=4)
    estimator = ProbabilityEstimator(fm, system=None, config=cfg, device="cpu")

    out = estimator.sample_endpoints(np.zeros((10, 2), dtype=np.float32), 2, verbose=False)

    assert out.shape == (10, 2, 2)
    # 10 states / batch 4 -> 3 batches, 2 passes each.
    assert fm.calls == 6


def test_sample_endpoints_accepts_torch_input():
    fm = _CountingFlowMatcher()
    cfg = ConformalConfig(attractor_radius=0.2)
    estimator = ProbabilityEstimator(fm, system=None, config=cfg, device="cpu")

    out = estimator.sample_endpoints(torch.zeros((3, 2)), 2, verbose=False)

    assert isinstance(out, np.ndarray)
    assert out.shape == (3, 2, 2)


def test_mc_backend_sample_endpoints_delegates():
    backend = EndpointMCProbabilityBackend(_mc_cfg(), system=None, device="cpu")
    backend.bind_model(_CountingFlowMatcher())

    out = backend.sample_endpoints(np.zeros((5, 2), dtype=np.float32), 4, verbose=False)

    assert out.shape == (5, 4, 2)


def test_mc_backend_sample_endpoints_requires_bind_model():
    backend = EndpointMCProbabilityBackend(_mc_cfg(), system=None, device="cpu")
    with pytest.raises(RuntimeError, match="before bind_model"):
        backend.sample_endpoints(np.zeros((2, 2), dtype=np.float32), 2, verbose=False)
```

Add `import pytest` to the top of the file if it is not already imported.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/adaptive_v2/test_probability_backends.py -v`
Expected: FAIL with `AttributeError: 'ProbabilityEstimator' object has no attribute 'sample_endpoints'`

- [ ] **Step 3: Implement on the estimator**

In `adaptive_roa/conformal/probability_estimator.py`, add after `estimate()` (which ends with its `return p_success, p_failure, p_invalid` at line 180) and before `estimate_single()`:

```python
    @torch.no_grad()
    def sample_endpoints(
        self,
        states: Union[torch.Tensor, np.ndarray],
        num_samples: int,
        verbose: bool = True,
    ) -> np.ndarray:
        """
        Sample raw endpoint clouds without classifying them.

        For each state, runs num_samples forward passes (each with a fresh
        z ~ N(0,I)) and returns the predicted endpoints as-is. Unlike
        estimate(), this performs no attractor classification and no
        refinement of invalid endpoints, so it never consults the success
        criteria. It is the input to distribution-based uncertainty scoring.

        Args:
            states: Initial states [N, state_dim] as torch tensor or numpy array
            num_samples: Number of endpoint samples K per state; overrides
                config.num_mc_samples
            verbose: Show a progress bar

        Returns:
            np.ndarray: Endpoint clouds [N, K, endpoint_dim] as float32
        """
        if isinstance(states, np.ndarray):
            states = torch.from_numpy(states).float()

        states = states.to(self.device)
        N = states.shape[0]
        K = int(num_samples)
        if K < 1:
            raise ValueError(f"num_samples must be >= 1, got {num_samples}")
        batch_size = self.config.mc_batch_size

        if verbose:
            print(f"      [Cloud] N={N} states, K={K} samples, batch_size={batch_size}")

        n_batches = (N + batch_size - 1) // batch_size
        out: torch.Tensor | None = None

        with tqdm(total=n_batches * K, desc="Endpoint sampling", disable=not verbose) as pbar:
            for batch_start in range(0, N, batch_size):
                batch_end = min(batch_start + batch_size, N)
                batch_states = states[batch_start:batch_end]

                for k in range(K):
                    endpoints = self.flow_matcher.predict_endpoint(batch_states)
                    if out is None:
                        out = torch.empty(
                            (N, K, endpoints.shape[1]), dtype=torch.float32
                        )
                    out[batch_start:batch_end, k, :] = endpoints.detach().float().cpu()
                    pbar.update(1)

        return out.numpy()
```

- [ ] **Step 4: Implement the backend delegate**

In `adaptive_roa/adaptive_v2/probability/endpoint_mc.py`, add after `estimate()` (which ends at line 57):

```python
    def sample_endpoints(
        self,
        start_states: np.ndarray,
        num_samples: int,
        verbose: bool = True,
    ) -> np.ndarray:
        """Return raw endpoint clouds [N, K, D] with no classification."""
        if self.estimator is None:
            raise RuntimeError("Probability backend used before bind_model")
        return self.estimator.sample_endpoints(
            start_states, num_samples=num_samples, verbose=verbose
        )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/adaptive_v2/test_probability_backends.py -v`
Expected: all previously passing tests still pass, plus 5 new ones

- [ ] **Step 6: Commit**

```bash
git add adaptive_roa/conformal/probability_estimator.py adaptive_roa/adaptive_v2/probability/endpoint_mc.py tests/adaptive_v2/test_probability_backends.py
git commit -m "feat(conformal): sample raw endpoint clouds without classification"
```

---

### Task 5: Dispersion acquisition strategy

**Files:**
- Create: `adaptive_roa/adaptive_v2/strategy/dispersion.py`
- Create: `configs/adaptive_v2/acquisition/dispersion.yaml`
- Test: `tests/adaptive_v2/test_dispersion_strategy.py`

**Interfaces:**
- Consumes: `get_normalization_scales()` (Task 1); `mean_pairwise_dispersion`, `select_greedy`, `select_greedy_diverse`, `select_proportional` (Tasks 2-3); `probability_backend.sample_endpoints()` (Task 4).
- Produces: `DispersionAcquisitionStrategy` with `mode = "dispersion"`, `.d2_ratio`, and `select(pool, probability_backend, threshold_backend, threshold_state, target_count, exclude=None) -> AcquisitionResult`.

**Context:** The engine constructs strategies as `cls(cfg_node)` with no system or device (`engine.py:28,87`), so the strategy gets both from the probability backend, which stores `self.system` and `self.device` (`endpoint_mc.py:24-25`). `threshold_state` is accepted for Protocol conformance and never read.

`exclude` is honoured by passing it through to `pool.sample_candidates_without_marking(n, exclude=exclude)` — the pool supports it (`pool/trajectory_pool.py:47-52`), even though `ranked` drops it.

- [ ] **Step 1: Write the failing test**

Create `tests/adaptive_v2/test_dispersion_strategy.py`:

```python
import math

import numpy as np
import pytest
from omegaconf import OmegaConf

from adaptive_roa.adaptive_v2.strategy.dispersion import DispersionAcquisitionStrategy
from adaptive_roa.adaptive_v2.types import ThresholdState
from adaptive_roa.systems.base import DynamicalSystem, ManifoldComponent


class _FakeSystem(DynamicalSystem):
    """Pendulum-like: SO2 angle + Real velocity in [-8, 8]."""

    def define_manifold_structure(self):
        return [
            ManifoldComponent("SO2", 1, "angle"),
            ManifoldComponent("Real", 1, "angular_velocity"),
        ]

    def define_state_bounds(self):
        return {"angle": (-math.pi, math.pi), "angular_velocity": (-8.0, 8.0)}

    def classify_attractor(self, state, radius=0.1):
        import torch

        # Success when |theta| < radius, failure when theta > 1.0, else invalid.
        theta = state[:, 0]
        labels = torch.zeros(state.shape[0], dtype=torch.int64)
        labels[theta.abs() < radius] = 1
        labels[theta > 1.0] = -1
        return labels


class _FakePool:
    """Serves a fixed candidate array and records what was excluded."""

    def __init__(self, states):
        self.states = np.asarray(states, dtype=np.float32)
        self.last_exclude = None
        self.last_n = None

    def sample_candidates_without_marking(self, n, exclude=None):
        self.last_exclude = exclude
        self.last_n = n
        take = min(n, len(self.states))
        return self.states[:take], list(range(100, 100 + take))


class _FakeBackend:
    """Returns a preset cloud per candidate; records the requested K."""

    def __init__(self, clouds, system):
        self.clouds = np.asarray(clouds, dtype=np.float32)
        self.system = system
        self.device = "cpu"
        self.attractor_radius = 0.2
        self.last_num_samples = None

    def sample_endpoints(self, start_states, num_samples, verbose=True):
        self.last_num_samples = num_samples
        return self.clouds[: len(start_states)]


def _cfg(**overrides):
    base = {
        "d2_ratio": 0.5,
        "n_dispersion_candidates": 50,
        "num_mc_samples_dispersion": 4,
        "selection_rule": "greedy",
        "diversity_pool_multiplier": 5,
        "temperature": 0.1,
        "seed": None,
        "chunk_size": 2048,
        "log_score_correlation": False,
        "verbose": False,
    }
    base.update(overrides)
    return OmegaConf.create(base)


def _threshold_state():
    return ThresholdState(lambda_star=0.5, delta_star=0.1)


def _clouds_with_spreads(spreads):
    """Build [M, 2, 2] clouds where candidate i has the given velocity spread."""
    return np.array([[[0.0, 0.0], [0.0, s]] for s in spreads], dtype=np.float32)


def test_stores_cfg():
    s = DispersionAcquisitionStrategy(_cfg())
    assert s.mode == "dispersion"
    assert s.d2_ratio == 0.5
    assert s.num_mc_samples_dispersion == 4
    assert s.selection_rule == "greedy"


def test_rejects_unknown_selection_rule():
    with pytest.raises(ValueError, match="selection_rule"):
        DispersionAcquisitionStrategy(_cfg(selection_rule="nonsense"))


def test_selects_highest_dispersion_candidates():
    system = _FakeSystem()
    states = np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0], [1.5, 0.0]])
    pool = _FakePool(states)
    backend = _FakeBackend(_clouds_with_spreads([0.1, 8.0, 0.2, 4.0]), system)

    result = DispersionAcquisitionStrategy(_cfg()).select(
        pool=pool,
        probability_backend=backend,
        threshold_backend=None,
        threshold_state=_threshold_state(),
        target_count=2,
    )

    # Candidates 1 (spread 8.0) and 3 (spread 4.0) -> pool indices 101 and 103.
    assert result.d2_indices == [101, 103]
    assert result.d1_indices == []
    assert result.n_candidates_evaluated == 4
    assert backend.last_num_samples == 4


def test_passes_exclude_through_to_pool():
    system = _FakeSystem()
    pool = _FakePool(np.array([[0.0, 0.0], [1.0, 0.0]]))
    backend = _FakeBackend(_clouds_with_spreads([1.0, 2.0]), system)

    DispersionAcquisitionStrategy(_cfg()).select(
        pool=pool,
        probability_backend=backend,
        threshold_backend=None,
        threshold_state=_threshold_state(),
        target_count=1,
        exclude={7, 9},
    )

    assert pool.last_exclude == {7, 9}
    assert pool.last_n == 50


def test_diagnostics_populated():
    system = _FakeSystem()
    pool = _FakePool(np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]))
    backend = _FakeBackend(_clouds_with_spreads([1.0, 8.0, 4.0]), system)

    result = DispersionAcquisitionStrategy(_cfg()).select(
        pool=pool,
        probability_backend=backend,
        threshold_backend=None,
        threshold_state=_threshold_state(),
        target_count=2,
    )

    d = result.diagnostics
    for key in (
        "dispersion_score_threshold",
        "dispersion_score_min",
        "dispersion_score_max",
        "dispersion_score_mean",
        "dispersion_score_median",
        "n_dispersion_candidates_evaluated",
        "n_nonfinite_excluded",
        "selection_rule",
    ):
        assert key in d, f"missing diagnostic: {key}"
    assert d["selection_rule"] == "greedy"
    assert d["n_nonfinite_excluded"] == 0
    # threshold is the lowest score among the two selected
    assert d["dispersion_score_threshold"] == pytest.approx(4.0 / 8.0, abs=1e-6)


def test_target_count_zero_skips():
    result = DispersionAcquisitionStrategy(_cfg()).select(
        pool=_FakePool(np.zeros((3, 2))),
        probability_backend=_FakeBackend(_clouds_with_spreads([1.0, 2.0, 3.0]), _FakeSystem()),
        threshold_backend=None,
        threshold_state=_threshold_state(),
        target_count=0,
    )
    assert result.d2_indices == []
    assert result.diagnostics["skipped_reason"] == "target_count_zero"


def test_empty_pool_skips():
    system = _FakeSystem()
    result = DispersionAcquisitionStrategy(_cfg()).select(
        pool=_FakePool(np.zeros((0, 2))),
        probability_backend=_FakeBackend(np.zeros((0, 2, 2)), system),
        threshold_backend=None,
        threshold_state=_threshold_state(),
        target_count=5,
    )
    assert result.d2_indices == []
    assert result.diagnostics["skipped_reason"] == "pool_exhausted"


def test_selects_all_when_pool_smaller_than_target():
    system = _FakeSystem()
    pool = _FakePool(np.array([[0.0, 0.0], [1.0, 0.0]]))
    backend = _FakeBackend(_clouds_with_spreads([1.0, 2.0]), system)

    result = DispersionAcquisitionStrategy(_cfg()).select(
        pool=pool,
        probability_backend=backend,
        threshold_backend=None,
        threshold_state=_threshold_state(),
        target_count=10,
    )

    assert sorted(result.d2_indices) == [100, 101]


def test_non_finite_endpoints_excluded_and_counted():
    system = _FakeSystem()
    pool = _FakePool(np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]))
    clouds = _clouds_with_spreads([1.0, 2.0, 3.0])
    clouds[1, 0, 1] = np.nan
    backend = _FakeBackend(clouds, system)

    result = DispersionAcquisitionStrategy(_cfg()).select(
        pool=pool,
        probability_backend=backend,
        threshold_backend=None,
        threshold_state=_threshold_state(),
        target_count=3,
    )

    assert 101 not in result.d2_indices
    assert result.diagnostics["n_nonfinite_excluded"] == 1


def test_rejects_backend_without_sample_endpoints():
    class _NoCloudBackend:
        system = _FakeSystem()
        device = "cpu"

    with pytest.raises(RuntimeError, match="sample_endpoints"):
        DispersionAcquisitionStrategy(_cfg()).select(
            pool=_FakePool(np.zeros((2, 2))),
            probability_backend=_NoCloudBackend(),
            threshold_backend=None,
            threshold_state=_threshold_state(),
            target_count=1,
        )


def test_proportional_rule_runs_and_is_seeded():
    system = _FakeSystem()
    states = np.random.default_rng(0).normal(size=(20, 2))
    pool = _FakePool(states)
    spreads = np.linspace(0.1, 8.0, 20)
    backend = _FakeBackend(_clouds_with_spreads(spreads), system)
    cfg = _cfg(selection_rule="proportional", seed=11)

    a = DispersionAcquisitionStrategy(cfg).select(
        pool=pool, probability_backend=backend, threshold_backend=None,
        threshold_state=_threshold_state(), target_count=5,
    )
    b = DispersionAcquisitionStrategy(cfg).select(
        pool=pool, probability_backend=backend, threshold_backend=None,
        threshold_state=_threshold_state(), target_count=5,
    )
    assert a.d2_indices == b.d2_indices
    assert len(set(a.d2_indices)) == 5


def test_greedy_diverse_rule_runs():
    system = _FakeSystem()
    states = np.random.default_rng(1).normal(size=(30, 2))
    pool = _FakePool(states)
    spreads = np.linspace(0.1, 8.0, 30)
    backend = _FakeBackend(_clouds_with_spreads(spreads), system)

    result = DispersionAcquisitionStrategy(_cfg(selection_rule="greedy_diverse")).select(
        pool=pool, probability_backend=backend, threshold_backend=None,
        threshold_state=_threshold_state(), target_count=6,
    )

    assert len(set(result.d2_indices)) == 6
    assert result.diagnostics["selection_rule"] == "greedy_diverse"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/adaptive_v2/test_dispersion_strategy.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'adaptive_roa.adaptive_v2.strategy.dispersion'`

- [ ] **Step 3: Write the strategy**

Create `adaptive_roa/adaptive_v2/strategy/dispersion.py`:

```python
"""Dispersion acquisition strategy (D2 only).

Scores candidates by the spread of their predicted final states rather than by
label counts, so selection never consults the success criteria. See
docs/superpowers/specs/2026-07-26-final-state-dispersion-acquisition-design.md
"""

from __future__ import annotations

from typing import Any

import numpy as np

from adaptive_roa.adaptive_v2.strategy.dispersion_score import (
    mean_pairwise_dispersion,
    select_greedy,
    select_greedy_diverse,
    select_proportional,
)
from adaptive_roa.adaptive_v2.types import AcquisitionResult, ThresholdState

_SELECTION_RULES = ("greedy", "greedy_diverse", "proportional")


class DispersionAcquisitionStrategy:
    mode = "dispersion"

    def __init__(self, cfg: Any):
        self.d2_ratio = float(cfg.d2_ratio)
        self.n_dispersion_candidates = int(cfg.n_dispersion_candidates)
        self.num_mc_samples_dispersion = int(cfg.num_mc_samples_dispersion)
        self.selection_rule = str(cfg.selection_rule)
        self.diversity_pool_multiplier = int(cfg.diversity_pool_multiplier)
        self.temperature = float(cfg.temperature)
        self.seed = None if cfg.seed is None else int(cfg.seed)
        self.chunk_size = int(cfg.chunk_size)
        self.log_score_correlation = bool(cfg.log_score_correlation)
        self.verbose = bool(cfg.verbose)

        if self.selection_rule not in _SELECTION_RULES:
            raise ValueError(
                f"selection_rule must be one of {_SELECTION_RULES}, "
                f"got {self.selection_rule!r}"
            )

    def select(
        self,
        pool: Any,
        probability_backend: Any,
        threshold_backend: Any,
        threshold_state: ThresholdState,
        target_count: int,
        exclude: set[int] | None = None,
    ) -> AcquisitionResult:
        # threshold_state is accepted for Protocol conformance and never read:
        # selection is independent of the success criteria by design.
        if target_count <= 0:
            return self._skip("target_count_zero")

        if not hasattr(probability_backend, "sample_endpoints"):
            raise RuntimeError(
                "DispersionAcquisitionStrategy requires a probability backend with "
                f"sample_endpoints(); got {type(probability_backend).__name__}"
            )

        system = getattr(probability_backend, "system", None)
        if system is None:
            raise RuntimeError(
                "DispersionAcquisitionStrategy requires probability_backend.system "
                "to compute the normalized state metric"
            )

        scales = system.get_normalization_scales().cpu().numpy().astype(np.float64)
        circular_mask = np.zeros(len(scales), dtype=bool)
        circular_indices = system.get_circular_indices()
        if circular_indices:
            circular_mask[np.asarray(circular_indices, dtype=int)] = True

        states, indices = pool.sample_candidates_without_marking(
            self.n_dispersion_candidates, exclude=exclude
        )
        n_actual = len(indices)
        if n_actual == 0:
            if self.verbose:
                print("    [Dispersion] Pool exhausted. No candidates available.")
            return self._skip("pool_exhausted")

        if self.verbose:
            print(
                f"    [Dispersion] Evaluating {n_actual} candidates with "
                f"K={self.num_mc_samples_dispersion}, rule={self.selection_rule}, "
                f"selecting {target_count}"
            )

        endpoints = probability_backend.sample_endpoints(
            states,
            num_samples=self.num_mc_samples_dispersion,
            verbose=self.verbose,
        )
        scores = mean_pairwise_dispersion(
            endpoints,
            scales,
            circular_mask,
            chunk_size=self.chunk_size,
            device=getattr(probability_backend, "device", "cpu"),
        )

        positions = self._apply_selection_rule(
            scores, np.asarray(states), scales, circular_mask, target_count
        )
        selected_indices = [indices[int(p)] for p in positions]

        diagnostics = self._build_diagnostics(scores, positions, n_actual)

        if self.verbose:
            print(
                f"    Selected {len(selected_indices)}/{n_actual}; "
                f"score min={diagnostics['dispersion_score_min']}, "
                f"max={diagnostics['dispersion_score_max']}, "
                f"threshold={diagnostics['dispersion_score_threshold']}"
            )

        return AcquisitionResult(
            d1_indices=[],
            d2_indices=selected_indices,
            n_candidates_evaluated=n_actual,
            n_certain_discarded=n_actual - len(selected_indices),
            n_invalid_added=0,
            diagnostics=diagnostics,
        )

    def _apply_selection_rule(
        self,
        scores: np.ndarray,
        states: np.ndarray,
        scales: np.ndarray,
        circular_mask: np.ndarray,
        target_count: int,
    ) -> np.ndarray:
        if self.selection_rule == "greedy":
            return select_greedy(scores, target_count)
        if self.selection_rule == "greedy_diverse":
            return select_greedy_diverse(
                scores,
                states,
                scales,
                circular_mask,
                target_count,
                pool_multiplier=self.diversity_pool_multiplier,
            )
        return select_proportional(
            scores, target_count, temperature=self.temperature, seed=self.seed
        )

    def _build_diagnostics(
        self,
        scores: np.ndarray,
        positions: np.ndarray,
        n_actual: int,
    ) -> dict[str, Any]:
        finite = scores[np.isfinite(scores)]
        selected = scores[positions] if len(positions) else np.array([])
        return {
            "dispersion_score_threshold": float(selected.min()) if len(selected) else None,
            "dispersion_score_min": float(finite.min()) if len(finite) else None,
            "dispersion_score_max": float(finite.max()) if len(finite) else None,
            "dispersion_score_mean": float(finite.mean()) if len(finite) else None,
            "dispersion_score_median": float(np.median(finite)) if len(finite) else None,
            "n_dispersion_candidates_evaluated": int(n_actual),
            "n_nonfinite_excluded": int((~np.isfinite(scores)).sum()),
            "selection_rule": self.selection_rule,
            "num_mc_samples_dispersion": self.num_mc_samples_dispersion,
        }

    @staticmethod
    def _skip(reason: str) -> AcquisitionResult:
        return AcquisitionResult(
            d1_indices=[],
            d2_indices=[],
            n_candidates_evaluated=0,
            n_certain_discarded=0,
            n_invalid_added=0,
            diagnostics={"skipped_reason": reason},
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/adaptive_v2/test_dispersion_strategy.py -v`
Expected: 12 passed

- [ ] **Step 5: Create the config**

Create `configs/adaptive_v2/acquisition/dispersion.yaml`:

```yaml
# @package _global_
sampling_mode: dispersion
acquisition:
  _target_: adaptive_roa.adaptive_v2.strategy.dispersion.DispersionAcquisitionStrategy
  d2_ratio: 0.5
  n_dispersion_candidates: 50000
  num_mc_samples_dispersion: 20
  selection_rule: greedy           # greedy | greedy_diverse | proportional
  diversity_pool_multiplier: 5     # greedy_diverse only
  temperature: 0.1                 # proportional only
  seed: null                       # proportional only
  chunk_size: 2048
  log_score_correlation: true
  verbose: true
```

Note the deliberate omissions versus the other four acquisition configs:
`decision_rule` (the engine reads only `.d2_ratio` and `.mode` off the strategy,
`engine.py:109,111`), and `batch_size_sampling` / `max_samples_per_epoch` (consumed only by
`UncertainSampler`, which this strategy does not use).

- [ ] **Step 6: Verify the config instantiates**

Append to `tests/adaptive_v2/test_dispersion_strategy.py`:

```python
def test_shipped_config_instantiates():
    from hydra.utils import get_class

    cfg = OmegaConf.load("configs/adaptive_v2/acquisition/dispersion.yaml")
    assert cfg.sampling_mode == "dispersion"

    node = cfg.acquisition
    strategy = get_class(node._target_)(node)   # mirrors engine.py:28 _instantiate

    assert strategy.mode == "dispersion"
    assert strategy.num_mc_samples_dispersion == 20
    assert strategy.selection_rule == "greedy"
    assert strategy.log_score_correlation is True
    assert strategy.seed is None
```

- [ ] **Step 7: Run the full test file**

Run: `python -m pytest tests/adaptive_v2/test_dispersion_strategy.py -v`
Expected: 13 passed

- [ ] **Step 8: Commit**

```bash
git add adaptive_roa/adaptive_v2/strategy/dispersion.py configs/adaptive_v2/acquisition/dispersion.yaml tests/adaptive_v2/test_dispersion_strategy.py
git commit -m "feat(adaptive_v2): dispersion acquisition strategy scoring endpoint spread"
```

---

### Task 6: Score-correlation diagnostic

**Files:**
- Modify: `adaptive_roa/adaptive_v2/strategy/dispersion.py`
- Test: `tests/adaptive_v2/test_dispersion_strategy.py` (append)

**Interfaces:**
- Consumes: `DispersionAcquisitionStrategy` (Task 5); `system.classify_attractor(state, radius)`; `probability_backend.attractor_radius`.
- Produces: `dispersion_label_uncertainty_spearman` in `AcquisitionResult.diagnostics` (a float, or `None` when disabled or undefined).

**Context:** This costs **zero extra forward passes** — the strategy already holds the cloud `[M,K,D]`, so classifying those same endpoints yields `p_success` directly. It also uses K=20 rather than `estimate()`'s K=10, so the correlation is measured against a better `p_success` than the `ranked` baseline itself consumes.

The correlation is `Spearman(dispersion, u)` where `u = −|p_success − 0.5|` is label-based uncertainty: `u` is maximal (0) at `p_success = 0.5` and minimal (−0.5) when the counts are unanimous. Both quantities rise with uncertainty, so **+1.0 means the two scores are redundant**.

This is the only place in the acquisition path that touches the success criteria, and its output must never influence selection — it is computed after `positions` is already decided.

- [ ] **Step 1: Write the failing test**

Append to `tests/adaptive_v2/test_dispersion_strategy.py`:

```python
def test_correlation_diagnostic_absent_when_disabled():
    system = _FakeSystem()
    pool = _FakePool(np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]))
    backend = _FakeBackend(_clouds_with_spreads([1.0, 8.0, 4.0]), system)

    result = DispersionAcquisitionStrategy(_cfg(log_score_correlation=False)).select(
        pool=pool, probability_backend=backend, threshold_backend=None,
        threshold_state=_threshold_state(), target_count=2,
    )

    assert result.diagnostics["dispersion_label_uncertainty_spearman"] is None


def test_correlation_diagnostic_computed_when_enabled():
    system = _FakeSystem()
    rng = np.random.default_rng(4)
    n = 25
    pool = _FakePool(rng.normal(size=(n, 2)))
    # Clouds whose theta values straddle the success/failure boundary by varying
    # amounts, so both dispersion and p_success vary across candidates.
    clouds = np.stack(
        [
            np.stack(
                [
                    np.array([0.0, 0.0]),
                    np.array([float(i) / n * 2.0, 0.0]),
                ]
            )
            for i in range(n)
        ]
    ).astype(np.float32)
    backend = _FakeBackend(clouds, system)

    result = DispersionAcquisitionStrategy(_cfg(log_score_correlation=True)).select(
        pool=pool, probability_backend=backend, threshold_backend=None,
        threshold_state=_threshold_state(), target_count=5,
    )

    rho = result.diagnostics["dispersion_label_uncertainty_spearman"]
    assert rho is not None
    assert -1.0 <= rho <= 1.0


def test_correlation_does_not_change_selection():
    system = _FakeSystem()
    states = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    clouds = _clouds_with_spreads([0.1, 8.0, 0.2, 4.0])

    off = DispersionAcquisitionStrategy(_cfg(log_score_correlation=False)).select(
        pool=_FakePool(states), probability_backend=_FakeBackend(clouds, system),
        threshold_backend=None, threshold_state=_threshold_state(), target_count=2,
    )
    on = DispersionAcquisitionStrategy(_cfg(log_score_correlation=True)).select(
        pool=_FakePool(states), probability_backend=_FakeBackend(clouds, system),
        threshold_backend=None, threshold_state=_threshold_state(), target_count=2,
    )

    assert off.d2_indices == on.d2_indices


def test_correlation_is_none_when_p_success_is_constant():
    system = _FakeSystem()
    n = 10
    pool = _FakePool(np.zeros((n, 2)))
    # Every endpoint has theta = 5.0 -> all classified failure -> p_success
    # constant -> Spearman undefined.
    clouds = np.stack(
        [np.array([[5.0, 0.0], [5.0, float(i)]]) for i in range(n)]
    ).astype(np.float32)
    backend = _FakeBackend(clouds, system)

    result = DispersionAcquisitionStrategy(_cfg(log_score_correlation=True)).select(
        pool=pool, probability_backend=backend, threshold_backend=None,
        threshold_state=_threshold_state(), target_count=3,
    )

    assert result.diagnostics["dispersion_label_uncertainty_spearman"] is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/adaptive_v2/test_dispersion_strategy.py -k correlation -v`
Expected: FAIL with `KeyError: 'dispersion_label_uncertainty_spearman'`

- [ ] **Step 3: Implement the diagnostic**

In `adaptive_roa/adaptive_v2/strategy/dispersion.py`, add these imports at the top:

```python
import torch
from scipy.stats import spearmanr
```

Add this method to the class:

```python
    def _label_uncertainty_correlation(
        self,
        endpoints: np.ndarray,
        scores: np.ndarray,
        system: Any,
        probability_backend: Any,
    ) -> float | None:
        """Spearman correlation between dispersion and label-based uncertainty.

        Costs no extra forward passes: the endpoint cloud is already in hand, so
        p_success comes from classifying those same endpoints. Label-based
        uncertainty is u = -|p_success - 0.5|, which peaks at p_success = 0.5.
        Both quantities rise with uncertainty, so +1.0 means the dispersion score
        and the label counts are redundant.

        This is diagnostic only. It runs after selection is decided and never
        influences which candidates are chosen.

        Returns:
            float | None: Spearman rho, or None when it is undefined (fewer than
                two finite candidates, or either series is constant)
        """
        radius = getattr(probability_backend, "attractor_radius", None)
        if radius is None:
            return None

        M, K, D = endpoints.shape
        success_counts = np.zeros(M, dtype=np.float64)

        for start in range(0, M, self.chunk_size):
            stop = min(start + self.chunk_size, M)
            flat = torch.as_tensor(
                endpoints[start:stop].reshape(-1, D), dtype=torch.float32
            )
            labels = system.classify_attractor(flat, radius=radius)
            labels = labels.reshape(stop - start, K)
            success_counts[start:stop] = (labels == 1).sum(dim=1).cpu().numpy()

        p_success = success_counts / K
        u = -np.abs(p_success - 0.5)

        finite = np.isfinite(scores)
        if finite.sum() < 2:
            return None
        s, u = scores[finite], u[finite]
        if np.ptp(s) == 0 or np.ptp(u) == 0:
            return None

        rho = spearmanr(s, u).statistic
        return None if not np.isfinite(rho) else float(rho)
```

- [ ] **Step 4: Wire it into `select()`**

In `select()`, replace the single line:

```python
        diagnostics = self._build_diagnostics(scores, positions, n_actual)
```

with:

```python
        diagnostics = self._build_diagnostics(scores, positions, n_actual)
        # Computed after selection is decided: diagnostic only, never an input.
        diagnostics["dispersion_label_uncertainty_spearman"] = (
            self._label_uncertainty_correlation(
                endpoints, scores, system, probability_backend
            )
            if self.log_score_correlation
            else None
        )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/adaptive_v2/test_dispersion_strategy.py -v`
Expected: 17 passed

- [ ] **Step 6: Commit**

```bash
git add adaptive_roa/adaptive_v2/strategy/dispersion.py tests/adaptive_v2/test_dispersion_strategy.py
git commit -m "feat(adaptive_v2): log dispersion vs label-uncertainty correlation"
```

---

### Task 7: Full-suite regression check

**Files:**
- Test: existing `tests/adaptive_v2/`

**Interfaces:**
- Consumes: everything from Tasks 1-6.
- Produces: nothing new.

**Context:** Tasks 1 and 4 modify shared code (`systems/base.py`, `probability_estimator.py`, `endpoint_mc.py`) that the other strategies and the classifier path depend on. This task confirms nothing regressed.

- [ ] **Step 1: Run the adaptive_v2 suite**

Run: `python -m pytest tests/adaptive_v2/ -v`
Expected: all pass, including the pre-existing `test_acquisition_strategies.py`, `test_probability_backends.py`, `test_calibration_backend.py`, `test_e2e_conformal.py`, `test_evaluator.py`

- [ ] **Step 2: Run the wider suite**

Run: `python -m pytest tests/ -x -q`
Expected: no new failures. Some tests require dataset files and may already be failing or skipping on this machine — compare against `git stash && python -m pytest tests/ -x -q` output if anything looks suspicious, and only investigate failures that this branch introduced.

- [ ] **Step 3: Confirm the other acquisition configs still resolve**

Run:
```bash
python -c "
from hydra.utils import get_class
from omegaconf import OmegaConf
for name in ['ranked', 'direct', 'conformal', 'dispersion']:
    cfg = OmegaConf.load(f'configs/adaptive_v2/acquisition/{name}.yaml')
    print(name, cfg.sampling_mode, cfg.acquisition._target_)
"
```
Expected: four lines printed, each with a matching `sampling_mode` and a resolvable target. Note `ranked`/`direct`/`conformal` interpolate `${decision_rule}`, which is unresolvable standalone — printing `_target_` and `sampling_mode` avoids touching it.

- [ ] **Step 4: Commit if anything needed fixing**

```bash
git add -A
git commit -m "test: verify dispersion additions leave existing strategies intact"
```

---

## Validation (manual, after implementation)

Not a task — run this once the plan is complete, per the spec's validation section.

Pendulum first (2-D, cheapest, existing `ranked`/`direct` baselines):

```bash
python scripts/run_adaptive.py system=pendulum acquisition=dispersion
```

`scripts/run_adaptive.py` is the entry point (`@hydra.main(config_path="../configs/adaptive_v2",
config_name="default")`), and `acquisition` is a defaults group in
`configs/adaptive_v2/default.yaml:7`, so `acquisition=dispersion` selects the new config. Outputs
land under a `..._sampling_mode_dispersion/` directory, so they will not collide with `ranked`
baselines.

Read `dispersion_label_uncertainty_spearman` from each epoch's `artifacts_v2.json`
(`acquisition.diagnostics`) before waiting for the full comparison: near **+1.0** means dispersion
and the NC score are redundant and the interesting question dies early; well short of that means
dispersion is finding points the label counts cannot see.

Then compare full-ROA eval metrics against a matched `acquisition=ranked` run with the same pool,
seed, and epoch count. Note the compute asymmetry: at `num_mc_samples_dispersion=20` the dispersion
run spends 2× the forward passes per epoch that `ranked` does. If dispersion wins, rerun at
`num_mc_samples_dispersion=10` to check whether the win survives a compute-matched budget.
