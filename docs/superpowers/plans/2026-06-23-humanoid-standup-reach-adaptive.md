# HumanoidStandUpReach Adaptive + Eval Implementation Plan (Plan 2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the `humanoid_standup_reach` system into ROA evaluation and the `adaptive_v2` conformal/adaptive-sampling pipeline, including intermediate-state acquisition (candidate = sub-trajectory start) with marking-based dedup.

**Architecture:** Mirror the `quadrotor3d` adaptive_v2 wiring (system config + `system_dims` + `_DATAMODULES` registration). Add a direct-file mode to the humanoid data module (the adaptive trainer feeds the pool's built endpoint files). Extend acquisition behind an opt-in `candidate_mode` flag using an opaque integer **candidate-id** abstraction, so the shared sampler/strategy/engine are untouched and existing systems keep their exact behavior.

**Tech Stack:** PyTorch Lightning, Hydra, the Facebook `flow_matching` library, the existing `adaptive_v2` engine, pytest.

**Spec:** `docs/superpowers/specs/2026-06-23-humanoid-standup-reach-design.md` (§5.4, §5.5)
**Depends on:** Plan 1 (core) — committed `0014708..9fb50f3` on `main`.

## Global Constraints

- **State 67-D**, sphere block dims 34:37. Model `output_dim=67`. System/flow-matcher/data-module classes from Plan 1: `adaptive_roa.systems.humanoid_standup_reach.HumanoidStandUpReachSystem`, `adaptive_roa.flow_matching.humanoid_standup_reach.latent_conditional.flow_matcher.HumanoidStandUpReachLatentConditionalFlowMatcher`, `adaptive_roa.data.humanoid_standup_reach_endpoint_data.HumanoidStandUpReachEndpointDataModule`.
- **Dataset dir** `${data_dir}/humanoid_get_up_medium` with `dataset_description.json`, `trajectories/sequence_{i}.txt` (comma/`", "`), `train_test_splits/{all_shuffled_indices,all_shuffled_labels,cal_set_fps,test_set_fps,cal_set_start_states,test_set_start_states,eval_fps,eval_start_states}.txt`.
- **Eval sets: FPS primary** — `cal_set_file → cal_set_fps.txt`, `test_set_file → test_set_fps.txt`; `*_start_states.txt` are the commented alternative.
- **Trajectory load: comma delimiter, RAISE on failure (never zero-fill).**
- **`candidate_mode` flag**: `start` (default, existing behavior for ALL systems) | `intermediate` (humanoid). Candidate-id stays an integer everywhere outside `AdaptiveDatasetBuilder`. In `intermediate` mode an id maps to `(traj_idx, row)`; marking id `(i,t)` marks the whole tail `(i, r≥t)`; adding it contributes the unmarked tail rows `[t..end-1] → final`. `start` mode behavior must be byte-identical to today (verified by backward-compat tests for pendulum/cartpole/quadrotor).
- **`prediction_mode=global`** (no trajectory/`flow_matcher_local` needed).
- **Tests** in `tests/`, run with `/common/home/st1122/Projects/adaptive_roa/env/bin/python -m pytest`. Full suite must stay green (Plan 1 = 40 tests).
- **Git**: branch `main`. Commit per task. No `Co-Authored-By`. End messages with `Claude-Session: https://claude.ai/code/session_01NZMXZPhQUpzSKYDqNqypoK`.

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `configs/evaluate_humanoid_standup_reach_roa.yaml` | create | ROA eval config |
| `adaptive_roa/data/humanoid_standup_reach_endpoint_data.py` | modify | add direct-file mode |
| `adaptive_roa/adaptive_v2/trainers/flow_matching_trainer.py` | modify | register in `_DATAMODULES` |
| `configs/adaptive_v2/system/humanoid_standup_reach.yaml` | create | adaptive system config |
| `configs/adaptive_v2/model/system_dims/humanoid_standup_reach.yaml` | create | model dims |
| `adaptive_roa/adaptive/data_source.py` | modify | `(traj,row)` state/length helpers |
| `adaptive_roa/adaptive/dataset_builder.py` | modify | `candidate_mode`, candidate-id mapping, marking dedup, tail expansion |
| `adaptive_roa/adaptive_v2/pool/trajectory_pool.py` | modify | pass `candidate_mode` through |
| `tests/test_humanoid_standup_reach_eval_config.py` | create | Task 1 |
| `tests/test_humanoid_standup_reach_datamodule_directfile.py` | create | Task 2 |
| `tests/test_humanoid_standup_reach_adaptive_config.py` | create | Task 3 |
| `tests/test_intermediate_acquisition.py` | create | Task 4 (incl. backward-compat) |
| `tests/test_humanoid_standup_reach_adaptive_smoke.py` | create | Task 5 |

---

### Task 1: ROA evaluation config

**Files:**
- Create: `configs/evaluate_humanoid_standup_reach_roa.yaml`
- Test: `tests/test_humanoid_standup_reach_eval_config.py`

**Interfaces:**
- Consumes: `evaluate_roa.py` config contract (`system.module`/`system.class`, `checkpoint.*`, `data.file`, `evaluation.*`, `output.*`); Plan 1 system's `is_in_attractor`/`classify_attractor`.
- Produces: a composable eval config; confidence that the 135-col humanoid eval file loads and classifies without pendulum/cartpole angle-wrapping.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_humanoid_standup_reach_eval_config.py
from pathlib import Path
import numpy as np
import torch
import pytest
from omegaconf import OmegaConf
import yaml

REPO = Path(__file__).resolve().parents[1]
DATASET_DIR = "/common/users/shared/pracsys/genMoPlan/data_trajectories/humanoid_get_up_medium"
pytestmark = pytest.mark.skipif(not Path(DATASET_DIR).exists(), reason="shared humanoid dataset not available")


def test_eval_config_targets_humanoid_flow_matcher():
    cfg = yaml.safe_load((REPO / "configs/evaluate_humanoid_standup_reach_roa.yaml").read_text())
    assert cfg["system"]["module"] == "adaptive_roa.flow_matching.humanoid_standup_reach.latent_conditional.flow_matcher"
    assert cfg["system"]["class"] == "HumanoidStandUpReachLatentConditionalFlowMatcher"
    assert cfg["checkpoint"]["base_name"] == "humanoid_standup_reach_latent_conditional_fm"
    # 67-D state → batch sized down; attractor_radius present (unused by humanoid classify but required by API)
    assert cfg["evaluation"]["num_steps"] >= 1


def test_eval_file_rows_classify_without_angle_wrapping():
    # The fps test set is (state[67], final[67], label) comma-delimited, 135 cols.
    fps = Path(DATASET_DIR) / "train_test_splits/test_set_fps.txt"
    data = np.loadtxt(fps, delimiter=",", max_rows=16)
    assert data.shape[1] == 135
    states = torch.tensor(data[:, :67], dtype=torch.float32)
    from adaptive_roa.systems.humanoid_standup_reach import HumanoidStandUpReachSystem
    system = HumanoidStandUpReachSystem(dataset_dir=DATASET_DIR)
    labels = system.classify_attractor(states)
    assert labels.shape == (16,)
    assert set(labels.tolist()) <= {1, -1}
```

- [ ] **Step 2: Run to verify it fails**

Run: `/common/home/st1122/Projects/adaptive_roa/env/bin/python -m pytest tests/test_humanoid_standup_reach_eval_config.py -v`
Expected: FAIL — config file missing.

- [ ] **Step 3: Create the config**

```yaml
# configs/evaluate_humanoid_standup_reach_roa.yaml
# Humanoid StandUpReach ROA evaluation (67-D, ℝ³⁴ × S² × ℝ³⁰).
data_dir: ${data_dir:}

name: humanoid_standup_reach_roa_evaluation
seed: 42

system:
  module: adaptive_roa.flow_matching.humanoid_standup_reach.latent_conditional.flow_matcher
  class: HumanoidStandUpReachLatentConditionalFlowMatcher

checkpoint:
  path: null
  auto_find: true
  base_name: humanoid_standup_reach_latent_conditional_fm

data:
  # FPS test set is primary; start_states alternative is commented.
  file: ${data_dir}/humanoid_get_up_medium/train_test_splits/test_set_fps.txt
  # file: ${data_dir}/humanoid_get_up_medium/train_test_splits/test_set_start_states.txt
  format: "67D start_state, 67D final_state, label"  # comma-separated, 135 cols

evaluation:
  batch_size: 512        # lower due to 67-D
  num_steps: 100
  attractor_radius: 1.0  # accepted by API; humanoid classify ignores it (uses head_height + CoM speed)
  probabilistic: true
  num_samples: 20
  confidence_threshold: 0.6

output:
  dir: humanoid_standup_reach_roa_evaluation
  save_plots: true
  save_data: true

hydra:
  run:
    dir: outputs/${name}/${now:%Y-%m-%d_%H-%M-%S}
```

- [ ] **Step 4: Run to verify it passes**

Run: `/common/home/st1122/Projects/adaptive_roa/env/bin/python -m pytest tests/test_humanoid_standup_reach_eval_config.py -v`
Expected: PASS. (If `evaluate_roa.py`'s data loader special-cases angle columns by state-dim and a 67-D path hits it, note it — but the test only exercises the loader format + classify, which are humanoid-side and safe. A full eval run needs a trained checkpoint and is the Optional follow-up below.)

- [ ] **Step 5: Commit**

```bash
git add configs/evaluate_humanoid_standup_reach_roa.yaml tests/test_humanoid_standup_reach_eval_config.py
git commit -m "feat: add HumanoidStandUpReach ROA evaluation config

Mirrors evaluate_cartpole_roa with humanoid flow-matcher routing, FPS-primary
eval file, 67-D batch sizing. Test verifies 135-col fps rows load and classify
(binary head-height/CoM) with no angle-wrapping.

Claude-Session: https://claude.ai/code/session_01NZMXZPhQUpzSKYDqNqypoK"
```

---

### Task 2: Data-module direct-file mode + `_DATAMODULES` registration

The adaptive_v2 trainer instantiates the data module with `data_file`/`validation_file`/`test_file` (the pool's built `[start(67) end(67)]` space-separated files), so the humanoid data module needs a direct-file mode and must be registered.

**Files:**
- Modify: `adaptive_roa/data/humanoid_standup_reach_endpoint_data.py`
- Modify: `adaptive_roa/adaptive_v2/trainers/flow_matching_trainer.py`
- Test: `tests/test_humanoid_standup_reach_datamodule_directfile.py`

**Interfaces:**
- Consumes: pool output format — `np.savetxt(hstack([starts, ends]), fmt='%.8f')` → space-separated, 134 cols (67+67, no label). The trainer also passes `data_file`/`validation_file`/`test_file`/`batch_size`/`val_batch_size`/`num_workers` (and may pass `dataset_dir`).
- Produces: `HumanoidStandUpReachEndpointDataModule(data_file=…, validation_file=…, test_file=…, batch_size=…, val_batch_size=…, num_workers=…, dataset_dir=None)` working in direct-file mode; `_DATAMODULES["humanoid_standup_reach"]` registered.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_humanoid_standup_reach_datamodule_directfile.py
import numpy as np
import torch
import pytest
from adaptive_roa.data.humanoid_standup_reach_endpoint_data import HumanoidStandUpReachEndpointDataModule


def _write_pairs(path, n, seed):
    rng = np.random.default_rng(seed)
    starts = rng.normal(size=(n, 67)); ends = rng.normal(size=(n, 67))
    np.savetxt(path, np.hstack([starts, ends]), fmt="%.8f")  # space-separated, 134 cols (pool format)


def test_direct_file_mode_loads_start_end(tmp_path):
    f = tmp_path / "train.txt"; _write_pairs(f, 8, 0)
    dm = HumanoidStandUpReachEndpointDataModule(
        data_file=str(f), validation_file=str(f), test_file=str(f),
        batch_size=4, num_workers=0)
    dm.setup("fit")
    item = dm.train_dataset[0]
    assert item["start_state"].shape == (67,) and item["end_state"].shape == (67,)
    assert torch.allclose(item["start_state"][34:37].norm(), torch.tensor(1.0), atol=1e-5)  # sphere unit-norm


def test_datamodule_registered_in_trainer():
    from adaptive_roa.adaptive_v2.trainers.flow_matching_trainer import _DATAMODULES
    assert _DATAMODULES["humanoid_standup_reach"] is HumanoidStandUpReachEndpointDataModule
```

- [ ] **Step 2: Run to verify it fails**

Run: `/common/home/st1122/Projects/adaptive_roa/env/bin/python -m pytest tests/test_humanoid_standup_reach_datamodule_directfile.py -v`
Expected: FAIL — `data_file` kwarg unexpected (TypeError) and `_DATAMODULES` has no humanoid key.

- [ ] **Step 3: Add direct-file mode + register**

In `adaptive_roa/data/humanoid_standup_reach_endpoint_data.py`:
- Add a `HumanoidStandUpReachDirectFileDataset(Dataset)` that loads a `np.loadtxt(path)` (whitespace; the pool writes space-separated) file, supporting 134 cols (`start[:67]`, `end[67:134]`) or 135 cols (drop the trailing label), unit-normalizing the sphere block of both via the existing `_unit_sphere` helper. RAISE on wrong column count.
- Extend `HumanoidStandUpReachEndpointDataModule.__init__` to accept `data_file=None, validation_file=None, test_file=None, dataset_dir=None` in addition to the existing shuffled-indices args. If `data_file` is provided, `setup` builds `HumanoidStandUpReachDirectFileDataset` for train/val/test; otherwise the existing shuffled-indices path. `dataset_dir` is accepted and ignored (API symmetry with the trainer).

```python
# Direct-file dataset (append to the data module file)
class HumanoidStandUpReachDirectFileDataset(Dataset):
    def __init__(self, data_file: str):
        data = np.loadtxt(data_file)  # pool writes space-separated
        if data.ndim == 1:
            data = data[None, :]
        n = data.shape[1]
        if n not in (134, 135):
            raise ValueError(f"{data_file}: expected 134 or 135 columns, got {n}")
        self.starts = data[:, :67].astype(np.float32)
        self.ends = data[:, 67:134].astype(np.float32)

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        return {
            "start_state": torch.from_numpy(_unit_sphere(self.starts[idx])),
            "end_state": torch.from_numpy(_unit_sphere(self.ends[idx])),
        }
```

DataModule `__init__` gains (keep existing params, all optional now):
```python
def __init__(self, train_indices_file: str = None, val_indices_file: str = None,
             test_indices_file: str = None, trajectories_dir: str = None,
             data_file: str = None, validation_file: str = None, test_file: str = None,
             query_mode: str = "random_intermediate", batch_size: int = 256,
             val_batch_size: Optional[int] = None, num_workers: int = 4, pin_memory: bool = True,
             dataset_dir: str = None, max_train_samples: Optional[int] = None,
             max_val_samples: Optional[int] = None):
    super().__init__()
    self.use_direct_file = data_file is not None
    # ... store all params ...
```
`setup` branches on `self.use_direct_file`:
```python
def setup(self, stage=None):
    if self.use_direct_file:
        if stage in ("fit", None):
            self.train_dataset = HumanoidStandUpReachDirectFileDataset(self.data_file)
            self.val_dataset = HumanoidStandUpReachDirectFileDataset(self.validation_file)
        if stage in ("test", None):
            self.test_dataset = HumanoidStandUpReachDirectFileDataset(self.test_file)
    else:
        # ... existing shuffled-indices setup ...
```

In `adaptive_roa/adaptive_v2/trainers/flow_matching_trainer.py`:
- Import `HumanoidStandUpReachEndpointDataModule` and add `"humanoid_standup_reach": HumanoidStandUpReachEndpointDataModule` to `_DATAMODULES`.
- If the trainer special-cases `dataset_dir` for `{"quadrotor3d"}`, add `"humanoid_standup_reach"` to that set (it accepts/ignores `dataset_dir`).

- [ ] **Step 4: Run to verify it passes**

Run: `/common/home/st1122/Projects/adaptive_roa/env/bin/python -m pytest tests/test_humanoid_standup_reach_datamodule_directfile.py -v` then the full suite `-q`.
Expected: both new tests pass; suite stays green.

- [ ] **Step 5: Commit**

```bash
git add adaptive_roa/data/humanoid_standup_reach_endpoint_data.py adaptive_roa/adaptive_v2/trainers/flow_matching_trainer.py tests/test_humanoid_standup_reach_datamodule_directfile.py
git commit -m "feat: HumanoidStandUpReach data-module direct-file mode + trainer registration

Adds a direct-file dataset (pool's space-separated [start end] files) and
registers humanoid_standup_reach in adaptive_v2 _DATAMODULES.

Claude-Session: https://claude.ai/code/session_01NZMXZPhQUpzSKYDqNqypoK"
```

---

### Task 3: adaptive_v2 system + model-dims configs

**Files:**
- Create: `configs/adaptive_v2/system/humanoid_standup_reach.yaml`
- Create: `configs/adaptive_v2/model/system_dims/humanoid_standup_reach.yaml`
- Test: `tests/test_humanoid_standup_reach_adaptive_config.py`

**Interfaces:**
- Consumes: `configs/adaptive_v2/system/_base.yaml`, the `_DATAMODULES` registration (Task 2), `system_dims` package convention.
- Produces: `python scripts/run_adaptive.py system=humanoid_standup_reach` composes; `data_source` points at humanoid trajectories + FPS cal/test; `adaptive_v2.system_name == "humanoid_standup_reach"`; `candidate_mode: intermediate`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_humanoid_standup_reach_adaptive_config.py
from pathlib import Path
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf
import pytest
from adaptive_roa.utils.env_config import get_net_id, get_exp_dir, get_shared_data_base

CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "configs/adaptive_v2")


def _resolvers():
    import os
    for name, fn in [("net_id", get_net_id), ("exp_dir", get_exp_dir), ("shared_data_base", get_shared_data_base)]:
        if not OmegaConf.has_resolver(name):
            OmegaConf.register_new_resolver(name, (lambda f: (lambda default="": f() or default))(fn))
    if not OmegaConf.has_resolver("data_dir"):
        OmegaConf.register_new_resolver("data_dir", lambda default="": os.environ.get("DATA_DIR", default) or default)


def test_adaptive_config_composes():
    _resolvers()
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(config_name="default", overrides=["system=humanoid_standup_reach"])
    assert cfg.adaptive_v2.system_name == "humanoid_standup_reach"
    assert cfg.system._target_ == "adaptive_roa.systems.humanoid_standup_reach.HumanoidStandUpReachSystem"
    assert cfg.flow_matcher._target_.endswith("HumanoidStandUpReachLatentConditionalFlowMatcher")
    assert str(cfg.data_source.test_set_file).endswith("test_set_fps.txt")
    assert str(cfg.data_source.cal_set_file).endswith("cal_set_fps.txt")
    assert cfg.get("candidate_mode") == "intermediate"
    assert int(cfg.model_dims.output_dim) == 67
```

- [ ] **Step 2: Run to verify it fails**

Run: `/common/home/st1122/Projects/adaptive_roa/env/bin/python -m pytest tests/test_humanoid_standup_reach_adaptive_config.py -v`
Expected: FAIL — `system=humanoid_standup_reach` not found.

- [ ] **Step 3: Create the configs**

```yaml
# configs/adaptive_v2/model/system_dims/humanoid_standup_reach.yaml
# @package model_dims
embedded_dim: 67
condition_dim: 67
output_dim: 67
time_emb_dim: 64

unet_target: adaptive_roa.model.quadrotor3d_unet.Quadrotor3DUNet
unet_hidden_dims: [512, 1024, 1024, 512]
mlp_hidden_dims: [512, 1024, 1024, 512]

adaln_hidden_dim: 512
adaln_num_blocks: 8
adaln_cond_dim: 256
adaln_mlp_ratio: 2
adaln_dropout: 0.0

dit_hidden_dim: 256
dit_num_blocks: 6
dit_num_heads: 4
dit_cond_dim: 256
dit_mlp_ratio: 4
dit_dropout: 0.0

use_input_embeddings: false
input_emb_dim: 128
```

```yaml
# configs/adaptive_v2/system/humanoid_standup_reach.yaml
# @package _global_
defaults:
  - _base
  - /model/system_dims: humanoid_standup_reach
  - _self_

name: adaptive_humanoid_standup_reach
output_dir: ${exp_dir}/${name}/outputs/training_index_${training_index}_d2_ratio_${d2_ratio}_warm_start_${warm_start}_manifold_${use_manifold}_threshold_mode_${threshold_mode}_adapt_iter_${n_epochs}_alpha_${alpha_sampling}_sampling_mode_${sampling_mode}/${now:%Y-%m-%d_%H-%M-%S}

use_manifold: true

# Intermediate-state acquisition (candidate = sub-trajectory start)
candidate_mode: intermediate

data_source:
  base_dir: ${data_dir}/humanoid_get_up_medium
  trajectories_dir: ${data_source.base_dir}/trajectories
  train_test_splits_dir: ${data_source.base_dir}/train_test_splits
  shuffled_indices_file: ${data_source.train_test_splits_dir}/all_shuffled_indices.txt
  shuffled_labels_file: ${data_source.train_test_splits_dir}/all_shuffled_labels.txt
  # FPS primary; start_states alternative commented
  eval_states_file: ${data_source.train_test_splits_dir}/eval_fps.txt
  cal_set_file: ${data_source.train_test_splits_dir}/cal_set_fps.txt
  test_set_file: ${data_source.train_test_splits_dir}/test_set_fps.txt
  # eval_states_file: ${data_source.train_test_splits_dir}/eval_start_states.txt
  # cal_set_file: ${data_source.train_test_splits_dir}/cal_set_start_states.txt
  # test_set_file: ${data_source.train_test_splits_dir}/test_set_start_states.txt

initial_train_size: 2000
n_epochs: 30
samples_per_epoch: 500

conformal:
  decision_rule: two_sided
  attractor_radius: 1.0

optimizer:
  lr: 5e-4

system:
  _target_: adaptive_roa.systems.humanoid_standup_reach.HumanoidStandUpReachSystem
  dataset_dir: ${data_dir}/humanoid_get_up_medium

flow_matcher:
  _target_: adaptive_roa.flow_matching.humanoid_standup_reach.latent_conditional.flow_matcher.HumanoidStandUpReachLatentConditionalFlowMatcher

adaptive_v2:
  system_name: humanoid_standup_reach
```

Note: no `flow_matcher_local` (global prediction mode only). If `_base.yaml` defaults `use_manifold: false`, the explicit `use_manifold: true` here enables the S² manifold.

- [ ] **Step 4: Run to verify it passes**

Run: `/common/home/st1122/Projects/adaptive_roa/env/bin/python -m pytest tests/test_humanoid_standup_reach_adaptive_config.py -v`
Expected: PASS. (If `model_dims` package access differs, mirror exactly how `configs/adaptive_v2/system/quadrotor3d.yaml` + `system_dims/quadrotor3d.yaml` expose it.)

- [ ] **Step 5: Commit**

```bash
git add configs/adaptive_v2/system/humanoid_standup_reach.yaml configs/adaptive_v2/model/system_dims/humanoid_standup_reach.yaml tests/test_humanoid_standup_reach_adaptive_config.py
git commit -m "feat: adaptive_v2 system + model-dims configs for humanoid_standup_reach

FPS-primary cal/test, candidate_mode=intermediate, output_dim 67, use_manifold.

Claude-Session: https://claude.ai/code/session_01NZMXZPhQUpzSKYDqNqypoK"
```

---

### Task 4: Intermediate-state acquisition (candidate-id abstraction + marking dedup)

The core change. Keep candidates as opaque integers everywhere except `AdaptiveDatasetBuilder`. Add `candidate_mode` (`start` default, `intermediate` opt-in). In `intermediate` mode the builder maps candidate-ids → `(traj_idx, row)`, scores the row's state, and on add/mark applies the tail/marking semantics from spec §5.5. `start` mode is byte-identical to today. The shared `UncertainSampler`, strategies, and engine are NOT modified — they pass integer ids through untouched.

**Files:**
- Modify: `adaptive_roa/adaptive/data_source.py` — add `get_state_at(traj_idx, row)` and `get_trajectory_length(traj_idx)` (with an internal per-trajectory load cache to avoid repeated disk reads). Comma delimiter; raise on failure.
- Modify: `adaptive_roa/adaptive/dataset_builder.py` — `candidate_mode` + candidate-id registry + marking dedup + tail expansion.
- Modify: `adaptive_roa/adaptive_v2/pool/trajectory_pool.py` — pass `candidate_mode` from `data_source_cfg` (or top-level cfg) into `AdaptiveDatasetBuilder`.
- Test: `tests/test_intermediate_acquisition.py` (incl. backward-compat).

**Interfaces:**
- Consumes: `TrajectoryDataSource`, `AdaptiveDatasetBuilder` current methods.
- Produces (contract):
  - `AdaptiveDatasetBuilder(data_source, output_dir, val_ratio, test_ratio, candidate_mode="start")`.
  - `sample_candidates_without_marking(n, exclude)` → `(states[n,D], cand_ids[n])`. `start`: cand_id == traj_idx (unchanged). `intermediate`: cand_id indexes an `(i,row)` registry; `states[k]` = state at that row.
  - `mark_indices_as_used(cand_ids)` / `add_to_training_balanced(cand_ids)` interpret ids per mode. `intermediate`: marking `(i,t)` marks all registry ids with `traj==i, row≥t`; adding contributes unmarked tail rows `[t..end-1] → final`.
  - `get_training_data()` / `build_all_datasets()` expand per spec §5.5 in `intermediate` mode (tail per trajectory = min added row → final), unchanged in `start`.

- [ ] **Step 1: Write the failing tests (contract + backward-compat)**

```python
# tests/test_intermediate_acquisition.py
import numpy as np
import pytest
from pathlib import Path
from adaptive_roa.adaptive.data_source import TrajectoryDataSource, TrajectoryDataSourceConfig
from adaptive_roa.adaptive.dataset_builder import AdaptiveDatasetBuilder


def _make_pool(tmp_path, lengths, seed=0):
    """Build a tiny trajectories dir + indices/labels; return a TrajectoryDataSource."""
    traj_dir = tmp_path / "trajectories"; traj_dir.mkdir()
    rng = np.random.default_rng(seed)
    names = []
    for i, T in enumerate(lengths):
        rows = rng.normal(size=(T, 4)).astype(np.float64)
        np.savetxt(traj_dir / f"sequence_{i}.txt", rows, delimiter=", ", fmt="%.6f")
        names.append(f"sequence_{i}.txt")
    idx = tmp_path / "idx.txt"; idx.write_text("\n".join(names) + "\n")
    lab = tmp_path / "lab.txt"; lab.write_text("\n".join("1" for _ in names) + "\n")
    cfg = TrajectoryDataSourceConfig(trajectories_dir=str(traj_dir), shuffled_indices_file=str(idx),
                                     shuffled_labels_file=str(lab))
    return TrajectoryDataSource(cfg)


def test_start_mode_unchanged(tmp_path):
    ds = _make_pool(tmp_path, [5, 6, 7])
    b = AdaptiveDatasetBuilder(ds, str(tmp_path / "out_s"), candidate_mode="start")
    states, ids = b.sample_candidates_without_marking(2)
    assert len(ids) == 2 and states.shape == (2, 4)
    # ids are trajectory indices; their states are row 0
    assert np.allclose(states[0], ds.get_start_state(ids[0]))


def test_intermediate_candidate_is_subtrajectory_start(tmp_path):
    ds = _make_pool(tmp_path, [5])  # one trajectory of length 5 → rows 0..4 → candidate rows 0..3 (non-terminal)
    b = AdaptiveDatasetBuilder(ds, str(tmp_path / "out_i"), candidate_mode="intermediate")
    states, ids = b.sample_candidates_without_marking(4)
    assert len(ids) == 4  # 4 non-terminal rows of the single trajectory
    traj = ds.load_trajectory(0)
    for k, cid in enumerate(ids):
        i, r = b.candidate_to_traj_row(cid)
        assert np.allclose(states[k], traj[r])


def test_marking_prevents_overlap_and_tail_expansion(tmp_path):
    ds = _make_pool(tmp_path, [6])  # rows 0..5, final=row5
    b = AdaptiveDatasetBuilder(ds, str(tmp_path / "out_m"), candidate_mode="intermediate", val_ratio=0.0)
    # add candidate (traj 0, row 3): tail rows 3,4 -> final(5)
    cid_3 = b.traj_row_to_candidate(0, 3)
    b.add_to_training_balanced([cid_3])
    starts, ends, labels = b.get_training_data()
    assert starts.shape[0] == 2  # rows 3,4
    traj = ds.load_trajectory(0)
    assert np.allclose(np.sort(starts[:, 0]), np.sort(traj[3:5, 0]))
    assert np.allclose(ends, np.tile(traj[5], (2, 1)))
    # now add (traj 0, row 1): only NEW rows 1,2 are added (3,4 already marked)
    cid_1 = b.traj_row_to_candidate(0, 1)
    b.add_to_training_balanced([cid_1])
    starts2, ends2, _ = b.get_training_data()
    assert starts2.shape[0] == 4  # rows 1,2,3,4 (no duplication of 3,4)


def test_marked_tail_unavailable(tmp_path):
    ds = _make_pool(tmp_path, [6])
    b = AdaptiveDatasetBuilder(ds, str(tmp_path / "out_a"), candidate_mode="intermediate", val_ratio=0.0)
    cid_2 = b.traj_row_to_candidate(0, 2)
    b.mark_indices_as_used([cid_2])  # marks rows 2,3,4,5-start... (>=2)
    _, ids = b.sample_candidates_without_marking(10)
    rows = sorted(b.candidate_to_traj_row(c)[1] for c in ids)
    assert all(r < 2 for r in rows)  # only rows 0,1 remain available
```

Add a backward-compat guard that the existing trajectory-based pool path is unchanged. Since the start path is the default and other systems never set `candidate_mode`, the **existing adaptive_v2 tests (if any) plus the full suite must stay green** — run it in Step 4.

- [ ] **Step 2: Run to verify it fails**

Run: `/common/home/st1122/Projects/adaptive_roa/env/bin/python -m pytest tests/test_intermediate_acquisition.py -v`
Expected: FAIL — `candidate_mode`, `candidate_to_traj_row`, `traj_row_to_candidate` not implemented.

- [ ] **Step 3: Implement (read the current bodies first, then edit)**

In `adaptive_roa/adaptive/data_source.py`, add (use the existing comma-split loader; cache loaded trajectories in a dict to avoid repeated reads):
```python
def get_trajectory_length(self, idx: int) -> int:
    return len(self.load_trajectory(idx))

def get_state_at(self, idx: int, row: int) -> np.ndarray:
    return self.load_trajectory(idx)[row]
```
(If `load_trajectory` is not already cached, add an `lru_cache`-style dict keyed by idx so per-candidate access is cheap.)

In `adaptive_roa/adaptive/dataset_builder.py`:
- `__init__(..., candidate_mode: str = "start")`; store it; in `intermediate` build a candidate registry lazily: a list `self._candidates` of `(traj_idx, row)` for `row in range(get_trajectory_length(i) - 1)` over the available trajectories, with `candidate_id == index into self._candidates`. Maintain `self._cid_used: set[int]` and `self._added_min_row: dict[int,int]` (per-traj minimum added row).
- `traj_row_to_candidate(i, row) -> int` and `candidate_to_traj_row(cid) -> (i,row)` helpers.
- `sample_candidates_without_marking(n, exclude)`:
  - `start` mode: existing body unchanged.
  - `intermediate`: take the first `n` candidate-ids whose id ∉ `_cid_used` and ∉ `exclude`; states via `data_source.get_state_at`. Return `(states, ids)`.
- `mark_indices_as_used(ids)`:
  - `start`: existing.
  - `intermediate`: for each id=(i,t), add to `_cid_used` ALL candidate-ids with `traj==i and row>=t`.
- `add_to_training_balanced(ids)`:
  - `start`: existing.
  - `intermediate`: for each id=(i,t): set `_added_min_row[i] = min(existing, t)`; then `mark_indices_as_used([id])`.
- `get_training_data()` / `build_all_datasets()` / `save`-paths:
  - `start`: existing (whole-trajectory expansion).
  - `intermediate`: for each trajectory `i` in `_added_min_row`, expand `rows[_added_min_row[i] : -1] → final` (tail). Add a `start_row` parameter to `TrajectoryDataSource.get_all_endpoint_pairs_from_trajectory` (default 0 = current behavior) so the tail can begin at `_added_min_row[i]`.
- `get_val_labels()` / `get_test_labels()`: keep trajectory-level (start-state) semantics in both modes — out of scope for intermediate acquisition; they feed in-loop threshold optimization, not the FPS cal/test eval.

In `adaptive_roa/adaptive_v2/pool/trajectory_pool.py`: read `candidate_mode` from the cfg (top-level `cfg.get("candidate_mode", "start")`, plumbed from the engine which constructs `TrajectoryPool`) and pass it into `AdaptiveDatasetBuilder`. (Confirm where `TrajectoryPool` is built in `engine.py` and thread `candidate_mode` from `self.cfg`.)

- [ ] **Step 4: Run to verify it passes (incl. backward-compat)**

Run: `/common/home/st1122/Projects/adaptive_roa/env/bin/python -m pytest tests/test_intermediate_acquisition.py -v`
Then the FULL suite: `/common/home/st1122/Projects/adaptive_roa/env/bin/python -m pytest -q` — must stay green (proves `start` mode and other systems are unaffected).
Expected: new tests pass; suite green.

- [ ] **Step 5: Commit**

```bash
git add adaptive_roa/adaptive/data_source.py adaptive_roa/adaptive/dataset_builder.py adaptive_roa/adaptive_v2/pool/trajectory_pool.py tests/test_intermediate_acquisition.py
git commit -m "feat: intermediate-state acquisition via candidate-id + marking dedup

candidate_mode=intermediate maps opaque candidate-ids to (traj,row); marking a
candidate marks its whole tail, adding contributes unmarked tail rows -> final.
start mode (default) is byte-identical; sampler/strategy/engine untouched.

Claude-Session: https://claude.ai/code/session_01NZMXZPhQUpzSKYDqNqypoK"
```

---

### Task 5: End-to-end adaptive_v2 smoke test

**Files:**
- Test: `tests/test_humanoid_standup_reach_adaptive_smoke.py`

**Interfaces:**
- Consumes: everything above + `scripts/run_adaptive.py` / `AdaptiveEngine`.
- Produces: confidence the full loop runs with `candidate_mode=intermediate` + FPS cal/test on a tiny CPU configuration.

- [ ] **Step 1: Write the test**

```python
# tests/test_humanoid_standup_reach_adaptive_smoke.py
from pathlib import Path
import pytest
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf

DATASET_DIR = "/common/users/shared/pracsys/genMoPlan/data_trajectories/humanoid_get_up_medium"
pytestmark = pytest.mark.skipif(not Path(DATASET_DIR).exists(), reason="shared humanoid dataset not available")
CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "configs/adaptive_v2")


def test_adaptive_engine_one_epoch_cpu(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", "/common/users/shared/pracsys/genMoPlan/data_trajectories")
    from adaptive_roa.utils.env_config import get_net_id, get_exp_dir, get_shared_data_base
    import os
    for name, fn in [("net_id", get_net_id), ("exp_dir", get_exp_dir), ("shared_data_base", get_shared_data_base)]:
        if not OmegaConf.has_resolver(name):
            OmegaConf.register_new_resolver(name, (lambda f: (lambda d="": f() or d))(fn))
    if not OmegaConf.has_resolver("data_dir"):
        OmegaConf.register_new_resolver("data_dir", lambda d="": os.environ.get("DATA_DIR", d) or d)
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(config_name="default", overrides=[
            "system=humanoid_standup_reach",
            "device=cpu", "n_epochs=1", "initial_train_size=8",
            "samples_per_epoch=4", "batch_size=4", "val_batch_size=8",
            "num_workers=0", "eval_every=0",
            "conformal.num_mc_samples=2", "conformal.num_mc_samples_eval=2",
            f"exp_dir={tmp_path}",
            "trainer.max_epochs=1", "trainer.accelerator=cpu", "trainer.devices=1",
        ])
    from adaptive_roa.adaptive_v2.engine import AdaptiveEngine
    engine = AdaptiveEngine(cfg)
    engine.run()  # should complete one epoch without error
```

- [ ] **Step 2: Run it**

Run: `/common/home/st1122/Projects/adaptive_roa/env/bin/python -m pytest tests/test_humanoid_standup_reach_adaptive_smoke.py -v`
Expected: PASS (runs one tiny epoch). This test is heavier; if it exceeds a reasonable time or a config override name differs from `_base.yaml`, adjust the overrides to the actual keys (read `configs/adaptive_v2/system/_base.yaml`). If a genuine integration bug surfaces, that is the test doing its job — fix the cause, don't weaken the test.

- [ ] **Step 3: Commit**

```bash
git add tests/test_humanoid_standup_reach_adaptive_smoke.py
git commit -m "test: end-to-end adaptive_v2 smoke for humanoid_standup_reach (intermediate + fps)

Tiny 1-epoch CPU run of AdaptiveEngine with candidate_mode=intermediate and
FPS cal/test, exercising the full pool->train->calibrate->eval loop.

Claude-Session: https://claude.ai/code/session_01NZMXZPhQUpzSKYDqNqypoK"
```

---

## Open items / risks
- **Task 1**: `evaluate_roa.py` may have angle-wrapping keyed on state-dim (pendulum col 1 / cartpole); the 67-D path should miss it, but a full eval run (needs a trained checkpoint) is deferred to the Optional follow-up.
- **Task 4**: candidate enumeration in `intermediate` mode needs trajectory lengths — use a per-trajectory load cache in `TrajectoryDataSource`; for very large pools consider a lengths cache or lazy registry growth (start with the available/initial subset). The smoke test (Task 5) uses a tiny pool so this is not exercised at scale.
- **Task 5**: heavier/slower; keep sizes tiny. May reveal real integration issues (the point of the test).

## Optional follow-up (not a task)
A real adaptive run + a real ROA eval on a trained checkpoint:
```bash
python scripts/run_adaptive.py system=humanoid_standup_reach   # full adaptive loop
python adaptive_roa/flow_matching/evaluate_roa.py --config-name=evaluate_humanoid_standup_reach_roa
```
