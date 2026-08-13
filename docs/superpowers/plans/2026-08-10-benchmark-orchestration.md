# Benchmark Orchestration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn eleven implemented predictor arms into a defensible benchmark — a declarative campaign manifest, an idempotent launcher, validity guards that refuse to compare things that are not comparable, and tier-separated reporting.

**Architecture:** A campaign is a *declarative manifest* expanded into `RunSpec` records with deterministic identities. The launcher is idempotent over those identities, so preemption and relaunch converge rather than duplicate. Aggregation reads only on-disk artifacts. Every comparison passes through a guard layer that raises rather than silently producing a number from incomparable inputs.

**Tech Stack:** Python 3, PyTorch, Hydra, pandas, SLURM (Amarel `gpu-redhat`), pytest.

## Context: why this plan is mostly guards

Ten arms exist and are tested. The scientific risk is no longer "does the code run" — it is **producing a plausible-looking table that is wrong**. Four concrete ways that happens here, each already observed at least once in this workstream:

1. Results predating a correctness fix get pooled with results after it.
2. Arms trained to different budgets get ranked against each other.
3. A quantity that is not comparable across arms (`best_val_nll`) gets put in one column.
4. A reference posterior that did not converge gets reported as a reference anyway.

Each is invisible in the output. So the guards are the deliverable, not scaffolding around it.

## Open decisions — resolve before Task 5

These do not block Tasks 1–4. They are flagged here so they are decided deliberately rather than by default.

- **`hmc_reg` does not mix** (`rhat_max` 7.9–91, ceiling agreement 0.0). Final-state posterior-fidelity claims are therefore unavailable. This plan builds the gate so the number is *withheld*, not fudged. Whether to (a) diagnose the conditioning — likely a diagonal mass matrix or a floor on `log_sigma` — or (b) ship the reference tier outcome-head-only and mark final-state fidelity as open, is a research call.
- **Campaign scope.** The full cross-product is ~450 runs (below). Default phasing in Task 2 is pendulum + cartpole first, quadrotors second. Adjust the manifest, not the code.
- **Tempering.** The spec calls for headline MFVI at `beta = 1` with tempered results in a separately labelled block. This plan implements the labelling; whether to run the tempered block at all is a scope call.

## Scale

Reference tier `[50,50]`: 4 MLP-backbone posterior arms (MFVI, ensemble, Laplace, HMC) × 2 heads × 4 systems.
Production tier `[256,512,256]`: 11 arms × 4 systems, everything except HMC.
With paired random controls (except non-adaptive `mlp_det`) and 3 seeds, this is **~450 runs**. HMC runs are the expensive tail: 3 chains × (200 warmup + 200 sampling) × 20 leapfrog steps at full-batch gradients.

The manifest exists so this is a data change, not a code change.

## Global Constraints

- Python interpreter is the repo-local env: **`./env/bin/python`**. Never plain `python`.
- Run tests as `./env/bin/python -m pytest <path> -v` from the repo root.
- **No new pip dependencies.** pandas, numpy, torch, hydra, omegaconf are available.
- **Commit messages contain zero AI/tool attribution**, and no `Claude-Session:` trailer.
- Amarel: partition `gpu-redhat` only (`cgpu-redhat` is Camden — never submit). Max walltime `3-00:00:00`. The `general` account **preempts jobs silently**, which is why the launcher must be idempotent.
- Amarel is a separate filesystem: code via git, results via `rsync`. `.env` is gitignored and does not travel — `SHARED_DATA_BASE` in particular falls back to a hardcoded iLab path rather than erroring.
- Never infer job liveness from a quiet `.out` log — count artifacts instead.
- New code lives under `adaptive_roa/benchmark/`; tests under `tests/benchmark/`.

---

## File Structure

| File | Responsibility |
|---|---|
| `adaptive_roa/benchmark/manifest.py` | `RunSpec`, deterministic run identity, YAML → specs |
| `adaptive_roa/benchmark/launcher.py` | specs → sbatch; idempotent skip/resume |
| `adaptive_roa/benchmark/aggregate.py` | on-disk artifacts → tidy DataFrame with provenance |
| `adaptive_roa/benchmark/guards.py` | validity guards; raise rather than return a wrong number |
| `adaptive_roa/benchmark/fidelity.py` | agreement / TV vs the HMC reference, convergence-gated |
| `adaptive_roa/benchmark/separatrix.py` | metric slicing by distance to the basin boundary |
| `adaptive_roa/benchmark/report.py` | tier-separated tables |
| `configs/benchmark/*.yaml` | campaign manifests (data, not code) |
| `scripts/run_benchmark.py` | CLI: expand, launch, status, aggregate, report |
| `tests/benchmark/` | unit + integration tests |

---

### Task 0: Record run provenance in the artifact

**Files:**
- Modify: `adaptive_roa/adaptive_v2/engine.py` (the `extra={...}` block in the `EpochArtifacts` construction)
- Create: `adaptive_roa/benchmark/provenance.py`
- Test: `tests/benchmark/test_provenance.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `git_sha(repo_root=None) -> str | None`, `assert_post_fix(df, fix_sha, repo_root)`.

**Why this comes first.** Verified against the current code: the engine writes `extra={"n_cal_eval": ..., "optimize_mode": ...}` and records **no commit SHA anywhere**. Every guard about stale results in Task 4 is therefore unimplementable until a run says what code produced it. Without this, the single most likely way this benchmark produces a wrong table — pooling results from before and after a correctness fix — has no detector at all.

`git_sha` must return `None` rather than raise when the code is not running from a git checkout (Amarel runs from a clone, but a tarball deploy would not be), and the guard must treat `None` as "unknown, refuse" rather than "fine".

- [ ] **Step 1: Write the failing test**

```python
# tests/benchmark/test_provenance.py
import pandas as pd
import pytest
from adaptive_roa.benchmark.provenance import git_sha, assert_post_fix


def test_git_sha_returns_a_sha_in_a_real_checkout():
    sha = git_sha()
    assert sha is not None and len(sha) >= 7


def test_git_sha_returns_none_outside_a_checkout(tmp_path):
    assert git_sha(repo_root=tmp_path) is None


def test_unknown_provenance_is_refused_not_assumed_valid():
    df = pd.DataFrame({"arm": ["bnn_mfvi"], "commit": [None]})
    with pytest.raises(ValueError, match="unknown"):
        assert_post_fix(df, fix_sha="HEAD")


def test_a_commit_that_is_not_an_ancestor_of_the_fix_is_refused():
    # A run from before the fix must not be pooled with runs after it.
    df = pd.DataFrame({"arm": ["bnn_mfvi"], "commit": ["0000000"]})
    with pytest.raises(ValueError, match="predates|not found|unknown"):
        assert_post_fix(df, fix_sha="HEAD")


def test_a_run_at_head_passes_against_an_ancestor_fix():
    head = git_sha()
    df = pd.DataFrame({"arm": ["bnn_mfvi"], "commit": [head]})
    assert_post_fix(df, fix_sha="HEAD~1")
```

- [ ] **Step 2: Run to verify it fails**

Run: `./env/bin/python -m pytest tests/benchmark/test_provenance.py -v`
Expected: FAIL — `ModuleNotFoundError: adaptive_roa.benchmark.provenance`

- [ ] **Step 3: Implement**

```python
# adaptive_roa/benchmark/provenance.py
"""Which code produced a run.

Pooling results from before and after a correctness fix is the single most
likely way this benchmark produces a confident wrong table, and it is invisible
in the numbers. That detector needs each run to record its own commit.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd


def git_sha(repo_root=None) -> str | None:
    """Short SHA of the working tree, or None if this is not a git checkout."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(repo_root) if repo_root else None,
            capture_output=True, text=True, check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return out.stdout.strip() or None


def assert_post_fix(df: pd.DataFrame, fix_sha: str, repo_root=None) -> None:
    """Every row must come from code at or after `fix_sha`."""
    commits = df["commit"].unique()
    for commit in commits:
        if commit is None or (isinstance(commit, float) and pd.isna(commit)):
            raise ValueError(
                "a run reports unknown provenance (no recorded commit). "
                "Refusing to assume it postdates the fix -- that assumption is "
                "what this guard exists to prevent. Re-run it, or exclude it "
                "explicitly."
            )
        try:
            subprocess.run(
                ["git", "merge-base", "--is-ancestor", fix_sha, str(commit)],
                cwd=str(repo_root) if repo_root else None,
                capture_output=True, check=True,
            )
        except subprocess.CalledProcessError:
            raise ValueError(
                f"run at commit {commit} predates fix {fix_sha} (or is not "
                f"found in this repository). Its results are not comparable to "
                f"runs after that fix."
            ) from None
```

Then in `adaptive_roa/adaptive_v2/engine.py`, add the SHA to the artifact's `extra` block:

```python
                extra={
                    "n_cal_eval": n_cal_eval,
                    "optimize_mode": self.threshold_backend.optimize_mode,
                    "commit": git_sha(),
                },
```

- [ ] **Step 4: Run tests**

Run: `./env/bin/python -m pytest tests/benchmark/test_provenance.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Verify the SHA actually reaches an artifact**

Run a 1-epoch smoke and read the artifact back:
```bash
./env/bin/python scripts/run_adaptive.py system=pendulum predictor=mlp n_epochs=1 \
    output_dir=/tmp/prov_check
./env/bin/python -c "
import json; d=json.load(open('/tmp/prov_check/epoch_000/artifacts_v2.json'))
print('commit:', d['extra'].get('commit'))"
```
Expected: a non-empty short SHA. If it prints `None`, the guard layer is inert and the rest of this plan rests on nothing — stop and fix it here.

- [ ] **Step 6: Commit**

```bash
git add adaptive_roa/benchmark/provenance.py adaptive_roa/adaptive_v2/engine.py \
        tests/benchmark/test_provenance.py
git commit -m "feat(benchmark): record the producing commit in every epoch artifact"
```

---

### Task 1: Campaign manifest and deterministic run identity

**Files:**
- Create: `adaptive_roa/benchmark/__init__.py`, `adaptive_roa/benchmark/manifest.py`
- Create: `configs/benchmark/pilot.yaml`
- Test: `tests/benchmark/test_manifest.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `RunSpec` (frozen dataclass with fields `arm, system, tier, acquisition, seed, n_epochs, overrides: tuple[str, ...]`), `RunSpec.run_id -> str`, `RunSpec.hydra_overrides() -> list[str]`, `expand_manifest(cfg: dict) -> list[RunSpec]`.

**Why the identity must be deterministic:** the launcher's idempotency, the aggregator's provenance join, and resume-after-preemption all key on `run_id`. If it depends on wall-clock time or dict ordering, a relaunch duplicates work instead of resuming it.

- [ ] **Step 1: Write the failing test**

```python
# tests/benchmark/test_manifest.py
import pytest
from adaptive_roa.benchmark.manifest import RunSpec, expand_manifest


def _spec(**kw):
    base = dict(arm="bnn_mfvi", system="pendulum", tier="production",
                acquisition="ranked", seed=42, n_epochs=10, overrides=())
    base.update(kw)
    return RunSpec(**base)


def test_run_id_is_stable_across_processes():
    # Must not depend on hash randomization, dict order, or wall clock.
    assert _spec().run_id == _spec().run_id
    assert _spec(overrides=("a=1", "b=2")).run_id == _spec(overrides=("a=1", "b=2")).run_id


def test_run_id_separates_every_field_that_changes_the_experiment():
    base = _spec()
    for field, other in [("arm", "bnn_laplace"), ("system", "cartpole"),
                         ("tier", "reference"), ("acquisition", "random"),
                         ("seed", 43), ("overrides", ("x=1",))]:
        assert _spec(**{field: other}).run_id != base.run_id, field


def test_n_epochs_does_not_change_identity():
    # A run extended from 10 to 20 epochs is the SAME run resumed, not a new one.
    assert _spec(n_epochs=10).run_id == _spec(n_epochs=20).run_id


def test_expand_builds_the_full_cross_product():
    specs = expand_manifest({
        "arms": ["bnn_mfvi", "gp_reg"],
        "systems": ["pendulum", "cartpole"],
        "tier": "production",
        "acquisition": ["ranked", "random"],
        "seeds": [42, 43],
        "n_epochs": 10,
    })
    assert len(specs) == 2 * 2 * 2 * 2
    assert len({s.run_id for s in specs}) == len(specs)


def test_mlp_det_is_never_expanded_adaptively():
    # Its outcome probability collapses to {0,1}: no ranking signal exists.
    specs = expand_manifest({
        "arms": ["mlp_det"], "systems": ["pendulum"], "tier": "production",
        "acquisition": ["ranked", "random"], "seeds": [42], "n_epochs": 10,
    })
    assert [s.acquisition for s in specs] == ["random"]


def test_hydra_overrides_use_the_baseline_experiment_for_mlp_det():
    # Bare `predictor=mlp_det` silently runs adaptively: Hydra's defaults list
    # puts `predictor` before `acquisition`, so a d2_ratio set in the predictor
    # group is discarded.
    ov = _spec(arm="mlp_det", acquisition="random").hydra_overrides()
    assert "+experiment=mlp_det_baseline" in ov


def test_reference_tier_emits_its_experiment_config():
    assert "+experiment=reference_tier" in _spec(tier="reference").hydra_overrides()


def test_unknown_arm_is_rejected_at_expansion():
    with pytest.raises(ValueError, match="unknown arm"):
        expand_manifest({"arms": ["bnn_typo"], "systems": ["pendulum"],
                         "tier": "production", "acquisition": ["ranked"],
                         "seeds": [42], "n_epochs": 10})


def test_hmc_is_rejected_outside_the_reference_tier():
    # HMC is only a valid reference at the width the approximations use.
    with pytest.raises(ValueError, match="reference tier"):
        expand_manifest({"arms": ["hmc"], "systems": ["pendulum"],
                         "tier": "production", "acquisition": ["ranked"],
                         "seeds": [42], "n_epochs": 10})
```

- [ ] **Step 2: Run to verify it fails**

Run: `./env/bin/python -m pytest tests/benchmark/test_manifest.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'adaptive_roa.benchmark'`

- [ ] **Step 3: Implement**

```python
# adaptive_roa/benchmark/manifest.py
"""Declarative campaign specification.

A campaign is data, not code: the full cross-product is ~450 runs, and which
subset to run is a scope decision that should never require editing Python.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

# Arms that may run adaptively, and the tier each is valid in.
OUTCOME_ARMS = ("mlp", "gp", "gp_optdelta", "bnn_mfvi", "bnn_ensemble", "bnn_laplace")
FINAL_STATE_ARMS = ("fm", "mlp_det", "gp_reg", "bnn_mfvi_reg",
                    "bnn_ensemble_reg", "bnn_laplace_reg")
REFERENCE_ONLY_ARMS = ("hmc", "hmc_reg")
ALL_ARMS = OUTCOME_ARMS + FINAL_STATE_ARMS + REFERENCE_ONLY_ARMS

# Fixed-dataset baselines: no ranking signal, so never expanded adaptively.
NON_ADAPTIVE_ARMS = ("mlp_det",)

# Arms whose posterior is over MLP weights, so an HMC reference applies to them.
MLP_BACKBONE_ARMS = ("bnn_mfvi", "bnn_ensemble", "bnn_laplace",
                     "bnn_mfvi_reg", "bnn_ensemble_reg", "bnn_laplace_reg")

TIERS = ("production", "reference")


@dataclass(frozen=True)
class RunSpec:
    arm: str
    system: str
    tier: str
    acquisition: str
    seed: int
    n_epochs: int
    overrides: tuple[str, ...] = ()

    @property
    def run_id(self) -> str:
        """Stable identity for this experiment.

        Deterministic across processes: built from a sorted, explicit tuple and
        hashed with sha1 rather than builtin hash(), which is randomized per
        process. n_epochs is deliberately EXCLUDED -- extending a run from 10 to
        20 epochs resumes the same experiment rather than starting a new one.
        """
        parts = (self.arm, self.system, self.tier, self.acquisition,
                 str(self.seed)) + tuple(sorted(self.overrides))
        digest = hashlib.sha1("|".join(parts).encode()).hexdigest()[:8]
        return f"{self.system}_{self.arm}_{self.tier}_{self.acquisition}_s{self.seed}_{digest}"

    def hydra_overrides(self) -> list[str]:
        ov = [f"system={self.system}", f"predictor={self.arm}",
              f"seed={self.seed}", f"n_epochs={self.n_epochs}"]
        if self.tier == "reference":
            ov.append("+experiment=reference_tier")
        if self.arm in NON_ADAPTIVE_ARMS:
            # MUST be the experiment config: a d2_ratio written in the predictor
            # group is silently discarded (defaults list orders predictor first).
            ov.append("+experiment=mlp_det_baseline")
        else:
            ov.append(f"acquisition={self.acquisition}")
        ov.extend(self.overrides)
        return ov


def expand_manifest(cfg: dict[str, Any]) -> list[RunSpec]:
    tier = str(cfg["tier"])
    if tier not in TIERS:
        raise ValueError(f"unknown tier {tier!r}; expected one of {TIERS}")

    arms = list(cfg["arms"])
    for arm in arms:
        if arm not in ALL_ARMS:
            raise ValueError(f"unknown arm {arm!r}; expected one of {ALL_ARMS}")
        if arm in REFERENCE_ONLY_ARMS and tier != "reference":
            raise ValueError(
                f"arm {arm!r} is valid in the reference tier only: HMC is a "
                f"reference for the SAME architecture the approximations use, "
                f"so an HMC run at [50,50] says nothing about a posterior at "
                f"[256,512,256]."
            )

    specs: list[RunSpec] = []
    for arm in arms:
        modes = ["random"] if arm in NON_ADAPTIVE_ARMS else list(cfg["acquisition"])
        for system in cfg["systems"]:
            for mode in modes:
                for seed in cfg["seeds"]:
                    specs.append(RunSpec(
                        arm=arm, system=system, tier=tier, acquisition=mode,
                        seed=int(seed), n_epochs=int(cfg["n_epochs"]),
                        overrides=tuple(cfg.get("overrides", ())),
                    ))
    return specs
```

```yaml
# configs/benchmark/pilot.yaml
# Pilot campaign: cheapest systems, full arm set, enough seeds for a variance
# claim. Validates the harness before the quadrotor phase is committed.
tier: production
arms: [mlp, gp, bnn_mfvi, bnn_ensemble, bnn_laplace,
       fm, mlp_det, gp_reg, bnn_mfvi_reg, bnn_ensemble_reg, bnn_laplace_reg]
systems: [pendulum, cartpole]
acquisition: [ranked, random]   # paired control per arm
seeds: [42, 43, 44]
n_epochs: 10
```

- [ ] **Step 4: Run tests**

Run: `./env/bin/python -m pytest tests/benchmark/test_manifest.py -v`
Expected: PASS (9 tests)

- [ ] **Step 5: Commit**

```bash
git add adaptive_roa/benchmark/__init__.py adaptive_roa/benchmark/manifest.py \
        configs/benchmark/pilot.yaml tests/benchmark/test_manifest.py
git commit -m "feat(benchmark): declarative campaign manifest with stable run identity"
```

---

### Task 2: Idempotent launcher

**Files:**
- Create: `adaptive_roa/benchmark/launcher.py`, `scripts/run_benchmark.py`
- Test: `tests/benchmark/test_launcher.py`

**Interfaces:**
- Consumes: `RunSpec`, `expand_manifest` from Task 1.
- Produces: `run_state(spec, exp_root) -> str` returning one of `"complete" | "partial" | "absent"`; `plan_launch(specs, exp_root) -> tuple[list[RunSpec], dict[str, str]]`; `sbatch_command(spec, exp_root) -> list[str]`.

**Why idempotency is the whole point:** the `general` account preempts jobs *silently*. A relaunch that cannot tell "done" from "half done" from "never started" either duplicates finished work or silently skips work that never ran. Completion is judged by counting `artifacts_v2.json` files — never by log contents, which stay quiet on healthy jobs.

- [ ] **Step 1: Write the failing test**

```python
# tests/benchmark/test_launcher.py
import json
import pytest
from adaptive_roa.benchmark.manifest import RunSpec
from adaptive_roa.benchmark.launcher import run_state, plan_launch, sbatch_command


def _spec(seed=42, arm="bnn_mfvi"):
    return RunSpec(arm=arm, system="pendulum", tier="production",
                   acquisition="ranked", seed=seed, n_epochs=3)


def _make_run(root, spec, n_epochs_done):
    d = root / spec.run_id
    for e in range(n_epochs_done):
        ed = d / f"epoch_{e:03d}"
        ed.mkdir(parents=True)
        (ed / "artifacts_v2.json").write_text(json.dumps({"epoch": e}))
    return d


def test_absent_when_nothing_on_disk(tmp_path):
    assert run_state(_spec(), tmp_path) == "absent"


def test_partial_when_some_epochs_present(tmp_path):
    s = _spec(); _make_run(tmp_path, s, 2)
    assert run_state(s, tmp_path) == "partial"


def test_complete_when_all_epochs_present(tmp_path):
    s = _spec(); _make_run(tmp_path, s, 3)
    assert run_state(s, tmp_path) == "complete"


def test_an_empty_epoch_dir_is_not_a_finished_epoch(tmp_path):
    # A preempted job leaves the directory but no artifact. Counting DIRS
    # instead of artifacts would call this run complete and skip it forever.
    s = _spec(); _make_run(tmp_path, s, 2)
    (tmp_path / s.run_id / "epoch_002").mkdir()
    assert run_state(s, tmp_path) == "partial"


def test_a_quiet_log_does_not_imply_completion(tmp_path):
    s = _spec(); _make_run(tmp_path, s, 1)
    (tmp_path / s.run_id / "job.out").write_text("Epoch 3/3 done\nfinished\n")
    assert run_state(s, tmp_path) == "partial"


def test_plan_launch_skips_complete_and_relaunches_partial(tmp_path):
    done, partial, fresh = _spec(seed=1), _spec(seed=2), _spec(seed=3)
    _make_run(tmp_path, done, 3)
    _make_run(tmp_path, partial, 1)
    to_launch, states = plan_launch([done, partial, fresh], tmp_path)
    assert [s.run_id for s in to_launch] == [partial.run_id, fresh.run_id]
    assert states[done.run_id] == "complete"


def test_relaunching_twice_launches_nothing_new(tmp_path):
    specs = [_spec(seed=i) for i in range(3)]
    for s in specs:
        _make_run(tmp_path, s, 3)
    to_launch, _ = plan_launch(specs, tmp_path)
    assert to_launch == []


def test_sbatch_targets_the_correct_partition_and_carries_overrides(tmp_path):
    cmd = sbatch_command(_spec(), tmp_path)
    joined = " ".join(cmd)
    assert "gpu-redhat" in joined
    assert "cgpu-redhat" not in joined          # Camden: never submit
    assert "predictor=bnn_mfvi" in joined
    assert "seed=42" in joined
    assert f"output_dir={tmp_path / _spec().run_id}" in joined
```

- [ ] **Step 2: Run to verify it fails**

Run: `./env/bin/python -m pytest tests/benchmark/test_launcher.py -v`
Expected: FAIL — `ImportError: cannot import name 'run_state'`

- [ ] **Step 3: Implement**

```python
# adaptive_roa/benchmark/launcher.py
"""Idempotent campaign launcher.

The `general` account preempts jobs silently, so relaunching a campaign is the
NORMAL path, not an error path. Completion is judged by counting
artifacts_v2.json files: a preempted job leaves epoch directories behind, and a
healthy job's log stays quiet for long stretches, so neither directories nor
logs are evidence of progress.
"""
from __future__ import annotations

from pathlib import Path

from .manifest import RunSpec

SBATCH_TEMPLATE = "scripts/sbatch_amarel.sh"


def completed_epochs(run_dir: Path) -> int:
    if not run_dir.exists():
        return 0
    return sum(1 for p in run_dir.glob("epoch_*/artifacts_v2.json") if p.is_file())


def run_state(spec: RunSpec, exp_root: Path) -> str:
    done = completed_epochs(Path(exp_root) / spec.run_id)
    if done == 0:
        return "absent"
    return "complete" if done >= spec.n_epochs else "partial"


def plan_launch(specs, exp_root):
    """Return (specs_to_launch, {run_id: state}). Complete runs are skipped."""
    exp_root = Path(exp_root)
    states, to_launch = {}, []
    for spec in specs:
        state = run_state(spec, exp_root)
        states[spec.run_id] = state
        if state != "complete":
            to_launch.append(spec)
    return to_launch, states


def sbatch_command(spec: RunSpec, exp_root: Path) -> list[str]:
    run_dir = Path(exp_root) / spec.run_id
    return [
        "sbatch",
        "-J", spec.run_id,
        "--partition=gpu-redhat",   # cgpu-redhat is Camden -- never submit there
        SBATCH_TEMPLATE,
        *spec.hydra_overrides(),
        f"output_dir={run_dir}",
    ]
```

```python
# scripts/run_benchmark.py
"""Campaign CLI: expand a manifest, report status, launch what is missing."""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import yaml

from adaptive_roa.benchmark.launcher import plan_launch, sbatch_command
from adaptive_roa.benchmark.manifest import expand_manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest", type=Path)
    ap.add_argument("--exp-root", type=Path, required=True)
    ap.add_argument("--launch", action="store_true",
                    help="actually submit; default is a dry run")
    args = ap.parse_args()

    specs = expand_manifest(yaml.safe_load(args.manifest.read_text()))
    to_launch, states = plan_launch(specs, args.exp_root)

    tally = {s: sum(1 for v in states.values() if v == s)
             for s in ("complete", "partial", "absent")}
    print(f"{len(specs)} runs: {tally['complete']} complete, "
          f"{tally['partial']} partial, {tally['absent']} absent")

    for spec in to_launch:
        cmd = sbatch_command(spec, args.exp_root)
        if args.launch:
            subprocess.run(cmd, check=True)
        else:
            print(" ".join(cmd))
    if not args.launch:
        print(f"\ndry run -- {len(to_launch)} would be submitted. Pass --launch.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests**

Run: `./env/bin/python -m pytest tests/benchmark/test_launcher.py -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Verify the dry run against a real manifest**

Run: `./env/bin/python scripts/run_benchmark.py configs/benchmark/pilot.yaml --exp-root /tmp/bench_dry`
Expected: prints `126 runs: 0 complete, 0 partial, 126 absent` followed by 126 sbatch lines, and submits nothing.

126 = 10 adaptively-run arms × 2 systems × 2 acquisition modes × 3 seeds (120), plus `mlp_det` × 2 systems × **1** mode × 3 seeds (6). The naive `11 × 2 × 2 × 3 = 132` forgets that `mlp_det` collapses to random-only.

- [ ] **Step 6: Commit**

```bash
git add adaptive_roa/benchmark/launcher.py scripts/run_benchmark.py tests/benchmark/test_launcher.py
git commit -m "feat(benchmark): idempotent launcher keyed on completed epoch artifacts"
```

---

### Task 3: Aggregation with provenance

**Files:**
- Create: `adaptive_roa/benchmark/aggregate.py`
- Test: `tests/benchmark/test_aggregate.py`

**Interfaces:**
- Consumes: run directories written by the engine.
- Produces: `collect_runs(exp_root) -> pd.DataFrame`, one row per `(run_id, epoch)`, with columns `run_id, arm, system, tier, acquisition, seed, epoch, commit, <metrics...>`.

**Why provenance travels with every row:** the guards in Task 4 decide validity per row. If `commit` is attached only at report time, a mixed-vintage pool is undetectable. The engine already copies `.hydra` into each epoch directory — read the arm identity from there rather than parsing the directory name, so a renamed directory cannot silently relabel an arm.

- [ ] **Step 1: Write the failing test**

```python
# tests/benchmark/test_aggregate.py
import json
import pytest
from adaptive_roa.benchmark.aggregate import collect_runs


def _write_run(root, run_id, arm, n_epochs=2, commit="abc1234", seed=42):
    for e in range(n_epochs):
        ed = root / run_id / f"epoch_{e:03d}"
        (ed / ".hydra").mkdir(parents=True)
        (ed / ".hydra" / "config.yaml").write_text(
            f"predictor:\n  name: {arm}\nsystem:\n  name: pendulum\nseed: {seed}\n")
        (ed / "artifacts_v2.json").write_text(json.dumps({
            "epoch": e, "sampling_mode": "ranked",
            "eval_metrics": {"accuracy": 0.8 + 0.01 * e, "f1": 0.7},
            "extra": {"commit": commit},
        }))


def test_one_row_per_run_epoch(tmp_path):
    _write_run(tmp_path, "r1", "bnn_mfvi", n_epochs=3)
    df = collect_runs(tmp_path)
    assert len(df) == 3
    assert set(df["epoch"]) == {0, 1, 2}


def test_arm_comes_from_the_config_not_the_directory_name(tmp_path):
    # A renamed directory must not relabel the arm.
    _write_run(tmp_path, "misleading_name_gp_reg", "bnn_laplace")
    df = collect_runs(tmp_path)
    assert set(df["arm"]) == {"bnn_laplace"}


def test_metrics_are_flattened_into_columns(tmp_path):
    _write_run(tmp_path, "r1", "bnn_mfvi")
    df = collect_runs(tmp_path)
    assert "accuracy" in df.columns and "f1" in df.columns
    assert df.loc[df.epoch == 1, "accuracy"].iloc[0] == pytest.approx(0.81)


def test_commit_provenance_is_carried_per_row(tmp_path):
    _write_run(tmp_path, "r1", "bnn_mfvi", commit="aaaaaaa")
    _write_run(tmp_path, "r2", "bnn_mfvi", commit="bbbbbbb")
    df = collect_runs(tmp_path)
    assert set(df["commit"]) == {"aaaaaaa", "bbbbbbb"}


def test_a_partial_run_contributes_only_its_finished_epochs(tmp_path):
    _write_run(tmp_path, "r1", "bnn_mfvi", n_epochs=2)
    (tmp_path / "r1" / "epoch_002").mkdir()      # preempted: dir but no artifact
    df = collect_runs(tmp_path)
    assert len(df) == 2


def test_empty_root_returns_an_empty_frame_not_a_crash(tmp_path):
    df = collect_runs(tmp_path)
    assert len(df) == 0
```

- [ ] **Step 2: Run to verify it fails**

Run: `./env/bin/python -m pytest tests/benchmark/test_aggregate.py -v`
Expected: FAIL — `ModuleNotFoundError: adaptive_roa.benchmark.aggregate`

- [ ] **Step 3: Implement**

```python
# adaptive_roa/benchmark/aggregate.py
"""Collect on-disk run artifacts into one tidy frame.

Provenance (commit, seed, arm, tier) is attached per ROW, not per report, so a
pool mixing pre- and post-fix vintages is detectable by the guard layer rather
than averaging silently into a plausible number.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml


def _epoch_rows(epoch_dir: Path) -> dict | None:
    art = epoch_dir / "artifacts_v2.json"
    if not art.is_file():
        return None                      # preempted: directory without artifact
    payload = json.loads(art.read_text())

    cfg = {}
    cfg_path = epoch_dir / ".hydra" / "config.yaml"
    if cfg_path.is_file():
        cfg = yaml.safe_load(cfg_path.read_text()) or {}

    predictor = cfg.get("predictor") or {}
    if isinstance(predictor, str):       # legacy configs record a bare string
        arm = predictor
    else:
        arm = predictor.get("name")
    system = (cfg.get("system") or {})
    system = system.get("name") if isinstance(system, dict) else system

    row = {
        "epoch": payload.get("epoch"),
        "arm": arm,
        "system": system,
        "seed": cfg.get("seed"),
        "acquisition": payload.get("sampling_mode"),
        "commit": (payload.get("extra") or {}).get("commit"),
    }
    for k, v in (payload.get("eval_metrics") or {}).items():
        if isinstance(v, (int, float)):
            row[k] = v
    return row


def collect_runs(exp_root) -> pd.DataFrame:
    exp_root = Path(exp_root)
    rows = []
    for run_dir in sorted(p for p in exp_root.iterdir() if p.is_dir()):
        for epoch_dir in sorted(run_dir.glob("epoch_*")):
            row = _epoch_rows(epoch_dir)
            if row is not None:
                rows.append({"run_id": run_dir.name, **row})
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Run tests**

Run: `./env/bin/python -m pytest tests/benchmark/test_aggregate.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add adaptive_roa/benchmark/aggregate.py tests/benchmark/test_aggregate.py
git commit -m "feat(benchmark): aggregate run artifacts with per-row provenance"
```

---

### Task 4: Validity guards

**Files:**
- Create: `adaptive_roa/benchmark/guards.py`
- Test: `tests/benchmark/test_guards.py`

**Interfaces:**
- Consumes: the DataFrame from Task 3.
- Produces: `INVALIDATING_COMMITS: dict[str, str]`, `assert_post_fix(df, repo)`, `assert_matched_budget(df)`, `NON_COMPARABLE_COLUMNS: frozenset[str]`, `assert_comparable(df, column)`.

**This is the core deliverable.** Each guard corresponds to a way this benchmark has already produced, or come close to producing, a confident wrong number. Every guard **raises**; none warns, filters silently, or returns a default.

The invalidating commits:

| Arm family | Valid only at or after | What changed |
|---|---|---|
| outcome BNN arms | `597db9f` | which weights ship; ensemble marginal now enumerated |
| final-state arms | `18ea67e` | head learns in normalized coordinates |
| `gp_reg` | `e642259` | sampling carried ~4% of intended variance |
| any multi-seed claim | the `_seeding.py` commit | seeds were ignored; replicates were identical |

- [ ] **Step 1: Write the failing test**

```python
# tests/benchmark/test_guards.py
import pandas as pd
import pytest
from adaptive_roa.benchmark.guards import (
    assert_matched_budget, assert_comparable, assert_distinct_seeds,
    NON_COMPARABLE_COLUMNS,
)


def _df(**cols):
    base = dict(arm=["bnn_mfvi", "bnn_laplace"], n_epochs=[10, 10],
                seed=[42, 42], accuracy=[0.8, 0.9])
    base.update(cols)
    return pd.DataFrame(base)


def test_matched_budget_passes_when_budgets_agree():
    assert_matched_budget(_df())


def test_matched_budget_raises_on_mismatch():
    with pytest.raises(ValueError, match="budget"):
        assert_matched_budget(_df(n_epochs=[10, 20]))


def test_val_nll_is_refused_as_a_cross_arm_column():
    # Same criterion CLASS, different quantity: the BNN arms use beta-NLL in
    # manifold-normalized coords summed over dims; gp_reg uses a plain Gaussian
    # NLL in embedded space meaned over tasks. Selection is valid within an arm.
    assert "best_val_nll" in NON_COMPARABLE_COLUMNS
    with pytest.raises(ValueError, match="not comparable across arms"):
        assert_comparable(_df(best_val_nll=[1.0, 2.0]), "best_val_nll")


def test_a_normal_metric_is_allowed():
    assert_comparable(_df(), "accuracy")


def test_identical_results_across_nominal_seeds_are_refused():
    # Three trainers ignored the run seed until the _seeding fix: replicates at
    # seeds 42/43/44 were bit-identical. A variance claim over them is fiction.
    df = pd.DataFrame({"arm": ["bnn_mfvi"] * 3, "seed": [42, 43, 44],
                       "accuracy": [0.8, 0.8, 0.8], "n_epochs": [10] * 3})
    with pytest.raises(ValueError, match="identical"):
        assert_distinct_seeds(df, "accuracy")


def test_genuinely_varying_seeds_pass():
    df = pd.DataFrame({"arm": ["bnn_mfvi"] * 3, "seed": [42, 43, 44],
                       "accuracy": [0.80, 0.83, 0.79], "n_epochs": [10] * 3})
    assert_distinct_seeds(df, "accuracy")


def test_single_seed_is_not_flagged():
    df = pd.DataFrame({"arm": ["bnn_mfvi"], "seed": [42],
                       "accuracy": [0.8], "n_epochs": [10]})
    assert_distinct_seeds(df, "accuracy")
```

- [ ] **Step 2: Run to verify it fails**

Run: `./env/bin/python -m pytest tests/benchmark/test_guards.py -v`
Expected: FAIL — `ModuleNotFoundError: adaptive_roa.benchmark.guards`

- [ ] **Step 3: Implement**

```python
# adaptive_roa/benchmark/guards.py
"""Guards that refuse to produce a number from incomparable inputs.

Every function here RAISES. None warns, filters silently, or substitutes a
default: the failure mode these exist to prevent is a confident wrong table, and
a warning in a log is not a defense against that.
"""
from __future__ import annotations

import pandas as pd

# Results before these commits are not comparable to results after them.
INVALIDATING_COMMITS = {
    "outcome_bnn": "597db9f",   # which weights ship; ensemble marginal enumerated
    "final_state": "18ea67e",   # head learns in normalized coordinates
    "gp_reg": "e642259",        # sampling carried ~4% of intended variance
}

# Quantities that are well-defined within an arm but mean different things
# across arms. Putting these in one column is a category error.
NON_COMPARABLE_COLUMNS = frozenset({"best_val_nll", "val_nll", "train_loss"})


def assert_matched_budget(df: pd.DataFrame) -> None:
    """Every arm in a comparison must have trained to the same budget."""
    budgets = df.groupby("arm")["n_epochs"].nunique()
    if (budgets > 1).any():
        offenders = budgets[budgets > 1].index.tolist()
        raise ValueError(
            f"arms {offenders} contain runs at more than one training budget; "
            f"weak baselines are the characteristic failure of this literature, "
            f"so ranking across budgets is refused."
        )
    per_arm = df.groupby("arm")["n_epochs"].max()
    if per_arm.nunique() > 1:
        raise ValueError(
            f"training budget differs across arms: {per_arm.to_dict()}. "
            f"Equalize the budget or compare within a budget."
        )


def assert_comparable(df: pd.DataFrame, column: str) -> None:
    if column in NON_COMPARABLE_COLUMNS and df["arm"].nunique() > 1:
        raise ValueError(
            f"{column!r} is not comparable across arms: the same criterion "
            f"class denotes a different quantity per arm (beta-NLL in "
            f"manifold-normalized coordinates summed over dims for the BNN "
            f"arms; a plain Gaussian NLL in embedded space meaned over tasks "
            f"for gp_reg). Model selection within an arm is unaffected."
        )


def assert_distinct_seeds(df: pd.DataFrame, column: str) -> None:
    """Refuse a variance claim over replicates that are not actually distinct."""
    for arm, grp in df.groupby("arm"):
        if grp["seed"].nunique() < 2:
            continue
        if grp[column].nunique() == 1:
            raise ValueError(
                f"arm {arm!r} reports identical {column!r} across "
                f"{grp['seed'].nunique()} nominal seeds. Three trainers ignored "
                f"the run-level seed until the _seeding fix, making replicates "
                f"bit-identical; a variance claim over them is fiction. Confirm "
                f"these runs postdate that fix."
            )
```

- [ ] **Step 4: Run tests**

Run: `./env/bin/python -m pytest tests/benchmark/test_guards.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add adaptive_roa/benchmark/guards.py tests/benchmark/test_guards.py
git commit -m "feat(benchmark): validity guards for budget, comparability, and seed distinctness"
```

---

### Task 5: Convergence-gated posterior fidelity

**Files:**
- Create: `adaptive_roa/benchmark/fidelity.py`
- Test: `tests/benchmark/test_fidelity.py`

**Interfaces:**
- Consumes: `hmc_diagnostics.json` written per run; per-point probabilities from the export layer.
- Produces: `FidelityResult` (dataclass: `available: bool`, `reason: str | None`, `agreement: float | None`, `total_variation: float | None`), `fidelity_vs_reference(approx_probs, ref_probs, ref_diagnostics, rhat_threshold=1.1) -> FidelityResult`.

**The gate is the point.** `hmc_reg` currently does not converge. A fidelity number computed against a non-converged reference is not a weak result — it is a meaningless one, and it looks exactly like a real result. When the reference fails its own diagnostic, this returns `available=False` with a reason, and the report prints the reason instead of a number.

- [ ] **Step 1: Write the failing test**

```python
# tests/benchmark/test_fidelity.py
import numpy as np
import pytest
from adaptive_roa.benchmark.fidelity import fidelity_vs_reference


CONVERGED = {"rhat_max": 1.02, "converged": True}
DIVERGED = {"rhat_max": 91.4, "converged": False}


def test_returns_a_number_when_the_reference_converged():
    a = np.array([0.9, 0.1, 0.8]); r = np.array([0.85, 0.15, 0.75])
    res = fidelity_vs_reference(a, r, CONVERGED)
    assert res.available
    assert res.total_variation == pytest.approx(0.05, abs=1e-9)


def test_withholds_the_number_when_the_reference_did_not_converge():
    a = np.array([0.9, 0.1]); r = np.array([0.85, 0.15])
    res = fidelity_vs_reference(a, r, DIVERGED)
    assert not res.available
    assert res.agreement is None and res.total_variation is None
    assert "rhat" in res.reason.lower()


def test_the_withheld_case_still_names_the_measured_value():
    res = fidelity_vs_reference(np.array([0.9]), np.array([0.8]), DIVERGED)
    assert "91.4" in res.reason


def test_identical_posteriors_score_perfectly():
    p = np.array([0.3, 0.7, 0.5])
    res = fidelity_vs_reference(p, p, CONVERGED)
    assert res.total_variation == pytest.approx(0.0)
    assert res.agreement == pytest.approx(1.0)


def test_a_missing_diagnostic_is_refused_rather_than_assumed_converged():
    with pytest.raises(ValueError, match="diagnostic"):
        fidelity_vs_reference(np.array([0.9]), np.array([0.8]), {})


def test_threshold_is_honoured():
    marginal = {"rhat_max": 1.05, "converged": True}
    assert fidelity_vs_reference(np.array([0.9]), np.array([0.8]), marginal,
                                 rhat_threshold=1.01).available is False


def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape"):
        fidelity_vs_reference(np.array([0.9, 0.1]), np.array([0.8]), CONVERGED)
```

- [ ] **Step 2: Run to verify it fails**

Run: `./env/bin/python -m pytest tests/benchmark/test_fidelity.py -v`
Expected: FAIL — `ModuleNotFoundError: adaptive_roa.benchmark.fidelity`

- [ ] **Step 3: Implement**

```python
# adaptive_roa/benchmark/fidelity.py
"""Posterior fidelity against the HMC reference, gated on the reference's own
convergence.

A fidelity number computed against a reference that did not converge is not a
weak result -- it is a meaningless one that is indistinguishable from a real
one. So the gate withholds the number and reports why, rather than emitting a
caveat nobody carries forward into the table.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FidelityResult:
    available: bool
    reason: str | None = None
    agreement: float | None = None
    total_variation: float | None = None


def fidelity_vs_reference(approx_probs, ref_probs, ref_diagnostics,
                          rhat_threshold: float = 1.1) -> FidelityResult:
    approx = np.asarray(approx_probs, dtype=float)
    ref = np.asarray(ref_probs, dtype=float)
    if approx.shape != ref.shape:
        raise ValueError(
            f"shape mismatch: approx {approx.shape} vs reference {ref.shape}"
        )
    if "rhat_max" not in (ref_diagnostics or {}):
        raise ValueError(
            "reference diagnostic 'rhat_max' is absent. Refusing to assume the "
            "reference converged -- that assumption is exactly what this gate "
            "exists to prevent."
        )

    rhat = float(ref_diagnostics["rhat_max"])
    if not np.isfinite(rhat) or rhat > rhat_threshold:
        return FidelityResult(
            available=False,
            reason=(f"reference did not converge: rhat_max={rhat} exceeds "
                    f"{rhat_threshold}. Fidelity against a non-converged "
                    f"posterior is undefined, so no number is reported."),
        )

    tv = float(np.mean(np.abs(approx - ref)))
    agreement = float(np.mean((approx >= 0.5) == (ref >= 0.5)))
    return FidelityResult(available=True, agreement=agreement,
                          total_variation=tv)
```

- [ ] **Step 4: Run tests**

Run: `./env/bin/python -m pytest tests/benchmark/test_fidelity.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add adaptive_roa/benchmark/fidelity.py tests/benchmark/test_fidelity.py
git commit -m "feat(benchmark): posterior fidelity gated on the reference's own convergence"
```

---

### Task 6: Separatrix-conditioned reporting

**Files:**
- Create: `adaptive_roa/benchmark/separatrix.py`
- Test: `tests/benchmark/test_separatrix.py`

**Interfaces:**
- Consumes: per-point states and probabilities; a system object exposing `normalize_state`.
- Produces: `separatrix_band(states, labels, k=5) -> np.ndarray[bool]`; `conditioned_metrics(states, labels, probs, k=5) -> dict` with keys `overall`, `near_boundary`, `interior`, plus `n_near`, `n_interior`.

**Why this slice exists:** aggregate accuracy is dominated by the interior of the basins, where every arm is correct and the task is easy. Arms differ where the outcome actually flips. A benchmark reported only in aggregate can rank arms identically while they behave very differently at the boundary — which is the regime adaptive sampling exists to resolve.

The band is defined empirically — a point is "near boundary" if its k nearest neighbours in normalized state space do not all share its label — so it needs no analytic separatrix and works on all four systems.

- [ ] **Step 1: Write the failing test**

```python
# tests/benchmark/test_separatrix.py
import numpy as np
import pytest
from adaptive_roa.benchmark.separatrix import separatrix_band, conditioned_metrics


def _two_blobs(n=60):
    # Two well-separated clusters: only points near x=0 should be flagged.
    left = np.linspace(-1.0, -0.05, n // 2).reshape(-1, 1)
    right = np.linspace(0.05, 1.0, n // 2).reshape(-1, 1)
    states = np.vstack([left, right])
    labels = np.array([0] * (n // 2) + [1] * (n // 2))
    return states, labels


def test_boundary_points_are_flagged_and_far_points_are_not():
    states, labels = _two_blobs()
    band = separatrix_band(states, labels, k=3)
    assert band[len(labels) // 2 - 1] and band[len(labels) // 2]   # straddle 0
    assert not band[0] and not band[-1]                            # extremes


def test_a_single_label_yields_an_empty_band():
    states = np.linspace(0, 1, 20).reshape(-1, 1)
    band = separatrix_band(states, np.zeros(20, dtype=int), k=3)
    assert not band.any()


def test_conditioned_metrics_split_the_population():
    states, labels = _two_blobs()
    probs = labels.astype(float)                # a perfect predictor
    m = conditioned_metrics(states, labels, probs, k=3)
    assert m["n_near"] + m["n_interior"] == len(labels)
    assert m["overall"] == pytest.approx(1.0)
    assert m["near_boundary"] == pytest.approx(1.0)


def test_boundary_errors_are_invisible_in_the_aggregate():
    # THE POINT of this module: an arm that fails only at the boundary still
    # scores well overall, so the aggregate alone cannot distinguish arms.
    states, labels = _two_blobs(n=100)
    probs = labels.astype(float)
    band = separatrix_band(states, labels, k=3)
    probs[band] = 1.0 - probs[band]             # wrong on every boundary point
    m = conditioned_metrics(states, labels, probs, k=3)
    assert m["overall"] > 0.85                  # aggregate still looks healthy
    assert m["near_boundary"] == pytest.approx(0.0)


def test_k_larger_than_the_population_is_rejected():
    states, labels = _two_blobs(n=10)
    with pytest.raises(ValueError, match="k"):
        separatrix_band(states, labels, k=50)
```

- [ ] **Step 2: Run to verify it fails**

Run: `./env/bin/python -m pytest tests/benchmark/test_separatrix.py -v`
Expected: FAIL — `ModuleNotFoundError: adaptive_roa.benchmark.separatrix`

- [ ] **Step 3: Implement**

```python
# adaptive_roa/benchmark/separatrix.py
"""Condition metrics on proximity to the basin boundary.

Aggregate accuracy is dominated by basin interiors, where every arm is correct.
Arms differ where the outcome flips -- which is precisely the regime adaptive
sampling exists to resolve -- so a benchmark reported only in aggregate can rank
arms identically while they behave very differently.

The band is empirical (a point whose k nearest neighbours do not all share its
label), so no analytic separatrix is needed and it applies to all four systems.
"""
from __future__ import annotations

import numpy as np


def separatrix_band(states, labels, k: int = 5) -> np.ndarray:
    states = np.asarray(states, dtype=float)
    labels = np.asarray(labels)
    n = len(labels)
    if k >= n:
        raise ValueError(f"k={k} must be smaller than the population n={n}")
    if len(np.unique(labels)) < 2:
        return np.zeros(n, dtype=bool)

    # Pairwise distances in the caller's coordinates. Callers pass NORMALIZED
    # states so that dimensions with different physical scales contribute
    # comparably -- an unnormalized angular velocity would otherwise dominate.
    d = np.linalg.norm(states[:, None, :] - states[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    nn = np.argsort(d, axis=1)[:, :k]
    return np.array([not np.all(labels[nn[i]] == labels[i]) for i in range(n)])


def conditioned_metrics(states, labels, probs, k: int = 5) -> dict:
    labels = np.asarray(labels)
    preds = np.asarray(probs, dtype=float) >= 0.5
    correct = preds == labels.astype(bool)
    band = separatrix_band(states, labels, k=k)

    def _acc(mask):
        return float(correct[mask].mean()) if mask.any() else float("nan")

    return {
        "overall": float(correct.mean()),
        "near_boundary": _acc(band),
        "interior": _acc(~band),
        "n_near": int(band.sum()),
        "n_interior": int((~band).sum()),
    }
```

- [ ] **Step 4: Run tests**

Run: `./env/bin/python -m pytest tests/benchmark/test_separatrix.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add adaptive_roa/benchmark/separatrix.py tests/benchmark/test_separatrix.py
git commit -m "feat(benchmark): separatrix-conditioned metric slicing"
```

---

### Task 7: Report generation

**Files:**
- Create: `adaptive_roa/benchmark/report.py`
- Modify: `scripts/run_benchmark.py` (add a `report` subcommand)
- Test: `tests/benchmark/test_report.py`

**Interfaces:**
- Consumes: the frame from Task 3, guards from Task 4, `FidelityResult` from Task 5.
- Produces: `build_report(df, fidelity=None) -> str` (markdown).

**The report must run the guards, not merely have them available.** A guard that exists but is never called on the reporting path is decoration. `build_report` calls `assert_matched_budget`, `assert_comparable`, and `assert_distinct_seeds` before emitting anything, and refuses to place production-tier and reference-tier rows in one table.

- [ ] **Step 1: Write the failing test**

```python
# tests/benchmark/test_report.py
import pandas as pd
import pytest
from adaptive_roa.benchmark.report import build_report
from adaptive_roa.benchmark.fidelity import FidelityResult


def _df(**kw):
    base = dict(
        arm=["bnn_mfvi", "bnn_mfvi", "gp_reg", "gp_reg"],
        tier=["production"] * 4, acquisition=["ranked", "random"] * 2,
        seed=[42, 42, 43, 43], n_epochs=[10] * 4, accuracy=[0.91, 0.86, 0.88, 0.84],
    )
    base.update(kw)
    return pd.DataFrame(base)


def test_report_contains_every_arm_and_both_acquisition_modes():
    out = build_report(_df())
    assert "bnn_mfvi" in out and "gp_reg" in out
    assert "ranked" in out and "random" in out


def test_report_runs_the_budget_guard():
    with pytest.raises(ValueError, match="budget"):
        build_report(_df(n_epochs=[10, 10, 20, 20]))


def test_report_refuses_to_mix_tiers_in_one_table():
    with pytest.raises(ValueError, match="tier"):
        build_report(_df(tier=["production", "production", "reference", "reference"]))


def test_withheld_fidelity_prints_the_reason_not_a_number():
    fid = {"bnn_mfvi_reg": FidelityResult(
        available=False, reason="reference did not converge: rhat_max=91.4")}
    out = build_report(_df(), fidelity=fid)
    assert "91.4" in out
    assert "did not converge" in out


def test_available_fidelity_prints_the_number():
    fid = {"bnn_mfvi": FidelityResult(available=True, agreement=0.93,
                                      total_variation=0.04)}
    out = build_report(_df(), fidelity=fid)
    assert "0.93" in out and "0.04" in out


def test_paired_control_delta_is_reported():
    # Foong et al. found MFVI-driven active learning LOSING to random. Without
    # the paired delta surfaced, that finding is invisible.
    out = build_report(_df())
    assert "delta" in out.lower()
```

- [ ] **Step 2: Run to verify it fails**

Run: `./env/bin/python -m pytest tests/benchmark/test_report.py -v`
Expected: FAIL — `ModuleNotFoundError: adaptive_roa.benchmark.report`

- [ ] **Step 3: Implement**

```python
# adaptive_roa/benchmark/report.py
"""Tier-separated benchmark tables.

Calls the guards on the reporting path rather than merely importing them: a
guard that is never invoked where the number is produced is decoration.
"""
from __future__ import annotations

import pandas as pd

from .guards import assert_comparable, assert_distinct_seeds, assert_matched_budget

METRIC = "accuracy"


def build_report(df: pd.DataFrame, fidelity: dict | None = None) -> str:
    if df["tier"].nunique() > 1:
        raise ValueError(
            f"refusing to place tiers {sorted(df['tier'].unique())} in one "
            f"table: the reference tier carries posterior-fidelity claims at "
            f"[50,50] and the production tier carries downstream task results "
            f"at [256,512,256]. They are not rows of the same table."
        )
    assert_matched_budget(df)
    assert_comparable(df, METRIC)
    assert_distinct_seeds(df, METRIC)

    tier = df["tier"].iloc[0]
    lines = [f"# Benchmark report — {tier} tier", ""]

    pivot = df.pivot_table(index="arm", columns="acquisition",
                           values=METRIC, aggfunc="mean")
    lines += [f"## Downstream task ({METRIC})", "",
              "| arm | ranked | random | delta (ranked − random) |",
              "|---|---|---|---|"]
    for arm, row in pivot.iterrows():
        ranked = row.get("ranked", float("nan"))
        random_ = row.get("random", float("nan"))
        delta = ranked - random_
        lines.append(f"| {arm} | {ranked:.3f} | {random_:.3f} | {delta:+.3f} |")

    lines += ["", "A negative delta means adaptive acquisition LOST to random "
                  "selection for that arm — a reportable finding, not a bug "
                  "(cf. Foong et al., NeurIPS 2020).", ""]

    if fidelity:
        lines += ["## Posterior fidelity vs the HMC reference", "",
                  "| arm | agreement | total variation |", "|---|---|---|"]
        for arm, res in sorted(fidelity.items()):
            if res.available:
                lines.append(f"| {arm} | {res.agreement:.3f} | "
                             f"{res.total_variation:.3f} |")
            else:
                lines.append(f"| {arm} | withheld | {res.reason} |")
        lines.append("")

    return "\n".join(lines)
```

- [ ] **Step 4: Run tests**

Run: `./env/bin/python -m pytest tests/benchmark/test_report.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Run the full suite**

Run: `./env/bin/python -m pytest tests/ -q -o addopts=""`
Expected: all prior tests still pass (baseline 698 passed / 31 skipped), plus the new `tests/benchmark/` tests.

- [ ] **Step 6: Commit**

```bash
git add adaptive_roa/benchmark/report.py scripts/run_benchmark.py tests/benchmark/test_report.py
git commit -m "feat(benchmark): tier-separated report with guards on the reporting path"
```

---

### Task 8: Pilot campaign end to end

**Files:**
- Modify: `configs/benchmark/pilot.yaml` if the dry run surfaces a bad override
- Create: `docs/benchmark/pilot-run-log.md`

**Interfaces:**
- Consumes: everything above.
- Produces: a real report from real runs, and a record of what was launched.

This task is operational rather than code: it proves the harness works before ~450 runs are committed to it.

- [ ] **Step 1: Dry run the pilot manifest**

Run: `./env/bin/python scripts/run_benchmark.py configs/benchmark/pilot.yaml --exp-root ${EXP_DIR}/benchmark_pilot`
Expected: 126 runs enumerated, nothing submitted. Inspect several printed sbatch lines and confirm the Hydra overrides are ones you would have typed by hand — in particular that `mlp_det` carries `+experiment=mlp_det_baseline` and no `acquisition=` override.

- [ ] **Step 2: Launch a single-arm slice first**

Run the same command with `arms: [bnn_mfvi]` and `seeds: [42]` in a scratch manifest, with `--launch`. Two runs (ranked + random) on pendulum.
Expected: two jobs queued on `gpu-redhat`.

- [ ] **Step 3: Confirm idempotency against the live runs**

Once both finish, re-run the launcher without `--launch`.
Expected: `2 runs: 2 complete, 0 partial, 0 absent` and nothing to submit. **This is the check that makes preemption survivable** — if it reports anything else, stop and fix Task 2 before scaling up.

- [ ] **Step 4: Aggregate and report the slice**

```bash
./env/bin/python -c "
from adaptive_roa.benchmark.aggregate import collect_runs
from adaptive_roa.benchmark.report import build_report
import os
df = collect_runs(os.environ['EXP_DIR'] + '/benchmark_pilot')
df['tier'] = 'production'
print(build_report(df))
"
```
Expected: a two-row table with a `delta` column. If a guard raises, that is the harness working — record the reason in the run log and resolve it before scaling.

- [ ] **Step 5: Launch the full pilot and record it**

Write `docs/benchmark/pilot-run-log.md` with the manifest used, the commit SHA the campaign ran at, the submission date, and the job-ID range. **The commit SHA is what lets a future reader tell whether these results predate a correctness fix** — the failure the guards in Task 4 exist to catch.

- [ ] **Step 6: Commit**

```bash
git add configs/benchmark/pilot.yaml docs/benchmark/pilot-run-log.md
git commit -m "docs(benchmark): pilot campaign manifest and run log"
```

---

## Self-Review

**Spec coverage.** Paired random-acquisition control — Task 1 (expansion) and Task 7 (delta column). Matched training budgets — Task 4. Prior-scale sensitivity sweep — expressible as a manifest `overrides` entry (`predictor.bnn.prior_sigma=...`) with no code change; Task 1's `run_id` includes overrides, so sweep points are distinct runs. Tempering — same mechanism, labelled via a manifest field. GP fairness (sampling from the predictive including likelihood noise) is a property of the already-merged `gp_reg` arm, not this plan. Separatrix reporting — Task 6. Tier separation — Task 7.

**Deliberately deferred.** Automatic `rsync` of Amarel results back to iLab: the existing manual workflow is adequate and automating a cross-filesystem copy invites a destructive mistake for little gain. Plot generation: `scripts/compile_adaptive_metrics.py` already plots, and Task 3's frame feeds it.

**Resolved during self-review.** `assert_post_fix` needs each run to record its producing commit, and the engine was verified to record none. That is now Task 0, ahead of everything else, because Task 4's staleness guards are inert without it.

**Type consistency.** `RunSpec.overrides` is a `tuple` throughout (not a list) so the frozen dataclass stays hashable and `run_id` stays stable. `FidelityResult` fields are `None` rather than `nan` when withheld, so a withheld result cannot be silently averaged into a table.
