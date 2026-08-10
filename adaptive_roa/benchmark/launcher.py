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
