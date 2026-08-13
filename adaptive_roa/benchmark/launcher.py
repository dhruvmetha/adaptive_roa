"""Idempotent campaign launcher.

The `general` account preempts jobs silently, so relaunching a campaign is the
NORMAL path, not an error path. Completion is judged by counting
artifacts_v2.json files: a preempted job leaves epoch directories behind, and a
healthy job's log stays quiet for long stretches, so neither directories nor
logs are evidence of progress.

engine.py now writes each artifact atomically (temp file + os.replace), so a
preempted job can no longer leave a half-written artifacts_v2.json on disk.
completed_epochs() still parses every artifact rather than trusting file
presence alone, as defense in depth: files already on disk from before that
fix -- or written by any other path that is not atomic -- can still be
truncated, and counting a truncated file as done would skip a dead run
forever, exactly the failure this launcher exists to prevent.
"""
from __future__ import annotations

import json
from pathlib import Path

from .manifest import RunSpec

SBATCH_TEMPLATE = "scripts/sbatch_amarel.sh"


def completed_epochs(run_dir: Path) -> int:
    if not run_dir.exists():
        return 0
    count = 0
    for p in run_dir.glob("epoch_*/artifacts_v2.json"):
        if not p.is_file():
            continue
        try:
            with open(p) as f:
                json.load(f)
        except (json.JSONDecodeError, OSError):
            # Truncated or unreadable: a preempted write, not a finished epoch.
            continue
        count += 1
    return count


def run_state(spec: RunSpec, exp_root: Path) -> str:
    done = completed_epochs(Path(exp_root) / spec.run_id)
    if done == 0:
        return "absent"
    return "complete" if done >= spec.n_epochs else "partial"


def plan_launch(specs, exp_root):
    """Return (specs_to_launch, {run_id: state}). Complete runs are skipped.

    Deduplicated by run_id, keeping first-seen order. n_epochs is deliberately
    excluded from run_id (extending a run resumes it rather than starting a
    new one), so two specs that differ only in n_epochs collide on the same
    run_id and the same output_dir. Without dedup both would be queued and two
    jobs would write into that directory concurrently. Keeping first-seen order
    (rather than e.g. last-seen, or silently overwriting states[run_id]) keeps
    submission order deterministic, which matters when reasoning about a
    partially-launched campaign.
    """
    exp_root = Path(exp_root)
    states, to_launch = {}, []
    seen_run_ids = set()
    for spec in specs:
        if spec.run_id in seen_run_ids:
            continue
        seen_run_ids.add(spec.run_id)
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
