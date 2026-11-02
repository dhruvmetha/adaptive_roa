"""Which decision rule a run's thresholds follow, for the post-hoc tools.

Idea 6 arms (threshold=fpr_cap, the *_fprcap1pct cells) fit tau_s/tau_f/tau_inv, and
their predictions_thrfit.npz holds the K=20 val states that fit saw, not the
lambda*/delta* threshold-fit pairs. A lambda/delta refit or replay on such a run would
read the wrong file and the wrong thresholds, so the lambda/delta-only tools refuse it
and score them with score_runs.py --thresholds fprcap instead.
"""
from __future__ import annotations

import json
from pathlib import Path


def uses_tau_rule(run_dir) -> bool:
    """True when the run's thresholds are the FPR-capped tau rule rather than lambda*/delta*."""
    run = Path(run_dir)
    if "fprcap" in run.name:
        return True
    for art in sorted(run.glob("epoch_*/artifacts_v2.json")):
        try:
            ts = json.loads(art.read_text()).get("threshold_state") or {}
        except (OSError, json.JSONDecodeError):
            continue
        return ts.get("tau_s") is not None
    return False


def require_lambda_delta(run_dir, tool: str) -> None:
    """Raise for a tau-rule run: ``tool`` reads lambda*/delta* and would misread it."""
    if uses_tau_rule(run_dir):
        raise ValueError(f"{tool}: {Path(run_dir).name} uses the FPR-capped tau rule, not lambda*/delta*; "
                         "score it with score_runs.py --thresholds fprcap")
