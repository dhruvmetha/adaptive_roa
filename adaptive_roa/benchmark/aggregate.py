"""Collect on-disk run artifacts into one tidy frame.

Provenance (commit, seed, arm, system, tier) is attached per ROW, not per
report, so a pool mixing pre- and post-fix vintages, or mismatched budgets, is
detectable by the guard layer (Task 4) rather than averaging silently into a
plausible number.

Malformed artifacts: an epoch whose artifacts_v2.json is missing, empty, or
fails to parse is SKIPPED, not raised. This matches the precedent set by
launcher.completed_epochs(), which already treats a truncated artifact as "not
finished" rather than an error -- the `general` account preempts jobs
silently, so a bad file here is the routine case, not the exceptional one, and
a single corrupt epoch out of a ~450-run campaign must not block aggregation
of the other 449. The cost is that a run whose LAST epoch is corrupt looks
shorter than it actually ran; Task 4's guards can detect that by comparing the
per-run row count against the `n_epochs` column this module attaches (the
budget recorded in that run's own Hydra config), which is a job for the guard
layer, not for this reader.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

# Columns every row carries regardless of which metrics a particular eval run
# happened to produce. Declared explicitly so an empty campaign still returns
# a frame guards can index into (df["commit"], df["arm"], ...) without a
# KeyError, and so provenance columns always sort first.
PROVENANCE_COLUMNS = [
    "run_id", "arm", "system", "tier", "acquisition", "seed", "epoch",
    "n_epochs", "commit",
]


def _resolve_arm(cfg: dict) -> str | None:
    """Arm identity from the copied Hydra config, never from the directory name.

    Mirrors ``adaptive_roa.probabilistic_classifier.export.resolve_predictor_name``
    (the sibling loader referenced by the Task 3 brief) rather than importing
    it: that module also imports torch and registers probabilistic-classifier
    subclasses as an import side effect, which this lightweight, frequently-run
    aggregator has no reason to pay for.

    Current run configs (configs/adaptive_v2/predictor/*.yaml) record
    `predictor` as a dict with an explicit `name` (e.g. classifier.yaml
    declares `predictor.name: mlp`). Older, pre-`name` run configs record
    `predictor` as a bare STRING -- the Hydra config group filename that was
    selected (e.g. "classifier") -- with the arm's own settings block
    flattened at the top level instead. For those, fall back to the family
    `type` tag the same way the sibling loader does.
    """
    predictor = cfg.get("predictor")
    if predictor is None:
        return None
    if isinstance(predictor, str):
        return predictor
    name = predictor.get("name")
    if name:
        return str(name)
    family = predictor.get("type")
    return str(family) if family else None


def _resolve_system(cfg: dict) -> str | None:
    """System identity from the copied Hydra config.

    Every configs/adaptive_v2/system/*.yaml sets `adaptive_v2.system_name` to
    the manifest's own `system=` value (e.g. "pendulum", "cartpole_pybullet",
    "quadrotor3d") -- that is the authoritative source on real run configs.
    The top-level `system:` block itself carries only `_target_` /
    `dataset_dir` there, with NO `name` key, so reading `system.name` first
    would silently resolve every production row to None. `system.name` is
    still checked as a fallback, for schemas (including this module's own
    test fixtures) that record identity there directly instead.
    """
    adaptive_v2 = cfg.get("adaptive_v2")
    if isinstance(adaptive_v2, dict):
        system_name = adaptive_v2.get("system_name")
        if system_name:
            return str(system_name)
    system = cfg.get("system")
    if isinstance(system, dict):
        name = system.get("name")
        if name:
            return str(name)
    return None


def _resolve_tier(epoch_dir: Path) -> str:
    """Tier ("production" or "reference"), read from the recorded CLI overrides.

    Nothing in the composed config.yaml itself says "reference": the
    reference tier (configs/adaptive_v2/experiment/reference_tier.yaml) only
    pins hidden_dims/activation/pos_weight deeper inside the predictor block,
    values a production run could equally set on its own, so they are not a
    reliable signal. The one unambiguous marker is the
    `+experiment=reference_tier` override RunSpec.hydra_overrides() emits
    for tier="reference" (manifest.py), preserved verbatim by Hydra in
    `.hydra/overrides.yaml` alongside config.yaml. Its absence -- including
    when overrides.yaml itself is missing, e.g. a hand-built fixture -- means
    "production", the manifest's other TIERS value and the default every
    non-reference run composes under.
    """
    overrides_path = epoch_dir / ".hydra" / "overrides.yaml"
    if not overrides_path.is_file():
        return "production"
    try:
        overrides = yaml.safe_load(overrides_path.read_text()) or []
    except yaml.YAMLError:
        return "production"
    for ov in overrides:
        key, _, value = str(ov).partition("=")
        if key.lstrip("+") == "experiment" and value == "reference_tier":
            return "reference"
    return "production"


def _flatten_metrics(d, prefix: str = "") -> dict:
    """Recursively flatten a metrics dict into scalar leaf columns.

    Real `eval_metrics` payloads (adaptive_v2/eval/full_roa.py's
    FullROAEvaluator) nest the numbers anyone actually wants to compare --
    accuracy, f1, precision, recall -- one level below variant names like
    `lambda_delta` / `fixed_threshold` / `conservative_qhat`. A shallow,
    top-level-only copy would silently produce a frame with NONE of those
    columns while still passing a test built on a flat fixture. Nested keys
    are joined with "." (e.g. "lambda_delta.accuracy"); non-numeric leaves
    (strings, None, lists, bools) are dropped rather than coerced.
    """
    out = {}
    for k, v in (d or {}).items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten_metrics(v, prefix=f"{key}."))
        elif isinstance(v, bool):
            continue
        elif isinstance(v, (int, float)):
            out[key] = v
    return out


def _epoch_row(epoch_dir: Path) -> dict | None:
    art = epoch_dir / "artifacts_v2.json"
    if not art.is_file():
        return None                      # preempted: directory without artifact
    try:
        payload = json.loads(art.read_text())
    except (json.JSONDecodeError, OSError):
        # Truncated or unreadable: a preempted write, not a finished epoch.
        # See module docstring for why this is skipped rather than raised.
        return None

    cfg = {}
    cfg_path = epoch_dir / ".hydra" / "config.yaml"
    if cfg_path.is_file():
        try:
            cfg = yaml.safe_load(cfg_path.read_text()) or {}
        except yaml.YAMLError:
            cfg = {}

    row = _flatten_metrics(payload.get("eval_metrics"))
    # Provenance is assigned AFTER the metrics so it always wins on a name
    # collision (none observed in practice, but identity columns must never
    # be silently clobbered by an oddly-named metric).
    row.update({
        "epoch": payload.get("epoch"),
        "arm": _resolve_arm(cfg),
        "system": _resolve_system(cfg),
        "tier": _resolve_tier(epoch_dir),
        "acquisition": payload.get("sampling_mode"),
        "seed": cfg.get("seed"),
        "n_epochs": cfg.get("n_epochs"),
        "commit": (payload.get("extra") or {}).get("commit"),
    })
    return row


def collect_runs(exp_root) -> pd.DataFrame:
    """One row per (run_id, epoch), with provenance attached at read time.

    `run_id` is the run's output directory name (trusted verbatim -- it is
    what launcher.sbatch_command wrote the run under, and is never parsed for
    meaning). Every other identity field (arm, system, tier, seed, commit) is
    read from the `.hydra` config and artifact JSON the engine copied into
    that epoch directory, so renaming the directory cannot relabel a run.
    """
    exp_root = Path(exp_root)
    rows = []
    if exp_root.is_dir():
        for run_dir in sorted(p for p in exp_root.iterdir() if p.is_dir()):
            for epoch_dir in sorted(run_dir.glob("epoch_*")):
                row = _epoch_row(epoch_dir)
                if row is not None:
                    rows.append({"run_id": run_dir.name, **row})

    if not rows:
        return pd.DataFrame(columns=PROVENANCE_COLUMNS)

    df = pd.DataFrame(rows)
    ordered = PROVENANCE_COLUMNS + [c for c in df.columns if c not in PROVENANCE_COLUMNS]
    return df[ordered]
