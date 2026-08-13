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
of the other 449.

The cost: a run whose LAST epoch is corrupt then looks IDENTICAL to a run
that simply has not gotten there yet -- both produce fewer rows than
`n_epochs` implies, and per launcher.py's own docs "still running" is the
campaign's normal steady state, not an edge case, so row count alone cannot
tell the two apart. That distinction is carried explicitly instead: every row
also gets `run_complete`, True iff the engine's own end-of-run marker
(`final_results.json`, written once by AdaptiveEngine.run() after its epoch
loop finishes, directly in the run directory -- see engine.py) exists and
parses. A downstream guard can then treat "short AND run_complete=False" as
still in progress, and "short AND run_complete=True" as a run that actually
finished early or lost its tail to corruption.

`run_complete` does not close the whole gap, though, and it takes TWO
counters to close it, not one -- an epoch can be lost in two different
places:

- `n_epochs_collected`: how many epoch rows this run contributed at all,
  i.e. how many epoch directories held an artifacts_v2.json that PARSED.
  A corrupt or never-written interior artifact is skipped above and shows
  up only here (`run_complete` stays True, because the run still reached
  its last epoch).
- `n_epochs_evaluated`: how many of those rows carried at least one numeric
  metric. An epoch on which eval did not run writes a perfectly valid
  artifact with `eval_metrics: {}`; it produces a row, so
  `n_epochs_collected` counts it, but every metric on that row is NaN and
  `pivot_table` silently skips it. Without this second counter a run that
  lost a third of its metric population is indistinguishable from a
  complete one -- and the report would state, in as many words, that every
  run contributed its full number of epochs.

Both are PROVENANCE columns, not metrics: guards must never treat either as
a quantity to compare across arms.
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
    "n_epochs", "n_epochs_collected", "n_epochs_evaluated", "run_complete",
    "commit",
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
    """Recursively flatten a metrics dict into scalar leaf columns, to any depth.

    Real `eval_metrics` payloads (adaptive_v2/eval/full_roa.py's
    FullROAEvaluator) nest the numbers anyone actually wants to compare --
    accuracy, f1, precision, recall -- one level below variant names like
    `lambda_delta` / `fixed_threshold` / `conservative_qhat`, and deeper still
    for e.g. `endpoint_errors.mean` under a regression arm. A shallow,
    top-level-only copy would silently produce a frame with NONE of those
    columns while still passing a test built on a flat fixture, so this
    recurses without a depth cap. Nested keys are joined with "."
    (e.g. "lambda_delta.accuracy").

    Deliberately DROPPED, not coerced:
      - non-numeric leaves (strings, None, bools);
      - list-valued leaves, e.g. `_compute_geodesic_error_stats`'s
        `mean_per_dim` / `median_per_dim` / `variance_per_dim` (one float per
        state dimension). A tidy (run_id, epoch) row is one scalar per column;
        forcing a per-dimension vector into that shape would need either N
        further columns whose count varies by system (pendulum is 2D,
        quadrotor3d is far more), or silently keeping only one element. Both
        are worse than the current, explicit drop. Comparing per-dimension
        error is left to a reader that opens the source artifact directly.

    Raises ValueError on a dotted-key collision: a literal top-level key
    containing "." (e.g. "a.b") that coincides with the flattened name of a
    nested key (e.g. {"a": {"b": ...}} also produces "a.b"). Silently
    picking whichever value happens to be visited last would make this
    frame's content depend on dict iteration order -- for a frame that Task
    4's guards treat as ground truth, a loud failure here is far preferable
    to quietly losing one of the two values.
    """
    out = {}
    for k, v in (d or {}).items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            leaves = _flatten_metrics(v, prefix=f"{key}.")
        elif isinstance(v, bool):
            continue
        elif isinstance(v, (int, float)):
            leaves = {key: v}
        else:
            continue                     # strings, None, lists: dropped, not coerced
        for leaf_key, leaf_value in leaves.items():
            if leaf_key in out:
                raise ValueError(
                    f"metric key collision while flattening eval_metrics: "
                    f"{leaf_key!r} is produced twice -- once as a literal key "
                    f"and once via nested flattening (or twice via nested "
                    f"flattening from two different parents). Rename the "
                    f"source key so flattening is unambiguous."
                )
            out[leaf_key] = leaf_value
    return out


def _run_is_complete(run_dir: Path) -> bool:
    """True iff the engine's own end-of-run marker exists and parses.

    AdaptiveEngine.run() writes `final_results.json` directly in the run
    directory exactly once, after its epoch loop finishes (engine.py,
    alongside the per-epoch `epoch_*/artifacts_v2.json` writes). It is the
    ONLY signal that says "this run reached its configured last epoch" --
    an epoch directory's presence, or even every epoch up to some point
    having a valid `artifacts_v2.json`, means only that those epochs
    finished, not that the run itself is done. Malformed the same way an
    epoch artifact can be (a preempted write): treated as absent, not raised.
    """
    final = run_dir / "final_results.json"
    if not final.is_file():
        return False
    try:
        json.loads(final.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    return True


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
    `run_complete` is computed once per run directory (see
    `_run_is_complete`) and copied onto every row that run contributes, as
    are `n_epochs_collected` (rows produced) and `n_epochs_evaluated` (rows
    that carried at least one numeric metric) -- the two counters that make
    a corrupt interior epoch and an eval-less interior epoch respectively
    visible at all. See the module docstring for why one counter is not
    enough.
    """
    exp_root = Path(exp_root)
    rows = []
    if exp_root.is_dir():
        for run_dir in sorted(p for p in exp_root.iterdir() if p.is_dir()):
            run_complete = _run_is_complete(run_dir)
            run_rows = []
            for epoch_dir in sorted(run_dir.glob("epoch_*")):
                row = _epoch_row(epoch_dir)
                if row is not None:
                    row["run_complete"] = run_complete
                    run_rows.append({"run_id": run_dir.name, **row})
            # Counted AFTER the loop, over the rows that SURVIVED: an epoch
            # whose artifact is missing or unparseable is skipped above, and
            # these columns are the only record that it happened. A row's
            # metric columns are whatever _flatten_metrics produced, i.e.
            # every key that is NOT one of PROVENANCE_COLUMNS -- an empty
            # set means eval wrote nothing numeric for that epoch.
            provenance = set(PROVENANCE_COLUMNS)
            evaluated = sum(1 for row in run_rows if set(row) - provenance)
            for row in run_rows:
                row["n_epochs_collected"] = len(run_rows)
                row["n_epochs_evaluated"] = evaluated
            rows.extend(run_rows)

    if not rows:
        return pd.DataFrame(columns=PROVENANCE_COLUMNS)

    df = pd.DataFrame(rows)
    ordered = PROVENANCE_COLUMNS + [c for c in df.columns if c not in PROVENANCE_COLUMNS]
    return df[ordered]
