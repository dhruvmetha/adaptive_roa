"""Read per-point eval artifacts off disk and drive separatrix + fidelity.

``separatrix.py`` and ``fidelity.py`` are both good modules that nothing
called. ``conditioned_metrics`` had zero call sites outside its own test;
``fidelity_vs_reference`` had zero anywhere, ``build_report`` accepted a
``fidelity=`` dict that no shipped code ever constructed, and nothing read
``hmc_diagnostics.json`` or a reference's predictive probabilities off
disk. By ``guards.py``'s own standard -- "a guard that is never invoked
where the number is produced is decoration" -- the near-boundary slice and
the posterior-fidelity table were two decorations. This module is the
missing half: it turns run directories into the arrays those two functions
take, so both are reachable from ``run_benchmark.py report``.

What is on disk (verified against a real run under
``outputs/partx_pendulum_dev/epoch_000/``):

- ``<run>/epoch_NNN/full_roa_per_point.npz`` -- ``start_states`` [n, d],
  ``p_success`` [n] in [0, 1], ``p_failure``, ``p_invalid``,
  ``true_labels`` [n] in {-1, 1}, plus the scalar ``lambda_star`` / ``delta``
  / ``attractor_radius`` the decision rule was calibrated at.
- ``<run>/epoch_NNN/checkpoints/hmc_diagnostics.json`` -- written once per
  HMC run by ``hmc_trainer.py``: ``rhat_max``, ``converged``, per-chain
  diagnostics, the HMC-vs-HMC ceiling.

Three things this module does NOT do, each stated rather than papered over:

1. **It uses raw ``p_success``**, never ``p_success / (p_success +
   p_failure)``. Renormalizing away ``p_invalid`` inflates confidence on
   exactly the states the model is least sure about (see the project's own
   stochastic-pendulum findings). ``_require_probabilities`` still applies:
   a value outside [0, 1] is refused, not clamped.
2. **It thresholds predictions at 0.5**, because that is what
   ``separatrix.conditioned_metrics`` does. That is NOT the calibrated
   lambda/delta rule the ``lambda_delta.*`` columns are computed under, so
   the conditioned accuracies are a different quantity from the headline
   table's and must not be read as a decomposition of it. The rendered
   section says so.
3. **It does not construct a ``DynamicalSystem`` to normalize states.**
   Every ``systems/*.py`` class loads its bounds from a
   ``dataset_description.json`` under the dataset root, so building one
   requires the training data to be present on whatever machine is
   rendering the report -- which, for a campaign whose results are rsynced
   off Amarel, it generally is not. States are therefore passed in the
   coordinates the evaluator recorded, and ``separatrix``'s own
   ``_guard_unnormalized_scale`` is what stands between that and a
   meaningless k-NN band: it REFUSES inputs whose per-dimension spans
   differ by more than 50x. A caller that does have the datasets can pass
   ``system_factory=`` and get exact normalization instead.

Per-run failures (a missing ``full_roa_per_point.npz``, a scale refusal)
are recorded IN the returned row as a ``refused`` reason and rendered
verbatim in the table, the same way ``fidelity.py`` renders a withheld
result -- never as a blank cell, and never dropped. A failure of the
REFERENCE itself (no ``hmc_diagnostics.json``, or a shape mismatch meaning
two arms were evaluated on different grids) propagates as an exception,
because it invalidates every row of the table at once rather than one.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .fidelity import FidelityResult, fidelity_vs_reference
from .separatrix import conditioned_metrics

__all__ = [
    "PER_POINT_FILE",
    "HMC_DIAGNOSTICS_FILE",
    "load_per_point",
    "separatrix_table",
    "fidelity_table",
]

PER_POINT_FILE = "full_roa_per_point.npz"
HMC_DIAGNOSTICS_FILE = Path("checkpoints") / "hmc_diagnostics.json"

_CELL_COLUMNS = ("system", "arm", "acquisition")


def epoch_dir(exp_root, run_id: str, epoch) -> Path:
    """``<exp_root>/<run_id>/epoch_NNN`` -- the engine's own layout."""
    return Path(exp_root) / str(run_id) / f"epoch_{int(epoch):03d}"


def _labels_to_success(labels: np.ndarray) -> np.ndarray:
    """{-1, 1} or {0, 1} -> {0, 1}. Anything else is refused, not coerced.

    ``full_roa_per_point.npz`` stores ``true_labels`` in the {-1, 1}
    convention; ``separatrix.conditioned_metrics`` requires {0, 1} and
    refuses anything else precisely because ``labels.astype(bool)`` maps -1
    and 1 alike to True, merging both classes. The remap belongs here, done
    explicitly and once.
    """
    unique = set(np.unique(labels).tolist())
    if unique <= {0, 1}:
        return labels.astype(int)
    if unique <= {-1, 1}:
        return (labels == 1).astype(int)
    raise ValueError(
        f"true_labels carries values {sorted(unique)}; expected the "
        f"{{-1, 1}} convention full_roa_per_point.npz writes, or {{0, 1}}. "
        f"Refusing to guess which of them means 'success'."
    )


def load_per_point(directory) -> dict:
    """states / probs / labels from one epoch's ``full_roa_per_point.npz``."""
    path = Path(directory) / PER_POINT_FILE
    if not path.is_file():
        raise FileNotFoundError(f"no {PER_POINT_FILE} at {path}")
    with np.load(path) as payload:
        missing = [k for k in ("start_states", "p_success", "true_labels")
                   if k not in payload.files]
        if missing:
            raise ValueError(
                f"{path} is missing {missing}; present: {sorted(payload.files)}"
            )
        states = np.asarray(payload["start_states"], dtype=float)
        probs = np.asarray(payload["p_success"], dtype=float)
        labels = np.asarray(payload["true_labels"])
    return {"states": states, "probs": probs,
            "labels": _labels_to_success(labels)}


def separatrix_table(df, exp_root, *, k: int = 5, system_factory=None) -> list[dict]:
    """Conditioned accuracy per (system, arm, acquisition), from disk.

    Aggregate accuracy is dominated by basin interiors where every arm is
    correct; arms differ where the outcome flips, which is the regime
    adaptive acquisition exists to resolve. ``df`` should already be reduced
    to the epochs being reported (``report.select_epochs``), so the rows
    here are the same rows the headline table summarizes.

    Runs in the same cell are averaged over seeds. A run this function
    cannot evaluate contributes its reason to the cell's ``refused`` list
    instead of being dropped.

    The k-NN band is recomputed per run rather than cached across the arms
    that share an eval grid. That is O(n log n) per run and deliberate: the
    band depends on (states, labels, k), and a cache keyed on anything less
    than all three is a correctness bug waiting to be introduced, while
    reusing one across runs would mean reimplementing
    ``conditioned_metrics``' own accuracy arithmetic here and letting the
    two copies drift. On the pendulum's ~50k-point grid this is a fraction
    of a second per run.
    """
    required = {"run_id", "epoch", "arm"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            f"separatrix_table needs {missing} to locate each run's "
            f"{PER_POINT_FILE} on disk; columns present: {sorted(df.columns)}"
        )

    cell_cols = [c for c in _CELL_COLUMNS if c in df.columns]
    runs = df[[*cell_cols, "run_id", "epoch"]].drop_duplicates()
    cells: dict[tuple, dict] = {}

    for row in runs.to_dict("records"):
        key = tuple(row[c] for c in cell_cols)
        cell = cells.setdefault(key, {"metrics": [], "refused": []})
        directory = epoch_dir(exp_root, row["run_id"], row["epoch"])
        try:
            data = load_per_point(directory)
            system = (system_factory(row["system"])
                      if system_factory is not None and "system" in row else None)
            metrics = conditioned_metrics(
                data["states"], data["labels"], data["probs"], k=k,
                system=system)
        except (FileNotFoundError, ValueError) as exc:
            cell["refused"].append(f"{row['run_id']}: {exc}")
            continue
        cell["metrics"].append(metrics)

    table = []
    for key, cell in sorted(cells.items(), key=repr):
        entry = dict(zip(cell_cols, key))
        entry["k"] = k
        entry["n_runs"] = len(cell["metrics"])
        entry["refused"] = "; ".join(cell["refused"]) or None
        for field in ("overall", "near_boundary", "interior"):
            values = [m[field] for m in cell["metrics"]]
            entry[field] = float(np.nanmean(values)) if values else float("nan")
        for field in ("n_near", "n_interior"):
            values = [m[field] for m in cell["metrics"]]
            entry[field] = int(np.mean(values)) if values else 0
        table.append(entry)
    return table


def _fidelity_label(arm, system, seed, disambiguate: bool) -> str:
    if not disambiguate:
        return str(arm)
    return f"{arm} [system={system}, seed={seed}]"


def fidelity_table(df, exp_root, *, reference_arm: str,
                   rhat_threshold: float = 1.1) -> dict[str, FidelityResult]:
    """Posterior fidelity of every arm against ``reference_arm``, from disk.

    The reference's own ``hmc_diagnostics.json`` gates the whole thing: a
    number computed against a posterior that did not mix is not a weak
    result, it is a meaningless one, and ``fidelity_vs_reference`` withholds
    it rather than reporting it with a caveat nobody carries forward. Arms
    are paired with the reference run that shares their (system, seed);
    failing that, with the system's single reference run if there is exactly
    one. An arm with no usable pairing is WITHHELD with that as the reason,
    never omitted.
    """
    required = {"run_id", "epoch", "arm"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            f"fidelity_table needs {missing} to locate each run's "
            f"{PER_POINT_FILE} on disk; columns present: {sorted(df.columns)}"
        )
    frame = df.copy()
    for column in ("system", "seed"):
        if column not in frame.columns:
            frame[column] = None

    runs = frame[["arm", "system", "seed", "run_id", "epoch"]].drop_duplicates()
    references = runs[runs["arm"] == reference_arm]
    if references.empty:
        raise ValueError(
            f"no runs for reference arm {reference_arm!r} in this frame; "
            f"arms present: {sorted(runs['arm'].unique().tolist())}. A "
            f"fidelity table without its reference is not a table."
        )

    others = runs[runs["arm"] != reference_arm]
    disambiguate = (others.groupby("arm")[["system", "seed"]]
                    .nunique().max().max() > 1) if not others.empty else False

    results: dict[str, FidelityResult] = {}
    for row in others.to_dict("records"):
        label = _fidelity_label(row["arm"], row["system"], row["seed"],
                                disambiguate)
        same_system = references[references["system"] == row["system"]]
        paired = same_system[same_system["seed"] == row["seed"]]
        if paired.empty:
            paired = same_system if len(same_system) == 1 else paired
        if paired.empty:
            results[label] = FidelityResult(
                available=False,
                reason=(f"no {reference_arm} reference run for "
                        f"system={row['system']!r} seed={row['seed']!r}"))
            continue
        ref = paired.iloc[0]

        ref_dir = epoch_dir(exp_root, ref["run_id"], ref["epoch"])
        diagnostics_path = ref_dir / HMC_DIAGNOSTICS_FILE
        if not diagnostics_path.is_file():
            # The reference's convergence check is the gate itself; its
            # absence invalidates every row, not this one.
            raise FileNotFoundError(
                f"reference run {ref['run_id']!r} has no "
                f"{HMC_DIAGNOSTICS_FILE} at {diagnostics_path}. Refusing to "
                f"assume the reference converged -- that assumption is what "
                f"this gate exists to prevent."
            )
        diagnostics = json.loads(diagnostics_path.read_text())

        try:
            approx = load_per_point(epoch_dir(exp_root, row["run_id"], row["epoch"]))
        except (FileNotFoundError, ValueError) as exc:
            results[label] = FidelityResult(available=False, reason=str(exc))
            continue
        reference = load_per_point(ref_dir)

        results[label] = fidelity_vs_reference(
            approx["probs"], reference["probs"], diagnostics,
            rhat_threshold=rhat_threshold)
    return results
