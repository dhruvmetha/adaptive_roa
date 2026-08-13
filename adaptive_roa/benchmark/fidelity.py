"""Posterior fidelity against the HMC reference, gated on the reference's own
convergence.

A fidelity number computed against a reference that did not converge is not a
weak result -- it is a meaningless one, and it is indistinguishable from a
real one in a table. So this module withholds the number and reports why,
rather than emitting a caveat that nobody carries forward past the printed
value. That is the entire design intent, implemented as a hard gate: a
withheld ``FidelityResult`` has ``available=False`` and every numeric field
``None`` (never ``nan``), so it cannot be silently averaged into a table the
way a NaN can slip through ``pandas`` aggregation.

Status of the two reference arms this gate exists for (see
``adaptive_roa/adaptive_v2/trainers/hmc_trainer.py``, ``RHAT_THRESHOLD =
1.1``, and ``docs/superpowers/plans/2026-08-10-benchmark-orchestration.md``):
the outcome-head reference (``hmc``) converges, function-space
``rhat_max`` ~1.0-1.19; the final-state reference (``hmc_reg``) does not,
``rhat_max`` measured 7.9-91 depending on seed. Any ``fidelity_vs_reference``
call against ``hmc_reg``'s diagnostics is therefore expected to return
``available=False`` until that sampler is fixed -- that is this gate working
as intended, not a bug in this module.

``hmc_diagnostics.json`` (written once per HMC run, next to the checkpoint)
records the convergence check as ``rhat_max`` (float) and ``converged``
(bool). This module reads ``rhat_max`` and recomputes the pass/fail decision
against ITS OWN ``rhat_threshold`` argument rather than trusting the
``converged`` field: ``converged`` was computed at write time against
whatever threshold that training run happened to be configured with
(``predictor.hmc.rhat_threshold``, defaulting to the same 1.1), and a caller
here may legitimately want a different -- typically stricter -- bar for a
benchmark table than whatever a given run was configured with. Recomputing
locally is what makes ``rhat_threshold`` in this function's signature actually
mean something (see ``test_threshold_is_honoured``); trusting the stored
``converged`` bool instead would silently ignore that argument.

Naming note on ``total_variation``: the value returned here is
``mean(|approx - ref|)`` over per-point Bernoulli probabilities. For a single
pair of Bernoulli(p), Bernoulli(q), the true total variation distance IS
``|p - q|``, so averaging that over points is an average TV, not one TV
number for a joint predictive over all points at once -- a defensible
quantity, but the name alone doesn't say "averaged". Kept as
``total_variation`` anyway (not renamed to e.g. ``mean_total_variation``)
because it is the exact name, and the exact formula, already established for
this identical computation in
``adaptive_roa/predictors/hmc/diagnostics.py`` (``total_variation``, the
building block of ``hmc_vs_hmc_ceiling``). Introducing a second name for the
same quantity computed the same way would be the inconsistency, not this one.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Reused directly, not reimplemented: one definition of "is this actually a
# probability" in this subsystem, not two that can drift apart. Underscore
# name means module-private by convention only -- Python does not enforce it,
# and duck typing (`.min()`/`.max()`/`float()`) means it works unmodified on
# the plain numpy arrays this module passes it, despite its torch.Tensor type
# hint (verified: raises correctly on values > 1, < 0, and NaN inputs here).
from adaptive_roa.predictors.hmc.diagnostics import _require_probabilities


@dataclass(frozen=True)
class FidelityResult:
    """Structurally unusable as a number when withheld.

    ``available=False`` always pairs with ``agreement=None`` and
    ``total_variation=None`` -- not ``nan`` -- specifically so a caller that
    forgets to check ``available`` and instead does something like
    ``df["fidelity"].mean()`` gets a ``TypeError`` from the ``None``, not a
    silently-computed average that quietly drops the withheld rows the way
    ``nan`` would under most pandas/numpy reductions.
    """

    available: bool
    reason: str | None = None
    agreement: float | None = None
    total_variation: float | None = None


def fidelity_vs_reference(approx_probs, ref_probs, ref_diagnostics,
                          rhat_threshold: float = 1.1) -> FidelityResult:
    """Compare an approximation's predictive probabilities to the HMC reference.

    Refuses (raises ``ValueError``) rather than silently proceeding when:
    - ``approx_probs`` and ``ref_probs`` disagree in shape -- they must be
      predictions at the same evaluation points, and a shape mismatch means
      they are not.
    - either array is not actually a probability (a value outside ``[0, 1]``,
      or NaN) -- via ``_require_probabilities``, reused from
      ``adaptive_roa/predictors/hmc/diagnostics.py`` rather than
      reimplemented. This is not a generic input check: this exact mistake
      already shipped in this subsystem once, when ``hmc_vs_hmc_ceiling`` was
      fed a raw, unbounded network output as if it were a probability and
      reported ``total_variation = 3.89`` -- a value a ``[0, 1]`` distance
      cannot produce -- as a plausible-looking number. The fix there was to
      raise, never clamp: clamping would turn an invalid input into a
      plausible in-range output, which is the exact failure mode this guard
      exists to prevent, so it is not offered as an option here either.
    - ``ref_diagnostics`` carries no ``rhat_max``. A missing diagnostic is
      refused, not treated as "assume converged": that assumption is exactly
      what this gate exists to prevent, and a reference run that somehow
      shipped without writing its own convergence check is a run this
      function cannot vouch for.

    Withholds (returns ``available=False``, no numbers) rather than raising
    when the reference's own diagnostic says it did not mix: ``rhat_max`` is
    non-finite or exceeds ``rhat_threshold``. This is not an error in the
    caller's inputs -- the shapes and diagnostics are all present and
    well-formed -- it is a property of the reference posterior itself, so it
    is reported through the result, not an exception.
    """
    approx = np.asarray(approx_probs, dtype=float)
    ref = np.asarray(ref_probs, dtype=float)
    if approx.shape != ref.shape:
        raise ValueError(
            f"shape mismatch: approx {approx.shape} vs reference {ref.shape}. "
            f"Both must be predictive probabilities at the same evaluation "
            f"points."
        )
    _require_probabilities(approx)
    _require_probabilities(ref)
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

    # Mean ABSOLUTE DIFFERENCE across points, not one joint TV distance for
    # the whole evaluation set -- see the module docstring's naming note.
    tv = float(np.mean(np.abs(approx - ref)))
    agreement = float(np.mean((approx >= 0.5) == (ref >= 0.5)))
    return FidelityResult(available=True, agreement=agreement,
                          total_variation=tv)
