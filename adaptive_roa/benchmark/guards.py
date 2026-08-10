"""Guards that refuse to produce a number from incomparable inputs.

Every function here RAISES. None warns, filters silently, or substitutes a
default: the failure mode these exist to prevent is a confident wrong table,
and a warning in a log is not a defense against that -- nobody reads it, and
the number still ships.

Each guard below corresponds to a way this benchmark has already produced, or
come close to producing, a confident wrong number:

- ``assert_matched_budget``: an arm compared at a shorter training budget than
  its rivals -- weak baselines are the characteristic failure of this
  literature.
- ``assert_comparable``: a criterion CLASS (e.g. "the validation NLL") that
  denotes a different quantity per arm, pooled into one column as if it were
  one quantity.
- ``assert_distinct_seeds``: "replicates" that are not actually distinct,
  because the seed never reached the model.
- ``assert_post_fix`` (imported, not reimplemented -- see provenance.py) plus
  ``INVALIDATING_COMMITS``: results computed by code that was later found to
  be wrong, pooled with results computed after the fix.

``validate_frame`` runs everything above that is fully determined by the
frame itself, so the reporting path in later tasks has one call to make
before it prints anything -- a guard that is never invoked where the number
is produced is decoration.
"""
from __future__ import annotations

import pandas as pd

from adaptive_roa.benchmark.aggregate import PROVENANCE_COLUMNS
from adaptive_roa.benchmark.provenance import assert_post_fix

__all__ = [
    "INVALIDATING_COMMITS",
    "SEEDING_FIX_COMMIT",
    "NON_COMPARABLE_COLUMNS",
    "assert_post_fix",
    "assert_matched_budget",
    "assert_comparable",
    "assert_distinct_seeds",
    "validate_frame",
]

# ---------------------------------------------------------------------------
# Which commit each arm's numbers first became trustworthy at.
# ---------------------------------------------------------------------------
#
# Keyed by arm name (the actual df["arm"] value), not by the family label
# used in the design doc's table: assert_post_fix needs a concrete row subset
# plus one fix_sha, and the only handle this frame gives on "family" is the
# arm column, so the family -> arms resolution has to happen once, here,
# rather than drifting between copies in every reporting call site.
#
# Scope verified against the actual diffs (`git show --stat <sha>`), not
# assumed from the family label alone:
#
#   597db9f  BayesianMLPTrainer selected on the ELBO (not val_nll), returned
#            last-epoch (not best-checkpoint) weights, and cold-started
#            warm-start ensembles silently. Touches ONLY
#            bayesian_mlp_trainer.py plus the bnn_{mfvi,ensemble,laplace}
#            configs -- the OUTCOME arms that route through it. mlp / gp /
#            gp_optdelta never touch BayesianMLPTrainer and are unaffected.
#   18ea67e  FinalStateHead.nll/sample/mean learned/emitted RAW-coordinate
#            state instead of normalized, so a wide-range-unit dimension
#            dominated the loss. final_state_trainer.py builds exactly ONE
#            FinalStateHead and uses it for every posterior kind it drives
#            (point estimate, mfvi, ensemble, laplace) -- i.e. mlp_det and
#            the three bnn_*_reg arms. `fm` has its own FlowMatchingTrainer
#            and never constructs a FinalStateHead, so it is not listed here.
#            `gp_reg` documents in gp_final_state_handle.py that it uses
#            FinalStateHead ONLY for naming/distance, never for its own
#            nll/sample/mean, so it is not gated on this commit directly --
#            see e642259, which postdates it (checked below).
#   e642259  gp_reg's own trainer/predictor fix: predictive draws collapsed
#            once a query batch's joint dimension exceeded gpytorch's
#            Cholesky-to-Lanczos threshold, and the arm had no
#            validation-based selection at all. Chronologically after
#            18ea67e on this branch, so a run that clears e642259 also
#            clears 18ea67e -- no separate gp_reg entry for 18ea67e is
#            needed.
INVALIDATING_COMMITS: dict[str, str] = {
    "bnn_mfvi": "597db9f",
    "bnn_ensemble": "597db9f",
    "bnn_laplace": "597db9f",
    "mlp_det": "18ea67e",
    "bnn_mfvi_reg": "18ea67e",
    "bnn_ensemble_reg": "18ea67e",
    "bnn_laplace_reg": "18ea67e",
    "gp_reg": "e642259",
}

# Three trainers (BayesianMLPTrainer, FinalStateTrainer) read a seed key no
# shipped config ever sets and silently fell back to a constant, so seeds
# 42/43/44 built bit-identical sub-models. Not folded into
# INVALIDATING_COMMITS above: it does not gate one arm family, it gates ANY
# claim that compares more than one seed, for ANY arm. assert_distinct_seeds
# is the primary defense here -- it checks the symptom directly, so it also
# catches a future, unrelated bug with the same effect -- and this constant
# lets a caller (including validate_frame) additionally check provenance.
SEEDING_FIX_COMMIT = "a012c7e"

# Quantities that are well-defined WITHIN an arm but mean a different thing
# ACROSS arms. Putting these in one column and comparing is a category error,
# not a measurement one: model selection within an arm is unaffected.
NON_COMPARABLE_COLUMNS = frozenset({"best_val_nll", "val_nll", "train_loss"})


def assert_matched_budget(df: pd.DataFrame) -> None:
    """Every arm in a comparison must have trained to the same budget.

    ``n_epochs`` is the CONFIGURED budget (see aggregate.py's module
    docstring), constant across every row a single run contributes. A
    `nunique() > 1` within one arm means two runs of that arm were
    configured with different budgets; a differing max ACROSS arms means the
    comparison itself is apples-to-oranges. Weak baselines trained for fewer
    epochs than the arm they are compared against are the characteristic
    failure of this literature, so both are refused rather than averaged
    over silently.
    """
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
    """Refuse to compare a column across arms when it means different things per arm."""
    if column in NON_COMPARABLE_COLUMNS and df["arm"].nunique() > 1:
        raise ValueError(
            f"{column!r} is not comparable across arms: the same criterion "
            f"class denotes a different quantity per arm (beta-NLL in "
            f"manifold-normalized coordinates summed over dims for the BNN "
            f"arms; a plain Gaussian NLL in embedded space meaned over tasks "
            f"for gp_reg). Model selection within an arm is unaffected."
        )


def _is_structurally_tied(value) -> bool:
    """True only for the two value classes that legitimately repeat by construction.

    ``assert_distinct_seeds`` exists to catch a seed that never reached the
    model -- identical output where genuinely different randomness should
    have produced different output. But a naive "all values equal" check
    also fires on ties that have nothing to do with that bug:

    - Boundary values (0.0 or 1.0): the saturation points of every
      NORMALIZED metric this codebase reports (accuracy, precision, recall,
      F1, AUC, and a bool like ``run_complete`` read back as 0/1). A model
      that gets every calibration point right, or every one wrong, ties
      there regardless of the seed -- e.g. a small eval set where three
      seeds all reach accuracy 1.0.
    - Integer-valued floats (``value == round(value)``): counts and other
      discrete-by-construction quantities (a raw hit count, an epoch index)
      tie whenever two runs land on the same integer, which genuinely
      different seeds do all the time for small counts -- this is also how
      a "degenerate early epoch" (e.g. epoch 0, before training has had a
      chance to diverge the seeds) most often shows up in an integer-valued
      column.

    Both classes have an objective test independent of *why* the tie
    happened, which is what makes them safe to exempt unconditionally.
    Everything else -- a continuous, non-boundary float exactly equal across
    nominally different seeds -- is NOT exempted. Independently seeded
    stochastic training does not produce bit-identical floats at the
    1e-15 level by coincidence; the only mechanism this codebase has ever
    produced that result through is the seed silently not reaching the
    model. Exempting more than these two classes would be the "too
    permissive" failure mode that let three trainers ship fake replicates
    in the first place, so ambiguous cases are deliberately left to raise
    rather than guessed as benign.
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False
    if v in (0.0, 1.0):
        return True
    return v == round(v)


def assert_distinct_seeds(df: pd.DataFrame, column: str) -> None:
    """Refuse a variance claim over replicates that are not actually distinct.

    Three trainers ignored the run-level seed until ``SEEDING_FIX_COMMIT``:
    nominally different seeds (42/43/44) trained bit-identical sub-models, so
    any reported spread over them was fiction, not measurement. This checks
    the SYMPTOM (identical values across a genuinely multi-seed group) rather
    than only the commit, so it also catches a future, different bug with the
    same effect -- see ``_is_structurally_tied`` for the narrow, objective
    carve-out for ties that are not evidence of that bug.
    """
    for arm, grp in df.groupby("arm"):
        if grp["seed"].nunique() < 2:
            continue
        if grp[column].nunique() != 1:
            continue
        value = grp[column].iloc[0]
        if _is_structurally_tied(value):
            continue
        raise ValueError(
            f"arm {arm!r} reports identical {column!r} ({value!r}) across "
            f"{grp['seed'].nunique()} nominal seeds. Three trainers ignored "
            f"the run-level seed until the seeding fix ({SEEDING_FIX_COMMIT}), "
            f"making replicates bit-identical; a variance claim over them is "
            f"fiction. Confirm these runs postdate that fix, or that the seed "
            f"genuinely reached the model."
        )


def validate_frame(df: pd.DataFrame, *, repo_root=None) -> None:
    """Run every guard whose inputs are already determined by the frame itself.

    Later tasks call the reporting path against a whole aggregated frame, not
    one column at a time -- a guard that only exists as a function nobody
    calls from that path is decoration. This is the one call to make before
    printing anything from ``df``:

    - ``assert_matched_budget``: unconditional, frame-wide.
    - ``assert_comparable``: every column in ``NON_COMPARABLE_COLUMNS`` that
      is actually present in ``df``.
    - provenance: every arm this module knows an invalidating commit for
      (``INVALIDATING_COMMITS``), checked against that arm's rows only; plus,
      for any arm reporting more than one seed, the seeding fix
      (``SEEDING_FIX_COMMIT``) against that arm's rows.
    - ``assert_distinct_seeds``: every metric column -- anything not in
      ``aggregate.PROVENANCE_COLUMNS`` -- that is numeric.

    This is a FLOOR, not a substitute for the column-specific calls: a caller
    about to report a specific NON_COMPARABLE_COLUMNS metric or a specific
    seed-variance claim should still call ``assert_comparable`` /
    ``assert_distinct_seeds`` directly for that column, the same way it would
    without this function existing.
    """
    assert_matched_budget(df)

    for column in sorted(NON_COMPARABLE_COLUMNS & set(df.columns)):
        assert_comparable(df, column)

    if {"arm", "commit"}.issubset(df.columns):
        for arm, fix_sha in INVALIDATING_COMMITS.items():
            subset = df[df["arm"] == arm]
            if not subset.empty:
                assert_post_fix(subset, fix_sha, repo_root=repo_root)

        if "seed" in df.columns:
            for arm, grp in df.groupby("arm"):
                if grp["seed"].nunique() > 1:
                    assert_post_fix(grp, SEEDING_FIX_COMMIT, repo_root=repo_root)

    if {"arm", "seed"}.issubset(df.columns):
        metric_columns = [
            c for c in df.columns
            if c not in PROVENANCE_COLUMNS and pd.api.types.is_numeric_dtype(df[c])
        ]
        for column in metric_columns:
            assert_distinct_seeds(df, column)
