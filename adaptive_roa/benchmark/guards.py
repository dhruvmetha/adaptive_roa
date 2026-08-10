"""Guards that refuse to produce a number from incomparable inputs.

Every function here RAISES. None warns, filters silently, or substitutes a
default: the failure mode these exist to prevent is a confident wrong table,
and a warning in a log is not a defense against that -- nobody reads it, and
the number still ships. That includes silently SKIPPING a check because a
column, or a row's identity, happened to be missing: a missing ``arm`` or
``commit`` is not "nothing to check", it is a row this module cannot vouch
for, and pandas' own default behavior (drop a null groupby key, quietly
ignore a null value in ``nunique()``) is exactly the kind of silent pass this
module exists to override. See ``_require_columns`` / ``_require_no_nulls``.

Each guard below corresponds to a way this benchmark has already produced, or
come close to producing, a confident wrong number:

- ``assert_matched_budget``: an arm compared at a shorter training budget than
  its rivals, WITHIN the same system/acquisition/tier context -- weak
  baselines are the characteristic failure of this literature. Scoped by
  context because different systems legitimately train to different budgets.
- ``assert_matched_coverage``: arms ranked against each other after being
  evaluated on DIFFERENT sets of cells. The mirror of the above, one level
  up: "same budget" is worthless if one arm's mean is taken over pendulum
  only and its rival's over pendulum plus quadrotor3d. This is the routine
  state of a partially-launched ~450-run campaign, and ``run_complete``
  cannot see it -- every run present really is complete; the missing ones
  simply contribute no rows.
- ``assert_comparable``: a criterion CLASS (e.g. "the validation NLL") that
  denotes a different quantity per arm, pooled into one column as if it were
  one quantity.
- ``assert_distinct_seeds``: "replicates" that are not actually distinct,
  because the seed never reached the model. Scoped to one (arm, system,
  acquisition, tier, epoch) cell at a time, and triggered only when the
  ENTIRE metric vector ties within that cell -- see the function docstring
  for why a single tied column is not, by itself, evidence of the bug.
- ``assert_all_complete``: a run that never finished (or was preempted)
  compared as if it were a settled result.
- ``assert_post_fix`` (imported, not reimplemented -- see provenance.py) plus
  ``INVALIDATING_COMMITS``: results computed by code that was later found to
  be wrong, pooled with results computed after the fix.

``validate_frame`` runs everything above that is fully determined by the
frame itself, so the reporting path in later tasks has one call to make
before it prints anything -- a guard that is never invoked where the number
is produced is decoration.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from adaptive_roa.benchmark.aggregate import PROVENANCE_COLUMNS
from adaptive_roa.benchmark.manifest import NON_ADAPTIVE_ARMS, acquisition_modes_for
from adaptive_roa.benchmark.provenance import assert_post_fix

__all__ = [
    "INVALIDATING_COMMITS",
    "SEEDING_FIX_COMMIT",
    "NON_COMPARABLE_COLUMNS",
    "assert_post_fix",
    "assert_matched_budget",
    "assert_matched_coverage",
    "assert_one_run_per_cell",
    "assert_comparable",
    "assert_distinct_seeds",
    "assert_all_complete",
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
#
# NOTE -- this guard is currently DORMANT against a real aggregated frame.
# best_val_nll / val_nll / train_loss are Lightning `self.log()` MONITOR
# names (see `_MONITOR = "val_nll"` in bayesian_mlp_trainer.py and
# final_state_trainer.py): they live in the PL CSVLogger's own metrics.csv
# under each run's checkpoint directory, never in artifacts_v2.json's
# eval_metrics, so aggregate.collect_runs() has no path that ever produces a
# column with these names (verified: `grep -rn '"val_nll"' adaptive_roa/`
# outside this file matches only the two `_MONITOR` assignments, never a
# dict key written to an artifact). Kept anyway, deliberately, as a
# forward-looking guard: the underlying hazard it describes is real (a
# beta-NLL in normalized, summed coordinates for the BNN arms is not the
# same quantity as gp_reg's plain Gaussian NLL in embedded, meaned
# coordinates), and if a future aggregation step starts joining the
# Lightning logs onto this frame, the guard should already be here rather
# than have to be reconstructed and remembered. See
# `test_non_comparable_columns_do_not_occur_in_a_real_aggregated_frame` for
# the check that keeps this claim honest, and the synthetic-column tests for
# the mechanism itself.
NON_COMPARABLE_COLUMNS = frozenset({"best_val_nll", "val_nll", "train_loss"})

# Columns that further scope a within-arm comparison: two rows can share an
# arm while representing entirely different experiments -- different system,
# different acquisition strategy, different tier, a different point in
# training. Grouping a comparison by "arm" alone pools all of those into one
# cell, and pooling is exactly how a genuine defect (or a genuine effect) in
# one cell gets masked by ordinary variation in another. Every guard that
# scopes a comparison below degrades gracefully to whatever subset of these
# is actually present (falling back to "arm" alone only when none of the
# rest exist), so the brief's own minimal fixtures -- which carry only arm,
# seed, n_epochs, and one metric -- still work.
_CONTEXT_COLUMNS = ("system", "acquisition", "tier")
_SEED_CELL_COLUMNS = ("system", "acquisition", "tier", "epoch")

# The cells every arm in one comparison must cover identically. Deliberately
# NOT including `epoch`: a run legitimately contributes a different number of
# epoch rows than its rival (preemption, a corrupt interior epoch, an eval
# that did not run), and the reporting path reduces over epochs explicitly
# rather than pooling them -- see report.py's epoch selector and
# aggregate.py's `n_epochs_collected`. `tier` is excluded because
# build_report already refuses a mixed-tier frame outright.
_COVERAGE_COLUMNS = ("system", "acquisition", "seed")

# The cell a single run occupies. Exactly one run_id may sit in one of these:
# `run_id` embeds a hash of the spec, so a stale run directory from an
# earlier campaign under the same exp_root lands in the SAME cell under a
# DIFFERENT run_id and is silently averaged in.
_RUN_CELL_COLUMNS = ("arm", "system", "acquisition", "seed", "epoch")


def _require_columns(df: pd.DataFrame, columns) -> None:
    """Refuse to guess what an absent column would have meant.

    A bare ``KeyError`` from pandas when a guard indexes a column that
    doesn't exist is not a diagnosis, and silently treating "column absent"
    as "nothing to check" is the exact failure mode this module exists to
    prevent (F11: pin the behavior instead of leaving it to whichever of
    those two accidents pandas happens to produce).
    """
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"frame is missing required column(s) {missing}; refusing to "
            f"guess what an absent column would have meant."
        )


def _require_no_nulls(df: pd.DataFrame, columns) -> None:
    """Refuse to silently drop a row whose identity is unknown.

    pandas' groupby silently DROPS a row whose grouping key is null,
    and ``Series.nunique()`` silently ignores a null value by default -- so a
    row with, say, a null ``arm`` would pass every check that groups or
    filters on ``arm`` not because it is valid, but because no check ever
    looked at it. That already happened to `commit` (see
    ``provenance.assert_post_fix``'s identical stance on a missing commit);
    the same silent drop is refused here for the other identity-bearing
    columns a guard groups or filters on.
    """
    for col in columns:
        if col not in df.columns:
            continue
        n_missing = int(df[col].isna().sum())
        if n_missing:
            raise ValueError(
                f"{col!r} is null on {n_missing} row(s); refusing to group or "
                f"filter on an unknown identity -- a null grouping key is "
                f"silently DROPPED by pandas groupby, so these rows would "
                f"otherwise pass every check that groups on {col!r} by never "
                f"being examined, not by being valid. Fix the aggregation "
                f"step (see aggregate.py) or exclude these rows explicitly."
            )


def assert_matched_budget(df: pd.DataFrame) -> None:
    """Every arm in a comparison must have trained to the same budget.

    ``n_epochs`` is the CONFIGURED budget (see aggregate.py's module
    docstring), constant across every row a single run contributes. Scoped
    by whichever of system/acquisition/tier are present: different systems
    (and, in principle, different acquisition modes or tiers -- the
    reference tier in particular is a deliberately different regime, not a
    weaker one) legitimately train to different budgets, so pooling budget
    comparisons across them refuses a perfectly honest frame. What is never
    allowed is two ARMS disagreeing on budget within the SAME context, or
    one arm itself carrying more than one budget within a context -- weak
    baselines trained for fewer epochs than their rivals are the
    characteristic failure of this literature.
    """
    context_cols = [c for c in _CONTEXT_COLUMNS if c in df.columns]
    _require_columns(df, ["arm", "n_epochs"])
    _require_no_nulls(df, ["arm", "n_epochs", *context_cols])

    cell_cols = ["arm", *context_cols]
    within = df.groupby(cell_cols)["n_epochs"].nunique()
    if (within > 1).any():
        offenders = within[within > 1].index.tolist()
        raise ValueError(
            f"{cell_cols} combination(s) {offenders} contain runs at more "
            f"than one training budget; weak baselines are the "
            f"characteristic failure of this literature, so ranking across "
            f"budgets is refused."
        )

    if context_cols:
        per_cell = df.groupby(cell_cols)["n_epochs"].max().reset_index()
        per_context = per_cell.groupby(context_cols)["n_epochs"].nunique()
        mismatched = per_context[per_context > 1]
        if not mismatched.empty:
            detail = per_cell.set_index(cell_cols)["n_epochs"].to_dict()
            raise ValueError(
                f"training budget differs across arms within context "
                f"{context_cols}={mismatched.index.tolist()}: {detail}. "
                f"Equalize the budget or compare within a budget."
            )
    else:
        per_arm = df.groupby("arm")["n_epochs"].max()
        if per_arm.nunique() > 1:
            raise ValueError(
                f"training budget differs across arms: {per_arm.to_dict()}. "
                f"Equalize the budget or compare within a budget."
            )


def assert_matched_coverage(df: pd.DataFrame) -> None:
    """Every arm in a comparison must cover the same (system, acquisition, seed) cells.

    ``assert_matched_budget`` asks whether two arms trained for the same
    number of epochs. This asks the question one level up: whether the two
    numbers being ranked were computed over the same POPULATION at all. An
    arm that has only finished on the easy system, or is missing one seed,
    has its mean taken over a different set of experiments than its rival's
    -- and a mean over an easier subset wins for a reason that has nothing
    to do with the arm:

        mlp  ran on [pendulum, quadrotor3d] -> 0.792
        gp   ran on [pendulum]              -> 1.002   <- "wins" by not running

    This is the ROUTINE state of a partially-launched ~450-run campaign, and
    nothing else catches it: ``assert_matched_budget`` is deliberately scoped
    BY system so a multi-system frame sails through it; ``run_complete`` is
    True on every row present, because the problem is the rows that are
    absent; and ``validate_frame`` checks relations among the rows it has.

    "The same cells" means the cells an arm is EXPECTED to cover, not a
    blunt global cross-product. ``mlp_det`` has no ``ranked`` cell BY
    DESIGN -- its outcome probability collapses to {0, 1}, so there is no
    ranking signal to acquire on, and ``expand_manifest`` never emits one.
    A cross-product rule refused the shipped pilot at 100% completion, with
    no flag able to rescue it (``--system`` cannot help when the missing
    axis is ``acquisition``). The per-arm expectation is therefore derived
    from ``manifest.acquisition_modes_for`` -- the same function
    ``expand_manifest`` builds the campaign with -- so the guard and the
    expander cannot drift apart. The modes an ADAPTIVE arm is held to are
    the modes the frame's adaptive arms actually ran, so a control-only or
    single-mode campaign is judged against itself rather than against a
    mode only the fixed-dataset baseline contributes.

    The ``system`` and ``seed`` axes stay union-based: a campaign that is
    equally incomplete across every arm (quadrotor3d still at seed 42 while
    pendulum has 42/43/44) is comparable and must not be refused.

    Scoped to whichever of ``system``/``acquisition``/``seed`` are present,
    degrading the same way every other guard here does, so the minimal
    fixtures in this module and in report.py's tests still work. With none
    of them present there is no cell structure to compare and the check is
    vacuous -- that is a frame carrying no evidence of coverage either way,
    not a frame this function has silently approved.

    Deliberately NOT called from ``validate_frame``, for the same reason
    ``assert_all_complete`` is not: unequal coverage is the normal condition
    of a live campaign, and a status query over one must not fail. It is
    called from ``build_report``, which is where arms are actually RANKED
    against each other and where unequal coverage stops being normal and
    becomes a wrong answer.
    """
    _require_columns(df, ["arm"])
    cell_cols = [c for c in _COVERAGE_COLUMNS if c in df.columns]
    _require_no_nulls(df, ["arm", *cell_cols])
    if not cell_cols:
        return

    covered = {
        arm: {tuple(row) for row in grp[cell_cols].drop_duplicates().to_numpy().tolist()}
        for arm, grp in df.groupby("arm")
    }
    if len(covered) < 2:
        return

    union = set().union(*covered.values())

    if "acquisition" in cell_cols:
        acq = cell_cols.index("acquisition")
        # What the ADAPTIVE arms in this frame actually ran. Using the frame-
        # wide set instead would hold every adaptive arm to a mode that only
        # a fixed-dataset baseline contributes.
        adaptive_modes = sorted(
            {cell[acq] for arm, cells in covered.items()
             if arm not in NON_ADAPTIVE_ARMS for cell in cells},
            key=repr,
        )

        def _expected(arm):
            allowed = set(acquisition_modes_for(arm, adaptive_modes))
            return {cell for cell in union if cell[acq] in allowed}
    else:
        def _expected(arm):
            return union

    missing = {}
    for arm, cells in covered.items():
        gap = sorted(_expected(arm) - cells, key=repr)
        if gap:
            missing[arm] = gap
    if not missing:
        return

    detail = "; ".join(
        f"{arm!r} is missing {len(cells)} of {len(_expected(arm))} expected: "
        f"{cells}"
        for arm, cells in sorted(missing.items())
    )
    raise ValueError(
        f"arms in this comparison do not cover the same {cell_cols} cells, so "
        f"their means are taken over different populations and are not "
        f"comparable -- an arm that has not run on the hard system, or is "
        f"missing a seed, wins for a reason that is not about the arm. "
        f"{detail}. (An arm's expected acquisition modes come from "
        f"manifest.acquisition_modes_for, so a non-adaptive arm is not held "
        f"to a 'ranked' cell it never runs.) Restrict the report to a slice "
        f"every arm covers (e.g. --system), or wait for the missing runs."
    )


def assert_one_run_per_cell(df: pd.DataFrame) -> None:
    """Exactly one run_id per (arm, system, acquisition, seed, epoch) cell.

    ``run_id`` embeds a sha1 of the spec, so a run from an earlier campaign
    under the same ``exp_root`` -- different ``overrides``, therefore a
    different hash, therefore a different directory -- lands in the SAME
    cell as the current run under a DIFFERENT ``run_id``. ``collect_runs``
    reads both, ``pivot_table`` averages both, and nothing anywhere notices:
    every run is complete, every budget matches, every arm covers every
    cell, and provenance is fine because the stale run was also produced by
    post-fix code.

    This is the one uncaught failure that REVERSES a conclusion rather than
    merely degrading it. Demonstrated: a stale directory pooling into one
    cell moved an arm from 0.95 to a reported 0.75 and flipped its paired
    delta from positive to -0.050, i.e. turned "adaptive helps" into
    "adaptive hurts".

    Degrades on a frame with no ``run_id`` column: run identity is exactly
    what this checks, so a frame that carries none has nothing to
    deduplicate. That is not a hole on the shipped path --
    ``validate_frame`` hard-requires every ``PROVENANCE_COLUMNS`` entry
    (``run_id`` among them) and ``run_benchmark.py report`` calls it by
    default -- it is what keeps the minimal hand-built fixtures working.
    """
    if "run_id" not in df.columns:
        return
    _require_columns(df, ["arm"])
    cell_cols = [c for c in _RUN_CELL_COLUMNS if c in df.columns]
    _require_no_nulls(df, ["run_id", *cell_cols])

    counts = df.groupby(cell_cols, dropna=False)["run_id"].nunique()
    offenders = counts[counts > 1]
    if offenders.empty:
        return

    detail = []
    for key in offenders.index:
        key_tuple = key if isinstance(key, tuple) else (key,)
        mask = pd.Series(True, index=df.index)
        for column, value in zip(cell_cols, key_tuple):
            mask &= df[column] == value
        run_ids = sorted(df.loc[mask, "run_id"].unique().tolist())
        detail.append(f"{dict(zip(cell_cols, key_tuple))} <- {run_ids}")

    raise ValueError(
        f"{len(offenders)} cell(s) contain more than one run: "
        + "; ".join(detail)
        + ". run_id embeds a hash of the run spec, so a stale run directory "
          "from an earlier campaign under the same exp_root lands in the "
          "same cell under a different run_id and is averaged in silently -- "
          "which can REVERSE a paired delta, not merely blur it. Remove the "
          "stale directories, or point --exp-root at a clean campaign root."
    )


def assert_comparable(df: pd.DataFrame, column: str) -> None:
    """Refuse to compare a column across arms when it means different things per arm."""
    _require_columns(df, ["arm"])
    if column in NON_COMPARABLE_COLUMNS and df["arm"].nunique() > 1:
        raise ValueError(
            f"{column!r} is not comparable across arms: the same criterion "
            f"class denotes a different quantity per arm (beta-NLL in "
            f"manifold-normalized coordinates summed over dims for the BNN "
            f"arms; a plain Gaussian NLL in embedded space meaned over tasks "
            f"for gp_reg). Model selection within an arm is unaffected."
        )


def assert_distinct_seeds(df: pd.DataFrame, column: str) -> None:
    """Refuse a variance claim over replicates that are not actually distinct.

    Three trainers ignored the run-level seed until ``SEEDING_FIX_COMMIT``:
    nominally different seeds (42/43/44) trained bit-identical sub-models, so
    any reported spread over them was fiction, not measurement.

    Two things this checks, both load-bearing:

    1. SCOPE. Comparisons are made within one (arm, system, acquisition,
       tier, epoch) cell at a time -- whichever of those columns are present
       -- not pooled across all of an arm's rows. Two rows can share an arm
       while being entirely different experiments; pooling them lets a
       genuine tie in one cell (identical output at seeds 42/43/44 for
       pendulum, epoch 9) be masked by ordinary variation somewhere else
       (a different value at quad3d, or at epoch 3), so the flagship check
       would silently pass on exactly the frame it exists to catch.
    2. EVIDENCE. Within a cell, this raises only when the ENTIRE metric
       vector ties across the cell's seeds -- `column` AND every other
       metric column that has a value for every seed in the cell -- not
       when `column` alone ties. A single tied column is routine on its
       own: accuracy/precision/F1/AUC are rationals over a fixed, finite
       evaluation grid, so two independently-seeded models land on the same
       accuracy constantly even while disagreeing on WHICH points they got
       right (different tp/fp/fn) -- refusing every such coincidence would
       abort honest comparisons for no reason, and was verified to do so
       against real seed-varying runs during review. But if every metric in
       the cell ties, the predictions themselves must be identical at every
       evaluation point, and the only mechanism this codebase has ever
       produced THAT through is the seed silently not reaching the model.
       A metric with a missing value for any seed in the cell is excluded
       from this vote rather than treated as tied OR as varying -- a
       missing value is not evidence either way, and letting it count as a
       "tie" is how a partially-populated frame produces a false alarm.
    """
    _require_columns(df, ["arm", "seed", column])
    cell_cols = [c for c in ("arm", *_SEED_CELL_COLUMNS) if c in df.columns]
    _require_no_nulls(df, [*cell_cols, "seed"])

    metric_cols = [
        c for c in df.columns
        if c not in PROVENANCE_COLUMNS and pd.api.types.is_numeric_dtype(df[c])
    ]
    if column not in metric_cols:
        metric_cols = metric_cols + [column]

    for key, cell in df.groupby(cell_cols):
        if cell["seed"].nunique() < 2:
            continue
        observed = [c for c in metric_cols if not cell[c].isna().any() and len(cell[c]) >= 2]
        if column not in observed:
            continue
        if any(cell[c].nunique() != 1 for c in observed):
            continue
        identity = dict(zip(cell_cols, key if isinstance(key, tuple) else (key,)))
        raise ValueError(
            f"cell {identity} reports an identical value for every metric "
            f"column with data for every seed ({sorted(observed)}), across "
            f"{cell['seed'].nunique()} nominal seeds -- including {column!r} "
            f"= {cell[column].iloc[0]!r}. Three trainers ignored the "
            f"run-level seed until the seeding fix ({SEEDING_FIX_COMMIT}), "
            f"making replicates bit-identical; a variance claim over them is "
            f"fiction. Confirm these runs postdate that fix, or that the "
            f"seed genuinely reached the model."
        )


def assert_all_complete(df: pd.DataFrame) -> None:
    """Refuse to treat a still-running or prematurely-terminated run as final.

    ``run_complete`` is True only when the engine's own end-of-run marker
    (``final_results.json``) exists and parses (aggregate.py). During an
    IN-FLIGHT campaign, ``run_complete=False`` is the ROUTINE state -- most
    rows, most of the time, per aggregate.py's own module docstring -- so
    ``validate_frame`` does NOT call this: doing so would make every query
    against a live ~450-run campaign fail outright, which is not what this
    guard is for.

    This is for the one place that DOES need every row settled: a final
    report or leaderboard, generated once a campaign is actually done,
    should call this in ADDITION to ``validate_frame`` so a run that was
    preempted mid-training -- which, by row count alone, looks identical to
    one that simply has not gotten there yet (see aggregate.py) -- cannot be
    silently reported as a finished result next to runs that really did
    finish.

    ``run_complete`` must be genuinely BOOLEAN, and that is enforced rather
    than assumed. ``~df["run_complete"].astype(bool)`` -- what this used to
    do -- passes silently on the STRING ``'False'`` (every non-empty string
    is truthy, so ``['False', 'True'] -> [True, True]``) and on ``NaN``
    (also truthy). This is the third appearance of that exact failure mode
    in this project: ``.astype(bool)`` on ``{-1, 1}`` labels already
    collapsed both classes into "success" once
    (``separatrix._labels_as_success_bool``), and once in
    ``full_roa``'s own history. ``collect_runs`` produces real bools today,
    so the direct CLI path was safe by accident of its inputs -- but a
    CSV/parquet round-trip, which is routine for a benchmark frame, turns
    every bool into a string, and a concat over a missing column introduces
    NaN. Either one silently disarms the one guard designated as the
    final-report gate. Object dtype holding genuine ``bool``/``np.bool_``
    values is accepted (a plain ``pd.concat`` produces it); anything else --
    strings, ints, None -- raises, naming the offending values.
    """
    _require_columns(df, ["run_complete"])
    _require_no_nulls(df, ["run_complete"])

    complete = df["run_complete"]
    if not pd.api.types.is_bool_dtype(complete):
        offenders = sorted(
            {f"{v!r} ({type(v).__name__})" for v in complete
             if not isinstance(v, (bool, np.bool_))}
        )
        if offenders:
            raise ValueError(
                f"'run_complete' is not boolean: {complete.dtype} dtype "
                f"carrying {offenders}. Refusing to coerce -- .astype(bool) "
                f"maps the string 'False' and NaN to True, which would pass "
                f"this guard on exactly the frame it exists to refuse (a "
                f"CSV/parquet round-trip stringifies every bool). Restore "
                f"the column to real booleans, e.g. "
                f"df['run_complete'].map({{'True': True, 'False': False}})."
            )

    incomplete = df.loc[~complete.astype(bool)]
    if not incomplete.empty:
        detail = (
            sorted(incomplete["run_id"].unique().tolist())
            if "run_id" in incomplete.columns
            else len(incomplete)
        )
        raise ValueError(
            f"{len(incomplete)} row(s) come from a run that has not written "
            f"final_results.json (run_complete=False) -- still running, or "
            f"preempted and never resumed. Not safe to report as a finished "
            f"result next to runs that did finish. Affected: {detail}."
        )


def validate_frame(df: pd.DataFrame, *, repo_root=None) -> None:
    """Run every guard whose inputs are already determined by the frame itself.

    Later tasks call the reporting path against a whole aggregated frame, not
    one column at a time -- a guard that only exists as a function nobody
    calls from that path is decoration. This is the one call to make before
    printing anything from ``df``:

    - requires every column ``aggregate.collect_runs()`` always produces
      (``aggregate.PROVENANCE_COLUMNS``) to actually be present -- a frame
      missing one is refused outright rather than having the checks below
      silently skip whatever they can't see (F4/F11: a silent skip here is
      exactly the failure mode this module exists to prevent).
    - ``assert_one_run_per_cell``: a stale run directory pooled into a cell
      that already has a run. Included here (unlike ``assert_all_complete``
      and ``assert_matched_coverage``) because it is never a legitimate
      state of a live campaign -- it is always an error, at every stage.
    - ``assert_matched_budget``: frame-wide, scoped by system/acquisition/tier.
      Called FIRST, and its own null-identity check (on ``arm``) is what
      catches a null ``arm`` before the provenance loop below ever runs --
      an EARLIER draft also had a separate, explicit null-arm check directly
      in this function, before this call; mutation-testing it (leaving
      ``assert_matched_budget``'s own check intact) proved that copy dead --
      disabling it broke no test -- so it was removed rather than kept as
      decoration (see task-4-report.md). ``assert_distinct_seeds`` at the
      bottom of this function ALSO independently checks ``arm`` for nulls
      (it needs to, standalone), so in practice this specific failure mode
      is caught twice over when reached through ``validate_frame`` -- again
      verified by mutation, disabling both at once (and only both at once)
      is what finally lets a null-arm row pass silently.
    - ``assert_comparable``: every column in ``NON_COMPARABLE_COLUMNS`` that
      is actually present in ``df`` (currently dormant against a real frame
      -- see that constant's docstring).
    - provenance: every arm this module knows an invalidating commit for
      (``INVALIDATING_COMMITS``), checked against that arm's rows only; plus,
      for any arm reporting more than one seed anywhere in the frame, the
      seeding fix (``SEEDING_FIX_COMMIT``) against that arm's rows. A row
      with a null ``arm`` would otherwise never match any of these per-arm
      filters and so never get its commit checked -- already ruled out by
      ``assert_matched_budget`` above.
    - ``assert_distinct_seeds``: every metric column -- anything not in
      ``aggregate.PROVENANCE_COLUMNS`` -- that is numeric.

    Deliberately NOT included: ``assert_all_complete``. See that function's
    docstring for why running it here would break routine use against a
    live campaign, and call it explicitly from a final-report path instead.

    This is a FLOOR, not a substitute for the column-specific calls: a caller
    about to report a specific NON_COMPARABLE_COLUMNS metric or a specific
    seed-variance claim should still call ``assert_comparable`` /
    ``assert_distinct_seeds`` directly for that column, the same way it would
    without this function existing.
    """
    _require_columns(df, PROVENANCE_COLUMNS)
    assert_matched_budget(df)
    # Fully determined by the frame itself, and never a legitimate state of a
    # live campaign the way an incomplete run or unequal coverage is: two
    # run_ids in one cell is always a stale directory, whatever stage the
    # campaign is at. So unlike assert_all_complete / assert_matched_coverage,
    # this one belongs in the floor.
    assert_one_run_per_cell(df)

    for column in sorted(NON_COMPARABLE_COLUMNS & set(df.columns)):
        assert_comparable(df, column)

    for arm, fix_sha in INVALIDATING_COMMITS.items():
        subset = df[df["arm"] == arm]
        if not subset.empty:
            assert_post_fix(subset, fix_sha, repo_root=repo_root)

    for arm, grp in df.groupby("arm"):
        if grp["seed"].nunique() > 1:
            assert_post_fix(grp, SEEDING_FIX_COMMIT, repo_root=repo_root)

    metric_columns = [
        c for c in df.columns
        if c not in PROVENANCE_COLUMNS and pd.api.types.is_numeric_dtype(df[c])
    ]
    for column in metric_columns:
        assert_distinct_seeds(df, column)
