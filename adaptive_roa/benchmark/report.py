"""Tier-separated benchmark tables.

Calls the guards on the reporting path rather than merely importing them: a
guard that exists but is never invoked where the number is produced is
decoration (the design point ``guards.py``'s own module docstring makes).
``build_report`` calls ``assert_matched_budget``, ``assert_comparable``, and
``assert_distinct_seeds`` -- in that order -- before it emits a single line,
and refuses outright to place production-tier and reference-tier rows in one
table. That tier check is NOT delegated to ``assert_matched_budget``: that
guard is deliberately scoped WITHIN a tier (different regimes legitimately
train to different budgets -- see its own docstring), so two tiers pooled
into one table would sail straight through it. The reference tier carries
posterior-fidelity claims computed at a different training budget
([50, 50]) than the production tier's downstream task results
([256, 512, 256]) -- they are not rows of the same table regardless of
whether their budgets happen to collide.

Does NOT call ``validate_frame`` in place of the three guards above --
deliberately. ``validate_frame`` (guards.py) documents itself as "a FLOOR,
not a substitute for the column-specific calls": it ADDS provenance and
seeding-fix checks on top of a frame-wide ``assert_matched_budget``, it does
not replace the reporting-specific calls this module needs (``assert_comparable``
/ ``assert_distinct_seeds`` scoped to the one metric actually being
reported). More concretely, ``validate_frame`` also HARD-requires every
column in ``aggregate.PROVENANCE_COLUMNS`` (``run_id``, ``system``,
``epoch``, ``run_complete``, ``commit``, ...) to be present, with no
degradation -- unlike the three guards called here, none of which requires
anything beyond ``arm`` plus whichever context columns happen to exist. The
test fixtures in this module's test file (``arm``, ``tier``, ``acquisition``,
``seed``, ``n_epochs``, one metric column -- several columns short of
``PROVENANCE_COLUMNS``) are deliberately that minimal, the same way
``guards.py``'s own fixtures are (see that module's docstring); calling
``validate_frame`` here would break every one of them, AND would break the
"aggregate and report a still-launching slice" workflow this benchmark
actually uses (Task 8, step 4: ``df['tier'] = 'production'; build_report(df)``
run directly against a live campaign, before every run's ``commit``/``run_id``
provenance is necessarily uniform). A caller reporting on a COMPLETE,
fully-provenanced campaign frame is free to call ``validate_frame`` itself
before ``build_report`` -- nothing here prevents that, and it is a
reasonable belt-and-suspenders addition for a final leaderboard -- but
making it mandatory INSIDE ``build_report`` would turn a floor into a wall
for every partial or synthetic frame that does not need it.

Headline metric: real aggregated frames (``aggregate.collect_runs``)
recursively flatten nested ``eval_metrics`` payloads into DOTTED column
names -- ``lambda_delta.accuracy``, ``fixed_threshold.f1``,
``conservative_qhat.recall``, and so on (see ``aggregate._flatten_metrics``)
-- because ``FullROAEvaluator`` nests every number of interest one or two
levels below a decision-rule name. A bare ``"accuracy"`` column, this
module's own default and every fixture in its test module, exists ONLY in
hand-built fixtures -- verified against ``adaptive_v2/eval/full_roa.py``
directly and against ``tests/benchmark/test_aggregate.py``'s
real-artifact-driven tests. So ``build_report`` takes an explicit ``metric``
keyword (defaulting to ``METRIC = "accuracy"`` for literal compatibility
with the brief's own fixtures) instead of hardcoding one column name, and
REFUSES LOUDLY -- naming the requested column and listing the numeric
columns actually available -- when the requested column is absent, rather
than letting a mistyped or wrong-tier metric name reach ``pandas`` (a bare
``KeyError`` from inside ``pivot_table``, with no explanation of the
dotted-column convention) or, worse, silently producing an empty table.
None of the three guards called here check for the metric column's
PRESENCE: ``assert_comparable`` only judges whether a column that DOES
exist is safe to compare across arms, and never indexes ``df[column]`` at
all if it isn't in ``NON_COMPARABLE_COLUMNS``. So that presence check is
this module's own responsibility, not something Task 4 already covers.

Never pools across systems (C2). ``build_report`` refuses to mix TIERS
outright; systems are handled differently, and deliberately: a tier carries
a different CLAIM at a different budget, so two tiers are not rows of one
table at all, whereas two systems are the same claim measured on two
problems -- they belong in one document, just never in one row. So this
module emits ONE TABLE PER SYSTEM rather than requiring a caller to invoke
it once per system: a cross-system comparison IS the benchmark's
deliverable, and a shipped command that cannot produce it without four
separate invocations invites exactly the pooling this avoids. Before the
split, ``df.pivot_table(index="arm")`` averaged pendulum's 0.95 with
quadrotor3d's 0.55 into a single 0.802 per arm, and no guard fired --
``assert_matched_budget`` is deliberately scoped BY system, so a
multi-system frame sails through it. ``--system`` on the CLI additionally
narrows to one system when that is what a caller wants.

Sectioning alone is not sufficient, though, because the arms in different
sections can be different arms (C3): the routine state of a partially
launched ~450-run campaign is that one arm has finished on pendulum only
while its rival has finished on pendulum AND quadrotor3d, and their means
are then taken over different populations. ``assert_matched_coverage``
(guards.py) refuses that here, naming what is missing;
``--require-complete`` does not catch it, because every run present really
is complete and the missing ones simply contribute no rows.

Acquisition levels are DERIVED from the frame, never hardcoded (I1). An
earlier version named ``ranked`` and ``random`` literally, so a frame
carrying a third level had those rows aggregated, passed through every
guard, and then vanished from the output with no mention -- and once the
manifest was corrected to name the real control group (``direct``, not
``random``), the control column rendered as ``nan`` while the control data
it had actually collected was discarded. This module refuses loudly when a
metric COLUMN is absent (``_require_metric_column``); an acquisition level
gets the same standard, which here means rendering every level present
rather than raising, since a level nobody asked about is still data.

The epoch reduction is explicit, selectable, and stated in the output (I2).
``collect_runs`` emits one row per (run_id, EPOCH) and the pivot's
``aggfunc="mean"`` used to average epoch 0 -- before adaptive acquisition
has done anything, where ranked and random are near-identical by
construction -- with the final epoch. The dilution is systematic and biases
the benchmark's headline claim toward "adaptive doesn't help" (measured on
synthetic runs: +0.100 at the final epoch became +0.090 pooled over 0..9),
and the header said only "Downstream task (accuracy)". The default is now
the final epoch of each run, and whichever selection is in force is printed
above the table.

Mixed fidelity: a fidelity dict routinely mixes ``available=True`` arms with
``available=False`` ones withheld for non-convergence -- this is the
EXPECTED shape of every real call against the ``hmc_reg`` reference (see
``fidelity.py``'s module docstring: that reference does not currently
converge). A withheld result is rendered as the literal word "withheld"
plus its ``reason`` verbatim, in the same cells that would otherwise hold
the numbers -- never as a blank or omitted cell. A blank cell reads as "not
measured yet"; this is a deliberate refusal to report a number that would
be meaningless, and the two must not be visually confusable in the one
place (a markdown table, scanned quickly) that matters.
"""
from __future__ import annotations

import pandas as pd

from .aggregate import PROVENANCE_COLUMNS
from .fidelity import FidelityResult
from .guards import (
    assert_comparable,
    assert_distinct_seeds,
    assert_matched_budget,
    assert_matched_coverage,
    assert_one_run_per_cell,
)

__all__ = ["METRIC", "EPOCH_SELECTIONS", "build_report", "select_epochs"]

# Default headline metric. Matches the brief's own constant name/value so its
# hand-built fixtures (a bare "accuracy" column) work unmodified -- see the
# module docstring's "Headline metric" section for why this is NOT the right
# default against a real aggregated frame, and why callers there must pass
# `metric=` explicitly (e.g. "lambda_delta.accuracy").
METRIC = "accuracy"

# The adaptive arm of the paired comparison, and the acquisition levels that
# can serve as its control. `direct` is the group name (acquisition/direct.yaml,
# DirectAcquisitionStrategy -- uniform sampling, no ranking signal) and is what
# a real frame carries, because the engine writes `sampling_mode: direct` into
# the artifact. `random` is kept because it is the word the literature and
# every hand-built fixture use, and dropping it would silently stop computing
# the delta on frames that say "random".
ADAPTIVE_LEVEL = "ranked"
CONTROL_LEVELS = ("direct", "random")

# Named epoch reductions. Anything else must be an int (that epoch alone).
EPOCH_SELECTIONS = ("final", "all")


def _require_metric_column(df: pd.DataFrame, metric: str) -> None:
    """Refuse to let an absent metric column reach pandas as a bare KeyError.

    Real frames don't carry a bare ``"accuracy"`` column (see module
    docstring) -- when a caller's requested (or defaulted) metric isn't
    present, name it explicitly and list what numeric columns ARE present,
    so the fix is "pass metric='lambda_delta.accuracy'", not a stack trace
    to reverse-engineer.
    """
    if metric in df.columns:
        return
    available = sorted(
        c for c in df.columns
        if c not in PROVENANCE_COLUMNS and pd.api.types.is_numeric_dtype(df[c])
    )
    raise ValueError(
        f"headline metric {metric!r} is not a column in this frame. Real "
        f"aggregated frames (aggregate.collect_runs) flatten nested "
        f"eval_metrics into dotted names, e.g. 'lambda_delta.accuracy' or "
        f"'fixed_threshold.f1' -- a bare {metric!r} column exists only in "
        f"hand-built fixtures. Pass metric=<one of the numeric columns "
        f"actually present>: {available or '(none numeric)'}"
    )


def _require_report_columns(df: pd.DataFrame, columns) -> None:
    """Refuse a missing identity column with a diagnosis, not a bare KeyError.

    ``df["tier"]`` on a frame without one used to raise ``KeyError: 'tier'``
    from inside this module -- the one thing ``_require_metric_column``
    exists to stop happening for metrics.
    """
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"frame is missing required column(s) {missing}. build_report "
            f"needs them to scope the table it prints (tier heading, one "
            f"section per system, one column per acquisition level); it will "
            f"not guess what an absent one would have meant. Columns "
            f"present: {sorted(df.columns)}"
        )


def _select_epochs(df: pd.DataFrame, epochs) -> tuple[pd.DataFrame, str]:
    """Reduce over epochs explicitly, and say which epochs survived.

    ``collect_runs`` emits one row per (run_id, epoch). Averaging all of
    them -- what ``pivot_table(aggfunc="mean")`` does on its own -- pools
    epoch 0, before adaptive acquisition has selected a single point, with
    the final epoch, and reports the result under a header that names
    neither. The returned description goes into the table's own heading.
    """
    if "epoch" not in df.columns:
        return df, ("no `epoch` column in this frame, so no epoch reduction "
                    "was applied: every row contributes exactly once")

    if epochs == "all":
        present = sorted(df["epoch"].dropna().unique().tolist())
        return df, (f"ALL epochs {present} pooled by mean -- including "
                    f"epoch 0, before adaptive acquisition has done anything, "
                    f"which dilutes the ranked-vs-control delta toward zero")

    if epochs == "final":
        # Per RUN, not per frame: runs legitimately end at different epochs
        # (preemption, a shorter budget), and `df["epoch"].max()` would drop
        # every run that stopped earlier instead of taking its last epoch.
        key = ["run_id"] if "run_id" in df.columns else [
            c for c in ("arm", "system", "tier", "acquisition", "seed")
            if c in df.columns
        ]
        if not key:
            raise ValueError(
                "cannot select the final epoch: this frame has neither a "
                "`run_id` column nor any of arm/system/tier/acquisition/seed "
                "to identify a run by, so there is no way to tell which rows "
                "belong to the same run. Pass epochs='all' if pooling every "
                "row is genuinely what you want."
            )
        last = df.groupby(key, dropna=False)["epoch"].transform("max")
        selected = df[df["epoch"] == last]
        present = sorted(selected["epoch"].dropna().unique().tolist())
        return selected, (f"FINAL epoch of each run (grouped by {key}); "
                          f"epochs contributing: {present}")

    if isinstance(epochs, bool) or not isinstance(epochs, int):
        raise ValueError(
            f"epochs={epochs!r} is not a valid epoch selection; expected one "
            f"of {EPOCH_SELECTIONS} or an integer epoch number."
        )
    selected = df[df["epoch"] == epochs]
    if selected.empty:
        raise ValueError(
            f"epochs={epochs} selects no rows; epochs present in this frame: "
            f"{sorted(df['epoch'].dropna().unique().tolist())}"
        )
    return selected, f"epoch {epochs} only"


# Public alias: a caller building the per-point tables (pointwise.py) needs
# the SAME rows the headline table summarizes, so it applies this reduction
# itself before reading each run's artifacts off disk. Re-applying it inside
# build_report is idempotent for all three selections.
select_epochs = _select_epochs


def _resolve_control(levels: list, control: str | None) -> str | None:
    """Which acquisition level the paired delta is taken against, or None."""
    if control is not None:
        if control not in levels:
            raise ValueError(
                f"control={control!r} is not an acquisition level in this "
                f"frame. Levels present: {levels}. Refusing to report a "
                f"delta against a control that was never collected."
            )
        return control
    candidates = [c for c in CONTROL_LEVELS if c in levels]
    return candidates[0] if len(candidates) == 1 else None


def _shortfall(df: pd.DataFrame, have: str, want: str) -> list[tuple]:
    """Runs where `have` is short of `want`, as (run_id, have, want)."""
    if not {have, want}.issubset(df.columns):
        return []
    known = df.dropna(subset=[have, want])
    short = known[known[have] < known[want]]
    if short.empty:
        return []
    labels = short["run_id"] if "run_id" in short.columns else short.index
    return sorted({(str(r), int(h), int(w))
                   for r, h, w in zip(labels, short[have], short[want])})


def _coverage_lines(df: pd.DataFrame) -> list[str]:
    """Say when a run lost epochs, instead of quietly averaging over fewer.

    TWO ways to lose one, and they need separate counters (aggregate.py):
    an artifact that never parsed leaves no row at all
    (``n_epochs_collected``), while an epoch on which eval did not run
    leaves a perfectly valid row of NaN metrics that ``pivot_table``
    silently skips (``n_epochs_evaluated``). Reporting only the first is how
    this section came to state "every run contributed its full configured
    number of epochs" for a run that had lost a third of its metric
    population.
    """
    if not {"n_epochs", "n_epochs_collected"}.issubset(df.columns):
        return []
    missing_rows = _shortfall(df, "n_epochs_collected", "n_epochs")
    missing_evals = _shortfall(df, "n_epochs_evaluated", "n_epochs_collected")
    if not missing_rows and not missing_evals:
        return ["Every run contributed its full configured number of epochs, "
                "and every epoch it contributed carried metrics.", ""]

    lines = []
    if missing_rows:
        lines += [
            f"**{len(missing_rows)} run(s) contributed fewer epochs than "
            f"configured** (an artifact that was missing or would not parse "
            f"is skipped at aggregation, so the mean below is over a smaller "
            f"population for these runs): "
            + ", ".join(f"`{r}` {h}/{w}" for r, h, w in missing_rows), "",
        ]
    if missing_evals:
        lines += [
            f"**{len(missing_evals)} run(s) have epochs that produced NO "
            f"metrics** (eval did not run for those epochs; the rows exist "
            f"but every metric on them is NaN and is skipped by the "
            f"aggregation below): "
            + ", ".join(f"`{r}` {h} of {w} epochs evaluated"
                        for r, h, w in missing_evals), "",
        ]
    return lines


def _fmt(value) -> str:
    return "n/a" if pd.isna(value) else f"{value:.3f}"


def _table_lines(section: pd.DataFrame, metric: str, levels: list,
                 control: str | None, heading: str) -> list[str]:
    pivot = section.pivot_table(index="arm", columns="acquisition",
                                values=metric, aggfunc="mean")
    header = ["arm", *[str(l) for l in levels]]
    if control is not None and ADAPTIVE_LEVEL in levels:
        header.append(f"delta ({ADAPTIVE_LEVEL} − {control})")
    lines = [heading, "",
             "| " + " | ".join(header) + " |",
             "|" + "---|" * len(header)]
    for arm, row in pivot.iterrows():
        cells = [str(arm)] + [_fmt(row.get(level, float("nan"))) for level in levels]
        if control is not None and ADAPTIVE_LEVEL in levels:
            delta = row.get(ADAPTIVE_LEVEL, float("nan")) - row.get(control, float("nan"))
            cells.append("n/a" if pd.isna(delta) else f"{delta:+.3f}")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def _conditioned_lines(conditioned: list) -> list[str]:
    """Render the near-boundary slice, refusals included verbatim.

    Aggregate accuracy is dominated by basin interiors where every arm is
    correct, so two arms can rank identically overall while behaving very
    differently exactly where the outcome flips -- which is the regime
    adaptive acquisition exists to resolve. Rows are built by
    ``pointwise.separatrix_table`` from each run's own
    ``full_roa_per_point.npz``.
    """
    lines = ["## Near-boundary conditioned accuracy", "",
             "Predictions here are thresholded at p(success) >= 0.5, NOT at "
             "the calibrated lambda/delta rule, so these numbers are a "
             "different quantity from the table above and are not a "
             "decomposition of it. A run whose band could not be computed "
             "shows the refusal verbatim rather than a blank cell.", ""]
    fields = [c for c in ("system", "arm", "acquisition") if c in conditioned[0]]
    header = [*fields, "k", "runs", "overall", "near boundary", "interior",
              "n near", "n interior"]
    lines += ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    refusals = []
    for row in conditioned:
        cells = [str(row[f]) for f in fields]
        cells += [str(row["k"]), str(row["n_runs"]), _fmt(row["overall"]),
                  _fmt(row["near_boundary"]), _fmt(row["interior"]),
                  str(row["n_near"]), str(row["n_interior"])]
        lines.append("| " + " | ".join(cells) + " |")
        if row.get("refused"):
            refusals.append(f"- `{'/'.join(str(row[f]) for f in fields)}`: "
                            f"{row['refused']}")
    lines.append("")
    if refusals:
        lines += ["Runs refused while computing the band above:", *refusals, ""]
    return lines


def _fidelity_lines(fidelity: dict) -> list[str]:
    lines = ["## Posterior fidelity vs the HMC reference", "",
             "| arm | agreement | total variation |", "|---|---|---|"]
    for arm, res in sorted(fidelity.items()):
        if not isinstance(res, FidelityResult):
            raise TypeError(
                f"fidelity[{arm!r}] is a {type(res).__name__}, not a "
                f"FidelityResult -- build_report only knows how to render "
                f"fidelity.fidelity_vs_reference's own result type."
            )
        if res.available:
            lines.append(f"| {arm} | {res.agreement:.3f} | "
                         f"{res.total_variation:.3f} |")
        else:
            # Withheld: the reason goes IN the table, verbatim, in the same
            # cells a number would occupy -- never a blank cell, which would
            # read as "not measured" rather than "refused on purpose".
            lines.append(f"| {arm} | withheld | {res.reason} |")
    lines.append("")
    return lines


def build_report(df: pd.DataFrame, fidelity: dict | None = None,
                 metric: str = METRIC, *, epochs="final",
                 control: str | None = None,
                 conditioned: list | None = None,
                 provenance_checked: bool = True) -> str:
    """Render the tier's tables: one per system, never pooled across them.

    Args:
        df: aggregated frame (``aggregate.collect_runs``, or a fixture
            carrying at least ``arm``, ``tier``, ``acquisition`` and the
            metric column).
        fidelity: optional ``{label: FidelityResult}``, rendered verbatim --
            a withheld result stays visibly withheld.
        metric: the headline column. Real frames use dotted names
            (``lambda_delta.accuracy``); see the module docstring.
        epochs: ``"final"`` (default -- each run at its own last epoch),
            ``"all"`` (pool the whole learning curve, epoch 0 included), or
            an int for one epoch. Whichever is in force is printed above
            every table.
        control: the acquisition level the paired delta is taken against.
            Auto-detected from ``CONTROL_LEVELS`` when exactly one of them
            is present; pass it explicitly to disambiguate, and this
            function raises if the named level is not in the frame.
        conditioned: optional rows from ``pointwise.separatrix_table`` --
            accuracy split by proximity to the empirical basin boundary,
            which is where arms actually differ.
        provenance_checked: False when the caller skipped
            ``guards.validate_frame`` (``run_benchmark.py report
            --no-validate``). The disclosure then goes into the RENDERED
            TEXT, not just the caller's stdout: the artifact is what gets
            saved, read weeks later and pasted into a thread, and a warning
            that lives only in a terminal scrollback is not attached to it.
    """
    _require_report_columns(df, ["arm", "tier", "acquisition"])
    if df["tier"].nunique() > 1:
        raise ValueError(
            f"refusing to place tiers {sorted(df['tier'].unique())} in one "
            f"table: the reference tier carries posterior-fidelity claims at "
            f"[50,50] and the production tier carries downstream task results "
            f"at [256,512,256]. They are not rows of the same table."
        )
    _require_metric_column(df, metric)

    assert_matched_budget(df)
    assert_matched_coverage(df)
    assert_one_run_per_cell(df)
    assert_comparable(df, metric)
    assert_distinct_seeds(df, metric)

    selected, epoch_note = _select_epochs(df, epochs)
    # Levels come from the WHOLE frame, not per section, so every table has
    # the same columns and a level that exists only on one system is still
    # visible on the others (as `n/a`) rather than silently absent there.
    levels = sorted(selected["acquisition"].dropna().unique().tolist(), key=str)
    if not levels:
        raise ValueError(
            "no acquisition level survives the epoch selection, so there is "
            "nothing to compare; `acquisition` is null on every selected row."
        )
    resolved_control = _resolve_control(levels, control)

    tier = df["tier"].iloc[0]
    lines = [f"# Benchmark report — {tier} tier", ""]
    if not provenance_checked:
        lines += [
            "> **PROVENANCE NOT CHECKED.** This report was generated with "
            "validation disabled (`--no-validate`), so `guards.validate_frame` "
            "never ran: no arm was checked against its invalidating commit "
            "(`INVALIDATING_COMMITS`), the multi-seed seeding fix was not "
            "verified, and no check for a stale duplicate run was made. The "
            "numbers below may pool pre- and post-fix vintages. Do not quote "
            "them without regenerating this report without that flag.", "",
        ]
    lines += [f"**Epoch selection:** {epoch_note}.", ""]
    lines += _coverage_lines(selected)

    if resolved_control is None or ADAPTIVE_LEVEL not in levels:
        # Not silent: the delta is the headline claim, so its ABSENCE is
        # stated as prominently as its value would have been.
        lines += [
            f"**No paired delta is reported.** It needs the "
            f"{ADAPTIVE_LEVEL!r} level plus exactly one control level from "
            f"{list(CONTROL_LEVELS)}; the levels present are {levels}. Every "
            f"level present is still tabulated below.", "",
        ]

    systems = ([None] if "system" not in df.columns
               else sorted(selected["system"].dropna().unique().tolist(), key=str))
    for system in systems:
        section = selected if system is None else selected[selected["system"] == system]
        heading = (f"## Downstream task ({metric})" if system is None
                   else f"## Downstream task ({metric}) — system: {system}")
        lines += _table_lines(section, metric, levels, resolved_control, heading)

    if "system" in df.columns and len(systems) > 1:
        lines += [
            f"Systems are tabulated separately and never averaged together: "
            f"pooling {systems} into one row per arm produces a number that "
            f"describes no system -- an easy system's score and a hard "
            f"system's score collapse into one value that is neither.", "",
        ]

    lines += [f"A negative delta means adaptive acquisition LOST to the "
              f"{resolved_control or 'control'} baseline for that arm — a "
              f"reportable finding, not a bug (cf. Foong et al., NeurIPS "
              f"2020).", ""]

    if conditioned:
        lines += _conditioned_lines(conditioned)

    if fidelity:
        lines += _fidelity_lines(fidelity)

    return "\n".join(lines)
