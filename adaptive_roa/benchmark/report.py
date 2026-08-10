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
from .guards import assert_comparable, assert_distinct_seeds, assert_matched_budget

__all__ = ["METRIC", "build_report"]

# Default headline metric. Matches the brief's own constant name/value so its
# hand-built fixtures (a bare "accuracy" column) work unmodified -- see the
# module docstring's "Headline metric" section for why this is NOT the right
# default against a real aggregated frame, and why callers there must pass
# `metric=` explicitly (e.g. "lambda_delta.accuracy").
METRIC = "accuracy"


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
                 metric: str = METRIC) -> str:
    if df["tier"].nunique() > 1:
        raise ValueError(
            f"refusing to place tiers {sorted(df['tier'].unique())} in one "
            f"table: the reference tier carries posterior-fidelity claims at "
            f"[50,50] and the production tier carries downstream task results "
            f"at [256,512,256]. They are not rows of the same table."
        )
    _require_metric_column(df, metric)

    assert_matched_budget(df)
    assert_comparable(df, metric)
    assert_distinct_seeds(df, metric)

    tier = df["tier"].iloc[0]
    lines = [f"# Benchmark report — {tier} tier", ""]

    pivot = df.pivot_table(index="arm", columns="acquisition",
                           values=metric, aggfunc="mean")
    lines += [f"## Downstream task ({metric})", "",
              "| arm | ranked | random | delta (ranked − random) |",
              "|---|---|---|---|"]
    for arm, row in pivot.iterrows():
        ranked = row.get("ranked", float("nan"))
        random_ = row.get("random", float("nan"))
        delta = ranked - random_
        lines.append(f"| {arm} | {ranked:.3f} | {random_:.3f} | {delta:+.3f} |")

    lines += ["", "A negative delta means adaptive acquisition LOST to random "
                  "selection for that arm — a reportable finding, not a bug "
                  "(cf. Foong et al., NeurIPS 2020).", ""]

    if fidelity:
        lines += _fidelity_lines(fidelity)

    return "\n".join(lines)
