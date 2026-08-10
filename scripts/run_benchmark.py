"""Campaign CLI: expand a manifest, report status, launch what is missing,
and render a tier-separated report from what has already been collected.

Two subcommands:

- ``launch`` (the default, for backward compatibility with every documented
  invocation of this script that predates the ``report`` subcommand --
  e.g. ``run_benchmark.py configs/benchmark/pilot.yaml --exp-root ...`` --
  see the argv-sniffing in ``main()``): expand a manifest, print what's
  complete/partial/absent, and launch what's missing.
- ``report``: aggregate whatever ``collect_runs`` finds under ``--exp-root``
  and render it with ``adaptive_roa.benchmark.report.build_report`` --
  which itself refuses (loudly) to report on a frame that mixes tiers, an
  unmatched budget, unequal arm coverage, an incomparable metric, or seeds
  that never actually diverged. A guard raising here is the harness working,
  not a bug in this script -- see ``report.py``'s module docstring.

``report`` calls ``guards.validate_frame`` on the collected frame BEFORE
handing it to ``build_report``, and that is where the provenance checks
live. ``build_report`` deliberately does not call ``validate_frame``
itself: that function hard-requires every ``PROVENANCE_COLUMNS`` entry, and
making it mandatory inside ``build_report`` would turn a floor into a wall
for every partial or hand-built frame (see report.py's module docstring for
the full argument, which stands). But the consequence, left unaddressed,
was that the ONLY reporting path this repo ships never checked provenance
at all -- a frame pooling pre- and post-fix vintages, the exact failure
mode ``INVALIDATING_COMMITS`` and Task 0 exist for, reached a printed table
unchecked. This function always has a fully-provenanced ``collect_runs``
frame, so it is the right place for the call. ``--no-validate`` exists as
an explicit escape for a frame whose provenance is genuinely not resolvable
(e.g. runs copied off another filesystem without their git history), and
says so in the output rather than being silent about it.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

from adaptive_roa.benchmark.aggregate import collect_runs
from adaptive_roa.benchmark.guards import assert_all_complete, validate_frame
from adaptive_roa.benchmark.launcher import plan_launch, sbatch_command
from adaptive_roa.benchmark.manifest import CONFIG_ROOT, expand_manifest
from adaptive_roa.benchmark.pointwise import fidelity_table, separatrix_table
from adaptive_roa.benchmark.report import (
    EPOCH_SELECTIONS,
    METRIC,
    build_report,
    select_epochs,
)

_SUBCOMMANDS = ("launch", "report")


def _parse_epochs(value: str):
    """"final" / "all" / an integer epoch. Anything else is refused here."""
    if value in EPOCH_SELECTIONS:
        return value
    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"{value!r} is not an epoch selection; expected one of "
            f"{EPOCH_SELECTIONS} or an integer epoch number"
        ) from None


def _assert_specs_compose(specs) -> None:
    """Compose every distinct override shape before anything is submitted.

    ``expand_manifest``'s own check covers a config GROUP whose value has no
    file (``system=cartpole``). It cannot cover a typo in the override's
    KEY: a manifest ``overrides:`` entry like ``acquisiton=ranked`` names no
    config group, so it is (correctly) treated as setting a config value,
    and only Hydra's struct-mode check can say whether that key exists. That
    check used to happen inside SLURM, once per job.

    So it happens here instead, on the launch path, where it costs one
    ``compose()`` per distinct override shape (42 for the pilot's 126 runs,
    a couple of seconds) and where the alternative is ~450 queued jobs
    dying one at a time. Imported lazily so the ``report`` subcommand never
    pays for Hydra.
    """
    from hydra import compose, initialize_config_dir

    seen = set()
    for spec in specs:
        overrides = spec.hydra_overrides()
        # seed / n_epochs set values, never groups: they cannot change
        # whether the set composes, so shapes that differ only in those are
        # the same composition check.
        shape = tuple(o for o in overrides
                      if not o.startswith(("seed=", "n_epochs=")))
        if shape in seen:
            continue
        seen.add(shape)
        try:
            with initialize_config_dir(config_dir=str(CONFIG_ROOT),
                                       version_base=None):
                compose(config_name="default", overrides=list(overrides))
        except Exception as exc:                       # noqa: BLE001
            raise SystemExit(
                f"run {spec.run_id!r} does not compose and would die inside "
                f"SLURM after being queued.\n  overrides: {overrides}\n"
                f"  {type(exc).__name__}: {str(exc).splitlines()[0]}"
            ) from None


def _run_launch(args) -> None:
    specs = expand_manifest(yaml.safe_load(args.manifest.read_text()))
    _assert_specs_compose(specs)
    to_launch, states = plan_launch(specs, args.exp_root)

    tally = {s: sum(1 for v in states.values() if v == s)
             for s in ("complete", "partial", "absent")}
    print(f"{len(specs)} runs: {tally['complete']} complete, "
          f"{tally['partial']} partial, {tally['absent']} absent")

    for spec in to_launch:
        cmd = sbatch_command(spec, args.exp_root)
        if args.launch:
            subprocess.run(cmd, check=True)
        else:
            print(" ".join(cmd))
    if not args.launch:
        print(f"\ndry run -- {len(to_launch)} would be submitted. Pass --launch.")


def _run_report(args) -> None:
    df = collect_runs(args.exp_root)
    if df.empty:
        print(f"no runs found under {args.exp_root}")
        return
    if args.tier is not None:
        df = df[df["tier"] == args.tier]
        if df.empty:
            print(f"no {args.tier!r}-tier runs found under {args.exp_root}")
            return
    if args.system is not None:
        available = sorted(df["system"].dropna().unique().tolist())
        df = df[df["system"] == args.system]
        if df.empty:
            # Refused, not "no runs found": a typo'd system name and an
            # empty campaign are different problems with different fixes.
            raise SystemExit(
                f"no runs for system={args.system!r} under {args.exp_root}; "
                f"systems present: {available}"
            )
    if args.require_complete:
        # Opt-in, not the default: run_complete=False is the ROUTINE state
        # for most rows of a still-launching campaign (aggregate.py's own
        # module docstring), so build_report itself does not call this --
        # only ask for it once a campaign is meant to be actually done.
        assert_all_complete(df)
    if args.validate:
        # The provenance gate. See this module's docstring for why it lives
        # here rather than inside build_report.
        validate_frame(df)
    else:
        print("WARNING: --no-validate -- provenance (INVALIDATING_COMMITS, "
              "the seeding fix) was NOT checked for this report.")

    # The near-boundary slice and the posterior-fidelity table are built
    # from each run's own per-point artifacts, at the SAME epochs the
    # headline table summarizes -- hence the explicit reduction here rather
    # than letting build_report do it and then guessing which rows it kept.
    conditioned = fidelity = None
    if args.separatrix or args.fidelity_vs:
        selected, _ = select_epochs(df, args.epochs)
        if args.separatrix:
            conditioned = separatrix_table(selected, args.exp_root,
                                           k=args.separatrix_k)
        if args.fidelity_vs:
            fidelity = fidelity_table(selected, args.exp_root,
                                      reference_arm=args.fidelity_vs,
                                      rhat_threshold=args.rhat_threshold)

    text = build_report(df, fidelity=fidelity, metric=args.metric,
                        epochs=args.epochs, control=args.control,
                        conditioned=conditioned,
                        provenance_checked=args.validate)
    if args.out is not None:
        args.out.write_text(text)
        print(f"wrote {args.out}")
    else:
        print(text)


def main():
    argv = sys.argv[1:]
    if not argv or argv[0] not in (*_SUBCOMMANDS, "-h", "--help"):
        argv = ["launch", *argv]

    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)

    launch_ap = sub.add_parser(
        "launch", help="expand a manifest, report status, launch what is "
                       "missing (default)")
    launch_ap.add_argument("manifest", type=Path)
    launch_ap.add_argument("--exp-root", type=Path, required=True)
    launch_ap.add_argument("--launch", action="store_true",
                           help="actually submit; default is a dry run")

    report_ap = sub.add_parser(
        "report", help="aggregate collected runs and render a "
                       "tier-separated markdown report")
    report_ap.add_argument("--exp-root", type=Path, required=True)
    report_ap.add_argument(
        "--metric", default=METRIC,
        help=f"headline column to report (default: {METRIC!r}; real "
             f"aggregated frames use dotted names like "
             f"'lambda_delta.accuracy' -- see report.py)")
    report_ap.add_argument(
        "--tier", choices=["production", "reference"], default=None,
        help="restrict to one tier before reporting; build_report refuses "
             "outright if the collected frame mixes tiers and this is "
             "omitted")
    report_ap.add_argument(
        "--system", default=None,
        help="restrict to one system before reporting; without it, every "
             "system gets its own table and none are averaged together")
    report_ap.add_argument(
        "--epochs", type=_parse_epochs, default="final",
        help="which epochs the reported number covers: 'final' (default, "
             "each run at its own last epoch), 'all' (pool the whole "
             "learning curve, epoch 0 included), or an integer epoch. The "
             "choice is printed above every table")
    report_ap.add_argument(
        "--control", default=None,
        help="acquisition level the paired delta is taken against "
             "(default: auto-detected when exactly one of direct/random is "
             "present)")
    report_ap.add_argument(
        "--require-complete", action="store_true",
        help="also refuse rows from runs that have not finished "
             "(assert_all_complete) -- for a final report, not a "
             "still-launching campaign")
    report_ap.add_argument(
        "--separatrix", action="store_true",
        help="also report accuracy split by proximity to the empirical "
             "basin boundary, computed from each run's own "
             "full_roa_per_point.npz (aggregate accuracy is dominated by "
             "interiors, where every arm is correct)")
    report_ap.add_argument(
        "--separatrix-k", type=int, default=5,
        help="neighbours examined per point when locating the boundary "
             "band (default: 5)")
    report_ap.add_argument(
        "--fidelity-vs", default=None, metavar="ARM",
        help="also report posterior fidelity of every arm against this "
             "reference arm (e.g. 'hmc'), gated on the reference's own "
             "hmc_diagnostics.json -- a non-converged reference withholds "
             "the number instead of reporting a meaningless one")
    report_ap.add_argument(
        "--rhat-threshold", type=float, default=1.1,
        help="convergence bar the reference's rhat_max is judged against, "
             "recomputed here rather than trusting its stored `converged` "
             "flag (default: 1.1)")
    report_ap.add_argument(
        "--no-validate", dest="validate", action="store_false",
        help="skip guards.validate_frame (provenance and seeding-fix "
             "checks). On by default; the report says so when it is off")
    report_ap.add_argument("--out", type=Path, default=None,
                           help="write the report here instead of stdout")

    args = ap.parse_args(argv)
    if args.command == "launch":
        _run_launch(args)
    else:
        _run_report(args)


if __name__ == "__main__":
    main()
